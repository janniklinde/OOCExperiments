#!/usr/bin/env python3
"""Small Vaex K-means sweep using delegated cgroup v2 without systemd.

Creates only child cgroups; never enables controllers or modifies ancestor limits.
"""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent


def values(path):
    return {key: int(value) for key, value in
            (line.split() for line in path.read_text().splitlines())}


def io_values(group):
    totals = {"rbytes": 0, "wbytes": 0}
    for line in (group / "io.stat").read_text().splitlines():
        for field in line.split()[1:]:
            key, value = field.split("=")
            if key in totals:
                totals[key] += int(value)
    return totals


def run(parent, directory, name, memory, command, timeout, environment=None, min_free_bytes=0):
    group = parent / f"vaex-{os.getpid()}-{name}"
    group.mkdir()
    process = None
    try:
        (group / "memory.max").write_text(str(memory))
        (group / "memory.swap.max").write_text("0")
        (group / "memory.oom.group").write_text("1")
        assert int((group / "memory.max").read_text()) == memory
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        env.update(environment or {})
        start = time.monotonic()
        timed_out = False
        disk_limit = False
        with (directory / f"{name}.log").open("w") as log, \
                (directory / f"{name}.telemetry.csv").open("w") as trace:
            writer = csv.DictWriter(trace, fieldnames=["seconds", "memory_bytes",
                                                       "cpu_seconds", "read_bytes", "write_bytes"])
            writer.writeheader()
            process = subprocess.Popen([sys.executable, str(__file__), "--enter", str(group),
                                        "--", *command], stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            while True:
                io = io_values(group)
                writer.writerow({"seconds": time.monotonic()-start,
                                 "memory_bytes": int((group / "memory.current").read_text()),
                                 "cpu_seconds": values(group / "cpu.stat")["usage_usec"]/1e6,
                                 "read_bytes": io["rbytes"], "write_bytes": io["wbytes"]})
                trace.flush()
                if process.poll() is not None:
                    break
                if min_free_bytes and shutil.disk_usage(directory).free < min_free_bytes:
                    disk_limit = True
                    (group / "cgroup.kill").write_text("1")
                    process.wait()
                    break
                if time.monotonic()-start > timeout:
                    timed_out = True
                    (group / "cgroup.kill").write_text("1")
                    process.wait()
                    break
                try:
                    process.wait(timeout=3 if name != "preflight" else .1)
                except subprocess.TimeoutExpired:
                    pass
        cpu = values(group / "cpu.stat")
        io = io_values(group)
        events = values(group / "memory.events")
        result = {"name": name, "status": "disk_limit" if disk_limit else "timeout" if timed_out else
                  "oom" if events.get("oom_kill", 0) else
                  "ok" if process.returncode == 0 else "failed",
                  "returncode": process.returncode, "wall_seconds": time.monotonic()-start,
                  "memory_limit_bytes": memory,
                  "peak_memory_bytes": int((group / "memory.peak").read_text()),
                  "cpu_seconds": cpu["usage_usec"]/1e6,
                  "cpu_user_seconds": cpu["user_usec"]/1e6,
                  "cpu_system_seconds": cpu["system_usec"]/1e6,
                  "read_bytes": io["rbytes"], "write_bytes": io["wbytes"],
                  "memory_events": events, "command": command}
        (directory / f"{name}.metrics.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)
        return result
    finally:
        if process is not None and process.poll() is None:
            (group / "cgroup.kill").write_text("1")
            process.wait()
        # Kernel may need a short interval to reap the last killed process.
        for _ in range(50):
            try:
                group.rmdir()
                break
            except OSError:
                time.sleep(.1)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--enter":
        (Path(sys.argv[2]) / "cgroup.procs").write_text(str(os.getpid()))
        os.execvpe(sys.argv[4], sys.argv[4:], os.environ)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--cgroup-parent", type=Path, default=Path("/sys/fs/cgroup/devcon"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=32)
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    invocation = args.results / stamp
    invocation.mkdir(parents=True)
    print(f"Results: {invocation}", flush=True)
    gate = run(args.cgroup_parent, invocation, "preflight", 64*1024**2,
               [sys.executable, "-c", "b=bytearray(256*1024**2); print('UNENFORCED')"], 30)
    if gate["status"] != "oom":
        raise RuntimeError("64 MiB enforcement gate failed; refusing sweep")
    preparation = run(args.cgroup_parent, invocation, "prepare", 8*1024**3,
                      [sys.executable, str(HERE / "prepare.py"), str(args.data)], 3600)
    if preparation["status"] != "ok":
        raise RuntimeError("columnar preparation failed")
    results = []
    for gib in (16, 8, 4):
        name = f"mem{gib}"
        directory = invocation / name
        directory.mkdir()
        cache = subprocess.run([sys.executable, str(HERE.parent / "drop_caches.py"),
                                str(args.data / "vaex" / "X.npy")], capture_output=True, text=True)
        (directory / "drop-caches.log").write_text(cache.stdout + cache.stderr)
        if cache.returncode:
            raise RuntimeError(f"cold-cache precondition failed; see {directory}")
        print(f"Starting {name}: {args.clusters} clusters, {args.iterations} iterations, "
              f"{args.threads} threads", flush=True)
        result = run(args.cgroup_parent, directory, name, gib*1024**3,
                     [sys.executable, str(HERE / "kmeans.py"), str(args.data),
                      "--layout", "columnar", "--threads", str(args.threads),
                      "--clusters", str(args.clusters), "--iterations", str(args.iterations),
                      "--output", str(directory / "vaex.json")], args.timeout)
        results.append(result)
        (invocation / "summary.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
