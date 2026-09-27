#!/usr/bin/env python3
"""Short MiniBatchKMeans memory sweep with strict cgroup-v2 enforcement."""
import argparse
import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("direct_cgroup", HERE.parent / "vaex/test_memory_sweep.py")
direct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(direct)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--memory", type=int, nargs="+", default=[16, 8, 4])
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = args.results / stamp
    directory.mkdir(parents=True)
    print(f"Results: {directory}", flush=True)
    parent = Path("/sys/fs/cgroup/devcon")
    gate = direct.run(parent, directory, "preflight", 64*1024**2,
                      [sys.executable, "-c", "b=bytearray(256*1024**2)"], 30)
    if gate["status"] != "oom":
        raise RuntimeError("memory enforcement preflight failed")
    records = []
    for gib in args.memory:
        name = f"mem{gib}"
        case = directory / name
        case.mkdir()
        cache = subprocess.run([sys.executable, str(HERE.parent / "drop_caches.py"),
                                str(args.data / "X.f64")], capture_output=True, text=True)
        (case / "drop-caches.log").write_text(cache.stdout+cache.stderr)
        if cache.returncode:
            raise RuntimeError("cold-cache precondition failed")
        result = direct.run(parent, case, name, gib*1024**3,
                            [sys.executable, str(HERE.parent / "kmeans/sklearn_batching.py"), str(args.data),
                             "--passes", str(args.passes), "--output", str(case / "sklearn.json")],
                            args.timeout, {"OMP_NUM_THREADS": str(args.threads)})
        records.append(result)
        (directory / "summary.json").write_text(json.dumps(records, indent=2)+"\n")
        # Avoid spending time on tighter limits when the initial pilot already times out.
        if result["status"] != "ok":
            print("Stopping sweep after unsuccessful pilot", flush=True)
            break


if __name__ == "__main__":
    main()
