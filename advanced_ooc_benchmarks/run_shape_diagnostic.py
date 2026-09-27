#!/usr/bin/env python3
"""Bounded shape diagnostic on existing 32 GB native inputs; no main-plan changes."""
import datetime
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

SUITE = Path(__file__).resolve().parent
DATA = Path("/workspace/data_dir")
JAVA = "/opt/devcon/env/java/current/bin/java"
JAR = Path("/workspace/systemds/target/SystemDS.jar")
spec = importlib.util.spec_from_file_location("cgroup", SUITE / "vaex/test_memory_sweep.py")
cg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cg)


def config(case):
    (case / "java-tmp").mkdir(exist_ok=True)
    path = case / "config.xml"
    path.write_text(f'''<root>
  <sysds.localtmpdir>{case}/tmp</sysds.localtmpdir>
  <sysds.scratch>{case}/scratch</sysds.scratch>
  <sysds.defaultblocksize>1000</sysds.defaultblocksize>
  <sysds.ooc.io.direct>true</sysds.ooc.io.direct>
  <sysds.ooc.io.reader.buffersize>8388608</sysds.ooc.io.reader.buffersize>
  <sysds.ooc.io.reader.threads>16</sysds.ooc.io.reader.threads>
  <sysds.ooc.io.reader.poolsize>16</sysds.ooc.io.reader.poolsize>
  <sysds.ooc.materialized.partition.bytes>0</sysds.ooc.materialized.partition.bytes>
  <sysds.ooc.memory.broker.max>4294967296</sysds.ooc.memory.broker.max>
  <sysds.ooc.memory.prefetch.max>1073741824</sysds.ooc.memory.prefetch.max>
  <sysds.ooc.memory.cache.fraction.soft>0.5</sysds.ooc.memory.cache.fraction.soft>
  <sysds.ooc.memory.cache.fraction.hard>0.6</sysds.ooc.memory.cache.fraction.hard>
  <sysds.ooc.source.replay.memory>402653184</sysds.ooc.source.replay.memory>
  <sysds.ooc.source.bulksize>402653184</sysds.ooc.source.bulksize>
  <sysds.ooc.source.replay.prefetch>16</sysds.ooc.source.replay.prefetch>
</root>''')
    return path


def command(case, script, arguments):
    return [JAVA, "-Xms12g", "-Xmx12g", "-XX:+UseG1GC", "-XX:G1HeapRegionSize=32m",
            "-XX:ActiveProcessorCount=16", "--add-modules=jdk.incubator.vector",
            f"-Djava.io.tmpdir={case}/java-tmp", "-Dsysds.ooc.watchdog.primitives=true",
            "-jar", str(JAR), "-f", str(script), "-exec", "singlenode",
            "-config", str(config(case)), "-ooc", "-oocStats", "-stats", "-args",
            *map(str, arguments)]


def run(case, name, script, arguments, cold=(), timeout=120):
    case.mkdir(parents=True, exist_ok=True)
    for path in cold:
        cache = subprocess.run([sys.executable, str(SUITE / "drop_caches.py"), str(path)],
                               capture_output=True, text=True)
        (case / "drop-caches.log").write_text(cache.stdout+cache.stderr)
        if cache.returncode:
            raise RuntimeError(f"cold-cache precondition failed: {path}")
    print(f"Starting {case.name} {name}", flush=True)
    result = cg.run(Path("/sys/fs/cgroup/devcon"), case, name, 16*1024**3,
                    command(case, script, arguments), timeout, min_free_bytes=20*1024**3)
    log = (case / f"{name}.log").read_text(errors="replace")
    if result["status"] == "ok" and ("An Error Occurred" in log or "Exception in thread" in log):
        result["status"] = "failed"
    result["numeric_outputs"] = re.findall(
        r"(?:residual_norm|inertia|model_norm|coefficient_norm|W checksum:|H checksum:)\s*=?\s*([^\n]+)", log)
    result["errors"] = [line[:500] for line in log.splitlines()
                        if any(term in line for term in ("Exception", "ERROR", "An Error", "OutOfMemory"))][:12]
    # Only invocation-owned runtime artifacts and outputs are removed, after logs/statistics persist.
    for leaf in ("tmp", "scratch", "java-tmp", "outputs"):
        target = case / leaf
        if target.exists():
            shutil.rmtree(target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cols", nargs="+", type=int, default=[1000, 10000, 100, 10])
    parser.add_argument("--prepare-baseline", action="store_true",
                        help="Refresh baseline labels using the same rule as reshaped inputs (reuse native X)")
    parser.add_argument("--algorithms", nargs="+", choices=["lmcg", "kmeans", "l2svm", "multilogreg", "gnmf"],
                        default=["lmcg", "kmeans", "l2svm", "multilogreg", "gnmf"])
    args = parser.parse_args()
    os.chdir(SUITE)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root = DATA / "shape-diagnostic" / stamp
    root.mkdir(parents=True)
    source = subprocess.check_output(["git", "-C", "/workspace/systemds", "rev-parse", "HEAD"], text=True).strip()
    manifest = {"source": source, "jar_sha256": hashlib.sha256(JAR.read_bytes()).hexdigest(),
                "memory_gib": 16, "heap_gib": 12, "threads": 16, "blocksize": 1000,
                "shape_cells": 4000000000, "iterations": {"lmcg": 3, "kmeans": 2,
                "gnmf": 2, "l2svm_outer": 2, "multilogreg_outer": 2, "inner": 3},
                "gnmf_rank": 8, "timeout_seconds": 120,
                "label_rule": "sign(first feature) for generated inputs; existing labels otherwise",
                "prepare_baseline": args.prepare_baseline,
                "selected_algorithms": args.algorithms, "results": []}
    summary = root / "summary.json"
    summary.write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Invocation: {root}", flush=True)
    gate = cg.run(Path("/sys/fs/cgroup/devcon"), root, "preflight", 64*1024**2,
                  [sys.executable, "-c", "b=bytearray(256*1024**2)"], 30)
    if gate["status"] != "oom":
        raise RuntimeError("memory enforcement failed")
    classes = root / "helper-classes"
    classes.mkdir()
    subprocess.run([str(Path(JAVA).with_name("javac")), "-cp",
                    f"{JAR}:{JAR.parent}/lib/*", "-d", str(classes),
                    str(SUITE / "FP64ShapeWriter.java")], check=True)
    for cols in args.cols:
        rows = 4000000000 // cols
        for family in ("dense-d32", "gnmf-d32"):
            selected = [a for a in args.algorithms if (a == "gnmf") == (family == "gnmf-d32")]
            if not selected:
                continue
            base = DATA / "bench-data" / family / "systemds"
            shape = root / f"r{rows}-c{cols}-{family}"
            shape.mkdir()
            prepared = shape / "prepared"
            prepared.mkdir()
            x = base / "X-bs1000"
            y = base / "binary_y-bs1000"
            if cols != 1000 or (args.prepare_baseline and family == "dense-d32"):
                labels_only = cols == 1000 and args.prepare_baseline
                if not labels_only:
                    x = prepared / "X"
                y = prepared / "y"
                prepare_case = shape / "prepare"
                prepare_case.mkdir()
                prep = cg.run(Path("/sys/fs/cgroup/devcon"), prepare_case, "prepare", 4*1024**3,
                              [JAVA, "-Xmx2g", "--add-modules=jdk.incubator.vector", "-cp", f"{classes}:{JAR}:{JAR.parent}/lib/*",
                               "FP64ShapeWriter", str(base.parent / "X.f64"), str(prepared),
                               str(rows), str(cols), str(family == "dense-d32").lower(),
                               str(labels_only).lower()],
                              240, min_free_bytes=20*1024**3)
                prep.update(rows=rows, cols=cols, algorithm="preparation", family=family)
                manifest["results"].append(prep)
                summary.write_text(json.dumps(manifest, indent=2)+"\n")
                if prep["status"] != "ok":
                    shutil.rmtree(prepared)
                    continue
            if not Path(str(x)+".mtd").exists():
                raise RuntimeError(f"missing prepared metadata: {x}")
            for algorithm in selected:
                case = shape / algorithm
                output = case / "outputs"
                output.mkdir(parents=True)
                if algorithm == "lmcg":
                    arguments = [x, y, 1e-7, 3, 0, output / "model"]
                    script = SUITE / "lmcg/systemds.dml"
                elif algorithm == "kmeans":
                    arguments = [x, 32, 2, output / "centers", output / "labels"]
                    script = SUITE / "kmeans/systemds.dml"
                elif algorithm == "l2svm":
                    arguments = [x, y, 1e-6, 1, 2, 3, output / "model"]
                    script = SUITE / "l2svm/systemds.dml"
                elif algorithm == "multilogreg":
                    script = case / "run.dml"
                    original = (SUITE / "multilogreg/systemds.dml").read_text()
                    # The suite's raw binary labels are {-1,+1}; softmax needs {1,2}.
                    original = original.replace("Y = read($2);", "Y = (read($2)+3)/2;")
                    script.write_text(original)
                    arguments = [x, y, 1e-6, 1, 2, 3, output / "model"]
                else:
                    script = SUITE / "gnmf/systemds.dml"
                    arguments = [x, 8, 2, 1e-8, 7, output / "W", output / "H"]
                result = run(case, algorithm, script, arguments, [x, y] if family == "dense-d32" else [x])
                result.update(rows=rows, cols=cols, algorithm=algorithm, family=family)
                manifest["results"].append(result)
                summary.write_text(json.dumps(manifest, indent=2)+"\n")
            shutil.rmtree(prepared)
            print(f"Cleaned generated preparation for {rows}x{cols} {family}", flush=True)
    print(f"Completed: {summary}", flush=True)


if __name__ == "__main__":
    main()
