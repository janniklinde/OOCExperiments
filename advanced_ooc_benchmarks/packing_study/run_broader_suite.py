#!/usr/bin/env python3
import argparse
import random
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("results", type=Path)
p.add_argument("--suite", choices=("joins", "algorithms", "bounds", "repeat"), required=True)
a = p.parse_args()
cases = []
def add(op, policy, rows, cols, rank, cache=16, order="random", seed=42):
    cases.append([op, policy, str(rows), str(rank), str(cache*2**20), "5", "0", str(cols), order, str(seed)])

if a.suite == "joins":
    for order in ("chunk", "correlated", "random"):
        for policy in ("single", "batch", "arrival", "stage50", "reorder"):
            add("equijoin", policy, 8000000, 2, 2, order=order)
    for policy in ("stage1", "stage4", "stage16"):
        add("equijoin", policy, 8000000, 2, 2)
    for policy in ("arrival", "stage4", "stage16", "stage50", "reorder"):
        for cache in (4, 128, 512):
            add("equijoin", policy, 8000000, 2, 2, cache=cache)
elif a.suite == "algorithms":
    for op in ("pca-stats", "logreg-hvp", "l2svm-line", "mlp-gate"):
        for rows, cols in (((16000000, 1),) if op == "l2svm-line" else ((8000000, 2), (500000, 32))):
            for order in ("chunk", "random"):
                for policy in ("arrival", "stage4", "stage50", "reorder"):
                    add(op, policy, rows, cols, 1 if op.endswith("hvp") else cols, order=order)
elif a.suite == "bounds":
    for op in ("map", "equijoin"):
        for policy in ("single", "batch", "arrival", "stage50", "reorder"):
            add(op, policy, 8000000, 2, 2, cache=512)
            add(op, policy, 500000, 128, 128)
    for policy in ("single", "batch", "arrival", "stage4", "stage16", "stage50", "reorder"):
        add("equijoin", policy, 32000000, 2, 2)
else:
    for op in ("equijoin", "mlp-gate", "logreg-hvp", "pca-stats"):
        for policy in ("arrival", "stage4", "stage50", "reorder"):
            add(op, policy, 8000000, 2, 1 if op.endswith("hvp") else 2, seed=73)
random.Random(901+len(cases)).shuffle(cases)
a.results.mkdir(parents=True, exist_ok=False)
plan = a.results.resolve() / "cases.txt"
plan.write_text("\n".join(" ".join(c) for c in cases)+"\n")
root = Path(__file__).resolve().parents[3] / "systemds"
subprocess.run(["/opt/devcon/env/java/current/bin/java", "-Xms512m", "-Xmx3g", "-XX:ActiveProcessorCount=4",
                "--add-modules=jdk.incubator.vector", "-cp", "target/test-classes:target/classes:target/lib/*",
                "org.apache.sysds.test.component.ooc.PartitionChainStudyTest", "--suite", str(plan), str(a.results.resolve())],
               cwd=root, check=True)
