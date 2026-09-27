#!/usr/bin/env python3
import argparse
import random
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("results", type=Path)
p.add_argument("--suite", choices=("main", "controls", "repeat", "large", "large-repeat", "partner"), default="main")
args = p.parse_args()
cases = []
def add(algo, rows, cols, rank, cache, order, policy, seed=42):
    cases.append([algo, policy, str(rows), str(rank), str(cache), "5", "0", str(cols), order, str(seed)])

if args.suite in ("main", "repeat"):
    for algo, rows, cols, rank in [("lmcg", 32000000, 2, 1), ("kmeans", 8000000, 2, 32),
                                  ("gnmf", 8000000, 2, 16), ("gnmf", 1000000, 128, 16)]:
        for order in ("chunk", "random"):
            for policy in (("arrival", "window", "stagefifo50", "stage50") if args.suite == "main" else ("arrival", "stage50")):
                add(algo, rows, cols, rank, 16777216, order, policy, 42 if args.suite == "main" else 73)
elif args.suite == "controls":
    for order in ("ordered", "window256", "random"):
        for policy in ("arrival", "stage4", "stage16", "stage50", "stage50x"):
            add("gnmf", 2000000, 2, 16, 16777216, order, policy, 73)
    for policy in ("arrival", "window", "stagefifo50", "stage50"):
        add("lmcg", 8000000, 1, 1, 268435456, "random", policy, 73)
        add("gnmf", 1000000, 2, 16, 4194304, "random", policy, 73)
    for policy in ("arrival", "stage50", "stage50x"):
        add("gnmf", 2000000, 2, 64, 16777216, "random", policy, 73)
elif args.suite == "large":
    for policy in ("arrival", "stage50"):
        add("lmcg", 256000000, 1, 1, 16777216, "random", policy, 73)
        add("gnmf", 8000000, 2, 64, 16777216, "random", policy, 73)
elif args.suite == "large-repeat":
    for seed, policies in [(73, ("stage50", "arrival", "stagefifo50", "stage50x")),
                           (91, ("arrival", "stage50"))]:
        for policy in policies:
            add("gnmf", 8000000, 2, 64, 16777216, "random", policy, seed)
else:
    for policy in ("arrival", "stage50", "stage50x", "stage50init", "partner", "stage50partner"):
        add("gnmf", 1000000, 128, 16, 16777216, "random", policy, 73)
    for order in ("chunk", "random"):
        for policy in ("arrival", "stage50", "stage50init"):
            add("gnmf", 8000000, 2, 16, 16777216, order, policy, 73)
if args.suite != "large-repeat":
    random.Random(917 if args.suite == "main" else 183).shuffle(cases)
args.results.mkdir(parents=True, exist_ok=False)
plan = args.results.resolve() / "cases.txt"
plan.write_text("\n".join(" ".join(case) for case in cases) + "\n")
root = Path(__file__).resolve().parents[3] / "systemds"
subprocess.run(["/opt/devcon/env/java/current/bin/java", "-Xms512m", "-Xmx3g", "-XX:ActiveProcessorCount=4",
                f"-Xlog:gc:file={args.results.resolve() / 'gc.log'}:time,uptime",
                "--add-modules=jdk.incubator.vector", "-cp", "target/test-classes:target/classes:target/lib/*",
                "org.apache.sysds.test.component.ooc.PartitionChainStudyTest", "--suite", str(plan), str(args.results.resolve())],
               cwd=root, check=True)
