#!/usr/bin/env python3
"""Create a separate Vaex-only plan for the existing cgroup benchmark runner."""
import argparse
import shlex
import sys
from pathlib import Path

import yaml
from support import source_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory", nargs="+", default=["16G", "8G", "4G"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    parser.add_argument("--layout", choices=["raw", "columnar"], default="columnar")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    if min(args.threads, args.clusters, args.iterations, args.chunk_rows,
           args.timeout, args.repetitions) < 1 or args.components < 2:
        parser.error("counts must be positive and components >= 2")
    scripts = Path(__file__).resolve().parent
    plan = {"version": 1, "root": str(args.results.resolve().parent),
            "results": str(args.results.resolve()), "tools": {"python": sys.executable},
            "defaults": {"repetitions": args.repetitions, "resources": {
                "threads": args.threads, "timeout_seconds": args.timeout,
                "timeout_grace_seconds": 30, "swap_max": 0}},
            "telemetry": {"interval_seconds": 3},
            "resource_profiles": {f"mem{i}": {"memory_max": memory}
                                  for i, memory in enumerate(args.memory)},
            "datasets": {}, "runs": []}
    for index, data in enumerate(args.data):
        _, shape, _ = source_info(data)
        if args.clusters > shape[0] or args.components > shape[1]:
            parser.error(f"clusters/components exceed shape of {data}")
        dataset_id = f"data{index}"
        plan["datasets"][dataset_id] = {"dir": str(data.resolve()),
                                        "ready": ["X.f64", "metadata.json"]}
        for algorithm in ("kmeans", "pca"):
            flags = (f"--clusters {args.clusters} --iterations {args.iterations}"
                     if algorithm == "kmeans" else f"--components {args.components}")
            command = (f'"${{tools.python}}" {shlex.quote(str(scripts / (algorithm + ".py")))} '
                       f'"${{dataset.dir}}" {flags} --layout {args.layout} '
                       f'--threads ${{resources.threads}} --chunk-rows {args.chunk_rows} '
                       '--output "${run.outputs}/vaex-r${rep}.json"')
            run = {"id": f"{algorithm}-{dataset_id}", "dataset": dataset_id,
                   "resource_profiles": list(plan["resource_profiles"]),
                   "implementations": [{"id": "vaex", "comparable": True,
                                          "command": command}],
                   "cold_cache": ["${dataset.dir}"]}
            if args.layout == "columnar":
                run["setup"] = [f'"${{tools.python}}" {shlex.quote(str(scripts / "prepare.py"))} '
                                '"${dataset.dir}"']
            plan["runs"].append(run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(plan, sort_keys=False))
    print(f"Wrote {args.output}: {len(args.data)*2*len(args.memory)*args.repetitions} executions")


if __name__ == "__main__":
    main()
