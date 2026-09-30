#!/usr/bin/env python3
"""Create the opt-in two-workload PreVision pilot from the main benchmark plan."""
import argparse
from pathlib import Path

import yaml


def main():
    suite = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=suite / "benchmark-plan.yaml")
    parser.add_argument("--output", type=Path,
                        default=suite / "benchmark-plan-prevision.generated.yaml")
    parser.add_argument("--arena", choices=("posix", "memfd"), default="posix")
    parser.add_argument("--threads", type=int, help="override the host CPU allowance")
    parser.add_argument("--blas-threads", type=int, help="override PreVision's BLAS threads")
    parser.add_argument("--elementwise-threads", type=int,
                        help="override PreVision's per-tile elementwise pthread count")
    parser.add_argument("--tile-rows", type=int, help="override PreVision input tile rows")
    parser.add_argument("--tile-cols", type=int, help="override PreVision input tile columns")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source.parent != output.parent:
        parser.error("write the generated plan beside the source so plan.dir stays valid")
    plan = yaml.safe_load(source.read_text())
    selected = {"gnmf_prevision", "gram_prevision"}
    plan["runs"] = [run for run in plan["runs"] if run["id"] in selected]
    if {run["id"] for run in plan["runs"]} != selected:
        parser.error("source plan is missing a PreVision pilot run")
    for run in plan["runs"]:
        run["enabled"] = True
        mode = "gnmf" if run["id"] == "gnmf_prevision" else "gram"
        for axis, value in (("", args.tile_rows), ("_cols", args.tile_cols)):
            if value is not None:
                if value < 1:
                    parser.error("tile extents must be positive")
                run["parameters"][f"prevision_{mode}_tile{axis}"] = value
    plan["defaults"]["resources"]["prevision_arena"] = args.arena
    for argument in ("threads", "blas_threads", "elementwise_threads"):
        value = getattr(args, argument)
        if value is not None and value < 1:
            parser.error(f"--{argument.replace('_', '-')} must be positive")
        if value is not None:
            key = argument if argument == "threads" else "prevision_" + argument
            plan["defaults"]["resources"][key] = value
    if args.threads is not None:
        for profile in plan["resource_profiles"].values():
            # Dask splits the allowance evenly, so round down to avoid exceeding it.
            workers = min(args.threads, int(profile.get("dask_workers", 1)))
            profile["dask_workers"] = workers
            profile["dask_threads"] = max(workers, args.threads // workers * workers)
            profile["spark_threads"] = args.threads
    output.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    print(f"wrote {output}: GNMF and Gram, each at mem16 ({args.arena} arena)")


if __name__ == "__main__":
    main()
