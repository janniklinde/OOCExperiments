#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0.

"""Derive the calibration plan from the benchmark plan.

The sweep that picks an OOC blocksize, a Spark blocksize, and a Dask chunk size has to
read exactly the data the main sweep reads. A dataset's directory is named by a hash of
its id, parameters, and prepare command, so a hand-maintained second plan regenerates
hundreds of GB into a second directory the moment one of those three drifts - silently,
with no error. This copies the source plan verbatim up to `runs:` (datasets, templates
and comments included) so that hash cannot differ, and substitutes only `results:` and
the run list.

Read the winners out of the calibration results, write them into the main plan as
constants, and run that. Selection is deliberately a human step: at one repetition a
difference under about 20% is noise, and the honest reading is then "flat, keep 1000"
rather than whichever case happened to finish first.

Usage: make_calibration_plan.py [SOURCE_PLAN] [-o TARGET_PLAN]
"""
import argparse
import re
from pathlib import Path

import yaml

# 95 MiB at 1000 columns is the middle point and the value the main plan already uses;
# the flanks are half and double it. Every one divides 4,000,000 rows exactly, so no
# trailing chunk is padded and the three stores differ only in chunk length.
DASK_ROW_CHUNKS = (6250, 12500, 25000)

# `dask_threads: auto` resolves to the host's core count, which is the one setting that
# changed when the suite moved to a 128-core machine: the same 12 GiB worker that finished
# in ~430s at 48 threads times out there while spilling ~1 TB. These two are measured at the
# middle chunk, where the auto case is already a third data point at the same geometry.
DASK_THREAD_COUNTS = (16, 48)
DASK_THREAD_CHUNK = 12500

RUNS_HEADER = """runs:
  # Calibration only. One workload - lmCG re-reads X in both orientations every
  # iteration, so it is the dense workload most sensitive to block geometry - at one
  # data:memory ratio. Everything else in the suite inherits the winners as constants.
"""

BLOCKSIZE_RUN = """
  - id: calibration_blocksize
    enabled: true
    dataset: {dataset}
    resource_profiles: [{profile}]
    entrypoint: ${{plan.dir}}/lmcg/systemds.dml
    blocksize_sweeps: {{ooc: {blocksizes}, spark: {blocksizes}}}
    parameters:
      iterations: 10
      reg: 1e-7
      tolerance: 0.0
      systemds_args: >-
        -args "${{input.X}}" "${{input.y}}" ${{run.reg}} ${{run.iterations}} ${{run.tolerance}}
        "${{run.outputs}}/systemds-beta-r${{rep}}"
    inputs: {{X: {{artifact: X}}, y: {{artifact: binary_y}}}}
    resources: {{timeout_seconds: {timeout}}}
    setup: *systemds_setup
    cold_cache: ["${{dataset.dir}}"]
    implementations:
      - {{id: systemds-ooc, template: systemds-ooc}}
      - {{id: systemds-spark, template: systemds-spark}}
"""

DASK_RUN = """
  - id: calibration_dask_c{chunk}
    enabled: true
    dataset: {dataset}
    resource_profiles: [{profile}]
    entrypoint: ${{plan.dir}}/lmcg/systemds.dml
    # No implementation names a sweep, so this expands to the single -baseline case that
    # carries the `baseline_only` Zarr variant. Without a `blocksize_sweeps` mapping the
    # run would not be a baseline case at all and the Zarr store would never be prepared.
    blocksize_sweeps: {{ooc: [1000]}}
    parameters:
      # resolve_zarr keeps one store per dataset and rebuilds it in place when this
      # changes, so each chunk costs a rebuild of X.zarr. That is affordable precisely
      # because calibration is its own plan: three cases, three rebuilds, none timed.
      dask_chunk: {chunk}
      iterations: 10
      reg: 1e-7
      tolerance: 0.0
    resources: {{timeout_seconds: {timeout}}}
    cold_cache: ["${{dataset.dir}}"]
    implementations:
      - id: dask-cg
        template: dask
        command: >-
          "${{tools.python}}" "${{plan.dir}}/lmcg/dask_array.py" "${{dataset.dir}}"
          --iterations ${{run.iterations}} --reg ${{run.reg}} --tolerance ${{run.tolerance}}
          --threads ${{resources.dask_threads}}
          --memory-limit ${{resources.dask_memory_limit}}
          --temporary-directory "${{run.results}}/dask-spill"
          --output "${{run.outputs}}/dask-r${{rep}}.json"
"""


DASK_THREADS_RUN = """
  - id: calibration_dask_t{threads}
    enabled: true
    dataset: {dataset}
    resource_profiles: [{profile}]
    entrypoint: ${{plan.dir}}/lmcg/systemds.dml
    blocksize_sweeps: {{ooc: [1000]}}
    parameters:
      # Same chunk as calibration_dask_c{chunk}, which supplies the `auto` point of this
      # sweep, and emitted next to it so the shared Zarr store is not rebuilt in between.
      dask_chunk: {chunk}
      iterations: 10
      reg: 1e-7
      tolerance: 0.0
    resources: {{timeout_seconds: {timeout}, dask_threads: {threads}}}
    cold_cache: ["${{dataset.dir}}"]
    implementations:
      - id: dask-cg
        template: dask
        command: >-
          "${{tools.python}}" "${{plan.dir}}/lmcg/dask_array.py" "${{dataset.dir}}"
          --iterations ${{run.iterations}} --reg ${{run.reg}} --tolerance ${{run.tolerance}}
          --threads ${{resources.dask_threads}}
          --memory-limit ${{resources.dask_memory_limit}}
          --temporary-directory "${{run.results}}/dask-spill"
          --output "${{run.outputs}}/dask-r${{rep}}.json"
"""


# The calibration workload is a literal template rather than a copy of the source run,
# so nothing stops the two drifting apart -- and a blocksize calibrated against a
# different iteration count is worse than no calibration, because it looks valid.
CALIBRATED_PARAMETERS = {"iterations": 10, "reg": 1e-7, "tolerance": 0.0}


def check_reference(source, run_id, dataset, profile):
    """Fail if the source plan's reference run no longer matches the template here."""
    plan = yaml.safe_load(source.read_text(encoding="utf-8"))
    run = next((r for r in plan.get("runs", []) if r.get("id") == run_id), None)
    if run is None:
        raise SystemExit(f"{source.name} has no run {run_id!r}; pass --reference-run")
    if dataset not in str(run.get("dataset")):
        raise SystemExit(f"Run {run_id} uses dataset {run.get('dataset')!r}, "
                         f"not the calibration dataset {dataset!r}")
    if profile not in str(run.get("resource_profiles")):
        raise SystemExit(f"Run {run_id} uses profiles {run.get('resource_profiles')!r}, "
                         f"not the calibration profile {profile!r}")
    parameters = run.get("parameters", {})
    drift = {key: (want, parameters.get(key)) for key, want in CALIBRATED_PARAMETERS.items()
             if float(parameters.get(key, "nan")) != want}
    if drift:
        raise SystemExit(
            f"Run {run_id} has drifted from the calibration template: "
            + "; ".join(f"{k} is {have!r}, template says {want!r}"
                        for k, (want, have) in sorted(drift.items()))
            + f"\nUpdate CALIBRATED_PARAMETERS in {Path(__file__).name} to match, "
              "or the calibrated blocksize will not describe the run that uses it.")
    if profile not in plan.get("resource_profiles", {}):
        raise SystemExit(f"{source.name} has no resource profile {profile!r}")
    if dataset not in plan.get("datasets", {}):
        raise SystemExit(f"{source.name} has no dataset {dataset!r}")


def main(argv=None):
    plan_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default=str(plan_dir / "benchmark-plan.yaml"))
    parser.add_argument("-o", "--output", default="")
    parser.add_argument("--dataset", default="dense_d32")
    parser.add_argument("--profile", default="mem16")
    parser.add_argument("--blocksizes", default="500,1000,2000")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--reference-run", default="lmcg",
                        help="run in the source plan whose parameters this must match")
    args = parser.parse_args(argv)

    source = Path(args.source).resolve()
    if not source.is_file():
        raise SystemExit(f"Missing source plan: {source}")
    target = Path(args.output).resolve() if args.output else \
        source.with_name(source.stem + "-calibration" + source.suffix)

    check_reference(source, args.reference_run, args.dataset, args.profile)

    text = source.read_text(encoding="utf-8")
    runs = re.search(r"^runs:[ \t]*$", text, re.MULTILINE)
    if not runs:
        raise SystemExit("Plan has no `runs:` section to replace")
    prefix = text[:runs.start()]

    results = re.search(r"^results:[ \t]*\S.*$", prefix, re.MULTILINE)
    if not results:
        raise SystemExit("Plan has no `results:` entry to redirect")
    prefix = (prefix[:results.start()] + "results: ${plan.root}/calibration-results"
              + prefix[results.end():])

    blocksizes = "[" + ", ".join(part.strip() for part in args.blocksizes.split(",")) + "]"
    body = RUNS_HEADER + BLOCKSIZE_RUN.format(
        dataset=args.dataset, profile=args.profile, blocksizes=blocksizes,
        timeout=args.timeout)
    for chunk in DASK_ROW_CHUNKS:
        body += DASK_RUN.format(dataset=args.dataset, profile=args.profile, chunk=chunk,
                                timeout=args.timeout)
        if chunk == DASK_THREAD_CHUNK:
            for threads in DASK_THREAD_COUNTS:
                body += DASK_THREADS_RUN.format(dataset=args.dataset, profile=args.profile,
                                                chunk=chunk, threads=threads,
                                                timeout=args.timeout)

    target.write_text(prefix + body, encoding="utf-8")
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
