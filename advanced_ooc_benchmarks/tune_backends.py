#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0.

"""Generate and summarize memory-policy tuning for LMCG on OOC, Spark, and Dask."""
import argparse
import copy
import csv
import itertools
import json
import math
from pathlib import Path
import random
import re
import statistics

import yaml

GIB = 1024**3
MIB = 1024**2
MEMORIES = (4, 8, 16, 32)


def policies():
    result = {}
    # Separate cache pressure from the value of allowing growth above a 12 GiB heap.
    for worker, prefetch, (cache, cap_scale) in itertools.product(
            (4, 5), (1, 2, 3), ((0.0625, 1), (0.125, 1), (0.125, 2))):
        key = f"ooc_w{worker}_p{prefetch}_c{int(cache * 10000)}_cap{cap_scale}"
        result[key] = dict(backend="ooc", worker_fraction=worker / 12,
                          worker_cap_gib=worker * cap_scale, prefetch_fraction=prefetch / 12,
                          prefetch_cap_gib=prefetch * cap_scale, cache_hard_fraction=cache,
                          cache_soft_fraction=cache * 2 / 3)
    for backend, chunks, retention in (("spark", (64, 128, 256), (0.15, 0.25)),
                                       ("dask", (50, 100, 200), (0.4, 0.6))):
        for chunk, share, retain in itertools.product(chunks, (0.25, 0.5, 0.75), retention):
            key = f"{backend}_p{chunk}_t{int(share * 100)}_m{int(retain * 100)}"
            result[key] = dict(backend=backend, partition_size=chunk,
                              concurrency_fraction=share, retention_fraction=retain)
    return result


def configured_run(plan, reference, policy, memory):
    run = copy.deepcopy(reference)
    run.pop("resource_profiles", None)
    run["resources"].update(plan["resource_profiles"][f"mem{memory}"])
    run["resources"]["threads"] = 64
    heap = memory * 3 / 4 * GIB
    backend = policy["backend"]
    extra = {}
    if backend == "ooc":
        worker = min(heap * policy["worker_fraction"], policy["worker_cap_gib"] * GIB)
        prefetch = min(heap * policy["prefetch_fraction"], policy["prefetch_cap_gib"] * GIB)
        hard = heap * policy["cache_hard_fraction"]
        assert worker + prefetch + hard + max(512 * MIB, heap * 0.1) <= heap
        replay = int(prefetch * 0.375)
        extra = {
            "memory.broker.fraction": policy["worker_fraction"],
            "memory.broker.max": policy["worker_cap_gib"] * GIB,
            "memory.prefetch.fraction": policy["prefetch_fraction"],
            "memory.prefetch.max": policy["prefetch_cap_gib"] * GIB,
            "memory.cache.fraction.soft": policy["cache_soft_fraction"],
            "memory.cache.fraction.hard": policy["cache_hard_fraction"],
            "source.replay.memory": replay, "source.bulksize": replay,
            "source.replay.prefetch": max(2, int(32 * prefetch / GIB)),
        }
        run["implementations"] = [{"id": "systemds-ooc", "template": "systemds-ooc"}]
        run["blocksize_sweeps"] = {"ooc": [1500]}
    elif backend == "spark":
        part = policy["partition_size"] * MIB
        threads = max(1, min(64, int(heap * policy["concurrency_fraction"] / part)))
        run["resources"]["spark_threads"] = threads
        run["parameters"].update(spark_partition_bytes=part,
                                 spark_memory_fraction=policy["retention_fraction"])
        run["implementations"] = [{"id": "systemds-spark", "template": "systemds-spark"}]
        run["blocksize_sweeps"] = {"spark": [1000]}
    else:
        part = policy["partition_size"] * 1_000_000
        threads = max(1, min(64, int(heap * policy["concurrency_fraction"] / part)))
        run["resources"]["dask_threads"] = threads
        # dense_d32 has 1000 columns. These chunk sizes divide its 4M rows.
        run["parameters"]["dask_chunk"] = part // (1000 * 8)
        impl = copy.deepcopy(next(i for i in reference["implementations"] if i["template"] == "dask"))
        impl["command"] += f" --memory-target {policy['retention_fraction']}"
        run["implementations"] = [impl]
        run["blocksize_sweeps"] = {"ooc": [1500]}
        run.pop("setup", None)
    if extra:
        xml = "".join(f"  <sysds.ooc.{key}>{value}</sysds.ooc.{key}>\n" for key, value in extra.items())
        run["setup"]["command"] = run["setup"]["command"].replace("</root>", xml + "</root>", 1)
    return run


def generate(args):
    plan = yaml.safe_load(args.source.read_text())
    reference = next(r for r in plan["runs"] if r["id"] == "lmcg")
    assert reference["dataset"] == "dense_d32"
    assert plan["datasets"]["dense_d32"]["parameters"]["cols"] == 1000
    plan["results"] = "${plan.root}/backend-tuning-results"
    plan["defaults"]["repetitions"] = 1
    for backend in ("systemds-ooc", "systemds-spark"):
        command = plan["templates"][backend]["command"].replace("G1HeapRegionSize=8m", "G1HeapRegionSize=32m")
        command = command.replace("-XX:G1HeapRegionSize=32m",
                                  "-XX:G1HeapRegionSize=32m -Xlog:gc*:file=${run.results}/gc.log:time,uptime,level,tags")
        if backend == "systemds-spark":
            command = command.replace("spark.memory.fraction=0.25", "spark.memory.fraction=${run.spark_memory_fraction}")
            command = command.replace("spark.hadoop.fs.local.block.size=134217728",
                                      "spark.hadoop.fs.local.block.size=${run.spark_partition_bytes}")
        plan["templates"][backend]["command"] = command
    candidates = policies()
    # Deduplicate capped concurrency settings while retaining every policy alias.
    concrete = {}
    for memory in MEMORIES:
        for name, policy in candidates.items():
            run = configured_run(plan, reference, policy, memory)
            run["resources"]["timeout_seconds"] = args.timeout
            signature_run = copy.deepcopy(run)
            if policy["backend"] == "ooc":
                heap = memory * 3 / 4 * GIB
                for broker, fraction, cap in (("broker", policy["worker_fraction"], policy["worker_cap_gib"]),
                                              ("prefetch", policy["prefetch_fraction"], policy["prefetch_cap_gib"])):
                    command = signature_run["setup"]["command"]
                    signature_run["setup"]["command"] = re.sub(
                        rf"(<sysds.ooc.memory.{broker}.max>)[0-9]+",
                        lambda m: m[1] + str(int(min(heap * fraction, cap * GIB))), command)
            signature = json.dumps(signature_run, sort_keys=True)
            if signature in concrete:
                concrete[signature]["aliases"].append(name)
            else:
                concrete[signature] = dict(run=run, aliases=[name], memory=memory,
                                           backend=policy["backend"])
    runs = []
    manifest = {}
    entries = list(concrete.values())
    for repetition in range(1, args.repetitions + 1):
        # Group by Zarr geometry to rebuild the shared store at most once per phase;
        # shuffle backends and memory sizes inside phases, independently per round.
        phases = [[], [], []]
        for index, entry in enumerate(entries):
            phase = ((entry["run"]["parameters"]["dask_chunk"] // 6250).bit_length() - 1
                     if entry["backend"] == "dask" else index % 3)
            phases[phase].append(entry)
        if repetition % 2 == 0:
            phases.reverse()
        rng = random.Random(args.seed + repetition)
        for phase in phases:
            rng.shuffle(phase)
            for entry in phase:
                run = copy.deepcopy(entry["run"])
                name = f"tune_{entry['aliases'][0]}_mem{entry['memory']}_round{repetition}"
                run["id"] = name
                runs.append(run)
                manifest[name] = dict(backend=entry["backend"], memory_gib=entry["memory"],
                                      policies=entry["aliases"], round=repetition)
    plan["runs"] = runs
    plan["tuning"] = dict(workload="lmcg/dense_d32/10 iterations", seed=args.seed,
                          repetitions=args.repetitions, policies=candidates, runs=manifest)
    args.output.write_text(yaml.safe_dump(plan, sort_keys=False, width=120))
    print(f"Wrote {args.output}: {len(concrete)} distinct configurations, {len(runs)} executions")


def summarize(args):
    root = args.invocation.resolve()
    plan = yaml.safe_load((root / "benchmark-plan.yaml").read_text())
    tuning = dict(plan["tuning"])
    configured = {run["id"] for run in plan["runs"]}
    # A shortened sweep may remove complete round entries from `runs:` without
    # rewriting the generated tuning manifest. Rank only executions that the
    # runner could actually have launched.
    tuning["runs"] = {name: meta for name, meta in tuning["runs"].items()
                      if name in configured}
    tuning["repetitions"] = len({meta["round"] for meta in tuning["runs"].values()})
    observations = []
    for name, meta in tuning["runs"].items():
        matches = list(root.glob(name + "-*/logs/*r1.metrics"))
        row = dict(run=name, **meta, status="missing", wall_seconds=None)
        if len(matches) == 1:
            path = matches[0]
            metrics = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
            log = path.with_suffix(".log").read_text(errors="replace")
            time_path = Path(str(path) + ".time")
            timing = time_path.read_text() if time_path.exists() else ""
            match = re.search(r"Elapsed .*\):\s*([\d:.]+)", timing)
            wall = 0.0
            if match:
                for part in match.group(1).split(":"):
                    wall = 60 * wall + float(part)
            residuals = re.findall(r'(?:residual_norm=|"residual_norm":\s*)([\deE.+-]+)', log)
            residual = float(residuals[-1]) if residuals else math.nan
            # A successful exit is insufficient: SystemDS can return zero on DML errors.
            valid = (metrics.get("exit_status") == "0" and wall > 0 and math.isfinite(wall)
                     and math.isfinite(residual) and residual <= 1e-6
                     and "An Error Occurred" not in log
                     and not re.search(r"(?:^|;)oom_kill [1-9]", metrics.get("memory_events", "")))
            row.update(status="ok" if valid else "failed", wall_seconds=wall if match else None,
                       residual_norm=residual if math.isfinite(residual) else None,
                       proc_read_bytes=int(metrics.get("proc_read_bytes", 0)),
                       memory_peak_bytes=int(metrics.get("memory_peak_bytes", 0)))
        observations.append(row)
    grouped = {}
    for row in observations:
        for policy in row["policies"]:
            grouped.setdefault((row["backend"], row["memory_gib"], policy), []).append(row)
    ranking = []
    for (backend, memory, policy), rows in grouped.items():
        ok = [r["wall_seconds"] for r in rows if r["status"] == "ok"]
        complete = len(ok) == tuning["repetitions"]
        ranking.append(dict(backend=backend, memory_gib=memory, policy=policy,
                            complete=complete, successful=len(ok), expected=tuning["repetitions"],
                            median_seconds=statistics.median(ok) if complete else None,
                            min_seconds=min(ok) if ok else None, max_seconds=max(ok) if ok else None))
    winners = []
    common = []
    lines = ["# LMCG backend tuning", "", "Best tested policies; medians of all required rounds. Missing/failed rounds disqualify a policy.",
             "Residual <= 1e-6 is a convergence screen, not a full coefficient comparison.", "",
             "| Backend | Cgroup GiB | Policy | Median seconds |", "|---|---:|---|---:|"]
    for backend in ("ooc", "spark", "dask"):
        best = {}
        for memory in MEMORIES:
            eligible = [r for r in ranking if r["backend"] == backend and r["memory_gib"] == memory and r["complete"]]
            if eligible:
                winner = min(eligible, key=lambda r: r["median_seconds"])
                best[memory] = winner["median_seconds"]
                winners.append(winner)
                lines.append(f"| {backend} | {memory} | {winner['policy']} | {winner['median_seconds']:.2f} |")
            else:
                lines.append(f"| {backend} | {memory} | No complete valid policy | — |")
        for policy, settings in tuning["policies"].items():
            if settings["backend"] != backend:
                continue
            rows = [r for r in ranking if r["policy"] == policy and r["complete"]]
            if len(rows) != len(MEMORIES) or len(best) != len(MEMORIES):
                continue
            ratios = [r["median_seconds"] / best[r["memory_gib"]] for r in rows]
            common.append(dict(backend=backend, policy=policy,
                               geometric_mean_slowdown=math.exp(statistics.mean(map(math.log, ratios))),
                               worst_slowdown=max(ratios), settings=settings))
    lines += ["", "## One policy across all four memory sizes", ""]
    selected = []
    for backend in ("ooc", "spark", "dask"):
        options = sorted((r for r in common if r["backend"] == backend), key=lambda r: r["geometric_mean_slowdown"])
        if options:
            selected.append(options[0])
            for r in options[:3]:
                lines.append(f"- {backend}: `{r['policy']}` — geometric mean slowdown {r['geometric_mean_slowdown']:.3f}×, worst {r['worst_slowdown']:.3f}×.")
        else:
            lines.append(f"- {backend}: no policy completed successfully at all four sizes.")
    report = dict(observations=observations, rankings=ranking, per_memory_winners=winners,
                  common_policies=common, selected_common_policies=selected,
                  policies=tuning["policies"])
    (root / "tuning-summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (root / "tuning-summary.md").write_text("\n".join(lines) + "\n")
    with (root / "tuning-ranking.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(ranking[0]))
        writer.writeheader()
        writer.writerows(ranking)
    # Ready-to-run confirmation plan: one common policy per backend, all memory sizes.
    # Keep source paths portable rather than copying the downloaded container's tools.
    if len(selected) == 3 and len(tuning["policies"]) > 3:
        source = yaml.safe_load(args.source.read_text())
        source["templates"] = plan["templates"]
        source["results"] = "${plan.root}/backend-tuning-confirm-results"
        source["defaults"]["repetitions"] = 1
        reference = next(r for r in source["runs"] if r["id"] == "lmcg")
        source["runs"] = []
        source["tuning"] = dict(tuning, repetitions=3, runs={},
                                policies={r["policy"]: r["settings"] for r in selected})
        chosen = {r["backend"]: r["policy"] for r in selected}
        for repetition in range(1, 4):
            for memory in MEMORIES:
                for backend, policy in chosen.items():
                    name = next(name for name, meta in tuning["runs"].items()
                                if meta["memory_gib"] == memory and policy in meta["policies"])
                    run = configured_run(source, reference, tuning["policies"][policy], memory)
                    run["resources"]["timeout_seconds"] = next(r for r in plan["runs"] if r["id"] == name)["resources"]["timeout_seconds"]
                    run["id"] = f"confirm_{policy}_mem{memory}_round{repetition}"
                    source["runs"].append(run)
                    source["tuning"]["runs"][run["id"]] = dict(
                        backend=backend, memory_gib=memory, policies=[policy], round=repetition)
        target = args.source.with_name("benchmark-plan-backend-confirm.yaml")
        target.write_text(yaml.safe_dump(source, sort_keys=False, width=120))
        lines.append(f"\nConfirmation plan: {target}")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    source = Path(__file__).with_name("benchmark-plan.yaml")
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--source", type=Path, default=source)
    generate_parser.add_argument("--output", type=Path, default=source.with_name("benchmark-plan-backend-tuning.yaml"))
    generate_parser.add_argument("--repetitions", type=int, default=2)
    generate_parser.add_argument("--timeout", type=int, default=1200)
    generate_parser.add_argument("--seed", type=int, default=1709)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("invocation", type=Path)
    summary_parser.add_argument("--source", type=Path, default=source)
    args = parser.parse_args()
    if args.command == "generate":
        if args.repetitions < 1 or args.timeout < 1:
            parser.error("repetitions and timeout must be positive")
        generate(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
