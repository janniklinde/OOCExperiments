#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0.

"""Execute a declarative OOC benchmark plan from YAML."""

import argparse
import copy
import csv
from datetime import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("benchmark_plan.py requires PyYAML", file=sys.stderr)
    raise

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z0-9_.-]+)\}")
_VALID_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def execution_timestamp(moment=None):
    """Return a filesystem-safe, timezone-qualified benchmark invocation ID."""
    moment = moment or datetime.now().astimezone()
    return moment.strftime("%Y%m%dT%H%M%S.%f%z")


def flatten(prefix, value, out):
    if isinstance(value, dict):
        for key, child in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), child, out)
    elif value is not None:
        out[prefix] = str(value).lower() if isinstance(value, bool) else str(value)


def expand(value, context):
    if not isinstance(value, str):
        return value

    def replace(match):
        key = match.group(1)
        if key not in context:
            raise ValueError(f"Unknown placeholder ${{{key}}}")
        return context[key]

    previous = None
    while value != previous:
        previous = value
        value = _PLACEHOLDER.sub(replace, value)
    return value


def expand_map(values, context):
    return {str(key): expand(str(value), context) for key, value in (values or {}).items()}


def command_run(command, cwd, env, log=None):
    output = None if log is None else open(log, "w", encoding="utf-8")
    try:
        return subprocess.run(["bash", "-lc", command], cwd=cwd, env=env,
                              stdout=output, stderr=subprocess.STDOUT).returncode
    finally:
        if output:
            output.close()


def metadata_matches(path, expected):
    try:
        actual = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False, f"missing or invalid metadata {path}"
    for key, wanted in expected.items():
        value = actual
        for component in str(key).split("."):
            if not isinstance(value, dict) or component not in value:
                return False, f"{path} has no metadata key {key}"
            value = value[component]
        if str(value).lower() != str(wanted).lower():
            return False, f"{path}: {key} is {value}, expected {wanted}"
    return True, ""


def dataset_status(dataset, directory, context):
    problems = []
    for ready in dataset.get("ready", []):
        path = Path(expand(str(ready), context))
        if not path.is_absolute():
            path = directory / path
        if not path.exists():
            problems.append(f"missing {path}")
    for artifact_id, artifact in dataset.get("artifacts", {}).items():
        path = Path(expand(str(artifact["path"]), context))
        if not path.is_absolute():
            path = directory / path
        context[f"artifact.{artifact_id}"] = str(path)
        if not path.exists():
            problems.append(f"missing {path}")
            continue
        expected_size = artifact.get("size_bytes")
        if expected_size is not None:
            wanted = int(expand(str(expected_size), context))
            actual = path.stat().st_size
            if actual != wanted:
                problems.append(f"{path}: size is {actual} bytes, expected {wanted}")
        size_product = artifact.get("size_bytes_product")
        if size_product is not None:
            try:
                if not isinstance(size_product, list) or not size_product:
                    raise ValueError("size_bytes_product must be a non-empty list")
                wanted = 1
                for factor in size_product:
                    wanted *= int(expand(str(factor), context))
                actual = path.stat().st_size
                if actual != wanted:
                    problems.append(f"{path}: size is {actual} bytes, expected {wanted}")
            except (TypeError, ValueError):
                problems.append(f"cannot derive expected size for {path} from size_bytes_product")
        size_metadata = artifact.get("size_bytes_from_metadata")
        if size_metadata:
            size_metadata_path = Path(expand(str(size_metadata["path"]), context))
            if not size_metadata_path.is_absolute():
                size_metadata_path = directory / size_metadata_path
            try:
                size_values = json.loads(size_metadata_path.read_text())
                size_value = size_values
                for component in str(size_metadata["key"]).split("."):
                    size_value = size_value[component]
                multiplier = int(expand(str(size_metadata.get("multiplier", 1)), context))
                add = int(expand(str(size_metadata.get("add", 0)), context))
                wanted = int(size_value) * multiplier + add
                actual = path.stat().st_size
                if actual != wanted:
                    problems.append(f"{path}: size is {actual} bytes, expected {wanted}")
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                problems.append(f"cannot derive expected size for {path} from {size_metadata_path}")
        metadata = artifact.get("metadata")
        if metadata:
            metadata_path = Path(expand(str(metadata.get("path", str(path) + ".mtd")), context))
            if not metadata_path.is_absolute():
                metadata_path = directory / metadata_path
            expected = expand_map(metadata.get("expect"), context)
            matches, problem = metadata_matches(metadata_path, expected)
            allow_fallback = str(expand(str(metadata.get("allow_fallback", "false")),
                                        context)).lower() == "true"
            if not matches and allow_fallback and metadata.get("fallback_expect"):
                fallback = expand_map(metadata.get("fallback_expect"), context)
                matches, _ = metadata_matches(metadata_path, fallback)
            if not matches:
                problems.append(problem)
    return problems


def remove_preparation_outputs(dataset, directory, context):
    for item in dataset.get("prepare", {}).get("clean", []):
        path = Path(expand(str(item), context))
        if not path.is_absolute():
            path = directory / path
        try:
            path.resolve().relative_to(directory.resolve())
        except ValueError as error:
            raise ValueError(f"Refusing to clean path outside dataset directory: {path}") from error
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def prepare_dataset(dataset_id, dataset, base_context, plan_dir, global_env):
    if not _VALID_ID.match(dataset_id):
        raise ValueError(f"Invalid dataset id {dataset_id!r}")
    parameters = dataset.get("parameters", {})
    identity = json.dumps({"id": dataset_id, "parameters": parameters,
                           "prepare": dataset.get("prepare", {}).get("command")},
                          sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()[:12]
    context = dict(base_context)
    context["dataset.id"] = dataset_id
    context["dataset.fingerprint"] = fingerprint
    flatten("dataset", parameters, context)
    default_dir = f"${{plan.root}}/datasets/{dataset_id}-{fingerprint}"
    directory = Path(expand(str(dataset.get("directory", default_dir)), context)).resolve()
    context["dataset.dir"] = str(directory)
    problems = dataset_status(dataset, directory, context)
    if not problems:
        return directory, context

    preparation = dataset.get("prepare")
    if not preparation:
        raise RuntimeError(f"Dataset {dataset_id} is not ready: {'; '.join(problems)}")
    policy = preparation.get("policy", "ask")
    should_prepare = policy == "auto"
    if policy == "ask":
        if not sys.stdin.isatty():
            raise RuntimeError(f"Dataset {dataset_id} is not ready and cannot prompt: {'; '.join(problems)}")
        print(f"Dataset {dataset_id} is not ready:\n  " + "\n  ".join(problems), file=sys.stderr)
        answer = input("[r]egenerate, [u]se intentionally despite mismatch, or [a]bort? ").strip().lower()
        if answer in ("u", "use"):
            print(f"WARNING: intentionally using incompatible dataset {dataset_id}", file=sys.stderr)
            return directory, context
        should_prepare = answer in ("r", "regenerate")
    elif policy != "fail" and policy != "auto":
        raise ValueError(f"Unknown preparation policy {policy!r}")
    if not should_prepare:
        raise RuntimeError(f"Dataset {dataset_id} is not ready: {'; '.join(problems)}")

    directory.mkdir(parents=True, exist_ok=True)
    remove_preparation_outputs(dataset, directory, context)
    command = expand(str(preparation["command"]), context)
    env = dict(global_env)
    env.update(expand_map(preparation.get("env"), context))
    prep_log = directory / "prepare.log"
    print(f"Preparing {dataset_id} in {directory}; follow {prep_log} for progress.", file=sys.stderr)
    rc = command_run(command, plan_dir, env, prep_log)
    if rc:
        raise RuntimeError(f"Generator for {dataset_id} failed ({rc}); see {prep_log}")
    problems = dataset_status(dataset, directory, context)
    if problems:
        raise RuntimeError(f"Generator for {dataset_id} did not produce valid data: {'; '.join(problems)}")
    return directory, context


def prepare_dataset_variant(dataset_id, variant_id, variant, directory, context,
                            plan_dir, global_env):
    """Ensure run-specific dataset artifacts, such as a native blocksize, exist."""
    problems = dataset_status(variant, directory, context)
    if not problems:
        return context

    preparation = variant.get("prepare")
    description = f"{dataset_id} {variant_id}={context[f'run.{variant_id}']}"
    if not preparation:
        raise RuntimeError(f"Dataset variant {description} is not ready: {'; '.join(problems)}")
    policy = preparation.get("policy", "ask")
    should_prepare = policy == "auto"
    if policy == "ask":
        if not sys.stdin.isatty():
            raise RuntimeError(f"Dataset variant {description} is not ready and cannot prompt: "
                               f"{'; '.join(problems)}")
        print(f"Dataset variant {description} is not ready:\n  " + "\n  ".join(problems),
              file=sys.stderr)
        answer = input("[r]egenerate, [u]se intentionally despite mismatch, or [a]bort? ").strip().lower()
        if answer in ("u", "use"):
            print(f"WARNING: intentionally using incompatible dataset variant {description}",
                  file=sys.stderr)
            return context
        should_prepare = answer in ("r", "regenerate")
    elif policy not in ("fail", "auto"):
        raise ValueError(f"Unknown preparation policy {policy!r}")
    if not should_prepare:
        raise RuntimeError(f"Dataset variant {description} is not ready: {'; '.join(problems)}")

    directory.mkdir(parents=True, exist_ok=True)
    remove_preparation_outputs(variant, directory, context)
    command = expand(str(preparation["command"]), context)
    env = dict(global_env)
    env.update(expand_map(preparation.get("env"), context))
    value = context[f"run.{variant_id}"]
    prep_log = directory / f"prepare-{variant_id}-{value}.log"
    print(f"Preparing dataset variant {description} in {directory}; follow {prep_log} for progress.",
          file=sys.stderr)
    rc = command_run(command, plan_dir, env, prep_log)
    if rc:
        raise RuntimeError(f"Generator for dataset variant {description} failed ({rc}); "
                           f"see {prep_log}")
    problems = dataset_status(variant, directory, context)
    if problems:
        raise RuntimeError(f"Generator for dataset variant {description} did not produce valid "
                           f"data: {'; '.join(problems)}")
    return context


def run_dataset_ids(run):
    """Return and validate the dataset IDs declared by one configured run."""
    value = run.get("dataset", "")
    values = value if isinstance(value, list) else [value]
    run_id = str(run.get("id", ""))
    if not values:
        raise ValueError(f"Run {run_id} dataset list must not be empty")
    dataset_ids = []
    seen = set()
    for dataset in values:
        dataset_id = str(dataset)
        if not _VALID_ID.match(dataset_id):
            raise ValueError(f"Run {run_id} has invalid dataset id {dataset_id!r}")
        if dataset_id in seen:
            raise ValueError(f"Run {run_id} repeats dataset {dataset_id!r}")
        seen.add(dataset_id)
        dataset_ids.append(dataset_id)
    return dataset_ids


def expand_dataset_cases(runs):
    """Expand list-valued dataset declarations before other run dimensions."""
    cases = []
    for configured_run in runs:
        if not isinstance(configured_run.get("dataset"), list):
            cases.append(configured_run)
            continue
        dataset_ids = run_dataset_ids(configured_run)
        configured_id = str(configured_run.get("id", ""))
        logical_base_id = str(configured_run.get("base_id", configured_id))
        for dataset_id in dataset_ids:
            run = copy.deepcopy(configured_run)
            run["id"] = f"{configured_id}-{dataset_id}"
            run["base_id"] = logical_base_id
            run["dataset"] = dataset_id
            cases.append(run)
    return cases


def expand_run_cases(runs, templates=None):
    """Expand dataset and legacy or backend-specific blocksize sweeps into run cases."""
    templates = templates or {}
    cases = []
    for configured_run in expand_dataset_cases(runs):
        sweeps = configured_run.get("blocksize_sweeps")
        if sweeps is not None:
            if not isinstance(sweeps, dict) or not sweeps:
                raise ValueError(f"Run {configured_run.get('id', '')} blocksize_sweeps must be a "
                                 "non-empty mapping")
            implementations = configured_run.get("implementations", [])
            resolved = [(implementation, resolve_implementation(implementation, templates))
                        for implementation in implementations]
            referenced_sweeps = {
                str(implementation["blocksize_sweep"])
                for _, implementation in resolved
                if implementation.get("enabled", True) and implementation.get("blocksize_sweep")
            }
            unknown = referenced_sweeps - {str(name) for name in sweeps}
            if unknown:
                raise ValueError(f"Run {configured_run.get('id', '')} implementations refer to "
                                 f"unknown blocksize sweeps: {', '.join(sorted(unknown))}")
            neutral = [original for original, implementation in resolved
                       if implementation.get("enabled", True)
                       and not implementation.get("blocksize_sweep")]
            if neutral:
                run = copy.deepcopy(configured_run)
                base_id = str(run.get("id", ""))
                run["id"] = f"{base_id}-baseline"
                run["base_id"] = str(run.get("base_id", base_id))
                run["_baseline_case"] = True
                run["implementations"] = copy.deepcopy(neutral)
                run["setup"] = copy.deepcopy(run.get("baseline_setup", []))
                run.setdefault("parameters", {}).pop("blocksize", None)
                run["parameters"].pop("blocksize_sweep", None)
                cases.append(run)
            for sweep_id, blocksizes in sweeps.items():
                sweep_id = str(sweep_id)
                if not _VALID_ID.match(sweep_id):
                    raise ValueError(f"Run {configured_run.get('id', '')} has invalid blocksize "
                                     f"sweep id {sweep_id!r}")
                if sweep_id not in referenced_sweeps:
                    continue
                if not isinstance(blocksizes, list) or not blocksizes:
                    raise ValueError(f"Run {configured_run.get('id', '')} blocksize sweep "
                                     f"{sweep_id!r} must be a non-empty list")
                selected = [original for original, implementation in resolved
                            if implementation.get("enabled", True)
                            and str(implementation.get("blocksize_sweep")) == sweep_id]
                seen = set()
                for blocksize in blocksizes:
                    blocksize = validate_blocksize(configured_run.get("id", ""), blocksize)
                    if blocksize in seen:
                        raise ValueError(f"Run {configured_run.get('id', '')} blocksize sweep "
                                         f"{sweep_id!r} repeats blocksize {blocksize}")
                    seen.add(blocksize)
                    run = copy.deepcopy(configured_run)
                    base_id = str(run.get("id", ""))
                    run["id"] = f"{base_id}-{sweep_id}-bs{blocksize}"
                    run["base_id"] = str(run.get("base_id", base_id))
                    run.setdefault("parameters", {})["blocksize"] = blocksize
                    run["parameters"]["blocksize_sweep"] = sweep_id
                    run["implementations"] = copy.deepcopy(selected)
                    cases.append(run)
            continue

        resolved_implementations = [resolve_implementation(implementation, templates)
                                    for implementation in configured_run.get("implementations", [])]
        grouped = [implementation for implementation in resolved_implementations
                   if implementation.get("enabled", True)
                   and implementation.get("blocksize_sweep")]
        if grouped:
            raise ValueError(f"Run {configured_run.get('id', '')} uses blocksize-specific "
                             "implementations but has no blocksize_sweeps mapping")
        parameters = configured_run.get("parameters", {})
        blocksizes = parameters.get("blocksize")
        if not isinstance(blocksizes, list):
            if blocksizes is not None:
                validate_blocksize(configured_run.get("id", ""), blocksizes)
            cases.append(configured_run)
            continue
        if not blocksizes:
            raise ValueError(f"Run {configured_run.get('id', '')} blocksize list must not be empty")
        seen = set()
        for blocksize in blocksizes:
            blocksize = validate_blocksize(configured_run.get("id", ""), blocksize)
            if blocksize in seen:
                raise ValueError(f"Run {configured_run.get('id', '')} repeats blocksize {blocksize}")
            seen.add(blocksize)
            run = copy.deepcopy(configured_run)
            base_id = str(run.get("id", ""))
            run["id"] = f"{base_id}-bs{blocksize}"
            run["base_id"] = str(run.get("base_id", base_id))
            run["parameters"]["blocksize"] = blocksize
            cases.append(run)
    return cases


def validate_blocksize(run_id, value):
    if isinstance(value, bool) or isinstance(value, float) and not value.is_integer():
        raise ValueError(f"Run {run_id} has invalid blocksize {value!r}")
    try:
        blocksize = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Run {run_id} has invalid blocksize {value!r}") from error
    if blocksize < 1:
        raise ValueError(f"Run {run_id} blocksize must be positive")
    return blocksize


def write_scope_runner(path):
    path.write_text(r"""#!/usr/bin/env bash
set +e
log="$1"; metrics="$2"; run_timeout="$3"; grace="$4"; shift 4
time_bin="${TIME_BIN:-/usr/bin/time}"
printf 'started_at=%s timeout_seconds=%s grace_seconds=%s\n' "$(date -Is)" "$run_timeout" "$grace" >"$log"
if [[ -x "$time_bin" ]]; then
  if [[ "$run_timeout" == 0 ]]; then
    "$time_bin" -v -o "${metrics}.time" bash -lc "$1" >>"$log" 2>&1
  else
    "$time_bin" -v -o "${metrics}.time" timeout --kill-after="$grace" "$run_timeout" bash -lc "$1" >>"$log" 2>&1
  fi
elif [[ "$run_timeout" == 0 ]]; then
  bash -lc "$1" >>"$log" 2>&1
else
  timeout --kill-after="$grace" "$run_timeout" bash -lc "$1" >>"$log" 2>&1
fi
rc=$?
cg=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
root="/sys/fs/cgroup${cg}"
{
  echo "exit_status=$rc"
  echo "memory_current_bytes=$(cat "$root/memory.current")"
  echo "memory_peak_bytes=$(cat "$root/memory.peak")"
  echo "memory_events=$(tr '\n' ';' < "$root/memory.events")"
  [[ -r "$root/io.stat" ]] && echo "io_stat=$(tr '\n' ';' < "$root/io.stat")"
} > "$metrics"
exit "$rc"
""")
    path.chmod(0o755)


def metric(path, pattern, default="nan"):
    try:
        match = re.search(pattern, path.read_text(errors="replace"), re.MULTILINE)
        return match.group(1) if match else default
    except OSError:
        return default


def reported_seconds(log):
    """Return an implementation's optional internal execution timer."""
    text = log.read_text(errors="replace")
    matches = re.findall(r'"seconds"\s*:\s*([0-9.eE+-]+)', text)
    if not matches:
        matches = re.findall(r"Total execution time:\s*([0-9.]+)", text)
    return matches[-1] if matches else "nan"


def elapsed_seconds(time_path):
    """Return GNU time's elapsed wall time in seconds, including process startup."""
    value = metric(time_path, r"^\s*Elapsed .*\):\s*((?:[0-9]+:){0,2}[0-9]+(?:\.[0-9]+)?)\s*$")
    if value == "nan":
        return value
    try:
        fields = [float(field) for field in value.split(":")]
    except ValueError:
        return "nan"
    seconds = 0.0
    for field in fields:
        seconds = seconds * 60.0 + field
    return f"{seconds:.6f}"


def resolve_implementation(implementation, templates):
    template_name = implementation.get("template")
    if not template_name:
        return implementation
    if template_name not in templates:
        raise ValueError(f"Unknown implementation template {template_name!r}")
    resolved = dict(templates[template_name])
    template_env = dict(resolved.get("environment", {}))
    resolved.update({key: value for key, value in implementation.items() if key != "template"})
    template_env.update(implementation.get("environment", {}))
    if template_env:
        resolved["environment"] = template_env
    return resolved


def drop_caches(paths, context, plan_dir, python, log):
    expanded = [expand(str(path), context) for path in paths]
    if not expanded:
        return
    script = plan_dir / "drop_caches.py"
    with open(log, "a", encoding="utf-8") as output:
        result = subprocess.run([python, str(script), *expanded], stdout=output,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Cold-cache precondition failed ({result.returncode}); see {log}")


def execute_plan(plan_path, validate_only=False):
    plan = yaml.safe_load(plan_path.read_text())
    if not isinstance(plan, dict) or plan.get("version") != 1:
        raise ValueError("Benchmark plan must be a mapping with version: 1")
    plan_dir = plan_path.parent.resolve()
    context = {"plan.dir": str(plan_dir)}
    root = Path(expand(str(plan.get("root", plan_dir.parent / "benchmark-data")), context)).resolve()
    context["plan.root"] = str(root)
    flatten("tools", plan.get("tools", {}), context)
    context.setdefault("tools.python", sys.executable)
    global_env = dict(os.environ)
    global_env.update(expand_map(plan.get("environment"), context))

    datasets = plan.get("datasets", {})
    templates = plan.get("templates", {})
    configured_runs = plan.get("runs", [])
    for run in configured_runs:
        run_id = str(run.get("id", ""))
        if not _VALID_ID.match(run_id):
            raise ValueError(f"Invalid run id {run_id!r}")
        for dataset_id in run_dataset_ids(run):
            if dataset_id not in datasets:
                raise ValueError(f"Run {run_id} refers to unknown dataset {dataset_id}")
        if not run.get("implementations"):
            raise ValueError(f"Run {run_id} has no implementations")
        for implementation in run["implementations"]:
            resolve_implementation(implementation, templates)
    expanded_runs = expand_run_cases(configured_runs, templates)
    enabled_runs = [run for run in expanded_runs if run.get("enabled", True) and any(
        resolve_implementation(implementation, templates).get("enabled", True)
        for implementation in run.get("implementations", []))]
    run_ids = set()
    for run in expanded_runs:
        run_id = str(run.get("id", ""))
        if not _VALID_ID.match(run_id):
            raise ValueError(f"Invalid run id {run_id!r}")
        if run_id in run_ids:
            raise ValueError(f"Duplicate expanded run id {run_id!r}")
        run_ids.add(run_id)
    if validate_only:
        print(f"Valid benchmark plan: {len(datasets)} datasets, {len(enabled_runs)} enabled run cases")
        return 0
    required_tools = set()
    for run in enabled_runs:
        for implementation in run.get("implementations", []):
            resolved = resolve_implementation(implementation, templates)
            if resolved.get("enabled", True):
                required_tools.update(str(tool) for tool in resolved.get("required_tools", []))
    for tool_id in sorted(required_tools):
        context_key = f"tools.{tool_id}"
        if context_key not in context:
            raise ValueError(f"Required tool {tool_id!r} is not configured")
        executable = expand(context[context_key], context)
        if not shutil.which(executable):
            raise RuntimeError(f"Required tool {tool_id!r} is not executable: {executable}")
    if not shutil.which("systemd-run") or subprocess.run(
            ["systemctl", "--user", "status"], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL).returncode:
        raise RuntimeError("A working user systemd instance and systemd-run are required")

    root.mkdir(parents=True, exist_ok=True)
    prepared = {}
    results_root = Path(expand(str(plan.get("results", "${plan.root}/results")), context)).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    invocation_id = execution_timestamp()
    invocation_dir = results_root / invocation_id
    invocation_dir.mkdir()
    shutil.copy2(plan_path, invocation_dir / "benchmark-plan.yaml")

    for run in enabled_runs:
        run_id = str(run["id"])
        if not _VALID_ID.match(run_id):
            raise ValueError(f"Invalid run id {run_id!r}")
        dataset_id = str(run["dataset"])
        if dataset_id not in datasets:
            raise ValueError(f"Run {run_id} refers to unknown dataset {dataset_id}")
        if dataset_id not in prepared:
            prepared[dataset_id] = prepare_dataset(dataset_id, datasets[dataset_id], context,
                                                    plan_dir, global_env)
        _, dataset_context = prepared[dataset_id]
        run_context = dict(dataset_context)
        run_context["run.id"] = run_id
        run_context["run.base_id"] = str(run.get("base_id", run_id))
        run_context["run.invocation_id"] = invocation_id
        flatten("run", run.get("parameters", {}), run_context)
        if not run.get("_baseline_case"):
            for variant_id, variant in datasets[dataset_id].get("variants", {}).items():
                context_key = f"run.{variant_id}"
                if context_key not in run_context:
                    raise ValueError(f"Run {run_id} must define parameters.{variant_id} for dataset "
                                     f"variant {dataset_id}.{variant_id}")
                prepare_dataset_variant(dataset_id, variant_id, variant, prepared[dataset_id][0],
                                        run_context, plan_dir, global_env)
        resources = dict(plan.get("defaults", {}).get("resources", {}))
        resources.update(run.get("resources", {}))
        threads = resources.get("threads", "auto")
        if str(threads).lower() == "auto":
            threads = len(os.sched_getaffinity(0))
        else:
            threads = int(threads)
        if threads < 1:
            raise ValueError(f"Run {run_id} resources.threads must be a positive integer or auto")
        resources["threads"] = threads
        spark_threads = resources.get("spark_threads", threads)
        if str(spark_threads).lower() == "auto":
            spark_threads = threads
        else:
            spark_threads = int(spark_threads)
        if spark_threads < 1:
            raise ValueError(f"Run {run_id} resources.spark_threads must be a positive integer "
                             "or auto")
        resources["spark_threads"] = spark_threads
        dask_threads = resources.get("dask_threads", threads)
        if str(dask_threads).lower() == "auto":
            dask_threads = threads
        else:
            dask_threads = int(dask_threads)
        if dask_threads < 1:
            raise ValueError(f"Run {run_id} resources.dask_threads must be a positive integer "
                             "or auto")
        resources["dask_threads"] = dask_threads
        flatten("resources", resources, run_context)
        run_context["run.entrypoint"] = expand(str(run.get("entrypoint", "")), run_context)
        for name, value in run.get("inputs", {}).items():
            if isinstance(value, dict) and "artifact" in value:
                artifact_key = f"artifact.{value['artifact']}"
                if artifact_key not in run_context:
                    if run.get("_baseline_case"):
                        continue
                    raise ValueError(f"Run {run_id} refers to unknown artifact {value['artifact']}")
                run_context[f"input.{name}"] = run_context[artifact_key]
            else:
                run_context[f"input.{name}"] = expand(str(value), run_context)

        # Keep previous measurements immutable, and group all cases of one benchmark
        # invocation together for straightforward comparison.
        run_dir = invocation_dir / run_id
        logs = run_dir / "logs"
        outputs = run_dir / "outputs"
        logs.mkdir(parents=True, exist_ok=True)
        outputs.mkdir(parents=True, exist_ok=True)
        run_context["run.results"] = str(run_dir)
        run_context["run.outputs"] = str(outputs)
        setup_steps = run.get("setup", [])
        if isinstance(setup_steps, (str, dict)):
            setup_steps = [setup_steps]
        for number, step in enumerate(setup_steps, 1):
            step = {"command": step} if isinstance(step, str) else step
            command = expand(str(step["command"]), run_context)
            env = dict(global_env)
            env.update(expand_map(step.get("environment"), run_context))
            setup_log = logs / f"setup-{number}.log"
            rc = command_run(command, plan_dir, env, setup_log)
            if rc:
                raise RuntimeError(f"Setup for run {run_id} failed ({rc}); see {setup_log}")
        (run_dir / "resolved-context.json").write_text(json.dumps(run_context, indent=2, sort_keys=True))
        scope_runner = run_dir / ".scope-runner.sh"
        write_scope_runner(scope_runner)
        csv_path = run_dir / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["run", "implementation", "comparable", "rep", "status", "wall_seconds",
                             "algorithm_seconds", "memory_peak_bytes", "major_faults",
                             "file_system_inputs", "log", "metrics"])
            repetitions = int(run.get("repetitions", plan.get("defaults", {}).get("repetitions", 1)))
            timeout_seconds = int(resources.get("timeout_seconds", 0))
            grace_seconds = int(resources.get("timeout_grace_seconds", 30))
            if timeout_seconds < 0 or grace_seconds < 0:
                raise ValueError(f"Run {run_id} timeout values must be non-negative")
            timeout = str(timeout_seconds)
            memory = str(resources["memory_max"])
            swap = str(resources.get("swap_max", 0))
            cache_paths = run.get("cold_cache", ["${dataset.dir}"])
            for configured_implementation in run.get("implementations", []):
                implementation = resolve_implementation(configured_implementation, templates)
                if not implementation.get("enabled", True):
                    continue
                impl_id = str(implementation["id"])
                for rep in range(1, repetitions + 1):
                    local_context = dict(run_context)
                    local_context["rep"] = str(rep)
                    local_context["implementation.id"] = impl_id
                    command = expand(str(implementation["command"]), local_context)
                    tag = f"{run_id}-{impl_id}-r{rep}"
                    log = logs / f"{tag}.log"
                    metrics = logs / f"{tag}.metrics"
                    print(f"Starting {tag} (timeout={timeout_seconds}s, memory={memory}, "
                          f"results={run_dir})", flush=True)
                    drop_caches(cache_paths, local_context, plan_dir, context["tools.python"],
                                logs / "drop-caches.log")
                    env = dict(global_env)
                    env.update(expand_map(run.get("environment"), local_context))
                    env.update(expand_map(implementation.get("environment"), local_context))
                    unit = re.sub(r"[^A-Za-z0-9_.-]", "-", f"ooc-{tag}-{os.getpid()}")
                    scope = ["systemd-run", "--user", "--scope", "--collect", "--quiet", f"--unit={unit}",
                             "-p", "MemoryAccounting=yes", "-p", f"MemoryMax={memory}",
                             "-p", f"MemorySwapMax={swap}", "-p", "TasksMax=infinity",
                             "-p", "KillMode=control-group", "-p", "SendSIGKILL=yes",
                             "-p", f"TimeoutStopSec={grace_seconds}s"]
                    if timeout_seconds:
                        scope += ["-p", f"RuntimeMaxSec={timeout_seconds + grace_seconds + 15}s"]
                    scope += [str(scope_runner), str(log), str(metrics), timeout, str(grace_seconds), command]
                    rc = subprocess.run(scope, cwd=plan_dir, env=env).returncode
                    status = "ok" if rc == 0 else "timeout" if rc == 124 else "killed" if rc == 137 else "failed"
                    if status == "ok" and "An Error Occurred" in log.read_text(errors="replace"):
                        status = "failed"
                    time_path = Path(str(metrics) + ".time")
                    wall_seconds = elapsed_seconds(time_path)
                    algorithm_seconds = reported_seconds(log)
                    peak = metric(metrics, r"^memory_peak_bytes=([0-9]+)$")
                    faults = metric(time_path, r"Major \(requiring I/O\) page faults:\s*([0-9]+)")
                    inputs = metric(time_path, r"File system inputs:\s*([0-9]+)")
                    writer.writerow([run_id, impl_id, implementation.get("comparable", True), rep, status,
                                     wall_seconds, algorithm_seconds, peak, faults, inputs, log, metrics])
                    csv_file.flush()
                    print(f"{tag}: {status} (wall={wall_seconds}s, peak={peak})")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--validate", action="store_true", help="validate structure without preparing or running")
    args = parser.parse_args()
    try:
        return execute_plan(args.plan.resolve(), args.validate)
    except (KeyError, ValueError, RuntimeError, OSError) as error:
        print(f"benchmark plan error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
