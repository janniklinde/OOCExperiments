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
import math
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
_DEFAULT_TEMPORARY_PATHS = [
    "${run.results}/systemds-tmp",
    "${run.results}/systemds-scratch",
    "${run.results}/spark-local",
    "${run.results}/java-tmp",
    "${run.results}/dask-spill",
    "${run.results}/python-tmp",
]

_RETENTION_METRICS = {
    "multilogreg": ("coefficient_norm",),
    "gnmf": ("w_checksum", "h_checksum"),
    "kmeans": ("inertia",),
    "pca": ("score_norm_sq",),
    "lmcg": ("residual_norm",),
    "l2svm": ("model_norm",),
}


def _validation_workload(base_id):
    for prefix, metrics in _RETENTION_METRICS.items():
        if base_id == prefix or base_id.startswith(prefix + "_"):
            return prefix, metrics
    return None, ()


def _log_validation_values(log, metric_names):
    aliases = {
        "w_checksum": r"W checksum:\s*([^\s]+)",
        "h_checksum": r"H checksum:\s*([^\s]+)",
    }
    text = log.read_text(encoding="utf-8", errors="replace")
    values = {}
    for name in metric_names:
        pattern = aliases.get(name, rf"{re.escape(name)}=([^\s]+)")
        matches = re.findall(pattern, text)
        if matches:
            try:
                values[name] = float(matches[-1])
            except ValueError:
                pass
    return values


def _json_validation_values(outputs, impl_id, rep, metric_names):
    candidates = sorted(outputs.glob(f"*-r{rep}.json"))
    if "dask" in impl_id:
        candidates = [path for path in candidates if path.name.startswith("dask-")]
    else:
        candidates = [path for path in candidates if not path.name.startswith("dask-")]
    for path in candidates:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if all(name in report for name in metric_names):
            try:
                return {name: float(report[name]) for name in metric_names}
            except (TypeError, ValueError):
                continue
    return {}


def _values_agree(summaries, relative_tolerance=1e-5, absolute_tolerance=1e-8):
    reference = summaries[0]
    return all(
        math.isclose(summary[name], reference[name], rel_tol=relative_tolerance,
                     abs_tol=absolute_tolerance)
        for summary in summaries[1:] for name in reference
    )


def apply_output_retention(records, invocation_dir):
    """Discard validated numeric artifacts while preserving anything diagnostically useful."""
    groups = {}
    for record in records:
        key = (record["base_id"], record["dataset"], record.get("resource_profile"),
               record.get("parameter_case"), record["rep"])
        groups.setdefault(key, []).append(record)

    invocation_report = []
    for key, members in groups.items():
        _, metric_names = _validation_workload(key[0])
        summaries = []
        reason = "validated"
        if not metric_names:
            reason = "no validation contract"
        elif len(members) < 2:
            reason = "fewer than two comparable executions"
        elif any(member["status"] != "ok" for member in members):
            reason = "one or more executions did not complete"
        else:
            for member in members:
                if member["impl_id"].startswith("systemds-"):
                    values = _log_validation_values(member["log"], metric_names)
                else:
                    values = _json_validation_values(
                        member["outputs"], member["impl_id"], member["rep"], metric_names)
                member["validation_values"] = values
                if set(values) != set(metric_names) or not all(
                        math.isfinite(value) for value in values.values()):
                    reason = f"missing or non-finite validation evidence for {member['impl_id']}"
                    break
                summaries.append(values)
            if reason == "validated" and not _values_agree(summaries):
                reason = "numerical outputs diverged"

        discard = reason == "validated"
        run_reports = {}
        for member in members:
            removed_bytes = 0
            if discard:
                for artifact in list(member["outputs"].iterdir()):
                    if artifact.suffix == ".json":
                        continue
                    if artifact.is_dir():
                        removed_bytes += sum(path.stat().st_size for path in artifact.rglob("*")
                                             if path.is_file())
                        shutil.rmtree(artifact)
                    else:
                        removed_bytes += artifact.stat().st_size
                        artifact.unlink()
            report = run_reports.setdefault(member["run_dir"], {
                "retention": "compact" if discard else "full", "reason": reason,
                "removed_bytes": 0, "executions": [],
            })
            report["removed_bytes"] += removed_bytes
            report["executions"].append({
                "implementation": member["impl_id"],
                "validation_values": member.get("validation_values", {}),
            })
        for run_dir, report in run_reports.items():
            (run_dir / "output-retention.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        invocation_report.append({
            "group": {"base_id": key[0], "dataset": key[1], "resource_profile": key[2],
                      "parameter_case": key[3], "rep": key[4]},
            "retention": "compact" if discard else "full",
            "reason": reason,
        })
    (invocation_dir / "output-validation.json").write_text(
        json.dumps(invocation_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execution_timestamp(moment=None):
    """Return a filesystem-safe, timezone-qualified benchmark invocation ID."""
    moment = moment or datetime.now().astimezone()
    return moment.strftime("%Y%m%dT%H%M%S.%f%z")


def file_digest(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_invocation_metadata(path, plan_dir, root, context):
    """Record compact host/tool/source provenance once per benchmark invocation."""
    source_files = {}
    for source in sorted(plan_dir.rglob("*")):
        if (not source.is_file() or source.is_symlink() or "__pycache__" in source.parts
                or source.suffix in {".pyc", ".pyo"}):
            continue
        source_files[str(source.relative_to(plan_dir))] = {
            "bytes": source.stat().st_size,
            "sha256": file_digest(source),
        }
    tool_files = {}
    for tool_id in ("systemds_jar",):
        configured = context.get(f"tools.{tool_id}")
        if not configured:
            continue
        candidate = Path(expand(configured, context)).resolve()
        if candidate.is_file():
            tool_files[tool_id] = {
                "path": str(candidate), "bytes": candidate.stat().st_size,
                "sha256": file_digest(candidate),
            }
    statvfs = os.statvfs(root)
    uname = os.uname()
    metadata = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "host": {
            "hostname": uname.nodename,
            "kernel": {"sysname": uname.sysname, "release": uname.release,
                       "version": uname.version, "machine": uname.machine},
            "logical_cpus": os.cpu_count(),
            "runner_cpu_affinity": sorted(os.sched_getaffinity(0)),
            "page_size_bytes": os.sysconf("SC_PAGE_SIZE"),
            "load_average": os.getloadavg(),
        },
        "filesystem": {
            "path": str(root),
            "block_size_bytes": statvfs.f_frsize,
            "total_bytes": statvfs.f_blocks * statvfs.f_frsize,
            "available_bytes": statvfs.f_bavail * statvfs.f_frsize,
        },
        "runner": {"python": sys.executable, "version": sys.version},
        "configured_tools": {key.removeprefix("tools."): value
                             for key, value in context.items() if key.startswith("tools.")},
        "tool_files": tool_files,
        "suite_sources": source_files,
    }
    for proc_name in ("meminfo", "cpuinfo"):
        proc_path = Path("/proc") / proc_name
        try:
            text = proc_path.read_text(errors="replace")
        except OSError:
            continue
        if proc_name == "cpuinfo":
            model = next((line.partition(":")[2].strip() for line in text.splitlines()
                          if line.lower().startswith("model name")), None)
            metadata["host"]["cpu_model"] = model
        else:
            wanted = {"MemTotal", "SwapTotal", "HugePages_Total", "Hugepagesize"}
            metadata["host"]["memory"] = {
                key.rstrip(":"): value.strip()
                for line in text.splitlines() if (key := line.partition(":")[0]) in wanted
                for value in [line.partition(":")[2]]
            }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


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


def named_mapping(plan, key):
    """Return and structurally validate a top-level mapping of named definitions."""
    values = plan.get(key, {})
    if not isinstance(values, dict):
        raise ValueError(f"{key} must be a mapping")
    for value_id in values:
        if not _VALID_ID.match(str(value_id)):
            raise ValueError(f"Invalid {key} id {value_id!r}")
    return values


def dataset_selection(run, dataset_groups=None):
    """Resolve a scalar/list dataset selection or a named dataset group."""
    value = run.get("dataset", "")
    if not isinstance(value, dict):
        return value
    if set(value) != {"group"}:
        raise ValueError(f"Run {run.get('id', '')} dataset mapping must contain only group")
    group_id = str(value["group"])
    if not _VALID_ID.match(group_id):
        raise ValueError(f"Run {run.get('id', '')} has invalid dataset group id {group_id!r}")
    dataset_groups = dataset_groups or {}
    if group_id not in dataset_groups:
        raise ValueError(f"Run {run.get('id', '')} refers to unknown dataset group {group_id}")
    return dataset_groups[group_id]


def run_dataset_ids(run, dataset_groups=None):
    """Return and validate the dataset IDs declared by one configured run."""
    value = dataset_selection(run, dataset_groups)
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


def expand_dataset_cases(runs, dataset_groups=None):
    """Expand list-valued dataset declarations before other run dimensions."""
    cases = []
    for configured_run in runs:
        selection = dataset_selection(configured_run, dataset_groups)
        if not isinstance(selection, list):
            cases.append(configured_run)
            continue
        dataset_ids = run_dataset_ids(configured_run, dataset_groups)
        configured_id = str(configured_run.get("id", ""))
        logical_base_id = str(configured_run.get("base_id", configured_id))
        for dataset_id in dataset_ids:
            run = copy.deepcopy(configured_run)
            run["id"] = f"{configured_id}-{dataset_id}"
            run["base_id"] = logical_base_id
            run["dataset"] = dataset_id
            cases.append(run)
    return cases


def selected_names(run, field, definitions):
    """Validate a scalar/list selection from a named top-level definition mapping."""
    value = run.get(field)
    values = value if isinstance(value, list) else [value]
    run_id = str(run.get("id", ""))
    if not values or any(value is None for value in values):
        raise ValueError(f"Run {run_id} {field} must not be empty")
    selected = []
    seen = set()
    for value in values:
        value_id = str(value)
        if not _VALID_ID.match(value_id):
            raise ValueError(f"Run {run_id} has invalid {field} id {value_id!r}")
        if value_id not in definitions:
            raise ValueError(f"Run {run_id} refers to unknown {field} {value_id}")
        if value_id in seen:
            raise ValueError(f"Run {run_id} repeats {field} {value_id!r}")
        seen.add(value_id)
        selected.append(value_id)
    return selected


def expand_resource_profile_cases(runs, resource_profiles=None):
    """Expand named, correlated resource configurations such as cgroup/heap pairs."""
    resource_profiles = resource_profiles or {}
    cases = []
    for configured_run in runs:
        if "resource_profiles" not in configured_run:
            cases.append(configured_run)
            continue
        for profile_id in selected_names(configured_run, "resource_profiles", resource_profiles):
            profile = resource_profiles[profile_id]
            if not isinstance(profile, dict):
                raise ValueError(f"Resource profile {profile_id} must be a mapping")
            run = copy.deepcopy(configured_run)
            base_id = str(run.get("id", ""))
            run["id"] = f"{base_id}-{profile_id}"
            run["base_id"] = str(run.get("base_id", base_id))
            resources = copy.deepcopy(run.get("resources", {}))
            resources.update(profile)
            run["resources"] = resources
            run["resource_profile"] = profile_id
            run.pop("resource_profiles", None)
            cases.append(run)
    return cases


def parameter_case_values(case_group, case_id, value):
    if not isinstance(value, dict):
        raise ValueError(f"Parameter case {case_group}.{case_id} must be a mapping")
    if "parameters" in value:
        if set(value) != {"id", "parameters"} or not isinstance(value["parameters"], dict):
            raise ValueError(f"Parameter case {case_group}.{case_id} with parameters must contain "
                             "only id and a parameter mapping")
        return copy.deepcopy(value["parameters"])
    return {key: copy.deepcopy(child) for key, child in value.items() if key != "id"}


def expand_parameter_cases(runs, parameter_cases=None):
    """Expand named lists of correlated algorithm parameter mappings."""
    parameter_cases = parameter_cases or {}
    cases = []
    for configured_run in runs:
        if "parameter_cases" not in configured_run:
            cases.append(configured_run)
            continue
        group_id = configured_run["parameter_cases"]
        if not isinstance(group_id, str) or not _VALID_ID.match(group_id):
            raise ValueError(f"Run {configured_run.get('id', '')} parameter_cases must name one "
                             "parameter case group")
        if group_id not in parameter_cases:
            raise ValueError(f"Run {configured_run.get('id', '')} refers to unknown parameter "
                             f"case group {group_id}")
        values = parameter_cases[group_id]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Parameter case group {group_id} must be a non-empty list")
        seen = set()
        for value in values:
            if not isinstance(value, dict) or "id" not in value:
                raise ValueError(f"Parameter case group {group_id} entries must be mappings with id")
            case_id = str(value["id"])
            if not _VALID_ID.match(case_id):
                raise ValueError(f"Invalid parameter case id {case_id!r} in group {group_id}")
            if case_id in seen:
                raise ValueError(f"Parameter case group {group_id} repeats case {case_id!r}")
            seen.add(case_id)
            run = copy.deepcopy(configured_run)
            base_id = str(run.get("id", ""))
            run["id"] = f"{base_id}-{case_id}"
            run["base_id"] = str(run.get("base_id", base_id))
            parameters = copy.deepcopy(run.get("parameters", {}))
            parameters.update(parameter_case_values(group_id, case_id, value))
            run["parameters"] = parameters
            run["parameter_case"] = case_id
            run.pop("parameter_cases", None)
            cases.append(run)
    return cases


def expand_run_cases(runs, templates=None, dataset_groups=None, resource_profiles=None,
                     parameter_cases=None):
    """Expand datasets, resources, parameters, and backend-specific block sizes."""
    templates = templates or {}
    cases = []
    expanded = expand_dataset_cases(runs, dataset_groups)
    expanded = expand_resource_profile_cases(expanded, resource_profiles)
    expanded = expand_parameter_cases(expanded, parameter_cases)
    for configured_run in expanded:
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
log="$1"; metrics="$2"; telemetry="$3"; telemetry_interval="$4"
run_timeout="$5"; grace="$6"; shift 6
time_bin="${TIME_BIN:-/usr/bin/time}"
printf 'started_at=%s timeout_seconds=%s grace_seconds=%s\n' "$(date -Is)" "$run_timeout" "$grace" >"$log"
printf 'exit_status=running\n' >"$metrics"
cg=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
root="/sys/fs/cgroup${cg}"

stat_value() {
  awk -v key="$2" '$1 == key {print $2; found=1; exit} END {if (!found) print 0}' "$1" 2>/dev/null
}

io_totals() {
  awk '{for (i=2; i<=NF; i++) {split($i, value, "="); totals[value[1]] += value[2]}} END {printf "%d,%d,%d,%d,%d,%d", totals["rbytes"], totals["wbytes"], totals["rios"], totals["wios"], totals["dbytes"], totals["dios"]}' "$1" 2>/dev/null
}

cgroup_values() {
  local files=() candidate
  for candidate in "$root/memory.stat" "$root/cpu.stat" "$root/io.stat" \
      "$root/cpu.pressure" "$root/memory.pressure" "$root/io.pressure"; do
    [[ -r "$candidate" ]] && files+=("$candidate")
  done
  awk -v pids="$1" '
    FILENAME ~ /memory\.stat$/ {mem[$1]=$2; next}
    FILENAME ~ /cpu\.stat$/ {cpu[$1]=$2; next}
    FILENAME ~ /io\.stat$/ {
      for (i=2; i<=NF; i++) {split($i, value, "="); io[value[1]] += value[2]}
      next
    }
    FILENAME ~ /\.pressure$/ {
      for (i=2; i<=NF; i++) if ($i ~ /^total=/) {
        split($i, value, "=")
        if (FILENAME ~ /cpu\.pressure$/) kind="cpu"
        else if (FILENAME ~ /memory\.pressure$/) kind="memory"
        else kind="io"
        pressure[kind ":" $1]=value[2]
      }
    }
    END {
      printf "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d", mem["anon"], mem["file"],
        mem["shmem"], mem["file_dirty"], mem["file_writeback"], mem["pgfault"],
        mem["pgmajfault"], mem["workingset_refault_anon"],
        mem["workingset_refault_file"], mem["workingset_activate_file"]
      printf ",%d,%d,%d,%d,%d,%d", cpu["usage_usec"], cpu["user_usec"],
        cpu["system_usec"], cpu["nr_periods"], cpu["nr_throttled"],
        cpu["throttled_usec"]
      printf ",%d,%d,%d,%d,%d,%d,%d", pids, io["rbytes"], io["wbytes"], io["rios"],
        io["wios"], io["dbytes"], io["dios"]
      printf ",%d,%d,%d,%d,%d,%d", pressure["cpu:some"], pressure["cpu:full"],
        pressure["memory:some"], pressure["memory:full"], pressure["io:some"],
        pressure["io:full"]
    }
  ' "${files[@]}" 2>/dev/null
}

sample_cgroup() {
  local now_ns elapsed_ms memory_current memory_peak memory_swap pids values
  now_ns=$(date +%s%N)
  elapsed_ms=$(( (now_ns - telemetry_start_ns) / 1000000 ))
  read -r memory_current <"$root/memory.current" 2>/dev/null || memory_current=0
  read -r memory_peak <"$root/memory.peak" 2>/dev/null || memory_peak=0
  read -r memory_swap <"$root/memory.swap.current" 2>/dev/null || memory_swap=0
  read -r pids <"$root/pids.current" 2>/dev/null || pids=0
  values=$(cgroup_values "$pids")
  printf '%s,%s,%s,%s,%s\n' "$elapsed_ms" "$memory_current" "$memory_peak" \
    "$memory_swap" "$values" >>"$telemetry"
}

monitor_cgroup() {
  local monitored_pid="$1"
  while kill -0 "$monitored_pid" 2>/dev/null; do
    sample_cgroup
    sleep "$telemetry_interval" || break
  done
}

printf '%s\n' 'elapsed_ms,memory_current_bytes,memory_peak_bytes,memory_swap_current_bytes,anon_bytes,file_bytes,shmem_bytes,file_dirty_bytes,file_writeback_bytes,pgfault,pgmajfault,workingset_refault_anon,workingset_refault_file,workingset_activate_file,cpu_usage_usec,cpu_user_usec,cpu_system_usec,cpu_nr_periods,cpu_nr_throttled,cpu_throttled_usec,pids_current,io_read_bytes,io_write_bytes,io_read_ops,io_write_ops,io_discard_bytes,io_discard_ops,cpu_pressure_some_usec,cpu_pressure_full_usec,memory_pressure_some_usec,memory_pressure_full_usec,io_pressure_some_usec,io_pressure_full_usec' >"$telemetry"
telemetry_start_ns=$(date +%s%N)
# Give the benchmark payload a higher OOM score than this small accounting
# wrapper. If MemoryMax is exhausted, the kernel can kill the payload while
# timeout, GNU time, and this wrapper remain alive to record the failure.
payload=(bash -c 'echo 500 > /proc/self/oom_score_adj 2>/dev/null || true; exec bash -lc "$1"' benchmark "$1")
if [[ -x "$time_bin" ]]; then
  if [[ "$run_timeout" == 0 ]]; then
    "$time_bin" -v -o "${metrics}.time" "${payload[@]}" >>"$log" 2>&1 &
  else
    "$time_bin" -v -o "${metrics}.time" timeout --kill-after="$grace" "$run_timeout" "${payload[@]}" >>"$log" 2>&1 &
  fi
elif [[ "$run_timeout" == 0 ]]; then
  "${payload[@]}" >>"$log" 2>&1 &
else
  timeout --kill-after="$grace" "$run_timeout" "${payload[@]}" >>"$log" 2>&1 &
fi
timed_pid=$!
monitor_cgroup "$timed_pid" &
monitor_pid=$!
wait "$timed_pid"
rc=$?
kill "$monitor_pid" 2>/dev/null
wait "$monitor_pid" 2>/dev/null
sample_cgroup
io=$(io_totals "$root/io.stat")
IFS=, read -r io_read_bytes io_write_bytes io_read_ops io_write_ops io_discard_bytes io_discard_ops <<<"$io"
{
  echo "exit_status=$rc"
  echo "memory_current_bytes=$(cat "$root/memory.current")"
  echo "memory_peak_bytes=$(cat "$root/memory.peak")"
  echo "memory_swap_current_bytes=$(cat "$root/memory.swap.current" 2>/dev/null || echo 0)"
  echo "memory_events=$(tr '\n' ';' < "$root/memory.events")"
  echo "cpu_usage_usec=$(stat_value "$root/cpu.stat" usage_usec)"
  echo "cpu_user_usec=$(stat_value "$root/cpu.stat" user_usec)"
  echo "cpu_system_usec=$(stat_value "$root/cpu.stat" system_usec)"
  echo "cpu_nr_throttled=$(stat_value "$root/cpu.stat" nr_throttled)"
  echo "cpu_throttled_usec=$(stat_value "$root/cpu.stat" throttled_usec)"
  echo "io_read_bytes=$io_read_bytes"
  echo "io_write_bytes=$io_write_bytes"
  echo "io_read_ops=$io_read_ops"
  echo "io_write_ops=$io_write_ops"
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


def validate_named_definitions(datasets, dataset_groups, resource_profiles, parameter_cases):
    """Validate reusable definitions even when no enabled run currently selects them."""
    for group_id, values in dataset_groups.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Dataset group {group_id} must be a non-empty list")
        fake_run = {"id": f"dataset-group-{group_id}", "dataset": values}
        for dataset_id in run_dataset_ids(fake_run):
            if dataset_id not in datasets:
                raise ValueError(f"Dataset group {group_id} refers to unknown dataset {dataset_id}")
    for profile_id, values in resource_profiles.items():
        if not isinstance(values, dict):
            raise ValueError(f"Resource profile {profile_id} must be a mapping")
    for group_id, values in parameter_cases.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Parameter case group {group_id} must be a non-empty list")
        seen = set()
        for value in values:
            if not isinstance(value, dict) or "id" not in value:
                raise ValueError(f"Parameter case group {group_id} entries must be mappings with id")
            case_id = str(value["id"])
            if not _VALID_ID.match(case_id):
                raise ValueError(f"Invalid parameter case id {case_id!r} in group {group_id}")
            if case_id in seen:
                raise ValueError(f"Parameter case group {group_id} repeats case {case_id!r}")
            seen.add(case_id)
            parameter_case_values(group_id, case_id, value)


def expanded_plan_manifest(plan, runs, invocation_id):
    """Create a selector-free record of the concrete cases chosen for one invocation."""
    manifest = copy.deepcopy(plan)
    manifest.pop("dataset_groups", None)
    manifest.pop("resource_profiles", None)
    manifest.pop("parameter_cases", None)
    templates = manifest.get("templates", {})
    concrete = []
    for run in runs:
        resolved = copy.deepcopy(run)
        implementations = [resolve_implementation(implementation, templates)
                           for implementation in run.get("implementations", [])]
        resolved["implementations"] = [implementation for implementation in implementations
                                       if implementation.get("enabled", True)]
        concrete.append(resolved)
    manifest["runs"] = concrete
    manifest["execution"] = {
        "invocation_id": invocation_id,
        "expanded_run_cases": len(concrete),
    }
    return manifest


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


def resolve_temporary_paths(configured, context, run_dir):
    """Resolve cleanup roots and constrain them to one immutable result case."""
    if not isinstance(configured, list) or not configured:
        raise ValueError("temporary_paths must be a non-empty list")
    run_root = Path(os.path.abspath(run_dir))
    paths = []
    seen = set()
    for value in configured:
        path = Path(os.path.abspath(expand(str(value), context)))
        try:
            path.relative_to(run_root)
        except ValueError as error:
            raise ValueError(f"Refusing temporary path outside run directory: {path}") from error
        if path == run_root:
            raise ValueError(f"Refusing to use the complete run directory as temporary path: {path}")
        if path in seen:
            raise ValueError(f"Duplicate temporary path {path}")
        seen.add(path)
        paths.append(path)
    return paths


def prepare_temporary_paths(paths):
    """Create fresh roots required by Java, Spark, SystemDS, and Dask."""
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(f"Refusing symlinked temporary path {path}")
        if path.exists() and not path.is_dir():
            raise RuntimeError(f"Temporary path is not a directory: {path}")
        path.mkdir(parents=True, exist_ok=True)


def cleanup_temporary_paths(paths):
    """Remove private temporary roots, returning errors without masking run status."""
    errors = []
    removed = 0
    for path in paths:
        try:
            if path.is_symlink():
                raise RuntimeError("refusing symlink")
            if path.is_dir():
                shutil.rmtree(path)
                removed += 1
            elif path.exists():
                path.unlink()
                removed += 1
        except OSError as error:
            errors.append(f"{path}: {error}")
        except RuntimeError as error:
            errors.append(f"{path}: {error}")
    return removed, errors


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
    dataset_groups = named_mapping(plan, "dataset_groups")
    resource_profiles = named_mapping(plan, "resource_profiles")
    parameter_cases = named_mapping(plan, "parameter_cases")
    validate_named_definitions(datasets, dataset_groups, resource_profiles, parameter_cases)
    configured_runs = plan.get("runs", [])
    for run in configured_runs:
        run_id = str(run.get("id", ""))
        if not _VALID_ID.match(run_id):
            raise ValueError(f"Invalid run id {run_id!r}")
        for dataset_id in run_dataset_ids(run, dataset_groups):
            if dataset_id not in datasets:
                raise ValueError(f"Run {run_id} refers to unknown dataset {dataset_id}")
        if not run.get("implementations"):
            raise ValueError(f"Run {run_id} has no implementations")
        for implementation in run["implementations"]:
            resolve_implementation(implementation, templates)
    expanded_runs = expand_run_cases(configured_runs, templates, dataset_groups,
                                     resource_profiles, parameter_cases)
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
    required_python_modules = set()
    for run in enabled_runs:
        for implementation in run.get("implementations", []):
            resolved = resolve_implementation(implementation, templates)
            if resolved.get("enabled", True):
                required_python_modules.update(
                    str(module) for module in resolved.get("required_python_modules", []))
    if required_python_modules:
        python = expand(context["tools.python"], context)
        module_check = (
            "import importlib,sys\n"
            "missing=[]\n"
            "for module in sys.argv[1:]:\n"
            " try: importlib.import_module(module)\n"
            " except ImportError: missing.append(module)\n"
            "print(', '.join(missing))\n"
            "raise SystemExit(bool(missing))\n"
        )
        result = subprocess.run([python, "-c", module_check, *sorted(required_python_modules)],
                                text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(f"Configured Python is missing required modules: "
                               f"{result.stdout.strip()}")
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
    write_invocation_metadata(
        invocation_dir / "invocation-metadata.json", plan_dir, root, context)
    manifest = expanded_plan_manifest(plan, enabled_runs, invocation_id)
    (invocation_dir / "expanded-plan.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    execution_records = []

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
        for field in ("resource_profile", "parameter_case"):
            if field in run:
                run_context[f"run.{field}"] = str(run[field])
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
        configured_temporary_paths = run.get(
            "temporary_paths", plan.get("temporary_paths", _DEFAULT_TEMPORARY_PATHS))
        temporary_paths = resolve_temporary_paths(
            configured_temporary_paths, run_context, run_dir)
        setup_steps = run.get("setup", [])
        if isinstance(setup_steps, (str, dict)):
            setup_steps = [setup_steps]
        resolved_setup = []
        for step in setup_steps:
            step = {"command": step} if isinstance(step, str) else step
            command = expand(str(step["command"]), run_context)
            resolved_setup.append({
                "command": command,
                "environment": expand_map(step.get("environment"), run_context),
            })
        repetitions = int(run.get("repetitions", plan.get("defaults", {}).get("repetitions", 1)))
        resolved_implementations = []
        for configured_implementation in run.get("implementations", []):
            implementation = resolve_implementation(configured_implementation, templates)
            if not implementation.get("enabled", True):
                continue
            executions = []
            for rep in range(1, repetitions + 1):
                local_context = dict(run_context)
                local_context["rep"] = str(rep)
                local_context["implementation.id"] = str(implementation["id"])
                executions.append({
                    "rep": rep,
                    "command": expand(str(implementation["command"]), local_context),
                    "environment": {
                        "BENCH_RUN_TMP": str(run_dir / "python-tmp"),
                        **expand_map(run.get("environment"), local_context),
                        **expand_map(implementation.get("environment"), local_context),
                    },
                })
            resolved_implementations.append({
                "id": str(implementation["id"]),
                "comparable": implementation.get("comparable", True),
                "executions": executions,
            })
        resolved_run = {
            "id": run_id,
            "base_id": run_context["run.base_id"],
            "dataset": dataset_id,
            "resource_profile": run.get("resource_profile"),
            "parameter_case": run.get("parameter_case"),
            "parameters": run.get("parameters", {}),
            "resources": resources,
            "inputs": {key.removeprefix("input."): value for key, value in run_context.items()
                       if key.startswith("input.")},
            "cold_cache": [expand(str(path), run_context)
                           for path in run.get("cold_cache", ["${dataset.dir}"])],
            "telemetry": {**plan.get("telemetry", {}), **run.get("telemetry", {})},
            "temporary_paths": [str(path) for path in temporary_paths],
            "setup": resolved_setup,
            "implementations": resolved_implementations,
        }
        (run_dir / "resolved-run.json").write_text(
            json.dumps(resolved_run, indent=2, sort_keys=True), encoding="utf-8")
        for number, step in enumerate(resolved_setup, 1):
            command = step["command"]
            env = dict(global_env)
            env.update(step["environment"])
            setup_log = logs / f"setup-{number}.log"
            rc = command_run(command, plan_dir, env, setup_log)
            if rc:
                raise RuntimeError(f"Setup for run {run_id} failed ({rc}); see {setup_log}")
        (run_dir / "resolved-context.json").write_text(json.dumps(run_context, indent=2, sort_keys=True))
        scope_runner = run_dir / ".scope-runner.sh"
        write_scope_runner(scope_runner)
        io_stat_warning_printed = False
        csv_path = run_dir / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["run", "implementation", "comparable", "rep", "status", "wall_seconds",
                             "algorithm_seconds", "memory_peak_bytes", "major_faults",
                             "file_system_inputs", "cpu_usage_usec", "cpu_user_usec",
                             "cpu_system_usec", "cpu_nr_throttled", "cpu_throttled_usec",
                             "io_read_bytes", "io_write_bytes", "io_read_ops", "io_write_ops",
                             "memory_max_events", "oom_kill_events", "log", "metrics",
                             "telemetry"])
            timeout_seconds = int(resources.get("timeout_seconds", 0))
            grace_seconds = int(resources.get("timeout_grace_seconds", 30))
            if timeout_seconds < 0 or grace_seconds < 0:
                raise ValueError(f"Run {run_id} timeout values must be non-negative")
            timeout = str(timeout_seconds)
            memory = str(resources["memory_max"])
            swap = str(resources.get("swap_max", 0))
            telemetry = dict(plan.get("telemetry", {}))
            telemetry.update(run.get("telemetry", {}))
            telemetry_interval = float(telemetry.get("interval_seconds", 1.0))
            if telemetry_interval <= 0:
                raise ValueError(f"Run {run_id} telemetry.interval_seconds must be positive")
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
                    telemetry_path = logs / f"{tag}.telemetry.csv"
                    print(f"Starting {tag} (timeout={timeout_seconds}s, memory={memory}, "
                          f"results={run_dir})", flush=True)
                    _, reset_errors = cleanup_temporary_paths(temporary_paths)
                    if reset_errors:
                        raise RuntimeError(f"Could not reset temporary paths for {tag}: "
                                           f"{'; '.join(reset_errors)}")
                    prepare_temporary_paths(temporary_paths)
                    drop_caches(cache_paths, local_context, plan_dir, context["tools.python"],
                                logs / "drop-caches.log")
                    env = dict(global_env)
                    env["BENCH_RUN_TMP"] = str(run_dir / "python-tmp")
                    env.update(expand_map(run.get("environment"), local_context))
                    env.update(expand_map(implementation.get("environment"), local_context))
                    unit = re.sub(r"[^A-Za-z0-9_.-]", "-", f"ooc-{tag}-{os.getpid()}")
                    scope = ["systemd-run", "--user", "--scope", "--collect", "--quiet", f"--unit={unit}",
                             "-p", "MemoryAccounting=yes", "-p", "IOAccounting=yes",
                             "-p", f"MemoryMax={memory}",
                             "-p", f"MemorySwapMax={swap}", "-p", "TasksMax=infinity",
                             "-p", "KillMode=control-group", "-p", "SendSIGKILL=yes",
                             "-p", f"TimeoutStopSec={grace_seconds}s"]
                    if timeout_seconds:
                        scope += ["-p", f"RuntimeMaxSec={timeout_seconds + grace_seconds + 15}s"]
                    scope += [str(scope_runner), str(log), str(metrics), str(telemetry_path),
                              str(telemetry_interval), timeout, str(grace_seconds), command]
                    try:
                        with open(log, "a", encoding="utf-8") as scope_output:
                            rc = subprocess.run(scope, cwd=plan_dir, env=env,
                                                stdout=scope_output,
                                                stderr=subprocess.STDOUT).returncode
                    finally:
                        removed, cleanup_errors = cleanup_temporary_paths(temporary_paths)
                        with open(log, "a", encoding="utf-8") as output:
                            output.write(f"temporary cleanup: removed {removed} roots\n")
                            for error in cleanup_errors:
                                output.write(f"temporary cleanup warning: {error}\n")
                        if cleanup_errors:
                            print(f"WARNING: {tag} temporary cleanup was incomplete; see {log}",
                                  file=sys.stderr, flush=True)
                    if (not io_stat_warning_printed and
                            metric(metrics, r"^io_read_bytes=([0-9]+)$", "") == ""):
                        print("WARNING: cgroup io.stat is unavailable in the benchmark scope; "
                              "byte-accurate read/write I/O metrics will not be recorded. "
                              "Ensure the cgroup-v2 io controller is delegated through the "
                              "systemd user hierarchy.", file=sys.stderr, flush=True)
                        io_stat_warning_printed = True
                    status = "ok" if rc == 0 else "timeout" if rc == 124 else "killed" if rc == 137 else "failed"
                    if metric(metrics, r"^exit_status=(.+)$") == "running":
                        status = "killed"
                        with open(log, "a", encoding="utf-8") as output:
                            output.write(f"scope exited with {rc} before the accounting wrapper "
                                         "could finalize; likely whole-scope termination\n")
                    if status == "ok" and "An Error Occurred" in log.read_text(errors="replace"):
                        status = "failed"
                    time_path = Path(str(metrics) + ".time")
                    wall_seconds = elapsed_seconds(time_path)
                    algorithm_seconds = reported_seconds(log)
                    peak = metric(metrics, r"^memory_peak_bytes=([0-9]+)$")
                    faults = metric(time_path, r"Major \(requiring I/O\) page faults:\s*([0-9]+)")
                    inputs = metric(time_path, r"File system inputs:\s*([0-9]+)")
                    cpu_usage = metric(metrics, r"^cpu_usage_usec=([0-9]+)$")
                    cpu_user = metric(metrics, r"^cpu_user_usec=([0-9]+)$")
                    cpu_system = metric(metrics, r"^cpu_system_usec=([0-9]+)$")
                    cpu_nr_throttled = metric(metrics, r"^cpu_nr_throttled=([0-9]+)$")
                    cpu_throttled = metric(metrics, r"^cpu_throttled_usec=([0-9]+)$")
                    io_read_bytes = metric(metrics, r"^io_read_bytes=([0-9]+)$")
                    io_write_bytes = metric(metrics, r"^io_write_bytes=([0-9]+)$")
                    io_read_ops = metric(metrics, r"^io_read_ops=([0-9]+)$")
                    io_write_ops = metric(metrics, r"^io_write_ops=([0-9]+)$")
                    memory_max_events = metric(
                        metrics, r"^memory_events=.*(?:^|;)max ([0-9]+)(?:;|$)")
                    oom_kill_events = metric(
                        metrics, r"^memory_events=.*(?:^|;)oom_kill ([0-9]+)(?:;|$)")
                    writer.writerow([run_id, impl_id, implementation.get("comparable", True), rep, status,
                                     wall_seconds, algorithm_seconds, peak, faults, inputs,
                                     cpu_usage, cpu_user, cpu_system, cpu_nr_throttled,
                                     cpu_throttled, io_read_bytes, io_write_bytes, io_read_ops,
                                     io_write_ops, memory_max_events, oom_kill_events, log,
                                     metrics, telemetry_path])
                    csv_file.flush()
                    if implementation.get("comparable", True):
                        execution_records.append({
                            "base_id": run_context["run.base_id"], "dataset": dataset_id,
                            "resource_profile": run.get("resource_profile"),
                            "parameter_case": run.get("parameter_case"), "rep": rep,
                            "impl_id": impl_id, "status": status, "log": log,
                            "outputs": outputs, "run_dir": run_dir,
                        })
                    print(f"{tag}: {status} (wall={wall_seconds}s, peak={peak})")
    apply_output_retention(execution_records, invocation_dir)
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
