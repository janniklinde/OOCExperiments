#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0.

import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

import benchmark_plan
import drop_caches


class RunExpansionTest(unittest.TestCase):
    def test_execution_timestamp_is_filesystem_safe_and_timezone_qualified(self):
        timestamp = benchmark_plan.execution_timestamp(
            datetime(2026, 8, 24, 17, 25, 16, 123456, tzinfo=timezone.utc))

        self.assertEqual(timestamp, "20260824T172516.123456+0000")

    def test_invocation_metadata_hashes_suite_sources_and_tool(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = root / "suite"
            suite.mkdir()
            (suite / "workload.dml").write_text("print('ok');\n")
            tool = root / "SystemDS.jar"
            tool.write_bytes(b"jar")
            destination = root / "metadata.json"

            benchmark_plan.write_invocation_metadata(
                destination, suite, root, {"tools.systemds_jar": str(tool)})

            metadata = json.loads(destination.read_text())
            self.assertEqual(metadata["suite_sources"]["workload.dml"]["bytes"], 13)
            self.assertEqual(metadata["tool_files"]["systemds_jar"]["sha256"],
                             benchmark_plan.file_digest(tool))

    def test_blocksize_list_expands_without_mutating_source(self):
        source = [{"id": "workload", "parameters": {"blocksize": [500, 1000]}}]

        cases = benchmark_plan.expand_run_cases(source)

        self.assertEqual([case["id"] for case in cases],
                         ["workload-bs500", "workload-bs1000"])
        self.assertEqual([case["parameters"]["blocksize"] for case in cases], [500, 1000])
        self.assertEqual(source[0]["parameters"]["blocksize"], [500, 1000])

    def test_scalar_blocksize_keeps_original_run_id(self):
        source = [{"id": "workload", "parameters": {"blocksize": 500}}]

        self.assertIs(benchmark_plan.expand_run_cases(source)[0], source[0])

    def test_duplicate_blocksize_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "repeats blocksize 500"):
            benchmark_plan.expand_run_cases([
                {"id": "workload", "parameters": {"blocksize": [500, 500]}}
            ])

    def test_fractional_blocksize_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid blocksize"):
            benchmark_plan.expand_run_cases([
                {"id": "workload", "parameters": {"blocksize": [500.5]}}
            ])

    def test_backend_sweeps_create_one_unblocked_baseline_case(self):
        templates = {
            "ooc": {"blocksize_sweep": "ooc"},
            "spark": {"blocksize_sweep": "spark"},
            "python": {},
        }
        source = [{
            "id": "workload",
            "blocksize_sweeps": {"ooc": [500, 1000], "spark": [2000]},
            "implementations": [
                {"id": "systemds-ooc", "template": "ooc"},
                {"id": "systemds-spark", "template": "spark"},
                {"id": "numpy", "template": "python"},
            ],
        }]

        cases = benchmark_plan.expand_run_cases(source, templates)

        self.assertEqual([case["id"] for case in cases], [
            "workload-baseline", "workload-ooc-bs500", "workload-ooc-bs1000",
            "workload-spark-bs2000"
        ])
        self.assertEqual([implementation["id"] for implementation in cases[0]["implementations"]],
                         ["numpy"])
        self.assertTrue(cases[0]["_baseline_case"])
        self.assertNotIn("blocksize", cases[0].get("parameters", {}))
        self.assertEqual(cases[0]["setup"], [])
        self.assertEqual([implementation["id"] for implementation in cases[1]["implementations"]],
                         ["systemds-ooc"])
        self.assertEqual([implementation["id"] for implementation in cases[3]["implementations"]],
                         ["systemds-spark"])
        self.assertEqual(cases[3]["parameters"]["blocksize"], 2000)
        self.assertEqual(cases[3]["parameters"]["blocksize_sweep"], "spark")

    def test_dataset_list_expands_before_backend_sweeps(self):
        templates = {
            "ooc": {"blocksize_sweep": "ooc"},
            "python": {},
        }
        source = [{
            "id": "workload",
            "dataset": ["dense_1m", "dense_2m"],
            "blocksize_sweeps": {"ooc": [500]},
            "implementations": [
                {"id": "systemds-ooc", "template": "ooc"},
                {"id": "numpy", "template": "python"},
            ],
        }]

        cases = benchmark_plan.expand_run_cases(source, templates)

        self.assertEqual([case["id"] for case in cases], [
            "workload-dense_1m-baseline", "workload-dense_1m-ooc-bs500",
            "workload-dense_2m-baseline", "workload-dense_2m-ooc-bs500",
        ])
        self.assertEqual([case["dataset"] for case in cases],
                         ["dense_1m", "dense_1m", "dense_2m", "dense_2m"])
        self.assertTrue(all(case["base_id"] == "workload" for case in cases))
        self.assertEqual(source[0]["dataset"], ["dense_1m", "dense_2m"])

    def test_named_dimensions_expand_in_stable_order(self):
        templates = {"ooc": {"blocksize_sweep": "ooc"}}
        source = [{
            "id": "mlp",
            "dataset": {"group": "dense"},
            "resource_profiles": ["mem8", "mem4"],
            "parameter_cases": "hidden",
            "resources": {"timeout_seconds": 90, "memory_max": "legacy"},
            "parameters": {"iterations": 1},
            "blocksize_sweeps": {"ooc": [500]},
            "implementations": [{"id": "systemds-ooc", "template": "ooc"}],
        }]

        cases = benchmark_plan.expand_run_cases(
            source, templates,
            dataset_groups={"dense": ["d4", "d8"]},
            resource_profiles={
                "mem8": {"memory_max": "8G", "java_heap": "6g"},
                "mem4": {"memory_max": "4G", "java_heap": "3g"},
            },
            parameter_cases={"hidden": [
                {"id": "act4", "hidden_size": 8192},
                {"id": "act8", "parameters": {"hidden_size": 16384}},
            ]})

        self.assertEqual(cases[0]["id"], "mlp-d4-mem8-act4-ooc-bs500")
        self.assertEqual(cases[-1]["id"], "mlp-d8-mem4-act8-ooc-bs500")
        self.assertEqual(len(cases), 8)
        self.assertEqual(cases[0]["base_id"], "mlp")
        self.assertEqual(cases[0]["resources"], {
            "timeout_seconds": 90, "memory_max": "8G", "java_heap": "6g"
        })
        self.assertEqual(cases[0]["parameters"]["hidden_size"], 8192)
        self.assertEqual(cases[0]["resource_profile"], "mem8")
        self.assertEqual(cases[0]["parameter_case"], "act4")
        self.assertEqual(source[0]["dataset"], {"group": "dense"})

    def test_resource_group_expands_like_an_inline_profile_list(self):
        source = [{"id": "workload", "resource_profiles": {"group": "scaling"}}]
        profiles = {"mem100": {"memory_max": "100G"}, "mem4": {"memory_max": "4G"}}

        grouped = benchmark_plan.expand_run_cases(
            source, resource_profiles=profiles,
            resource_groups={"scaling": ["mem100", "mem4"]})
        inline = benchmark_plan.expand_run_cases(
            [{"id": "workload", "resource_profiles": ["mem100", "mem4"]}],
            resource_profiles=profiles)

        self.assertEqual([case["id"] for case in grouped],
                         ["workload-mem100", "workload-mem4"])
        self.assertEqual([case["resources"] for case in grouped],
                         [case["resources"] for case in inline])
        # A profile added to the group must not require touching the run.
        self.assertEqual(source[0]["resource_profiles"], {"group": "scaling"})

    def test_unknown_or_malformed_resource_groups_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown resource group missing"):
            benchmark_plan.expand_run_cases(
                [{"id": "workload", "resource_profiles": {"group": "missing"}}],
                resource_profiles={"mem4": {"memory_max": "4G"}},
                resource_groups={"scaling": ["mem4"]})
        with self.assertRaisesRegex(ValueError, "must contain only group"):
            benchmark_plan.expand_run_cases(
                [{"id": "workload", "resource_profiles": {"group": "scaling", "extra": 1}}],
                resource_profiles={"mem4": {"memory_max": "4G"}},
                resource_groups={"scaling": ["mem4"]})
        with self.assertRaisesRegex(ValueError, "Resource group scaling must be a non-empty list"):
            benchmark_plan.validate_named_definitions(
                {}, {}, {"mem4": {"memory_max": "4G"}}, {}, {"scaling": []})
        with self.assertRaisesRegex(ValueError, "unknown resource_profiles mem8"):
            benchmark_plan.validate_named_definitions(
                {}, {}, {"mem4": {"memory_max": "4G"}}, {}, {"scaling": ["mem8"]})

    def test_unknown_named_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown dataset group missing"):
            benchmark_plan.expand_run_cases([
                {"id": "workload", "dataset": {"group": "missing"}}
            ])
        with self.assertRaisesRegex(ValueError, "unknown resource_profiles mem4"):
            benchmark_plan.expand_run_cases([
                {"id": "workload", "resource_profiles": ["mem4"]}
            ])
        with self.assertRaisesRegex(ValueError, "unknown parameter case group hidden"):
            benchmark_plan.expand_run_cases([
                {"id": "workload", "parameter_cases": "hidden"}
            ])

    def test_expanded_manifest_contains_only_concrete_cases(self):
        plan = {
            "version": 1,
            "dataset_groups": {"dense": ["d4"]},
            "resource_profiles": {"mem4": {"memory_max": "4G"}},
            "parameter_cases": {"iterations": [{"id": "i1", "iterations": 1}]},
            "templates": {"python": {"command": "python baseline.py"}},
        }
        runs = [{
            "id": "run-d4-mem4-i1",
            "dataset": "d4",
            "resources": {"memory_max": "4G"},
            "implementations": [{"id": "numpy", "template": "python"}],
        }]

        manifest = benchmark_plan.expanded_plan_manifest(plan, runs, "invocation")

        self.assertNotIn("dataset_groups", manifest)
        self.assertNotIn("resource_profiles", manifest)
        self.assertNotIn("parameter_cases", manifest)
        self.assertEqual(manifest["execution"]["expanded_run_cases"], 1)
        self.assertEqual(manifest["runs"][0]["implementations"][0]["command"],
                         "python baseline.py")

    def test_scalar_dataset_does_not_change_case_names(self):
        source = [{"id": "workload", "dataset": "dense_1m", "parameters": {}}]

        self.assertIs(benchmark_plan.expand_run_cases(source)[0], source[0])

    def test_empty_or_duplicate_dataset_lists_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "dataset list must not be empty"):
            benchmark_plan.expand_run_cases([{"id": "workload", "dataset": []}])
        with self.assertRaisesRegex(ValueError, "repeats dataset 'dense_1m'"):
            benchmark_plan.expand_run_cases([{
                "id": "workload", "dataset": ["dense_1m", "dense_1m"]
            }])

    def test_unknown_backend_sweep_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown blocksize sweeps: spark"):
            benchmark_plan.expand_run_cases([{
                "id": "workload",
                "blocksize_sweeps": {"ooc": [500]},
                "implementations": [{"id": "spark", "blocksize_sweep": "spark"}],
            }])

    def test_disabled_runs_are_still_structurally_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = Path(temporary) / "plan.yaml"
            plan.write_text("""
version: 1
datasets: {dense: {}}
runs:
  - id: future-workload
    enabled: false
    dataset: missing
    implementations: [{id: numpy, command: true}]
""")

            with self.assertRaisesRegex(ValueError, "unknown dataset missing"):
                benchmark_plan.execute_plan(plan, validate_only=True)

    def test_dataset_list_members_are_structurally_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = Path(temporary) / "plan.yaml"
            plan.write_text("""
version: 1
datasets: {dense_1m: {}}
runs:
  - id: workload
    dataset: [dense_1m, missing]
    implementations: [{id: numpy, command: true}]
""")

            with self.assertRaisesRegex(ValueError, "unknown dataset missing"):
                benchmark_plan.execute_plan(plan, validate_only=True)


class DatasetVariantTest(unittest.TestCase):
    def test_variant_is_prepared_and_publishes_artifact_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            context = {"run.blocksize": "500", "dataset.dir": str(directory)}
            variant = {
                "artifacts": {
                    "X": {
                        "path": "systemds/X-bs${run.blocksize}",
                        "metadata": {
                            "expect": {"rows": "2", "rows_in_block": "${run.blocksize}"}
                        },
                    }
                },
                "prepare": {
                    "policy": "auto",
                    "command": (
                        "mkdir -p '${dataset.dir}/systemds/X-bs${run.blocksize}'; "
                        "printf '%s' '{\"rows\":2,\"rows_in_block\":500}' > "
                        "'${dataset.dir}/systemds/X-bs${run.blocksize}.mtd'"
                    ),
                },
            }

            benchmark_plan.prepare_dataset_variant(
                "dense", "blocksize", variant, directory, context, directory, {})

            self.assertEqual(context["artifact.X"],
                             str(directory / "systemds" / "X-bs500"))
            self.assertEqual(json.loads((directory / "systemds" / "X-bs500.mtd").read_text()),
                             {"rows": 2, "rows_in_block": 500})
            self.assertTrue((directory / "prepare-blocksize-500.log").exists())


class DropCachesTest(unittest.TestCase):
    def test_walk_ignores_tool_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data = directory / "X.f64"
            tool = directory / "SystemDS.jar"
            data.write_bytes(b"data")
            tool.symlink_to(data)

            self.assertEqual(list(drop_caches.walk([str(directory)])), [str(data)])


class TemporaryCleanupTest(unittest.TestCase):
    def test_private_temporary_roots_are_removed_without_touching_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "case"
            outputs = run_dir / "outputs"
            outputs.mkdir(parents=True)
            result = outputs / "model.bin"
            result.write_bytes(b"result")
            context = {"run.results": str(run_dir)}
            paths = benchmark_plan.resolve_temporary_paths(
                ["${run.results}/systemds-tmp", "${run.results}/dask-spill"],
                context, run_dir)

            benchmark_plan.prepare_temporary_paths(paths)
            (paths[0] / "spill.bin").write_bytes(b"spill")
            nested = paths[1] / "worker" / "storage"
            nested.mkdir(parents=True)
            (nested / "part").write_bytes(b"spill")

            removed, errors = benchmark_plan.cleanup_temporary_paths(paths)

            self.assertEqual(removed, 2)
            self.assertEqual(errors, [])
            self.assertTrue(result.exists())
            self.assertTrue(all(not path.exists() for path in paths))

    def test_temporary_roots_must_be_strict_children_of_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "case"
            run_dir.mkdir()
            context = {"run.results": str(run_dir)}

            with self.assertRaisesRegex(ValueError, "complete run directory"):
                benchmark_plan.resolve_temporary_paths(
                    ["${run.results}"], context, run_dir)
            with self.assertRaisesRegex(ValueError, "outside run directory"):
                benchmark_plan.resolve_temporary_paths(
                    [str(Path(temporary) / "outside")], context, run_dir)


class OutputRetentionTest(unittest.TestCase):
    def _record(self, root, case, implementation, value, status="ok"):
        run_dir = root / case
        outputs = run_dir / "outputs"
        logs = run_dir / "logs"
        outputs.mkdir(parents=True)
        logs.mkdir()
        (outputs / f"{implementation}-artifact-r1").write_bytes(b"large result")
        log = logs / "run.log"
        if implementation.startswith("systemds-"):
            log.write_text(f"inertia={value}\n", encoding="utf-8")
        else:
            log.write_text("", encoding="utf-8")
            (outputs / f"{implementation}-r1.json").write_text(
                json.dumps({"inertia": value}), encoding="utf-8")
        return {
            "base_id": "kmeans_scaling", "dataset": "dense", "resource_profile": "mem4",
            "parameter_case": None, "rep": 1, "impl_id": implementation,
            "status": status, "log": log, "outputs": outputs, "run_dir": run_dir,
        }

    def test_matching_outputs_are_compacted_after_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [
                self._record(root, "ooc", "systemds-ooc", 42.0),
                self._record(root, "baseline", "numpy-lloyd", 42.00001),
            ]

            benchmark_plan.apply_output_retention(records, root)

            self.assertFalse(any(records[0]["outputs"].iterdir()))
            self.assertEqual([path.name for path in records[1]["outputs"].iterdir()],
                             ["numpy-lloyd-r1.json"])
            self.assertEqual(json.loads((records[0]["run_dir"] /
                                         "output-retention.json").read_text())["retention"],
                             "compact")

    def test_divergent_outputs_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [
                self._record(root, "ooc", "systemds-ooc", 42.0),
                self._record(root, "baseline", "numpy-lloyd", 84.0),
            ]

            benchmark_plan.apply_output_retention(records, root)

            self.assertTrue((records[0]["outputs"] / "systemds-ooc-artifact-r1").exists())
            self.assertTrue((records[1]["outputs"] / "numpy-lloyd-artifact-r1").exists())
            report = json.loads((records[0]["run_dir"] / "output-retention.json").read_text())
            self.assertEqual(report["reason"], "numerical outputs diverged")


class ScopeRunnerTest(unittest.TestCase):
    def test_payload_failure_is_recorded_by_accounting_wrapper(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runner = directory / "runner.sh"
            log = directory / "run.log"
            metrics = directory / "run.metrics"
            benchmark_plan.write_scope_runner(runner)

            result = subprocess.run(
                [runner, log, metrics, directory / "telemetry.csv", "0.05", "10", "1",
                 "echo payload-started; exit 23"])

            self.assertEqual(result.returncode, 23)
            self.assertIn("payload-started", log.read_text())
            self.assertIn("exit_status=23", metrics.read_text())
            telemetry = (directory / "telemetry.csv").read_text().splitlines()
            self.assertIn("cpu_usage_usec", telemetry[0])
            self.assertGreaterEqual(len(telemetry), 2)


if __name__ == "__main__":
    unittest.main()


class PreparationTimeoutTest(unittest.TestCase):
    def test_timeout_kills_the_whole_process_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pid_file = directory / "child.pid"
            command = f"sleep 60 & echo $! > '{pid_file}'; sleep 60"

            with self.assertRaises(subprocess.TimeoutExpired):
                benchmark_plan.command_run(command, directory, dict(os.environ),
                                           directory / "run.log", timeout=1)

            # bash -lc dies on its own, so the grandchild is the thing worth asserting.
            child = int(pid_file.read_text().strip())
            deadline = time.time() + 10
            while time.time() < deadline:
                try:
                    os.kill(child, 0)
                except ProcessLookupError:
                    return
                time.sleep(0.1)
            self.fail(f"process {child} survived the timeout")

    def test_dataset_preparation_reports_the_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            dataset = {
                "ready": ["metadata.json"],
                "prepare": {"policy": "auto", "command": "sleep 60", "timeout_seconds": 1},
            }
            context = {"plan.root": str(directory)}

            with self.assertRaisesRegex(RuntimeError, "exceeded 1s and was killed"):
                benchmark_plan.prepare_dataset("dense", dataset, context, directory, {})


class ImplementationSelectionTest(unittest.TestCase):
    """--implementation narrows the arms so a smoke test need not run the comparison."""

    def _plan(self, directory):
        plan = {
            "version": 1, "root": str(directory),
            "datasets": {"dense": {"ready": ["metadata.json"]}},
            "templates": {"python": {"command": "true"}},
            "runs": [
                {"id": "lmcg", "dataset": "dense", "entrypoint": "x.dml",
                 "implementations": [{"id": "systemds-ooc", "template": "python"},
                                     {"id": "systemds-cp", "template": "python"},
                                     {"id": "numpy-cg", "template": "python"}]},
                {"id": "kmeans", "dataset": "dense", "entrypoint": "x.dml",
                 "implementations": [{"id": "numpy-lloyd", "template": "python"}]},
            ],
        }
        path = directory / "plan.yaml"
        path.write_text(json.dumps(plan))
        return path

    def _select(self, plan, patterns):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            benchmark_plan.execute_plan(plan, validate_only=True, implementations=patterns)
        return output.getvalue()

    def test_a_single_arm_drops_the_cases_without_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self._plan(Path(temporary))
            output = self._select(plan, ["systemds-ooc"])
            # kmeans has no systemds-ooc arm, so only lmcg survives.
            self.assertIn("1 enabled run case", output)

    def test_a_glob_keeps_every_matching_arm(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self._plan(Path(temporary))
            output = self._select(plan, ["numpy-*"])
            self.assertIn("2 enabled run case", output)

    def test_an_unmatched_arm_is_an_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self._plan(Path(temporary))
            with self.assertRaises(ValueError) as raised:
                benchmark_plan.execute_plan(plan, validate_only=True,
                                            implementations=["dask-cg"])
            self.assertIn("No enabled implementation matches", str(raised.exception))

    def test_it_composes_with_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self._plan(Path(temporary))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                benchmark_plan.execute_plan(plan, validate_only=True, only=["lmcg"],
                                            implementations=["systemds-*"])
            self.assertIn("1 enabled run case", output.getvalue())


class RunSelectionTest(unittest.TestCase):
    def _plan(self, directory):
        plan = {
            "version": 1, "root": str(directory),
            "datasets": {"dense": {"ready": ["metadata.json"]}},
            "templates": {"python": {"command": "true"}},
            "runs": [
                {"id": "lmcg_scaling", "dataset": "dense", "entrypoint": "x.dml",
                 "implementations": [{"id": "numpy", "template": "python"}]},
                {"id": "kmeans_scaling", "dataset": "dense", "entrypoint": "x.dml",
                 "implementations": [{"id": "numpy", "template": "python"}]},
            ],
        }
        path = directory / "plan.yaml"
        path.write_text(json.dumps(plan))
        return path

    def test_only_selects_a_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self._plan(Path(temporary))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                benchmark_plan.execute_plan(plan, validate_only=True, only=["lmcg_scaling"])
            self.assertIn("1 enabled run case", output.getvalue())

    def test_only_accepts_a_glob(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self._plan(Path(temporary))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                benchmark_plan.execute_plan(plan, validate_only=True, only=["*_scaling"])
            self.assertIn("2 enabled run cases", output.getvalue())

    def test_unmatched_selection_is_an_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = self._plan(Path(temporary))
            with self.assertRaisesRegex(ValueError, "No enabled run case matches"):
                benchmark_plan.execute_plan(plan, validate_only=True, only=["pca_scaling"])


class SkipVariantsTest(unittest.TestCase):
    def test_base_dataset_is_staged_without_its_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = {
                "version": 1, "root": str(directory), "results": str(directory / "out"),
                "datasets": {"dense": {
                    "directory": str(directory / "dense"),
                    "ready": ["metadata.json"],
                    "artifacts": {"X_raw": {"path": "X.f64"}},
                    "prepare": {"policy": "auto", "command":
                                "mkdir -p '${dataset.dir}'; touch '${dataset.dir}/X.f64'; "
                                "printf '{}' > '${dataset.dir}/metadata.json'"},
                    "variants": {"blocksize": {
                        "artifacts": {"X": {"path": "X-bs${run.blocksize}"}},
                        "prepare": {"policy": "auto",
                                    "command": "touch '${dataset.dir}/X-bs${run.blocksize}'"},
                    }},
                }},
                "templates": {"python": {"command": "true"}},
                "runs": [{"id": "lmcg", "dataset": "dense", "entrypoint": "x.dml",
                          "parameters": {"blocksize": 1000},
                          "implementations": [{"id": "numpy", "template": "python"}]}],
            }
            path = directory / "plan.yaml"
            path.write_text(json.dumps(plan))

            with contextlib.redirect_stdout(io.StringIO()):
                benchmark_plan.execute_plan(path, prepare_only=True, skip_variants=True)
            self.assertTrue((directory / "dense" / "X.f64").exists())
            self.assertFalse((directory / "dense" / "X-bs1000").exists())

            with contextlib.redirect_stdout(io.StringIO()):
                benchmark_plan.execute_plan(path, prepare_only=True)
            self.assertTrue((directory / "dense" / "X-bs1000").exists())


class PrepareKeepGoingTest(unittest.TestCase):
    def _plan(self, directory):
        def dataset(name, command):
            return {"directory": str(directory / name), "ready": ["metadata.json"],
                    "prepare": {"policy": "auto", "command": command}}
        plan = {
            "version": 1, "root": str(directory), "results": str(directory / "out"),
            "datasets": {
                "broken": dataset("broken", "definitely-not-a-command"),
                "fine": dataset("fine",
                                "mkdir -p '${dataset.dir}'; printf '{}' > '${dataset.dir}/metadata.json'"),
            },
            "templates": {"python": {"command": "true"}},
            "runs": [{"id": name, "dataset": name, "entrypoint": "x.dml",
                      "implementations": [{"id": "numpy", "template": "python"}]}
                     for name in ("broken", "fine")],
        }
        path = directory / "plan.yaml"
        path.write_text(json.dumps(plan))
        return path

    def test_a_failing_dataset_does_not_block_the_others(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                rc = benchmark_plan.execute_plan(self._plan(directory), prepare_only=True)

            self.assertEqual(rc, 1)  # non-zero, so a wrapper still sees the failure
            self.assertTrue((directory / "fine" / "metadata.json").exists())
            self.assertIn("FAILED broken", output.getvalue())
            self.assertIn("Prepared 1 of 2 datasets", output.getvalue())

    @unittest.skipUnless(
        shutil.which("systemd-run") and subprocess.run(
            ["systemctl", "--user", "status"], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL).returncode == 0,
        "a real run needs a user systemd manager before it reaches preparation")
    def test_a_real_run_still_fails_fast(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "Generator for broken failed"):
                benchmark_plan.execute_plan(self._plan(directory))
