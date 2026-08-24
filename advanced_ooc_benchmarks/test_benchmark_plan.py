#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0.

import json
import tempfile
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


if __name__ == "__main__":
    unittest.main()
