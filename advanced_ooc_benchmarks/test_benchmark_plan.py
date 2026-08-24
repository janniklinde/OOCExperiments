#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0.

import json
import tempfile
import unittest
from pathlib import Path

import benchmark_plan


class RunExpansionTest(unittest.TestCase):
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

    def test_unknown_backend_sweep_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown blocksize sweeps: spark"):
            benchmark_plan.expand_run_cases([{
                "id": "workload",
                "blocksize_sweeps": {"ooc": [500]},
                "implementations": [{"id": "spark", "blocksize_sweep": "spark"}],
            }])


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


if __name__ == "__main__":
    unittest.main()
