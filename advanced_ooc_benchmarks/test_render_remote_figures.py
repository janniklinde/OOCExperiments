"""Regression tests for invocation fallback priority (no figures are generated)."""
import unittest
from unittest.mock import patch

import render_remote_figures as renderer
import visualize_invocation as visualizer


class FallbackPriorityTests(unittest.TestCase):
    def select(self, statuses, prefer_successful=False):
        candidates = list(range(len(statuses)))
        def tagged(candidate):
            return [("dense", "", {
                "base_id": "lmcg", "memory_profile": "mem16",
                "implementation": "systemds-spark", "status": statuses[candidate],
                "source": candidate,
            })]
        with patch.object(renderer, "tagged_rows", side_effect=tagged):
            partitions = list(renderer.partition_rows(
                candidates[0], candidates[1:], prefer_successful))
        return partitions[0][2][0]["source"]

    def test_default_keeps_failure(self):
        self.assertEqual(self.select(["failed", "ok"]), 0)

    def test_success_replaces_failure(self):
        self.assertEqual(self.select(["failed", "timeout", "ok"], True), 2)

    def test_first_success_wins(self):
        self.assertEqual(self.select(["failed", "ok", "ok"], True), 1)
        self.assertEqual(self.select(["ok", "ok"], True), 0)

    def test_all_unsuccessful_keeps_first(self):
        self.assertEqual(self.select(["timeout", "failed", "killed"], True), 0)


class FigureGroupingTests(unittest.TestCase):
    def test_equal_size_shapes_are_labeled_by_dimensions(self):
        rows = [
            {"memory_profile": "mem16", "dataset": "tall", "dataset_bytes": 32_000_000_000,
             "dataset_shape": (40_000_000, 100)},
            {"memory_profile": "mem16", "dataset": "thin", "dataset_bytes": 32_000_000_000,
             "dataset_shape": (400_000_000, 10)},
        ]
        axis, _, labels, title = visualizer.bar_axis(rows)
        self.assertEqual(axis, "dataset")
        self.assertEqual(title, "Dataset Shape (32GB FP64)")
        self.assertEqual(labels, ["40M × 100", "400M × 10"])

    def test_specialized_memory_profile_displays_its_cgroup_size(self):
        axis, _, labels, title = visualizer.bar_axis([
            {"memory_profile": "mem16_mlp", "dataset": "mlp_wide"},
        ])
        self.assertEqual((axis, labels, title), ("memory_profile", ["16GB"], "CGroup Size"))

    def test_spoof_references_join_memory_sweep_as_distinct_bars(self):
        def tagged(_):
            for base_id, implementation, profile in (
                ("kmeans_memory", "systemds-ooc", "mem128"),
                ("kmeans_spoof", "systemds-cp", "mem128"),
                ("kmeans_spoof", "systemds-cp-spoof", "mem128"),
                ("kmeans", "systemds-ooc", "mem16"),
            ):
                yield ("dense_d32", "", {
                    "base_id": base_id, "memory_profile": profile,
                    "implementation": implementation, "status": "ok",
                })
        with patch.object(renderer, "tagged_rows", side_effect=tagged):
            partitions = list(renderer.partition_rows("invocation"))
        self.assertEqual(len(partitions), 1)
        base_id, suffix, rows = partitions[0]
        self.assertEqual((base_id, suffix), ("kmeans_memory", ""))
        self.assertEqual(len(rows), 4)
        self.assertNotEqual(visualizer.implementation_label("systemds-cp"),
                            visualizer.implementation_label("systemds-cp-spoof"))

    def test_blocksize_variants_survive_candidate_selection(self):
        def tagged(_):
            for blocksize in ("500", "1000"):
                yield ("dense", blocksize, {
                    "base_id": "kmeans", "memory_profile": "mem16",
                    "implementation": "systemds-ooc", "status": "ok",
                })
        with patch.object(renderer, "tagged_rows", side_effect=tagged):
            partitions = list(renderer.partition_rows("invocation"))
        self.assertEqual([suffix for _, suffix, _ in partitions], ["bs1000", "bs500"])

    def test_older_different_blocksize_is_not_mixed_into_current_case(self):
        def tagged(candidate):
            size = "1000" if candidate == "latest" else "500"
            yield ("dense", size, {
                "base_id": "kmeans", "memory_profile": "mem16",
                "implementation": "systemds-ooc", "status": "failed", "source": candidate,
            })
        with patch.object(renderer, "tagged_rows", side_effect=tagged):
            partitions = list(renderer.partition_rows("latest", ["old"], True))
        self.assertEqual(len(partitions), 1)
        self.assertEqual(partitions[0][2][0]["source"], "latest")

    def test_older_profile_at_same_memory_is_not_an_extra_axis_value(self):
        def tagged(candidate):
            profile = "mem16_mlp" if candidate == "latest" else "mem16"
            yield ("mlp_wide", "", {
                "base_id": "mlp_wide", "memory_profile": profile,
                "memory_limit": "16G", "case_signature": "h40" if candidate == "latest" else "old",
                "implementation": "systemds-ooc", "status": "ok",
            })
        with patch.object(renderer, "tagged_rows", side_effect=tagged):
            partitions = list(renderer.partition_rows("latest", ["old"], True))
        self.assertEqual(len(partitions[0][2]), 1)
        self.assertEqual(partitions[0][2][0]["memory_profile"], "mem16_mlp")


if __name__ == "__main__":
    unittest.main()
