"""Prepared-input compatibility across PreVision workload-only rebuilds."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "prevision_prepare", Path(__file__).parent / "prevision" / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PreparedInputCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "generator": "prevision/prepare.py", "generator_version": 2,
            "mode": "gnmf", "rows": 96, "cols": 64, "tile_rows": 48,
            "tile_cols": 32, "rank": 16, "seed": 23,
            "source": {"path": "/source/X.f64", "size": 96 * 64 * 8, "mtime_ns": 123},
            "adapter_sha256": "old-binary",
        }

    def test_workload_rebuild_reuses_inputs(self):
        rebuilt = deepcopy(self.metadata)
        rebuilt["adapter_sha256"] = "new-binary"
        self.assertTrue(prepare.compatible_manifest(self.metadata, rebuilt))
        self.assertEqual(self.metadata["adapter_sha256"], "old-binary")

    def test_layout_initialization_or_format_change_invalidates(self):
        for field, value in (("rows", 192), ("cols", 128), ("tile_rows", 96),
                             ("tile_cols", 64), ("rank", 8), ("seed", 24),
                             ("generator_version", 3), ("mode", "gram")):
            with self.subTest(field=field):
                changed = deepcopy(self.metadata)
                changed[field] = value
                self.assertFalse(prepare.compatible_manifest(self.metadata, changed))

    def test_changed_source_invalidates(self):
        changed = deepcopy(self.metadata)
        changed["source"]["mtime_ns"] += 1
        self.assertFalse(prepare.compatible_manifest(self.metadata, changed))

    def test_legacy_square_manifest_reuses_inputs(self):
        legacy = deepcopy(self.metadata)
        legacy["generator_version"] = 1
        legacy["tile"] = 32
        del legacy["tile_rows"], legacy["tile_cols"]
        rebuilt = dict(legacy, adapter_sha256="new-binary")
        self.assertTrue(prepare.compatible_manifest(legacy, rebuilt))

    def test_malformed_manifest_is_not_compatible(self):
        self.assertFalse(prepare.compatible_manifest([], self.metadata))


if __name__ == "__main__":
    unittest.main()
