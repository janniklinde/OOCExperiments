import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class IncrementalKMeans(unittest.TestCase):
    def test_full_passes_and_final_objective(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            x = np.random.default_rng(5).normal(size=(400, 5))
            x.astype("<f8").tofile(data / "X.f64")
            (data / "metadata.json").write_text(json.dumps(
                {"rows": 400, "cols": 5, "dtype": "float64"}))
            result = data / "result.json"
            script = Path(__file__).resolve().parent.parent / "kmeans/sklearn_batching.py"
            subprocess.run([sys.executable, str(script),
                            str(data), "--clusters", "4", "--passes", "3",
                            "--batch-rows", "64", "--output", str(result)],
                           check=True, capture_output=True)
            report = json.loads(result.read_text())
            self.assertEqual(report["rows_processed_training"], 1200)
            self.assertEqual(report["partial_fit_steps"], 21)
            centers = np.load(data / "result-centers.npy")
            labels = np.load(data / "result-labels.npy") - 1
            self.assertEqual(centers.dtype, np.float64)
            expected = np.sum((x-centers[labels])**2)
            self.assertAlmostEqual(report["inertia"], expected, places=10)


if __name__ == "__main__":
    unittest.main()
