"""Small numerical checks against the existing NumPy benchmark entrypoints."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from prepare import prepare
from support import load

HERE = Path(__file__).resolve().parent


class Baselines(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        self.x = np.random.default_rng(37).normal(size=(321, 5))
        self.x += np.arange(5) * 2
        self.x.astype("<f8").tofile(self.data / "X.f64")
        (self.data / "metadata.json").write_text(json.dumps(
            {"rows": len(self.x), "cols": 5, "dtype": "float64", "seed": 37}))

    def run_script(self, path, output, *flags):
        subprocess.run([sys.executable, str(path), str(self.data), *flags,
                        "--output", str(output)], check=True, capture_output=True, text=True)

    def test_layouts_and_preparation(self):
        path = prepare(self.data, 1)
        timestamp = path.stat().st_mtime_ns
        prepare(self.data, 1)
        self.assertEqual(timestamp, path.stat().st_mtime_ns)
        np.testing.assert_array_equal(np.load(path), self.x)
        for layout in ("raw", "columnar"):
            df, matrix, features = load(self.data, layout, 2, 17)
            self.assertEqual(df.executor.thread_pool._max_workers, 2)
            np.testing.assert_array_equal(df.evaluate(features[2]), self.x[:, 2])

    def test_stale_columnar(self):
        prepare(self.data, 1)
        with (self.data / "X.f64").open("r+b") as stream:
            stream.write(np.float64(99).tobytes())
        with self.assertRaisesRegex(ValueError, "stale"):
            load(self.data, "columnar")

    def test_multiband_preparation(self):
        matrix = np.random.default_rng(19).normal(size=(60000, 5))
        matrix.astype("<f8").tofile(self.data / "X.f64")
        (self.data / "metadata.json").write_text(json.dumps(
            {"rows": len(matrix), "cols": 5, "dtype": "float64"}))
        np.testing.assert_array_equal(np.load(prepare(self.data, 1)), matrix)

    def test_generated_plan(self):
        plan = self.data / "plan.yaml"
        subprocess.run([sys.executable, str(HERE / "make_plan.py"),
                        "--data", str(self.data), "--results", str(self.data / "results"),
                        "--output", str(plan), "--clusters", "4", "--components", "3",
                        "--memory", "16G", "8G", "4G"], check=True, capture_output=True)
        subprocess.run([sys.executable, str(HERE.parent / "benchmark_plan.py"),
                        str(plan), "--validate"], check=True, capture_output=True)

    def test_kmeans(self):
        expected = self.data / "numpy.json"
        self.run_script(HERE.parent / "kmeans/numpy.py", expected,
                        "--clusters", "4", "--iterations", "3")
        prepare(self.data, 1)
        for layout in ("raw", "columnar"):
            output = self.data / f"{layout}.json"
            self.run_script(HERE / "kmeans.py", output, "--clusters", "4",
                            "--iterations", "3", "--threads", "2",
                            "--chunk-rows", "17", "--layout", layout)
            np.testing.assert_allclose(np.load(self.data / f"{layout}-centers.npy"),
                                       np.load(self.data / "numpy-centers.npy"), atol=1e-12)
            np.testing.assert_array_equal(np.load(self.data / f"{layout}-labels.npy").ravel(),
                                          np.load(self.data / "numpy-labels.npy"))
            result = json.loads(output.read_text())
            self.assertEqual(result["iterations"], 3)
            self.assertAlmostEqual(result["inertia"], json.loads(expected.read_text())["inertia"], places=9)

    def test_pca(self):
        self.run_script(HERE.parent / "pca/numpy.py", self.data / "numpy.json",
                        "--components", "3")
        prepare(self.data, 1)
        for layout in ("raw", "columnar"):
            self.run_script(HERE / "pca.py", self.data / f"{layout}.json",
                            "--components", "3", "--threads", "2", "--chunk-rows", "17",
                            "--layout", layout)
            vectors = np.load(self.data / f"{layout}-components.npy")
            reference = np.load(self.data / "numpy-components.npy")
            signs = np.sign(np.sum(vectors * reference, axis=0))
            np.testing.assert_allclose(vectors * signs, reference, atol=1e-11)
            np.testing.assert_allclose(np.load(self.data / f"{layout}-eigenvalues.npy"),
                                       np.load(self.data / "numpy-eigenvalues.npy"), atol=1e-11)
            np.testing.assert_allclose(np.load(self.data / f"{layout}-scores.npy") * signs,
                                       np.load(self.data / "numpy-scores.npy"), atol=1e-10)


if __name__ == "__main__":
    unittest.main()
