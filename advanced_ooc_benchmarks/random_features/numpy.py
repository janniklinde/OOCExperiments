#!/usr/bin/env python3
"""Random Fourier feature ridge regression with the feature map held in memory.

The idiomatic NumPy formulation, and deliberately so: the closed form only ever
needs row-separable accumulations of Z, but expressing it as whole-array NumPy
materializes the full n-by-D feature map. That is the measurement -- the input is
under a gigabyte and fits everywhere, while Z is 4 to 32 GB, so this arm shows
where the intermediate rather than the input ends a run.
"""

import argparse
import json
import sys
import time
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

import numpy as np


def read_projection(path, cols):
    """Read the prepared projection and the scale that goes with it."""
    metadata = json.loads(Path(str(path) + ".json").read_text())
    if metadata["cols"] != cols:
        raise ValueError(f"projection was built for {metadata['cols']} columns, not {cols}")
    weights = np.fromfile(str(path) + ".f64", dtype="<f8",
                          count=cols * metadata["half"]).reshape(cols, metadata["half"])
    if weights.shape[1] != metadata["half"]:
        raise ValueError(f"short read from {path}.f64")
    return weights, metadata["half"], metadata["scale"], metadata["features"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--projection", type=Path, required=True,
                        help="stem of the prepared projection, without .f64/.json")
    parser.add_argument("--reg", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.reg < 0:
        raise ValueError("reg must be non-negative")

    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    shape = (metadata["rows"], metadata["cols"])
    matrix = np.memmap(args.data / "X.f64", dtype=np.float64, mode="r", shape=shape)
    response = np.fromfile(args.data / "binary_y.f64", dtype=np.float64,
                           count=shape[0]).reshape(-1, 1)
    weights, half, scale, width = read_projection(args.projection, shape[1])

    # The projection lands in the left half of the map and is then read once more
    # to fill the right half, so the only arrays alive at the peak are the map
    # itself and that half-width view of it.
    features = np.empty((shape[0], width), dtype=np.float64)
    projected = features[:, :half]
    np.matmul(matrix, weights, out=projected)
    np.sin(projected, out=features[:, half:])
    np.cos(projected, out=projected)
    features *= scale

    gram = features.T @ features
    rhs = features.T @ response
    del features
    beta = np.linalg.solve(gram + args.reg * np.eye(width), rhs)

    squared = float(np.squeeze(beta.T @ gram @ beta - 2 * beta.T @ rhs))
    squared += float(np.squeeze(response.T @ response))
    residual_norm = float(np.sqrt(max(squared, 0.0)))
    model_norm = float(np.linalg.norm(beta))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-beta.npy"), beta)
    report = {"implementation": "numpy-random-features",
              "seconds": time.perf_counter() - start, "features": width,
              "model_norm": model_norm, "residual_norm": residual_norm}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
