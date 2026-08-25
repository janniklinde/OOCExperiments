#!/usr/bin/env python3
"""Fixed-iteration conjugate-gradient regression using automatic Dask chunks."""

import argparse
import json
import sys
import time
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

import dask.array as da
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--reg", type=float, default=1e-7)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.reg < 0 or args.tolerance < 0 or args.threads < 1:
        raise ValueError("iterations/threads must be positive and reg/tolerance non-negative")
    compute_options = {"scheduler": "threads", "num_workers": args.threads}

    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    shape = (metadata["rows"], metadata["cols"])
    mapped = np.memmap(args.data / "X.f64", dtype=np.float64, mode="r", shape=shape)
    mapped_y = np.memmap(args.data / "binary_y.f64", dtype=np.float64, mode="r",
                         shape=(shape[0], 1))
    matrix = da.from_array(mapped, chunks="auto")
    response = da.from_array(mapped_y, chunks=(matrix.chunks[0], (1,)))
    beta = np.zeros((shape[1], 1), dtype=np.float64)
    residual = -(matrix.T @ response).compute(**compute_options)
    direction = -residual
    residual_sq = (residual.T @ residual).item()
    target = residual_sq * args.tolerance * args.tolerance

    completed = 0
    while completed < args.iterations and residual_sq > target:
        projected = matrix @ direction
        curvature = (matrix.T @ projected).compute(**compute_options) + args.reg * direction
        alpha = residual_sq / (direction.T @ curvature).item()
        beta += alpha * direction
        residual += alpha * curvature
        old_residual_sq = residual_sq
        residual_sq = (residual.T @ residual).item()
        direction = -residual + (residual_sq / old_residual_sq) * direction
        completed += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-beta.npy"), beta)
    report = {"implementation": "dask-lmcg", "seconds": time.perf_counter() - start,
              "iterations": completed, "residual_norm": residual_sq ** 0.5,
              "beta_checksum": float(beta.sum())}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
