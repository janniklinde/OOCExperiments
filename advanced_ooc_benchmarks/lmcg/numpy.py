#!/usr/bin/env python3
"""Fixed-iteration whole-memmap conjugate-gradient linear regression."""

import argparse
import json
import sys
import time
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--reg", type=float, default=1e-7)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.reg < 0 or args.tolerance < 0:
        raise ValueError("iterations must be positive and reg/tolerance non-negative")

    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    shape = (metadata["rows"], metadata["cols"])
    X = np.memmap(args.data / "X.f64", dtype=np.float64, mode="r", shape=shape)
    y = np.memmap(args.data / "binary_y.f64", dtype=np.float64, mode="r",
                         shape=(shape[0], 1))
    beta = np.zeros((shape[1], 1), dtype=np.float64)
    r = -(X.T @ y)
    p = -r
    norm_r2 = (r.T @ r).item()
    target = norm_r2 * args.tolerance * args.tolerance

    i = 0
    while i < args.iterations and norm_r2 > target:
        Xp = X @ p
        q = X.T @ Xp + args.reg * p
        alpha = norm_r2 / (p.T @ q).item()
        beta += alpha * p
        r += alpha * q
        old_norm_r2 = norm_r2
        norm_r2 = (r.T @ r).item()
        p = -r + (norm_r2 / old_norm_r2) * p
        i += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-beta.npy"), beta)
    report = {"implementation": "numpy-lmcg", "seconds": time.perf_counter() - start,
              "iterations": i, "residual_norm": norm_r2 ** 0.5,
              "beta_checksum": float(beta.sum())}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
