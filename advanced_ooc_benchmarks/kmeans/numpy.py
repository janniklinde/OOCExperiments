#!/usr/bin/env python3
"""Deterministic fixed-iteration whole-memmap Lloyd KMeans baseline."""

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
    parser.add_argument("--clusters", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.clusters, args.iterations) < 1:
        raise ValueError("clusters and iterations must be positive")

    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    shape = (metadata["rows"], metadata["cols"])
    if args.clusters > shape[0]:
        raise ValueError("clusters cannot exceed the number of rows")
    X = np.memmap(args.data / "X.f64", dtype=np.float64, mode="r", shape=shape)
    C = np.array(X[:args.clusters], dtype=np.float64, copy=True)
    sum_x_sq = float(np.einsum("ij,ij->", X, X, optimize=True))

    for _ in range(args.iterations):
        D = -2.0 * (X @ C.T) + np.sum(C * C, axis=1)
        min_d = np.min(D, axis=1)
        P = (D <= min_d[:, None]).astype(np.float64)
        P /= P.sum(axis=1, keepdims=True)
        counts = P.sum(axis=0)
        C = (P.T @ X) / counts[:, None]
        del D, P

    D = -2.0 * (X @ C.T) + np.sum(C * C, axis=1)
    # rowIndexMin selects the last column when distances tie.
    Y = args.clusters - 1 - np.argmin(D[:, ::-1], axis=1)
    inertia = float(sum_x_sq + np.min(D, axis=1).sum())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-centers.npy"), C)
    np.save(args.output.with_name(args.output.stem + "-labels.npy"), Y + 1)
    report = {"implementation": "numpy-kmeans", "seconds": time.perf_counter() - start,
              "clusters": args.clusters, "iterations": args.iterations, "inertia": inertia}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
