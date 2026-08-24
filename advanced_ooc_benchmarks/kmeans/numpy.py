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
    matrix = np.memmap(args.data / "X.f64", dtype=np.float64, mode="r", shape=shape)
    centers = np.array(matrix[:args.clusters], dtype=np.float64, copy=True)
    cluster_ids = np.arange(args.clusters)

    for _ in range(args.iterations):
        distances = -2.0 * (matrix @ centers.T) + np.sum(centers * centers, axis=1)
        labels = np.argmin(distances, axis=1)
        del distances
        membership = (labels[:, None] == cluster_ids).astype(np.float64)
        counts = membership.sum(axis=0)
        if np.any(counts == 0):
            raise RuntimeError("an empty cluster was encountered")
        centers = (membership.T @ matrix) / counts[:, None]
        del membership

    distances = -2.0 * (matrix @ centers.T) + np.sum(centers * centers, axis=1)
    labels = np.argmin(distances, axis=1)
    inertia = float(np.einsum("ij,ij->", matrix, matrix, optimize=True)
                    + np.min(distances, axis=1).sum())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-centers.npy"), centers)
    np.save(args.output.with_name(args.output.stem + "-labels.npy"), labels + 1)
    report = {"implementation": "numpy-kmeans", "seconds": time.perf_counter() - start,
              "clusters": args.clusters, "iterations": args.iterations, "inertia": inertia}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
