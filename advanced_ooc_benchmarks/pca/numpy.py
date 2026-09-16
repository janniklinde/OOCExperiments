#!/usr/bin/env python3
"""Centered covariance PCA over a whole row-major FP64 memmap."""

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
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    shape = (metadata["rows"], metadata["cols"])
    if not 1 <= args.components <= shape[1] or shape[0] < 2:
        raise ValueError("components must be in [1, cols] and at least two rows are required")
    X = np.memmap(args.data / "X.f64", dtype=np.float64, mode="r", shape=shape)
    sums = np.sum(X, axis=0)
    gram = X.T @ X
    center_correction = np.outer(sums, sums) / shape[0]
    covariance = (gram - center_correction) / (shape[0] - 1)
    all_values, all_vectors = np.linalg.eigh(covariance)
    decreasing_idx = np.argsort(all_values)[::-1][:args.components]
    eigenvalues = all_values[decreasing_idx]
    components = all_vectors[:, decreasing_idx]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    score_path = args.output.with_name(args.output.stem + "-scores.npy")
    scores = np.lib.format.open_memmap(
        score_path, mode="w+", dtype=np.float64, shape=(shape[0], args.components))
    np.matmul(X, components, out=scores)
    sum_projection = sums @ components
    np.subtract(scores, sum_projection / shape[0], out=scores)
    score_norm_sq = float(np.einsum("ij,ij->", scores, scores, optimize=True))
    scores.flush()
    np.save(args.output.with_name(args.output.stem + "-components.npy"), components)
    np.save(args.output.with_name(args.output.stem + "-eigenvalues.npy"), eigenvalues.reshape(-1, 1))
    report = {"implementation": "numpy-pca", "seconds": time.perf_counter() - start,
              "components": args.components, "score_norm_sq": score_norm_sq,
              "eigenvalues": eigenvalues.tolist()}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
