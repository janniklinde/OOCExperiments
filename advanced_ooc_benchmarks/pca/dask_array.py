#!/usr/bin/env python3
"""Centered covariance PCA expressed using automatically chunked Dask arrays."""

import argparse
import json
import sys
import time
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dask.array as da
import numpy as np
from dask_support import create_client, load_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="3GiB")
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    client = create_client(args.threads, args.memory_limit, args.temporary_directory)
    compute_options = {}

    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    shape = (metadata["rows"], metadata["cols"])
    if not 1 <= args.components <= shape[1] or shape[0] < 2:
        raise ValueError("components must be in [1, cols] and at least two rows are required")
    matrix = load_matrix(args.data / "X.f64", shape)
    center, gram = da.compute(matrix.mean(axis=0), matrix.T @ matrix, **compute_options)
    covariance = gram / (shape[0] - 1)
    covariance -= (shape[0] / (shape[0] - 1)) * np.outer(center, center)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1][:args.components]
    eigenvalues = values[order]
    components = vectors[:, order]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    score_path = args.output.with_name(args.output.stem + "-scores.npy")
    scores = np.lib.format.open_memmap(
        score_path, mode="w+", dtype=np.float64, shape=(shape[0], args.components))
    score_array = matrix @ components - center @ components
    # Do not materialize the tall score matrix in the Python heap.  Returning
    # the stored chunks lets Dask share their computation with the checksum.
    stored = da.store(score_array, scores, lock=False, compute=False, return_stored=True)
    score_norm_sq = float((stored * stored).sum().compute(**compute_options))
    scores.flush()
    np.save(args.output.with_name(args.output.stem + "-components.npy"), components)
    np.save(args.output.with_name(args.output.stem + "-eigenvalues.npy"), eigenvalues.reshape(-1, 1))
    report = {"implementation": "dask-pca", "seconds": time.perf_counter() - start,
              "components": args.components, "score_norm_sq": score_norm_sq,
              "eigenvalues": eigenvalues.tolist()}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    client.close()


if __name__ == "__main__":
    main()
