#!/usr/bin/env python3
"""Brute-force kNN classification over automatically chunked Dask arrays.

The distance matrix is never named as a concrete array. Each neighbour round
submits the whole expression -- projection, masking, per-query argmin -- and Dask
streams it block by block, so the peak footprint is a few blocks rather than the
4 to 32 GB the matrix would occupy.

The cost of that is recomputation: the matrix is far too large to persist inside
any of these memory budgets, so every round rebuilds it from the inputs, and the
arm performs k products where the NumPy and DML arms perform one. That is not a
handicap imposed by this script but the actual choice Dask leaves open when an
intermediate exceeds memory, and reporting it is part of the comparison.
"""

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
from dask_support import create_client, load_zarr, read_vector, resolve_zarr

MASKED = 1e300


def _mask_taken(block, taken=None, block_info=None):
    """Push already-chosen cells out of range within one block of the matrix."""
    (first_row, last_row), (first_column, last_column) = block_info[0]["array-location"]
    relative = taken[first_row:last_row] - first_column
    inside = (relative >= 0) & (relative < last_column - first_column)
    if not inside.any():
        return block
    rows, columns = np.nonzero(inside)
    masked = block.copy()
    masked[rows, relative[rows, columns]] += MASKED
    return masked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--zarr", type=Path,
                        help="override the prepared Zarr store for X "
                             "(default: <data>/zarr/X.zarr)")
    parser.add_argument("--reference-rows", type=int, required=True)
    parser.add_argument("--query-rows", type=int, required=True)
    parser.add_argument("--neighbours", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="3GiB")
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.reference_rows, args.query_rows, args.neighbours) < 1:
        raise ValueError("reference-rows, query-rows and neighbours must be positive")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    client = create_client(args.threads, args.memory_limit, args.temporary_directory)

    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    shape = (metadata["rows"], metadata["cols"])
    if args.reference_rows + args.query_rows > shape[0]:
        raise ValueError("the reference and query blocks overlap")
    if args.neighbours > args.reference_rows:
        raise ValueError("neighbours cannot exceed the number of reference rows")
    matrix = load_zarr(resolve_zarr(args.data, args.zarr))
    response = read_vector(args.data / "nn_y.f64", shape[0])
    first_query = shape[0] - args.query_rows
    reference = matrix[:args.reference_rows]
    labels = response[:args.reference_rows]
    queries = matrix[first_query:]
    query_labels = response[first_query:]

    # The query norms are omitted: constant along a row, so they cannot change
    # which reference is nearest. See the note in implementation.dml.
    distances = queries @ reference.T * -2.0 + (reference * reference).sum(axis=1)
    votes = np.zeros((args.query_rows, 1), dtype=np.float64)
    taken = np.empty((args.query_rows, 0), dtype=np.int64)

    for _ in range(args.neighbours):
        candidate = distances
        if taken.shape[1]:
            candidate = da.map_blocks(_mask_taken, distances, taken=taken,
                                      dtype=distances.dtype)
        nearest = np.asarray(candidate.argmin(axis=1).compute())
        votes += labels[nearest]
        taken = np.concatenate([taken, nearest.reshape(-1, 1)], axis=1)

    predictions = (votes * 2 >= args.neighbours).astype(np.float64)
    accuracy = float(np.mean(predictions == query_labels))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-predictions.npy"), predictions)
    report = {"implementation": "dask-knn", "seconds": time.perf_counter() - start,
              "reference_rows": args.reference_rows, "neighbours": args.neighbours,
              "vote_sum": float(votes.sum()), "accuracy": accuracy}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    client.close()


if __name__ == "__main__":
    main()
