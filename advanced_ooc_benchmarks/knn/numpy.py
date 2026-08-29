#!/usr/bin/env python3
"""Brute-force kNN classification with the distance matrix held in memory.

The whole query-by-reference distance matrix is one NumPy array, matching the
formulation the DML arm submits. Selection then walks it k times in place: the
per-query argmin gathers the neighbour's label and the chosen cell is pushed out
of range by a scatter, so no second array of that size is ever allocated.
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

MASKED = 1e300


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--reference-rows", type=int, required=True)
    parser.add_argument("--query-rows", type=int, required=True)
    parser.add_argument("--neighbours", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.reference_rows, args.query_rows, args.neighbours) < 1:
        raise ValueError("reference-rows, query-rows and neighbours must be positive")

    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    shape = (metadata["rows"], metadata["cols"])
    if args.reference_rows + args.query_rows > shape[0]:
        raise ValueError("the reference and query blocks overlap")
    if args.neighbours > args.reference_rows:
        raise ValueError("neighbours cannot exceed the number of reference rows")
    matrix = np.memmap(args.data / "X.f64", dtype=np.float64, mode="r", shape=shape)
    response = np.fromfile(args.data / "nn_y.f64", dtype=np.float64,
                           count=shape[0]).reshape(-1, 1)
    first_query = shape[0] - args.query_rows
    reference = np.asarray(matrix[:args.reference_rows], dtype=np.float64)
    labels = response[:args.reference_rows]
    queries = np.asarray(matrix[first_query:], dtype=np.float64)
    query_labels = response[first_query:]

    # The query norms are omitted: constant along a row, so they cannot change
    # which reference is nearest. See the note in implementation.dml.
    distances = queries @ reference.T
    distances *= -2.0
    distances += np.einsum("ij,ij->i", reference, reference)
    votes = np.zeros((args.query_rows, 1), dtype=np.float64)

    rows = np.arange(args.query_rows)
    for _ in range(args.neighbours):
        nearest = np.argmin(distances, axis=1)
        votes += labels[nearest]
        distances[rows, nearest] += MASKED

    predictions = (votes * 2 >= args.neighbours).astype(np.float64)
    accuracy = float(np.mean(predictions == query_labels))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-predictions.npy"), predictions)
    report = {"implementation": "numpy-knn", "seconds": time.perf_counter() - start,
              "reference_rows": args.reference_rows, "neighbours": args.neighbours,
              "vote_sum": float(votes.sum()), "accuracy": accuracy}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
