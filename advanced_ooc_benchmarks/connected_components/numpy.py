#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0.

"""Connected components by max-label propagation over whole CSR memmaps."""

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
    parser.add_argument("--iterations", type=int, default=0,
                        help="maximum label-propagation iterations, 0 = until convergence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 0:
        raise ValueError("iterations must be non-negative")

    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    vertices, slots = metadata["vertices"], metadata["edge_slots"]
    csr = args.data / "csr"
    # Deliberately not a scipy.sparse.csr_matrix. Above two billion stored edges the row
    # pointer needs int64, scipy unifies both index arrays to one dtype, and the int32
    # column indices would be upcast -- doubling the largest array in the dataset, in RAM,
    # before any work starts. The reductions below are what a CSR matrix would run anyway.
    pointer = np.memmap(csr / "row_ptr.i64", dtype=np.int64, mode="r", shape=(vertices + 1,))
    columns = np.memmap(csr / "col_idx.i32", dtype=np.int32, mode="r", shape=(slots,))
    values = np.memmap(csr / "values.f64", dtype=np.float64, mode="r", shape=(slots,))

    starts = np.asarray(pointer[:-1])
    if not np.all(np.diff(np.asarray(pointer)) > 0):
        raise ValueError("input graph has an isolated vertex; segment reductions assume "
                         "every row carries at least one edge")

    # The vendored DML performs this symmetry guard before propagating. Both sides are
    # segment reductions over the stored edges, so neither needs an edge-sized temporary.
    row_sums = np.add.reduceat(values, starts)
    column_sums = np.bincount(columns, weights=values, minlength=vertices)
    if not np.array_equal(row_sums, column_sums):
        raise ValueError("input graph is not symmetric: row and column sums differ")

    labels = np.arange(1, vertices + 1, dtype=np.float64)
    completed = 0
    while args.iterations == 0 or completed < args.iterations:
        # max(rowMaxs(G * t(c)), c). Like the other NumPy arms this is expressed over the
        # whole input with no user-controlled row-block loop, which here means the gather
        # `labels[columns]` materializes one float per stored edge -- eight bytes times the
        # nonzero count -- before the segment max reduces it. Nothing in the algebra needs
        # that temporary; streaming it is exactly what the out-of-core arms do instead, so
        # this arm is expected to exhaust memory once the graph is large.
        propagated = np.maximum.reduceat(labels[columns], starts)
        updated = np.maximum(propagated, labels)
        difference = int(np.count_nonzero(updated != labels))
        labels = updated
        completed += 1
        if difference == 0:
            break

    identity = np.arange(1, vertices + 1, dtype=np.float64)
    components = int(np.count_nonzero(labels == identity))
    label_sum = int(labels.sum())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-C.npy"), labels.reshape(vertices, 1))
    report = {"implementation": "numpy-connected-components",
              "seconds": time.perf_counter() - start, "vertices": vertices,
              "edge_slots": slots, "iterations": completed,
              "components": components, "label_sum": label_sum}
    expected_components = metadata.get("expected_components")
    expected_label_sum = metadata.get("expected_label_sum")
    if expected_components is not None and components != expected_components:
        raise RuntimeError(f"found {components} components, expected {expected_components}")
    if expected_label_sum is not None and label_sum != expected_label_sum:
        raise RuntimeError(f"label sum {label_sum}, expected {expected_label_sum}")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
