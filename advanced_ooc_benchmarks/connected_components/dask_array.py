#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0.

"""Connected components by max-label propagation over sparse COO Dask chunks."""

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
from dask_support import create_client, load_csr_coo


def densify(value):
    """Dask reductions over sparse chunks return sparse results; make them ndarray."""
    return np.asarray(value.todense()) if hasattr(value, "todense") else np.asarray(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--band-rows", type=int, required=True,
                        help="graph rows per COO chunk")
    parser.add_argument("--iterations", type=int, default=0,
                        help="maximum label-propagation iterations, 0 = until convergence")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="3GiB")
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 0 or min(args.threads, args.band_rows) < 1:
        raise ValueError("iterations must be non-negative; threads and band-rows positive")

    client = create_client(args.threads, args.memory_limit, args.temporary_directory)
    try:
        start = time.perf_counter()
        metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
        vertices = metadata["vertices"]
        graph = load_csr_coo(args.data / "csr", vertices, args.band_rows)

        # The symmetry guard the vendored DML performs before propagating. One `compute`
        # so both aggregates share a single pass over the graph.
        row_sums, column_sums = da.compute(graph.sum(axis=1), graph.sum(axis=0))
        if not np.array_equal(densify(row_sums), densify(column_sums)):
            raise ValueError("input graph is not symmetric: row and column sums differ")

        labels = np.arange(1, vertices + 1, dtype=np.float64)
        completed = 0
        while args.iterations == 0 or completed < args.iterations:
            # max(rowMaxs(G * t(c)), c). Multiplying a COO chunk by the dense label row
            # stays sparse -- the implicit zeros are unchanged -- and the row max over a
            # fill value of 0 is the max over each vertex's neighbours, because every label
            # is at least 1. So a chunk's temporary is its own nonzero count, not its
            # dense width, and the label vector is the only dense operand.
            propagated = densify((graph * labels).max(axis=1).compute())
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
        np.save(args.output.with_name(args.output.stem + "-C.npy"),
                labels.reshape(vertices, 1))
        report = {"implementation": "dask-connected-components",
                  "seconds": time.perf_counter() - start, "vertices": vertices,
                  "edge_slots": metadata["edge_slots"], "band_rows": args.band_rows,
                  "iterations": completed, "components": components, "label_sum": label_sum}
        expected_components = metadata.get("expected_components")
        expected_label_sum = metadata.get("expected_label_sum")
        if expected_components is not None and components != expected_components:
            raise RuntimeError(f"found {components} components, expected {expected_components}")
        if expected_label_sum is not None and label_sum != expected_label_sum:
            raise RuntimeError(f"label sum {label_sum}, expected {expected_label_sum}")
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        client.close()


if __name__ == "__main__":
    main()
