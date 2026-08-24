#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0.

"""Fixed-iteration PageRank through a SciPy CSR matrix backed by memmaps."""

import argparse
import json
import sys
import time
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

import numpy as np
from scipy.sparse import csr_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", type=Path)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 0 or not 0 <= args.alpha <= 1:
        raise ValueError("iterations must be non-negative and alpha must be in [0, 1]")

    start = time.perf_counter()
    metadata = json.loads((args.graph / "metadata.json").read_text(encoding="utf-8"))
    vertices, edges = metadata["vertices"], metadata["edges"]
    csr = args.graph / "csr"
    indptr = np.memmap(csr / "row_ptr.i64", dtype=np.int64, mode="r", shape=vertices + 1)
    index_name = "col_idx.i32" if metadata["dtype"]["col_idx"] == "int32" else "col_idx.i64"
    index_dtype = np.int32 if index_name.endswith("i32") else np.int64
    indices = np.memmap(csr / index_name, dtype=index_dtype, mode="r", shape=edges)
    values = np.memmap(csr / "values.f64", dtype=np.float64, mode="r", shape=edges)
    dangling = np.memmap(args.graph / "dangling.u8", dtype=np.uint8, mode="r", shape=vertices)
    transition = csr_matrix((values, indices, indptr), shape=(vertices, vertices), copy=False)

    rank = np.full(vertices, 1.0 / vertices, dtype=np.float64)
    for _ in range(args.iterations):
        rank = args.alpha * (transition.dot(rank) + rank[dangling != 0].sum() / vertices) \
            + (1 - args.alpha) * rank.sum() / vertices
    # SystemDS writes its rank vector in binary format. Keep output materialization
    # binary here as well so end-to-end wall time does not include text formatting.
    np.save(args.output, rank)
    seconds = time.perf_counter() - start
    print(json.dumps({"implementation": "scipy-csr-memmap", "vertices": vertices,
                      "edges": edges, "iterations": args.iterations, "alpha": args.alpha,
                      "seconds": seconds, "rank_sum": float(rank.sum())}))


if __name__ == "__main__":
    main()
