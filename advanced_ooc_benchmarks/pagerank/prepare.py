#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0.

"""Prepare a deterministic sparse PageRank graph and SystemDS import artifact."""

import argparse
import json
import sys
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertices", type=int, required=True)
    parser.add_argument("--out-degree", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.vertices < 1 or args.out_degree < 1:
        raise ValueError("vertices and out-degree must be positive")

    metadata_path = args.out / "metadata.json"
    ijv_path = args.out / "systemds" / "G.ijv"
    if metadata_path.exists() and ijv_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("vertices") == args.vertices
                and metadata.get("out_degree") == args.out_degree
                and metadata.get("seed") == args.seed):
            print(f"Reusing prepared PageRank import artifact at {ijv_path}")
            return

    rng = np.random.default_rng(args.seed)
    degrees = rng.integers(1, 2 * args.out_degree + 1, size=args.vertices, dtype=np.int32)
    sources = np.repeat(np.arange(args.vertices, dtype=np.int64), degrees)
    destinations = rng.integers(args.vertices, size=sources.size, dtype=np.int64)
    first_edge = np.r_[0, np.cumsum(degrees[:-1], dtype=np.int64)]
    destinations[first_edge] = np.arange(args.vertices, dtype=np.int64)
    values = 1.0 / degrees[sources]
    order = np.lexsort((sources, destinations))
    rows, cols, values = destinations[order], sources[order], values[order]
    starts = np.r_[True, (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])]
    offsets = np.flatnonzero(starts)
    rows, cols, values = rows[offsets], cols[offsets], np.add.reduceat(values, offsets)
    args.out.mkdir(parents=True, exist_ok=True)
    csr = args.out / "csr"
    csr.mkdir(exist_ok=True)
    row_ptr = np.zeros(args.vertices + 1, dtype=np.int64)
    np.add.at(row_ptr, rows + 1, 1)
    np.cumsum(row_ptr, out=row_ptr)
    row_ptr.tofile(csr / "row_ptr.i64")
    cols.astype(np.int32, copy=False).tofile(csr / "col_idx.i32")
    values.tofile(csr / "values.f64")
    np.zeros(args.vertices, dtype=np.uint8).tofile(args.out / "dangling.u8")

    systemds = args.out / "systemds"
    systemds.mkdir(exist_ok=True)
    with open(systemds / "G.ijv", "w", encoding="ascii") as handle:
        for start in range(0, values.size, 1 << 20):
            stop = min(start + (1 << 20), values.size)
            handle.write("\n".join(f"{row + 1} {col + 1} {value:.17g}"
                                    for row, col, value in zip(rows[start:stop], cols[start:stop], values[start:stop])))
            handle.write("\n")

    metadata_path.write_text(json.dumps({
        "vertices": args.vertices, "edges": int(values.size),
        "out_degree": args.out_degree, "mean_out_degree": float(degrees.mean()),
        "seed": args.seed,
        "dtype": {"row_ptr": "int64", "col_idx": "int32", "values": "float64"},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
