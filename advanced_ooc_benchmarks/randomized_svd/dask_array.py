#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0.

"""Two-pass randomized SVD expressed as whole-matrix Dask operations."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="3GiB")
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    client = create_client(args.threads, args.memory_limit, args.temporary_directory)
    compute_options = {}
    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    shape = (metadata["rows"], metadata["cols"])
    matrix = load_matrix(args.data / "X.f64", shape)
    omega = np.random.default_rng(args.seed).standard_normal((shape[1], args.rank))

    sketch = (matrix @ omega).persist(**compute_options)
    gram = (sketch.T @ sketch).compute(**compute_options)
    inverse = np.linalg.inv(np.linalg.cholesky(gram).T)
    basis = sketch @ inverse
    projection = (basis.T @ matrix).compute(**compute_options)
    left_small, singular, right_transposed = np.linalg.svd(projection, full_matrices=False)
    left = (basis @ left_small).compute(**compute_options)
    nnz_u = int(np.count_nonzero(left))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output.with_name(args.output.stem + "-S.npy"), singular.reshape(-1, 1))
        np.save(args.output.with_name(args.output.stem + "-V.npy"), right_transposed.T)
    report = {"implementation": "dask-randomized-svd", "seconds": time.perf_counter() - start,
              "singular_values": singular[:args.rank].tolist(), "nnzU": nnz_u}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    client.close()


if __name__ == "__main__":
    main()
