#!/usr/bin/env python3
"""Two-pass whole-memmap randomized SVD baseline."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    matrix = np.memmap(args.data / "X.f64", dtype=np.float64, mode="r",
                       shape=(metadata["rows"], metadata["cols"]))
    omega = np.random.default_rng(args.seed).standard_normal((metadata["cols"], args.rank))
    sketch = matrix @ omega
    inverse = np.linalg.inv(np.linalg.cholesky(sketch.T @ sketch).T)
    basis = sketch @ inverse
    projection = basis.T @ matrix
    left_small, singular_values, right_transposed = np.linalg.svd(projection, full_matrices=False)
    left = basis @ left_small
    # Match the DML script: form U, write S and V, then report a U checksum.
    nnz_u = int(np.count_nonzero(left))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output.with_name(args.output.stem + "-S.npy"), singular_values.reshape(-1, 1))
        np.save(args.output.with_name(args.output.stem + "-V.npy"), right_transposed.T)
    report = {"implementation": "python-randomized_svd", "seconds": time.perf_counter() - start,
              "singular_values": singular_values[:args.rank].tolist(), "nnzU": nnz_u}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
