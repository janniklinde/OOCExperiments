#!/usr/bin/env python3
"""Whole-CSR SciPy ALS-CG baseline."""
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
    parser.add_argument("data", type=Path)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--reg", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    users, items, nnz = metadata["rows"], metadata["cols"], metadata["nnz"]
    root = args.data / "csr"
    indptr = np.memmap(root / "row_ptr.i64", dtype=np.int64, mode="r", shape=users + 1)
    indices = np.memmap(root / "col_idx.i32", dtype=np.int32, mode="r", shape=nnz)
    values = np.memmap(root / "values.f64", dtype=np.float64, mode="r", shape=nnz)
    user_factors = np.random.default_rng(args.seed).uniform(-0.5, 0.5, (users, args.rank))
    item_factors = np.random.default_rng(args.seed + 1).uniform(-0.5, 0.5, (items, args.rank))
    row_ids = np.repeat(np.arange(users), np.diff(indptr))

    def observed_product(left, right):
        return np.einsum("ij,ij->i", left[row_ids], right[indices])

    def cg_update(target, update_users):
        prediction = observed_product(user_factors, item_factors)
        residual_matrix = csr_matrix((prediction - values, indices, indptr), shape=(users, items))
        gradient = residual_matrix @ item_factors if update_users else residual_matrix.T @ user_factors
        gradient += args.reg * target
        residual, direction = -gradient, -gradient.copy()
        gradient_norm2 = residual_norm2 = float((gradient * gradient).sum())
        for _ in range(args.rank):
            if residual_norm2 <= 1e-9 * gradient_norm2:
                break
            product = observed_product(direction, item_factors) if update_users else observed_product(user_factors, direction)
            hessian = csr_matrix((product, indices, indptr), shape=(users, items))
            hessian_direction = hessian @ item_factors if update_users else hessian.T @ user_factors
            hessian_direction += args.reg * direction
            alpha = residual_norm2 / float((direction * hessian_direction).sum())
            target += alpha * direction
            residual -= alpha * hessian_direction
            previous, residual_norm2 = residual_norm2, float((residual * residual).sum())
            direction = residual + residual_norm2 / previous * direction

    for _ in range(args.iterations):
        cg_update(user_factors, True)
        cg_update(item_factors, False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output.with_name(args.output.stem + "-U.npy"), user_factors)
        np.save(args.output.with_name(args.output.stem + "-V.npy"), item_factors)
    report = {"implementation": "scipy-als-cg", "seconds": time.perf_counter() - start,
              "factor_norm": float(np.linalg.norm(user_factors) + np.linalg.norm(item_factors))}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
