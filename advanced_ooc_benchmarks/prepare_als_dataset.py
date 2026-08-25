#!/usr/bin/env python3
"""Prepare deterministic sparse ratings plus a SystemDS binary-block input for ALS."""
import argparse
import json
import os
import sys
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

import numpy as np


def valid_size(path, size):
    return path.is_file() and path.stat().st_size == size


def existing_csr_is_reusable(csr_dir, users, items, ratings_per_user, nnz):
    """Recognize a complete prior raw phase, including runs predating its metadata."""
    indptr_path = csr_dir / "row_ptr.i64"
    indices_path = csr_dir / "col_idx.i32"
    values_path = csr_dir / "values.f64"
    if not (valid_size(indptr_path, (users + 1) * 8) and
            valid_size(indices_path, nnz * 4) and valid_size(values_path, nnz * 8)):
        return False
    indptr = np.memmap(indptr_path, dtype=np.int64, mode="r", shape=users + 1)
    indices = np.memmap(indices_path, dtype=np.int32, mode="r", shape=nnz)
    values = np.memmap(values_path, dtype=np.float64, mode="r", shape=nnz)
    rows = (0, users // 2, users - 1)
    return (int(indptr[0]) == 0 and int(indptr[-1]) == nnz and
            all(int(indptr[row + 1] - indptr[row]) == ratings_per_user for row in rows) and
            all(np.all(np.diff(indices[indptr[row]:indptr[row + 1]]) > 0) for row in rows) and
            all(0 <= int(indices[indptr[row]]) < items and
                0 <= int(indices[indptr[row + 1] - 1]) < items for row in rows) and
            all(np.all((values[indptr[row]:indptr[row + 1]] >= 1) &
                       (values[indptr[row]:indptr[row + 1]] <= 5)) for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--users", type=int, required=True)
    parser.add_argument("--items", type=int, required=True)
    parser.add_argument("--ratings-per-user", type=int, default=40)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    if min(args.users, args.items, args.ratings_per_user) < 1:
        raise ValueError("ALS dimensions must be positive")
    if args.ratings_per_user > args.items:
        raise ValueError("ratings-per-user cannot exceed items")

    args.out.mkdir(parents=True, exist_ok=True)
    csr_dir = args.out / "csr"
    csr_dir.mkdir(exist_ok=True)
    nnz = args.users * args.ratings_per_user
    if existing_csr_is_reusable(csr_dir, args.users, args.items,
                                args.ratings_per_user, nnz):
        print(f"Reusing existing complete CSR data in {csr_dir}", flush=True)
    else:
        expected_paths = [csr_dir / "row_ptr.i64", csr_dir / "col_idx.i32",
                          csr_dir / "values.f64"]
        if any(path.exists() for path in expected_paths):
            raise RuntimeError(f"refusing to overwrite incomplete or incompatible CSR data in {csr_dir}")
        indptr = np.arange(0, nnz + 1, args.ratings_per_user, dtype=np.int64)
        indices_tmp = csr_dir / f".col_idx.i32.tmp-{os.getpid()}"
        values_tmp = csr_dir / f".values.f64.tmp-{os.getpid()}"
        row_ptr_tmp = csr_dir / f".row_ptr.i64.tmp-{os.getpid()}"
        indices = np.memmap(indices_tmp, dtype=np.int32, mode="w+", shape=nnz)
        values = np.memmap(values_tmp, dtype=np.float64, mode="w+", shape=nnz)
        rng = np.random.default_rng(args.seed)
        for user in range(args.users):
            first, last = indptr[user], indptr[user + 1]
            indices[first:last] = np.sort(
                rng.choice(args.items, args.ratings_per_user, replace=False))
            values[first:last] = rng.integers(1, 6, size=args.ratings_per_user)
        indices.flush()
        values.flush()
        indptr.tofile(row_ptr_tmp)
        del indices, values
        os.replace(indices_tmp, csr_dir / "col_idx.i32")
        os.replace(values_tmp, csr_dir / "values.f64")
        os.replace(row_ptr_tmp, csr_dir / "row_ptr.i64")

    indptr = np.memmap(csr_dir / "row_ptr.i64", dtype=np.int64, mode="r",
                       shape=args.users + 1)
    indices = np.memmap(csr_dir / "col_idx.i32", dtype=np.int32, mode="r", shape=nnz)
    values = np.memmap(csr_dir / "values.f64", dtype=np.float64, mode="r", shape=nnz)

    metadata = {"rows": args.users, "cols": args.items, "nnz": nnz,
                "ratings_per_user": args.ratings_per_user, "seed": args.seed,
                "dtype": "float64", "generator": "prepare_als_dataset.py",
                "generator_version": 1}
    # Publish raw provenance before native conversion so a failed conversion can resume.
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Prepared ALS CSR data in {args.out}")


if __name__ == "__main__":
    main()
