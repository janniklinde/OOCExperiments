#!/usr/bin/env python3
"""Dask kernel ridge regression: RBF Gram matrix, conjugate-gradient solve.

The Gram matrix is loop-invariant and read once per CG iteration, which puts
Dask's default policy under exactly the wrong incentive: with a lazy graph the
matrix is *recomputed* from the input on every `.compute()`, paying 2*n^2*d
flops per iteration to avoid 8*n^2 bytes of reads. `--persist-kernel` (the
default) materialises it once so Dask's spill machinery keeps it in
`local_directory` and re-reads it, which is what the SystemDS arm does and so is
the like-for-like comparison. `--no-persist-kernel` measures the recompute
policy instead; both are legitimate, and the gap between them is the finding.
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
from dask_support import create_client, load_zarr, resolve_zarr, read_vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--train-rows", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=1.0)
    parser.add_argument("--cg-iterations", type=int, default=20)
    parser.add_argument("--chunk-rows", type=int, required=True)
    parser.add_argument("--zarr", type=Path,
                        help="prepared Zarr store for X, the Dask arm's counterpart to "
                             "the SystemDS binary blocks (default: <data>/zarr/X.zarr)")
    parser.add_argument("--persist-kernel", dest="persist_kernel",
                        action="store_true", default=True)
    parser.add_argument("--no-persist-kernel", dest="persist_kernel",
                        action="store_false")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="3GiB")
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.train_rows, args.cg_iterations, args.chunk_rows, args.threads) < 1:
        raise ValueError("train-rows, cg-iterations, chunk-rows and threads must be positive")
    client = create_client(args.threads, args.memory_limit, args.temporary_directory)

    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    shape = (metadata["rows"], metadata["cols"])
    if args.train_rows > shape[0]:
        raise ValueError(f"train-rows {args.train_rows} exceeds the dataset's {shape[0]}")
    n = args.train_rows
    features = load_zarr(resolve_zarr(args.data, args.zarr))[:n]
    features = features.rechunk((args.chunk_rows, shape[1]))
    y = read_vector(args.data / "nn_y.f64", shape[0])[:n].reshape(-1, 1)

    squared_norms = (features ** 2).sum(axis=1, keepdims=True)
    sqdist = -2 * (features @ features.T) + squared_norms.T + squared_norms
    kernel = da.exp(-args.gamma * sqdist)
    # Square the row chunking onto the columns as well: a chunk of the Gram
    # matrix is then chunk_rows^2 cells, the unit Dask spills and reloads.
    kernel = kernel.rechunk((args.chunk_rows, args.chunk_rows))
    if args.persist_kernel:
        kernel = client.persist(kernel)

    alpha = np.zeros((n, 1), dtype=np.float64)
    r = y.astype(np.float64, copy=True)
    p = r.copy()
    rs = float(r.ravel() @ r.ravel())
    for _ in range(args.cg_iterations):
        dask_p = da.from_array(p, chunks=(args.chunk_rows, 1))
        kp = np.asarray((kernel @ dask_p).compute()) + args.reg * p
        step = rs / float(p.ravel() @ kp.ravel())
        alpha += step * p
        r -= step * kp
        rs_next = float(r.ravel() @ r.ravel())
        p = r + (rs_next / rs) * p
        rs = rs_next

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-alpha.npy"), alpha)
    report = {"implementation": "dask-kernel-ridge",
              "seconds": time.perf_counter() - start,
              "train_rows": n, "cg_iterations": args.cg_iterations,
              "kernel_bytes": n * n * 8, "kernel_persisted": bool(args.persist_kernel),
              "alpha_sum": float(alpha.sum()), "residual_norm": float(np.sqrt(rs))}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    client.close()


if __name__ == "__main__":
    main()
