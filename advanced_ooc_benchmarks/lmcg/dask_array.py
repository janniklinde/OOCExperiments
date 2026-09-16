#!/usr/bin/env python3
"""Fixed-iteration conjugate-gradient regression using automatic Dask chunks."""

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
    parser.add_argument("--zarr", type=Path,
                        help="override the prepared Zarr store for X "
                             "(default: <data>/zarr/X.zarr)")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--reg", type=float, default=1e-7)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0,
                        help="worker processes; 0 derives one per 3 GiB of the memory limit")
    parser.add_argument("--memory-limit", default="3GiB")
    parser.add_argument("--memory-target", type=float, default=0.60)
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.reg < 0 or args.tolerance < 0 or args.threads < 1:
        raise ValueError("iterations/threads must be positive and reg/tolerance non-negative")
    if not 0 < args.memory_target < 0.70:
        raise ValueError("memory-target must be between zero and the 0.70 spill threshold")
    client = create_client(args.threads, args.memory_limit, args.temporary_directory,
                           memory_target=args.memory_target, workers=args.workers)
    compute_options = {}

    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    shape = (metadata["rows"], metadata["cols"])
    X = load_zarr(resolve_zarr(args.data, args.zarr))
    y = read_vector(args.data / "binary_y.f64", shape[0])
    beta = np.zeros((shape[1], 1), dtype=np.float64)
    r = -(X.T @ y).compute(**compute_options)
    p = -r
    norm_r2 = (r.T @ r).item()
    target = norm_r2 * args.tolerance * args.tolerance

    i = 0
    while i < args.iterations and norm_r2 > target:
        Xp = X @ p
        q = (X.T @ Xp).compute(**compute_options) + args.reg * p
        alpha = norm_r2 / (p.T @ q).item()
        beta += alpha * p
        r += alpha * q
        old_norm_r2 = norm_r2
        norm_r2 = (r.T @ r).item()
        p = -r + (norm_r2 / old_norm_r2) * p
        i += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-beta.npy"), beta)
    report = {"implementation": "dask-lmcg", "seconds": time.perf_counter() - start,
              "iterations": i, "residual_norm": norm_r2 ** 0.5,
              "beta_checksum": float(beta.sum())}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    client.close()


if __name__ == "__main__":
    main()
