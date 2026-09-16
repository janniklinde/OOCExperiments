#!/usr/bin/env python3
"""Fixed-iteration Lee-Seung GNMF using automatic Dask chunks."""

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
from dask_support import create_client, load_zarr, resolve_zarr, store_tall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--zarr", type=Path,
                        help="override the prepared Zarr store for X "
                             "(default: <data>/zarr/X.zarr)")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0,
                        help="worker processes; 0 derives one per 3 GiB of the memory limit")
    parser.add_argument("--memory-limit", default="3GiB")
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rank, args.iterations, args.threads) < 1 or args.epsilon <= 0:
        raise ValueError("rank/iterations/threads and epsilon must be positive")
    client = create_client(args.threads, args.memory_limit, args.temporary_directory, workers=args.workers)
    compute_options = {}

    start = time.perf_counter()
    metadata = json.loads((args.data / "X.f64.json").read_text(encoding="utf-8"))
    rows, cols = metadata["rows"], metadata["cols"]
    X = load_zarr(resolve_zarr(args.data, args.zarr))
    row_ids = da.arange(1, rows + 1, chunks=X.chunks[0])[:, None]
    component_ids = np.arange(1, args.rank + 1, dtype=np.float64)[None, :]
    col_ids = np.arange(1, cols + 1, dtype=np.float64)[None, :]
    W = 0.01 + da.remainder(row_ids * component_ids + args.seed, 97.0) / 97.0
    H = 0.01 + np.remainder(component_ids.T * col_ids + 3 * args.seed, 89.0) / 89.0

    for _ in range(args.iterations):
        WtX, WtW = da.compute(W.T @ X, W.T @ W, **compute_options)
        H *= WtX / (WtW @ H + args.epsilon)
        numerator = X @ H.T
        denominator = W @ (H @ H.T)
        W = (W * numerator / (denominator + args.epsilon)).persist(**compute_options)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    w_path = args.output.with_name(args.output.stem + "-W.zarr")
    stored = store_tall(W, w_path)
    w_checksum = float(stored.sum().compute(**compute_options))
    np.save(args.output.with_name(args.output.stem + "-H.npy"), H)
    report = {
        "implementation": "dask-gnmf",
        "seconds": time.perf_counter() - start,
        "iterations": args.iterations,
        "w_checksum": w_checksum,
        "h_checksum": float(H.sum()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    client.close()


if __name__ == "__main__":
    main()
