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
from dask_support import create_client, load_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="3GiB")
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rank, args.iterations, args.threads) < 1 or args.epsilon <= 0:
        raise ValueError("rank/iterations/threads and epsilon must be positive")
    client = create_client(args.threads, args.memory_limit, args.temporary_directory)
    compute_options = {}

    start = time.perf_counter()
    metadata = json.loads((args.data / "X.f64.json").read_text(encoding="utf-8"))
    rows, cols = metadata["rows"], metadata["cols"]
    matrix = load_matrix(args.data / "X.f64", (rows, cols))
    row_ids = da.arange(1, rows + 1, chunks=matrix.chunks[0])[:, None]
    component_ids = np.arange(1, args.rank + 1, dtype=np.float64)[None, :]
    col_ids = np.arange(1, cols + 1, dtype=np.float64)[None, :]
    w = 0.01 + da.remainder(row_ids * component_ids + args.seed, 97.0) / 97.0
    h = 0.01 + np.remainder(component_ids.T * col_ids + 3 * args.seed, 89.0) / 89.0

    for _ in range(args.iterations):
        wt_x, wt_w = da.compute(w.T @ matrix, w.T @ w, **compute_options)
        h *= wt_x / (wt_w @ h + args.epsilon)
        numerator = matrix @ h.T
        denominator = w @ (h @ h.T)
        w = (w * numerator / (denominator + args.epsilon)).persist(**compute_options)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    w_path = args.output.with_name(args.output.stem + "-W.npy")
    materialized_w = np.lib.format.open_memmap(
        w_path, mode="w+", dtype=np.float64, shape=(rows, args.rank))
    stored = da.store(w, materialized_w, lock=False, compute=False, return_stored=True)
    w_checksum = float(stored.sum().compute(**compute_options))
    materialized_w.flush()
    np.save(args.output.with_name(args.output.stem + "-H.npy"), h)
    report = {
        "implementation": "dask-gnmf",
        "seconds": time.perf_counter() - start,
        "iterations": args.iterations,
        "w_checksum": w_checksum,
        "h_checksum": float(h.sum()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    client.close()


if __name__ == "__main__":
    main()
