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

import dask.array as da
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rank, args.iterations, args.threads) < 1 or args.epsilon <= 0:
        raise ValueError("rank/iterations/threads and epsilon must be positive")
    compute_options = {"scheduler": "threads", "num_workers": args.threads}

    start = time.perf_counter()
    metadata = json.loads((args.data / "X.f64.json").read_text(encoding="utf-8"))
    rows, cols = metadata["rows"], metadata["cols"]
    mapped = np.memmap(args.data / "X.f64", dtype=np.float64, mode="r",
                       shape=(rows, cols))
    matrix = da.from_array(mapped, chunks="auto")
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

    materialized_w = w.compute(**compute_options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-W.npy"), materialized_w)
    np.save(args.output.with_name(args.output.stem + "-H.npy"), h)
    report = {
        "implementation": "dask-gnmf",
        "seconds": time.perf_counter() - start,
        "iterations": args.iterations,
        "w_checksum": float(materialized_w.sum()),
        "h_checksum": float(h.sum()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
