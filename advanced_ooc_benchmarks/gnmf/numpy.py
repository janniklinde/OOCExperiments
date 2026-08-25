#!/usr/bin/env python3
"""Fixed-iteration Lee-Seung GNMF over one complete FP64 memmap."""

import argparse
import json
import sys
import time
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

import numpy as np


def initialize(rows, cols, rank, seed):
    row_ids = np.arange(1, rows + 1, dtype=np.float64)[:, None]
    component_ids = np.arange(1, rank + 1, dtype=np.float64)[None, :]
    col_ids = np.arange(1, cols + 1, dtype=np.float64)[None, :]
    w = 0.01 + np.remainder(row_ids * component_ids + seed, 97.0) / 97.0
    h = 0.01 + np.remainder(component_ids.T * col_ids + 3 * seed, 89.0) / 89.0
    return w, h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rank < 1 or args.iterations < 1 or args.epsilon <= 0:
        raise ValueError("rank/iterations must be positive and epsilon must be greater than zero")

    start = time.perf_counter()
    metadata = json.loads((args.data / "X.f64.json").read_text(encoding="utf-8"))
    rows, cols = metadata["rows"], metadata["cols"]
    matrix = np.memmap(args.data / "X.f64", dtype=np.float64, mode="r",
                       shape=(rows, cols))
    w, h = initialize(rows, cols, args.rank, args.seed)
    for _ in range(args.iterations):
        h *= (w.T @ matrix) / ((w.T @ w) @ h + args.epsilon)
        w *= (matrix @ h.T) / (w @ (h @ h.T) + args.epsilon)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-W.npy"), w)
    np.save(args.output.with_name(args.output.stem + "-H.npy"), h)
    report = {
        "implementation": "numpy-gnmf",
        "seconds": time.perf_counter() - start,
        "iterations": args.iterations,
        "w_checksum": float(w.sum()),
        "h_checksum": float(h.sum()),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
