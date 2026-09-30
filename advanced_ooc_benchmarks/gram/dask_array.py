#!/usr/bin/env python3
"""Materialize the FP64 Gram matrix X.T @ X from prepared Zarr chunks."""
import argparse
import json
import sys
import time
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from dask_support import create_client, load_zarr, resolve_zarr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--memory-limit", required=True)
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = create_client(args.threads, args.memory_limit,
                           args.temporary_directory, workers=args.workers)
    try:
        start = time.perf_counter()
        X = load_zarr(resolve_zarr(args.data))
        G = (X.T @ X).compute()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output.with_name(args.output.stem + "-G.npy"), G)
        report = {"implementation": "dask-gram",
                  "seconds": time.perf_counter() - start,
                  "gram_checksum": float(G.sum())}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        client.close()


if __name__ == "__main__":
    main()
