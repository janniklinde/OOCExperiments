#!/usr/bin/env python3
"""Materialize the FP64 Gram matrix X.T @ X from the canonical whole memmap."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    X = np.memmap(args.data / "X.f64", mode="r", dtype="<f8",
                  shape=(metadata["rows"], metadata["cols"]))
    G = X.T @ X
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-G.npy"), G)
    report = {"implementation": "numpy-gram",
              "seconds": time.perf_counter() - start,
              "gram_checksum": float(G.sum())}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
