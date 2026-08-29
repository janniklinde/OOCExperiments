#!/usr/bin/env python3
"""Write the random Fourier projection every arm reads.

The projection is experimental setup, not workload: it does not depend on the
data, it is the same for every engine, and generating it inside each arm would
measure three random number generators instead of one feature map. Preparing it
once as a file also keeps the arms bit-identical by construction rather than by
argument, and keeps a small dense generator out of the streamed dataflow, which
is where it collides with backend limits that have nothing to do with the
benchmark.

Two encodings of one array: `.f64` for the NumPy and Dask arms and `.csv` with a
`.mtd` sidecar for DML, which has no reader for raw row-major files. The CSV is
written at full FP64 precision, so the two encodings hold the same doubles.
"""

import argparse
import json
import sys
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)
sys.path.append(script_dir)

import numpy as np
from projection import projection

GENERATOR_VERSION = 1


def prepare(out, cols, features, gamma, seed):
    weights, half, scale = projection(cols, features, gamma, seed)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"W-f{features}"
    weights.astype("<f8").tofile(str(stem) + ".f64")
    np.savetxt(str(stem) + ".csv", weights, delimiter=",", fmt="%.17g")
    (Path(str(stem) + ".csv.mtd")).write_text(json.dumps({
        "data_type": "matrix", "value_type": "double", "rows": cols, "cols": half,
        "format": "csv", "header": False, "sep": ","}, indent=2) + "\n")
    (Path(str(stem) + ".json")).write_text(json.dumps({
        "cols": cols, "features": features, "half": half, "gamma": gamma, "seed": seed,
        "scale": scale, "generator": "random_features/prepare_projection.py",
        "generator_version": GENERATOR_VERSION}, indent=2) + "\n")
    print(f"Prepared {cols}x{half} projection for {features} features in {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--features", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.cols < 1 or args.gamma <= 0:
        raise ValueError("cols must be positive and gamma must be positive")
    prepare(args.out, args.cols, args.features, args.gamma, args.seed)


if __name__ == "__main__":
    main()


