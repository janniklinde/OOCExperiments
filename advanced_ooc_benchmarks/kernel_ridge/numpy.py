#!/usr/bin/env python3
"""Exact kernel ridge regression with an RBF kernel, solved by conjugate gradient.

The Gram matrix is the whole point of the workload, so how it is stored is a
deliberate choice rather than an implementation detail:

* By default it is one in-RAM array, which is what `sklearn.kernel_ridge` builds
  and what knn/numpy.py does with its distance matrix. Above the memory limit
  this arm is OOM-killed, which is the honest outcome for the formulation a
  practitioner actually writes.
* `--kernel-memmap` builds the same matrix in row bands into a disk-backed
  memmap instead, so the arm survives and the comparison becomes one of I/O
  efficiency rather than of allocation strategy. Report whichever is stronger.

Either way the matrix is built once and re-read by every CG iteration; it is
never modified, so the ridge term is applied to the vector inside the product.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

import numpy as np

# Row band for the blocked kernel build. 64 MiB of float64 keeps the transient
# squared-distance band far below any profile's limit at every supported n.
BAND_BYTES = 64 * 1024 * 1024


def build_kernel(features, squared_norms, gamma, destination):
    """Fill `destination` with exp(-gamma * squared distances), one band at a time."""
    rows = features.shape[0]
    band = max(1, BAND_BYTES // (rows * 8))
    for first in range(0, rows, band):
        last = min(rows, first + band)
        chunk = features[first:last] @ features.T
        chunk *= -2.0
        chunk += squared_norms
        chunk += squared_norms[first:last, None]
        np.multiply(chunk, -gamma, out=chunk)
        np.exp(chunk, out=chunk)
        destination[first:last] = chunk
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--train-rows", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--reg", type=float, default=1.0)
    parser.add_argument("--cg-iterations", type=int, default=20)
    parser.add_argument("--kernel-memmap", action="store_true",
                        help="hold the Gram matrix on disk instead of in RAM")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.train_rows, args.cg_iterations) < 1:
        raise ValueError("train-rows and cg-iterations must be positive")

    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    shape = (metadata["rows"], metadata["cols"])
    if args.train_rows > shape[0]:
        raise ValueError(f"train-rows {args.train_rows} exceeds the dataset's {shape[0]}")
    matrix = np.memmap(args.data / "X.f64", dtype=np.float64, mode="r", shape=shape)
    response = np.fromfile(args.data / "nn_y.f64", dtype=np.float64,
                           count=shape[0]).reshape(-1, 1)
    n = args.train_rows
    features = np.asarray(matrix[:n], dtype=np.float64)
    y = response[:n]
    squared_norms = np.einsum("ij,ij->i", features, features)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = None
    succeeded = False
    try:
        if args.kernel_memmap:
            work_root = Path(os.environ.get("BENCH_RUN_TMP", args.output.parent))
            work_root.mkdir(parents=True, exist_ok=True)
            work = work_root / f".{args.output.stem}-kernel-work"
            if work.exists():
                raise FileExistsError(f"stale kernel work directory exists: {work}")
            work.mkdir()
            kernel = np.memmap(work / "kernel.f64", dtype=np.float64, mode="w+",
                               shape=(n, n))
        else:
            kernel = np.empty((n, n), dtype=np.float64)
        build_kernel(features, squared_norms, args.gamma, kernel)

        alpha = np.zeros((n, 1), dtype=np.float64)
        r = y.astype(np.float64, copy=True)
        p = r.copy()
        rs = float(r.ravel() @ r.ravel())
        for _ in range(args.cg_iterations):
            kp = kernel @ p + args.reg * p
            step = rs / float(p.ravel() @ kp.ravel())
            alpha += step * p
            r -= step * kp
            rs_next = float(r.ravel() @ r.ravel())
            p = r + (rs_next / rs) * p
            rs = rs_next
        succeeded = True
    finally:
        if work is not None:
            if succeeded:
                del kernel
                shutil.rmtree(work)
            else:
                print(f"kernel memmap awaits runner cleanup after failure: {work}",
                      file=sys.stderr)

    np.save(args.output.with_name(args.output.stem + "-alpha.npy"), alpha)
    report = {"implementation": "numpy-kernel-ridge",
              "seconds": time.perf_counter() - start,
              "train_rows": n, "cg_iterations": args.cg_iterations,
              "kernel_bytes": n * n * 8, "kernel_on_disk": bool(args.kernel_memmap),
              "alpha_sum": float(alpha.sum()), "residual_norm": float(np.sqrt(rs))}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
