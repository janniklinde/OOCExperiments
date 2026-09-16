#!/usr/bin/env python3
"""Fixed-iteration whole-memmap binary L2-SVM baseline."""

import argparse
import json
import sys
import time
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

import numpy as np


def normalized_labels(path, rows):
    Y = np.memmap(path, dtype=np.float64, mode="r", shape=(rows, 1))
    label_min = float(Y.min())
    label_max = float(Y.max())
    if int(np.count_nonzero(Y == label_min) + np.count_nonzero(Y == label_max)) != rows:
        raise ValueError("L2-SVM requires exactly two label values")
    if label_min == label_max:
        raise ValueError("L2-SVM requires two distinct label values")
    if label_min != -1.0 or label_max != 1.0:
        return 2.0 / (label_max - label_min) * Y - (
            label_min + label_max) / (label_max - label_min)
    return Y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--inner-iterations", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--reg", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.inner_iterations < 1:
        raise ValueError("iterations and inner-iterations must be positive")
    if args.reg < 0 or args.tolerance < 0:
        raise ValueError("reg and tolerance must be non-negative")

    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    rows, cols = metadata["rows"], metadata["cols"]
    X = np.memmap(args.data / "X.f64", dtype=np.float64, mode="r",
                       shape=(rows, cols))
    Y = normalized_labels(args.data / "binary_y.f64", rows)
    w = np.zeros((cols, 1), dtype=np.float64)
    Xw = np.zeros((rows, 1), dtype=np.float64)
    g_old = X.T @ Y
    s = g_old.copy()
    obj = 0.5 * rows
    iter = 0

    while iter < args.iterations:
        Xd = X @ s
        step_sz = 0.0
        wd = args.reg * float((w.T @ s).item())
        dd = args.reg * float((s.T @ s).item())
        for _ in range(args.inner_iterations):
            out = np.maximum(0.0, 1.0 - Y * (Xw + step_sz * Xd))
            g = wd + step_sz * dd - float(np.sum(out * Y * Xd))
            sv = out > 0
            h = dd + float(np.sum(Xd * sv * Xd))
            step_sz -= g / h
            if not (g * g / h >= args.tolerance):
                break

        w += step_sz * s
        Xw += step_sz * Xd
        out = np.maximum(0.0, 1.0 - Y * Xw)
        obj = (0.5 * float(np.sum(out * out))
                     + args.reg / 2.0 * float(np.sum(w * w)))
        g_new = X.T @ (out * Y) - args.reg * w
        continuation = (step_sz * float((s.T @ g_old).item())
                        >= args.tolerance * obj
                        and float(np.sum(s * s)) != 0.0)
        be = (float((g_new.T @ g_new).item())
                / float((g_old.T @ g_old).item()))
        s = be * s + g_new
        g_old = g_new
        iter += 1
        if not continuation:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-model.npy"), w)
    report = {
        "implementation": "numpy-l2svm",
        "seconds": time.perf_counter() - start,
        "iterations": iter,
        "objective": obj,
        "model_norm": float(np.linalg.norm(w)),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
