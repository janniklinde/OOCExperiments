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
    labels = np.memmap(path, dtype=np.float64, mode="r", shape=(rows, 1))
    label_min = float(labels.min())
    label_max = float(labels.max())
    if int(np.count_nonzero(labels == label_min) + np.count_nonzero(labels == label_max)) != rows:
        raise ValueError("L2-SVM requires exactly two label values")
    if label_min == label_max:
        raise ValueError("L2-SVM requires two distinct label values")
    if label_min != -1.0 or label_max != 1.0:
        return 2.0 / (label_max - label_min) * labels - (
            label_min + label_max) / (label_max - label_min)
    return labels


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
    matrix = np.memmap(args.data / "X.f64", dtype=np.float64, mode="r",
                       shape=(rows, cols))
    labels = normalized_labels(args.data / "binary_y.f64", rows)
    weights = np.zeros((cols, 1), dtype=np.float64)
    scores = np.zeros((rows, 1), dtype=np.float64)
    gradient = matrix.T @ labels
    direction = gradient.copy()
    objective = 0.5 * rows
    completed = 0

    while completed < args.iterations:
        projected = matrix @ direction
        step = 0.0
        wd = args.reg * float((weights.T @ direction).item())
        dd = args.reg * float((direction.T @ direction).item())
        for _ in range(args.inner_iterations):
            slack = np.maximum(0.0, 1.0 - labels * (scores + step * projected))
            line_gradient = wd + step * dd - float(np.sum(slack * labels * projected))
            active = slack > 0
            line_hessian = dd + float(np.sum(projected * active * projected))
            step -= line_gradient / line_hessian

        weights += step * direction
        scores += step * projected
        slack = np.maximum(0.0, 1.0 - labels * scores)
        objective = (0.5 * float(np.sum(slack * slack))
                     + args.reg / 2.0 * float(np.sum(weights * weights)))
        new_gradient = matrix.T @ (slack * labels) - args.reg * weights
        continuation = (step * float((direction.T @ gradient).item())
                        >= args.tolerance * objective
                        and float(np.sum(direction * direction)) != 0.0)
        beta = (float((new_gradient.T @ new_gradient).item())
                / float((gradient.T @ gradient).item()))
        direction = beta * direction + new_gradient
        gradient = new_gradient
        completed += 1
        if not continuation:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-model.npy"), weights)
    report = {
        "implementation": "numpy-l2svm",
        "seconds": time.perf_counter() - start,
        "iterations": completed,
        "objective": objective,
        "model_norm": float(np.linalg.norm(weights)),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
