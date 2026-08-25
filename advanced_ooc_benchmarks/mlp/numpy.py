#!/usr/bin/env python3
"""NumPy mini-batch baseline matching the ffTrain workload shape."""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

import numpy as np


def nesterov_update(value, gradient, velocity, learning_rate, momentum):
    previous = velocity.copy()
    velocity *= momentum
    velocity -= learning_rate * gradient
    value += -momentum * previous + (1.0 + momentum) * velocity


def create_memmap(path, shape, dtype=np.float64):
    """Create a disk-backed intermediate without first allocating it in RAM."""
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        raise ValueError("--output is required for disk-backed activation intermediates")
    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    n, cols, hidden = metadata["rows"], metadata["cols"], args.hidden_size
    if min(args.epochs, args.batch_size, hidden) < 1:
        raise ValueError("epochs, batch-size, and hidden-size must be positive")
    matrix = np.memmap(args.data / "X.f64", dtype=np.float64, mode="r", shape=(n, cols))
    labels = np.memmap(args.data / "nn_y.f64", dtype=np.float64, mode="r", shape=(n, 1))
    # ffTrain seeds each affine::init call independently; its dropout calls use
    # an unseeded random mask for every batch.
    w1 = np.random.default_rng(args.seed).standard_normal((cols, hidden)) * math.sqrt(2 / cols)
    w2 = np.random.default_rng(args.seed).standard_normal((hidden, 1)) * math.sqrt(2 / hidden)
    b1, b2 = np.zeros((1, hidden)), np.zeros((1, 1))
    dropout_rng = np.random.default_rng()
    velocities = [np.zeros_like(value) for value in (w1, b1, w2, b2)]
    learning_rate, momentum, keep = args.learning_rate, 0.0, 0.35
    epoch_loss = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(os.environ.get("BENCH_RUN_TMP", args.output.parent))
    work_root.mkdir(parents=True, exist_ok=True)
    work = work_root / f".{args.output.stem}-memmap-work"
    if work.exists():
        raise FileExistsError(f"stale memmap work directory exists: {work}")
    work.mkdir()
    succeeded = False
    try:
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            for first in range(0, n, args.batch_size):
                last = min(n, first + args.batch_size)
                x, y = matrix[first:last], labels[first:last]
                activation_shape = (last - first, hidden)
                activation = create_memmap(work / "activation.f64", activation_shape)
                active = create_memmap(work / "active.u8", activation_shape, np.uint8)
                mask = create_memmap(work / "mask.u8", activation_shape, np.uint8)

                np.matmul(x, w1, out=activation)
                np.add(activation, b1, out=activation)
                np.greater(activation, 0, out=active)
                np.maximum(activation, 0, out=activation)
                # Generate dropout in bounded row slices. A single RNG call for
                # this benchmark's 4.5 GiB activation would itself require a
                # full-size anonymous temporary before it could reach memmap.
                random_rows = max(1, (32 * 1024 * 1024) // (hidden * 8))
                for row in range(0, activation_shape[0], random_rows):
                    stop = min(activation_shape[0], row + random_rows)
                    mask[row:stop] = dropout_rng.random((stop - row, hidden)) < keep
                np.multiply(activation, mask, out=activation)
                np.divide(activation, keep, out=activation)

                prediction = activation @ w2 + b2
                # Match ffTrain's log_loss::forward/backward followed by sigmoid::backward.
                prediction = 1 / (1 + np.exp(-prediction))
                epoch_loss += float((-y * np.log(prediction) - (1 - y) * np.log(1 - prediction)).sum()) / len(x)
                d_prediction = (prediction - y) / (prediction * (1 - prediction)) / len(x)
                dout = d_prediction * prediction * (1 - prediction)
                dw2, db2 = activation.T @ dout, dout.sum(axis=0, keepdims=True)

                hidden_gradient = create_memmap(work / "hidden-gradient.f64", activation_shape)
                np.matmul(dout, w2.T, out=hidden_gradient)
                np.multiply(hidden_gradient, mask, out=hidden_gradient)
                np.divide(hidden_gradient, keep, out=hidden_gradient)
                np.multiply(hidden_gradient, active, out=hidden_gradient)
                dw1, db1 = x.T @ hidden_gradient, hidden_gradient.sum(axis=0, keepdims=True)
                for value, gradient, velocity in zip((w2, b2, w1, b1), (dw2, db2, dw1, db1),
                                                      (velocities[2], velocities[3], velocities[0], velocities[1])):
                    nesterov_update(value, gradient, velocity, learning_rate, momentum)
                del activation, active, mask, hidden_gradient
            momentum += (0.999 - momentum) / (args.epochs - epoch)
            learning_rate *= 0.99
        succeeded = True
    finally:
        if succeeded:
            shutil.rmtree(work)
        else:
            print(f"memmap intermediates await runner cleanup after failure: {work}",
                  file=sys.stderr)
    report = {"implementation": "python-mlp", "seconds": time.perf_counter() - start,
              "model_checksum": float(w1.sum() + w2.sum()), "last_epoch_loss": epoch_loss}
    np.save(args.output.with_name(args.output.stem + "-W1.npy"), w1)
    np.save(args.output.with_name(args.output.stem + "-W2.npy"), w2)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
