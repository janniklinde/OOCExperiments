#!/usr/bin/env python3
"""NumPy mini-batch baseline matching the ffTrain workload shape."""
import argparse
import json
import math
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
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for first in range(0, n, args.batch_size):
            last = min(n, first + args.batch_size)
            x, y = matrix[first:last], labels[first:last]
            affine = x @ w1 + b1
            relu = np.maximum(affine, 0)
            mask = dropout_rng.random(relu.shape) < keep
            dropped = relu * mask / keep
            prediction = 1 / (1 + np.exp(-(dropped @ w2 + b2)))
            # Match ffTrain's log_loss::forward/backward followed by sigmoid::backward.
            epoch_loss += float((-y * np.log(prediction) - (1 - y) * np.log(1 - prediction)).sum()) / len(x)
            d_prediction = (prediction - y) / (prediction * (1 - prediction)) / len(x)
            dout = d_prediction * prediction * (1 - prediction)
            dw2, db2 = dropped.T @ dout, dout.sum(axis=0, keepdims=True)
            hidden_gradient = (dout @ w2.T) * mask / keep * (affine > 0)
            dw1, db1 = x.T @ hidden_gradient, hidden_gradient.sum(axis=0, keepdims=True)
            for value, gradient, velocity in zip((w2, b2, w1, b1), (dw2, db2, dw1, db1),
                                                  (velocities[2], velocities[3], velocities[0], velocities[1])):
                nesterov_update(value, gradient, velocity, learning_rate, momentum)
        momentum += (0.999 - momentum) / (args.epochs - epoch)
        learning_rate *= 0.99
    report = {"implementation": "python-mlp", "seconds": time.perf_counter() - start,
              "model_checksum": float(w1.sum() + w2.sum()), "last_epoch_loss": epoch_loss}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output.with_name(args.output.stem + "-W1.npy"), w1)
        np.save(args.output.with_name(args.output.stem + "-W2.npy"), w2)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
