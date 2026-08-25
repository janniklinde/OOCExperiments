#!/usr/bin/env python3
"""Dask mini-batch MLP with chunks derived from the hidden activation shape."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dask.array as da
import numpy as np
from dask_support import create_client, load_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="3GiB")
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden_size, args.threads) < 1:
        raise ValueError("epochs, batch-size, hidden-size, and threads must be positive")
    client = create_client(args.threads, args.memory_limit, args.temporary_directory)
    compute_options = {}

    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    rows, cols, hidden = metadata["rows"], metadata["cols"], args.hidden_size
    features_path = args.data / "X.f64"
    labels_path = args.data / "nn_y.f64"
    w1 = np.random.default_rng(args.seed).standard_normal((cols, hidden)) * math.sqrt(2 / cols)
    w2 = np.random.default_rng(args.seed).standard_normal((hidden, 1)) * math.sqrt(2 / hidden)
    b1, b2 = np.zeros((1, hidden)), np.zeros((1, 1))
    velocities = [np.zeros_like(value) for value in (w1, b1, w2, b2)]
    learning_rate, momentum, keep = args.learning_rate, 0.0, 0.35
    epoch_loss = 0.0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for first in range(0, rows, args.batch_size):
            last = min(rows, first + args.batch_size)
            count = last - first
            # Ask Dask to size chunks for the expanded hidden activation, then
            # reuse those row/column chunks for the operands producing it.
            activation_layout = da.empty((count, hidden), chunks="auto", dtype=np.float64)
            row_chunks, hidden_chunks = activation_layout.chunks
            #row_chunks is uniform except for its tail, so its head is the chunk height
            batch_row_chunk = row_chunks[0]
            x = load_matrix(features_path, (rows, cols), row_chunk=batch_row_chunk,
                            row_range=(first, last))
            y = load_matrix(labels_path, (rows, 1), row_chunk=batch_row_chunk,
                            row_range=(first, last))
            dask_w1 = da.from_array(w1, chunks=((cols,), hidden_chunks))
            dask_w2 = da.from_array(w2, chunks=(hidden_chunks, (1,)))

            affine = x @ dask_w1 + b1
            relu = da.maximum(affine, 0)
            dropout_rng = da.random.RandomState(
                int(np.random.SeedSequence([args.seed, epoch, first]).generate_state(1)[0]))
            mask = dropout_rng.random_sample(relu.shape, chunks=relu.chunks) < keep
            dropped = relu * mask / keep
            prediction = 1 / (1 + da.exp(-(dropped @ dask_w2 + b2)))
            clipped = da.clip(prediction, np.finfo(np.float64).eps,
                              1 - np.finfo(np.float64).eps)
            loss = (-y * da.log(clipped) - (1 - y) * da.log(1 - clipped)).sum() / count
            d_prediction = (prediction - y) / (prediction * (1 - prediction)) / count
            dout = d_prediction * prediction * (1 - prediction)
            dw2 = dropped.T @ dout
            db2 = dout.sum(axis=0, keepdims=True)
            hidden_gradient = (dout @ dask_w2.T) * mask / keep * (affine > 0)
            dw1 = x.T @ hidden_gradient
            db1 = hidden_gradient.sum(axis=0, keepdims=True)
            grad_w1, grad_b1, grad_w2, grad_b2, batch_loss = da.compute(
                dw1, db1, dw2, db2, loss, **compute_options)
            epoch_loss += float(batch_loss)

            for value, gradient, velocity in zip(
                    (w2, b2, w1, b1),
                    (grad_w2, grad_b2, grad_w1, grad_b1),
                    (velocities[2], velocities[3], velocities[0], velocities[1])):
                previous = velocity.copy()
                velocity *= momentum
                velocity -= learning_rate * gradient
                value += -momentum * previous + (1.0 + momentum) * velocity
        momentum += (0.999 - momentum) / (args.epochs - epoch)
        learning_rate *= 0.99

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-W1.npy"), w1)
    np.save(args.output.with_name(args.output.stem + "-W2.npy"), w2)
    report = {
        "implementation": "dask-mlp",
        "seconds": time.perf_counter() - start,
        "model_checksum": float(w1.sum() + w2.sum()),
        "last_epoch_loss": epoch_loss,
        "activation_bytes": rows * hidden * 8,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    client.close()


if __name__ == "__main__":
    main()
