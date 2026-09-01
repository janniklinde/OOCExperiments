#!/usr/bin/env python3
"""NumPy mini-batch baseline matching the ffTrain workload shape.

Both large objects of this workload are disk-backed. The hidden activation is
the obvious one; the first-layer weight matrix is the one that matters here.
At D features and H hidden units the model is D*H*8 bytes and Nesterov SGD
needs three of them live at once -- the weights, their gradient, and the
velocity -- so above a few tens of thousands of hidden units the *parameters*
exceed the budget while the activations still fit comfortably.

That is the regime activation checkpointing does not reach: there is nothing to
recompute, because the object that does not fit is the model. Holding these as
np.memmap gives the baseline the strongest single-node formulation available
without a framework, so the comparison is about I/O efficiency rather than
about a naive allocation.
"""
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


# Column band for elementwise passes over the (D, H) parameter arrays. 64 MiB of
# float64 bounds the transient of every update regardless of how wide the layer is.
BAND_BYTES = 64 * 1024 * 1024


def column_band(rows, dtype=np.float64):
    """Columns per pass so that one band of a `rows`-tall array stays under BAND_BYTES."""
    return max(1, BAND_BYTES // max(1, rows * np.dtype(dtype).itemsize))


def nesterov_update(value, gradient, velocity, learning_rate, momentum):
    """In-place Nesterov step, banded so it never allocates a full copy of `velocity`.

    The original formulation needs the pre-update velocity, which for an in-RAM
    array is a cheap `.copy()`. At D*H = 10 GB that copy is the whole problem, so
    the pass runs over column bands instead: each band copies only its own slice.
    """
    if value.ndim == 1 or value.shape[0] * value.shape[1] * 8 <= BAND_BYTES:
        previous = np.array(velocity, copy=True)
        velocity *= momentum
        velocity -= learning_rate * gradient
        value += -momentum * previous + (1.0 + momentum) * velocity
        return
    band = column_band(value.shape[0])
    for first in range(0, value.shape[1], band):
        last = min(value.shape[1], first + band)
        previous = np.array(velocity[:, first:last], copy=True)
        updated = velocity[:, first:last] * momentum
        updated -= learning_rate * gradient[:, first:last]
        velocity[:, first:last] = updated
        value[:, first:last] += -momentum * previous + (1.0 + momentum) * updated


def init_affine(destination, fan_in, seed):
    """Fill a (fan_in, units) memmap with He-scaled normals, one column band at a time.

    ffTrain's affine::init draws the whole matrix in one call. Doing that here
    would materialise the model in RAM before it ever reached disk, so the draw
    is banded. The generator is advanced once per band from a single seeded
    stream, so the result is deterministic in `seed` and the layer width.
    """
    generator = np.random.default_rng(seed)
    scale = math.sqrt(2 / fan_in)
    band = column_band(fan_in)
    for first in range(0, destination.shape[1], band):
        last = min(destination.shape[1], first + band)
        destination[:, first:last] = generator.standard_normal((fan_in, last - first)) * scale
    return destination


def banded_sum(array):
    """Sum a possibly disk-backed (D, H) array without reading it all at once."""
    if array.ndim == 1 or array.shape[0] * array.shape[1] * 8 <= BAND_BYTES:
        return float(np.asarray(array).sum())
    band = column_band(array.shape[0])
    return float(sum(float(np.asarray(array[:, first:min(array.shape[1], first + band)]).sum())
                     for first in range(0, array.shape[1], band)))


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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work_root = Path(os.environ.get("BENCH_RUN_TMP", args.output.parent))
    work_root.mkdir(parents=True, exist_ok=True)
    work = work_root / f".{args.output.stem}-memmap-work"
    if work.exists():
        raise FileExistsError(f"stale memmap work directory exists: {work}")
    work.mkdir()
    succeeded = False
    try:
        # ffTrain seeds each affine::init call independently; its dropout calls
        # use an unseeded random mask for every batch. W1 and its two SGD
        # companions are the model, and at wide hidden layers they are what does
        # not fit, so all three live on disk. W2 is (hidden, 1) and b1 is
        # (1, hidden): both are a single vector and stay resident.
        w1 = init_affine(create_memmap(work / "w1.f64", (cols, hidden)), cols, args.seed)
        w2 = np.random.default_rng(args.seed).standard_normal((hidden, 1)) * math.sqrt(2 / hidden)
        b1, b2 = np.zeros((1, hidden)), np.zeros((1, 1))
        dropout_rng = np.random.default_rng()
        velocity_w1 = create_memmap(work / "velocity-w1.f64", (cols, hidden))
        velocity_w1[:] = 0
        dw1 = create_memmap(work / "dw1.f64", (cols, hidden))
        velocities = [velocity_w1] + [np.zeros_like(value) for value in (b1, w2, b2)]
        learning_rate, momentum, keep = args.learning_rate, 0.0, 0.35
        epoch_loss = 0.0
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
                # (cols, hidden), the same size as the model: written straight
                # into its memmap rather than returned as a fresh array.
                np.matmul(np.asarray(x).T, hidden_gradient, out=dw1)
                db1 = hidden_gradient.sum(axis=0, keepdims=True)
                for value, gradient, velocity in zip((w2, b2, w1, b1), (dw2, db2, dw1, db1),
                                                      (velocities[2], velocities[3], velocities[0], velocities[1])):
                    nesterov_update(value, gradient, velocity, learning_rate, momentum)
                del activation, active, mask, hidden_gradient
            momentum += (0.999 - momentum) / (args.epochs - epoch)
            learning_rate *= 0.99
        model_checksum = banded_sum(w1) + float(w2.sum())
        # Written before the work directory goes away; W1 is the model, so the
        # SystemDS arm writes it too and the volume is symmetric across arms.
        np.save(args.output.with_name(args.output.stem + "-W1.npy"), np.asarray(w1))
        succeeded = True
    finally:
        if succeeded:
            del w1, velocity_w1, dw1
            shutil.rmtree(work)
        else:
            print(f"memmap intermediates await runner cleanup after failure: {work}",
                  file=sys.stderr)
    report = {"implementation": "python-mlp", "seconds": time.perf_counter() - start,
              "model_checksum": model_checksum, "last_epoch_loss": epoch_loss,
              "model_bytes": cols * hidden * 8, "activation_bytes": args.batch_size * hidden * 8}
    np.save(args.output.with_name(args.output.stem + "-W2.npy"), w2)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
