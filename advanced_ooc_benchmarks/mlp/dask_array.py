#!/usr/bin/env python3
"""Dask mini-batch MLP with the first-layer model held out of core.

Two objects grow with the hidden width: the activation, which is (batch, H) and
is rebuilt every batch, and the first-layer weights, which are (D, H) and
survive the whole run. At wide layers the second dominates, and Nesterov SGD
needs three of them live -- weights, gradient, velocity.

Dask arrays are immutable, so a parameter that outlives a graph cannot simply be
mutated. Calling `.compute()` on it would pull the whole model into the client
process, which is the failure this workload is about. Instead W1 and its
velocity live in Zarr stores and are double-buffered: each batch reads the
current stores as Dask arrays, builds one graph for both updated arrays, and
writes them to the alternate stores. That is the same read-modify-write against
disk that a parameter-offload framework performs, expressed in Dask's own terms.

W2 is (H, 1) and b1 is (1, H) -- one vector each -- so they stay resident.
"""

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dask.array as da
import numpy as np
from dask_support import create_client, load_zarr, resolve_zarr, read_vector

# Column band for seeding W1 on disk without ever holding it whole.
BAND_BYTES = 64 * 1024 * 1024


def init_w1_store(path, cols, hidden, hidden_chunk, seed):
    """Write He-scaled normals into a Zarr store one column band at a time."""
    import zarr

    store = zarr.open(str(path), mode="w", shape=(cols, hidden),
                      chunks=(cols, hidden_chunk), dtype=np.float64)
    generator = np.random.default_rng(seed)
    scale = math.sqrt(2 / cols)
    band = max(hidden_chunk, (BAND_BYTES // (cols * 8)) // hidden_chunk * hidden_chunk)
    for first in range(0, hidden, band):
        last = min(hidden, first + band)
        store[:, first:last] = generator.standard_normal((cols, last - first)) * scale
    return store


def zeros_store(path, cols, hidden, hidden_chunk):
    import zarr

    return zarr.open(str(path), mode="w", shape=(cols, hidden),
                     chunks=(cols, hidden_chunk), dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--zarr", type=Path,
                        help="prepared Zarr store for X, the Dask arm's counterpart to "
                             "the SystemDS binary blocks (default: <data>/zarr/X.zarr)")
    parser.add_argument("--hidden-chunk", type=int, default=2048,
                        help="column chunk for the (D, H) parameter stores")
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="3GiB")
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden_size, args.hidden_chunk,
           args.threads) < 1:
        raise ValueError("epochs, batch-size, hidden-size, hidden-chunk and threads"
                         " must be positive")
    client = create_client(args.threads, args.memory_limit, args.temporary_directory)

    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    rows, cols, hidden = metadata["rows"], metadata["cols"], args.hidden_size
    hidden_chunk = min(args.hidden_chunk, hidden)
    matrix = load_zarr(resolve_zarr(args.data, args.zarr))
    labels = read_vector(args.data / "nn_y.f64", rows).reshape(-1, 1)

    # Parameter stores live beside the spill directory: they are working state of
    # this execution, and the runner clears that path before and after the run.
    model_root = args.temporary_directory / "model"
    if model_root.exists():
        shutil.rmtree(model_root)
    model_root.mkdir(parents=True, exist_ok=True)
    w1_stores = [model_root / "w1-a.zarr", model_root / "w1-b.zarr"]
    velocity_stores = [model_root / "velocity-a.zarr", model_root / "velocity-b.zarr"]
    init_w1_store(w1_stores[0], cols, hidden, hidden_chunk, args.seed)
    zeros_store(velocity_stores[0], cols, hidden, hidden_chunk)
    zeros_store(w1_stores[1], cols, hidden, hidden_chunk)
    zeros_store(velocity_stores[1], cols, hidden, hidden_chunk)
    current = 0

    w2 = np.random.default_rng(args.seed).standard_normal((hidden, 1)) * math.sqrt(2 / hidden)
    b1, b2 = np.zeros((1, hidden)), np.zeros((1, 1))
    velocity_b1 = np.zeros_like(b1)
    velocity_w2, velocity_b2 = np.zeros_like(w2), np.zeros_like(b2)
    learning_rate, momentum, keep = args.learning_rate, 0.0, 0.35
    epoch_loss = 0.0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for first in range(0, rows, args.batch_size):
            last = min(rows, first + args.batch_size)
            count = last - first
            x = matrix[first:last]
            y = labels[first:last]
            dask_w1 = da.from_zarr(str(w1_stores[current]))
            dask_velocity = da.from_zarr(str(velocity_stores[current]))
            dask_w2 = da.from_array(w2, chunks=((hidden_chunk,) * (hidden // hidden_chunk)
                                                + ((hidden % hidden_chunk,)
                                                   if hidden % hidden_chunk else ()), (1,)))

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
            dw1 = (x.T @ hidden_gradient).rechunk(dask_w1.chunks)
            db1 = hidden_gradient.sum(axis=0, keepdims=True)

            # Nesterov, fused into the same graph as the gradient so dw1 is built
            # once and never lands anywhere but the destination store.
            next_velocity = momentum * dask_velocity - learning_rate * dw1
            next_w1 = dask_w1 + (-momentum * dask_velocity
                                 + (1.0 + momentum) * next_velocity)
            other = 1 - current
            write_w1 = da.to_zarr(next_w1, str(w1_stores[other]), overwrite=True,
                                  compute=False)
            write_velocity = da.to_zarr(next_velocity, str(velocity_stores[other]),
                                        overwrite=True, compute=False)
            _, _, grad_b1, grad_w2, grad_b2, batch_loss = da.compute(
                write_w1, write_velocity, db1, dw2, db2, loss)
            current = other
            epoch_loss += float(batch_loss)

            for value, gradient, velocity in ((w2, grad_w2, velocity_w2),
                                              (b2, grad_b2, velocity_b2),
                                              (b1, grad_b1, velocity_b1)):
                previous = velocity.copy()
                velocity *= momentum
                velocity -= learning_rate * gradient
                value += -momentum * previous + (1.0 + momentum) * velocity
        momentum += (0.999 - momentum) / (args.epochs - epoch)
        learning_rate *= 0.99

    model_checksum = float(da.from_zarr(str(w1_stores[current])).sum().compute()) \
        + float(w2.sum())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-W2.npy"), w2)
    da.to_zarr(da.from_zarr(str(w1_stores[current])),
               str(args.output.with_name(args.output.stem + "-W1.zarr")), overwrite=True)
    report = {
        "implementation": "dask-mlp",
        "seconds": time.perf_counter() - start,
        "model_checksum": model_checksum,
        "last_epoch_loss": epoch_loss,
        "model_bytes": cols * hidden * 8,
        "activation_bytes": args.batch_size * hidden * 8,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    shutil.rmtree(model_root, ignore_errors=True)
    client.close()


if __name__ == "__main__":
    main()
