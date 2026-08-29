#!/usr/bin/env python3
"""Random Fourier feature ridge regression over automatically chunked Dask arrays.

The contrast with the NumPy arm is that the feature map is never named as a
concrete array: `Z.T @ Z` and `Z.T @ y` are submitted as one graph, so each block
of Z is projected, shifted, cosined, accumulated into the two small results and
dropped. Only the D-by-D Gram and the D-by-1 right-hand side survive a block.
"""

import argparse
import json
import sys
import time
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dask.array as da
import numpy as np
from dask_support import create_client, load_zarr, read_vector, resolve_zarr


def read_projection(path, cols):
    """Read the prepared projection and the scale that goes with it."""
    metadata = json.loads(Path(str(path) + ".json").read_text())
    if metadata["cols"] != cols:
        raise ValueError(f"projection was built for {metadata['cols']} columns, not {cols}")
    weights = np.fromfile(str(path) + ".f64", dtype="<f8",
                          count=cols * metadata["half"]).reshape(cols, metadata["half"])
    if weights.shape[1] != metadata["half"]:
        raise ValueError(f"short read from {path}.f64")
    return weights, metadata["half"], metadata["scale"], metadata["features"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--zarr", type=Path,
                        help="override the prepared Zarr store for X "
                             "(default: <data>/zarr/X.zarr)")
    parser.add_argument("--projection", type=Path, required=True,
                        help="stem of the prepared projection, without .f64/.json")
    parser.add_argument("--reg", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="3GiB")
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.reg < 0:
        raise ValueError("reg must be non-negative")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    client = create_client(args.threads, args.memory_limit, args.temporary_directory)

    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    shape = (metadata["rows"], metadata["cols"])
    matrix = load_zarr(resolve_zarr(args.data, args.zarr))
    target = read_vector(args.data / "binary_y.f64", shape[0])
    response = da.from_array(target, chunks=(matrix.chunks[0], 1))
    weights, _, scale, width = read_projection(args.projection, shape[1])

    projected = matrix @ weights
    features = da.concatenate([da.cos(projected), da.sin(projected)], axis=1) * scale
    # One compute call so the two reductions share a single pass over the map.
    gram, rhs = da.compute(features.T @ features, features.T @ response)
    beta = np.linalg.solve(gram + args.reg * np.eye(width), rhs)

    squared = float(np.squeeze(beta.T @ gram @ beta - 2 * beta.T @ rhs))
    squared += float(np.squeeze(target.T @ target))
    residual_norm = float(np.sqrt(max(squared, 0.0)))
    model_norm = float(np.linalg.norm(beta))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-beta.npy"), beta)
    report = {"implementation": "dask-random-features",
              "seconds": time.perf_counter() - start, "features": width,
              "model_norm": model_norm, "residual_norm": residual_norm}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    client.close()


if __name__ == "__main__":
    main()
