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
from dask_support import create_client, load_zarr, resolve_zarr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--zarr", type=Path)
    parser.add_argument("--clusters", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--memory-limit", default="3GiB")
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if min(args.clusters, args.iterations, args.threads) < 1:
        raise ValueError("clusters, iterations, and threads must be positive")

    client = create_client(
        args.threads, args.memory_limit, args.temporary_directory,
        workers=args.workers
    )

    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())

    if args.clusters > metadata["rows"]:
        raise ValueError("clusters cannot exceed the number of rows")

    X = load_zarr(resolve_zarr(args.data, args.zarr))
    k = args.clusters
    iterations = args.iterations

    C = np.asarray(X[:k])
    sum_x_sq = da.sum(X * X)

    for i in range(iterations):
        D = -2.0 * (X @ C.T) + np.sum(C * C, axis=1)
        min_d = da.min(D, axis=1)
        P = (D <= min_d[:, None]).astype(np.float64)
        P = P / da.sum(P, axis=1)[:, None]
        counts, C_num = da.compute(da.sum(P, axis=0), P.T @ X)
        C = C_num / counts[:, None]

    D = -2.0 * (X @ C.T) + np.sum(C * C, axis=1)
    # rowIndexMin selects the last column when distances tie.
    Y = k - 1 - da.argmin(D[:, ::-1], axis=1)
    min_d = da.min(D, axis=1)

    Y, sum_x_sq, min_d = da.compute(Y, sum_x_sq, da.sum(min_d))
    Y = Y + 1
    inertia = float(sum_x_sq + min_d)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_name(args.output.stem + "-centers.npy"), C)
    np.save(args.output.with_name(args.output.stem + "-labels.npy"), Y)

    report = {
        "implementation": "dask-kmeans",
        "seconds": time.perf_counter() - start,
        "clusters": k,
        "iterations": iterations,
        "inertia": inertia,
    }

    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    client.close()


if __name__ == "__main__":
    main()
