#!/usr/bin/env python3
"""Native Vaex Lloyd K-means with deterministic first-k-row initialization."""
import argparse
import time
from pathlib import Path

import numpy as np
from support import load, output_path, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--layout", choices=["raw", "columnar"], default="raw")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    df, matrix, features = load(args.data, args.layout, args.threads, args.chunk_rows)
    if not 1 <= args.clusters <= len(df) or args.iterations < 1:
        raise ValueError("invalid clusters or iterations")
    from vaex.ml.cluster import KMeans

    class FixedIterations(KMeans):
        # Preserve Vaex's kernels; only disable its absolute-inertia early stop.
        def _is_done(self, old, new):
            return [False] * len(new)

    model = FixedIterations(features=features, n_clusters=args.clusters,
                            init=np.array(matrix[:args.clusters], copy=True).tolist(),
                            n_init=1, max_iter=args.iterations, verbose=False)
    model.fit(df)
    centers = np.asarray(model.cluster_centers, dtype=np.float64)
    if not np.isfinite(centers).all():
        raise RuntimeError("Vaex produced nonfinite centers (possibly an empty cluster)")
    np.save(output_path(args.output, "-centers.npy"), centers)
    prediction = model.transform(df)
    # Vaex assigns ties to the first centroid; preserve native semantics.
    labels = np.lib.format.open_memmap(output_path(args.output, "-labels.npy"),
                                      mode="w+", dtype="<i8", shape=(len(df),))
    # Native prediction is evaluated with the input columns in one output pass.
    # Its fit inertia describes pre-update centers; compute the final objective
    # from these same bounded chunks, without introducing another input scan.
    inertia = 0.0
    for first, last, columns in prediction.evaluate_iterator(
            features + [model.prediction_label], chunk_size=args.chunk_rows, array_type="numpy"):
        assigned = np.asarray(columns[-1], dtype=np.int64)
        labels[first:last] = assigned + 1
        for j, column in enumerate(columns[:-1]):
            residual = column - centers[assigned, j]
            inertia += float(np.dot(residual, residual))
    labels.flush()
    report(args.output, {"implementation": "vaex-kmeans", "seconds": time.perf_counter()-start,
                        "clusters": args.clusters, "iterations": len(model.inertias),
                        "inertia": inertia, "layout": args.layout, "threads": args.threads,
                        "chunk_rows": args.chunk_rows, "tie_policy": "first centroid"})


if __name__ == "__main__":
    main()
