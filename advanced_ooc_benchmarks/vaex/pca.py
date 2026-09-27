#!/usr/bin/env python3
"""Native Vaex centered, unscaled covariance PCA with materialized FP64 scores."""
import argparse
import time
from pathlib import Path

import numpy as np
from support import load, materialize, output_path, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--layout", choices=["raw", "columnar"], default="raw")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    df, matrix, features = load(args.data, args.layout, args.threads, args.chunk_rows)
    if len(df) < 2 or not 2 <= args.components <= len(features):
        raise ValueError("native Vaex PCA requires 2 <= components <= cols and rows >= 2")
    from vaex.ml.transformations import PCA
    model = PCA(features=features, n_components=args.components, whiten=False)
    model.fit(df)
    components = np.asarray(model.eigen_vectors_, dtype=np.float64)[:, :args.components]
    # Vaex uses population covariance; the suite uses sample covariance.
    values = np.asarray(model.eigen_values_, dtype=np.float64)[:args.components] * len(df)/(len(df)-1)
    np.save(output_path(args.output, "-components.npy"), components)
    np.save(output_path(args.output, "-eigenvalues.npy"), values.reshape(-1, 1))
    transformed = model.transform(df)
    norm = materialize(transformed, [f"{model.prefix}{i}" for i in range(args.components)],
                       output_path(args.output, "-scores.npy"), args.chunk_rows)
    report(args.output, {"implementation": "vaex-pca", "seconds": time.perf_counter()-start,
                        "components": args.components, "eigenvalues": values.tolist(),
                        "score_norm_sq": norm, "layout": args.layout, "threads": args.threads,
                        "chunk_rows": args.chunk_rows})


if __name__ == "__main__":
    main()
