#!/usr/bin/env python3
"""FP64 MiniBatchKMeans: explicit full-data passes over canonical row-major input."""
import argparse
import json
from pathlib import Path
import sys
import time

# The sibling numpy.py is a workload, not the NumPy package.
script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--clusters", type=int, default=32)
    parser.add_argument("--passes", "--iterations", dest="passes", type=int, default=10,
                        help="complete sequential training passes (not Lloyd iterations)")
    parser.add_argument("--batch-rows", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    shape = int(metadata["rows"]), int(metadata["cols"])
    if metadata.get("dtype") != "float64" or min(shape) < 1:
        raise ValueError("expected nonempty canonical FP64 data")
    if not 1 <= args.clusters <= min(shape[0], args.batch_rows) or args.passes < 1:
        raise ValueError("invalid clusters, passes or batch size")
    if (args.data / "X.f64").stat().st_size != np.prod(shape) * 8:
        raise ValueError("input byte size disagrees with metadata")
    matrix = np.memmap(args.data / "X.f64", mode="r", dtype="<f8", shape=shape)
    model = MiniBatchKMeans(n_clusters=args.clusters,
                            init=np.array(matrix[:args.clusters], copy=True), n_init=1,
                            batch_size=args.batch_rows, random_state=104,
                            reassignment_ratio=0, compute_labels=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress = args.output.with_name(args.output.stem + "-progress.json")
    for epoch in range(args.passes):
        for first in range(0, shape[0], args.batch_rows):
            model.partial_fit(matrix[first:first+args.batch_rows])
        elapsed = time.perf_counter()-start
        status = {"completed_passes": epoch+1, "seconds": elapsed,
                  "rows_processed": (epoch+1)*shape[0]}
        progress.write_text(json.dumps(status, indent=2)+"\n")
        print(json.dumps(status), flush=True)
    training_seconds = time.perf_counter()-start
    centers = model.cluster_centers_
    assert centers.dtype == np.float64
    np.save(args.output.with_name(args.output.stem+"-centers.npy"), centers)
    labels = np.lib.format.open_memmap(args.output.with_name(args.output.stem+"-labels.npy"),
                                      mode="w+", dtype="<i8", shape=(shape[0],))
    inertia = 0.0
    for first in range(0, shape[0], args.batch_rows):
        last = min(first+args.batch_rows, shape[0])
        block = matrix[first:last]
        assigned = model.predict(block)
        labels[first:last] = assigned+1
        residual = block-centers[assigned]
        inertia += float(np.einsum("ij,ij->", residual, residual))
    labels.flush()
    result = {"implementation": "sklearn-minibatch-kmeans", "seconds": time.perf_counter()-start,
              "training_seconds": training_seconds, "passes": args.passes,
              "clusters": args.clusters, "batch_rows": args.batch_rows, "dtype": "float64",
              "inertia": inertia, "sklearn_version": sklearn.__version__,
              "initialization": "first k rows", "reassignment_ratio": 0,
              "rows_processed_training": shape[0]*args.passes, "partial_fit_steps": model.n_steps_}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
