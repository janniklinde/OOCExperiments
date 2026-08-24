#!/usr/bin/env python3
"""Whole-memmap sklearn diagonal-GMM baseline."""
import argparse
import json
import sys
import time
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

import numpy as np
from sklearn.mixture import GaussianMixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--model", choices=("VVV", "EEE", "VVI", "VII"), default="VVI")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--reg", type=float, default=1e-6)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    matrix = np.memmap(args.data / "X.f64", dtype=np.float64, mode="r",
                       shape=(metadata["rows"], metadata["cols"]))
    covariance = {"VVV": "full", "EEE": "tied", "VVI": "diag", "VII": "spherical"}[args.model]
    model = GaussianMixture(n_components=args.components, covariance_type=covariance, tol=args.tolerance,
                            reg_covar=args.reg, max_iter=args.iterations, n_init=1,
                            init_params="random", random_state=args.seed)
    model.fit_predict(matrix)
    # GaussianMixture.fit_predict already performs the final E-step used for labels,
    # matching the final E-step in the vendored DML implementation. Calling
    # predict_proba here would add another full, unmatched E-step.
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output.with_name(args.output.stem + "-mu.npy"), model.means_)
        np.save(args.output.with_name(args.output.stem + "-weight.npy"), model.weights_.reshape(-1, 1))
    report = {"implementation": "sklearn-gmm", "seconds": time.perf_counter() - start,
              "lower_bound": float(model.lower_bound_), "weights": model.weights_.tolist()}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
