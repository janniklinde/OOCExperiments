#!/usr/bin/env python3
"""XGBoost histogram-tree baseline."""
import argparse
import json
import sys
import time
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--trees", type=int, default=20)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--reg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--binned", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    import xgboost as xgb
    metadata = json.loads((args.data / "metadata.json").read_text())
    name, dtype = ("X_binned.u8", np.uint8) if args.binned else ("X.f64", np.float64)
    matrix = np.memmap(args.data / name, dtype=dtype, mode="r", shape=(metadata["rows"], metadata["cols"]))
    labels = np.memmap(args.data / "nn_y.f64", dtype=np.float64, mode="r", shape=metadata["rows"])
    model = xgb.XGBClassifier(n_estimators=args.trees, max_depth=args.depth, learning_rate=args.learning_rate,
                              reg_lambda=args.reg, tree_method="hist", objective="binary:logistic",
                              n_jobs=args.threads, random_state=args.seed).fit(matrix, labels)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(args.output.with_name(args.output.stem + "-model.ubj")))
    report = {"implementation": "xgboost-hist", "seconds": time.perf_counter() - start,
              "trees": len(model.get_booster().get_dump())}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
