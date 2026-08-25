#!/usr/bin/env python3
"""Whole-memmap sklearn RandomForest baseline over the prepared binned input."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--trees", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--min-leaf", type=int, required=True)
    parser.add_argument("--min-split", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    start = time.perf_counter()
    metadata = json.loads((args.data / "randomforest-metadata.json").read_text(
        encoding="utf-8"))
    matrix = np.memmap(args.data / "X_binned.u8", dtype=np.uint8, mode="r",
                       shape=(metadata["rows"], metadata["cols"]))
    labels = np.memmap(args.data / "class_y.u8", dtype=np.uint8, mode="r",
                       shape=metadata["rows"])

    from joblib import dump
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=args.trees, criterion="gini", max_depth=args.depth,
        min_samples_split=args.min_split, min_samples_leaf=args.min_leaf,
        max_features="sqrt", bootstrap=False, n_jobs=args.threads,
        random_state=args.seed).fit(matrix, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_path = args.output.with_name(args.output.stem + "-model.joblib")
    dump(model, model_path, compress=0)
    report = {
        "implementation": "sklearn-random-forest",
        "seconds": time.perf_counter() - start,
        "trees": len(model.estimators_),
        "nodes": int(sum(tree.tree_.node_count for tree in model.estimators_)),
        "model": str(model_path),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
