#!/usr/bin/env python3
"""Fixed-iteration multinomial logistic-regression TRON using automatic Dask chunks."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dask
import dask.array as da
import numpy as np
from dask_support import create_client, default_row_chunk, load_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--inner-iterations", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--reg", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", default="3GiB")
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.inner_iterations < 1 or args.threads < 1:
        raise ValueError("iterations, inner-iterations, and threads must be positive")
    if args.reg < 0 or args.tolerance < 0:
        raise ValueError("reg and tolerance must be non-negative")

    client = create_client(args.threads, args.memory_limit, args.temporary_directory)
    try:
        start = time.perf_counter()
        metadata = json.loads((args.data / "metadata.json").read_text())
        n, d, classes = metadata["rows"], metadata["cols"], metadata["classes"]
        k = classes - 1
        row_chunk = default_row_chunk(d)
        matrix = load_matrix(args.data / "X.f64", (n, d), row_chunk=row_chunk)
        # nn_y holds {0,1}; the vendored SystemDS implementation turns the same file into a
        # {1,2} label column and then an indicator matrix whose first K columns are the
        # non-baseline categories. Class index 0 is therefore the positive label.
        raw_labels = load_matrix(args.data / "nn_y.f64", (n, 1), row_chunk=row_chunk)
        label_index = da.where(raw_labels > 0, 0.0, 1.0)
        indicators = da.concatenate(
            [(label_index == c).astype(np.float64) for c in range(classes)], axis=1)
        indicators = indicators.persist()

        # The vendored SystemDS implementation performs this full-input robustness scan
        # before training. The benchmark dataset contract excludes missing values, but
        # retain the scan so every arm performs the same logical input check. It shares
        # one pass over X with the row-norm bound the trust region is initialized from.
        has_nan, max_norm = da.compute(
            da.isnan(matrix).any(), da.sqrt((matrix * matrix).sum(axis=1)).max())
        if bool(has_nan):
            raise ValueError("benchmark input X.f64 contains NaN values")
        max_norm = float(max_norm)

        beta = np.zeros((d, k))
        probabilities = da.full((n, classes), 1.0 / classes, chunks=(row_chunk, classes))

        def gradient(probability, value):
            residual = probability[:, :k] - indicators[:, :k]
            return (matrix.T @ residual).compute() + args.reg * value

        def evaluate(value):
            """Probabilities and objective at `value`, both from a single pass over X."""
            logits = da.concatenate(
                [matrix @ value, da.zeros((n, 1), chunks=(row_chunk, 1))], axis=1)
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = da.exp(logits)
            total = exp_logits.sum(axis=1, keepdims=True)
            probability = exp_logits / total
            negative_likelihood = da.log(total[:, 0]).sum() - (logits * indicators).sum()
            probability, negative_likelihood = dask.persist(probability, negative_likelihood)
            objective = (0.5 * args.reg * float((value * value).sum())
                         + float(negative_likelihood.compute()))
            return probability, objective

        delta = 0.5 * math.sqrt(d) / max_norm
        objective = n * math.log(classes)
        grad = gradient(probabilities, beta)
        initial_norm = np.linalg.norm(grad)
        completed = 0
        for outer in range(args.iterations):
            grad_norm = np.linalg.norm(grad)
            if grad_norm < args.tolerance * initial_norm:
                break
            completed += 1
            step = np.zeros_like(beta)
            residual = -grad
            direction = residual.copy()
            residual_norm2 = float((residual * residual).sum())
            boundary = False
            for _ in range(args.inner_iterations):
                # One compute() so the two matmuls share a single read of each X block.
                p = probabilities[:, :k]
                q = p * (matrix @ direction)
                hv = (matrix.T @ (q - p * q.sum(axis=1, keepdims=True))).compute()
                hv += args.reg * direction
                alpha = residual_norm2 / float((direction * hv).sum())
                candidate = step + alpha * direction
                if float((candidate * candidate).sum()) > delta * delta:
                    boundary = True
                    sv = float((step * direction).sum())
                    v2 = float((direction * direction).sum())
                    s2 = float((step * step).sum())
                    radius = math.sqrt(sv * sv + v2 * (delta * delta - s2))
                    alpha = (delta * delta - s2) / (sv + radius) if sv >= 0 else (radius - sv) / v2
                    step += alpha * direction
                    residual -= alpha * hv
                    break
                step = candidate
                residual -= alpha * hv
                old = residual_norm2
                residual_norm2 = float((residual * residual).sum())
                if math.sqrt(residual_norm2) <= 0.1 * grad_norm:
                    break
                direction = residual + residual_norm2 / old * direction
            candidate_probability, candidate_objective = evaluate(beta + step)
            gs = float((step * grad).sum())
            predicted_reduction = -0.5 * (gs - float((step * residual).sum()))
            actual_reduction = objective - candidate_objective
            rho = actual_reduction / predicted_reduction
            step_norm = np.linalg.norm(step)
            if outer == 0:
                delta = min(delta, step_norm)
            alpha2 = candidate_objective - objective - gs
            alpha = 4.0 if alpha2 <= 0 else max(0.25, -0.5 * gs / alpha2)
            if rho < 0.0001:
                delta = min(max(alpha, 0.25) * step_norm, 0.5 * delta)
            elif rho < 0.25:
                delta = max(0.25 * delta, min(alpha * step_norm, 0.5 * delta))
            elif rho < 0.75:
                delta = max(0.25 * delta, min(alpha * step_norm, 4.0 * delta))
            else:
                delta = max(delta, min(alpha * step_norm, 4.0 * delta))
            if rho > 0.0001:
                beta += step
                probabilities = candidate_probability
                objective = candidate_objective
                grad = gradient(probabilities, beta)
            if not boundary and abs(actual_reduction) < (abs(objective) + abs(candidate_objective)) * 1e-14:
                break

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output.with_name(args.output.stem + "-B.npy"), beta)
        report = {
            "implementation": "dask-multilogreg",
            "seconds": time.perf_counter() - start,
            "iterations": completed,
            "coefficient_norm": float(np.linalg.norm(beta)),
            "objective": float(objective),
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        client.close()


if __name__ == "__main__":
    main()
