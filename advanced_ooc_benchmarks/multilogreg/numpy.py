#!/usr/bin/env python3
"""Whole-memmap multinomial logistic-regression TRON baseline."""
import argparse
import json
import math
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
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--inner-iterations", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--reg", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()
    metadata = json.loads((args.data / "metadata.json").read_text())
    n, d, classes = metadata["rows"], metadata["cols"], metadata["classes"]
    matrix = np.memmap(args.data / "X.f64", dtype=np.float64, mode="r", shape=(n, d))
    # The vendored SystemDS implementation performs this full-input robustness
    # scan before training. The benchmark dataset contract excludes missing values,
    # but retain the scan so both arms perform the same logical input check.
    if np.isnan(matrix).any():
        raise ValueError("benchmark input X.f64 contains NaN values")
    labels = np.memmap(args.data / "nn_y.f64", dtype=np.float64, mode="r", shape=n)
    labels = np.where(labels > 0, 1, 2).astype(np.int64) - 1
    k = classes - 1
    beta = np.zeros((d, k))
    probabilities = np.full((n, classes), 1.0 / classes)

    def gradient(probability, value):
        residual = probability[:, :k].copy()
        rows = np.arange(n)
        active = labels < k
        residual[rows[active], labels[active]] -= 1
        return matrix.T @ residual + args.reg * value

    max_norm = float(np.sqrt(np.einsum("ij,ij->i", matrix, matrix)).max())

    def evaluate(value):
        logits = np.column_stack((matrix @ value, np.zeros(n)))
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probability = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        objective = 0.5 * args.reg * float((value * value).sum())
        objective -= float(logits[np.arange(n), labels].sum())
        objective += float(np.log(exp_logits.sum(axis=1)).sum())
        return probability, objective

    delta = 0.5 * math.sqrt(d) / max_norm
    objective = n * math.log(classes)
    grad = gradient(probabilities, beta)
    initial_norm = np.linalg.norm(grad)
    for outer in range(args.iterations):
        grad_norm = np.linalg.norm(grad)
        if grad_norm < args.tolerance * initial_norm:
            break
        step = np.zeros_like(beta)
        residual = -grad
        direction = residual.copy()
        residual_norm2 = float((residual * residual).sum())
        boundary = False
        for _ in range(args.inner_iterations):
            xv = matrix @ direction
            p = probabilities[:, :k]
            q = p * xv
            hv = matrix.T @ (q - p * q.sum(axis=1, keepdims=True)) + args.reg * direction
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
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output.with_name(args.output.stem + "-B.npy"), beta)
    report = {"implementation": "python-multilogreg", "seconds": time.perf_counter() - start,
              "coefficient_norm": float(np.linalg.norm(beta)), "objective": float(objective)}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
