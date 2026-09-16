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
    N, D, classes = metadata["rows"], metadata["cols"], metadata["classes"]
    X = np.memmap(args.data / "X.f64", dtype=np.float64, mode="r", shape=(N, D))
    # The vendored SystemDS implementation performs this full-input robustness
    # scan before training. The benchmark dataset contract excludes missing values,
    # but retain the scan so both arms perform the same logical input check.
    if np.isnan(X).any():
        X = np.where(np.isnan(X), 0.0, X)
    labels = np.memmap(args.data / "nn_y.f64", dtype=np.float64, mode="r", shape=N)
    max_y = int(labels.max())
    if labels.min() <= 0:
        labels = np.where(labels <= 0, max_y + 1, labels)
        max_y += 1
    classes = max_y
    labels = labels.astype(np.int64) - 1
    K = classes - 1
    B = np.zeros((D, K))
    P = np.full((N, classes), 1.0 / classes)

    def gradient(P, value):
        R = P[:, :K].copy()
        rows = np.arange(N)
        active = labels < K
        R[rows[active], labels[active]] -= 1
        return X.T @ R + args.reg * value

    max_norm = float(np.sqrt(np.einsum("ij,ij->i", X, X)).max())

    def evaluate(value):
        logits = np.column_stack((X @ value, np.zeros(N)))
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        P = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        obj = 0.5 * args.reg * float((value * value).sum())
        obj -= float(logits[np.arange(N), labels].sum())
        obj += float(np.log(exp_logits.sum(axis=1)).sum())
        return P, obj

    delta = 0.5 * math.sqrt(D) / max_norm
    obj = N * math.log(classes)
    Grad = gradient(P, B)
    norm_Grad_initial = np.linalg.norm(Grad)
    for outer in range(args.iterations):
        norm_Grad = np.linalg.norm(Grad)
        if norm_Grad < args.tolerance * (1.0 if outer == 0 else norm_Grad_initial):
            break
        S = np.zeros_like(B)
        R = -Grad
        V = R.copy()
        norm_R2 = float((R * R).sum())
        is_trust_boundary_reached = False
        for _ in range(args.inner_iterations):
            XV = X @ V
            p = P[:, :K]
            Q = p * XV
            HV = X.T @ (Q - p * Q.sum(axis=1, keepdims=True)) + args.reg * V
            alpha = norm_R2 / float((V * HV).sum())
            candidate = S + alpha * V
            if float((candidate * candidate).sum()) > delta * delta:
                is_trust_boundary_reached = True
                sv = float((S * V).sum())
                v2 = float((V * V).sum())
                s2 = float((S * S).sum())
                radius = math.sqrt(sv * sv + v2 * (delta * delta - s2))
                alpha = (delta * delta - s2) / (sv + radius) if sv >= 0 else (radius - sv) / v2
                S += alpha * V
                R -= alpha * HV
                break
            S = candidate
            R -= alpha * HV
            old = norm_R2
            norm_R2 = float((R * R).sum())
            if math.sqrt(norm_R2) <= 0.1 * norm_Grad:
                break
            V = R + norm_R2 / old * V
        P_new, obj_new = evaluate(B + S)
        gs = float((S * Grad).sum())
        qk = -0.5 * (gs - float((S * R).sum()))
        actred = obj - obj_new
        rho = actred / qk
        snorm = np.linalg.norm(S)
        if outer == 0:
            delta = min(delta, snorm)
        alpha2 = obj_new - obj - gs
        alpha = 4.0 if alpha2 <= 0 else max(0.25, -0.5 * gs / alpha2)
        if rho < 0.0001:
            delta = min(max(alpha, 0.25) * snorm, 0.5 * delta)
        elif rho < 0.25:
            delta = max(0.25 * delta, min(alpha * snorm, 0.5 * delta))
        elif rho < 0.75:
            delta = max(0.25 * delta, min(alpha * snorm, 4.0 * delta))
        else:
            delta = max(delta, min(alpha * snorm, 4.0 * delta))
        if rho > 0.0001:
            B += S
            P = P_new
            obj = obj_new
            Grad = gradient(P, B)
        if not is_trust_boundary_reached and abs(actred) < (abs(obj) + abs(obj_new)) * 1e-14:
            break
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output.with_name(args.output.stem + "-B.npy"), B)
    report = {"implementation": "python-multilogreg", "seconds": time.perf_counter() - start,
              "coefficient_norm": float(np.linalg.norm(B)), "objective": float(obj)}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
