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
from dask_support import create_client, load_zarr, resolve_zarr, read_vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--zarr", type=Path,
                        help="override the prepared Zarr store for X "
                             "(default: <data>/zarr/X.zarr)")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--inner-iterations", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--reg", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0,
                        help="worker processes; 0 derives one per 3 GiB of the memory limit")
    parser.add_argument("--memory-limit", default="3GiB")
    parser.add_argument("--temporary-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.inner_iterations < 1 or args.threads < 1:
        raise ValueError("iterations, inner-iterations, and threads must be positive")
    if args.reg < 0 or args.tolerance < 0:
        raise ValueError("reg and tolerance must be non-negative")

    client = create_client(args.threads, args.memory_limit, args.temporary_directory, workers=args.workers)
    try:
        start = time.perf_counter()
        metadata = json.loads((args.data / "metadata.json").read_text())
        N, D, classes = metadata["rows"], metadata["cols"], metadata["classes"]
        K = classes - 1
        X = load_zarr(resolve_zarr(args.data, args.zarr))
        row_chunk = X.chunks[0][0]
        # Match DML's conversion of nonpositive labels into the baseline category.
        raw_labels = read_vector(args.data / "nn_y.f64", N)
        max_y = int(raw_labels.max())
        if raw_labels.min() <= 0:
            raw_labels = np.where(raw_labels <= 0, max_y + 1, raw_labels)
            max_y += 1
        classes = max_y
        K = classes - 1
        label_index = raw_labels - 1
        # Build the indicators in NumPy (n-by-classes is 64 MB at four million rows), then
        # hand them back to Dask on X's exact row chunking. A bare NumPy operand would be
        # captured whole by every task that touches it rather than sliced per block.
        Y = da.from_array(
            np.concatenate([(label_index == c).astype(np.float64) for c in range(classes)],
                           axis=1),
            chunks=(X.chunks[0], classes))

        # The vendored SystemDS implementation performs this full-input robustness scan
        # before training. The benchmark dataset contract excludes missing values, but
        # retain the scan so every arm performs the same logical input check. It shares
        # one pass over X with the row-norm bound the trust region is initialized from.
        has_nan, max_norm = da.compute(
            da.isnan(X).any(), da.sqrt((X * X).sum(axis=1)).max())
        if bool(has_nan):
            X = da.where(da.isnan(X), 0.0, X)
            max_norm = float(da.sqrt((X * X).sum(axis=1)).max().compute())
        max_norm = float(max_norm)

        B = np.zeros((D, K))
        P = da.full((N, classes), 1.0 / classes, chunks=(row_chunk, classes))

        def gradient(P, value):
            R = P[:, :K] - Y[:, :K]
            return (X.T @ R).compute() + args.reg * value

        def evaluate(value):
            """Probabilities and objective at `value`, both from a single pass over X."""
            logits = da.concatenate(
                [X @ value, da.zeros((N, 1), chunks=(row_chunk, 1))], axis=1)
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = da.exp(logits)
            total = exp_logits.sum(axis=1, keepdims=True)
            P = exp_logits / total
            negative_likelihood = da.log(total[:, 0]).sum() - (logits * Y).sum()
            P, negative_likelihood = dask.persist(P, negative_likelihood)
            obj = (0.5 * args.reg * float((value * value).sum())
                         + float(negative_likelihood.compute()))
            return P, obj

        delta = 0.5 * math.sqrt(D) / max_norm
        obj = N * math.log(classes)
        Grad = gradient(P, B)
        norm_Grad_initial = np.linalg.norm(Grad)
        completed = 0
        for outer in range(args.iterations):
            norm_Grad = np.linalg.norm(Grad)
            if norm_Grad < args.tolerance * (1.0 if outer == 0 else norm_Grad_initial):
                break
            completed += 1
            S = np.zeros_like(B)
            R = -Grad
            V = R.copy()
            norm_R2 = float((R * R).sum())
            is_trust_boundary_reached = False
            for _ in range(args.inner_iterations):
                # One compute() so the two matmuls share a single read of each X block.
                p = P[:, :K]
                Q = p * (X @ V)
                HV = (X.T @ (Q - p * Q.sum(axis=1, keepdims=True))).compute()
                HV += args.reg * V
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

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output.with_name(args.output.stem + "-B.npy"), B)
        report = {
            "implementation": "dask-multilogreg",
            "seconds": time.perf_counter() - start,
            "iterations": completed,
            "coefficient_norm": float(np.linalg.norm(B)),
            "objective": float(obj),
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        client.close()


if __name__ == "__main__":
    main()
