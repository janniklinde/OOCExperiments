#!/usr/bin/env python3
"""Generate an arbitrarily large synthetic row-major FP64 matrix with bounded RAM."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def prepare_numpy(output, rows, cols, sparsity=1.0, seed=7, chunk_mib=256,
                  distribution="normal", force=False):
    """Write a raw little-endian FP64 matrix and return its metadata."""
    if min(rows, cols, chunk_mib) < 1:
        raise ValueError("rows, cols, and chunk_mib must be positive")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must be in [0, 1]")
    if distribution not in ("normal", "uniform"):
        raise ValueError("distribution must be 'normal' or 'uniform'")
    output = Path(output)
    metadata_path = Path(str(output) + ".json")
    if not force and (output.exists() or metadata_path.exists()):
        raise FileExistsError(f"refusing to replace {output} or {metadata_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name("." + output.name + f".tmp-{os.getpid()}")
    temporary_metadata = Path(str(temporary) + ".json")

    # During masking, the FP64 values, FP32 random field, and boolean mask can
    # briefly coexist. Account for all three in the requested working-set bound.
    bytes_per_cell = 8 + (4 + 1 if 0.0 < sparsity < 1.0 else 0)
    chunk_rows = max(1, (chunk_mib << 20) // (cols * bytes_per_cell))
    value_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    mask_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    nnz = 0
    try:
        with temporary.open("wb") as stream:
            for start in range(0, rows, chunk_rows):
                count = min(chunk_rows, rows - start)
                if sparsity == 0.0:
                    chunk = np.zeros((count, cols), dtype="<f8")
                elif distribution == "normal":
                    chunk = value_rng.standard_normal((count, cols), dtype=np.float64)
                else:
                    chunk = value_rng.uniform(-1.0, 1.0, size=(count, cols))
                if 0.0 < sparsity < 1.0:
                    mask = mask_rng.random((count, cols), dtype=np.float32) >= sparsity
                    chunk[mask] = 0.0
                    del mask
                nnz += int(np.count_nonzero(chunk))
                np.asarray(chunk, dtype="<f8", order="C").tofile(stream)
                print(f"generated rows {start}:{start + count} / {rows}", flush=True)
            stream.flush()
            os.fsync(stream.fileno())
        expected_size = rows * cols * 8
        if temporary.stat().st_size != expected_size:
            raise RuntimeError(f"generated {temporary.stat().st_size} bytes; expected {expected_size}")
        metadata = {
            "generator": "prepare_numpy.py",
            "generator_version": 1,
            "rows": rows,
            "cols": cols,
            "dtype": "float64",
            "endianness": "little",
            "layout": "row-major raw FP64",
            "distribution": distribution,
            "seed": seed,
            "sparsity_model": "independent Bernoulli",
            "sparsity_requested": sparsity,
            "nnz": nnz,
            "sparsity_actual": nnz / (rows * cols),
            "chunk_rows": chunk_rows,
            "size_bytes": expected_size,
        }
        temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
        os.replace(temporary, output)
        os.replace(temporary_metadata, metadata_path)
        return metadata
    except BaseException:
        temporary.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--sparsity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chunk-mib", type=int, default=256)
    parser.add_argument("--distribution", choices=("normal", "uniform"), default="normal")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    metadata = prepare_numpy(args.output, args.rows, args.cols, args.sparsity, args.seed,
                             args.chunk_mib, args.distribution, args.force)
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
