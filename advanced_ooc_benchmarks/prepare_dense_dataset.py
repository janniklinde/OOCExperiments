#!/usr/bin/env python3
"""Prepare a reproducible dense benchmark bundle in bounded memory.

The canonical bundle contains row-major X.f64, correlated {-1,+1} binary_y.f64,
correlated {0,1} nn_y.f64, and metadata.json. SystemDS conversion is separate.
"""

import argparse
import json
import os
import sys
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)
sys.path.insert(0, script_dir)

import numpy as np

from prepare_numpy import prepare_numpy


LAYOUT = "row-major raw files; matching SystemDS binary blocks written through the Python API"


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def basic_metadata_matches(metadata, rows, cols, classes):
    expected = {"rows": rows, "cols": cols, "classes": classes, "dtype": "float64"}
    return isinstance(metadata, dict) and all(metadata.get(key) == value
                                               for key, value in expected.items())


def generator_metadata_matches(metadata, rows, cols, classes, sparsity, seed, distribution):
    expected = {
        "rows": rows, "cols": cols, "classes": classes, "dtype": "float64",
        "sparsity_requested": sparsity, "seed": seed, "distribution": distribution,
        "generator": "prepare_dense_dataset.py", "generator_version": 1,
    }
    return isinstance(metadata, dict) and all(metadata.get(key) == value
                                               for key, value in expected.items())


def valid_size(path, size):
    return path.is_file() and path.stat().st_size == size


def write_labels(matrix_path, binary_path, nn_path, rows, cols, seed, chunk_mib):
    missing_binary = not binary_path.exists()
    missing_nn = not nn_path.exists()
    vector_bytes = rows * 8
    if binary_path.exists() and not valid_size(binary_path, vector_bytes):
        raise RuntimeError(f"refusing to replace incompatible label file {binary_path}")
    if nn_path.exists() and not valid_size(nn_path, vector_bytes):
        raise RuntimeError(f"refusing to replace incompatible label file {nn_path}")
    if not missing_binary and not missing_nn:
        return

    binary_tmp = binary_path.with_name("." + binary_path.name + f".tmp-{os.getpid()}")
    nn_tmp = nn_path.with_name("." + nn_path.name + f".tmp-{os.getpid()}")
    matrix = np.memmap(matrix_path, dtype="<f8", mode="r", shape=(rows, cols), order="C")
    weights = np.random.default_rng(np.random.SeedSequence([seed, 2])).standard_normal(cols)
    # Matrix view + score + labels stay under this approximate primary-array budget.
    chunk_rows = max(1, (chunk_mib << 20) // max(cols * 8 + 24, 1))
    binary_stream = binary_tmp.open("wb") if missing_binary else None
    nn_stream = nn_tmp.open("wb") if missing_nn else None
    try:
        for start in range(0, rows, chunk_rows):
            count = min(chunk_rows, rows - start)
            score = matrix[start:start + count] @ weights
            binary = np.where(score >= 0, 1.0, -1.0)
            if binary_stream:
                np.asarray(binary, dtype="<f8").tofile(binary_stream)
            if nn_stream:
                np.asarray((binary + 1.0) / 2.0, dtype="<f8").tofile(nn_stream)
            print(f"generated labels {start}:{start + count} / {rows}", flush=True)
        for stream in (binary_stream, nn_stream):
            if stream:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
        if missing_binary:
            if binary_tmp.stat().st_size != vector_bytes:
                raise RuntimeError("binary label generation produced an unexpected byte count")
            os.replace(binary_tmp, binary_path)
        if missing_nn:
            if nn_tmp.stat().st_size != vector_bytes:
                raise RuntimeError("neural-network label generation produced an unexpected byte count")
            os.replace(nn_tmp, nn_path)
    except BaseException:
        for stream in (binary_stream, nn_stream):
            if stream and not stream.closed:
                stream.close()
        binary_tmp.unlink(missing_ok=True)
        nn_tmp.unlink(missing_ok=True)
        raise


def prepare_dataset(output, rows, cols, classes=2, sparsity=1.0, seed=7,
                    distribution="normal", chunk_mib=256, accept_legacy=False):
    if classes != 2:
        raise ValueError("the current binary_y/nn_y benchmark contract requires classes=2")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    matrix_path = output / "X.f64"
    matrix_metadata_path = output / "X.f64.json"
    binary_path = output / "binary_y.f64"
    nn_path = output / "nn_y.f64"
    metadata_path = output / "metadata.json"
    matrix_bytes = rows * cols * 8
    vector_bytes = rows * 8
    metadata = read_json(metadata_path)
    complete = (valid_size(matrix_path, matrix_bytes) and
                valid_size(binary_path, vector_bytes) and valid_size(nn_path, vector_bytes))
    if complete and generator_metadata_matches(metadata, rows, cols, classes, sparsity,
                                                seed, distribution):
        print(f"Keeping complete generated raw dataset in {output}.", flush=True)
        return
    if complete and accept_legacy and basic_metadata_matches(metadata, rows, cols, classes):
        print(f"Keeping complete legacy raw dataset in {output}; seed/sparsity cannot be verified.",
              flush=True)
        return
    if metadata_path.exists() and not generator_metadata_matches(
            metadata, rows, cols, classes, sparsity, seed, distribution):
        if accept_legacy and basic_metadata_matches(metadata, rows, cols, classes):
            raise RuntimeError("an incomplete legacy dataset cannot be extended because its "
                               "seed and sparsity are unverifiable")
        raise RuntimeError(f"refusing to replace incompatible dataset metadata {metadata_path}")

    if matrix_path.exists():
        if not valid_size(matrix_path, matrix_bytes):
            raise RuntimeError(f"refusing to replace incompatible matrix {matrix_path}")
        matrix_metadata = read_json(matrix_metadata_path)
        if matrix_metadata is not None:
            expected = {"rows": rows, "cols": cols, "seed": seed,
                        "sparsity_requested": sparsity, "distribution": distribution}
            if not all(matrix_metadata.get(key) == value for key, value in expected.items()):
                raise RuntimeError(f"matrix provenance does not match the plan: {matrix_metadata_path}")
        else:
            raise RuntimeError(f"cannot verify existing matrix provenance without {matrix_metadata_path}")
    else:
        if matrix_metadata_path.exists():
            raise RuntimeError(f"orphan matrix metadata exists: {matrix_metadata_path}")
        prepare_numpy(matrix_path, rows, cols, sparsity, seed, chunk_mib, distribution)

    write_labels(matrix_path, binary_path, nn_path, rows, cols, seed, chunk_mib)
    matrix_metadata = read_json(matrix_metadata_path)
    dataset_metadata = {
        "rows": rows,
        "cols": cols,
        "classes": classes,
        "dtype": "float64",
        "layout": LAYOUT,
        "generator": "prepare_dense_dataset.py",
        "generator_version": 1,
        "distribution": distribution,
        "sparsity_requested": sparsity,
        "sparsity_actual": matrix_metadata.get("sparsity_actual") if matrix_metadata else None,
        "nnz": matrix_metadata.get("nnz") if matrix_metadata else None,
        "seed": seed,
    }
    temporary_metadata = output / f".metadata.json.tmp-{os.getpid()}"
    temporary_metadata.write_text(json.dumps(dataset_metadata, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    os.replace(temporary_metadata, metadata_path)
    print(f"Prepared canonical raw dataset in {output}.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--sparsity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--distribution", choices=("normal", "uniform"), default="normal")
    parser.add_argument("--chunk-mib", type=int, default=256)
    parser.add_argument("--accept-legacy", action="store_true")
    args = parser.parse_args()
    if min(args.rows, args.cols, args.classes, args.chunk_mib) < 1:
        raise ValueError("rows, cols, classes, and chunk-mib must be positive")
    if not 0.0 <= args.sparsity <= 1.0:
        raise ValueError("sparsity must be in [0, 1]")
    prepare_dataset(args.out, args.rows, args.cols, args.classes, args.sparsity,
                    args.seed, args.distribution, args.chunk_mib, args.accept_legacy)


if __name__ == "__main__":
    main()
