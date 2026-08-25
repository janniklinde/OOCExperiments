#!/usr/bin/env python3
"""Prepare bounded-memory binned inputs for the RandomForest benchmark."""

import argparse
import json
import os
import sys
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

import numpy as np


def valid_size(path, size):
    return path.is_file() and path.stat().st_size == size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=256)
    args = parser.parse_args()
    if min(args.rows, args.cols, args.bins, args.chunk_mib) < 1:
        raise ValueError("rows, cols, bins, and chunk-mib must be positive")

    source_x = args.out / "X.f64"
    source_y = args.out / "nn_y.f64"
    output_x = args.out / "X_binned.u8"
    output_y = args.out / "class_y.u8"
    metadata_path = args.out / "randomforest-metadata.json"
    if not valid_size(source_x, args.rows * args.cols * 8):
        raise RuntimeError(f"missing or incompatible source matrix {source_x}")
    if not valid_size(source_y, args.rows * 8):
        raise RuntimeError(f"missing or incompatible source labels {source_y}")

    expected = {
        "rows": args.rows, "cols": args.cols, "bins": args.bins,
        "dtype": "uint8", "generator": "prepare_randomforest_dataset.py",
        "generator_version": 1,
    }
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = None
    metadata_matches = isinstance(metadata, dict) and all(
        metadata.get(key) == value for key, value in expected.items())
    if metadata_path.exists() and not metadata_matches:
        raise RuntimeError("refusing to replace incompatible RandomForest metadata")
    for path, size in ((output_x, args.rows * args.cols), (output_y, args.rows)):
        if path.exists() and not valid_size(path, size):
            raise RuntimeError(f"refusing to replace incompatible RandomForest input {path}")
    complete = valid_size(output_x, args.rows * args.cols) and valid_size(output_y, args.rows)
    if complete and metadata_matches:
        print(f"Keeping complete RandomForest inputs in {args.out}.", flush=True)
        return
    if not metadata_path.exists():
        if output_x.exists() or output_y.exists():
            raise RuntimeError("cannot verify existing RandomForest inputs without metadata")
        temporary_metadata = metadata_path.with_name(
            f".{metadata_path.name}.tmp-{os.getpid()}")
        temporary_metadata.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
        os.replace(temporary_metadata, metadata_path)

    missing_x = not output_x.exists()
    missing_y = not output_y.exists()
    x_tmp = output_x.with_name(f".{output_x.name}.tmp-{os.getpid()}") if missing_x else None
    y_tmp = output_y.with_name(f".{output_y.name}.tmp-{os.getpid()}") if missing_y else None
    source_matrix = np.memmap(source_x, dtype="<f8", mode="r",
                              shape=(args.rows, args.cols), order="C")
    source_labels = np.memmap(source_y, dtype="<f8", mode="r", shape=args.rows)
    boundaries = np.linspace(-2.0, 2.0, args.bins - 1)
    chunk_rows = max(1, (args.chunk_mib << 20) // (args.cols * 9))
    x_stream = x_tmp.open("wb") if missing_x else None
    y_stream = y_tmp.open("wb") if missing_y else None
    try:
        for start in range(0, args.rows, chunk_rows):
            end = min(args.rows, start + chunk_rows)
            if x_stream:
                binned = np.searchsorted(boundaries, source_matrix[start:end],
                                         side="right").astype(np.uint8)
                binned += 1
                binned.tofile(x_stream)
            if y_stream:
                np.asarray(source_labels[start:end] + 1, dtype=np.uint8).tofile(y_stream)
            print(f"binned rows {start}:{end} / {args.rows}", flush=True)
        for stream in (x_stream, y_stream):
            if stream:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
        if missing_x:
            if not valid_size(x_tmp, args.rows * args.cols):
                raise RuntimeError("RandomForest feature generation produced an unexpected byte count")
            os.replace(x_tmp, output_x)
        if missing_y:
            if not valid_size(y_tmp, args.rows):
                raise RuntimeError("RandomForest label generation produced an unexpected byte count")
            os.replace(y_tmp, output_y)
    except BaseException:
        for stream in (x_stream, y_stream):
            if stream and not stream.closed:
                stream.close()
        if x_tmp:
            x_tmp.unlink(missing_ok=True)
        if y_tmp:
            y_tmp.unlink(missing_ok=True)
        raise
    print(f"Prepared RandomForest inputs in {args.out}.", flush=True)


if __name__ == "__main__":
    main()
