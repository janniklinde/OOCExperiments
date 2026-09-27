"""Shared FP64 dataset and output helpers; no input-sized copies."""
import json
from pathlib import Path

import numpy as np


def source_info(data):
    data = Path(data)
    metadata = json.loads((data / "metadata.json").read_text())
    shape = (int(metadata["rows"]), int(metadata["cols"]))
    if min(shape) < 1 or metadata.get("dtype", "float64") != "float64":
        raise ValueError("expected a nonempty FP64 dataset")
    path = data / "X.f64"
    stat = path.stat()
    if stat.st_size != shape[0] * shape[1] * 8:
        raise ValueError("X.f64 size disagrees with metadata")
    fingerprint = {"path": str(path.resolve()), "bytes": stat.st_size,
                   "mtime_ns": stat.st_mtime_ns, "shape": list(shape),
                   "metadata": metadata}
    return path, shape, fingerprint


def load(data, layout="raw", threads=4, chunk_rows=65536):
    if threads < 1 or chunk_rows < 1:
        raise ValueError("threads and chunk-rows must be positive")
    import vaex
    vaex.settings.main.thread_count = threads
    path, shape, fingerprint = source_info(data)
    if layout == "columnar":
        prepared = Path(data) / "vaex"
        if json.loads((prepared / "provenance.json").read_text()) != fingerprint:
            raise ValueError("stale columnar data; rerun prepare.py")
        matrix = np.load(prepared / "X.npy", mmap_mode="r")
        if matrix.shape != shape or matrix.dtype != np.dtype("<f8") or not matrix.flags.f_contiguous:
            raise ValueError("expected a Fortran-order FP64 columnar matrix")
    else:
        matrix = np.memmap(path, dtype="<f8", mode="r", shape=shape)
    features = [f"x{i}" for i in range(shape[1])]
    df = vaex.from_arrays(**{name: matrix[:, i] for i, name in enumerate(features)})
    # Bound the native map/reduce chunks as well as output materialization.
    df.executor.thread_pool = vaex.multithreading.ThreadPoolIndex(max_workers=threads)
    df.executor.chunk_size = chunk_rows
    return df, matrix, features


def output_path(output, suffix):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output.with_name(output.stem + suffix)


def materialize(df, expressions, path, chunk_rows, dtype="<f8"):
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype,
                                      shape=(len(df), len(expressions)))
    norm_sq = 0.0
    for first, last, columns in df.evaluate_iterator(expressions, chunk_size=chunk_rows,
                                                    array_type="numpy"):
        for j, column in enumerate(columns):
            array[first:last, j] = column
            if dtype == "<f8":
                norm_sq += float(np.dot(column, column))
    array.flush()
    return norm_sq


def report(output, values):
    import importlib.metadata
    values["versions"] = {name: importlib.metadata.version(name)
                          for name in ("vaex-core", "vaex-ml", "numpy")}
    Path(output).write_text(json.dumps(values, indent=2) + "\n")
    print(json.dumps(values))
