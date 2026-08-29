#!/usr/bin/env python3
"""Convert a canonical row-major FP64 matrix into an uncompressed Zarr store.

Dask's idiomatic out-of-core input is a chunked native store read through
``da.from_zarr``.  Giving the Dask arm its own prepared representation mirrors the
native SystemDS ``X-bs<blocksize>`` conversion, so neither runtime is measured
against a layout chosen for the other.

Two properties are deliberate and load-bearing:

* ``compressors=None``.  The suite compares physical read volume across arms, so the
  store must hold the same bytes as the raw input.  Blosc/zstd on normally
  distributed FP64 buys almost nothing and would both distort ``read.pdf`` and move
  work into the CPU chart.
* Chunks span whole rows by default.  Every workload in this suite is a row-local
  reduction (``t(X)%*%X``, ``t(X)%*%(X%*%v)``, ``t(P)%*%X``); splitting columns forces
  a cross-chunk combine that no arm's algorithm asks for.

The transfer is bounded: rows are copied in bands sized by ``--chunk-mib`` rather than
mapping the whole matrix, so preparing a 32 GB input does not need 32 GB of RAM.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

GENERATOR_VERSION = 1


def sidecar_path(store):
    return store.with_name(store.name + ".json")


def describe(store, rows, cols, row_chunk, col_chunk, dtype):
    return {"rows": rows, "cols": cols, "row_chunk": row_chunk, "col_chunk": col_chunk,
            "dtype": np.dtype(dtype).name, "compressor": "none",
            "layout": "zarr v3, C order, uncompressed; identical values to the raw FP64 input",
            "generator": "prepare_zarr.py", "generator_version": GENERATOR_VERSION}


def padding_percent(extent, chunk):
    """Percent of stored values that are trailing-chunk padding along one axis."""
    stored = -(-extent // chunk) * chunk
    return 100.0 * (stored - extent) / extent


def nearby_divisors(extent, chunk, count=3):
    """Exact chunk lengths closest to `chunk`, so the warning can suggest a fix."""
    divisors = [d for d in range(1, extent + 1) if extent % d == 0 and d <= extent] \
        if extent <= 10_000 else \
        sorted({d for d in range(max(1, chunk // 4), min(extent, chunk * 4) + 1)
                if extent % d == 0})
    return sorted(divisors, key=lambda d: abs(d - chunk))[:count] or [extent]


def store_problems(store, expected):
    """Return why an existing store cannot be reused, or an empty list."""
    if not store.exists():
        return [f"missing {store}"]
    try:
        actual = json.loads(sidecar_path(store).read_text())
    except (OSError, json.JSONDecodeError):
        return [f"missing or invalid {sidecar_path(store)}"]
    problems = [f"{key} is {actual.get(key)!r}, expected {value!r}"
                for key, value in expected.items() if actual.get(key) != value]
    if problems:
        return problems
    try:
        import zarr
        array = zarr.open_array(store=str(store), mode="r")
    except Exception as error:                      # noqa: BLE001 - report, do not crash
        return [f"cannot open {store}: {error}"]
    if tuple(array.shape) != (expected["rows"], expected["cols"]):
        return [f"{store}: shape is {tuple(array.shape)}"]
    if tuple(array.chunks) != (expected["row_chunk"], expected["col_chunk"]):
        return [f"{store}: chunks are {tuple(array.chunks)}"]
    return []


def convert(source, store, rows, cols, row_chunk, col_chunk, dtype, chunk_mib):
    import zarr

    itemsize = np.dtype(dtype).itemsize
    expected_bytes = rows * cols * itemsize
    actual_bytes = source.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"{source}: {actual_bytes} bytes, expected {expected_bytes} "
                         f"for {rows}x{cols} {np.dtype(dtype).name}")

    # Copy in whole row-chunk multiples so no chunk is written twice.
    band = max(row_chunk, (chunk_mib * 2**20 // (cols * itemsize) // row_chunk) * row_chunk)
    array = zarr.create_array(store=str(store), shape=(rows, cols),
                              chunks=(row_chunk, col_chunk), dtype=dtype,
                              compressors=None, overwrite=False)
    started = time.perf_counter()
    with source.open("rb") as handle:
        for first in range(0, rows, band):
            height = min(band, rows - first)
            values = np.fromfile(handle, dtype=dtype, count=height * cols)
            if values.size != height * cols:
                raise ValueError(f"short read from {source} at row {first}")
            array[first:first + height] = values.reshape(height, cols)
            done = (first + height) / rows
            print(f"  {done:6.1%}  {first + height}/{rows} rows "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)
    return array


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="headerless row-major FP64 input")
    parser.add_argument("store", type=Path, help="output .zarr directory")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--row-chunk", type=int, required=True)
    parser.add_argument("--col-chunk", type=int, default=0,
                        help="columns per chunk; 0 (default) spans the whole row")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--chunk-mib", type=int, default=256,
                        help="approximate RAM bound for one transfer band")
    parser.add_argument("--replace-invalid", action="store_true",
                        help="quarantine an incompatible store instead of failing")
    args = parser.parse_args()

    col_chunk = args.col_chunk or args.cols
    if min(args.rows, args.cols, args.row_chunk, args.chunk_mib) < 1 or col_chunk < 1:
        raise ValueError("rows, cols, row-chunk, col-chunk, and chunk-mib must be positive")
    if args.row_chunk > args.rows or col_chunk > args.cols:
        raise ValueError("row-chunk and col-chunk cannot exceed the matrix dimensions")
    if not args.source.is_file():
        raise RuntimeError(f"missing source matrix {args.source}")

    expected = describe(args.store, args.rows, args.cols, args.row_chunk, col_chunk, args.dtype)
    problems = store_problems(args.store, expected)
    if not problems:
        print(f"{args.store} is already valid; nothing to do.")
        return 0
    if args.store.exists():
        if not args.replace_invalid:
            raise RuntimeError(f"{args.store} exists but is not usable: {'; '.join(problems)}")
        quarantine = args.store.with_name(f"{args.store.name}.invalid-{int(time.time())}")
        print(f"Quarantining incompatible store as {quarantine}: {'; '.join(problems)}",
              file=sys.stderr)
        args.store.rename(quarantine)
        sidecar = sidecar_path(args.store)
        if sidecar.exists():
            sidecar.rename(sidecar_path(quarantine))

    args.store.parent.mkdir(parents=True, exist_ok=True)
    chunk_mib = args.row_chunk * col_chunk * np.dtype(args.dtype).itemsize / 2**20
    print(f"Writing {args.store} with {args.row_chunk}x{col_chunk} chunks "
          f"({chunk_mib:.1f} MiB each), uncompressed.")
    # Zarr writes every chunk at full size, so a trailing partial chunk is padded on
    # disk. The suite compares physical read volume, so an oversized store would
    # silently shift the Dask arm's read bars; a divisor of `rows` costs nothing.
    padding = padding_percent(args.rows, args.row_chunk) + padding_percent(args.cols, col_chunk)
    if padding:
        print(f"WARNING: {args.row_chunk}x{col_chunk} does not divide {args.rows}x{args.cols}; "
              f"padded chunks inflate the store by about {padding:.2f}%. "
              f"Nearby exact row chunks: {', '.join(str(v) for v in nearby_divisors(args.rows, args.row_chunk))}.",
              file=sys.stderr)
    try:
        convert(args.source, args.store, args.rows, args.cols, args.row_chunk, col_chunk,
                args.dtype, args.chunk_mib)
    except BaseException:
        # A partial store is worse than none: it would satisfy an existence check.
        shutil.rmtree(args.store, ignore_errors=True)
        raise
    sidecar_path(args.store).write_text(json.dumps(expected, indent=2) + "\n")

    remaining = store_problems(args.store, expected)
    if remaining:
        raise RuntimeError(f"{args.store} failed validation after writing: {'; '.join(remaining)}")
    print(f"Wrote and validated {args.store}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
