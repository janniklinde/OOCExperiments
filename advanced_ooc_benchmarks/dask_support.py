#!/usr/bin/env python3
"""Shared single-process Dask scheduler configuration for benchmark baselines."""

from pathlib import Path

import dask
import dask.array as da
import numpy as np
from dask.base import tokenize
from dask.delayed import delayed
from dask.utils import parse_bytes
from distributed import Client



def _read_block(path, dtype, cols, first_row, rows):
    """Read one row block of a row-major raw file into a fresh array."""
    itemsize = np.dtype(dtype).itemsize
    values = np.fromfile(path, dtype=dtype, count=rows * cols,
                         offset=first_row * cols * itemsize)
    if values.size != rows * cols:
        raise ValueError(f"short read from {path}: wanted {rows * cols}, got {values.size}")
    return values.reshape(rows, cols)


def read_rows(path, shape, first_row, rows, dtype=np.float64):
    """Eagerly read a small row range into memory (initial centers, label scans)."""
    return _read_block(str(path), dtype, shape[1], first_row, rows)


def default_row_chunk(cols, dtype=np.float64):
    """Rows per chunk that keep a whole-row block near the configured chunk size."""
    limit = parse_bytes(dask.config.get("array.chunk-size"))
    return max(1, int(limit // max(1, cols * np.dtype(dtype).itemsize)))


def load_matrix(path, shape, dtype=np.float64, row_chunk=None, row_range=None):
    """Load a row-major raw file as a Dask array of managed, spillable blocks.

    np.memmap + da.from_array looks lazy but hands Dask file-backed pages: they land in the
    worker's RSS as *unmanaged* memory, which the memory monitor counts toward its pause
    threshold and the spill machinery cannot evict. A worker then pauses at the threshold,
    spills nothing, and never resumes. Reading each block with np.fromfile instead produces
    ordinary arrays that Dask owns, tracks, and can spill to local_directory.

    Blocks span whole rows: the file is row-major, so a row block is one contiguous read,
    and every algorithm here partitions over rows anyway.
    """
    rows, cols = shape
    first_row, last_row = (0, rows) if row_range is None else row_range
    if not (0 <= first_row <= last_row <= rows):
        raise ValueError(f"row_range {row_range} outside 0..{rows}")
    if row_chunk is None:
        row_chunk = default_row_chunk(cols, dtype)

    path = str(path)
    name = f"read-{tokenize(path, shape, dtype, row_chunk, row_range)}"
    blocks = []
    for start in range(first_row, last_row, row_chunk):
        height = min(row_chunk, last_row - start)
        block = delayed(_read_block, name=f"{name}-{start}", pure=True)(
            path, dtype, cols, start, height)
        blocks.append(da.from_delayed(block, shape=(height, cols), dtype=dtype))
    if not blocks:
        return da.zeros((0, cols), dtype=dtype, chunks=(1, cols))
    return da.concatenate(blocks, axis=0)


def create_client(threads, memory_limit, temporary_directory):
    """Create one spill-capable in-process worker with bounded concurrency."""
    temporary_directory = Path(temporary_directory)
    temporary_directory.mkdir(parents=True, exist_ok=True)
    dask.config.set({
        # Repeated fixed-iteration submissions reuse the same array keys. Keep
        # their task specifications stable across submissions; otherwise newer
        # distributed schedulers can see a fused Alias first and a getter task
        # later for the same key.
        "optimization.fuse.active": False,
        "distributed.worker.memory.target": 0.60,
        "distributed.worker.memory.spill": 0.70,
        "distributed.worker.memory.pause": 0.82,
        "distributed.worker.memory.terminate": 0.95,
    })
    return Client(
        processes=False,
        n_workers=1,
        threads_per_worker=threads,
        memory_limit=memory_limit,
        local_directory=str(temporary_directory),
        dashboard_address=None,
    )
