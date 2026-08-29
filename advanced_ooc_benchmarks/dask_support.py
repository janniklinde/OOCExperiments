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


DEFAULT_ZARR_STORE = Path("zarr") / "X.zarr"


def resolve_zarr(data, override=None):
    """The prepared Zarr store for a dataset directory.

    One fixed location per dataset rather than a chunk-qualified name: the chunking lives
    in the sidecar, so changing `dask_chunk` makes the plan's preflight see a mismatch and
    rebuild in place instead of accumulating stores nobody selected.
    """
    store = Path(override) if override else Path(data) / DEFAULT_ZARR_STORE
    if not store.exists():
        raise SystemExit(
            f"missing prepared Zarr store {store}\n"
            f"The benchmark plan prepares it automatically; to build one by hand run:\n"
            f"  prepare_zarr.py {Path(data) / 'X.f64'} {store} "
            f"--rows <rows> --cols <cols> --row-chunk <rows-per-chunk>")
    return store


def load_zarr(path):
    """Load the prepared uncompressed Zarr store as a Dask array.

    This is the idiomatic Dask out-of-core input and the one the benchmark uses for the
    large matrix. It matters beyond style: `from_delayed` + `concatenate` puts an *alias*
    at the top of the array, whose task specification is literally the name of another
    key. Low-level fusion renames that key differently depending on the downstream graph,
    so reusing one such array across several `compute()` calls makes the scheduler see
    two different specifications for one stable key and warn about possible deadlocks.
    `from_zarr` emits a real getter task per chunk instead, with no alias to invalidate,
    so fusion can stay enabled.
    """
    import dask.array as _da  # local: only the zarr arm needs the dependency

    return _da.from_zarr(str(path))


def _read_csr_band(block, directory=None, vertices=0, band_rows=0):
    """Build one row band of a CSR graph as a `sparse.COO` chunk.

    The band's own slice of `row_ptr` is read inside the task rather than passed in: it is
    a few hundred kilobytes, while the whole pointer array would be embedded in every task
    specification in the graph.
    """
    import sparse

    band = int(np.asarray(block).ravel()[0])
    first = band * band_rows
    last = min(first + band_rows, vertices)
    pointer = np.fromfile(f"{directory}/row_ptr.i64", dtype="<i8",
                          count=last - first + 1, offset=first * 8)
    start, stop = int(pointer[0]), int(pointer[-1])
    columns = np.fromfile(f"{directory}/col_idx.i32", dtype="<i4",
                          count=stop - start, offset=start * 4)
    values = np.fromfile(f"{directory}/values.f64", dtype="<f8",
                         count=stop - start, offset=start * 8)
    if columns.size != stop - start or values.size != stop - start:
        raise ValueError(f"short read from {directory} for rows {first}:{last}")
    rows = np.repeat(np.arange(last - first, dtype=np.int64), np.diff(pointer))
    return sparse.COO(np.stack([rows, columns.astype(np.int64)]), values,
                      shape=(last - first, vertices), has_duplicates=False, sorted=True)


def load_csr_coo(directory, vertices, band_rows):
    """A lazy Dask array of `sparse.COO` row bands over raw CSR files.

    Built by mapping over a tiny index array rather than with `from_delayed` +
    `concatenate`. That construction puts an *alias* at the top of the array whose task
    specification is the name of another key; low-level fusion renames it differently
    depending on the downstream graph, so reusing one such array across several
    `compute()` calls -- which is exactly what a fixed-point iteration does -- makes the
    scheduler see two specifications for one stable key and warn about possible deadlocks.
    `map_blocks` emits ordinary Blockwise tasks with no alias, so fusion can stay enabled.
    """
    import sparse

    directory = Path(directory)
    for name in ("row_ptr.i64", "col_idx.i32", "values.f64"):
        if not (directory / name).is_file():
            raise SystemExit(f"missing CSR component {directory / name}")
    bands = -(-vertices // band_rows)
    heights = tuple(min(band_rows, vertices - band * band_rows) for band in range(bands))
    meta = sparse.COO(np.zeros((2, 0), dtype=np.int64), np.zeros(0), shape=(0, 0))
    return da.arange(bands, chunks=1).map_blocks(
        _read_csr_band, directory=str(directory), vertices=vertices, band_rows=band_rows,
        chunks=(heights, (vertices,)), new_axis=1, dtype=np.float64, meta=meta)


def read_vector(path, rows, dtype=np.float64):
    """Read a response/label column eagerly as NumPy.

    These are one column wide, so chunking them to match X would produce one tiny task
    per row block of the large matrix -- thousands of keys carrying kilobytes each -- for
    an array that is a rounding error beside X (32 MB at four million rows). Reading it
    once keeps the graph to the operand that is actually out of core.
    """
    values = np.fromfile(str(path), dtype=dtype, count=rows)
    if values.size != rows:
        raise ValueError(f"short read from {path}: wanted {rows}, got {values.size}")
    return values.reshape(rows, 1)


def store_tall(array, path, compute_options=None):
    """Materialize a tall Dask result to an uncompressed Zarr store, streaming.

    `da.store` into a `np.memmap` looks like it works and does not: under a distributed
    client the memmap is serialized to the worker, which writes into its own copy, so the
    file stays zero-filled while the returned checksum is still correct. A Zarr target is
    opened inside the worker instead, so the chunks actually land on disk.

    Returns the stored array so a checksum can share the same single pass.
    """
    import dask.array as _da
    import zarr

    chunks = tuple(dimension[0] for dimension in array.chunks)
    target = zarr.create_array(store=str(path), shape=array.shape, chunks=chunks,
                               dtype=array.dtype, compressors=None, overwrite=True)
    return _da.store(array, target, lock=False, compute=False, return_stored=True)


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
        # Low-level fusion stays enabled. The alias layer that made it unsafe came from
        # `load_matrix`'s from_delayed/concatenate construction; `load_zarr` has no alias,
        # so repeated fixed-iteration submissions no longer redefine a stable key. Keep
        # `optimization.fuse.active: False` if a workload falls back to `load_matrix`.
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
