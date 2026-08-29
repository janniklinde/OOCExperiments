#!/usr/bin/env python3
"""Convert a canonical CSR graph to SystemDS binary blocks in bounded chunks.

The sparse counterpart to `prepare_dense_systemds.py`, and it reuses that module's staging
manifest, validation, quarantine, and native OOC `rbind` assembly. Only the transfer differs:
each row band is handed to the Python binding as a `scipy.sparse.csr_matrix` so the edges
cross into the JVM in their sparse form. Materializing a band densely is not an option --
one 22,000-row band of a 17-million-vertex graph would be three terabytes.
"""

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)
sys.path.insert(0, script_dir)

import numpy as np

from prepare_dense_systemds import (assemble, native_state, quarantine_native, read_json,
                                    write_manifest)


def csr_identity(directory, vertices, slots):
    files = {}
    for name in ("row_ptr.i64", "col_idx.i32", "values.f64"):
        stat = (directory / name).stat()
        files[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {"path": str(directory.resolve()), "vertices": vertices, "edge_slots": slots,
            "files": files, "layout": "csr"}


def aligned_band_rows(degree, blocksize, chunk_bytes):
    """Rows per transferred band, rounded down to a whole number of SystemDS blocks."""
    raw_rows = max(1, chunk_bytes // max(1, degree * 12))
    if raw_rows < blocksize:
        return blocksize
    return (raw_rows // blocksize) * blocksize


def read_band(directory, vertices, first, last):
    """One row band of the CSR graph as a scipy CSR matrix, read directly from disk."""
    from scipy.sparse import csr_matrix

    pointer = np.fromfile(directory / "row_ptr.i64", dtype="<i8",
                          count=last - first + 1, offset=first * 8)
    start, stop = int(pointer[0]), int(pointer[-1])
    columns = np.fromfile(directory / "col_idx.i32", dtype="<i4",
                          count=stop - start, offset=start * 4)
    values = np.fromfile(directory / "values.f64", dtype="<f8",
                         count=stop - start, offset=start * 8)
    if columns.size != stop - start or values.size != stop - start:
        raise RuntimeError(f"short read from {directory} for rows {first}:{last}")
    # A band's own pointer array must start at zero, and stays int32 because a band holds
    # far fewer than two billion edges -- which also keeps scipy from unifying the column
    # indices to int64 the way a whole-graph matrix would.
    local = (pointer - pointer[0]).astype(np.int32)
    return csr_matrix((values, columns, local), shape=(last - first, vertices))


def stage_bands(directory, vertices, slots, degree, blocksize, chunk_bytes, staging):
    band_rows = aligned_band_rows(degree, blocksize, chunk_bytes)
    count = (vertices + band_rows - 1) // band_rows
    manifest_path = staging / "manifest.json"
    manifest = {
        "version": 1, "source": csr_identity(directory, vertices, slots),
        "blocksize_alignment": blocksize, "chunk_bytes_requested": chunk_bytes,
        "chunk_rows": band_rows, "chunks": count,
    }
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing != manifest:
            raise RuntimeError(
                f"staging data at {staging} belongs to a different source or chunk geometry; "
                "move it aside and retry")
    else:
        staging.mkdir(parents=True, exist_ok=True)
        write_manifest(manifest_path, manifest)

    chunks, missing = [], []
    for index, first in enumerate(range(0, vertices, band_rows)):
        rows = min(band_rows, vertices - first)
        chunk = staging / f"part-{index:05d}"
        completion = staging / f"part-{index:05d}.complete.json"
        record = {"source": manifest["source"], "start_row": first, "rows": rows,
                  "cols": vertices}
        state = native_state(chunk, rows, vertices)
        if state == "valid" and read_json(completion) == record:
            chunks.append(chunk)
        elif state == "missing" and not completion.exists():
            chunks.append(chunk)
            missing.append((index, first, rows, chunk, completion, record))
        else:
            quarantine_native(chunk)
            completion.unlink(missing_ok=True)
            chunks.append(chunk)
            missing.append((index, first, rows, chunk, completion, record))

    if not missing:
        print(f"Reusing {count} validated staged band(s).", flush=True)
        return chunks

    print(f"Staging {count} band(s) of at most {band_rows} rows.", flush=True)
    from systemds.context import SystemDSContext

    for position, (index, first, rows, chunk, completion, record) in enumerate(missing, 1):
        band = read_band(directory, vertices, first, first + rows)
        print(f"  transfer {position}/{len(missing)}: rows {first}:{first + rows} "
              f"({band.nnz} edges)", flush=True)
        # data_transfer_mode=0 forces the py4j transfer path. The default pipe transfer has
        # no sparse implementation: it falls through and hands the JVM a null MatrixBlock,
        # which surfaces only as a NullPointerException inside setMatrix.
        with SystemDSContext(data_transfer_mode=0) as sds:
            operation = sds.from_py(band).write(str(chunk), format="binary")
            operation.compute()
        del operation, band
        gc.collect()
        if native_state(chunk, rows, vertices) != "valid":
            raise RuntimeError(f"SystemDS wrote an invalid staged band: {chunk}")
        write_manifest(completion, record)
    return chunks


def convert(data, output, blocksize, chunk_mib, java, jar, config, java_heap, java_tmp,
            replace_invalid, keep_staging, staging):
    data, output = Path(data), Path(output)
    metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    vertices, slots = metadata["vertices"], metadata["edge_slots"]
    degree = max(metadata["degrees"])
    directory = data / "csr"
    expected = {"row_ptr.i64": (vertices + 1) * 8, "col_idx.i32": slots * 4,
                "values.f64": slots * 8}
    for name, size in expected.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"missing or incompatible CSR component {path}; "
                               f"expected {size} bytes")

    state = native_state(output, vertices, vertices, blocksize)
    if state == "valid":
        print(f"Keeping valid output: {output}", flush=True)
        return
    if state in ("incomplete", "incompatible"):
        if not replace_invalid:
            raise RuntimeError(f"requested output is {state}: {output}; "
                               "retry with --replace-invalid to quarantine it")
        quarantine_native(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if staging is None:
        staging = output.parent / "systemds-staging" / f"{output.name}-bs{blocksize}"
    staging = Path(staging)
    chunks = stage_bands(directory, vertices, slots, degree, blocksize,
                         chunk_mib << 20, staging)
    assemble(chunks, output, vertices, vertices, blocksize, java, jar, config,
             java_heap, java_tmp)
    if not keep_staging:
        shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="dataset directory holding csr/ and metadata.json")
    parser.add_argument("output", type=Path)
    parser.add_argument("--blocksize", type=int, required=True)
    parser.add_argument("--chunk-mib", type=int, default=128)
    parser.add_argument("--java", default="java")
    parser.add_argument("--systemds-jar", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--java-heap", default="3g")
    parser.add_argument("--java-tmp", type=Path)
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--replace-invalid", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()
    if min(args.blocksize, args.chunk_mib) < 1:
        raise ValueError("blocksize and chunk-mib must be positive")
    if not args.systemds_jar.is_file():
        raise RuntimeError(f"SystemDS JAR does not exist: {args.systemds_jar}")
    if not args.config.is_file():
        raise RuntimeError(f"SystemDS config does not exist: {args.config}")
    convert(args.data, args.output, args.blocksize, args.chunk_mib, args.java,
            args.systemds_jar, args.config, args.java_heap, args.java_tmp,
            args.replace_invalid, args.keep_staging, args.staging)


if __name__ == "__main__":
    main()
