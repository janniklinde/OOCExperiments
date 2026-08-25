#!/usr/bin/env python3
"""Convert canonical row-major FP64 files to SystemDS binary blocks in bounded chunks.

Each raw input is transferred through the Python binding as a sequence of small,
row-aligned temporary matrices. A native SystemDS OOC program then rbinds those
matrices and writes one matrix with the requested final block size.
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

import numpy as np


def native_state(path, rows, cols, blocksize=None):
    metadata_path = Path(str(path) + ".mtd")
    if not path.exists() and not metadata_path.exists():
        return "missing"
    if not path.exists() or not metadata_path.exists():
        return "incomplete"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "incomplete"
    expected = {"rows": rows, "cols": cols}
    if blocksize is not None:
        expected.update({"rows_in_block": blocksize, "cols_in_block": blocksize})
    try:
        matches = all(int(metadata.get(key, -1)) == value for key, value in expected.items())
    except (TypeError, ValueError):
        matches = False
    return "valid" if matches else "incompatible"


def raw_identity(path, rows, cols, dtype):
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "rows": rows,
        "cols": cols,
        "dtype": np.dtype(dtype).name,
        "layout": "row-major",
    }


def dml_string(path):
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def quarantine_native(path):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = path.with_name(path.name + f".invalid-{stamp}")
    suffix = 1
    while quarantine.exists() or Path(str(quarantine) + ".mtd").exists():
        quarantine = path.with_name(path.name + f".invalid-{stamp}-{suffix}")
        suffix += 1
    moved = []
    if path.exists():
        path.rename(quarantine)
        moved.append((path, quarantine))
    metadata = Path(str(path) + ".mtd")
    if metadata.exists():
        destination = Path(str(quarantine) + ".mtd")
        metadata.rename(destination)
        moved.append((metadata, destination))
    checksum = path.parent / ("." + path.name + ".mtd.crc")
    if checksum.exists():
        destination = quarantine.parent / ("." + quarantine.name + ".mtd.crc")
        checksum.rename(destination)
        moved.append((checksum, destination))
    if moved:
        print("Quarantined incompatible output as " + str(quarantine), flush=True)


def aligned_chunk_rows(cols, blocksize, chunk_bytes, dtype=np.float64):
    raw_rows = max(1, chunk_bytes // (cols * np.dtype(dtype).itemsize))
    if raw_rows < blocksize:
        return blocksize
    return (raw_rows // blocksize) * blocksize


def write_manifest(path, manifest):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def stage_chunks(raw_path, rows, cols, blocksize, chunk_bytes, staging, dtype=np.float64):
    dtype = np.dtype(dtype)
    chunk_rows = aligned_chunk_rows(cols, blocksize, chunk_bytes, dtype)
    count = (rows + chunk_rows - 1) // chunk_rows
    manifest_path = staging / "manifest.json"
    manifest = {
        "version": 1,
        "source": raw_identity(raw_path, rows, cols, dtype),
        "blocksize_alignment": blocksize,
        "chunk_bytes_requested": chunk_bytes,
        "chunk_rows": chunk_rows,
        "chunks": count,
    }
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid staging manifest {manifest_path}") from error
        if existing != manifest:
            raise RuntimeError(
                f"staging data at {staging} belongs to a different source or chunk geometry; "
                "move it aside and retry")
    else:
        staging.mkdir(parents=True, exist_ok=True)
        write_manifest(manifest_path, manifest)

    chunks = []
    missing = []
    for index, start in enumerate(range(0, rows, chunk_rows)):
        current_rows = min(chunk_rows, rows - start)
        chunk = staging / f"part-{index:05d}"
        completion = staging / f"part-{index:05d}.complete.json"
        completion_record = {
            "source": manifest["source"], "start_row": start,
            "rows": current_rows, "cols": cols,
        }
        state = native_state(chunk, current_rows, cols)
        if state == "valid" and read_json(completion) == completion_record:
            chunks.append(chunk)
        elif state == "missing" and not completion.exists():
            chunks.append(chunk)
            missing.append((index, start, current_rows, chunk, completion, completion_record))
        else:
            quarantine_native(chunk)
            completion.unlink(missing_ok=True)
            chunks.append(chunk)
            missing.append((index, start, current_rows, chunk, completion, completion_record))

    if missing:
        actual_mib = chunk_rows * cols * dtype.itemsize / (1 << 20)
        print(f"Staging {raw_path.name} as {count} chunk(s) of at most {chunk_rows} rows "
              f"({actual_mib:.1f} MiB aligned; {chunk_bytes / (1 << 20):.0f} MiB requested).",
              flush=True)
        from systemds.context import SystemDSContext

        mapped = np.memmap(raw_path, dtype=dtype, mode="r", shape=(rows, cols), order="C")
        for position, (index, start, current_rows, chunk, completion,
                       completion_record) in enumerate(missing, 1):
            end = start + current_rows
            mib = current_rows * cols * dtype.itemsize / (1 << 20)
            print(f"  transfer {position}/{len(missing)}: rows {start}:{end} ({mib:.1f} MiB)",
                  flush=True)
            # A fresh context per chunk guarantees that Java-side transfer state is
            # released before the next chunk, at the cost of preparation startup time.
            with SystemDSContext() as sds:
                operation = sds.from_py(np.asarray(mapped[start:end]))
                operation.write(str(chunk), format="binary").compute()
            del operation
            gc.collect()
            if native_state(chunk, current_rows, cols) != "valid":
                raise RuntimeError(f"SystemDS wrote an invalid staged chunk: {chunk}")
            write_manifest(completion, completion_record)
    else:
        print(f"Reusing {count} validated staged chunk(s) for {raw_path.name}.", flush=True)
    return chunks


def assemble(chunks, output, rows, cols, blocksize, java, jar, config, heap, java_tmp):
    assembly = output.with_name(output.name + ".assembling")
    completion = Path(str(assembly) + ".complete.json")
    completion_record = {
        "rows": rows, "cols": cols, "blocksize": blocksize,
        "chunks": [str(chunk.resolve()) for chunk in chunks],
    }
    state = native_state(assembly, rows, cols, blocksize)
    if state == "valid" and read_json(completion) == completion_record:
        print(f"Reusing validated assembled matrix {assembly}.", flush=True)
    else:
        if state != "missing":
            quarantine_native(assembly)
        completion.unlink(missing_ok=True)
        reads = ",\n    ".join(f"read({dml_string(chunk)})" for chunk in chunks)
        expression = reads if len(chunks) == 1 else f"rbind(\n    {reads}\n)"
        program = (
            f"X = {expression};\n"
            f"write(X, {dml_string(assembly)}, format=\"binary\", "
            f"rows_in_block={blocksize}, cols_in_block={blocksize});\n"
        )
        dml_path = chunks[0].parent / "assemble.dml"
        dml_path.write_text(program, encoding="utf-8")
        command = [java, f"-Xmx{heap}", "-XX:+UseG1GC", "--add-modules=jdk.incubator.vector"]
        if java_tmp:
            command.append(f"-Djava.io.tmpdir={java_tmp}")
        command.extend(["-jar", str(jar), "-f", str(dml_path), "-exec", "singlenode",
                        "-config", str(config), "-ooc", "-oocStats", "-stats"])
        print(f"Assembling {len(chunks)} chunk(s) with native SystemDS OOC rbind...", flush=True)
        subprocess.run(command, check=True)
        if native_state(assembly, rows, cols, blocksize) != "valid":
            raise RuntimeError(f"SystemDS produced an invalid assembled matrix: {assembly}")
        write_manifest(completion, completion_record)

    if output.exists() or Path(str(output) + ".mtd").exists():
        raise RuntimeError(f"refusing to publish over existing output: {output}")
    assembly.rename(output)
    Path(str(assembly) + ".mtd").rename(Path(str(output) + ".mtd"))
    checksum = assembly.parent / ("." + assembly.name + ".mtd.crc")
    if checksum.exists():
        checksum.rename(output.parent / ("." + output.name + ".mtd.crc"))
    completion.unlink()
    if native_state(output, rows, cols, blocksize) != "valid":
        raise RuntimeError(f"published output failed validation: {output}")


def convert_fp64(raw, output, rows, cols, blocksize, chunk_mib, java, jar, config,
                  java_heap="3g", java_tmp=None, replace_invalid=False, keep_staging=False,
                  staging=None, dtype=np.float64):
    """Convert one raw row-major numeric matrix to one native SystemDS matrix."""
    raw = Path(raw)
    output = Path(output)
    dtype = np.dtype(dtype)
    expected_size = rows * cols * dtype.itemsize
    if not raw.is_file() or raw.stat().st_size != expected_size:
        raise RuntimeError(f"missing or incompatible canonical raw input {raw}; "
                           f"expected {expected_size} bytes")
    state = native_state(output, rows, cols, blocksize)
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
    chunks = stage_chunks(raw, rows, cols, blocksize, chunk_mib << 20, staging, dtype)
    assemble(chunks, output, rows, cols, blocksize, java, jar, config, java_heap, java_tmp)
    if not keep_staging:
        shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="directory containing canonical raw FP64 files")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--blocksize", type=int, required=True)
    parser.add_argument("--chunk-mib", type=int, default=128,
                        help="maximum raw transfer chunk in MiB (default: 128)")
    parser.add_argument("--java", default="java")
    parser.add_argument("--systemds-jar", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--java-heap", default="3g")
    parser.add_argument("--java-tmp", type=Path)
    parser.add_argument("--replace-invalid", action="store_true",
                        help="quarantine incomplete/incompatible requested outputs")
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--x-output", type=Path, required=True)
    parser.add_argument("--binary-y-output", type=Path, required=True)
    parser.add_argument("--nn-y-output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.cols, args.blocksize, args.chunk_mib) < 1:
        raise ValueError("rows, cols, blocksize, and chunk-mib must be positive")
    if not args.systemds_jar.is_file():
        raise RuntimeError(f"SystemDS JAR does not exist: {args.systemds_jar}")
    if not args.config.is_file():
        raise RuntimeError(f"SystemDS config does not exist: {args.config}")

    matrices = {
        "X": (args.data / "X.f64", args.x_output, args.rows, args.cols),
        "binary_y": (args.data / "binary_y.f64", args.binary_y_output, args.rows, 1),
        "nn_y": (args.data / "nn_y.f64", args.nn_y_output, args.rows, 1),
    }
    for raw, _, rows, cols in matrices.values():
        expected_size = rows * cols * np.dtype("<f8").itemsize
        if not raw.is_file() or raw.stat().st_size != expected_size:
            raise RuntimeError(f"missing or incompatible canonical raw input {raw}; "
                               f"expected {expected_size} bytes")
    for name, (raw, output, rows, cols) in matrices.items():
        staging = args.data / "systemds-staging" / f"{name}-bs{args.blocksize}"
        convert_fp64(raw, output, rows, cols, args.blocksize, args.chunk_mib, args.java,
                     args.systemds_jar, args.config, args.java_heap, args.java_tmp,
                     args.replace_invalid, args.keep_staging, staging)
        print(f"Prepared {name} at blocksize {args.blocksize}: {output}", flush=True)


if __name__ == "__main__":
    main()
