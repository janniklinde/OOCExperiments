#!/usr/bin/env python3
"""Build a bounded-memory induced Twitter window for comparable PageRank runs.

The input CSR stores the column-stochastic transition matrix as destination
rows and source columns. Dropping vertices changes outgoing degrees, so the
retained columns are renormalized and dangling vertices are recomputed.
"""

import argparse
import json
import sys
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

import numpy as np


def prepare(source: Path, output: Path, start_vertex: int, vertices: int, chunk_edges: int) -> None:
    source_meta = json.loads((source / "metadata.json").read_text())
    source_vertices = int(source_meta["vertices"])
    source_edges = int(source_meta["edges"])
    end_vertex = start_vertex + vertices
    if start_vertex < 0 or vertices < 1 or end_vertex > source_vertices:
        raise ValueError(f"vertex interval must lie in [0, {source_vertices})")
    if chunk_edges < 1:
        raise ValueError("chunk_edges must be positive")
    if output.resolve() == source.resolve():
        raise ValueError("output must differ from source")

    csr = source / "csr"
    row_ptr = np.memmap(csr / "row_ptr.i64", dtype="<i8", mode="r", shape=(source_vertices + 1,))
    columns = np.memmap(csr / "col_idx.i32", dtype="<i4", mode="r", shape=(source_edges,))
    values = np.memmap(csr / "values.f64", dtype="<f8", mode="r", shape=(source_edges,))
    input_first = int(row_ptr[start_vertex])
    input_limit = int(row_ptr[end_vertex])
    edge_count = sum(
        int(np.count_nonzero((columns[first:min(first + chunk_edges, input_limit)] >= start_vertex)
                             & (columns[first:min(first + chunk_edges, input_limit)] < end_vertex)))
        for first in range(input_first, input_limit, chunk_edges)
    )
    print(f"induced graph: {vertices:,} vertices, {edge_count:,} edges", flush=True)

    out_csr = output / "csr"
    if (output / "metadata.json").exists() or any(
        (out_csr / name).exists() for name in ("row_ptr.i64", "col_idx.i32", "values.f64")
    ):
        raise FileExistsError(f"refusing to replace existing dataset at {output}")
    out_csr.mkdir(parents=True, exist_ok=True)
    out_ptr_path = out_csr / "row_ptr.i64.partial"
    out_col_path = out_csr / "col_idx.i32.partial"
    out_val_path = out_csr / "values.f64.partial"
    out_dangling_path = output / "dangling.u8.partial"
    out_ptr = np.memmap(out_ptr_path, dtype="<i8", mode="w+", shape=(vertices + 1,))
    out_col = np.memmap(out_col_path, dtype="<i4", mode="w+", shape=(edge_count,))
    out_val = np.memmap(out_val_path, dtype="<f8", mode="w+", shape=(edge_count,))
    column_sums = np.zeros(vertices, dtype=np.float64)

    row = 0
    written = 0
    next_progress = 1_000_000
    while row < vertices:
        first = int(row_ptr[start_vertex + row])
        last_row = min(vertices, max(row + 1, int(np.searchsorted(
            row_ptr, first + chunk_edges, side="right")) - 1 - start_vertex))
        last = int(row_ptr[start_vertex + last_row])
        selected = (columns[first:last] >= start_vertex) & (columns[first:last] < end_vertex)
        count = int(np.count_nonzero(selected))
        if count:
            selected_col = np.asarray(columns[first:last][selected]) - start_vertex
            selected_val = np.asarray(values[first:last][selected])
            out_col[written:written + count] = selected_col
            out_val[written:written + count] = selected_val
            column_sums += np.bincount(selected_col, weights=selected_val, minlength=vertices)
        prefix = np.empty(len(selected) + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(selected, dtype=np.int64, out=prefix[1:])
        out_ptr[row:last_row + 1] = written + prefix[
            row_ptr[start_vertex + row:start_vertex + last_row + 1] - first]
        written += count
        row = last_row
        if row >= next_progress:
            print(f"processed rows through {row:,}/{vertices:,}", flush=True)
            next_progress = row + 1_000_000
    if written != edge_count or int(out_ptr[-1]) != edge_count:
        raise RuntimeError(f"edge count mismatch: expected {edge_count}, wrote {written}")

    for first in range(0, edge_count, chunk_edges):
        last = min(first + chunk_edges, edge_count)
        out_val[first:last] /= column_sums[out_col[first:last]]
    dangling = (column_sums == 0).astype(np.uint8)
    dangling.tofile(out_dangling_path)
    out_ptr.flush()
    out_col.flush()
    out_val.flush()
    del out_ptr, out_col, out_val

    out_ptr_path.replace(out_csr / "row_ptr.i64")
    out_col_path.replace(out_csr / "col_idx.i32")
    out_val_path.replace(out_csr / "values.f64")
    out_dangling_path.replace(output / "dangling.u8")
    metadata = {
        "name": "twitter-2010-induced-window",
        "vertices": vertices,
        "edges": edge_count,
        "input_edges": input_limit - input_first,
        "dangling_vertices": int(dangling.sum()),
        "orientation": source_meta["orientation"],
        "normalization": "retained source columns renormalized to one; empty columns marked dangling",
        "source": str(source.resolve()),
        "source_vertices": source_vertices,
        "source_vertex_start": start_vertex,
        "source_vertex_end": end_vertex,
        "dtype": {"row_ptr": "int64", "col_idx": "int32", "values": "float64"},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"prepared {output}: {edge_count:,} edges, {metadata['dangling_vertices']:,} dangling", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=int, default=0, help="first published vertex ID, zero-based")
    parser.add_argument("--vertices", type=int, required=True)
    parser.add_argument("--chunk-edges", type=int, default=8_000_000)
    args = parser.parse_args()
    prepare(args.source, args.output, args.start, args.vertices, args.chunk_edges)


if __name__ == "__main__":
    main()
