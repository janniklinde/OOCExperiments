#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0.

"""Prepare a deterministic sparse symmetric graph with planted connected components.

The graph is stored as CSR over raw files, the representation `pagerank/` already uses:
`csr/row_ptr.i64`, `csr/col_idx.i32`, `csr/values.f64`. That is 12 bytes per stored edge,
which is also what a SystemDS `SparseBlockCSR` costs (a 4-byte column index and an 8-byte
value), so the two arms read comparable volume rather than being separated by their
representations. The values are all 1.0 and carry no information; they are written and read
anyway, because SystemDS has no unweighted matrix type and must read them, and omitting them
here would cut the Dask arm's read volume by two thirds for free.

Two structural properties are load-bearing:

* **Symmetric by construction, so generation never sorts.** Within a component the vertices
  are placed on a ring by their rank, and vertex `r` is adjacent to `r ± o (mod m)` for a
  fixed offset set. Because the offset set is applied in both directions the edge relation is
  symmetric without a transpose-and-merge pass, which at this scale would be an external sort
  of billions of pairs. Every row is computed independently, so the writer stays streaming.
* **Offsets are a short band plus powers of two.** A band alone gives a ring of diameter
  `m/(2*band)` -- hundreds of thousands of propagation passes. Powers of two alone give
  diameter about `log2(m)`, since any offset is reachable through its binary representation.
  Together the diameter is roughly `log2(m) - log2(band)`, which is what sets the iteration
  count: `band` therefore tunes degree (hence dataset size) and the power ladder tunes depth.

Component membership and within-component rank are both scattered across vertex ids by one
seeded permutation. Without it the ring would sit in a band around the diagonal and both the
SystemDS block structure and the Dask row chunks would see wildly uneven nonzero counts.

Ground truth is exact: SystemDS labels a component by its maximum vertex id, so the generator
records `expected_components` and `expected_label_sum` and every arm checks against them.
"""

import argparse
import json
import os
import sys
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)

import numpy as np

GENERATOR_VERSION = 1
LAYOUT = ("CSR over raw files: row_ptr int64, col_idx int32, values float64; symmetric "
          "0/1 adjacency with a zero diagonal and sorted column indices per row")


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def offsets_for(size, band):
    """Ring offsets for one component: 1..band, then powers of two, all below size/2.

    Staying strictly below size/2 keeps `+o` and `-o` distinct, so every vertex has exactly
    twice as many neighbours as there are offsets and the row lengths need no deduplication.
    """
    limit = size // 2
    chosen = [o for o in range(1, band + 1) if o < limit]
    power = 1
    while power < limit:
        if power > band:
            chosen.append(power)
        power *= 2
    return np.array(sorted(set(chosen)), dtype=np.int64)


def component_layout(vertices, components, minority_fraction, seed):
    """Component sizes plus the position->vertex permutation that scatters them."""
    if components < 1:
        raise ValueError("components must be positive")
    minority = int(round(vertices * minority_fraction))
    small = components - 1
    sizes = np.zeros(components, dtype=np.int64)
    if small:
        base, extra = divmod(minority, small)
        if base < 8:
            raise ValueError(f"minority-fraction {minority_fraction} leaves only {base} "
                             f"vertices for each of the {small} small components; a ring "
                             f"needs at least 8 to carry any offsets")
        sizes[1:] = base
        sizes[1:1 + extra] += 1
    sizes[0] = vertices - sizes[1:].sum()
    if sizes[0] < 8:
        raise ValueError("the dominant component is too small; lower --minority-fraction")

    order = np.random.default_rng(np.random.SeedSequence([seed, 1])).permutation(
        vertices).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    component = np.empty(vertices, dtype=np.int32)
    rank = np.empty(vertices, dtype=np.int64)
    component[order] = np.repeat(np.arange(components, dtype=np.int32), sizes)
    rank[order] = np.concatenate([np.arange(size, dtype=np.int64) for size in sizes])
    return order, starts, sizes, component, rank


def write_csr(directory, order, starts, sizes, component, rank, offsets, vertices, band_rows):
    """Stream CSR row bands; column indices are sorted within each row."""
    directory.mkdir(parents=True, exist_ok=True)
    index_tmp = directory / f".col_idx.i32.tmp-{os.getpid()}"
    value_tmp = directory / f".values.f64.tmp-{os.getpid()}"
    pointer_tmp = directory / f".row_ptr.i64.tmp-{os.getpid()}"
    lengths = np.empty(vertices, dtype=np.int64)
    total = 0
    try:
        with index_tmp.open("wb") as index_stream, value_tmp.open("wb") as value_stream:
            for start in range(0, vertices, band_rows):
                stop = min(start + band_rows, vertices)
                rows = np.arange(start, stop, dtype=np.int64)
                width = 2 * max(len(offsets[c]) for c in np.unique(component[start:stop]))
                neighbours = np.full((rows.size, width), -1, dtype=np.int64)
                for index in np.unique(component[start:stop]):
                    local = np.flatnonzero(component[start:stop] == index)
                    ring, size = offsets[index], sizes[index]
                    if ring.size == 0:
                        continue
                    positions = rank[rows[local]][:, None]
                    forward = (positions + ring) % size
                    backward = (positions - ring) % size
                    both = np.concatenate([forward, backward], axis=1)
                    neighbours[local, :both.shape[1]] = order[starts[index] + both]
                # -1 sorts to the front, so the padding of shorter rows is dropped by the
                # same mask that keeps the sorted column indices CSR wants.
                neighbours.sort(axis=1)
                keep = neighbours >= 0
                counts = keep.sum(axis=1)
                lengths[start:stop] = counts
                flat = neighbours[keep]
                total += flat.size
                flat.astype("<i4").tofile(index_stream)
                np.ones(flat.size, dtype="<f8").tofile(value_stream)
                print(f"generated rows {start}:{stop} / {vertices} ({total} edge slots)",
                      flush=True)
            for stream in (index_stream, value_stream):
                stream.flush()
                os.fsync(stream.fileno())
        pointer = np.zeros(vertices + 1, dtype="<i8")
        np.cumsum(lengths, out=pointer[1:])
        pointer.tofile(pointer_tmp)
        if pointer[-1] != total:
            raise RuntimeError("row pointer total disagrees with the written index count")
        if index_tmp.stat().st_size != total * 4 or value_tmp.stat().st_size != total * 8:
            raise RuntimeError("CSR generation produced an unexpected byte count")
        os.replace(index_tmp, directory / "col_idx.i32")
        os.replace(value_tmp, directory / "values.f64")
        os.replace(pointer_tmp, directory / "row_ptr.i64")
    except BaseException:
        for path in (index_tmp, value_tmp, pointer_tmp):
            path.unlink(missing_ok=True)
        raise
    return total


def prepare(out, vertices, components, band, minority_fraction, seed, chunk_mib):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    csr = out / "csr"
    metadata_path = out / "metadata.json"

    expected_metadata = {
        "vertices": vertices, "components": components, "band": band,
        "minority_fraction": minority_fraction, "seed": seed,
        "generator": "connected_components/prepare.py",
        "generator_version": GENERATOR_VERSION,
    }
    metadata = read_json(metadata_path)
    files = [csr / "row_ptr.i64", csr / "col_idx.i32", csr / "values.f64"]
    if all(path.is_file() for path in files) and isinstance(metadata, dict) and all(
            metadata.get(key) == value for key, value in expected_metadata.items()):
        expected_sizes = [(vertices + 1) * 8, metadata["edge_slots"] * 4,
                          metadata["edge_slots"] * 8]
        if all(path.stat().st_size == size for path, size in zip(files, expected_sizes)):
            print(f"Keeping complete generated graph in {out}.", flush=True)
            return
    if metadata_path.exists() and isinstance(metadata, dict) and not all(
            metadata.get(key) == value for key, value in expected_metadata.items()):
        raise RuntimeError(f"refusing to replace incompatible metadata {metadata_path}")

    order, starts, sizes, component, rank = component_layout(
        vertices, components, minority_fraction, seed)
    offsets = [offsets_for(int(size), band) for size in sizes]
    degrees = [2 * ring.size for ring in offsets]
    if degrees[0] == 0:
        raise ValueError("the dominant component carries no offsets; raise --vertices")

    # One band holds rows * 2 * |offsets| int64 candidates plus the same again while sorting.
    band_rows = max(1, (chunk_mib << 20) // max(1, degrees[0] * 8 * 3))
    edge_slots = write_csr(csr, order, starts, sizes, component, rank, offsets,
                           vertices, band_rows)

    labels = np.empty(components, dtype=np.int64)
    for index in range(components):
        members = order[starts[index]:starts[index] + sizes[index]]
        labels[index] = int(members.max()) + 1
    label_sum = int((labels * sizes).sum())

    dataset_metadata = dict(expected_metadata)
    dataset_metadata.update({
        "layout": LAYOUT,
        "dtype": {"row_ptr": "int64", "col_idx": "int32", "values": "float64"},
        "component_sizes": [int(size) for size in sizes],
        "degrees": [int(degree) for degree in degrees],
        "edge_slots": int(edge_slots),
        "edges": int(edge_slots // 2),
        "bytes": int(edge_slots * 12 + (vertices + 1) * 8),
        "density_actual": edge_slots / float(vertices) / float(vertices),
        "expected_components": components,
        "expected_label_sum": label_sum,
    })
    temporary = out / f".metadata.json.tmp-{os.getpid()}"
    temporary.write_text(json.dumps(dataset_metadata, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, metadata_path)
    print(f"Prepared connected-components graph in {out} "
          f"({edge_slots} edge slots, degree {degrees[0]}, "
          f"{dataset_metadata['bytes'] / 1e9:.2f} GB).", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vertices", type=int, required=True)
    parser.add_argument("--components", type=int, default=4,
                        help="planted components: one dominant plus components-1 small ones")
    parser.add_argument("--band", type=int, default=64,
                        help="consecutive ring offsets before the power-of-two ladder; "
                             "sets degree, and so dataset size, at fixed vertex count")
    parser.add_argument("--minority-fraction", type=float, default=0.015,
                        help="share of vertices spread over the small components")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chunk-mib", type=int, default=256)
    args = parser.parse_args()
    if min(args.vertices, args.components, args.band, args.chunk_mib) < 1:
        raise ValueError("vertices, components, band, and chunk-mib must be positive")
    if not 0.0 <= args.minority_fraction < 1.0:
        raise ValueError("minority-fraction must be in [0, 1)")
    prepare(args.out, args.vertices, args.components, args.band,
            args.minority_fraction, args.seed, args.chunk_mib)


if __name__ == "__main__":
    main()
