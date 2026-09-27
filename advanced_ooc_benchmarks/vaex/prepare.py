#!/usr/bin/env python3
"""Optional lossless, bounded-memory row-major FP64 to columnar conversion."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from support import source_info


def prepare(data, buffer_mib=64):
    if buffer_mib < 1:
        raise ValueError("buffer-mib must be positive")
    path, shape, fingerprint = source_info(data)
    destination = Path(data) / "vaex"
    destination.mkdir(exist_ok=True)
    manifest = destination / "provenance.json"
    target = destination / "X.npy"
    if manifest.exists() and target.exists() and json.loads(manifest.read_text()) == fingerprint:
        existing = np.load(target, mmap_mode="r")
        if existing.shape == shape and existing.dtype == np.dtype("<f8") and existing.flags.f_contiguous:
            print(f"Already prepared: {target}", flush=True)
            return target
    # Only this utility's derived artifacts are replaced; never canonical inputs.
    temporary = destination / "X.partial.npy"
    source = np.memmap(path, mode="r", dtype="<f8", shape=shape)
    converted = np.lib.format.open_memmap(temporary, mode="w+", dtype="<f8",
                                        shape=shape, fortran_order=True)
    rows = max(1, buffer_mib * 1024**2 // (8 * shape[1]))
    for first in range(0, shape[0], rows):
        last = min(first + rows, shape[0])
        band = np.array(source[first:last], copy=True, order="F")
        converted[first:last] = band
        print(f"Prepared rows {last}/{shape[0]}", flush=True)
    converted.flush()
    del converted
    if source_info(data)[2] != fingerprint:
        raise RuntimeError("source changed during conversion")
    os.replace(temporary, target)
    manifest.write_text(json.dumps(fingerprint, indent=2) + "\n")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--buffer-mib", type=int, default=64)
    args = parser.parse_args()
    prepare(args.data, args.buffer_mib)
