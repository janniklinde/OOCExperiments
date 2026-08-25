#!/usr/bin/env python3
"""Convert one arbitrarily large row-major numeric matrix to SystemDS in bounded chunks."""

import argparse
import sys
from pathlib import Path

script_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == script_dir:
    sys.path.pop(0)
sys.path.insert(0, script_dir)

from prepare_dense_systemds import convert_fp64


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--blocksize", type=int, required=True)
    parser.add_argument("--chunk-mib", type=int, default=128)
    parser.add_argument("--dtype", default="float64",
                        help="NumPy dtype of the headerless row-major input (default: float64)")
    parser.add_argument("--java", default="java")
    parser.add_argument("--systemds-jar", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--java-heap", default="3g")
    parser.add_argument("--java-tmp", type=Path)
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--replace-invalid", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()
    if min(args.rows, args.cols, args.blocksize, args.chunk_mib) < 1:
        raise ValueError("rows, cols, blocksize, and chunk-mib must be positive")
    if not args.systemds_jar.is_file():
        raise RuntimeError(f"SystemDS JAR does not exist: {args.systemds_jar}")
    if not args.config.is_file():
        raise RuntimeError(f"SystemDS config does not exist: {args.config}")
    try:
        import numpy as np
        dtype = np.dtype(args.dtype)
    except TypeError as error:
        raise ValueError(f"invalid NumPy dtype: {args.dtype}") from error
    convert_fp64(args.input, args.output, args.rows, args.cols, args.blocksize,
                 args.chunk_mib, args.java, args.systemds_jar, args.config,
                 args.java_heap, args.java_tmp, args.replace_invalid,
                 args.keep_staging, args.staging, dtype)


if __name__ == "__main__":
    main()
