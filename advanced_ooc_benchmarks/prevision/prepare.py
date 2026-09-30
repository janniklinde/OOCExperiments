#!/usr/bin/env python3
"""Convert canonical row-major FP64 data to PreVision TileStore outside timed runs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def compatible_manifest(stored: dict, expected: dict) -> bool:
    """An algorithm-only rebuild must not invalidate unchanged prepared inputs.

    The adapter hash is provenance, not the input-format version. Bump the
    generator version when import/layout/initialization semantics change.
    """
    if not isinstance(stored, dict):
        return False
    return ({key: value for key, value in stored.items() if key != "adapter_sha256"}
            == {key: value for key, value in expected.items() if key != "adapter_sha256"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("gnmf", "gram"), required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--tile", type=int, help="legacy square tile extent")
    parser.add_argument("--tile-rows", type=int, help="rectangular tile row extent")
    parser.add_argument("--tile-cols", type=int, help="rectangular tile column extent")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    if args.tile is not None:
        if args.tile_rows is not None or args.tile_cols is not None:
            parser.error("use --tile or --tile-rows/--tile-cols, not both")
        tile_rows = tile_cols = args.tile
    else:
        if args.tile_rows is None or args.tile_cols is None:
            parser.error("specify both --tile-rows and --tile-cols, or legacy --tile")
        tile_rows, tile_cols = args.tile_rows, args.tile_cols
    if min(args.rows, args.cols, tile_rows, tile_cols, args.rank, args.seed) < 1:
        parser.error("rows, cols, tile, rank, and seed must be positive")
    if args.rows % tile_rows or args.cols % tile_cols:
        parser.error("PreVision import currently needs exact tile divisibility")
    if not args.adapter.is_file() or not os.access(args.adapter, os.X_OK):
        parser.error(f"PreVision adapter is not executable: {args.adapter}")
    raw = args.raw.resolve()
    if raw.stat().st_size != args.rows * args.cols * 8:
        parser.error("raw FP64 file size does not match rows × cols × 8")
    adapter = args.adapter.resolve()
    metadata = {
        "generator": "prevision/prepare.py",
        "generator_version": 1 if args.tile is not None else 2,
        "mode": args.mode, "rows": args.rows, "cols": args.cols,
        "rank": args.rank if args.mode == "gnmf" else None,
        "seed": args.seed if args.mode == "gnmf" else None,
        "source": fingerprint(raw),
        "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
    }
    # Preserve the old manifest contract when using the legacy square CLI, so
    # a valid already-prepared variant is not regenerated after this upgrade.
    if args.tile is not None:
        metadata["tile"] = args.tile
    else:
        metadata.update(tile_rows=tile_rows, tile_cols=tile_cols)
    output = args.output.resolve()
    manifest = output / "manifest.json"
    required = ["X.tilestore"]
    if args.mode == "gnmf":
        required += ["W.tilestore", "H.tilestore"]
    if manifest.is_file() and all((output / name).is_dir() for name in required):
        try:
            if compatible_manifest(json.loads(manifest.read_text()), metadata):
                print(f"valid PreVision data already present at {output}")
                return
        except json.JSONDecodeError:
            pass

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        def imported(name: str, rows: int, cols: int, tr: int, tc: int, mode: str) -> None:
            command = [str(adapter), "import", str(raw), str(staging / name),
                       str(rows), str(cols), str(tr), str(tc), mode, str(args.seed)]
            subprocess.run(command, check=True)

        imported("X", args.rows, args.cols, tile_rows, tile_cols, "copy")
        if args.mode == "gnmf":
            imported("W", args.rows, args.rank, tile_rows, args.rank, "gnmf-w")
            imported("H", args.rank, args.cols, args.rank, tile_cols, "gnmf-h")
        (staging / "manifest.json").write_text(
            json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        # The path is owned by this converter and the replacement is completed only
        # after every TileStore array and its provenance have been prepared.
        if output.exists():
            shutil.rmtree(output)
        staging.rename(output)
        print(f"prepared PreVision {args.mode} data at {output}")
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
