#!/usr/bin/env python3
"""Convert staged Criteo Parquet shards to the benchmark's canonical TSV stream."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded-memory Criteo Parquet to TSV conversion.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--subset", choices=("all_rows", "every_second_row"), default="all_rows")
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    if args.out.is_file() and args.out.stat().st_size:
        return

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    shard_dir = args.manifest.parent / "parquet" / "day=2015-03-08"
    shards = [shard_dir / entry["path"] for entry in manifest["shards"]]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing staged Parquet shards, beginning with {missing[0]}")

    from pyspark.sql import SparkSession

    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.unlink(missing_ok=True)
    spark = SparkSession.builder.appName("SystemDS-SliceLine-Criteo-stage").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    selected = 0
    source_rows = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="") as target:
            # Publisher part numbers are order-preserving. One file at a time keeps
            # the driver bounded by one Parquet partition and retains that order.
            for shard in shards:
                frame = spark.read.parquet(str(shard))
                if len(frame.columns) != 40:
                    raise RuntimeError(f"{shard.name} has {len(frame.columns)} columns; expected 40")
                for row in frame.toLocalIterator(prefetchPartitions=False):
                    source_rows += 1
                    if args.subset == "every_second_row" and source_rows % 2:
                        continue
                    target.write("\t".join("" if value is None else str(value) for value in row))
                    target.write("\n")
                    selected += 1
                    if selected == args.rows:
                        break
                if selected == args.rows:
                    break
        if selected != args.rows:
            raise RuntimeError(f"Only {selected} every-second rows available; expected {args.rows}")
        temporary.replace(args.out)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
