#!/usr/bin/env python3
"""Stage the published Criteo day_21 Parquet shards for SliceLine."""

import argparse
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


REPOSITORY = "criteo/CriteoClickLogs"
REVISION = "main"
# The historic files are zero-based: day_21 is 2015-03-08, not 2015-02-21.
DAY_PATH = "data/day=2015-03-08"
API_URL = "https://huggingface.co/api/datasets/{repo}/tree/{revision}/{path}?recursive=true&expand=false&limit=1000"
RESOLVE_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
VERSION = 2


def expected_metadata(rows: int, revision: str, subset: str) -> dict:
    return {
        "rows": rows,
        "raw_columns": 40,
        "feature_columns": 39,
        "subset": subset,
        "source": {"repository": REPOSITORY, "revision": revision, "path": DAY_PATH,
                   "historical_file": "day_21.gz", "format": "snappy-parquet"},
        "generator": "sliceline/prepare.py",
        "generator_version": VERSION,
    }


def request_json(url: str) -> tuple[list[dict], object]:
    request = urllib.request.Request(url, headers={"User-Agent": "SystemDS-OOC-bench"})
    with urllib.request.urlopen(request) as response:  # fixed Hugging Face API endpoint
        return json.load(response), response.headers


def next_link(header: str | None) -> str | None:
    if not header:
        return None
    match = re.search(r"<([^>]+)>;\s*rel=\"?next\"?", header)
    return match.group(1) if match else None


def fetch_manifest() -> tuple[str, list[dict]]:
    revision = os.environ.get("CRITEO_HF_REVISION", REVISION)
    url = API_URL.format(repo=urllib.parse.quote(REPOSITORY, safe="/"),
                         revision=urllib.parse.quote(revision, safe=""),
                         path=urllib.parse.quote(DAY_PATH, safe="/="))
    files: list[dict] = []
    resolved_revision = revision
    while url:
        page, headers = request_json(url)
        if not isinstance(page, list):
            raise RuntimeError(f"Unexpected Hugging Face listing response at {url}")
        resolved_revision = headers.get("x-repo-commit", resolved_revision)
        files.extend(entry for entry in page if entry.get("type") == "file"
                     and str(entry.get("path", "")).endswith(".parquet"))
        url = next_link(headers.get("Link"))
    if not files:
        raise RuntimeError(f"No Parquet shards found below {REPOSITORY}/{DAY_PATH}")
    files.sort(key=lambda entry: str(entry["path"]))
    return resolved_revision, files


def valid_file(path: Path, size: object) -> bool:
    return path.is_file() and (not isinstance(size, int) or path.stat().st_size == size)


def download_shard(output: Path, revision: str, entry: dict) -> dict:
    source_path = str(entry["path"])
    target = output / Path(source_path).name
    size = entry.get("size")
    if valid_file(target, size):
        return {"path": target.name, "size": target.stat().st_size, "source_path": source_path}
    target.unlink(missing_ok=True)
    partial = target.with_name(target.name + ".download")
    url = RESOLVE_URL.format(repo=urllib.parse.quote(REPOSITORY, safe="/"),
                             revision=urllib.parse.quote(revision, safe=""),
                             path=urllib.parse.quote(source_path, safe="/="))
    print(f"downloading {target.name}", flush=True)
    subprocess.run(["curl", "--fail", "--location", "--retry", "5", "--retry-delay", "5",
                    "--continue-at", "-", "--output", str(partial), url], check=True)
    if isinstance(size, int) and partial.stat().st_size != size:
        raise RuntimeError(f"Downloaded {target.name} has {partial.stat().st_size} bytes; expected {size}")
    os.replace(partial, target)
    return {"path": target.name, "size": target.stat().st_size, "source_path": source_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage Criteo day_21 Parquet shards from Hugging Face.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--raw-name", default="D21.tsv")
    parser.add_argument("--subset", choices=("all_rows", "every_second_row"), default="all_rows")
    parser.add_argument("--finalize", action="store_true",
                        help="write the ready marker after the Spark conversion completed")
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")

    raw = args.out / args.raw_name
    metadata = args.out / "metadata.json"
    manifest = args.out / "parquet-manifest.json"
    if args.finalize:
        if not raw.is_file() or raw.stat().st_size == 0:
            raise RuntimeError(f"Missing converted source: {raw}")
        source = json.loads(manifest.read_text(encoding="utf-8"))
        metadata.write_text(json.dumps(expected_metadata(args.rows, source["revision"], args.subset),
                                       indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return

    if raw.is_file() and metadata.is_file():
        try:
            current = json.loads(metadata.read_text(encoding="utf-8"))
            if current == expected_metadata(args.rows, current["source"]["revision"], args.subset):
                return
        except (KeyError, OSError, json.JSONDecodeError):
            pass

    shard_dir = args.out / "parquet" / "day=2015-03-08"
    shard_dir.mkdir(parents=True, exist_ok=True)
    try:
        revision, entries = fetch_manifest()
        shards = [download_shard(shard_dir, revision, entry) for entry in entries]
    except (OSError, subprocess.CalledProcessError, urllib.error.URLError) as error:
        raise RuntimeError(
            "Unable to stage Criteo day_21 from Hugging Face. Set CRITEO_HF_REVISION to a "
            "known revision if main changed, or retry when network access is available."
        ) from error
    manifest.write_text(json.dumps({"repository": REPOSITORY, "revision": revision,
                                    "path": DAY_PATH, "historical_file": "day_21.gz",
                                    "format": "snappy-parquet", "shards": shards},
                                   indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
