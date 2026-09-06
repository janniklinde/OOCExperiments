#!/usr/bin/env python3
"""Summarise one invocation: the derived numbers, not the raw columns.

results.csv carries cumulative counters. What a reader actually wants is how they
relate to the run -- CPU against wall time, bytes read against the size of the input,
stall time against the duration -- so this prints those ratios beside the raw values.

Usage: inspect_invocation.py INVOCATION_DIR [--sort COLUMN] [--csv]
"""
import argparse
import csv
import json
from pathlib import Path

COLUMNS = ["case", "implementation", "status", "wall_s", "cores", "peak_GiB",
           "read_GB", "write_GB", "passes", "io_stall_%", "oom"]


def number(value, default=0.0):
    """Parse a counter that may be absent, empty, or the literal `nan` a failed arm writes."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return default if parsed != parsed else parsed


def stalled_usec(telemetry):
    """Cumulative io.pressure `some` total from the last telemetry sample."""
    try:
        with open(telemetry, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return None
    for row in reversed(rows):
        value = row.get("io_pressure_some_usec", "")
        if value not in ("", None):
            try:
                return int(value)
            except ValueError:
                return None
    return None


def summarise(case_dir):
    results = case_dir / "results.csv"
    if not results.is_file():
        return []
    context = {}
    resolved = case_dir / "resolved-context.json"
    if resolved.is_file():
        context = json.loads(resolved.read_text(encoding="utf-8"))
    try:
        input_bytes = int(context["dataset.rows"]) * int(context["dataset.cols"]) * 8
    except (KeyError, TypeError, ValueError):
        input_bytes = 0

    rows = []
    with open(results, newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            wall = number(record.get("wall_seconds"))
            cpu = number(record.get("cpu_usage_usec"))
            read = number(record.get("io_read_bytes"))
            write = number(record.get("io_write_bytes"))
            # The telemetry path in the CSV is the container's; resolve it locally.
            telemetry = case_dir / "logs" / Path(record.get("telemetry", "")).name
            stalled = stalled_usec(telemetry)
            rows.append({
                "case": case_dir.name,
                "implementation": record.get("implementation", ""),
                "status": record.get("status", ""),
                "wall_s": f"{wall:.1f}",
                # Mean cores busy: >1 means real parallelism, ~1 means a serial phase.
                "cores": f"{cpu / (wall * 1e6):.1f}" if wall else "-",
                "peak_GiB": f"{number(record.get('memory_peak_bytes')) / 2**30:.2f}",
                "read_GB": f"{read / 1e9:.1f}",
                "write_GB": f"{write / 1e9:.1f}",
                # How many times the engine re-read the input. The headline number for
                # an out-of-core comparison: 1.0 is a single streaming pass.
                "passes": f"{read / input_bytes:.2f}" if input_bytes else "-",
                "io_stall_%": f"{stalled / (wall * 1e6) * 100:.0f}"
                              if stalled is not None and wall else "-",
                "oom": f"{number(record.get('oom_kill_events')):.0f}",
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("invocation", type=Path)
    parser.add_argument("--sort", default="case", choices=COLUMNS)
    parser.add_argument("--csv", action="store_true", help="emit CSV instead of a table")
    args = parser.parse_args()

    rows = []
    for case_dir in sorted(p for p in args.invocation.iterdir() if p.is_dir()):
        rows.extend(summarise(case_dir))
    if not rows:
        raise SystemExit(f"No results.csv under {args.invocation}")

    def key(row):
        try:
            return (0, float(row[args.sort]))
        except ValueError:
            return (1, row[args.sort])
    rows.sort(key=key)

    if args.csv:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        return 0

    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in COLUMNS}
    print("  ".join(c.ljust(widths[c]) for c in COLUMNS))
    print("  ".join("-" * widths[c] for c in COLUMNS))
    for row in rows:
        print("  ".join(str(row[c]).ljust(widths[c]) for c in COLUMNS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
