#!/usr/bin/env python3
import argparse
import csv
import math
import re
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("directories", nargs="+", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
rows, layouts, failures = [], [], []
reference = {}
for directory in args.directories:
    for path in sorted(directory.glob("*.log")):
        if path.name == "gc.log":
            continue
        lines = path.read_text().splitlines()
        meta = next((line for line in lines if line.startswith("# columns=")), None)
        header = next((line for line in lines if line.startswith("phase,algorithm,")), None)
        if meta is None or header is None:
            failures.append(str(path))
            continue
        metadata = dict(re.findall(r"(\w+)=([^ ]+)", meta))
        samples = list(csv.DictReader([header] + [line for line in lines if line.startswith(("iteration,", "prepare,", "reorder,"))]))
        if len([row for row in samples if row["phase"] == "iteration"]) != 5 or any("Exception" in line or "AssertionError" in line for line in lines):
            failures.append(str(path))
            continue
        for row in samples:
            row.update(metadata, source=str(path))
            rows.append(row)
            if row["phase"] == "iteration":
                key = tuple(row[k] for k in ("algorithm", "rows", "columns", "rank", "iteration"))
                value = float(row["checksum"])
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite checksum: {path}")
                old = reference.setdefault(key, value)
                if not math.isclose(old, value, rel_tol=1e-8, abs_tol=1e-8):
                    raise ValueError(f"Checksum mismatch {path}: {old} vs {value}")
        for line in lines:
            if line.startswith(("# packs ", "# layout ", "# traffic ")):
                layouts.append(dict(re.findall(r"(\w+)=([^ ]+)", line), source=str(path)))

args.output.parent.mkdir(parents=True, exist_ok=True)
if rows:
    with args.output.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
if layouts:
    with args.output.with_name(args.output.name + "-layouts.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(row.keys() for row in layouts))))
        writer.writeheader()
        writer.writerows(layouts)

text = ["# Staging policy measurements", "", "Five measured iterations per invocation after at least 30 s JVM warmup. Medians below. Bytes are serialized cache reload/spill, not necessarily physical SSD traffic.", "",
        "| Case | Order | Policy | Cache MiB | Seed | Iter s | CPU s | Reload GB | Write GB | Groups | Prep s | Total prep + iterations s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for source in sorted({row["source"] for row in rows}):
    samples = [row for row in rows if row["source"] == source]
    it = [row for row in samples if row["phase"] == "iteration"]
    r = it[0]
    def median(field):
        return statistics.median(float(row[field]) for row in it)
    prep = sum(float(row["seconds"]) for row in samples if row["phase"] != "iteration")
    total = sum(float(row["seconds"]) for row in samples)
    text.append(f'| {r["algorithm"]} {r["rows"]}×{r["columns"]}/r{r["rank"]} | {r["input_order"]} | {r["policy"]} | {int(r["cache_bytes"])/2**20:g} | {r["seed"]} | {median("seconds"):.3f} | {median("cpu_seconds"):.2f} | {median("read_bytes")/1e9:.3f} | {median("write_bytes")/1e9:.3f} | {median("tasks"):.0f} | {prep:.3f} | {total:.3f} |')
text.extend(["", f"Successful invocations: {len({row['source'] for row in rows})}; measured iterations: {sum(row['phase']=='iteration' for row in rows)}. Matching checksums agree.",
             f"Storage-accounted reads in measured iterations: {sum(int(row['process_read_bytes']) for row in rows if row['phase']=='iteration'):,} bytes.", "", "## Failed/incomplete logs", ""])
text.extend(f"- {name}" for name in failures)
args.output.with_suffix(".md").write_text("\n".join(text) + "\n")
print("\n".join(text))
