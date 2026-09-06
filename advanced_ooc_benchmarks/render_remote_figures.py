#!/usr/bin/env python3
"""Render selected remote benchmark invocations into per-invocation figure directories.

visualize_invocation.py writes fixed filenames below <workload>/results/<invocation>/, which
collapses several invocations onto each other and cannot express a workload that the suite
has no directory for. This renders the same figures through the same code, into one directory
per invocation named for its start time, with each file named for the run case it came from.

Its lookup keys figures by (memory profile, implementation), so a base id carrying two OOC
blocksizes or two datasets at one profile would silently keep only the last row. Those are
split into separate figures here instead.
"""

import argparse
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import visualize_invocation as vi

FIGURES = ("runtime", "cpu", "read", "write", "io")
_SWEEP = re.compile(r"-(?:ooc|spark)-bs(\d+)$")


def blocksize(case_id):
    sweep = _SWEEP.search(case_id)
    return sweep.group(1) if sweep else ""


def collisions(tagged):
    """The (profile, implementation) bars that more than one row would be drawn into."""
    counted = Counter((row["memory_profile"], vi.implementation_label(row["implementation"]))
                      for _, _, row in tagged)
    return {bar for bar, count in counted.items() if count > 1}


def split(tagged, index):
    """Separate (dataset, blocksize, row) triples along one component, but only far enough
    to resolve collisions: rows that already own their bar are shared into every partition,
    so a Spark arm swept at a different blocksize than OOC stays in the same figure."""
    contested = collisions(tagged)
    if not contested:
        return [("", tagged)]
    def bar(tag):
        return (tag[2]["memory_profile"], vi.implementation_label(tag[2]["implementation"]))
    split_off = [tag for tag in tagged if bar(tag) in contested and tag[index]]
    values = {tag[index] for tag in split_off}
    if len(values) <= 1:
        return [("", tagged)]
    shared = [tag for tag in tagged if tag not in split_off]
    return [(value, shared + [tag for tag in split_off if tag[index] == value])
            for value in sorted(values)]


def partition_rows(invocation):
    """Split a base id only where two rows would land on the same bar.

    grouped_bars keys its lookup by (memory profile, implementation), so a base id spanning
    several datasets or several OOC blocksizes silently keeps whichever row was read last.
    Dataset comes first because it is the coarser axis; blocksize splits only what remains.
    Baseline cases carry no blocksize and are shared into every partition.
    """
    by_base = defaultdict(list)
    for case in sorted(path for path in invocation.iterdir() if (path / "results.csv").is_file()):
        run = vi.json.loads((case / "resolved-run.json").read_text(encoding="utf-8"))
        tag = (str(run.get("dataset") or ""), blocksize(str(run["id"])))
        for row in vi.load_case(case):
            by_base[row["base_id"]].append((*tag, row))
    for base_id, tagged in by_base.items():
        for dataset, by_dataset in split(tagged, 0):
            for size, by_size in split(by_dataset, 1):
                suffix = "-".join(part for part in
                                  (dataset, f"bs{size}" if size else "") if part)
                yield base_id, suffix, [tag[2] for tag in by_size]


def render(invocation, out_dir):
    written = []
    for base_id, suffix, rows in partition_rows(invocation):
        stem = "-".join(part for part in (base_id, suffix) if part)
        with tempfile.TemporaryDirectory() as staging:
            target = Path(staging)
            vi.save_runtime(target, base_id, rows)
            vi.save_cpu(target, base_id, rows)
            vi.save_read(target, rows)
            vi.save_write(target, rows)
            vi.save_io(target, base_id, rows)
            for figure in FIGURES:
                for extension in ("png", "pdf"):
                    source = target / f"{figure}.{extension}"
                    if not source.exists():
                        continue
                    destination = out_dir / f"{stem}-{figure}.{extension}"
                    shutil.move(str(source), destination)
                    written.append(destination)
    return written


DEFAULT_INVOCATIONS = (
    "results-remote/20260902T181411.384560+0000",
    "results-remote-calibration/20260903T143550.348060+0000",
)


def main():
    suite = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("invocations", nargs="*", type=Path,
                        default=[suite / name for name in DEFAULT_INVOCATIONS],
                        help="invocation directories to render (default: the two complete runs)")
    parser.add_argument("--out", type=Path, default=suite / "results-remote-figures")
    args = parser.parse_args()
    total = 0
    for invocation in args.invocations:
        if not (invocation / "expanded-plan.yaml").exists():
            raise SystemExit(f"not an invocation directory: {invocation}")
        out_dir = args.out / invocation.name.split(".")[0].replace("+0000", "")
        out_dir.mkdir(parents=True, exist_ok=True)
        written = render(invocation, out_dir)
        total += len(written)
        print(f"{invocation.name}\t{len(written)} files -> {out_dir}")
    print(f"wrote {total} files to {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
