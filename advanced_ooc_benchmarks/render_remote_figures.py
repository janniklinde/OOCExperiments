#!/usr/bin/env python3
"""Render selected remote benchmark invocations into per-invocation figure directories.

visualize_invocation.py writes fixed filenames below <workload>/results/<invocation>/, which
collapses several invocations onto each other and cannot express a workload that the suite
has no directory for. This renders the same figures through the same code, into one directory
per invocation named for its start time, with each file named for the run case it came from.

Its lookup keys figures by (memory profile, implementation), so a base id carrying two OOC
blocksizes or two datasets at one profile would silently keep only the last row. Those are
split into separate figures here instead. A workload's standalone 16 GB case is folded into
its matching ``*_memory`` sweep.
"""

import argparse
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import visualize_invocation as vi

FIGURES = ("runtime", "cpu", "read", "write", "io")
_SWEEP = re.compile(r"-(?:ooc|spark)-bs(\d+)$")


def blocksize(case_id):
    sweep = _SWEEP.search(case_id)
    return sweep.group(1) if sweep else ""


def axis_value(tag, axis):
    return tag[0] if axis == "dataset" else tag[2]["memory_profile"]


def collisions(tagged, axis):
    """The (profile, implementation) bars that more than one row would be drawn into."""
    counted = Counter((axis_value(tag, axis), vi.implementation_label(tag[2]["implementation"]))
                      for tag in tagged)
    return {bar for bar, count in counted.items() if count > 1}


def split(tagged, index, axis):
    """Separate (dataset, blocksize, row) triples along one component, but only far enough
    to resolve collisions: rows that already own their bar are shared into every partition,
    so a Spark arm swept at a different blocksize than OOC stays in the same figure."""
    contested = collisions(tagged, axis)
    if not contested:
        return [("", tagged)]
    def bar(tag):
        return (axis_value(tag, axis), vi.implementation_label(tag[2]["implementation"]))
    split_off = [tag for tag in tagged if bar(tag) in contested and tag[index]]
    values = {tag[index] for tag in split_off}
    if len(values) <= 1:
        return [("", tagged)]
    shared = [tag for tag in tagged if tag not in split_off]
    return [(value, shared + [tag for tag in split_off if tag[index] == value])
            for value in sorted(values)]


def dataset_bytes(invocation):
    plan = yaml.safe_load((invocation / "expanded-plan.yaml").read_text(encoding="utf-8"))
    result = {}
    for name, dataset in plan.get("datasets", {}).items():
        parameters = dataset.get("parameters", {})
        rows, cols = parameters.get("rows"), parameters.get("cols")
        if isinstance(rows, int) and isinstance(cols, int):
            result[name] = rows * cols * 8
    return result


def tagged_rows(invocation):
    """Yield dataset, blocksize and result rows from one invocation."""
    sizes = dataset_bytes(invocation)
    for case in sorted(path for path in invocation.iterdir() if (path / "results.csv").is_file()):
        run = vi.json.loads((case / "resolved-run.json").read_text(encoding="utf-8"))
        dataset = str(run.get("dataset") or "")
        tag = (dataset, blocksize(str(run["id"])))
        for row in vi.load_case(case):
            yield (*tag, dict(row, dataset=dataset, dataset_bytes=sizes.get(dataset)))


def partition_rows(invocation, fallbacks=()):
    """Split a base id only where two rows would land on the same bar.

    grouped_bars keys its lookup by one x-axis value and implementation, so a base id spanning
    several datasets or several OOC blocksizes silently keeps whichever row was read last.
    A dataset-only sweep uses dataset size as that axis; otherwise dataset comes first and
    blocksize splits only what remains.
    Baseline cases carry no blocksize and are shared into every partition.
    """
    by_base = defaultdict(list)
    selected = set()
    for candidate in (invocation, *fallbacks):
        for tag in tagged_rows(candidate):
            row = tag[2]
            key = (row["base_id"], tag[0], row["memory_profile"], row["implementation"])
            if key not in selected:
                selected.add(key)
                by_base[row["base_id"]].append(tag)

    # The main workload case is the 16 GB point omitted from the corresponding memory sweep.
    # Keep the memory-sweep name so the rendered artifact represents one continuous experiment.
    for base_id in list(by_base):
        memory_id = f"{base_id}_memory"
        if memory_id not in by_base:
            continue
        main_datasets = {tag[0] for tag in by_base[base_id]}
        memory_datasets = {tag[0] for tag in by_base[memory_id]}
        if main_datasets <= memory_datasets:
            by_base[memory_id].extend(by_base.pop(base_id))

    for base_id, tagged in by_base.items():
        profiles = {tag[2]["memory_profile"] for tag in tagged}
        use_dataset_axis = len(profiles) == 1 and len({tag[0] for tag in tagged}) > 1
        dataset_partitions = [("", tagged)] if use_dataset_axis else split(tagged, 0, "memory")
        for dataset, by_dataset in dataset_partitions:
            axis = "dataset" if use_dataset_axis else "memory"
            for size, by_size in split(by_dataset, 1, axis):
                suffix = "-".join(part for part in
                                  (dataset, f"bs{size}" if size else "") if part)
                yield base_id, suffix, [tag[2] for tag in by_size]


def render(invocation, out_dir, figures, fallbacks=()):
    written = []
    for base_id, suffix, rows in partition_rows(invocation, fallbacks):
        stem = "-".join(part for part in (base_id, suffix) if part)
        with tempfile.TemporaryDirectory() as staging:
            target = Path(staging)
            vi.save_runtime(target, base_id, rows)
            if "cpu" in figures:
                vi.save_cpu(target, base_id, rows)
            if "read" in figures:
                vi.save_read(target, rows)
            if "write" in figures:
                vi.save_write(target, rows)
            if "io" in figures:
                vi.save_io(target, base_id, rows)
            for figure in figures:
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
    parser.add_argument("--figures", choices=("all", "runtime-io"), default="all",
                        help="figure set to render (default: all)")
    parser.add_argument("--fallback", type=Path, nargs="*", default=[], metavar="INVOCATION",
                        help="older invocations, in descending priority, used only for missing bars")
    args = parser.parse_args()
    total = 0
    for invocation in args.invocations:
        if not (invocation / "expanded-plan.yaml").exists():
            raise SystemExit(f"not an invocation directory: {invocation}")
        for fallback in args.fallback:
            if not (fallback / "expanded-plan.yaml").exists():
                raise SystemExit(f"not an invocation directory: {fallback}")
        out_dir = args.out / invocation.name.split(".")[0].replace("+0000", "")
        out_dir.mkdir(parents=True, exist_ok=True)
        figures = FIGURES if args.figures == "all" else ("runtime", "io")
        written = render(invocation, out_dir, figures, args.fallback)
        total += len(written)
        print(f"{invocation.name}\t{len(written)} files -> {out_dir}")
    print(f"wrote {total} files to {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
