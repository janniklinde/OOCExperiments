#!/usr/bin/env python3
"""Render grouped workload comparisons for one benchmark invocation.

For each workload this writes ``runtime``, ``read``, ``write``, ``cpu``, and ``io`` figures below
``<workload>/results/<invocation-id>/``. Every plot groups bars by memory profile and,
within each group, compares all executed implementations/backends. Cgroup I/O is preferred;
when unavailable, GNU time's Linux rusage block counters are converted with 512-byte blocks.
"""

import argparse
import csv
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ooc-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Figures are scaled down heavily in the paper, so keep every typographic element large.
plt.rcParams.update({
    "font.size": 28,
    "axes.labelsize": "medium",
    "axes.titlesize": "large",
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "legend.fontsize": "small",
    "figure.titlesize": "large",
})


# Every figure shares this width so that panels stacked in the paper are scaled by the same
# factor and therefore end up with identical on-page font sizes.
FIGURE_WIDTH = 7.5

# Bold, slightly below the body size so the legend reads clearly without crowding the plot.
LEGEND_FONT = {"weight": "bold", "size": 26}

_SYSTEM_STYLES = {
    "systemds-ooc": ("#202020", ""),
    "systemds-spark": ("#666666", "//"),
    "numpy": ("#a0a0a0", ".."),
    "dask": ("#dedede", "xx"),
}
_SYSTEM_ORDER = ("systemds-ooc", "systemds-spark", "numpy", "dask")
# Short display names; the internal labels stay untouched because they key the OOC replacement.
_LEGEND_NAMES = {
    "systemds-ooc": "ACES",
    "systemds-spark": "SysDS-SP",
    "numpy": "NumPy",
    "dask": "Dask",
}
_GNU_TIME_IO = re.compile(r"File system (inputs|outputs):\s*([0-9]+)")
_OOC_SPILL = re.compile(
    r"evict writes:\s*[0-9]+\s*\(time\s*[0-9.]+\s*sec,\s*([0-9.]+)\s*GB\)", re.I)


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def implementation_style(name):
    return next((style for prefix, style in _SYSTEM_STYLES.items() if name.startswith(prefix)),
                ("#888888", "++"))


def implementation_label(name):
    """Use framework/backend names instead of workload-specific implementation suffixes."""
    return next((prefix for prefix in _SYSTEM_STYLES if name.startswith(prefix)), name)


def legend_label(name):
    """Compact framework name for legends only."""
    label = implementation_label(name)
    return _LEGEND_NAMES.get(label, label)


def implementation_key(name):
    label = implementation_label(name)
    return (_SYSTEM_ORDER.index(label) if label in _SYSTEM_ORDER else len(_SYSTEM_ORDER), label, name)


def resolved_path(case_dir, recorded):
    """Recorded paths are host paths; only their log filename is portable."""
    return case_dir / "logs" / Path(recorded).name


def gnu_time_io(path):
    values = {"inputs": math.nan, "outputs": math.nan}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return values
    for kind, blocks in _GNU_TIME_IO.findall(text):
        values[kind] = int(blocks) * 512
    return values


def systemds_spill(path):
    try:
        matches = _OOC_SPILL.findall(path.read_text(errors="replace"))
    except OSError:
        return math.nan
    return float(matches[-1]) * 1_000_000_000 if matches else math.nan


def failure_label(status, path):
    if status == "ok":
        return ""
    if status == "timeout":
        return "TIMEOUT"
    try:
        text = path.read_text(errors="replace").lower()
    except OSError:
        text = ""
    if "outofmemory" in text or "out of memory" in text or "oom-kill" in text:
        return "OOM"
    return "FAILURE"


def load_case(case_dir):
    with (case_dir / "resolved-run.json").open(encoding="utf-8") as source:
        run = json.load(source)
    with (case_dir / "results.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        row["memory_profile"] = run.get("resource_profile") or "default"
        row["base_id"] = run["base_id"]
        row["wall"] = number(row.get("wall_seconds"))
        row["cpu"] = number(row.get("cpu_usage_usec")) / 1_000_000
        cgroup_read = number(row.get("io_read_bytes"))
        cgroup_write = number(row.get("io_write_bytes"))
        if math.isfinite(cgroup_read):
            row["read_bytes"], row["write_bytes"], row["io_source"] = (
                cgroup_read, cgroup_write, "cgroup io.stat")
        else:
            fallback = gnu_time_io(resolved_path(case_dir, row.get("metrics", "") + ".time"))
            row["read_bytes"], row["write_bytes"], row["io_source"] = (
                fallback["inputs"], fallback["outputs"], "GNU time blocks × 512 B")
        log_path = resolved_path(case_dir, row.get("log", ""))
        row["spill_bytes"] = systemds_spill(log_path)
        row["failure_label"] = failure_label(row["status"], log_path)
    return rows


def profile_key(profile):
    match = re.fullmatch(r"mem(\d+)", profile)
    return int(match.group(1)) if match else math.inf


def workload_name(base_id):
    """Strip the plan's scaling/size suffixes to get the suite's workload directory name."""
    return str(base_id).removesuffix("_scaling").removesuffix("_3g")


def workload_dir(suite_root, base_id, invocation):
    workload = workload_name(base_id)
    root = suite_root / workload
    if not root.is_dir():
        raise ValueError(f"No workload directory for {base_id!r}: {root}")
    target = root / "results" / invocation.name
    target.mkdir(parents=True, exist_ok=True)
    return target


def grouped_bars(axis, rows, value_key, title, ylabel, log_scale=False, log_limits=None):
    profiles = sorted({row["memory_profile"] for row in rows}, key=profile_key, reverse=True)
    implementations = sorted({row["implementation"] for row in rows}, key=implementation_key)
    lookup = {(row["memory_profile"], row["implementation"]): row for row in rows}
    width = 0.60 / max(1, len(implementations))
    centers = list(range(len(profiles)))
    for index, implementation in enumerate(implementations):
        offset = (index - (len(implementations) - 1) / 2) * width
        for center, profile in zip(centers, profiles):
            row = lookup.get((profile, implementation))
            value = number(row.get(value_key)) if row else math.nan
            color, hatch = implementation_style(implementation)
            shown = value if math.isfinite(value) and value > 0 else 0
            if row and row["status"] == "ok":
                axis.bar(center + offset, shown, width=width, color=color, hatch=hatch,
                         edgecolor="black", linewidth=0.55)
            elif row and row["failure_label"]:
                axis.text(center + offset, 0.04, row["failure_label"],
                          transform=axis.get_xaxis_transform(), ha="left", va="center_baseline",
                          rotation=90, rotation_mode="anchor",
                          fontsize=plt.rcParams["font.size"] * 0.8, color="#b00020",
                          fontweight="bold", clip_on=True)
    labels = [f"{profile_key(profile):g}GB" if math.isfinite(profile_key(profile)) else profile
              for profile in profiles]
    axis.set(xlabel="CGroup Size", ylabel=ylabel, xticks=centers, xticklabels=labels)
    axis.set_xlim(-0.5, len(profiles) - 0.5)
    if title:
        axis.set_title(title, loc="left", pad=34)
    if log_scale:
        axis.set_yscale("log")
        if log_limits:
            axis.set_ylim(*log_limits)
            axis.set_yticks([10 ** exponent for exponent in
                             range(round(math.log10(log_limits[0])),
                                   round(math.log10(log_limits[1])) + 1)])
    axis.grid(axis="y", color="#cccccc", linewidth=0.6, which="both")
    axis.set_axisbelow(True)
    axis.margins(y=0.16)
    legend = [Patch(facecolor=implementation_style(name)[0], hatch=implementation_style(name)[1],
                    edgecolor="black", label=legend_label(name))
              for name in implementations]
    axis.legend(handles=legend, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.97),
                frameon=False, prop=LEGEND_FONT, handlelength=1.3, handletextpad=0.4,
                columnspacing=1.0, labelspacing=0.25, borderaxespad=0.0)


def save_runtime(target, base_id, rows):
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 6.5), constrained_layout=True)
    grouped_bars(axis, rows, "wall", "",
                 "Elapsed Time [s]", log_scale=True, log_limits=(1, 10_000))
    figure.savefig(target / "runtime.png", dpi=180)
    figure.savefig(target / "runtime.pdf")
    plt.close(figure)


def save_cpu(target, base_id, rows):
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 6.5), constrained_layout=True)
    grouped_bars(axis, rows, "cpu", "CPU Consumption",
                 "CPU Time [s]", log_scale=True)
    figure.savefig(target / "cpu.png", dpi=180)
    figure.savefig(target / "cpu.pdf")
    plt.close(figure)


def mirrored_io_bars(axis, rows, log_limits=(1, 10_000), legend=True):
    """Draw read volume upwards and write volume mirrored downwards on one symmetric log axis."""
    profiles = sorted({row["memory_profile"] for row in rows}, key=profile_key, reverse=True)
    implementations = sorted({row["implementation"] for row in rows}, key=implementation_key)
    lookup = {(row["memory_profile"], row["implementation"]): row for row in rows}
    width = 0.60 / max(1, len(implementations))
    centers = list(range(len(profiles)))
    low, high = log_limits
    for index, implementation in enumerate(implementations):
        offset = (index - (len(implementations) - 1) / 2) * width
        color, hatch = implementation_style(implementation)
        for center, profile in zip(centers, profiles):
            row = lookup.get((profile, implementation))
            if not row:
                continue
            if row["status"] == "ok":
                for value_key, sign in (("read_bytes", 1), ("write_bytes", -1)):
                    value = number(row.get(value_key)) / 2**30
                    shown = value if math.isfinite(value) and value > 0 else 0
                    axis.bar(center + offset, sign * shown, width=width, color=color, hatch=hatch,
                             edgecolor="black", linewidth=0.55)
            elif row["failure_label"]:
                axis.text(center + offset, 0.52, row["failure_label"],
                          transform=axis.get_xaxis_transform(), ha="left", va="center_baseline",
                          rotation=90, rotation_mode="anchor",
                          fontsize=plt.rcParams["font.size"] * 0.8, color="#b00020",
                          fontweight="bold", clip_on=True)

    labels = [f"{profile_key(profile):g}GB" if math.isfinite(profile_key(profile)) else profile
              for profile in profiles]
    axis.set(xlabel="CGroup Size", ylabel="Data Volume [GiB]",
             xticks=centers, xticklabels=labels)
    axis.set_xlim(-0.5, len(profiles) - 0.5)
    axis.set_yscale("symlog", linthresh=low, linscale=1.0)
    axis.set_ylim(-high, high)
    exponents = range(round(math.log10(low)), round(math.log10(high)) + 1)
    ticks = [-(10 ** exponent) for exponent in reversed(exponents)] + [0] + [10 ** exponent
                                                                            for exponent in exponents]
    axis.set_yticks(ticks)
    axis.set_yticklabels([f"$10^{{{abs(exponent)}}}$" if tick else "0"
                          for tick, exponent in zip(ticks, list(reversed(exponents)) + [0] + list(exponents))])
    axis.axhline(0, color="black", linewidth=1.1)
    axis.grid(axis="y", color="#cccccc", linewidth=0.6, which="major")
    axis.set_axisbelow(True)

    axis.text(0.015, 0.985, "read", transform=axis.transAxes, ha="left", va="top",
              fontweight="bold", color="#404040")
    axis.text(0.015, 0.015, "write", transform=axis.transAxes, ha="left", va="bottom",
              fontweight="bold", color="#404040")
    if legend:
        handles = [Patch(facecolor=implementation_style(name)[0],
                         hatch=implementation_style(name)[1],
                         edgecolor="black", label=legend_label(name))
                   for name in implementations]
        axis.legend(handles=handles, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                    frameon=False, prop=LEGEND_FONT, handlelength=1.3, handletextpad=0.4,
                    columnspacing=1.0, borderaxespad=0.4)


def save_io(target, base_id, rows):
    """Read above the axis, write mirrored below, on a shared 10^0..10^4 GiB log scale."""
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 7.5), constrained_layout=True)
    mirrored_io_bars(axis, rows, log_limits=(1, 10_000), legend=False)
    figure.savefig(target / "io.png", dpi=180)
    figure.savefig(target / "io.pdf")
    plt.close(figure)


def save_read(target, rows):
    """Write the standalone input-read-volume figure."""
    reads = [dict(row, io_gib=number(row["read_bytes"]) / 2**30) for row in rows]
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 6.5), constrained_layout=True)
    grouped_bars(axis, reads, "io_gib", "", "Data Read [GiB]")
    figure.savefig(target / "read.png", dpi=180)
    figure.savefig(target / "read.pdf")
    plt.close(figure)


def save_write(target, rows):
    """Write the standalone output-write-volume figure."""
    writes = [dict(row, io_gib=number(row["write_bytes"]) / 2**30) for row in rows]
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, 6.5), constrained_layout=True)
    grouped_bars(axis, writes, "io_gib", "", "Data Written [GiB]")
    figure.savefig(target / "write.png", dpi=180)
    figure.savefig(target / "write.pdf")
    plt.close(figure)


def invocation_path(path):
    if path.is_dir() and (path / "expanded-plan.yaml").exists():
        return path
    candidates = sorted((candidate for candidate in path.iterdir()
                         if candidate.is_dir() and (candidate / "expanded-plan.yaml").exists()),
                        key=lambda candidate: candidate.stat().st_mtime, reverse=True)
    if not candidates:
        raise ValueError(f"No benchmark invocation under {path}")
    return candidates[0]


def replacement_ooc_rows(invocation):
    """Load an OOC-only invocation and reject accidental mixed-run replacement."""
    grouped = defaultdict(list)
    for case in sorted(path for path in invocation.iterdir() if (path / "results.csv").is_file()):
        for row in load_case(case):
            if implementation_label(row["implementation"]) != "systemds-ooc":
                raise ValueError(
                    f"OOC replacement invocation contains non-OOC row: {case.name} "
                    f"({row['implementation']})")
            grouped[row["base_id"]].append(row)
    if not grouped:
        raise ValueError(f"No OOC result rows under {invocation}")
    return grouped


def replace_ooc_rows(rows, replacements, base_id):
    """Keep every original series except SystemDS OOC, then substitute refreshed rows."""
    if base_id not in replacements:
        return rows
    replacement_rows = replacements[base_id]
    original_profiles = {row["memory_profile"] for row in rows
                         if implementation_label(row["implementation"]) == "systemds-ooc"}
    replacement_profiles = {row["memory_profile"] for row in replacement_rows}
    if original_profiles != replacement_profiles:
        raise ValueError(
            f"OOC replacement profiles for {base_id} differ: "
            f"original={sorted(original_profiles)}, replacement={sorted(replacement_profiles)}")
    return [row for row in rows if implementation_label(row["implementation"]) != "systemds-ooc"] + replacement_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path("/media/jannik/data/OOCExperiments/bench-results"),
                        help="invocation directory, or its bench-results parent (default: latest invocation)")
    parser.add_argument("--replace-ooc-from", type=Path, metavar="INVOCATION",
                        help="replace SystemDS-OOC rows with those from this OOC-only invocation")
    parser.add_argument("--figures", choices=("all", "runtime-read", "runtime-read-write"), default="all",
                        help="figure set to render (default: all)")
    parser.add_argument("--skip-workloads", default="", metavar="A,B,C",
                        help="workloads to leave alone, e.g. because a newer invocation already covered them")
    args = parser.parse_args()
    skipped = {name.strip() for name in args.skip_workloads.split(",") if name.strip()}
    invocation = invocation_path(args.path.resolve())
    replacements = (replacement_ooc_rows(invocation_path(args.replace_ooc_from.resolve()))
                    if args.replace_ooc_from else {})
    grouped = defaultdict(list)
    for case in sorted(path for path in invocation.iterdir() if (path / "results.csv").is_file()):
        for row in load_case(case):
            grouped[row["base_id"]].append(row)
    suite_root = Path(__file__).resolve().parent
    rendered = []
    for base_id, rows in grouped.items():
        if workload_name(base_id) in skipped:
            continue
        rows = replace_ooc_rows(rows, replacements, base_id)
        target = workload_dir(suite_root, base_id, invocation)
        save_runtime(target, base_id, rows)
        if args.figures in ("runtime-read", "runtime-read-write"):
            save_read(target, rows)
            if args.figures == "runtime-read-write":
                save_write(target, rows)
        else:
            save_cpu(target, base_id, rows)
            save_read(target, rows)
            save_write(target, rows)
            save_io(target, base_id, rows)
        rendered.append(workload_name(base_id))
    for workload in sorted(set(rendered)):
        print(f"RENDERED\t{workload}\t{invocation}")
    print(f"Generated grouped figures for {len(set(rendered))} workloads under {invocation}")


if __name__ == "__main__":
    main()
