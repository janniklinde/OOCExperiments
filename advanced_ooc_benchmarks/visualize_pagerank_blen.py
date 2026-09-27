#!/usr/bin/env python3
"""Render the 16 GiB Twitter PageRank block-size comparison."""

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ooc-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ooc-cache"))

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import visualize_invocation as vi


SERIES = (
    ("pagerank_blen-ooc-bs2500", "ACES 2.5k", "#00507E"),
    ("pagerank_blen-ooc-bs5000", "ACES 5k", "#A74800"),
    ("pagerank_blen-ooc-bs10000", "ACES 10k", "#007B5A"),
    ("pagerank_scipy", "SciPy", "#709EB9"),
)


def telemetry_seconds(path):
    try:
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        return float(rows[-1]["elapsed_ms"]) / 1000 if rows else math.nan
    except (OSError, KeyError, TypeError, ValueError):
        return math.nan


def load_result(invocation, case_id):
    case = invocation / case_id
    result_path = case / "results.csv"
    if not result_path.is_file() or result_path.stat().st_size == 0:
        return None
    with result_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        return None
    row = rows[-1]
    wall = vi.number(row.get("wall_seconds"))
    if not math.isfinite(wall):
        wall = telemetry_seconds(vi.resolved_path(case, row["telemetry"]))
    return {
        "status": row["status"],
        "wall": wall,
        "read_gib": vi.number(row.get("io_read_bytes")) / 2**30,
        "write_gib": vi.number(row.get("io_write_bytes")) / 2**30,
    }


def load_series(ooc_invocation, scipy_invocation):
    values = []
    for case_id, label, color in SERIES:
        invocation = scipy_invocation if case_id == "pagerank_scipy" else ooc_invocation
        result = load_result(invocation, case_id) if invocation else None
        if result is not None:
            values.append((label, color, result))
    return values


def legend(axis, values, location="upper center"):
    handles = [Patch(facecolor=color, edgecolor="black", label=label)
               for label, color, _ in values]
    axis.legend(handles=handles, ncol=2, loc=location, bbox_to_anchor=(0.5, 0.97),
                frameon=False, prop={"weight": "bold", "size": 18},
                handlelength=1.3, handletextpad=0.4, columnspacing=0.8,
                labelspacing=0.2, borderaxespad=0.0)


def offsets(count, width=0.16):
    return [(index - (count - 1) / 2) * width for index in range(count)], width


def save_runtime(target, values):
    figure, axis = plt.subplots(figsize=(vi.FIGURE_WIDTH, 6.5), constrained_layout=True)
    positions, width = offsets(len(values))
    for position, (label, color, result) in zip(positions, values):
        if result["status"] == "ok" and math.isfinite(result["wall"]):
            axis.bar(position, result["wall"], width=width, color=color,
                     edgecolor="black", linewidth=0.55)
        else:
            axis.text(position, 0.04, result["status"].upper(),
                      transform=axis.get_xaxis_transform(), ha="left", va="center_baseline",
                      rotation=90, rotation_mode="anchor", color="#b00020",
                      fontweight="bold", clip_on=True)
    axis.set(xlabel="CGroup Size", ylabel="Elapsed Time [s]", xticks=[0], xticklabels=["16GB"])
    axis.set_xlim(-0.5, 0.5)
    axis.set_yscale("log")
    axis.set_ylim(1, 10_000)
    axis.set_yticks([1, 10, 100, 1_000, 10_000])
    axis.grid(axis="y", color="#cccccc", linewidth=0.6, which="both")
    axis.set_axisbelow(True)
    legend(axis, values)
    figure.savefig(target / "runtime.png", dpi=180)
    figure.savefig(target / "runtime.pdf")
    plt.close(figure)


def save_io(target, values):
    figure, axis = plt.subplots(figsize=(vi.FIGURE_WIDTH, 7.5), constrained_layout=True)
    positions, width = offsets(len(values))
    for position, (label, color, result) in zip(positions, values):
        if result["status"] != "ok":
            axis.text(position, 0.52, result["status"].upper(),
                      transform=axis.get_xaxis_transform(), ha="left", va="center_baseline",
                      rotation=90, rotation_mode="anchor", color="#b00020",
                      fontweight="bold", clip_on=True)
            continue
        axis.bar(position, result["read_gib"], width=width, color=color,
                 edgecolor="black", linewidth=0.55)
        axis.bar(position, -result["write_gib"], width=width, color=color,
                 edgecolor="black", linewidth=0.55)
    axis.set(xlabel="CGroup Size", ylabel="Data Volume [GiB]", xticks=[0], xticklabels=["16GB"])
    axis.set_xlim(-0.5, 0.5)
    axis.set_yscale("symlog", linthresh=1, linscale=1.0)
    axis.set_ylim(-10_000, 10_000)
    ticks = [-10_000, -1_000, -100, -10, -1, 0, 1, 10, 100, 1_000, 10_000]
    axis.set_yticks(ticks)
    axis.axhline(0, color="black", linewidth=1.1)
    axis.grid(axis="y", color="#cccccc", linewidth=0.6, which="major")
    axis.set_axisbelow(True)
    axis.text(0.015, 0.985, "read", transform=axis.transAxes, ha="left", va="top",
              fontweight="bold", color="#404040")
    axis.text(0.015, 0.015, "write", transform=axis.transAxes, ha="left", va="bottom",
              fontweight="bold", color="#404040")
    legend(axis, values)
    figure.savefig(target / "io.png", dpi=180)
    figure.savefig(target / "io.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ooc_invocation", type=Path)
    parser.add_argument("--scipy-invocation", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    ooc_invocation = args.ooc_invocation.resolve()
    scipy_invocation = args.scipy_invocation.resolve() if args.scipy_invocation else None
    target = (args.out.resolve() if args.out else
              Path(__file__).resolve().parent / "pagerank" / "results" / ooc_invocation.name)
    target.mkdir(parents=True, exist_ok=True)
    values = load_series(ooc_invocation, scipy_invocation)
    if not values:
        raise SystemExit("no completed PageRank results found")
    save_runtime(target, values)
    save_io(target, values)
    for label, _, result in values:
        print(f"{label}\t{result['status']}\t{result['wall']:.3f}s\t"
              f"read={result['read_gib']:.3f}GiB\twrite={result['write_gib']:.3f}GiB")
    print(target)


if __name__ == "__main__":
    main()
