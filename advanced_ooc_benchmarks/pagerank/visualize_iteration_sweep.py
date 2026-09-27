#!/usr/bin/env python3
"""Plot comparable CP/OOC PageRank iteration sweeps from reference-run logs."""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

# This directory contains numpy.py, which would shadow the installed NumPy package.
script_dir = Path(__file__).resolve().parent
sys.path = [str(script_dir.parent)] + [entry for entry in sys.path
                                      if Path(entry or ".").resolve() != script_dir]
from visualize_invocation import FIGURE_WIDTH, implementation_style, legend_label  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


ITERATIONS = (1, 3, 5, 10, 15, 30, 50, 75, 100)
BACKENDS = (
    ("ooc", "systemds-ooc", "o"),
    ("cp", "systemds-cp", "s"),
)
EXECUTION_TIME = re.compile(r"Total execution time:\s*([0-9.]+) sec\.")


def read_run(run_dir: Path, backend: str, iterations: int) -> dict:
    if not (run_dir / "output" / "rank.mtd").is_file():
        raise ValueError(f"missing rank output: {run_dir}")
    log = (run_dir / "systemds.log").read_text(errors="replace")
    if "An Error Occurred" in log:
        raise ValueError(f"SystemDS failed: {run_dir}")
    timing = {}
    for line in (run_dir / "time.txt").read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep:
            timing[key] = float(value)
    execution = EXECUTION_TIME.findall(log)
    if not execution or not all(key in timing for key in
                            ("wall_seconds", "user_seconds", "system_seconds")):
        raise ValueError(f"incomplete timing: {run_dir}")
    return {
        "backend": backend,
        "iterations": iterations,
        "wall_seconds": timing["wall_seconds"],
        "cpu_seconds": timing["user_seconds"] + timing["system_seconds"],
        "execution_seconds": float(execution[-1]),
    }


def plot(rows: list[dict], metric: str, ylabel: str, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 5.7))
    for backend, style_key, marker in BACKENDS:
        points = [row for row in rows if row["backend"] == backend]
        color, _ = implementation_style(style_key)
        ax.plot(
            [row["iterations"] for row in points],
            [row[metric] for row in points],
            color=color,
            marker=marker,
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.2,
            linewidth=2.7,
            label=legend_label(style_key),
        )
    ax.set_xlabel("PageRank iterations")
    ax.set_ylabel(ylabel)
    ax.set_xticks((1, 15, 30, 50, 75, 100))
    ax.set_xlim(0, 102)
    ax.set_ylim(bottom=0, top=max(row[metric] for row in rows) * 1.14)
    ax.grid(axis="y", color="#D7D7D7", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=20, handlelength=2.4)
    fig.tight_layout(pad=0.8)
    for suffix in ("png", "pdf"):
        fig.savefig(destination.with_suffix(f".{suffix}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    rows = [read_run(args.input_dir / f"{backend}-{iterations}", backend, iterations)
            for backend, _, _ in BACKENDS for iterations in ITERATIONS]
    if not all(math.isfinite(row[metric]) and row[metric] > 0
               for row in rows for metric in ("wall_seconds", "cpu_seconds", "execution_seconds")):
        raise ValueError("non-finite or non-positive measurements")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "iteration-metrics.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, "wall_seconds", "Elapsed Time [s]", args.output_dir / "runtime-vs-iterations")
    plot(rows, "cpu_seconds", "CPU Time [s]", args.output_dir / "cpu-vs-iterations")


if __name__ == "__main__":
    main()
