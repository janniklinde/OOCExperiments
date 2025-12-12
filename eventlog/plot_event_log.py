#!/usr/bin/env python3
"""
Visualize compute vs idle time per thread as a Gantt-style chart.

Input CSV columns (case-sensitive):
    ThreadID, CallerID, StartNanos, EndNanos

Usage examples:
    python plot_event_log.py EventLog.csv
    python plot_event_log.py EventLog.csv --output timeline.png --unit us
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


TimeEntry = Tuple[int, str, int, int]  # thread, caller, start_ns, end_ns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot compute/idle timeline per thread.")
    parser.add_argument("csv", type=Path, help="Path to EventLog.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional image output path (png/svg/pdf). If omitted, show interactively.",
    )
    parser.add_argument(
        "--unit",
        choices=["ns", "us", "ms", "s"],
        default="s",
        help="Time unit to display on the x-axis (default: seconds).",
    )
    parser.add_argument(
        "--show-idle",
        action="store_true",
        help="Draw idle gaps as gray bars (default is to leave gaps).",
    )
    parser.add_argument(
        "--max-legend",
        type=int,
        default=25,
        help="Maximum number of CallerID entries to include in the legend.",
    )
    return parser.parse_args()


def load_entries(path: Path) -> List[TimeEntry]:
    entries: List[TimeEntry] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"ThreadID", "CallerID", "StartNanos", "EndNanos"}
        if not required.issubset(reader.fieldnames or []):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            entries.append(
                (
                    int(row["ThreadID"]),
                    row["CallerID"],
                    int(row["StartNanos"]),
                    int(row["EndNanos"]),
                )
            )
    if not entries:
        raise ValueError("No data rows found.")
    return entries


def unit_factor(unit: str) -> float:
    if unit == "ns":
        return 1.0
    if unit == "us":
        return 1e3
    if unit == "ms":
        return 1e6
    if unit == "s":
        return 1e9
    raise ValueError(f"Unsupported unit: {unit}")


def build_segments(
    entries: List[TimeEntry], include_idle: bool
) -> Tuple[Dict[int, List[Tuple[float, float, str]]], Dict[str, str], float, float]:
    """
    Returns per-thread segments, color map, global min/max time.
    Segments are tuples of (start_ns, duration_ns, label).
    """
    by_thread: Dict[int, List[TimeEntry]] = defaultdict(list)
    for ent in entries:
        by_thread[ent[0]].append(ent)

    for thread_entries in by_thread.values():
        thread_entries.sort(key=lambda e: e[2])

    all_callers = sorted({e[1] for e in entries})
    cmap = plt.get_cmap("tab20")
    colors: Dict[str, str] = {
        caller: cmap(i % cmap.N) for i, caller in enumerate(all_callers)
    }

    segments: Dict[int, List[Tuple[float, float, str]]] = defaultdict(list)
    min_start = min(e[2] for e in entries)
    max_end = max(e[3] for e in entries)

    for thread_id, thread_entries in by_thread.items():
        for idx, (thr, caller, start, end) in enumerate(thread_entries):
            duration = end - start
            if duration <= 0:
                continue  # skip malformed spans
            segments[thread_id].append((start, duration, caller))

            if not include_idle and idx + 1 < len(thread_entries):
                continue

            if include_idle and idx + 1 < len(thread_entries):
                next_start = thread_entries[idx + 1][2]
                gap = next_start - end
                if gap > 0:
                    segments[thread_id].append((end, gap, "__idle__"))

        # Trailing idle to max_end is omitted to keep the plot compact.

    if include_idle:
        colors["__idle__"] = "#d3d3d3"

    return segments, colors, min_start, max_end


def plot_gantt(
    segments: Dict[int, List[Tuple[float, float, str]]],
    colors: Dict[str, str],
    min_start: float,
    max_end: float,
    unit: str,
    output: Optional[Path],
    max_legend: int,
) -> None:
    factor = unit_factor(unit)
    fig_height = max(4, 0.7 * len(segments))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    y_ticks = []
    y_labels = []
    y_base = 10
    height = 8

    for idx, (thread_id, segs) in enumerate(sorted(segments.items())):
        y = idx * (height + 6) + y_base
        y_ticks.append(y + height / 2)
        y_labels.append(str(thread_id))
        for start_ns, dur_ns, label in segs:
            ax.broken_barh(
                [( (start_ns - min_start) / factor, dur_ns / factor )],
                (y, height),
                facecolors=colors.get(label, "#999999"),
                edgecolors="black",
                linewidth=0.4,
            )

    ax.set_xlabel(f"Time ({unit}) relative to first event")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_title("Thread timelines")
    margin = (max_end - min_start) * 0.02
    ax.set_xlim(0, (max_end - min_start + margin) / factor)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    caller_labels = [c for c in colors if c != "__idle__"]
    caller_labels.sort()
    legend_items = caller_labels[:max_legend]
    legend_handles = [Patch(facecolor=colors[c], edgecolor="black", label=c) for c in legend_items]
    if "__idle__" in colors:
        legend_handles.append(Patch(facecolor=colors["__idle__"], edgecolor="black", label="Idle"))
    if legend_handles:
        ncol = max(1, math.floor(len(legend_handles) / 12) + 1)
        ax.legend(handles=legend_handles, title="CallerID", bbox_to_anchor=(1.02, 1), loc="upper left", ncol=ncol)

    plt.tight_layout()
    if output:
        fig.savefig(output, dpi=200)
        print(f"Wrote {output}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    entries = load_entries(args.csv)
    segments, colors, min_start, max_end = build_segments(entries, include_idle=args.show_idle)
    plot_gantt(
        segments,
        colors,
        min_start,
        max_end,
        unit=args.unit,
        output=args.output,
        max_legend=args.max_legend,
    )


if __name__ == "__main__":
    main()
