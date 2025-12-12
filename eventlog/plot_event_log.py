#!/usr/bin/env python3
"""
Visualize compute vs idle time per thread as a Gantt-style chart, with disk IO
tracks aligned to the same timeline.

Input CSV columns (case-sensitive):
    ThreadID, CallerID, StartNanos, EndNanos
Disk read/write CSVs may include an extra NumBytes column, which is ignored here.

Usage examples:
    python plot_event_log.py                  # uses Compute/Disk*EventLog.csv automatically
    python plot_event_log.py --output timeline.png --unit us
    python plot_event_log.py CustomCompute.csv --disk-read-csv CustomRead.csv --disk-write-csv CustomWrite.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


TimeEntry = Tuple[int, str, int, int]  # thread, caller, start_ns, end_ns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot compute/idle timeline per thread.")
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=Path("ComputeEventLog.csv"),
        help="Path to ComputeEventLog.csv (default: ComputeEventLog.csv)",
    )
    parser.add_argument(
        "--disk-read-csv",
        type=Path,
        default=Path("DiskReadEventLog.csv"),
        help="Path to DiskReadEventLog.csv (default: DiskReadEventLog.csv).",
    )
    parser.add_argument(
        "--disk-write-csv",
        type=Path,
        default=Path("DiskWriteEventLog.csv"),
        help="Path to DiskWriteEventLog.csv (default: DiskWriteEventLog.csv).",
    )
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
    return parser.parse_args()


def load_entries(path: Path, allow_empty: bool = False) -> List[TimeEntry]:
    entries: List[TimeEntry] = []
    if not path.exists():
        if allow_empty:
            print(f"Warning: {path} not found; skipping.")
            return entries
        raise FileNotFoundError(f"{path} does not exist.")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            if allow_empty:
                print(f"Warning: {path} has no data; skipping.")
                return entries
            raise ValueError(f"{path} is empty or missing a header.")
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
        if allow_empty:
            print(f"Warning: {path} has no events; skipping.")
            return entries
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


def find_bounds(entries: List[TimeEntry]) -> Optional[Tuple[int, int]]:
    if not entries:
        return None
    return min(e[2] for e in entries), max(e[3] for e in entries)


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


def plot_disk_track(
    ax: plt.Axes,
    entries: List[TimeEntry],
    min_start: float,
    unit: str,
    label: str,
    color: str,
) -> None:
    factor = unit_factor(unit)
    by_thread: Dict[int, List[TimeEntry]] = defaultdict(list)
    for ent in entries:
        by_thread[ent[0]].append(ent)

    y_ticks: List[float] = []
    y_labels: List[str] = []
    y_base = 3
    height = 3
    gap = 2

    if by_thread:
        for idx, (thread_id, thread_entries) in enumerate(
            sorted(by_thread.items(), key=lambda kv: kv[0])
        ):
            y = idx * (height + gap) + y_base
            y_ticks.append(y + height / 2)
            y_labels.append(str(thread_id))
            for _, _, start, end in sorted(thread_entries, key=lambda e: e[2]):
                duration = end - start
                if duration <= 0:
                    continue
                ax.broken_barh(
                    [((start - min_start) / factor, duration / factor)],
                    (y, height),
                    facecolors=color,
                    edgecolors="none",
                    linewidth=0.0,
                )
    else:
        ax.text(
            0.5,
            0.5,
            "No events",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="#555555",
        )

    if by_thread:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
    ax.set_ylabel(label, rotation=0, labelpad=30, fontsize=9, ha="center", va="center")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def plot_compute_axis(
    ax: plt.Axes,
    segments: Dict[int, List[Tuple[float, float, str]]],
    colors: Dict[str, str],
    min_start: float,
    max_end: float,
    unit: str,
) -> None:
    factor = unit_factor(unit)
    y_ticks = []
    y_labels = []
    y_base = 6
    height = 4
    gap = 3

    for idx, (thread_id, segs) in enumerate(sorted(segments.items())):
        y = idx * (height + gap) + y_base
        y_ticks.append(y + height / 2)
        y_labels.append(str(thread_id))
        for start_ns, dur_ns, label in segs:
            ax.broken_barh(
                [((start_ns - min_start) / factor, dur_ns / factor)],
                (y, height),
                facecolors=colors.get(label, "#999999"),
                edgecolors="none",
                linewidth=0.0,
            )

    ax.set_xlabel(f"Time ({unit}) relative to first event")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_title("Thread timelines")
    span = max_end - min_start
    margin = span * 0.02 if span > 0 else 1.0
    ax.set_xlim(0, (span + margin) / factor)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    caller_labels = [c for c in colors if c != "__idle__"]
    caller_labels.sort()
    if caller_labels:
        handles = [
            Patch(facecolor=colors[c], edgecolor="none", label=c) for c in caller_labels
        ]
        ax.legend(
            handles=handles,
            title="CallerID",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=4,
            frameon=False,
        )


def plot_timeline(
    segments: Dict[int, List[Tuple[float, float, str]]],
    colors: Dict[str, str],
    disk_reads: List[TimeEntry],
    disk_writes: List[TimeEntry],
    min_start: float,
    max_end: float,
    unit: str,
    output: Optional[Path],
) -> None:
    disk_tracks = [
        ("Disk Reads", disk_reads, "#1f77b4"),
        ("Disk Writes", disk_writes, "#ff7f0e"),
    ]
    compute_ratio = max(3.0, 0.7 * len(segments))

    def disk_ratio(entries: List[TimeEntry]) -> float:
        threads = {e[0] for e in entries}
        return max(1.5, 0.6 * len(threads) if threads else 1.5)

    height_ratios = [disk_ratio(disk_reads), disk_ratio(disk_writes), compute_ratio]
    fig_height = max(5, 1.2 * sum(height_ratios))

    fig, axes = plt.subplots(
        nrows=len(disk_tracks) + 1,
        ncols=1,
        sharex=True,
        figsize=(12, fig_height),
        gridspec_kw={"height_ratios": height_ratios},
    )
    try:
        axes_list = list(axes)
    except TypeError:
        axes_list = [axes]

    for idx, (label, entries, color) in enumerate(disk_tracks):
        plot_disk_track(axes_list[idx], entries, min_start, unit, label, color)
        axes_list[idx].tick_params(labelbottom=False)

    compute_ax = axes_list[-1]
    plot_compute_axis(
        compute_ax,
        segments,
        colors,
        min_start,
        max_end,
        unit,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if output:
        fig.savefig(output, dpi=200)
        print(f"Wrote {output}")
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    compute_entries = load_entries(args.csv)
    disk_read_entries = load_entries(args.disk_read_csv, allow_empty=True)
    disk_write_entries = load_entries(args.disk_write_csv, allow_empty=True)

    segments, colors, compute_min, compute_max = build_segments(
        compute_entries, include_idle=args.show_idle
    )

    bounds = [(compute_min, compute_max)]
    for extra_bounds in (
        find_bounds(disk_read_entries),
        find_bounds(disk_write_entries),
    ):
        if extra_bounds:
            bounds.append(extra_bounds)

    min_start = min(b[0] for b in bounds)
    max_end = max(b[1] for b in bounds)

    plot_timeline(
        segments,
        colors,
        disk_read_entries,
        disk_write_entries,
        min_start,
        max_end,
        unit=args.unit,
        output=args.output,
    )


if __name__ == "__main__":
    main()
