from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for visualization. Install it with 'pip install matplotlib'."
    ) from exc


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate bar chart visualizations for experiment result CSV files. "
            "Each chart is saved next to its source CSV."
        )
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("experiments"),
        help="Root directory that contains experiment folders with results.csv files.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="results.png",
        help="Filename for the generated plot inside each experiment directory.",
    )
    return parser.parse_args()


def collect_experiment_csvs(root: Path) -> Iterable[Tuple[Path, Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Experiments root '{root}' does not exist.")

    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        csv_path = path / "results.csv"
        if csv_path.exists():
            yield path, csv_path


def read_results(csv_path: Path) -> List[Tuple[str, str, float]]:
    rows: List[Tuple[str, str, float]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"mode", "conf", "rep1", "rep2", "rep3"}
        if set(reader.fieldnames or []) < expected:
            missing = expected - set(reader.fieldnames or [])
            raise ValueError(f"{csv_path} missing column(s): {', '.join(sorted(missing))}")
        for row in reader:
            reps = [float(row[f"rep{i}"]) for i in (1, 2, 3)]
            avg = sum(reps) / len(reps)
            rows.append((row["mode"], row["conf"], avg))
    return rows


def prepare_series(data: List[Tuple[str, str, float]]) -> Tuple[List[str], List[str], Dict[str, List[float]]]:
    modes = sorted({mode for mode, _, _ in data})
    confs = sorted({conf for _, conf, _ in data})
    series: Dict[str, List[float]] = defaultdict(list)

    lookup: Dict[Tuple[str, str], float] = {(mode, conf): avg for mode, conf, avg in data}
    for mode in modes:
        for conf in confs:
            series[mode].append(lookup.get((mode, conf), math.nan))

    return modes, confs, series


def create_plot(
    modes: List[str],
    confs: List[str],
    series: Dict[str, List[float]],
    title: str,
    output_path: Path,
) -> None:
    if not modes or not confs:
        raise ValueError("No data available to plot.")

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = range(len(confs))
    bar_width = 0.8 / max(len(modes), 1)

    for idx, mode in enumerate(modes):
        offsets = [x + idx * bar_width for x in x_positions]
        bars = ax.bar(offsets, series[mode], width=bar_width, label=mode)
        for bar, value in zip(bars, series[mode]):
            if math.isnan(value):
                continue
            ax.annotate(
                f"{value:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(title)
    ax.set_xlabel("JVM Memory Configuration")
    ax.set_ylabel("Average Runtime [s]")
    ax.set_xticks([x + (len(modes) - 1) * bar_width / 2 for x in x_positions])
    ax.set_xticklabels(confs)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_arguments()
    experiment_count = 0

    for experiment_dir, csv_path in collect_experiment_csvs(args.experiments_root):
        data = read_results(csv_path)
        modes, confs, series = prepare_series(data)
        title = f"{experiment_dir.name} Results"
        output_path = experiment_dir / args.output_name
        create_plot(modes, confs, series, title, output_path)
        experiment_count += 1
        print(f"Saved plot to {output_path}")

    if experiment_count == 0:
        raise FileNotFoundError(
            f"No experiments with results.csv found under '{args.experiments_root}'."
        )


if __name__ == "__main__":
    main()
