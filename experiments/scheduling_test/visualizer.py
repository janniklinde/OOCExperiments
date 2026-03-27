from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for visualization. Install it with 'pip install matplotlib'."
    ) from exc


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a publication-style runtime bar chart for scheduling strategy "
            "comparisons across matrix setups."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).with_name("results.csv"),
        help="Path to the scheduling results CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path stem (without extension). Default: next to CSV as results",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Scheduling Strategy Runtime by Matrix Setup",
        help="Plot title.",
    )
    return parser.parse_args()


def ordered_unique(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def read_results(csv_path: Path) -> List[Tuple[str, str, float]]:
    rows: List[Tuple[str, str, float]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"Setup", "ExecMode", "Runtime [s]"}
        if set(reader.fieldnames or []) < expected:
            missing = expected - set(reader.fieldnames or [])
            raise ValueError(f"{csv_path} missing column(s): {', '.join(sorted(missing))}")

        for row in reader:
            rows.append((row["ExecMode"], row["Setup"], float(row["Runtime [s]"])))

    return rows


def prepare_series(data: List[Tuple[str, str, float]]) -> Tuple[List[str], List[str], Dict[str, List[float]]]:
    modes = ordered_unique([mode for mode, _, _ in data])
    setups = ordered_unique([setup for _, setup, _ in data])

    lookup: Dict[Tuple[str, str], float] = {}
    for mode, setup, value in data:
        key = (mode, setup)
        if key in lookup:
            raise ValueError(f"Duplicate row detected for mode={mode}, setup={setup}")
        lookup[key] = value

    series: Dict[str, List[float]] = {}
    for mode in modes:
        series[mode] = [lookup[(mode, setup)] for setup in setups]

    return modes, setups, series


def apply_scientific_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "axes.titleweight": "semibold",
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def format_setup_label(setup: str) -> str:
    parts = [part for part in setup.split("_") if part]
    if not parts:
        return setup

    operation = parts[0].upper()
    dims: List[str] = []
    for part in parts[1:]:
        if part == "x":
            continue
        dims.append(part[:-1] if part.endswith("x") else part)

    if not dims:
        return operation
    return f"{operation}\n" + " x ".join(dims)


def create_plot(
    modes: List[str],
    setups: List[str],
    series: Dict[str, List[float]],
    title: str,
    output_stem: Path,
) -> List[Path]:
    if not modes or not setups:
        raise ValueError("No data available to plot.")

    finite_values = [value for mode in modes for value in series[mode]]
    y_limit = max(finite_values) * 1.18 if finite_values else 1.0

    mode_palette = [
        "#CDA2BE",
        "#E4C890",
        "#CCFFFF",
        "#FFD9CC",
    ]

    fig_width = max(6.2, 1.8 * len(setups) + 1.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    x_positions = list(range(len(setups)))
    bar_width = 0.8 / len(modes)

    for idx, mode in enumerate(modes):
        offsets = [x - 0.4 + bar_width / 2.0 + idx * bar_width for x in x_positions]
        ax.bar(
            offsets,
            series[mode],
            width=bar_width,
            color=mode_palette[idx % len(mode_palette)],
            edgecolor="#2E2E2E",
            linewidth=0.6,
            label=mode,
            zorder=3,
        )

    ax.set_title(title)
    ax.set_xlabel("Matrix Setup")
    ax.set_ylabel("Runtime [s]")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([format_setup_label(setup) for setup in setups])
    ax.set_ylim(0, y_limit)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output_stem.with_suffix(".pdf")
    output_png = output_stem.with_suffix(".png")
    fig.savefig(output_pdf, dpi=300)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    return [output_pdf, output_png]


def main() -> None:
    args = parse_arguments()
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file '{args.csv}' does not exist.")

    apply_scientific_style()
    data = read_results(args.csv)
    modes, setups, series = prepare_series(data)
    output_stem = args.output.with_suffix("") if args.output else args.csv.parent / "results"
    outputs = create_plot(modes, setups, series, args.title, output_stem)
    for path in outputs:
        print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
