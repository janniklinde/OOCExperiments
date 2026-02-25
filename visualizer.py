from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for visualization. Install it with 'pip install matplotlib'."
    ) from exc


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a publication-style runtime bar chart from a results.csv file. "
            "NaN values are rendered as full-height textured bars to indicate infinite runtime."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("experiments/lmcg/results.csv"),
        help="Path to a results.csv file.",
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
        default=None,
        help="Optional plot title. Default: '<experiment> Runtime'.",
    )
    return parser.parse_args()


def _as_float(raw_value: str) -> float:
    value = raw_value.strip()
    if value == "" or value.lower() == "nan":
        return math.nan
    return float(value)


def read_results(csv_path: Path) -> List[Tuple[str, str, float]]:
    rows: List[Tuple[str, str, float]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"mode", "conf", "rep1"}
        if set(reader.fieldnames or []) < expected:
            missing = expected - set(reader.fieldnames or [])
            raise ValueError(f"{csv_path} missing column(s): {', '.join(sorted(missing))}")

        for row in reader:
            rows.append((row["mode"], row["conf"], _as_float(row["rep1"])))

    return rows


def ordered_unique(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _memory_conf_bytes(conf: str) -> Optional[float]:
    units = {
        "": 1.0,
        "k": 1024.0,
        "m": 1024.0**2,
        "g": 1024.0**3,
        "t": 1024.0**4,
        "p": 1024.0**5,
    }

    # Prefer explicit JVM heap config like -Xmx4g when present.
    xmx_match = re.search(r"-xmx\s*(\d+(?:\.\d+)?)\s*([kmgtp]?)(?:i?b)?", conf, flags=re.IGNORECASE)
    match = xmx_match or re.search(
        r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*([kmgtp]?)(?:i?b)?(?![a-z0-9])",
        conf,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2).lower()
    return value * units[unit]


def sort_jvm_memory_confs(confs: List[str]) -> List[str]:
    indexed = list(enumerate(confs))

    def conf_key(item: Tuple[int, str]) -> Tuple[int, float, int]:
        idx, conf = item
        bytes_value = _memory_conf_bytes(conf)
        if bytes_value is None:
            return (1, 0.0, idx)
        return (0, bytes_value, idx)

    return [conf for _, conf in sorted(indexed, key=conf_key)]


def prepare_series(data: List[Tuple[str, str, float]]) -> Tuple[List[str], List[str], Dict[str, List[float]]]:
    modes = ordered_unique([mode for mode, _, _ in data])
    confs = sort_jvm_memory_confs(ordered_unique([conf for _, conf, _ in data]))

    lookup: Dict[Tuple[str, str], float] = {}
    for mode, conf, value in data:
        key = (mode, conf)
        if key in lookup:
            raise ValueError(f"Duplicate row detected for mode={mode}, conf={conf}")
        lookup[key] = value

    series: Dict[str, List[float]] = {}
    for mode in modes:
        series[mode] = [lookup.get((mode, conf), math.nan) for conf in confs]

    return modes, confs, series


def apply_scientific_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "hatch.linewidth": 0.35,
        }
    )


def create_plot(
    modes: List[str],
    confs: List[str],
    series: Dict[str, List[float]],
    title: str,
    output_stem: Path,
) -> List[Path]:
    if not modes or not confs:
        raise ValueError("No data available to plot.")

    finite_values = [value for mode in modes for value in series[mode] if math.isfinite(value)]
    finite_max = max(finite_values) if finite_values else 1.0
    y_limit = finite_max * 1.15 if finite_max > 0 else 1.0
    infinite_bar_height = y_limit * 2.0

    # Okabe-Ito style palette (colorblind-friendly), combined with hatches for grayscale print.
    mode_palette = [
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#D55E00",
        "#CC79A7",
        "#56B4E9",
        "#F0E442",
        "#000000",
    ]
    failed_hatch = "////"

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x_positions = list(range(len(confs)))
    bar_width = 0.8 / max(len(modes), 1)

    for idx, mode in enumerate(modes):
        offsets = [x - 0.4 + bar_width / 2.0 + idx * bar_width for x in x_positions]
        finite_x: List[float] = []
        finite_y: List[float] = []
        mode_color = mode_palette[idx % len(mode_palette)]

        for x_pos, value in zip(offsets, series[mode]):
            if math.isfinite(value):
                finite_x.append(x_pos)
                finite_y.append(value)
                continue

            ax.bar(
                x_pos,
                infinite_bar_height,
                width=bar_width,
                facecolor="none",
                edgecolor=mode_color,
                linewidth=0.8,
                hatch=failed_hatch,
                zorder=2,
            )

        if finite_x:
            ax.bar(
                finite_x,
                finite_y,
                width=bar_width,
                color=mode_color,
                edgecolor="#2e2e2e",
                linewidth=0.6,
                label=mode.upper(),
                zorder=3,
            )
        else:
            ax.bar(
                [],
                [],
                color=mode_color,
                edgecolor="#2e2e2e",
                label=mode.upper(),
            )

    ax.set_title(title)
    ax.set_xlabel("JVM Memory Configuration")
    ax.set_ylabel("Runtime [s]")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(confs)
    ax.set_ylim(0, y_limit)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.grid(axis="x", visible=False)

    ax.legend(frameon=False, ncol=2, loc="upper right")

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
    modes, confs, series = prepare_series(data)
    title = args.title or f"{args.csv.parent.name} Runtime"
    output_stem = (args.output.with_suffix("") if args.output else (args.csv.parent / "results"))
    outputs = create_plot(modes, confs, series, title, output_stem)
    for path in outputs:
        print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
