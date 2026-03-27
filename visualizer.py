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
            "NaN values are rendered as compact failure markers near the runtime axis."
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
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "axes.titleweight": "semibold",
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "hatch.linewidth": 0.35,
        }
    )


def format_experiment_name(name: str) -> str:
    token_map = {
        "pca": "PCA",
        "lmcg": "LMCG",
        "kmeans": "K-MEANS",
        "warm": "Warm",
    }
    parts = [part for part in name.split("_") if part]
    if not parts:
        return name
    return " ".join(token_map.get(part.lower(), part.upper()) for part in parts)


def _format_dimension(value: str) -> str:
    number = int(value)
    suffixes = [
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "k"),
    ]
    for divisor, suffix in suffixes:
        if number >= divisor and number % divisor == 0:
            return f"{number // divisor}{suffix}"
    return str(number)


def _infer_dimensions_from_run_script(run_script_path: Path) -> Optional[str]:
    if not run_script_path.exists():
        return None

    text = run_script_path.read_text()
    match = re.search(r"run_args=\(\s*(\d+)\s+(\d+)\s+", text)
    if not match:
        return None

    rows = _format_dimension(match.group(1))
    cols = _format_dimension(match.group(2))
    return f"{rows} x {cols}"


def _infer_dimensions_from_blob_setup(experiment_dir: Path) -> Optional[str]:
    exp_path = experiment_dir / "exp.dml"
    setup_path = Path("setup.sh")
    if not exp_path.exists() or not setup_path.exists():
        return None

    exp_text = exp_path.read_text()
    blob_match = re.search(r'read\("(?P<path>\.\./\.\./data/blobs/[^"]+)_X"\)', exp_text)
    if not blob_match:
        return None

    blob_base = blob_match.group("path").replace("../../", "")
    setup_text = setup_path.read_text()
    setup_match = re.search(
        rf'blob_base="{re.escape(blob_base)}".*?-args\s+(\d+)\s+(\d+)\s+\d+\s+[-+0-9.]+\s+[-+0-9.]+\s+\d+\s+"\$blob_base"',
        setup_text,
        flags=re.DOTALL,
    )
    if not setup_match:
        return None

    rows = _format_dimension(setup_match.group(1))
    cols = _format_dimension(setup_match.group(2))
    return f"{rows} x {cols}"


def infer_experiment_dimensions(experiment_dir: Path) -> Optional[str]:
    return _infer_dimensions_from_run_script(experiment_dir / "run.sh") or _infer_dimensions_from_blob_setup(
        experiment_dir
    )


def build_plot_title(csv_path: Path, explicit_title: Optional[str]) -> str:
    if explicit_title:
        return explicit_title

    experiment_name = format_experiment_name(csv_path.parent.name)
    dimensions = infer_experiment_dimensions(csv_path.parent)
    if dimensions:
        return f"{experiment_name} Runtime ({dimensions})"
    return f"{experiment_name} Runtime"


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
    y_limit = finite_max * 1.22 if finite_max > 0 else 1.0
    failed_marker_y = y_limit * 0.035

    # Soft pastel palette, combined with hatches for grayscale print.
    mode_palette = [
        "#CDA2BE",
        "#E4C890",
        "#CCFFFF",
        "#FFD9CC",
        "#CDA2BE",
        "#E4C890",
        "#CCFFFF",
        "#FFD9CC",
    ]
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
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

            ax.plot(
                [x_pos],
                [failed_marker_y],
                linestyle="none",
                marker="x",
                markersize=14,
                markeredgewidth=4.2,
                color="black",
                zorder=4,
            )
            ax.plot(
                [x_pos],
                [failed_marker_y],
                linestyle="none",
                marker="x",
                markersize=12,
                markeredgewidth=2.6,
                color=mode_color,
                zorder=5,
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
    title = build_plot_title(args.csv, args.title)
    output_stem = (args.output.with_suffix("") if args.output else (args.csv.parent / "results"))
    outputs = create_plot(modes, confs, series, title, output_stem)
    for path in outputs:
        print(f"Saved plot to {path}")


if __name__ == "__main__":
    main()
