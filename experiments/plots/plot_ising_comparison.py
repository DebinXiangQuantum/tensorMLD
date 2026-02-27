#!/usr/bin/env python3
"""Plot Ising decoder vs BP comparison: LER curves and latency bar chart.

Reads benchmark JSON from test_ising_decoder.py and produces:
  1. LER vs physical error rate (one subplot per code, lines per decoder)
  2. Latency bar chart at p=0.01 (QUBO-SA uses hardware latency = cycles*4ns)

Usage:
  python experiments/plots/plot_ising_comparison.py
  python experiments/plots/plot_ising_comparison.py --input experiments/results/ising_vs_bp_v3.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from experiments.plots.paper_style import (
    DOUBLE_COLUMN_PT,
    apply_paper_style,
    save_figure,
)

DEFAULT_INPUT = WORKSPACE / "experiments" / "results" / "ising_vs_bp_v3.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "out" / "ising_comparison"

# Decoder display config
DECODER_ORDER = ["qubo_sa", "bp", "bp_osd", "bp_lsd"]
DECODER_DISPLAY = {
    "qubo_sa": "QUBO-SA",
    "bp": "BP",
    "bp_osd": "BP-OSD",
    "bp_lsd": "BP-LSD",
}
DECODER_COLORS = {
    "qubo_sa": "#e41a1c",
    "bp": "#377eb8",
    "bp_osd": "#4daf4a",
    "bp_lsd": "#984ea3",
}
DECODER_MARKERS = {
    "qubo_sa": "s",
    "bp": "o",
    "bp_osd": "^",
    "bp_lsd": "D",
}


def load_records(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data)}")
    return [x for x in data if isinstance(x, dict)]


def _sorted_codes(records: list[dict]) -> list[str]:
    """Get unique code names sorted by N."""
    code_n = {}
    for r in records:
        code = str(r.get("code", ""))
        n_val = int(r.get("N", 10**9))
        if code not in code_n or n_val < code_n[code]:
            code_n[code] = n_val
    return [c for c, _ in sorted(code_n.items(), key=lambda kv: (kv[1], kv[0]))]


def _build_ler_data(
    records: list[dict], codes: list[str]
) -> dict[str, dict[str, dict[float, float]]]:
    """Build {code: {decoder: {p: ler}}} mapping."""
    data: dict[str, dict[str, dict[float, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for rec in records:
        code = str(rec.get("code", ""))
        if code not in codes:
            continue
        p = float(rec.get("noise_p", 0))
        decoders = rec.get("decoders", {})
        for dec_name, dec_info in decoders.items():
            if not isinstance(dec_info, dict) or dec_info.get("status") != "ok":
                continue
            ler = dec_info.get("logical_error_rate")
            if ler is not None:
                data[code][dec_name][p] = float(ler)
    return data


def plot_ler_curves(
    records: list[dict], codes: list[str], output_dir: Path
) -> list[Path]:
    """Plot LER vs p subplots, one per code."""
    ncols = min(3, len(codes))
    nrows = (len(codes) + ncols - 1) // ncols
    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT, ncols=ncols, nrows=nrows,
        panel_aspect=1.0,
    )
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, constrained_layout=True, squeeze=False,
    )

    ler_data = _build_ler_data(records, codes)

    for idx, code in enumerate(codes):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        code_data = ler_data.get(code, {})
        for dec_name in DECODER_ORDER:
            if dec_name not in code_data:
                continue
            p_ler = code_data[dec_name]
            ps = sorted(p_ler.keys())
            lers = [p_ler[p] for p in ps]

            # Replace 0 LER with small value for log scale
            lers_plot = [max(v, 1e-7) for v in lers]

            ax.plot(
                ps, lers_plot,
                marker=DECODER_MARKERS.get(dec_name, "o"),
                color=DECODER_COLORS.get(dec_name, "#333333"),
                label=DECODER_DISPLAY.get(dec_name, dec_name),
                markersize=4,
            )

        # Find N, K, D for title
        for rec in records:
            if rec.get("code") == code:
                n_val = rec.get("N", "?")
                k_val = rec.get("K", "?")
                d_val = rec.get("D", "?")
                ax.set_title(f"{code}\n[[{n_val},{k_val},{d_val}]]")
                break

        ax.set_yscale("log")
        ax.set_xlabel("Physical error rate $p$")
        ax.set_ylabel("Logical error rate")
        ax.legend(fontsize=6, loc="lower right")

    # Hide unused axes
    for idx in range(len(codes), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    saved = save_figure(fig, output_dir, "ising_ler_vs_p")
    plt.close(fig)
    return saved


def plot_latency_bars(
    records: list[dict], codes: list[str], output_dir: Path,
    target_p: float = 0.01,
) -> list[Path]:
    """Plot latency bar chart at a specific p value.

    QUBO-SA uses hardware latency (cycles * cycle_time_ns) from ising_machine metrics.
    BP decoders use software wall-clock latency (per_shot_ms converted to ns).
    """
    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT, ncols=1, nrows=1, panel_aspect=2.0,
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    # Collect latency per code per decoder at target_p
    latency_data: dict[str, dict[str, float]] = defaultdict(dict)
    for rec in records:
        code = str(rec.get("code", ""))
        if code not in codes:
            continue
        p = float(rec.get("noise_p", 0))
        if abs(p - target_p) > 1e-6:
            continue
        decoders = rec.get("decoders", {})
        hw = rec.get("ising_machine", {})

        for dec_name in DECODER_ORDER:
            dec_info = decoders.get(dec_name, {})
            if not isinstance(dec_info, dict) or dec_info.get("status") != "ok":
                continue

            if dec_name == "qubo_sa":
                # Use hardware latency: total_cycles * cycle_time_ns
                hw_latency_ns = hw.get("hardware_latency_ns", 0)
                latency_data[code][dec_name] = hw_latency_ns  # in ns
            else:
                # Use software latency: per_shot_ms -> ns
                ms = dec_info.get("per_shot_ms", 0)
                latency_data[code][dec_name] = ms * 1e6  # ms -> ns

    if not latency_data:
        ax.text(0.5, 0.5, f"No data at p={target_p}", ha="center", va="center",
                transform=ax.transAxes)
        saved = save_figure(fig, output_dir, "ising_latency_bars")
        plt.close(fig)
        return saved

    # Build grouped bar chart
    n_codes = len(codes)
    n_decoders = len(DECODER_ORDER)
    bar_width = 0.8 / n_decoders
    x = np.arange(n_codes)

    for i, dec_name in enumerate(DECODER_ORDER):
        vals = []
        for code in codes:
            v = latency_data.get(code, {}).get(dec_name, 0)
            vals.append(v)
        bars = ax.bar(
            x + i * bar_width - 0.4 + bar_width / 2,
            vals,
            bar_width * 0.9,
            label=DECODER_DISPLAY.get(dec_name, dec_name),
            color=DECODER_COLORS.get(dec_name, "#333333"),
        )
        # Add value labels
        for bar, val in zip(bars, vals):
            if val > 0:
                if val < 1e3:
                    label = f"{val:.0f}ns"
                elif val < 1e6:
                    label = f"{val/1e3:.1f}us"
                else:
                    label = f"{val/1e6:.2f}ms"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    label,
                    ha="center", va="bottom", fontsize=5, rotation=45,
                )

    ax.set_yscale("log")
    ax.set_ylabel("Decode latency (ns)")
    ax.set_title(f"Decode latency at p={target_p}")
    ax.set_xticks(x)
    ax.set_xticklabels(codes, rotation=35, ha="right")
    ax.legend(fontsize=6, loc="upper left")

    saved = save_figure(fig, output_dir, "ising_latency_bars")
    plt.close(fig)
    return saved


def generate_all_plots(input_path: Path, output_dir: Path) -> list[Path]:
    records = load_records(input_path)
    codes = _sorted_codes(records)
    if not codes:
        raise ValueError("No code records found.")

    all_saved: list[Path] = []
    all_saved.extend(plot_ler_curves(records, codes, output_dir))
    all_saved.extend(plot_latency_bars(records, codes, output_dir))
    return all_saved


def main():
    parser = argparse.ArgumentParser(
        description="Plot Ising decoder vs BP comparison."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    saved = generate_all_plots(args.input, args.output_dir)
    for path in saved:
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
