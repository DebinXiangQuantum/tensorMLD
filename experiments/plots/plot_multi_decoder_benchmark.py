#!/usr/bin/env python3
"""Plot multi-decoder benchmark results in publication style."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from experiments.plots.paper_style import (
    DOUBLE_COLUMN_PT,
    DECODER_COLORS,
    apply_paper_style,
    decoder_display_name,
    save_figure,
)

DEFAULT_INPUT = WORKSPACE / "experiments" / "results" / "multi_decoder_benchmark.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "out" / "multi_decoder_benchmark"

DECODER_ORDER = ["mps_tn", "exact_tn", "bp", "bp_osd", "bp_lsd"]


def load_records(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON for benchmark, got {type(data)}")
    return [x for x in data if isinstance(x, dict)]


def _sorted_codes(records: list[dict[str, Any]]) -> list[str]:
    sortable = []
    for r in records:
        code = str(r.get("code", "unknown"))
        n_val = int(r.get("N", 10**9))
        sortable.append((n_val, code))
    unique = {}
    for n_val, code in sortable:
        unique[code] = min(n_val, unique.get(code, n_val))
    return [code for code, _ in sorted(unique.items(), key=lambda kv: (kv[1], kv[0]))]


def _build_metric_matrices(
    records: list[dict[str, Any]],
    codes: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    ler = np.full((len(DECODER_ORDER), len(codes)), np.nan, dtype=float)
    per_shot = np.full((len(DECODER_ORDER), len(codes)), np.nan, dtype=float)
    code_to_idx = {code: idx for idx, code in enumerate(codes)}

    for rec in records:
        code = str(rec.get("code", "unknown"))
        if code not in code_to_idx:
            continue
        j = code_to_idx[code]
        decoders = rec.get("decoders", {})
        if not isinstance(decoders, dict):
            continue

        for i, dec_name in enumerate(DECODER_ORDER):
            d = decoders.get(dec_name)
            if not isinstance(d, dict) or d.get("status") != "ok":
                continue
            if d.get("logical_error_rate") is not None:
                ler[i, j] = float(d["logical_error_rate"])
            if d.get("per_shot_ms") is not None:
                per_shot[i, j] = float(d["per_shot_ms"])

    return ler, per_shot


def _plot_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    codes: list[str],
    decoder_labels: list[str],
    title: str,
    cbar_label: str,
    cmap: str,
    use_log: bool,
    fmt: str,
) -> None:
    valid = matrix[np.isfinite(matrix)]
    if valid.size == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    masked = np.ma.masked_invalid(matrix)

    if use_log:
        pos = valid[valid > 0]
        if pos.size == 0:
            norm = None
        else:
            vmin = max(float(np.nanmin(pos)), 1e-8)
            vmax = float(np.nanmax(pos))
            if vmax <= vmin:
                vmax = vmin * 1.001
            norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    img = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(codes)))
    ax.set_xticklabels(codes, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(decoder_labels)))
    ax.set_yticklabels(decoder_labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text = "NA" if not np.isfinite(val) else format(val, fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=6)

    cbar = ax.figure.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(cbar_label)


def _plot_decoder_summary(
    ler: np.ndarray,
    per_shot: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    figsize = apply_paper_style(width_pt=DOUBLE_COLUMN_PT, ncols=2, nrows=1)
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    labels = [decoder_display_name(x) for x in DECODER_ORDER]
    x = np.arange(len(labels))

    with np.errstate(all="ignore"):
        avg_ler = np.nanmean(ler, axis=1)
        avg_lat = np.nanmean(per_shot, axis=1)

    colors = [DECODER_COLORS.get(x, "#4c72b0") for x in DECODER_ORDER]

    bars1 = axes[0].bar(x, np.clip(avg_ler, 1e-8, None), color=colors)
    axes[0].set_yscale("log")
    axes[0].set_title("Average LER across codes")
    axes[0].set_ylabel("Logical error rate")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    for bar, val in zip(bars1, avg_ler):
        if np.isfinite(val):
            axes[0].text(bar.get_x() + bar.get_width() / 2, max(val, 1e-8), f"{val:.2e}", ha="center", va="bottom", fontsize=6)

    bars2 = axes[1].bar(x, avg_lat, color=colors)
    axes[1].set_title("Average latency across codes")
    axes[1].set_ylabel("Per-shot latency (ms)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    for bar, val in zip(bars2, avg_lat):
        if np.isfinite(val):
            axes[1].text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    saved = save_figure(fig, output_dir, "multi_decoder_summary")
    plt.close(fig)
    return saved


def generate_plots(input_path: Path, output_dir: Path) -> list[Path]:
    records = load_records(input_path)
    codes = _sorted_codes(records)
    if not codes:
        raise ValueError("No code records found in benchmark JSON.")

    ler, per_shot = _build_metric_matrices(records, codes)
    decoder_labels = [decoder_display_name(x) for x in DECODER_ORDER]

    figsize = apply_paper_style(width_pt=DOUBLE_COLUMN_PT, ncols=2, nrows=1)
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    _plot_heatmap(
        axes[0],
        ler,
        codes=codes,
        decoder_labels=decoder_labels,
        title="Logical error rate by decoder/code",
        cbar_label="LER",
        cmap="YlOrRd",
        use_log=True,
        fmt=".1e",
    )
    _plot_heatmap(
        axes[1],
        per_shot,
        codes=codes,
        decoder_labels=decoder_labels,
        title="Per-shot latency by decoder/code",
        cbar_label="Latency (ms)",
        cmap="Blues",
        use_log=True,
        fmt=".3f",
    )

    saved = save_figure(fig, output_dir, "multi_decoder_heatmaps")
    plt.close(fig)

    saved.extend(_plot_decoder_summary(ler, per_shot, output_dir))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multi-decoder benchmark results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    saved = generate_plots(args.input, args.output_dir)
    for path in saved:
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
