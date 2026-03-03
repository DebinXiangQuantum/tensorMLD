#!/usr/bin/env python3
"""Plot decomposition ablation study results in publication style.

Supports two input formats:
  1. Cost estimation JSON (from estimate_ablation_cost.py):
     List of per-code dicts with nested method sub-dicts.
  2. Full ablation JSON (from test_ablation_decomposition.py):
     Flat list of per-method-per-code records.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from experiments.plots.paper_style import (
    DOUBLE_COLUMN_PT,
    SINGLE_COLUMN_PT,
    METHOD_COLORS,
    apply_paper_style,
    method_display_name,
    save_figure,
)

DEFAULT_ESTIMATE_INPUT = WORKSPACE / "experiments" / "results" / "ablation_cost_estimates.json"
DEFAULT_ABLATION_INPUT = WORKSPACE / "experiments" / "results" / "ablation_decomposition_results.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "out" / "ablation_decomposition"
METHOD_ORDER = ["naive_direct", "quimb_compress", "our_mps"]
METHOD_KEYS = {"naive": "naive_direct", "quimb_compress": "quimb_compress", "our_mps": "our_mps"}


# ---------------------------------------------------------------------------
# Detect and load input format
# ---------------------------------------------------------------------------
def _is_estimate_format(data: list[dict]) -> bool:
    """Check if data is from estimate_ablation_cost.py (nested per-code dicts)."""
    if not data:
        return False
    first = data[0]
    return "naive" in first or "quimb_compress" in first or "our_mps" in first


def load_estimate_data(path: Path) -> tuple[list[str], dict[str, dict[str, dict]]]:
    """Load cost estimation JSON. Returns (codes, {code: {method: metrics}})."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    codes_with_n = []
    result: dict[str, dict[str, dict]] = {}
    for entry in raw:
        code = entry["code"]
        n = entry.get("N", 0)
        codes_with_n.append((code, n))
        result[code] = {}
        for key, method_name in METHOD_KEYS.items():
            if key in entry:
                result[code][method_name] = entry[key]
    # Sort by N
    codes_with_n.sort(key=lambda x: (x[1], x[0]))
    codes = [c for c, _ in codes_with_n]
    return codes, result


def load_ablation_data(path: Path) -> tuple[list[str], dict[str, dict[str, dict]]]:
    """Load flat ablation JSON. Returns (codes, {code: {method: metrics}})."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    code_n: dict[str, int] = {}
    result: dict[str, dict[str, dict]] = {}
    for r in raw:
        code = str(r.get("code", "unknown"))
        method = str(r.get("method", ""))
        n = int(r.get("N", 0))
        code_n[code] = min(code_n.get(code, n), n)
        if code not in result:
            result[code] = {}
        result[code][method] = r

    codes = [k for k, _ in sorted(code_n.items(), key=lambda kv: (kv[1], kv[0]))]
    return codes, result


# ---------------------------------------------------------------------------
# Extract metrics from either format
# ---------------------------------------------------------------------------
def _get_flops(code_data: dict[str, dict], method: str) -> float:
    """Get estimated FLOPs per shot."""
    d = code_data.get(method, {})
    if method == "our_mps":
        return float(d.get("est_online_flops_per_shot", np.nan))
    # Estimate format
    if "est_flops" in d:
        return float(d["est_flops"])
    # Ablation format
    return float(d.get("est_flops_per_shot", np.nan))


def _get_peak_memory_gb(code_data: dict[str, dict], method: str) -> float:
    """Get estimated peak memory in GB."""
    d = code_data.get(method, {})
    if method == "our_mps":
        mem_bytes = float(d.get("est_online_peak_memory_bytes", np.nan))
        return mem_bytes / 1e9
    # Estimate format
    if "est_peak_memory_bytes" in d:
        return float(d["est_peak_memory_bytes"]) / 1e9
    # Ablation format
    if "actual_peak_memory_bytes" in d:
        return float(d["actual_peak_memory_bytes"]) / 1e9
    return np.nan


def _get_storage_mb(code_data: dict[str, dict], method: str) -> float:
    """Get TN/MPS storage in MB."""
    d = code_data.get(method, {})
    if method == "our_mps":
        return float(d.get("est_mps_storage_MB", d.get("est_mps_storage_bytes", 0) / 1e6))
    if "storage_bytes" in d:
        return float(d["storage_bytes"]) / 1e6
    return np.nan


def _get_contraction_width(code_data: dict[str, dict], method: str) -> float:
    """Get contraction width (log2 of peak intermediate)."""
    d = code_data.get(method, {})
    if method == "our_mps":
        # MPS online: peak is chi x chi matrix, log2(chi^2) ~ 2*log2(chi)
        chi = d.get("chi", 32)
        return round(math.log2(chi * chi * 2), 2)  # chi x 2 x chi
    if "est_peak_size_log2" in d:
        return float(d["est_peak_size_log2"])
    if "contraction_width" in d:
        return float(d["contraction_width"])
    return np.nan


def _build_matrix(codes: list[str], code_data: dict[str, dict[str, dict]],
                  metric_fn) -> np.ndarray:
    """Build (num_methods x num_codes) matrix using metric_fn."""
    out = np.full((len(METHOD_ORDER), len(codes)), np.nan, dtype=float)
    for j, code in enumerate(codes):
        if code not in code_data:
            continue
        cd = code_data[code]
        for i, method in enumerate(METHOD_ORDER):
            out[i, j] = metric_fn(cd, method)
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _short_code_label(code: str) -> str:
    """Shorten code names for x-axis labels."""
    # e.g. "QCC_144_12_12" -> "QCC\n144"
    parts = code.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}\n[[{parts[1]}]]"
    return code


def _plot_grouped_bars(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    codes: list[str],
    ylabel: str,
    log_scale: bool = True,
    show_values: bool = False,
) -> None:
    x = np.arange(len(codes))
    width = 0.22
    sep =0.25
    labels = [_short_code_label(c) for c in codes]

    for i, method in enumerate(METHOD_ORDER):
        y = matrix[i]
        offset = (i - (len(METHOD_ORDER) - 1) / 2) * sep
        mask = ~np.isnan(y) & (y > 0)
        y_plot = np.where(mask, y, np.nan)
        bars = ax.bar(
            x + offset,
            y_plot,
            width,
            label=method_display_name(method),
            color=METHOD_COLORS.get(method, "#4c72b0"),
            alpha=0.92,
            edgecolor="k",
            linewidth=0.3,
        )
        if show_values:
            for bar, val in zip(bars, y):
                if not np.isnan(val) and val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.0e}", ha="center", va="bottom", fontsize=5,
                            rotation=90)

    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlim(x[0]-0.5,x[-1]+0.5)
    ax.tick_params(axis='x', rotation=0)


def generate_estimate_plots(input_path: Path, output_dir: Path) -> list[Path]:
    """Generate plots from cost estimation data."""
    codes, code_data = load_estimate_data(input_path)
    if not codes:
        raise ValueError("No codes found in estimation JSON.")

    flops = _build_matrix(codes, code_data, _get_flops)
    peak_gb = _build_matrix(codes, code_data, _get_peak_memory_gb)
    width_log2 = _build_matrix(codes, code_data, _get_contraction_width)
    storage = _build_matrix(codes, code_data, _get_storage_mb)

    # --- Figure 1: Main comparison (FLOPs + Peak Memory) ---
    apply_paper_style(width_pt=SINGLE_COLUMN_PT, ncols=2, nrows=1)
    w_in = SINGLE_COLUMN_PT / 72.27
    fig, axes = plt.subplots(2,1, figsize=(w_in, w_in * 0.8))
    fig.subplots_adjust(left=0.0, right=0.97, hspace=0.1, top=0.78, bottom=0.20)
    _plot_grouped_bars(axes[0], flops, codes=codes,
                       ylabel="FLOPs", log_scale=True)
    # axes[0].set_title("(a) Computational Cost", fontsize=8, fontweight="bold")

    _plot_grouped_bars(axes[1], peak_gb, codes=codes,
                       ylabel="Peak Memory (GB)", log_scale=True)
    # axes[1].set_title("(b) Memory Requirement", fontsize=8, fontweight="bold")

    # Shared legend above all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 0.9), fontsize=7,
                   columnspacing=1.5, handletextpad=0.5)
    saved = save_figure(fig, output_dir, "ablation_cost_comparison")

    plt.close(fig)

    # --- Figure 2: Contraction Width + Storage ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(w_in, w_in * 0.38))
    fig2.subplots_adjust(left=0.0, right=0.98, wspace=0.30, top=0.78, bottom=0.20)

    _plot_grouped_bars(axes2[0], width_log2, codes=codes,
                       ylabel="Contraction Width (log$_2$)", log_scale=False)
    axes2[0].set_title("(a) Contraction Width", fontsize=8, fontweight="bold")

    _plot_grouped_bars(axes2[1], storage, codes=codes,
                       ylabel="TN/MPS Storage (MB)", log_scale=True)
    axes2[1].set_title("(b) Storage Size", fontsize=8, fontweight="bold")

    handles2, labels2 = axes2[0].get_legend_handles_labels()
    if handles2:
        fig2.legend(handles2, labels2, loc="upper center", ncol=3,
                    bbox_to_anchor=(0.5, 0.96), fontsize=7,
                    columnspacing=1.5, handletextpad=0.5)

    saved += save_figure(fig2, output_dir, "ablation_width_storage")
    plt.close(fig2)

    # --- Figure 3: Scaling trend (line plot FLOPs vs N) ---
    fig3, ax3 = plt.subplots(1, 1, figsize=(w_in * 0.5, w_in * 0.38),
                              constrained_layout=True)

    # Extract N values
    with open(input_path, "r") as f:
        raw = json.load(f)
    n_values = {r["code"]: r["N"] for r in raw}
    ns = np.array([n_values.get(c, 0) for c in codes], dtype=float)

    markers = {"naive_direct": "s", "quimb_compress": "^", "our_mps": "o"}
    for i, method in enumerate(METHOD_ORDER):
        y = flops[i]
        mask = ~np.isnan(y) & (y > 0)
        ax3.plot(ns[mask], y[mask],
                 marker=markers.get(method, "o"),
                 color=METHOD_COLORS.get(method, "#4c72b0"),
                 label=method_display_name(method),
                 linewidth=1.2, markersize=4)

    ax3.set_xlabel("Code size $N$ (qubits)")
    ax3.set_ylabel("Estimated FLOPs / shot")
    ax3.set_yscale("log")
    ax3.set_title("FLOPs Scaling vs Code Size", fontsize=8, fontweight="bold")
    ax3.legend(fontsize=6, loc="upper left")

    saved += save_figure(fig3, output_dir, "ablation_flops_scaling")
    plt.close(fig3)

    return saved


def generate_ablation_plots(input_path: Path, output_dir: Path) -> list[Path]:
    """Generate plots from full ablation run data (original format)."""
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    records = [x for x in records if isinstance(x, dict)]

    codes, code_data = load_ablation_data(input_path)

    def _ablation_flops(cd, method):
        d = cd.get(method, {})
        if d.get("status", "ok") != "ok" and "logical_error_rate" not in d:
            return np.nan
        if method == "our_mps":
            return float(d.get("est_online_flops_per_shot", np.nan))
        return float(d.get("est_flops_per_shot", np.nan))

    def _ablation_latency(cd, method):
        d = cd.get(method, {})
        if d.get("status", "ok") != "ok" and "logical_error_rate" not in d:
            return np.nan
        if method == "our_mps":
            return float(d.get("online_per_shot_ms", np.nan))
        return float(d.get("actual_per_shot_ms", np.nan))

    def _ablation_peak_mb(cd, method):
        d = cd.get(method, {})
        if d.get("status", "ok") != "ok" and "logical_error_rate" not in d:
            return np.nan
        if method == "our_mps":
            return float(d.get("online_peak_memory_bytes", np.nan)) / (1024 * 1024)
        return float(d.get("actual_peak_memory_bytes", np.nan)) / (1024 * 1024)

    def _ablation_ler(cd, method):
        d = cd.get(method, {})
        return float(d.get("logical_error_rate", np.nan))

    flops = _build_matrix(codes, code_data, _ablation_flops)
    latency = _build_matrix(codes, code_data, _ablation_latency)
    peak_mb = _build_matrix(codes, code_data, _ablation_peak_mb)
    ler = _build_matrix(codes, code_data, _ablation_ler)

    figsize = apply_paper_style(width_pt=SINGLE_COLUMN_PT, ncols=2, nrows=2)
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    _plot_grouped_bars(axes[0, 0], latency, codes=codes,
                       ylabel="Latency (ms/shot)", log_scale=True)
    axes[0, 0].set_title("(a) Per-shot Decode Latency", fontsize=8, fontweight="bold")

    _plot_grouped_bars(axes[0, 1], peak_mb, codes=codes,
                       ylabel="Peak Memory (MB)", log_scale=True)
    axes[0, 1].set_title("(b) Peak Memory", fontsize=8, fontweight="bold")

    _plot_grouped_bars(axes[1, 0], flops, codes=codes,
                       ylabel="FLOPs / shot", log_scale=True)
    axes[1, 0].set_title("(c) Estimated FLOPs", fontsize=8, fontweight="bold")

    _plot_grouped_bars(axes[1, 1], np.clip(ler, 1e-8, None), codes=codes,
                       ylabel="Logical Error Rate", log_scale=True)
    axes[1, 1].set_title("(d) Accuracy (LER)", fontsize=8, fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 1.04), fontsize=7)

    saved = save_figure(fig, output_dir, "ablation_decomposition_dashboard")
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ablation decomposition results (estimation or full run).")
    parser.add_argument("--input", type=Path, default=None,
                        help="Input JSON. Auto-detects format.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    saved_all: list[Path] = []

    # If explicit input given, use it
    if args.input is not None:
        path = Path(args.input)
        with open(path) as f:
            data = json.load(f)
        if _is_estimate_format(data):
            print(f"[info] Detected cost estimation format: {path}")
            saved_all += generate_estimate_plots(path, args.output_dir)
        else:
            print(f"[info] Detected ablation run format: {path}")
            saved_all += generate_ablation_plots(path, args.output_dir)
    else:
        # Try both files
        if DEFAULT_ESTIMATE_INPUT.exists():
            print(f"[info] Plotting cost estimates: {DEFAULT_ESTIMATE_INPUT}")
            saved_all += generate_estimate_plots(DEFAULT_ESTIMATE_INPUT, args.output_dir)
        if DEFAULT_ABLATION_INPUT.exists():
            print(f"[info] Plotting ablation results: {DEFAULT_ABLATION_INPUT}")
            saved_all += generate_ablation_plots(DEFAULT_ABLATION_INPUT, args.output_dir)

    if not saved_all:
        print("[error] No input files found. Run estimate_ablation_cost.py first.")
        sys.exit(1)

    for path in saved_all:
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
