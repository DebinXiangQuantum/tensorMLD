#!/usr/bin/env python3
"""Plot decomposition ablation study results in publication style."""
from __future__ import annotations

import argparse
import json
import sys
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
    METHOD_COLORS,
    apply_paper_style,
    method_display_name,
    save_figure,
)

DEFAULT_INPUT = WORKSPACE / "experiments" / "results" / "ablation_decomposition_results.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "out" / "ablation_decomposition"
METHOD_ORDER = ["naive_direct", "quimb_compress", "our_mps"]


def load_records(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON for ablation, got {type(data)}")
    return [x for x in data if isinstance(x, dict)]


def _is_ok(record: dict[str, Any]) -> bool:
    status = str(record.get("status", "ok"))
    return status == "ok" and record.get("logical_error_rate") is not None


def _collect_codes(records: list[dict[str, Any]]) -> list[str]:
    code_n = {}
    for r in records:
        code = str(r.get("code", "unknown"))
        n_val = int(r.get("N", 10**9))
        code_n[code] = min(code_n.get(code, n_val), n_val)
    return [k for k, _ in sorted(code_n.items(), key=lambda kv: (kv[1], kv[0]))]


def _metric_value(record: dict[str, Any], metric: str) -> float:
    method = str(record.get("method", ""))
    if method == "our_mps":
        if metric == "per_shot_ms":
            return float(record.get("online_per_shot_ms", np.nan))
        if metric == "peak_mb":
            return float(record.get("online_peak_memory_bytes", np.nan)) / (1024.0 * 1024.0)
        if metric == "flops":
            return float(record.get("est_online_flops_per_shot", np.nan))
    else:
        if metric == "per_shot_ms":
            return float(record.get("actual_per_shot_ms", np.nan))
        if metric == "peak_mb":
            return float(record.get("actual_peak_memory_bytes", np.nan)) / (1024.0 * 1024.0)
        if metric == "flops":
            return float(record.get("est_flops_per_shot", np.nan))

    if metric == "ler":
        return float(record.get("logical_error_rate", np.nan))

    return np.nan


def _build_metric_matrix(records: list[dict[str, Any]], codes: list[str], metric: str) -> np.ndarray:
    out = np.full((len(METHOD_ORDER), len(codes)), np.nan, dtype=float)
    code_to_idx = {code: i for i, code in enumerate(codes)}

    for r in records:
        if not _is_ok(r):
            continue
        method = str(r.get("method", ""))
        if method not in METHOD_ORDER:
            continue
        code = str(r.get("code", "unknown"))
        if code not in code_to_idx:
            continue

        i = METHOD_ORDER.index(method)
        j = code_to_idx[code]
        out[i, j] = _metric_value(r, metric)

    return out


def _plot_grouped(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    codes: list[str],
    title: str,
    ylabel: str,
    log_scale: bool,
) -> None:
    x = np.arange(len(codes))
    width = 0.24

    for i, method in enumerate(METHOD_ORDER):
        y = matrix[i]
        offset = (i - (len(METHOD_ORDER) - 1) / 2) * width
        ax.bar(
            x + offset,
            y,
            width,
            label=method_display_name(method),
            color=METHOD_COLORS.get(method, "#4c72b0"),
            alpha=0.92,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(codes, rotation=35, ha="right")
    if log_scale:
        ax.set_yscale("log")


def generate_plots(input_path: Path, output_dir: Path) -> list[Path]:
    records = load_records(input_path)
    codes = _collect_codes(records)
    if not codes:
        raise ValueError("No code records found in ablation JSON.")

    per_shot = _build_metric_matrix(records, codes, "per_shot_ms")
    peak_mb = _build_metric_matrix(records, codes, "peak_mb")
    flops = _build_metric_matrix(records, codes, "flops")
    ler = _build_metric_matrix(records, codes, "ler")

    figsize = apply_paper_style(width_pt=DOUBLE_COLUMN_PT, ncols=2, nrows=2)
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    _plot_grouped(
        axes[0, 0],
        per_shot,
        codes=codes,
        title="Per-shot decode latency",
        ylabel="Latency (ms)",
        log_scale=True,
    )
    _plot_grouped(
        axes[0, 1],
        peak_mb,
        codes=codes,
        title="Peak memory",
        ylabel="Peak memory (MB)",
        log_scale=True,
    )
    _plot_grouped(
        axes[1, 0],
        flops,
        codes=codes,
        title="Estimated FLOPs per shot",
        ylabel="FLOPs/shot",
        log_scale=True,
    )
    _plot_grouped(
        axes[1, 1],
        np.clip(ler, 1e-8, None),
        codes=codes,
        title="Logical error rate",
        ylabel="LER",
        log_scale=True,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(3, len(labels)),
            bbox_to_anchor=(0.5, 1.03),
        )

    saved = save_figure(fig, output_dir, "ablation_decomposition_dashboard")
    plt.close(fig)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ablation decomposition results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    saved = generate_plots(args.input, args.output_dir)
    for path in saved:
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
