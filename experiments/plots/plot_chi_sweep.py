#!/usr/bin/env python3
"""Plot chi sweep benchmark results with publication style.

Supports multiple noise rates (noise_p) and BP-OSD baselines.
Generates per-noise-rate panels and a combined overview.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, MultipleLocator

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from experiments.plots.paper_style import (
    DOUBLE_COLUMN_PT,
    apply_paper_style,
    save_figure,
)

DEFAULT_INPUT = WORKSPACE / "experiments" / "results" / "chi_sweep_results.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "out" / "chi_sweep"

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
COLOR_MAP = plt.cm.tab10


def load_records(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON for chi sweep, got {type(data)}")
    return [x for x in data if isinstance(x, dict)]


def _split_by_noise_p(
    records: list[dict[str, Any]],
) -> dict[float, list[dict[str, Any]]]:
    """Split records by noise_p value."""
    by_p: dict[float, list[dict[str, Any]]] = {}
    for r in records:
        p = float(r.get("noise_p", 0.01))
        by_p.setdefault(p, []).append(r)
    return dict(sorted(by_p.items()))


def _group_ok_records(
    records: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[int], list[str]]:
    """Group ok records by code name, extract chi values and backends."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    chi_values: set[int] = set()
    backends: set[str] = set()

    for r in records:
        if r.get("status") != "ok":
            continue
        code = str(r.get("code", "unknown"))
        grouped.setdefault(code, []).append(r)
        if r.get("chi") is not None:
            chi_values.add(int(r["chi"]))
        if r.get("backend"):
            backends.add(str(r["backend"]))

    for values in grouped.values():
        values.sort(key=lambda item: int(item.get("chi", 0)))

    return grouped, sorted(chi_values), sorted(backends)


def _get_bposd_baselines(records: list[dict[str, Any]]) -> dict[str, float]:
    """Extract BP-OSD LER baselines per code (first valid entry)."""
    baselines: dict[str, float] = {}
    for r in records:
        code = str(r.get("code", "unknown"))
        if code in baselines:
            continue
        bposd = r.get("bposd")
        if isinstance(bposd, dict) and bposd.get("status") == "ok":
            ler = bposd.get("logical_error_rate")
            if ler is not None:
                baselines[code] = float(ler)
    return baselines


def _sort_codes(grouped: dict[str, list[dict[str, Any]]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Sort codes by N (number of physical qubits)."""
    return sorted(
        grouped.items(),
        key=lambda kv: int(kv[1][0].get("N", 10**9)) if kv[1] else 10**9,
    )


def _plot_per_p_panels(
    records_by_p: dict[float, list[dict[str, Any]]],
    output_dir: Path,
) -> list[Path]:
    """Generate a 3-column figure per noise_p: LER, Latency, Truncation."""
    saved: list[Path] = []

    for noise_p, recs in records_by_p.items():
        grouped, chi_values, backends = _group_ok_records(recs)
        if not grouped:
            continue

        bposd_baselines = _get_bposd_baselines(recs)
        sort_items = _sort_codes(grouped)

        figsize = apply_paper_style(
            width_pt=DOUBLE_COLUMN_PT,
            ncols=3,
            nrows=1,
            panel_aspect=4.0 / 3.0,
        )
        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        ax_ler, ax_time, ax_trunc = axes

        for idx, (code, rows) in enumerate(sort_items):
            marker = MARKERS[idx % len(MARKERS)]
            color = COLOR_MAP(idx % 10)
            chis = np.asarray([int(r.get("chi", 0)) for r in rows], dtype=float)
            ler = np.asarray([
                float(r["logical_error_rate"])
                if r.get("logical_error_rate") is not None else np.nan
                for r in rows
            ], dtype=float)
            per_shot = np.asarray([
                float(r["per_shot_ms"])
                if r.get("per_shot_ms") is not None else np.nan
                for r in rows
            ], dtype=float)
            trunc = np.asarray([
                float(r["truncation_error"])
                if r.get("truncation_error") is not None else np.nan
                for r in rows
            ], dtype=float)

            valid_ler = np.isfinite(ler)
            valid_time = np.isfinite(per_shot)
            valid_trunc = np.isfinite(trunc) & (trunc > 0)

            if np.any(valid_ler):
                ax_ler.plot(
                    chis[valid_ler],
                    np.clip(ler[valid_ler], 1e-8, None),
                    marker=marker, color=color, label=code,
                )

            # BP-OSD baseline as horizontal dashed line
            if code in bposd_baselines:
                bp_ler = bposd_baselines[code]
                if bp_ler > 0:
                    ax_ler.axhline(
                        bp_ler, color=color, linestyle="--", alpha=0.5, linewidth=0.8,
                    )

            if np.any(valid_time):
                ax_time.plot(
                    chis[valid_time], per_shot[valid_time],
                    marker=marker, color=color, label=code,
                )

            if np.any(valid_trunc):
                ax_trunc.plot(
                    chis[valid_trunc], trunc[valid_trunc],
                    marker=marker, color=color, label=code,
                )

        ax_ler.set_title(f"LER vs $\\chi$  (p={noise_p})")
        ax_ler.set_xlabel("Bond dimension $\\chi$")
        ax_ler.set_ylabel("Logical error rate")
        ax_ler.set_yscale("log")
        ax_ler.set_xscale("log", base=2)

        ax_time.set_title(f"Latency vs $\\chi$  (p={noise_p})")
        ax_time.set_xlabel("Bond dimension $\\chi$")
        ax_time.set_ylabel("Per-shot decode (ms)")
        ax_time.set_xscale("log", base=2)

        ax_trunc.set_title(f"Truncation error vs $\\chi$  (p={noise_p})")
        ax_trunc.set_xlabel("Bond dimension $\\chi$")
        ax_trunc.set_ylabel("Truncation error")
        ax_trunc.set_yscale("log")
        ax_trunc.set_xscale("log", base=2)

        if chi_values:
            for ax in axes:
                ax.set_xticks(chi_values)
                ax.set_xticklabels([str(c) for c in chi_values], rotation=45, fontsize=6)
                ax.grid(True, alpha=0.3)

        handles, labels = ax_ler.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels,
                loc="upper center",
                ncol=min(5, len(labels)),
                bbox_to_anchor=(0.5, 1.10),
                columnspacing=0.8,
                handlelength=1.6,
                fontsize=6,
            )

        p_str = f"{noise_p:.4f}".rstrip("0").rstrip(".")
        saved.extend(save_figure(fig, output_dir, f"chi_sweep_p{p_str}"))
        plt.close(fig)

    return saved


def _plot_combined_ler(
    records_by_p: dict[float, list[dict[str, Any]]],
    output_dir: Path,
) -> list[Path]:
    """Combined figure: one LER-vs-chi subplot per noise_p."""
    n_p = len(records_by_p)
    if n_p == 0:
        return []

    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT,
        ncols=n_p,
        nrows=1,
        panel_aspect=4.0 / 3.0,
    )
    fig, axes = plt.subplots(1, n_p, figsize=figsize, constrained_layout=True, squeeze=False)

    # Collect all code names across all p values for consistent colors
    all_codes: list[str] = []
    for recs in records_by_p.values():
        grouped, _, _ = _group_ok_records(recs)
        for code in _sort_codes(grouped):
            if code[0] not in all_codes:
                all_codes.append(code[0])
    code_idx = {c: i for i, c in enumerate(all_codes)}

    for col, (noise_p, recs) in enumerate(records_by_p.items()):
        ax = axes[0, col]
        grouped, chi_values, _ = _group_ok_records(recs)
        bposd_baselines = _get_bposd_baselines(recs)
        sort_items = _sort_codes(grouped)

        for code, rows in sort_items:
            idx = code_idx.get(code, 0)
            marker = MARKERS[idx % len(MARKERS)]
            color = COLOR_MAP(idx % 10)
            chis = np.asarray([int(r.get("chi", 0)) for r in rows], dtype=float)
            ler = np.asarray([
                float(r["logical_error_rate"])
                if r.get("logical_error_rate") is not None else np.nan
                for r in rows
            ], dtype=float)

            valid = np.isfinite(ler)
            if np.any(valid):
                ax.plot(
                    chis[valid], np.clip(ler[valid], 1e-8, None),
                    marker=marker, color=color, label=code, markersize=4,
                )

            if code in bposd_baselines and bposd_baselines[code] > 0:
                ax.axhline(
                    bposd_baselines[code], color=color,
                    linestyle="--", alpha=0.4, linewidth=0.8,
                )

        ax.set_title(f"p = {noise_p}")
        ax.set_xlabel("$\\chi$")
        if col == 0:
            ax.set_ylabel("Logical error rate")
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        if chi_values:
            ax.set_xticks(chi_values)
            ax.set_xticklabels([str(c) for c in chi_values], rotation=45, fontsize=6)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=min(5, len(labels)),
            bbox_to_anchor=(0.5, 1.10),
            columnspacing=0.8,
            handlelength=1.6,
            fontsize=6,
        )

    saved = save_figure(fig, output_dir, "chi_sweep_combined_ler")
    plt.close(fig)
    return saved


def generate_plots(input_path: Path, output_dir: Path) -> list[Path]:
    records = load_records(input_path)
    records_by_p = _split_by_noise_p(records)

    # If only one noise_p, fall back to simple grouping
    if not records_by_p:
        raise ValueError("No chi sweep records found in input JSON.")

    saved: list[Path] = []
    saved.extend(_plot_per_p_panels(records_by_p, output_dir))
    if len(records_by_p) > 1:
        saved.extend(_plot_combined_ler(records_by_p, output_dir))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot chi sweep results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    saved = generate_plots(args.input, args.output_dir)
    for path in saved:
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
