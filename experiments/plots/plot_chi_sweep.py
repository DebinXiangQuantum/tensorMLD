#!/usr/bin/env python3
"""Plot chi sweep benchmark results with publication style."""
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


def load_records(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON for chi sweep, got {type(data)}")
    return [x for x in data if isinstance(x, dict)]


def _group_ok_records(records: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], list[int], list[str]]:
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

    chi_sorted = sorted(chi_values)
    backend_sorted = sorted(backends)
    return grouped, chi_sorted, backend_sorted


def _plot_main_panels(
    grouped: dict[str, list[dict[str, Any]]],
    chi_values: list[int],
    backends: list[str],
    output_dir: Path,
) -> list[Path]:
    figsize = apply_paper_style(
        width_pt=DOUBLE_COLUMN_PT,
        ncols=2,
        nrows=1,
        panel_aspect=4.0 / 3.0,
    )
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    ax_ler, ax_time = axes

    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    color_map = plt.cm.tab10

    sort_items = sorted(
        grouped.items(),
        key=lambda kv: int(kv[1][0].get("N", 10**9)) if kv[1] else 10**9,
    )

    for idx, (code, rows) in enumerate(sort_items):
        marker = markers[idx % len(markers)]
        color = color_map(idx % 10)
        chis = np.asarray([int(r.get("chi", 0)) for r in rows], dtype=float)
        ler = np.asarray([
            float(r["logical_error_rate"])
            if r.get("logical_error_rate") is not None
            else np.nan
            for r in rows
        ], dtype=float)
        per_shot = np.asarray([
            float(r["per_shot_ms"]) if r.get("per_shot_ms") is not None else np.nan
            for r in rows
        ], dtype=float)

        valid_ler = np.isfinite(ler)
        valid_time = np.isfinite(per_shot)

        if np.any(valid_ler):
            ax_ler.plot(
                chis[valid_ler],
                np.clip(ler[valid_ler], 1e-8, None),
                marker=marker,
                color=color,
                label=code,
            )

        if np.any(valid_time):
            ax_time.plot(
                chis[valid_time],
                per_shot[valid_time],
                marker=marker,
                color=color,
                label=code,
            )

    backend_str = ", ".join(backends) if backends else "unknown"
    ax_ler.set_title(f"LER vs chi (backend: {backend_str})")
    ax_ler.set_xlabel("Bond dimension chi")
    ax_ler.set_ylabel("Logical error rate")
    ax_ler.set_yscale("log")
    ax_ler.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))

    ax_time.set_title("Latency vs chi")
    ax_time.set_xlabel("Bond dimension chi")
    ax_time.set_ylabel("Per-shot decode latency (ms)")
    finite_time = [
        float(r.get("per_shot_ms"))
        for rows in grouped.values()
        for r in rows
        if r.get("per_shot_ms") is not None
    ]
    if finite_time:
        upper = max(finite_time)
        step = max(0.01, round(upper / 6.0, 2))
        ax_time.yaxis.set_major_locator(MultipleLocator(step))

    if chi_values:
        for ax in (ax_ler, ax_time):
            ax.set_xticks(chi_values)

    handles, labels = ax_ler.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(5, len(labels)),
            bbox_to_anchor=(0.5, 1.08),
            columnspacing=0.8,
            handlelength=1.6,
        )

    saved = save_figure(fig, output_dir, "chi_sweep_ler_latency")
    plt.close(fig)
    return saved


def _plot_truncation_panel(grouped: dict[str, list[dict[str, Any]]], output_dir: Path) -> list[Path]:
    figsize = apply_paper_style(width_pt=DOUBLE_COLUMN_PT / 2.0)
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
    color_map = plt.cm.tab10

    plotted = False
    sort_items = sorted(
        grouped.items(),
        key=lambda kv: int(kv[1][0].get("N", 10**9)) if kv[1] else 10**9,
    )

    for idx, (code, rows) in enumerate(sort_items):
        chis = np.asarray([int(r.get("chi", 0)) for r in rows], dtype=float)
        trunc = np.asarray([
            float(r["truncation_error"])
            if r.get("truncation_error") is not None
            else np.nan
            for r in rows
        ], dtype=float)
        valid = np.isfinite(trunc) & (trunc > 0)
        if not np.any(valid):
            continue
        plotted = True
        ax.plot(
            chis[valid],
            trunc[valid],
            marker=markers[idx % len(markers)],
            color=color_map(idx % 10),
            label=code,
        )

    if not plotted:
        plt.close(fig)
        return []

    ax.set_title("Offline truncation error vs chi")
    ax.set_xlabel("Bond dimension chi")
    ax.set_ylabel("Truncation error")
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.legend(loc="best")

    saved = save_figure(fig, output_dir, "chi_sweep_truncation")
    plt.close(fig)
    return saved


def generate_plots(input_path: Path, output_dir: Path) -> list[Path]:
    records = load_records(input_path)
    grouped, chi_values, backends = _group_ok_records(records)
    if not grouped:
        raise ValueError("No successful chi sweep records found in input JSON.")

    saved = []
    saved.extend(_plot_main_panels(grouped, chi_values, backends, output_dir))
    saved.extend(_plot_truncation_panel(grouped, output_dir))
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
