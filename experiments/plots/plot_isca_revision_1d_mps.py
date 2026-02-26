#!/usr/bin/env python3
"""Plot ISCA revision MPS export manifest in publication style."""
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

from experiments.plots.paper_style import DOUBLE_COLUMN_PT, apply_paper_style, save_figure

DEFAULT_INPUT = WORKSPACE / "experiments" / "results" / "isca_revision_1d_mps" / "manifest.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "out" / "isca_revision_1d_mps"


def load_manifest(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON for manifest, got {type(data)}")
    return data


def _sorted_codes(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = manifest.get("codes", [])
    if not isinstance(rows, list):
        return []
    rows = [r for r in rows if isinstance(r, dict)]
    rows.sort(key=lambda r: (int(r.get("N", 10**9)), str(r.get("name", ""))))
    return rows


def generate_plots(input_path: Path, output_dir: Path) -> list[Path]:
    manifest = load_manifest(input_path)
    rows = _sorted_codes(manifest)
    if not rows:
        raise ValueError("No code records found in manifest JSON.")

    codes = [str(r.get("name", "unknown")) for r in rows]
    x = np.arange(len(codes))

    compile_ms = np.asarray([float(r.get("compile_ms", np.nan)) for r in rows], dtype=float)
    num_sites = np.asarray([float(r.get("num_sites", np.nan)) for r in rows], dtype=float)
    num_syn = np.asarray([float(r.get("num_syndrome_indices", np.nan)) for r in rows], dtype=float)
    num_log = np.asarray([float(r.get("num_logical_indices", np.nan)) for r in rows], dtype=float)

    trunc_err = []
    max_bond = []
    for r in rows:
        stats = r.get("offline_stats", {})
        if not isinstance(stats, dict):
            trunc_err.append(np.nan)
            max_bond.append(np.nan)
            continue
        trunc_err.append(float(stats.get("truncation_error", np.nan)))
        max_bond.append(float(stats.get("max_bond_dim", np.nan)))

    trunc_err_arr = np.asarray(trunc_err, dtype=float)
    max_bond_arr = np.asarray(max_bond, dtype=float)

    figsize = apply_paper_style(width_pt=DOUBLE_COLUMN_PT, ncols=3, nrows=1)
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    axes[0].bar(x, compile_ms, color="#1f77b4")
    axes[0].set_title("Offline compile time")
    axes[0].set_ylabel("Compile time (ms)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(codes, rotation=35, ha="right")

    w = 0.25
    axes[1].bar(x - w, num_sites, w, color="#2ca02c", label="Sites")
    axes[1].bar(x, num_syn, w, color="#ff7f0e", label="Syndrome idx")
    axes[1].bar(x + w, num_log, w, color="#9467bd", label="Logical idx")
    axes[1].set_title("Exported chain structure")
    axes[1].set_ylabel("Count")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(codes, rotation=35, ha="right")
    axes[1].legend(loc="upper left")

    axes[2].plot(x, np.clip(trunc_err_arr, 1e-16, None), marker="o", color="#d62728", label="Trunc. error")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Truncation error")
    axes[2].set_title("Compression quality")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(codes, rotation=35, ha="right")

    ax2b = axes[2].twinx()
    ax2b.plot(x, max_bond_arr, marker="s", color="#17becf", label="Max bond")
    ax2b.set_ylabel("Max bond dimension")
    ax2b.grid(False)

    h1, l1 = axes[2].get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    axes[2].legend(h1 + h2, l1 + l2, loc="upper left")

    saved = save_figure(fig, output_dir, "isca_revision_1d_mps_manifest")
    plt.close(fig)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ISCA revision 1D MPS manifest.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    saved = generate_plots(args.input, args.output_dir)
    for path in saved:
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
