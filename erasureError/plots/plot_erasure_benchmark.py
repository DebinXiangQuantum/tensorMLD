#!/usr/bin/env python3
"""Plot erasure benchmark: TN-MLD vs BP-OSD comparison.

Panel (a): LER vs physical error rate at 5% erasure, per code.
Panel (b): LER at fixed p vs erasure rate, showing threshold degradation.

Supports two input formats:
  1. Comprehensive benchmark (nested: codes → data → decoders)
  2. Adaptive flat list (list of result dicts)

Use --adaptive to overlay adaptive results on top of comprehensive data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ── Import paper style ──────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from experiments.plots.paper_style import (
    SINGLE_COLUMN_PT,
    apply_paper_style,
    save_figure,
)

DEFAULT_INPUT = SCRIPT_DIR.parent / "results" / "comprehensive_benchmark_20260302_140627.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "out" / "erasure_benchmark"

# ── Visual config ───────────────────────────────────────────────────────
CODE_ORDER = [
    "QCC_18_4_4",
    "LDPC_25_3_4",
    "LDPC_30_6_4",
    # "TOR_50_2_5",
    "QCC_60_8_4",
    # "QCC_72_12_6",
]

CODE_DISPLAY = {
    "QCC_18_4_4":    "QCC [[18,4,4]]",
    "LDPC_25_3_4":   "LDPC [[25,3,4]]",
    "LDPC_30_6_4":   "LDPC [[30,6,4]]",
    "TOR_50_2_5":    "Toric [[50,2,5]]",
    "QCC_60_8_4":    "QCC [[60,8,4]]",
    "QCC_72_12_6":   "QCC [[72,12,6]]",
}

MARKERS = ["o", "^"]
COLOR_MAP = plt.cm.tab10


# ── Unified data structure ──────────────────────────────────────────────
# Internal format: dict[code_name][decoder][(p, erasure)] = ler

def _build_lookup_from_comprehensive(data: dict) -> dict:
    """Parse comprehensive benchmark JSON into lookup table."""
    lookup: dict[str, dict[str, dict[tuple, float]]] = {}
    for code_name, code_data in data.get("codes", {}).items():
        lookup[code_name] = {}
        for config in code_data.get("data", []):
            p = config["error_rate"]
            er = config["erasure_rate"]
            for dec_name, dec_result in config.get("decoders", {}).items():
                ler = dec_result.get("logical_error_rate")
                if ler is not None and np.isfinite(ler):
                    lookup[code_name].setdefault(dec_name, {})
                    lookup[code_name][dec_name][(p, er)] = ler
    return lookup


def _build_lookup_from_flat_list(records: list) -> dict:
    """Parse adaptive flat-list results into lookup table."""
    lookup: dict[str, dict[str, dict[tuple, float]]] = {}
    for r in records:
        code = r.get("code_name", "")
        dec = r.get("decoder", "")
        p = r.get("error_rate")
        er = r.get("erasure_rate")
        ler = r.get("logical_error_rate")
        if code and dec and p is not None and er is not None:
            if ler is not None and np.isfinite(ler):
                lookup.setdefault(code, {}).setdefault(dec, {})
                lookup[code][dec][(p, er)] = ler
    return lookup


def load_and_merge(
    base_path: Path,
    adaptive_path: Path | None = None,
) -> dict:
    """Load base data and optionally overlay adaptive results."""
    with open(base_path, "r") as f:
        raw = json.load(f)

    # Detect format
    if "codes" in raw:
        lookup = _build_lookup_from_comprehensive(raw)
    elif "results" in raw:
        lookup = _build_lookup_from_flat_list(raw["results"])
    elif isinstance(raw, list):
        lookup = _build_lookup_from_flat_list(raw)
    else:
        raise ValueError(f"Unknown data format in {base_path}")

    # Overlay adaptive results (prefer adaptive over base where both exist,
    # since adaptive has more shots → more reliable LER)
    if adaptive_path and adaptive_path.exists():
        with open(adaptive_path, "r") as f:
            adaptive_raw = json.load(f)
        if isinstance(adaptive_raw, list):
            adaptive_records = adaptive_raw
        elif "results" in adaptive_raw:
            adaptive_records = adaptive_raw["results"]
        else:
            adaptive_records = []

        for r in adaptive_records:
            code = r.get("code_name", "")
            dec = r.get("decoder", "")
            p = r.get("error_rate")
            er = r.get("erasure_rate")
            ler = r.get("logical_error_rate")
            if code and dec and p is not None and er is not None:
                if ler is not None and np.isfinite(ler) and ler > 0:
                    lookup.setdefault(code, {}).setdefault(dec, {})
                    existing = lookup[code][dec].get((p, er))
                    # Prefer adaptive result if base had 0 or NaN
                    if existing is None or existing == 0 or not np.isfinite(existing):
                        lookup[code][dec][(p, er)] = ler
                    # Also prefer if adaptive has more shots (more accurate)
                    elif r.get("num_shots", 0) > 0:
                        lookup[code][dec][(p, er)] = ler

    return lookup


def _extract_ler_vs_p(
    lookup: dict, code_name: str, erasure_rate: float, decoder: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (p_array, ler_array) from lookup."""
    dec_data = lookup.get(code_name, {}).get(decoder, {})
    ps, lers = [], []
    for (p, er), ler in dec_data.items():
        if abs(er - erasure_rate) < 1e-9 and ler > 0:
            ps.append(p)
            lers.append(ler)
    if not ps:
        return np.array([]), np.array([])
    order = np.argsort(ps)
    return np.array(ps)[order], np.array(lers)[order]


def _extract_ler_at_fixed_p(
    lookup: dict, code_name: str, target_p: float, decoder: str
) -> tuple[list[float], list[float]]:
    """Extract (erasure_rates, lers) at a fixed physical error rate."""
    dec_data = lookup.get(code_name, {}).get(decoder, {})
    ers, lers = [], []
    for (p, er), ler in dec_data.items():
        if abs(p - target_p) < 1e-9 and ler > 0:
            ers.append(er)
            lers.append(ler)
    if not ers:
        return [], []
    order = np.argsort(ers)
    return [ers[i] for i in order], [lers[i] for i in order]


def plot_erasure_benchmark(lookup: dict, output_dir: Path) -> list[Path]:
    """Generate the 2-panel figure."""
    figsize = apply_paper_style(
        width_pt=SINGLE_COLUMN_PT,
        ncols=2,
        nrows=1,
        panel_aspect=9/8,
    )
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # ── Panel (a): LER vs p at erasure_rate=0.05 ────────────────────────
    target_erasure = 0.05

    for idx, code_name in enumerate(CODE_ORDER):
        if code_name not in lookup:
            continue
        marker = MARKERS[idx % len(MARKERS)]
        color = COLOR_MAP(idx % 10)
        label = CODE_DISPLAY.get(code_name, code_name)

        # TN MLD (solid)
        ps_tn, lers_tn = _extract_ler_vs_p(lookup, code_name, target_erasure, "tensor_network_mld")
        if len(ps_tn) > 0:
            ax_a.plot(
                ps_tn, np.clip(lers_tn, 1e-7, None),
                marker='^', color=color, linestyle="-", linewidth=1,
                label=f"{label} TN",
                markersize=3.5, zorder=3,
            )

        # BP-OSD (dashed)
        ps_bp, lers_bp = _extract_ler_vs_p(lookup, code_name, target_erasure, "bp_osd")
        if len(ps_bp) > 0:
            ax_a.plot(
                ps_bp, np.clip(lers_bp, 1e-7, None),
                marker='o', color=color, linestyle="--",linewidth=1,
                label=f"{label} BP-OSD",
                markersize=3.5, alpha=0.7, zorder=2,
            )

    # Reference line LER = p
    p_ref = np.linspace(0.0005, 0.012, 100)
    ax_a.plot(p_ref, p_ref, "k-", linewidth=0.8, alpha=0.7, label="LER = $p$")

    ax_a.set_xlabel("Physical error rate $p$")
    ax_a.set_ylabel("Logical error rate")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(8e-4, 1.2e-2)
    ax_a.set_ylim(1e-5, 2e-2)
    # ax_a.set_title(f"(a) LER vs $p$  (erasure = {target_erasure:.0%})")
    ax_a.grid(True, which="major", alpha=0.3,axis='y')

    handles_a, labels_a = ax_a.get_legend_handles_labels()
    # ax_a.legend(
    #     handles_a, labels_a,
    #     fontsize=5, loc="upper left",
    #     ncol=2, columnspacing=0.5, handlelength=1.5,
    #     handletextpad=0.3, labelspacing=0.3,
    # )

    # ── Panel (b): LER at p=0.01 vs erasure rate ───────────────────────
    ref_p = 0.01

    for idx, code_name in enumerate(CODE_ORDER):
        if code_name not in lookup:
            continue
        marker = MARKERS[idx % len(MARKERS)]
        color = COLOR_MAP(idx % 10)
        label = CODE_DISPLAY.get(code_name, code_name)

        for dec_name, ls in [("tensor_network_mld", "-"), ("bp_osd", "--")]:
            dec_label = "TN" if "tensor" in dec_name else "BP-OSD"
            marker = '^' if dec_label == "TN" else "o"
            ers, lers = _extract_ler_at_fixed_p(lookup, code_name, ref_p, dec_name)
            if ers:
                ax_b.plot(
                    ers, np.clip(lers, 1e-7, None),
                    marker=marker, color=color, linestyle=ls,
                    label=f"{label} {dec_label}",
                    markersize=3,
                    zorder=3 if "TN" in dec_label else 2,
                    alpha=1.0 if "TN" in dec_label else 0.7,
                )

    # Reference line: LER = p (threshold boundary)
    ax_b.axhline(
        ref_p, color="k", linestyle="-", linewidth=0.8, alpha=0.5,
        label=f"LER = $p$ = {ref_p}",
    )

    ax_b.set_xlabel("Erasure rate $\\epsilon$")
    ax_b.set_ylabel(f"Logical error rate)")
    ax_b.set_yscale("log")
    # ax_b.set_title(f"(b) LER vs erasure rate  ($p={ref_p}$)")
    ax_b.set_xlim(-0.005, 0.16)
    ax_b.set_ylim(1e-4, 0.06)
    ax_b.grid(True, which="major", alpha=0.3,axis='y')

    handles_b, labels_b = ax_b.get_legend_handles_labels()
    # ax_b.legend(
    #     handles_b, labels_b,
    #     fontsize=5, loc="upper left",
    #     ncol=2, columnspacing=0.5, handlelength=1.5,
    #     handletextpad=0.3, labelspacing=0.3,
    # )

    saved = save_figure(fig, output_dir, "erasure_tn_vs_bposd")
    plt.close(fig)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot erasure benchmark results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--adaptive", type=Path, default=None,
                        help="Adaptive results to overlay (flat list JSON)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    lookup = load_and_merge(args.input, args.adaptive)

    # Summary
    for code in CODE_ORDER:
        if code not in lookup:
            continue
        for dec in ["tensor_network_mld", "bp_osd"]:
            n = len(lookup.get(code, {}).get(dec, {}))
            if n > 0:
                print(f"  {code:15s} {dec:25s}: {n} data points")

    saved = plot_erasure_benchmark(lookup, args.output_dir)
    for p in saved:
        print(f"[saved] {p}")


if __name__ == "__main__":
    main()
