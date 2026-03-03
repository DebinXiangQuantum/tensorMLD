#!/usr/bin/env python3
"""Shared publication-style plotting utilities for experiments."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

INCHES_PER_PT = 1.0 / 72.27
SINGLE_COLUMN_PT = 240.0
DOUBLE_COLUMN_PT = 510.0
BASE_FONT_SIZE = 7.0

DECODER_COLORS = {
    "mps_tn": "#1b9e77",
    "exact_tn": "#7570b3",
    "bp": "#d95f02",
    "bp_osd": "#e7298a",
    "bp_lsd": "#66a61e",
}

METHOD_COLORS = {
    "naive_direct": "#d95f02",
    "quimb_compress": "#7570b3",
    "our_mps": "#1b9e77",
}

# 5-method ablation study colors
METHOD_COLORS_5 = {
    "onthefly_oc": "#e7298a",       # magenta-pink
    "full_tn_oc": "#d95f02",        # orange
    "quimb_compress": "#7570b3",    # purple
    "exact_mps": "#66a61e",         # green
    "compressed_mps": "#1b9e77",    # teal
}

METHOD_DISPLAY_5 = {
    "onthefly_oc": "On-the-fly + OC",
    "full_tn_oc": "Full TN + OC",
    "quimb_compress": "Quimb Compress",
    "exact_mps": "Exact 1D MPS",
    "compressed_mps": "Compressed MPS",
}


def figure_size(
    *,
    width_pt: float = SINGLE_COLUMN_PT,
    ncols: int = 1,
    nrows: int = 1,
    panel_aspect: float = 4.0 / 2.0,
) -> tuple[float, float]:
    """Return (width, height) in inches from publication point units.

    panel_aspect is width / height for one panel.
    """
    width_in = float(width_pt) * INCHES_PER_PT
    panel_width = width_in / max(1, int(ncols))
    panel_height = panel_width / float(panel_aspect)
    height_in = panel_height * max(1, int(nrows))
    return width_in, height_in


def apply_paper_style(
    *,
    width_pt: float = SINGLE_COLUMN_PT,
    ncols: int = 1,
    nrows: int = 1,
    panel_aspect: float = 4.0 / 3.0,
    font_size: float = BASE_FONT_SIZE,
) -> tuple[float, float]:
    """Apply paper plotting style and return figure size."""
    figsize = figure_size(
        width_pt=width_pt,
        ncols=ncols,
        nrows=nrows,
        panel_aspect=panel_aspect,
    )

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": font_size,
            "figure.figsize": figsize,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "xtick.major.pad": 0.3,
            "ytick.major.pad": 0.3,
            "legend.fontsize": font_size,
            "lines.markersize": 5,
            "lines.linewidth": 1.2,
            "lines.markeredgewidth": 0.3,
            "grid.linewidth": 0.3,
            "grid.alpha": 0.4,
            "grid.color": "gray",
            "grid.linestyle": "--",
            "axes.grid.axis": "y",
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "axes.grid": True,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return figsize


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    """Save figure as both PDF and PNG and return paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for ext in ("pdf", "png", "svg"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(path)
        saved.append(path)
    return saved


def decoder_display_name(name: str) -> str:
    mapping = {
        "mps_tn": "MPS-TN",
        "exact_tn": "Exact-TN",
        "bp": "BP",
        "bp_osd": "BP-OSD",
        "bp_lsd": "BP-LSD",
    }
    return mapping.get(str(name), str(name))


def method_display_name(name: str) -> str:
    mapping = {
        "naive_direct": "Naive Direct",
        "quimb_compress": "Quimb Compress",
        "our_mps": "Our MPS",
    }
    return mapping.get(str(name), str(name))


def method_display_name_5(name: str) -> str:
    return METHOD_DISPLAY_5.get(str(name), str(name))
