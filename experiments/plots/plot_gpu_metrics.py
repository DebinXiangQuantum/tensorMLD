#!/usr/bin/env python3
"""Plot GPU metrics benchmark results with publication style."""
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
    apply_paper_style,
    save_figure,
)

DEFAULT_INPUT = WORKSPACE / "experiments" / "results" / "gpu_metrics.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "out" / "gpu_metrics"


def load_records(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON for gpu metrics, got {type(data)}")
    return [x for x in data if isinstance(x, dict)]


def _ok_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok = [r for r in records if r.get("status") == "ok"]
    ok.sort(key=lambda r: (int(r.get("N", 10**9)), str(r.get("code", ""))))
    return ok


def _extract(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    data: dict[str, list[float]] = {
        "compile_ms": [],
        "decode_total_ms": [],
        "per_shot_ms": [],
        "latency_ms": [],
        "ler": [],
        "gpu_mem_peak_mb": [],
        "gpu_util_avg": [],
    }

    for r in records:
        data["compile_ms"].append(float(r.get("compile_ms", np.nan)))
        data["decode_total_ms"].append(float(r.get("total_decode_ms", np.nan)))
        data["per_shot_ms"].append(float(r.get("per_shot_ms", np.nan)))

        lat = r.get("single_shot_latency", {})
        if isinstance(lat, dict) and lat.get("avg_ms") is not None:
            data["latency_ms"].append(float(lat["avg_ms"]))
        else:
            data["latency_ms"].append(np.nan)

        data["ler"].append(float(r.get("logical_error_rate", np.nan)))

        dec_prof = r.get("decode_gpu_profile", {})
        if isinstance(dec_prof, dict) and dec_prof.get("mem_used_peak_mb") is not None:
            data["gpu_mem_peak_mb"].append(float(dec_prof["mem_used_peak_mb"]))
        else:
            data["gpu_mem_peak_mb"].append(np.nan)

        if isinstance(dec_prof, dict) and dec_prof.get("gpu_util_avg") is not None:
            data["gpu_util_avg"].append(float(dec_prof["gpu_util_avg"]))
        else:
            data["gpu_util_avg"].append(np.nan)

    return {k: np.asarray(v, dtype=float) for k, v in data.items()}


def generate_plots(input_path: Path, output_dir: Path) -> list[Path]:
    records = _ok_records(load_records(input_path))
    if not records:
        raise ValueError("No successful gpu metrics records found.")

    codes = [str(r.get("code", "unknown")) for r in records]
    m = _extract(records)

    figsize = apply_paper_style(width_pt=DOUBLE_COLUMN_PT, ncols=2, nrows=2)
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    x = np.arange(len(codes))
    w = 0.38

    ax_a.bar(x - w / 2, m["compile_ms"], w, color="#1f77b4", label="Compile")
    ax_a.bar(x + w / 2, m["decode_total_ms"], w, color="#ff7f0e", label="Total decode")
    ax_a.set_title("Offline/online total time")
    ax_a.set_ylabel("Time (ms)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(codes, rotation=35, ha="right")
    ax_a.legend(loc="upper left")

    ax_b.bar(x - w / 2, m["per_shot_ms"], w, color="#2ca02c", label="Per-shot")
    ax_b.bar(x + w / 2, m["latency_ms"], w, color="#9467bd", label="Single-shot")
    ax_b.set_title("Latency metrics")
    ax_b.set_ylabel("Latency (ms)")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(codes, rotation=35, ha="right")
    ax_b.legend(loc="upper left")

    ax_c.bar(x, m["gpu_mem_peak_mb"], color="#17becf", label="GPU memory peak")
    ax_c.set_title("GPU memory/utilization")
    ax_c.set_ylabel("Peak memory (MB)")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(codes, rotation=35, ha="right")

    ax_c2 = ax_c.twinx()
    ax_c2.plot(x, m["gpu_util_avg"], color="#d62728", marker="o", label="GPU util avg")
    ax_c2.set_ylabel("GPU util avg (%)")
    ax_c2.grid(False)

    h1, l1 = ax_c.get_legend_handles_labels()
    h2, l2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(h1 + h2, l1 + l2, loc="upper left")

    ax_d.bar(x, np.clip(m["ler"], 1e-8, None), color="#8c564b")
    ax_d.set_title("Logical error rate")
    ax_d.set_ylabel("LER")
    ax_d.set_yscale("log")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(codes, rotation=35, ha="right")

    saved = save_figure(fig, output_dir, "gpu_metrics_dashboard")
    plt.close(fig)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot gpu metrics results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    saved = generate_plots(args.input, args.output_dir)
    for path in saved:
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
