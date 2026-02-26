#!/usr/bin/env python3
"""Generate all experiment plots in one command."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from experiments.plots.plot_ablation_decomposition import generate_plots as plot_ablation
from experiments.plots.plot_chi_sweep import generate_plots as plot_chi
from experiments.plots.plot_gpu_metrics import generate_plots as plot_gpu
from experiments.plots.plot_isca_revision_1d_mps import generate_plots as plot_isca
from experiments.plots.plot_multi_decoder_benchmark import generate_plots as plot_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all experiments plots.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=WORKSPACE / "experiments" / "results",
        help="Directory that contains experiment JSON results.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR / "out",
        help="Root directory for exported plot files.",
    )
    args = parser.parse_args()

    tasks = [
        (
            "chi_sweep",
            plot_chi,
            args.results_dir / "chi_sweep_results.json",
            args.output_root / "chi_sweep",
        ),
        (
            "multi_decoder_benchmark",
            plot_benchmark,
            args.results_dir / "multi_decoder_benchmark.json",
            args.output_root / "multi_decoder_benchmark",
        ),
        (
            "gpu_metrics",
            plot_gpu,
            args.results_dir / "gpu_metrics.json",
            args.output_root / "gpu_metrics",
        ),
        (
            "ablation_decomposition",
            plot_ablation,
            args.results_dir / "ablation_decomposition_results.json",
            args.output_root / "ablation_decomposition",
        ),
        (
            "isca_revision_1d_mps",
            plot_isca,
            args.results_dir / "isca_revision_1d_mps" / "manifest.json",
            args.output_root / "isca_revision_1d_mps",
        ),
    ]

    for name, fn, input_path, output_dir in tasks:
        try:
            print(f"[plot] {name}: input={input_path}")
            saved = fn(input_path, output_dir)
            for path in saved:
                print(f"  [saved] {path}")
        except FileNotFoundError:
            print(f"  [skip] missing input: {input_path}")
        except Exception as exc:
            print(f"  [error] {name}: {exc}")


if __name__ == "__main__":
    main()
