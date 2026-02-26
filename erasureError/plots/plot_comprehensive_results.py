# ============================================================================ #
# Plotting Script for Comprehensive qLDPC Benchmark Results                    #
# ============================================================================ #
"""Generate plots from comprehensive benchmark JSON results."""

import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from experiments.plots.paper_style import (
    DOUBLE_COLUMN_PT,
    SINGLE_COLUMN_PT,
    apply_paper_style,
    save_figure,
)


def _prepare_style(*, width_pt: float, ncols: int = 1, nrows: int = 1) -> tuple[float, float]:
    return apply_paper_style(
        width_pt=width_pt,
        ncols=ncols,
        nrows=nrows,
        panel_aspect=4.0 / 3.0,
        font_size=8.0,
    )


def _save_plot(fig: plt.Figure, output_dir: str, filename: str) -> None:
    stem = Path(filename).stem
    save_figure(fig, Path(output_dir), stem)
    print(f"Saved: {Path(output_dir) / f'{stem}.pdf'}")


def load_latest_results(results_dir: str) -> Dict[str, Any]:
    """Load the most recent comprehensive benchmark results."""
    pattern = os.path.join(results_dir, "comprehensive_benchmark_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No comprehensive_benchmark_*.json in {results_dir}")
    latest_file = max(files, key=os.path.getmtime)
    print(f"Loading results from: {latest_file}")
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_decoder_data(
    code_result: Dict[str, Any],
    erasure_rate: float
) -> Dict[str, Dict[str, List[float]]]:
    """Extract LER vs error rate data for each decoder."""
    decoder_data: Dict[str, Dict[str, List[float]]] = {}

    for entry in code_result.get('data', []):
        if abs(float(entry.get('erasure_rate', -1.0)) - erasure_rate) > 1e-9:
            continue
        error_rate = float(entry['error_rate'])
        for decoder_name, decoder_result in entry.get('decoders', {}).items():
            decoder_data.setdefault(decoder_name, {
                'error_rates': [],
                'lers': [],
            })
            decoder_data[decoder_name]['error_rates'].append(error_rate)
            decoder_data[decoder_name]['lers'].append(
                float(decoder_result.get('logical_error_rate', float('nan'))))

    # Sort by error rate
    for decoder_name, values in decoder_data.items():
        order = np.argsort(values['error_rates'])
        values['error_rates'] = [values['error_rates'][i] for i in order]
        values['lers'] = [values['lers'][i] for i in order]

    return decoder_data


def plot_all_codes_comparison(
    code_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    show: bool = False,
):
    """Plot LER vs error rate for all codes in a grid."""
    code_names = list(code_results.keys())
    if not code_names:
        return

    # Separate codes by type
    bb_codes = [c for c in code_names if c.startswith('BB_')]
    tb_codes = [c for c in code_names if c.startswith('TB_')]
    hp_codes = [c for c in code_names if c.startswith('HP_')]

    # Colors and markers
    colors = {
        'tensor_network_mld': '#0b5ed7',
        'bp_osd': '#198754',
        'min_sum_bp': '#fd7e14',
    }
    markers = {
        'tensor_network_mld': 'o',
        'bp_osd': 's',
        'min_sum_bp': '^',
    }

    # Plot for each code type and erasure rate
    for code_type, codes in [('BB', bb_codes), ('TB', tb_codes), ('HP', hp_codes)]:
        if not codes:
            continue

        ncols = len(codes)
        figsize = _prepare_style(
            width_pt=max(DOUBLE_COLUMN_PT, SINGLE_COLUMN_PT * ncols),
            ncols=ncols,
            nrows=3,
        )
        fig, axes = plt.subplots(3, ncols, figsize=figsize, squeeze=False)
        fig.suptitle(f'{code_type} Codes: LER vs Physical Error Rate', fontsize=14, fontweight='bold')

        for col, code_name in enumerate(codes):
            code_result = code_results[code_name]
            code_params = f"[[{code_result['N']}, {code_result['K']}, {code_result['D']}]]"

            for row, erasure_rate in enumerate([0.0, 0.1, 0.2]):
                ax = axes[row, col]
                decoder_data = extract_decoder_data(code_result, erasure_rate)

                for decoder_name, values in decoder_data.items():
                    # Replace zeros with small value for log scale
                    lers = [max(v, 1e-4) for v in values['lers']]
                    ax.semilogy(
                        values['error_rates'],
                        lers,
                        marker=markers.get(decoder_name, 'o'),
                        color=colors.get(decoder_name, 'black'),
                        linewidth=2,
                        markersize=5,
                        label=decoder_name,
                    )

                # Reference line y=x
                if decoder_data:
                    x = next(iter(decoder_data.values()))['error_rates']
                    ax.semilogy(x, x, 'k--', alpha=0.6, linewidth=1.5, label='p')

                ax.set_xlabel('Physical Error Rate (p)')
                ax.set_ylabel('LER')
                ax.set_title(f'{code_name} {code_params}\nErasure={erasure_rate:.0%}')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7, loc='lower right')
                ax.set_ylim([1e-4, 1])

        plt.tight_layout()
        _save_plot(fig, output_dir, f'{code_type.lower()}_codes_comparison.png')
        if show:
            plt.show()
        else:
            plt.close()


def plot_decoder_performance_summary(
    code_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    show: bool = False,
):
    """Plot summary of decoder performance across all codes."""
    # Count how often each decoder achieves LER < p
    decoder_stats = {}

    for code_name, code_result in code_results.items():
        for entry in code_result.get('data', []):
            p = float(entry['error_rate'])
            for decoder_name, decoder_result in entry.get('decoders', {}).items():
                ler = float(decoder_result.get('logical_error_rate', float('nan')))
                if decoder_name not in decoder_stats:
                    decoder_stats[decoder_name] = {'below': 0, 'total': 0}
                decoder_stats[decoder_name]['total'] += 1
                if ler < p:
                    decoder_stats[decoder_name]['below'] += 1

    # Plot bar chart
    figsize = _prepare_style(width_pt=SINGLE_COLUMN_PT)
    fig, ax = plt.subplots(figsize=figsize)

    decoders = list(decoder_stats.keys())
    ratios = [decoder_stats[d]['below'] / decoder_stats[d]['total'] for d in decoders]

    colors = ['#0b5ed7', '#198754', '#fd7e14']
    bars = ax.bar(decoders, ratios, color=colors[:len(decoders)])

    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, ratio + 0.02,
                f'{ratio:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Fraction of test points with LER < p', fontsize=11)
    ax.set_xlabel('Decoder', fontsize=11)
    ax.set_title('Decoder Performance: How often does decoding help?', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save_plot(fig, output_dir, 'decoder_performance_summary.png')
    if show:
        plt.show()
    else:
        plt.close()


def plot_erasure_impact(
    code_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    fixed_error_rate: float = 0.02,
    show: bool = False,
):
    """Plot how erasure rate affects LER for each code."""
    code_names = list(code_results.keys())
    if not code_names:
        return

    ncols = min(4, len(code_names))
    nrows = (len(code_names) + ncols - 1) // ncols
    figsize = _prepare_style(
        width_pt=max(DOUBLE_COLUMN_PT, SINGLE_COLUMN_PT * ncols),
        ncols=ncols,
        nrows=nrows,
    )
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.suptitle(f'Impact of Erasure Rate on LER (p={fixed_error_rate})', fontsize=14, fontweight='bold')

    colors = {
        'tensor_network_mld': '#0b5ed7',
        'bp_osd': '#198754',
        'min_sum_bp': '#fd7e14',
    }

    for idx, code_name in enumerate(code_names):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        code_result = code_results[code_name]

        # Extract data at fixed error rate
        decoder_data: Dict[str, Dict[str, List[float]]] = {}
        for entry in code_result.get('data', []):
            if abs(float(entry['error_rate']) - fixed_error_rate) > 1e-6:
                continue
            erasure_rate = float(entry['erasure_rate'])
            for decoder_name, decoder_result in entry.get('decoders', {}).items():
                decoder_data.setdefault(decoder_name, {
                    'erasure_rates': [],
                    'lers': [],
                })
                decoder_data[decoder_name]['erasure_rates'].append(erasure_rate)
                decoder_data[decoder_name]['lers'].append(
                    float(decoder_result.get('logical_error_rate', float('nan'))))

        for decoder_name, values in decoder_data.items():
            order = np.argsort(values['erasure_rates'])
            x = [values['erasure_rates'][i] for i in order]
            y = [values['lers'][i] for i in order]
            ax.plot(x, y, marker='o', linewidth=2, markersize=6,
                   color=colors.get(decoder_name, 'black'), label=decoder_name)

        code_params = f"[[{code_result['N']}, {code_result['K']}, {code_result['D']}]]"
        ax.set_title(f'{code_name} {code_params}')
        ax.set_xlabel('Erasure Rate')
        ax.set_ylabel('LER')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.axhline(y=fixed_error_rate, color='black', linestyle='--', alpha=0.5, label='p')

    # Hide unused subplots
    for idx in range(len(code_names), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    _save_plot(fig, output_dir, 'erasure_impact_analysis.png')
    if show:
        plt.show()
    else:
        plt.close()


def plot_code_comparison_at_fixed_params(
    code_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    error_rate: float = 0.01,
    erasure_rate: float = 0.0,
    show: bool = False,
):
    """Compare all codes at fixed error rate and erasure rate."""
    code_names = list(code_results.keys())
    if not code_names:
        return

    # Extract LER for each code at fixed params
    data = {decoder: {'codes': [], 'lers': []} for decoder in ['tensor_network_mld', 'bp_osd', 'min_sum_bp']}

    for code_name in code_names:
        code_result = code_results[code_name]
        for entry in code_result.get('data', []):
            if abs(float(entry['error_rate']) - error_rate) > 1e-6:
                continue
            if abs(float(entry.get('erasure_rate', 0)) - erasure_rate) > 1e-6:
                continue
            for decoder_name, decoder_result in entry.get('decoders', {}).items():
                if decoder_name in data:
                    data[decoder_name]['codes'].append(code_name)
                    data[decoder_name]['lers'].append(
                        float(decoder_result.get('logical_error_rate', float('nan'))))

    figsize = _prepare_style(width_pt=DOUBLE_COLUMN_PT)
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(code_names))
    width = 0.25

    colors = ['#0b5ed7', '#198754', '#fd7e14']
    for i, (decoder_name, values) in enumerate(data.items()):
        ax.bar(x + i * width, values['lers'], width, label=decoder_name, color=colors[i])

    ax.axhline(y=error_rate, color='red', linestyle='--', linewidth=2, label=f'p={error_rate}')

    ax.set_ylabel('Logical Error Rate')
    ax.set_xlabel('Code')
    ax.set_title(f'Decoder Comparison Across Codes\n(p={error_rate}, erasure={erasure_rate:.0%})', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(code_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _save_plot(fig, output_dir, 'code_comparison_fixed_params.png')
    if show:
        plt.show()
    else:
        plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', 'results')
    output_dir = script_dir

    try:
        results = load_latest_results(results_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("Run comprehensive benchmark first: python tests/comprehensive_benchmark.py")
        return

    code_results = results.get('codes', {})
    if not code_results:
        print("No code results found in benchmark JSON.")
        return

    print(f"\nFound {len(code_results)} codes: {list(code_results.keys())}")
    print("\nGenerating plots...")

    # Generate all plots
    plot_all_codes_comparison(code_results, output_dir, show=False)
    plot_decoder_performance_summary(code_results, output_dir, show=False)
    plot_erasure_impact(code_results, output_dir, fixed_error_rate=0.02, show=False)
    plot_code_comparison_at_fixed_params(code_results, output_dir, error_rate=0.01, erasure_rate=0.0, show=False)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
