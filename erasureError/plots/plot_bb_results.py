# ============================================================================ #
# Plotting Script for BB Code Benchmark Results                                #
# ============================================================================ #
"""Generate plots from BB code benchmark JSON results."""

import glob
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_latest_bb_results(results_dir: str) -> Dict[str, Any]:
    """Load the most recent BB benchmark results."""
    pattern = os.path.join(results_dir, "bb_benchmark_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No bb_benchmark_*.json in {results_dir}")
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


def plot_bb_ler_vs_error_rate(
    code_name: str,
    code_result: Dict[str, Any],
    output_dir: str,
    show: bool = False,
):
    """Plot LER vs error rate for different erasure rates."""
    all_erasure_rates = sorted(
        list({
            float(entry.get('erasure_rate', 0.0))
            for entry in code_result.get('data', [])
        }))
    if not all_erasure_rates:
        return

    ncols = len(all_erasure_rates)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5), squeeze=False)

    colors = {
        'tensor_network_mld': '#0b5ed7',
        'bp_osd': '#198754',
        'min_sum_bp': '#fd7e14',
        'bp_lsd': '#dc3545',
    }
    markers = {
        'tensor_network_mld': 'o',
        'bp_osd': 's',
        'min_sum_bp': '^',
        'bp_lsd': 'v',
    }

    for idx, erasure_rate in enumerate(all_erasure_rates):
        ax = axes[0, idx]
        decoder_data = extract_decoder_data(code_result, erasure_rate)

        for decoder_name, values in decoder_data.items():
            ax.semilogy(
                values['error_rates'],
                values['lers'],
                marker=markers.get(decoder_name, 'o'),
                color=colors.get(decoder_name, 'black'),
                linewidth=2,
                markersize=6,
                label=decoder_name,
            )

        # Plot y=x reference line (physical error rate)
        if decoder_data:
            x = next(iter(decoder_data.values()))['error_rates']
            ax.semilogy(x, x, 'k--', linewidth=1.5, alpha=0.6, label='p (physical)')

        ax.set_xlabel('Physical Error Rate (p)')
        ax.set_ylabel('Logical Error Rate (LER)')
        ax.set_title(f'{code_name}\nErasure Rate = {erasure_rate:.0%}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_ylim([1e-3, 1])

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'bb_ler_vs_error_{code_name}.png')
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_bb_decoder_comparison(
    code_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    show: bool = False,
):
    """Plot decoder comparison summary across all BB codes."""
    code_names = list(code_results.keys())
    if not code_names:
        return

    # Create figure with 2 rows: erasure=0 and erasure=0.2
    ncols = len(code_names)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8), squeeze=False)

    colors = ['#0b5ed7', '#198754', '#fd7e14', '#dc3545', '#6f42c1']
    markers = ['o', 's', '^', 'v', 'd']

    for idx, code_name in enumerate(code_names):
        code_result = code_results[code_name]

        for row, erasure_rate in enumerate([0.0, 0.2]):
            ax = axes[row, idx]
            decoder_data = extract_decoder_data(code_result, erasure_rate)

            for i, (decoder_name, values) in enumerate(decoder_data.items()):
                ax.semilogy(
                    values['error_rates'],
                    values['lers'],
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    linewidth=2,
                    markersize=5,
                    label=decoder_name,
                )

            if decoder_data:
                x = next(iter(decoder_data.values()))['error_rates']
                ax.semilogy(x, x, 'k--', alpha=0.6, linewidth=1.5, label='p')

            ax.set_title(f'{code_name}\nErasure={erasure_rate:.0%}')
            ax.set_xlabel('Physical Error Rate')
            ax.set_ylabel('Logical Error Rate')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            ax.set_ylim([1e-3, 1])

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'bb_decoder_comparison_summary.png')
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_threshold_analysis(
    code_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    show: bool = False,
):
    """Analyze where LER < physical error rate (below threshold)."""
    code_names = list(code_results.keys())
    if not code_names:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    x_labels = []
    below_threshold_counts = {}

    for code_name in code_names:
        code_result = code_results[code_name]

        for entry in code_result.get('data', []):
            error_rate = float(entry['error_rate'])
            erasure_rate = float(entry.get('erasure_rate', 0.0))

            for decoder_name, decoder_result in entry.get('decoders', {}).items():
                ler = float(decoder_result.get('logical_error_rate', float('nan')))

                if decoder_name not in below_threshold_counts:
                    below_threshold_counts[decoder_name] = {'total': 0, 'below': 0}

                below_threshold_counts[decoder_name]['total'] += 1
                if ler < error_rate:
                    below_threshold_counts[decoder_name]['below'] += 1

    # Plot bar chart
    decoders = list(below_threshold_counts.keys())
    ratios = [below_threshold_counts[d]['below'] / below_threshold_counts[d]['total']
              for d in decoders]

    bars = ax.bar(decoders, ratios, color=['#0b5ed7', '#198754', '#fd7e14'][:len(decoders)])

    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, ratio + 0.02,
                f'{ratio:.1%}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Fraction of test points with LER < p')
    ax.set_title('Threshold Analysis: How often does decoding help?')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'bb_threshold_analysis.png')
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, '..', 'results')
    output_dir = script_dir

    try:
        results = load_latest_bb_results(results_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("Run BB code benchmark first: python tests/bb_code_benchmark.py")
        return

    print("\nGenerating BB code plots...")

    # Plot for each code
    for code_name, code_result in results.items():
        if not isinstance(code_result, dict) or 'data' not in code_result:
            continue
        print(f"\nPlotting {code_name}...")
        plot_bb_ler_vs_error_rate(
            code_name=code_name,
            code_result=code_result,
            output_dir=output_dir,
            show=False,
        )

    # Summary plots
    code_results = {k: v for k, v in results.items()
                    if isinstance(v, dict) and 'data' in v}

    print("\nGenerating summary plots...")
    plot_bb_decoder_comparison(code_results, output_dir, show=False)
    plot_threshold_analysis(code_results, output_dir, show=False)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
