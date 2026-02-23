# ============================================================================ #
# Plotting Script for qLDPC Decoder Comparison Results                         #
# ============================================================================ #
"""Generate plots from decoder comparison JSON results."""

import glob
import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_latest_results(results_dir: str) -> Dict[str, Any]:
    pattern = os.path.join(results_dir, "decoder_comparison_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No decoder_comparison_*.json in {results_dir}")
    latest_file = max(files, key=os.path.getmtime)
    print(f"Loading results from: {latest_file}")
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def _case_map(results: Dict[str, Any]) -> Dict[str, Any]:
    # New format
    if 'cases' in results and isinstance(results['cases'], dict):
        return results['cases']
    # Backward compatibility with old format
    return {
        key: value for key, value in results.items() if isinstance(value, dict) and
        'data' in value
    }


def extract_ler_vs_error_rate(
    case_result: Dict[str, Any],
    erasure_rate: float,
) -> Dict[str, Dict[str, List[float]]]:
    data_list = case_result.get('data', [])
    decoder_data: Dict[str, Dict[str, List[float]]] = {}

    for entry in data_list:
        if abs(float(entry.get('erasure_rate', -1.0)) - erasure_rate) > 1e-9:
            continue
        error_rate = float(entry['error_rate'])
        for decoder_name, decoder_result in entry.get('decoders', {}).items():
            decoder_data.setdefault(decoder_name, {
                'error_rates': [],
                'lers': [],
                'ler_no_decoding': [],
            })
            decoder_data[decoder_name]['error_rates'].append(error_rate)
            decoder_data[decoder_name]['lers'].append(
                float(decoder_result.get('logical_error_rate', float('nan'))))
            decoder_data[decoder_name]['ler_no_decoding'].append(
                float(decoder_result.get('ler_without_decoder', float('nan'))))

    for decoder_name, values in decoder_data.items():
        order = np.argsort(values['error_rates'])
        values['error_rates'] = [values['error_rates'][i] for i in order]
        values['lers'] = [values['lers'][i] for i in order]
        values['ler_no_decoding'] = [values['ler_no_decoding'][i] for i in order]

    return decoder_data


def plot_ler_vs_error_rate(
    case_name: str,
    case_result: Dict[str, Any],
    erasure_rates: Optional[List[float]],
    output_dir: str,
    show: bool,
):
    all_erasure_rates = sorted(
        list({
            float(entry.get('erasure_rate', 0.0))
            for entry in case_result.get('data', [])
        }))
    if not all_erasure_rates:
        return
    if erasure_rates is None:
        erasure_rates = all_erasure_rates

    ncols = len(erasure_rates)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), squeeze=False)

    colors = {
        'tensor_network_mld': '#0b5ed7',
        'bp_osd': '#198754',
        'min_sum_bp': '#fd7e14',
        'sequential_relay_bp': '#dc3545',
        'lookup_table': '#6f42c1',
    }
    markers = {
        'tensor_network_mld': 'o',
        'bp_osd': 's',
        'min_sum_bp': '^',
        'sequential_relay_bp': 'v',
        'lookup_table': 'd',
    }

    for idx, erasure_rate in enumerate(erasure_rates):
        ax = axes[0, idx]
        decoder_data = extract_ler_vs_error_rate(case_result, erasure_rate)
        for decoder_name, values in decoder_data.items():
            ax.semilogy(
                values['error_rates'],
                values['lers'],
                marker=markers.get(decoder_name, 'o'),
                color=colors.get(decoder_name, 'black'),
                linewidth=2,
                markersize=5,
                label=decoder_name,
            )

        if decoder_data:
            reference_x = next(iter(decoder_data.values()))['error_rates']
            ax.semilogy(
                reference_x,
                reference_x,
                'k--',
                linewidth=1,
                alpha=0.6,
                label='p (physical)',
            )

        ax.set_xlabel('Physical Error Rate')
        ax.set_ylabel('Logical Error Rate')
        ax.set_title(f'{case_name}\nErasure={erasure_rate:.1%}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'ler_vs_error_{case_name}.png')
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_erasure_effect(
    case_name: str,
    case_result: Dict[str, Any],
    fixed_error_rate: float,
    output_dir: str,
    show: bool,
):
    decoder_data: Dict[str, Dict[str, List[float]]] = {}
    for entry in case_result.get('data', []):
        if abs(float(entry.get('error_rate', -1.0)) - fixed_error_rate) > 1e-9:
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

    if not decoder_data:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for decoder_name, values in decoder_data.items():
        order = np.argsort(values['erasure_rates'])
        x = [values['erasure_rates'][i] for i in order]
        y = [values['lers'][i] for i in order]
        ax.plot(x, y, marker='o', linewidth=2, label=decoder_name)

    ax.set_xlabel('Erasure Rate')
    ax.set_ylabel('Logical Error Rate')
    ax.set_title(f'{case_name}: LER vs Erasure (p={fixed_error_rate:.3f})')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f'erasure_effect_{case_name}.png')
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_decoder_comparison_summary(
    case_results: Dict[str, Any],
    output_dir: str,
    show: bool,
):
    case_names = list(case_results.keys())
    if not case_names:
        return

    ncols = len(case_names)
    fig, axes = plt.subplots(2, ncols, figsize=(4.4 * ncols, 8), squeeze=False)
    colors = ['#0b5ed7', '#198754', '#fd7e14', '#dc3545', '#6f42c1']
    markers = ['o', 's', '^', 'v', 'd']

    for idx, case_name in enumerate(case_names):
        case_result = case_results[case_name]
        top_ax = axes[0, idx]
        bottom_ax = axes[1, idx]

        no_erasure = extract_ler_vs_error_rate(case_result, erasure_rate=0.0)
        for i, (decoder_name, values) in enumerate(no_erasure.items()):
            top_ax.semilogy(
                values['error_rates'],
                values['lers'],
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linewidth=2,
                label=decoder_name,
            )
        if no_erasure:
            x = next(iter(no_erasure.values()))['error_rates']
            top_ax.semilogy(x, x, 'k--', alpha=0.6, linewidth=1, label='p')
        top_ax.set_title(f'{case_name}\nErasure=0')
        top_ax.set_xlabel('Physical Error Rate')
        top_ax.set_ylabel('Logical Error Rate')
        top_ax.grid(True, alpha=0.3)
        top_ax.legend(fontsize=7)

        erasure_20 = extract_ler_vs_error_rate(case_result, erasure_rate=0.2)
        for i, (decoder_name, values) in enumerate(erasure_20.items()):
            bottom_ax.semilogy(
                values['error_rates'],
                values['lers'],
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                linewidth=2,
                label=decoder_name,
            )
        if erasure_20:
            x = next(iter(erasure_20.values()))['error_rates']
            bottom_ax.semilogy(x, x, 'k--', alpha=0.6, linewidth=1, label='p')
        bottom_ax.set_title(f'{case_name}\nErasure=20%')
        bottom_ax.set_xlabel('Physical Error Rate')
        bottom_ax.set_ylabel('Logical Error Rate')
        bottom_ax.grid(True, alpha=0.3)
        bottom_ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'decoder_comparison_summary.png')
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"Saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_verification_summary(
    case_results: Dict[str, Any],
    global_summary: Dict[str, Any],
    output_dir: str,
    show: bool,
):
    case_names = []
    ratios = []
    for case_name, result in case_results.items():
        summary = result.get('verification_summary', {})
        case_names.append(case_name)
        ratios.append(float(summary.get('ratio', 0.0)))

    if not case_names:
        return

    fig, ax = plt.subplots(figsize=(max(7, len(case_names) * 1.2), 4.2))
    bars = ax.bar(case_names, ratios, color='#0d6efd', alpha=0.85)
    ax.axhline(
        float(global_summary.get('ratio', 0.0)),
        color='red',
        linestyle='--',
        linewidth=1.5,
        label='global ratio',
    )
    for bar, ratio in zip(bars, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ratio + 0.01,
            f"{ratio:.1%}",
            ha='center',
            va='bottom',
            fontsize=8,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Fraction of points with LER < p')
    ax.set_title('Threshold-Region Verification Summary')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'verification_summary.png')
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
        results = load_latest_results(results_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        print("Run decoder comparison first: python tests/decoder_comparison.py")
        return

    case_results = _case_map(results)
    if not case_results:
        print("No case results found in decoder comparison JSON.")
        return

    print("\nGenerating plots...")
    for case_name, case_result in case_results.items():
        print(f"\nPlotting {case_name}...")
        plot_ler_vs_error_rate(
            case_name=case_name,
            case_result=case_result,
            erasure_rates=None,
            output_dir=output_dir,
            show=False,
        )
        try:
            plot_erasure_effect(
                case_name=case_name,
                case_result=case_result,
                fixed_error_rate=0.01,
                output_dir=output_dir,
                show=False,
            )
        except Exception as exc:
            print(f"  Skipped erasure-effect plot ({case_name}): {exc}")

    print("\nGenerating summary plots...")
    plot_decoder_comparison_summary(
        case_results=case_results,
        output_dir=output_dir,
        show=False,
    )
    plot_verification_summary(
        case_results=case_results,
        global_summary=results.get('verification_summary', {}),
        output_dir=output_dir,
        show=False,
    )
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
