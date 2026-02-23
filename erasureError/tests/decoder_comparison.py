# ============================================================================ #
# Decoder Comparison for qLDPC Erasure Error Scenarios                         #
# ============================================================================ #
"""
Compare decoders on six qLDPC benchmark codes:
- BB: [[18,4,4]], [[60,8,4]], [[72,12,6]]
- TB: [[25,3,4]], [[30,6,4]], [[48,4,8]]

Expected matrices:
data/cases/<case_name>/H.npy
data/cases/<case_name>/logical.npy
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from erasure_tensor_network_decoder import ErasureTensorNetworkDecoder
from qldpc_cases import default_cases_root, load_qldpc_cases
from simple_decoders import (
    LookupTableDecoder,
    LdpcBpOsdDecoder,
    LdpcMinSumBpDecoder,
    LdpcSequentialRelayBpDecoder,
    CudaqBpOsdDecoder,
    CudaqMinSumBpDecoder,
    CudaqSequentialRelayBpDecoder,
    LDPC_AVAILABLE,
    CUDAQ_QEC_AVAILABLE,
)


def _parse_float_list(raw: str) -> List[float]:
    values = []
    for token in raw.split(','):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def _select_decoders() -> List[Tuple[Any, str]]:
    decoders: List[Tuple[Any, str]] = [
        (ErasureTensorNetworkDecoder, 'tensor_network_mld'),
    ]

    if CUDAQ_QEC_AVAILABLE:
        decoders.extend([
            (CudaqBpOsdDecoder, 'bp_osd'),
            (CudaqMinSumBpDecoder, 'min_sum_bp'),
            (CudaqSequentialRelayBpDecoder, 'sequential_relay_bp'),
        ])
    elif LDPC_AVAILABLE:
        decoders.extend([
            (LdpcBpOsdDecoder, 'bp_osd'),
            (LdpcMinSumBpDecoder, 'min_sum_bp'),
            (LdpcSequentialRelayBpDecoder, 'sequential_relay_bp'),
        ])
    else:
        decoders.append((LookupTableDecoder, 'lookup_table'))

    return decoders


def generate_case_data(
    h: np.ndarray,
    logical_obs: np.ndarray,
    error_rate: float,
    erasure_rate: float,
    num_shots: int,
    random_seed: int,
) -> Dict[str, Any]:
    """Generate syndrome/logical data for one case and one noise point."""
    rng = np.random.default_rng(random_seed)
    num_checks, num_errors = h.shape

    syndromes = np.zeros((num_shots, num_checks), dtype=np.float64)
    actual_logicals = np.zeros(num_shots, dtype=bool)
    erasure_masks = np.zeros((num_shots, num_errors), dtype=bool)
    error_probs = [error_rate] * num_errors

    logical_row = logical_obs[0].astype(np.int32)

    for shot in range(num_shots):
        erasure_mask = rng.random(num_errors) < erasure_rate
        errors = np.zeros(num_errors, dtype=np.int32)

        for i in range(num_errors):
            if erasure_mask[i]:
                errors[i] = int(rng.random() < 0.5)
            else:
                errors[i] = int(rng.random() < error_rate)

        syndrome = (h @ errors) % 2
        syndromes[shot] = syndrome.astype(np.float64)
        actual_logicals[shot] = bool(np.dot(logical_row, errors) % 2)
        erasure_masks[shot] = erasure_mask

    return {
        'H': h,
        'logical_obs': logical_obs,
        'syndromes': syndromes,
        'actual_logicals': actual_logicals,
        'error_probs': error_probs,
        'erasure_masks': erasure_masks,
    }


def decode_with_decoder(
    decoder_class: Any,
    decoder_name: str,
    data: Dict[str, Any],
    debug: bool = False,
) -> Dict[str, Any]:
    """Decode all shots with one decoder."""
    h = data['H']
    logical_obs = data['logical_obs']
    syndromes = data['syndromes']
    actual_logicals = data['actual_logicals']
    error_probs = data['error_probs']
    erasure_masks = data['erasure_masks']

    num_shots = len(syndromes)
    correct_count = 0
    converged_count = 0
    raw_logical_count = int(np.sum(actual_logicals))

    for shot in range(num_shots):
        kwargs: Dict[str, Any] = {
            'H': h,
            'logical_obs': logical_obs,
            'error_probabilities': error_probs,
            'erasure_mask': erasure_masks[shot],
        }
        if decoder_class == ErasureTensorNetworkDecoder:
            kwargs['debug'] = bool(debug and shot == 0)

        decoder = decoder_class(**kwargs)
        result = decoder.decode(syndromes[shot].tolist())

        predicted_logical = bool(result['logical_error_prob'] > 0.5)
        if bool(result.get('converged', True)):
            converged_count += 1

        if predicted_logical == bool(actual_logicals[shot]):
            correct_count += 1

    ler_with_decoder = 1.0 - (correct_count / num_shots)
    ler_without_decoder = raw_logical_count / num_shots

    return {
        'decoder': decoder_name,
        'logical_error_rate': float(ler_with_decoder),
        'ler_without_decoder': float(ler_without_decoder),
        'num_shots': int(num_shots),
        'correct_count': int(correct_count),
        'converged_count': int(converged_count),
    }


def run_case_comparison(
    case_name: str,
    family: str,
    n: int,
    k: int,
    d: int,
    h: np.ndarray,
    logical_obs: np.ndarray,
    error_rates: List[float],
    erasure_rates: List[float],
    num_shots: int,
    seed: int,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run full decoder comparison for one qLDPC case."""
    decoders = _select_decoders()
    case_results: Dict[str, Any] = {
        'name': case_name,
        'family': family,
        'n': n,
        'k': k,
        'd': d,
        'timestamp': datetime.now().isoformat(),
        'num_shots': num_shots,
        'h_shape': list(h.shape),
        'logical_shape': list(logical_obs.shape),
        'error_rates': error_rates,
        'erasure_rates': erasure_rates,
        'decoder_names': [name for _, name in decoders],
        'data': [],
    }

    print(f"\n{'=' * 80}")
    print(f"Case: {case_name} ({family} [[{n},{k},{d}]])")
    print(f"H shape: {h.shape}, logical shape: {logical_obs.shape}")
    print(f"Decoders: {[name for _, name in decoders]}")
    print(f"{'=' * 80}")

    for erasure_rate in erasure_rates:
        for error_rate in error_rates:
            print(
                f"\n--- p={error_rate:.4f}, erasure={erasure_rate:.2f}, shots={num_shots} ---"
            )

            data = generate_case_data(
                h=h,
                logical_obs=logical_obs,
                error_rate=error_rate,
                erasure_rate=erasure_rate,
                num_shots=num_shots,
                random_seed=seed + int(1000 * erasure_rate) + int(100000 * error_rate),
            )

            config_result: Dict[str, Any] = {
                'error_rate': float(error_rate),
                'erasure_rate': float(erasure_rate),
                'decoders': {},
            }

            for decoder_class, decoder_name in decoders:
                print(f"  Running {decoder_name}...", end=' ')
                try:
                    result = decode_with_decoder(
                        decoder_class=decoder_class,
                        decoder_name=decoder_name,
                        data=data,
                        debug=debug,
                    )
                    config_result['decoders'][decoder_name] = result
                    print(f"LER={result['logical_error_rate']:.6f}")
                except Exception as exc:
                    print(f"ERROR: {exc}")
                    config_result['decoders'][decoder_name] = {
                        'decoder': decoder_name,
                        'logical_error_rate': float('nan'),
                        'error': str(exc),
                    }

            tensor_ler = config_result['decoders'].get(
                'tensor_network_mld', {}).get('logical_error_rate', float('nan'))
            config_result['verification'] = {
                'tensor_ler_below_physical': bool(
                    np.isfinite(tensor_ler) and tensor_ler < error_rate),
                'tensor_ler': float(tensor_ler)
                if np.isfinite(tensor_ler) else float('nan'),
                'physical_error_rate': float(error_rate),
            }

            case_results['data'].append(config_result)

    below_count = 0
    total_count = 0
    for entry in case_results['data']:
        total_count += 1
        if entry['verification']['tensor_ler_below_physical']:
            below_count += 1

    case_results['verification_summary'] = {
        'tensor_below_physical_points': below_count,
        'total_points': total_count,
        'ratio': (below_count / total_count) if total_count > 0 else 0.0,
    }
    return case_results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="qLDPC decoder comparison")
    parser.add_argument(
        '--cases-root',
        type=str,
        default=str(default_cases_root()),
        help='Path to data/cases root',
    )
    parser.add_argument(
        '--case-filter',
        type=str,
        default='',
        help='Regex filter on case names',
    )
    parser.add_argument(
        '--skip-missing',
        action='store_true',
        help='Skip cases with missing H.npy/logical.npy',
    )
    parser.add_argument(
        '--max-logicals',
        type=int,
        default=1,
        help='Max number of logical rows per case',
    )
    parser.add_argument(
        '--error-rates',
        type=str,
        default='0.001,0.003,0.005,0.01,0.02,0.05',
    )
    parser.add_argument(
        '--erasure-rates',
        type=str,
        default='0.0,0.1,0.2,0.3',
    )
    parser.add_argument('--shots', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    return parser


def main() -> Dict[str, Any]:
    args = build_arg_parser().parse_args()

    cases, skipped = load_qldpc_cases(
        cases_root=args.cases_root,
        case_filter=args.case_filter,
        skip_missing=args.skip_missing,
        max_logicals=args.max_logicals,
    )

    for msg in skipped:
        print(f"[SKIP] {msg}")

    if not cases:
        raise RuntimeError(
            f"No qLDPC cases loaded from {args.cases_root}. "
            "Place H.npy and logical.npy under data/cases/<case_name>/."
        )

    error_rates = _parse_float_list(args.error_rates)
    erasure_rates = _parse_float_list(args.erasure_rates)

    all_results: Dict[str, Any] = {
        'timestamp': datetime.now().isoformat(),
        'num_shots': int(args.shots),
        'error_rates': error_rates,
        'erasure_rates': erasure_rates,
        'decoder_backend': (
            'cudaq_qec' if CUDAQ_QEC_AVAILABLE else
            ('ldpc' if LDPC_AVAILABLE else 'fallback')
        ),
        'cases': {},
    }

    for case in cases:
        # Evaluate only the first logical row by default for runtime control.
        logical_obs = case.logical[:1, :]
        case_result = run_case_comparison(
            case_name=case.spec.name,
            family=case.spec.family,
            n=case.spec.n,
            k=case.spec.k,
            d=case.spec.d,
            h=case.h,
            logical_obs=logical_obs,
            error_rates=error_rates,
            erasure_rates=erasure_rates,
            num_shots=args.shots,
            seed=args.seed,
            debug=args.debug,
        )
        all_results['cases'][case.spec.name] = case_result

    global_total = 0
    global_below = 0
    for case_result in all_results['cases'].values():
        summary = case_result['verification_summary']
        global_total += int(summary['total_points'])
        global_below += int(summary['tensor_below_physical_points'])

    all_results['verification_summary'] = {
        'tensor_below_physical_points': global_below,
        'total_points': global_total,
        'ratio': (global_below / global_total) if global_total > 0 else 0.0,
    }

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"decoder_comparison_{timestamp}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"Saved results: {out_path}")
    print(
        "Tensor MLD points with LER < physical p: "
        f"{global_below}/{global_total} ({all_results['verification_summary']['ratio']:.2%})"
    )
    print(f"{'=' * 80}")
    return all_results


if __name__ == "__main__":
    main()
