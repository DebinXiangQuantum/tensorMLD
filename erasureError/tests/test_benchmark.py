# ============================================================================ #
# qLDPC Benchmark for Erasure Tensor Network Decoder                           #
# ============================================================================ #
"""
Benchmark only the erasure-aware tensor-network decoder on the six qLDPC codes.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from erasure_tensor_network_decoder import ErasureTensorNetworkDecoder
from qldpc_cases import default_cases_root, load_qldpc_cases


def _parse_float_list(raw: str) -> List[float]:
    values = []
    for token in raw.split(','):
        token = token.strip()
        if token:
            values.append(float(token))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def _simulate_data(
    h: np.ndarray,
    logical_obs: np.ndarray,
    error_rate: float,
    erasure_rate: float,
    num_shots: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    num_checks, num_errors = h.shape
    syndromes = np.zeros((num_shots, num_checks), dtype=np.float64)
    actual_logicals = np.zeros(num_shots, dtype=bool)
    erasure_masks = np.zeros((num_shots, num_errors), dtype=bool)
    logical_row = logical_obs[0].astype(np.int32)

    for shot in range(num_shots):
        erasure_mask = rng.random(num_errors) < erasure_rate
        errors = np.zeros(num_errors, dtype=np.int32)
        for i in range(num_errors):
            if erasure_mask[i]:
                errors[i] = int(rng.random() < 0.5)
            else:
                errors[i] = int(rng.random() < error_rate)

        syndromes[shot] = ((h @ errors) % 2).astype(np.float64)
        actual_logicals[shot] = bool(np.dot(logical_row, errors) % 2)
        erasure_masks[shot] = erasure_mask

    return {
        'syndromes': syndromes,
        'actual_logicals': actual_logicals,
        'erasure_masks': erasure_masks,
    }


def _run_case(
    case_name: str,
    h: np.ndarray,
    logical_obs: np.ndarray,
    error_rates: List[float],
    erasure_rates: List[float],
    num_shots: int,
    seed: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        'name': case_name,
        'h_shape': list(h.shape),
        'logical_shape': list(logical_obs.shape),
        'num_shots': num_shots,
        'data': [],
    }

    for erasure_rate in erasure_rates:
        for error_rate in error_rates:
            sim = _simulate_data(
                h=h,
                logical_obs=logical_obs,
                error_rate=error_rate,
                erasure_rate=erasure_rate,
                num_shots=num_shots,
                seed=seed + int(1000 * erasure_rate) + int(100000 * error_rate),
            )

            syndromes = sim['syndromes']
            actual_logicals = sim['actual_logicals']
            erasure_masks = sim['erasure_masks']

            correct = 0
            converged = 0
            raw_errors = int(np.sum(actual_logicals))

            for shot in range(num_shots):
                decoder = ErasureTensorNetworkDecoder(
                    H=h,
                    logical_obs=logical_obs,
                    error_probabilities=[error_rate] * h.shape[1],
                    erasure_mask=erasure_masks[shot],
                    debug=False,
                )
                result = decoder.decode(syndromes[shot].tolist())
                pred = bool(result['logical_error_prob'] > 0.5)
                if pred == bool(actual_logicals[shot]):
                    correct += 1
                if bool(result.get('converged', True)):
                    converged += 1

            ler = 1.0 - (correct / num_shots)
            ler_raw = raw_errors / num_shots
            out['data'].append({
                'error_rate': float(error_rate),
                'erasure_rate': float(erasure_rate),
                'ler_with_decoder': float(ler),
                'ler_without_decoder': float(ler_raw),
                'decoding_gain': float(ler_raw - ler),
                'tensor_ler_below_physical': bool(ler < error_rate),
                'converged_fraction': float(converged / num_shots),
            })

    return out


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="qLDPC benchmark (tensor decoder)")
    parser.add_argument('--cases-root', type=str, default=str(default_cases_root()))
    parser.add_argument('--case-filter', type=str, default='')
    parser.add_argument('--skip-missing', action='store_true')
    parser.add_argument('--max-logicals', type=int, default=1)
    parser.add_argument('--error-rates', type=str, default='0.001,0.003,0.005,0.01,0.02')
    parser.add_argument('--erasure-rates', type=str, default='0.0,0.1,0.2,0.3')
    parser.add_argument('--shots', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

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
            "Place H.npy/logical.npy under data/cases/<case_name>/."
        )

    error_rates = _parse_float_list(args.error_rates)
    erasure_rates = _parse_float_list(args.erasure_rates)

    all_results: Dict[str, Any] = {
        'timestamp': datetime.now().isoformat(),
        'num_shots': int(args.shots),
        'error_rates': error_rates,
        'erasure_rates': erasure_rates,
        'cases': {},
    }

    for case in cases:
        print(f"Benchmarking {case.spec.name}...")
        all_results['cases'][case.spec.name] = _run_case(
            case_name=case.spec.name,
            h=case.h,
            logical_obs=case.logical[:1, :],
            error_rates=error_rates,
            erasure_rates=erasure_rates,
            num_shots=args.shots,
            seed=args.seed,
        )

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"benchmark_results_{timestamp}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved benchmark results: {out_path}")
    return all_results


if __name__ == "__main__":
    main()
