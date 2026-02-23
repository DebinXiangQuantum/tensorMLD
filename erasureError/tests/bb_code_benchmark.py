# ============================================================================ #
# BB Code Benchmark for Erasure Error Decoding                                 #
# ============================================================================ #
"""
Benchmark tensor network MLD decoder against ldpc decoders on BB codes.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from codes.codes import gen_BB_code
from erasure_tensor_network_decoder import ErasureTensorNetworkDecoder
from simple_decoders import (
    LdpcBpOsdDecoder,
    LdpcMinSumBpDecoder,
    LdpcBpLsdDecoder,
    LDPC_AVAILABLE
)


def generate_bb_code_data(
    bb_code,
    error_rate: float,
    erasure_rate: float,
    num_shots: int,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Generate syndrome data for a BB code.

    For CSS codes, we only test X errors (hx detects X errors).
    """
    np.random.seed(random_seed)

    # Use hx as parity check matrix (detects X errors)
    H = bb_code.hx.astype(np.float64)
    # Use lz as logical observable (X errors affect Z logicals)
    logical_obs = bb_code.lz.astype(np.float64)

    num_checks, num_qubits = H.shape
    num_logicals = logical_obs.shape[0]

    print(f"  Code: n={num_qubits}, k={num_logicals}, checks={num_checks}")

    # Simulate syndrome data
    syndromes = np.zeros((num_shots, num_checks), dtype=np.float64)
    actual_logicals = np.zeros((num_shots, num_logicals), dtype=bool)
    erasure_masks = np.zeros((num_shots, num_qubits), dtype=bool)
    error_probs = np.full(num_qubits, error_rate)

    for shot in range(num_shots):
        # Generate erasure mask
        erasure_mask = np.random.rand(num_qubits) < erasure_rate

        # Generate X errors
        errors = np.zeros(num_qubits, dtype=bool)
        for i in range(num_qubits):
            if erasure_mask[i]:
                # Erased qubit: X error with p=0.5
                errors[i] = np.random.rand() < 0.5
            else:
                # Normal qubit: X error with given rate
                errors[i] = np.random.rand() < error_rate

        # Compute syndrome (hx @ errors mod 2)
        syndrome = (H @ errors.astype(np.int32)) % 2
        syndromes[shot] = syndrome.astype(np.float64)

        # Compute logical errors (lz @ errors mod 2)
        actual_logicals[shot] = ((logical_obs @ errors.astype(np.int32)) % 2).astype(bool)

        erasure_masks[shot] = erasure_mask

    return {
        'H': H,
        'logical_obs': logical_obs,
        'syndromes': syndromes,
        'actual_logicals': actual_logicals,
        'error_probs': error_probs.tolist(),
        'erasure_masks': erasure_masks
    }


def decode_with_decoder(
    decoder_class,
    decoder_name: str,
    data: Dict[str, Any],
    debug: bool = False,
    max_shots: int = None
) -> Dict[str, float]:
    """Decode using a specified decoder class."""
    H = data['H']
    logical_obs = data['logical_obs']
    syndromes = data['syndromes']
    actual_logicals = data['actual_logicals']
    error_probs = data['error_probs']
    erasure_masks = data['erasure_masks']
    num_shots = len(syndromes)
    if max_shots is not None:
        num_shots = min(num_shots, max_shots)
    num_logicals = logical_obs.shape[0]

    correct_count = 0
    converged_count = 0

    for shot in range(num_shots):
        try:
            # Create decoder with this shot's erasure mask
            if decoder_class == ErasureTensorNetworkDecoder:
                # For tensor network, we decode one logical at a time
                decoder = decoder_class(
                    H=H,
                    logical_obs=logical_obs[:1],  # First logical only for simplicity
                    error_probabilities=error_probs,
                    erasure_mask=erasure_masks[shot],
                    debug=(debug and shot == 0)
                )
            else:
                decoder = decoder_class(
                    H=H,
                    logical_obs=logical_obs[:1],
                    error_probabilities=error_probs,
                    erasure_mask=erasure_masks[shot]
                )

            # Decode
            result = decoder.decode(syndromes[shot].tolist())
            predicted = result['logical_error_prob'] > 0.5

            if result.get('converged', True):
                converged_count += 1

            # Check if prediction matches actual (first logical)
            actual = actual_logicals[shot, 0]
            if predicted == actual:
                correct_count += 1
        except Exception as e:
            if debug:
                print(f"  Shot {shot} error: {e}")
            # Count as incorrect on error
            pass

    ler_with_decoder = 1.0 - (correct_count / num_shots)

    return {
        'decoder': decoder_name,
        'logical_error_rate': ler_with_decoder,
        'num_shots': num_shots,
        'correct_count': correct_count,
        'converged_count': converged_count
    }


def run_bb_benchmark(
    bb_code_size: int,
    error_rates: List[float],
    erasure_rates: List[float],
    num_shots: int = 200,
    debug: bool = False
) -> Dict[str, Any]:
    """Run benchmark on a BB code."""

    # Generate BB code
    print(f"\n{'='*60}")
    print(f"BB Code Benchmark: N={bb_code_size}")
    print(f"{'='*60}")

    bb_code = gen_BB_code(bb_code_size)
    print(f"Code parameters: [[{bb_code.N}, {bb_code.K}, {bb_code.D}]]")

    if bb_code.K == 0:
        print("WARNING: Code has 0 logical qubits, skipping...")
        return None

    results = {
        'code_name': f'BB_{bb_code_size}',
        'N': int(bb_code.N),
        'K': int(bb_code.K),
        'D': int(bb_code.D) if not np.isnan(bb_code.D) else None,
        'timestamp': datetime.now().isoformat(),
        'num_shots': num_shots,
        'error_rates': error_rates,
        'erasure_rates': erasure_rates,
        'data': []
    }

    # Define decoders
    decoders = [
        (ErasureTensorNetworkDecoder, 'tensor_network_mld'),
    ]

    if LDPC_AVAILABLE:
        decoders.extend([
            (LdpcBpOsdDecoder, 'bp_osd'),
            (LdpcMinSumBpDecoder, 'min_sum_bp'),
        ])

    print(f"Decoders: {[d[1] for d in decoders]}")
    print(f"Error rates: {error_rates}")
    print(f"Erasure rates: {erasure_rates}")

    for erasure_rate in erasure_rates:
        for error_rate in error_rates:
            print(f"\n--- Error rate: {error_rate:.4f}, Erasure rate: {erasure_rate:.2f} ---")

            # Generate data
            data = generate_bb_code_data(
                bb_code=bb_code,
                error_rate=error_rate,
                erasure_rate=erasure_rate,
                num_shots=num_shots
            )

            config_results = {
                'error_rate': error_rate,
                'erasure_rate': erasure_rate,
                'decoders': {}
            }

            for decoder_class, decoder_name in decoders:
                print(f"  Running {decoder_name}...", end=' ', flush=True)

                try:
                    # Limit shots for slow tensor network decoder
                    max_shots = 20 if decoder_name == 'tensor_network_mld' else None
                    result = decode_with_decoder(
                        decoder_class, decoder_name, data, debug=debug,
                        max_shots=max_shots
                    )
                    config_results['decoders'][decoder_name] = result
                    print(f"LER = {result['logical_error_rate']:.4f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    config_results['decoders'][decoder_name] = {
                        'decoder': decoder_name,
                        'logical_error_rate': float('nan'),
                        'error': str(e)
                    }

            results['data'].append(config_results)

    return results


def main():
    """Run BB code benchmarks."""
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}

    # Test configurations - comprehensive range to find threshold
    error_rates = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    erasure_rates = [0.0, 0.1, 0.2]

    # Test BB_72 (smallest with K > 0)
    print("\n" + "="*60)
    print("BB CODE BENCHMARK")
    print("="*60)

    result_72 = run_bb_benchmark(
        bb_code_size=72,
        error_rates=error_rates,
        erasure_rates=erasure_rates,
        num_shots=100,  # More shots for better statistics
        debug=False
    )
    if result_72:
        all_results['BB_72'] = result_72

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"bb_benchmark_{timestamp}.json")

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}")

    # Print summary
    print("\n=== SUMMARY ===")
    for code_name, code_results in all_results.items():
        print(f"\n{code_name}:")
        for config in code_results['data'][:4]:
            er = config['error_rate']
            erasure = config['erasure_rate']
            print(f"  p={er:.3f}, erasure={erasure:.1f}:")
            for decoder_name, decoder_result in config['decoders'].items():
                ler = decoder_result.get('logical_error_rate', float('nan'))
                print(f"    {decoder_name}: LER={ler:.4f}")

    return all_results


if __name__ == "__main__":
    main()
