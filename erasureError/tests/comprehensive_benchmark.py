# ============================================================================ #
# Comprehensive qLDPC Code Benchmark for Erasure Error Decoding               #
# ============================================================================ #
"""
Benchmark tensor network MLD decoder against ldpc decoders on various qLDPC codes.

Codes tested:
- BB codes: [[18,4,4]], [[60,8,4]], [[72,12,6]]
- TB codes: [[25,3,4]], [[30,6,4]]
- HP codes: [[162,8,6]], [[50,2,5]]
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from codes.codes import gen_BB_code, gen_HP_ring_code, hypergraph_product, rep_code
from codes.gnd_ldpc_codes import (
    bb_18_4_4, bb_60_8_4, bb_72_12_6,
    tb_25_3_4, tb_30_6_4
)
from erasure_tensor_network_decoder import ErasureTensorNetworkDecoder
from simple_decoders import (
    LdpcBpOsdDecoder,
    LdpcMinSumBpDecoder,
    LdpcSequentialRelayBpDecoder,
    LDPC_AVAILABLE
)


def generate_syndrome_data(
    code,
    error_rate: float,
    erasure_rate: float,
    num_shots: int,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Generate syndrome data for a CSS code.
    For CSS codes, we only test X errors (hx detects X errors).
    """
    np.random.seed(random_seed)

    # Use hx as parity check matrix (detects X errors)
    H = code.hx.astype(np.float64)
    # Use lz as logical observable (X errors affect Z logicals)
    logical_obs = code.lz.astype(np.float64)

    num_checks, num_qubits = H.shape
    num_logicals = logical_obs.shape[0]

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
    max_shots: Optional[int] = None
) -> Dict[str, Any]:
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
    start_time = time.time()

    for shot in range(num_shots):
        try:
            # Create decoder with this shot's erasure mask
            if decoder_class == ErasureTensorNetworkDecoder:
                decoder = decoder_class(
                    H=H,
                    logical_obs=logical_obs[:1],  # First logical only
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
            pass

    elapsed_time = time.time() - start_time
    ler_with_decoder = 1.0 - (correct_count / num_shots) if num_shots > 0 else float('nan')

    return {
        'decoder': decoder_name,
        'logical_error_rate': ler_with_decoder,
        'num_shots': num_shots,
        'correct_count': correct_count,
        'converged_count': converged_count,
        'time_seconds': elapsed_time
    }


def run_code_benchmark(
    code,
    code_name: str,
    error_rates: List[float],
    erasure_rates: List[float],
    num_shots: int = 500,
    tn_max_shots: int = 50,
    debug: bool = False
) -> Dict[str, Any]:
    """Run benchmark on a single code."""

    print(f"\n{'='*60}")
    print(f"Benchmarking: {code_name}")
    print(f"Code parameters: [[{code.N}, {code.K}, {code.D}]]")
    print(f"{'='*60}")

    if code.K == 0:
        print("WARNING: Code has 0 logical qubits, skipping...")
        return None

    results = {
        'code_name': code_name,
        'N': int(code.N),
        'K': int(code.K),
        'D': int(code.D) if not np.isnan(code.D) else None,
        'num_shots': num_shots,
        'tn_max_shots': tn_max_shots,
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
            (LdpcSequentialRelayBpDecoder, 'relaybp'),
        ])

    print(f"Decoders: {[d[1] for d in decoders]}")
    print(f"Error rates: {error_rates}")
    print(f"Erasure rates: {erasure_rates}")
    print(f"Shots: {num_shots} (TN max: {tn_max_shots})")

    for erasure_rate in erasure_rates:
        for error_rate in error_rates:
            print(f"\n--- p={error_rate:.4f}, erasure={erasure_rate:.2f} ---")

            # Generate data
            data = generate_syndrome_data(
                code=code,
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
                print(f"  {decoder_name}...", end=' ', flush=True)

                try:
                    max_shots = tn_max_shots if decoder_name == 'tensor_network_mld' else None
                    result = decode_with_decoder(
                        decoder_class, decoder_name, data,
                        debug=debug, max_shots=max_shots
                    )
                    config_results['decoders'][decoder_name] = result
                    ler = result['logical_error_rate']
                    t = result['time_seconds']
                    print(f"LER={ler:.4f} ({t:.1f}s)")
                except Exception as e:
                    print(f"ERROR: {e}")
                    config_results['decoders'][decoder_name] = {
                        'decoder': decoder_name,
                        'logical_error_rate': float('nan'),
                        'error': str(e)
                    }

            results['data'].append(config_results)

    return results


def get_all_codes() -> List[tuple]:
    """Get all codes to benchmark."""
    codes = []

    # BB codes from GND files
    print("Loading BB codes from GND files...")
    try:
        codes.append(('BB_18_4_4', bb_18_4_4()))
        print("  BB_18_4_4 loaded")
    except Exception as e:
        print(f"  BB_18_4_4 failed: {e}")

    try:
        codes.append(('BB_60_8_4', bb_60_8_4()))
        print("  BB_60_8_4 loaded")
    except Exception as e:
        print(f"  BB_60_8_4 failed: {e}")

    try:
        codes.append(('BB_72_12_6', bb_72_12_6()))
        print("  BB_72_12_6 loaded")
    except Exception as e:
        print(f"  BB_72_12_6 failed: {e}")

    # TB codes from GND files
    print("\nLoading TB codes from GND files...")
    try:
        codes.append(('TB_25_3_4', tb_25_3_4()))
        print("  TB_25_3_4 loaded")
    except Exception as e:
        print(f"  TB_25_3_4 failed: {e}")

    try:
        codes.append(('TB_30_6_4', tb_30_6_4()))
        print("  TB_30_6_4 loaded")
    except Exception as e:
        print(f"  TB_30_6_4 failed: {e}")

    # HP codes from code generation
    print("\nGenerating HP codes...")
    try:
        hp_code = gen_HP_ring_code(5, 5)  # Small HP code
        codes.append(('HP_50', hp_code))
        print(f"  HP_50 generated: [[{hp_code.N}, {hp_code.K}, {hp_code.D}]]")
    except Exception as e:
        print(f"  HP_50 failed: {e}")

    try:
        hp_code = gen_HP_ring_code(7, 7)  # Larger HP code
        codes.append(('HP_98', hp_code))
        print(f"  HP_98 generated: [[{hp_code.N}, {hp_code.K}, {hp_code.D}]]")
    except Exception as e:
        print(f"  HP_98 failed: {e}")

    return codes


def main():
    """Run comprehensive benchmark."""
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("COMPREHENSIVE qLDPC CODE BENCHMARK")
    print("=" * 70)

    # Get all codes
    all_codes = get_all_codes()

    if not all_codes:
        print("ERROR: No codes available for benchmarking")
        return

    print(f"\nTotal codes to benchmark: {len(all_codes)}")

    # Benchmark parameters
    error_rates = [0.001,0.003, 0.005, 0.008, 0.01]
    erasure_rates = [0.0, 0.05, 0.1,0.15]
    num_shots = 50000  # More shots for better statistics
    tn_max_shots = 50000  # Limit for slow tensor network decoder

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'error_rates': error_rates,
        'erasure_rates': erasure_rates,
        'num_shots': num_shots,
        'tn_max_shots': tn_max_shots,
        'codes': {}
    }

    for code_name, code in all_codes:
        try:
            result = run_code_benchmark(
                code=code,
                code_name=code_name,
                error_rates=error_rates,
                erasure_rates=erasure_rates,
                num_shots=num_shots,
                tn_max_shots=tn_max_shots,
                debug=False
            )
            if result:
                all_results['codes'][code_name] = result
        except Exception as e:
            print(f"\nERROR benchmarking {code_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"comprehensive_benchmark_{timestamp}.json")

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}")

    # Print summary
    print("\n=== SUMMARY ===")
    for code_name, code_results in all_results['codes'].items():
        print(f"\n{code_name} [[{code_results['N']}, {code_results['K']}, {code_results['D']}]]:")
        for config in code_results['data'][:3]:  # Show first 3 configs
            er = config['error_rate']
            erasure = config['erasure_rate']
            print(f"  p={er:.3f}, erasure={erasure:.1f}:")
            for decoder_name, decoder_result in config['decoders'].items():
                ler = decoder_result.get('logical_error_rate', float('nan'))
                below = "< p" if ler < er else ">= p"
                print(f"    {decoder_name}: LER={ler:.4f} ({below})")

    return all_results


if __name__ == "__main__":
    main()
