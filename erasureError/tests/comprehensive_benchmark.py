# ============================================================================ #
# Comprehensive qLDPC Code Benchmark for Erasure Error Decoding               #
# Multi-threaded version                                                       #
# ============================================================================ #
"""
Benchmark tensor network MLD decoder against ldpc decoders on various qLDPC codes.

Uses multiprocessing to parallelise across (code, erasure_rate, error_rate, decoder)
configurations. Each worker handles one full (code, params, decoder) job so that
the GIL is not a bottleneck.

Codes tested (ISCA revision benchmark set):
- LDPC codes: [[25,3,4]], [[30,6,4]]
- Toric code: [[50,2,5]]
- QCC codes: [[18,4,4]], [[60,8,4]], [[72,12,6]], [[90,8,10]], [[108,8,10]], [[144,12,12]]
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from codes.codes import gen_BB_code, gen_HP_ring_code, hypergraph_product, rep_code
from codes.gnd_ldpc_codes import (
    qcc_18_4_4, qcc_60_8_4, qcc_72_12_6,
    qcc_90_8_10, qcc_108_8_10, qcc_144_12_12,
    ldpc_25_3_4, ldpc_30_6_4, tor_50_2_5,
)
from erasure_tensor_network_decoder import ErasureTensorNetworkDecoder
from simple_decoders import (
    LdpcBpOsdDecoder,
    LdpcMinSumBpDecoder,
    LdpcSequentialRelayBpDecoder,
    LDPC_AVAILABLE
)

# ---------------------------------------------------------------------------
# Decoder registry – maps name → class so we can pickle job descriptions
# ---------------------------------------------------------------------------
DECODER_REGISTRY: Dict[str, type] = {
    'tensor_network_mld': ErasureTensorNetworkDecoder,
}
if LDPC_AVAILABLE:
    DECODER_REGISTRY['bp_osd'] = LdpcBpOsdDecoder
    DECODER_REGISTRY['min_sum_bp'] = LdpcMinSumBpDecoder
    DECODER_REGISTRY['relaybp'] = LdpcSequentialRelayBpDecoder


# ---------------------------------------------------------------------------
# Data generation (unchanged, runs in worker)
# ---------------------------------------------------------------------------
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

    H = code.hx.astype(np.float64)
    logical_obs = code.lz.astype(np.float64)

    num_checks, num_qubits = H.shape
    num_logicals = logical_obs.shape[0]

    syndromes = np.zeros((num_shots, num_checks), dtype=np.float64)
    actual_logicals = np.zeros((num_shots, num_logicals), dtype=bool)
    erasure_masks = np.zeros((num_shots, num_qubits), dtype=bool)
    error_probs = np.full(num_qubits, error_rate)

    for shot in range(num_shots):
        erasure_mask = np.random.rand(num_qubits) < erasure_rate

        errors = np.zeros(num_qubits, dtype=bool)
        for i in range(num_qubits):
            if erasure_mask[i]:
                errors[i] = np.random.rand() < 0.5
            else:
                errors[i] = np.random.rand() < error_rate

        syndrome = (H @ errors.astype(np.int32)) % 2
        syndromes[shot] = syndrome.astype(np.float64)
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


# ---------------------------------------------------------------------------
# Single-decoder decoding loop
# ---------------------------------------------------------------------------
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

    correct_count = 0
    converged_count = 0
    start_time = time.time()

    for shot in range(num_shots):
        try:
            if decoder_class == ErasureTensorNetworkDecoder:
                decoder = decoder_class(
                    H=H,
                    logical_obs=logical_obs[:1],
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

            result = decoder.decode(syndromes[shot].tolist())
            predicted = result['logical_error_prob'] > 0.5

            if result.get('converged', True):
                converged_count += 1

            actual = actual_logicals[shot, 0]
            if predicted == actual:
                correct_count += 1
        except Exception as e:
            if debug:
                print(f"  Shot {shot} error: {e}")

    elapsed_time = time.time() - start_time
    ler = 1.0 - (correct_count / num_shots) if num_shots > 0 else float('nan')

    return {
        'decoder': decoder_name,
        'logical_error_rate': ler,
        'num_shots': num_shots,
        'correct_count': correct_count,
        'converged_count': converged_count,
        'time_seconds': elapsed_time
    }


# ---------------------------------------------------------------------------
# Code loaders – picklable factories
# ---------------------------------------------------------------------------
CODE_LOADER_REGISTRY: Dict[str, Any] = {
    'LDPC_25_3_4': lambda: ldpc_25_3_4(),
    'LDPC_30_6_4': lambda: ldpc_30_6_4(),
    'TOR_50_2_5': lambda: tor_50_2_5(),
    'QCC_18_4_4': lambda: qcc_18_4_4(),
    'QCC_60_8_4': lambda: qcc_60_8_4(),
    'QCC_72_12_6': lambda: qcc_72_12_6(),
    'QCC_90_8_10': lambda: qcc_90_8_10(),
    'QCC_108_8_10': lambda: qcc_108_8_10(),
    'QCC_144_12_12': lambda: qcc_144_12_12(),
}


# ---------------------------------------------------------------------------
# Worker function — runs one (code, erasure_rate, error_rate, decoder) job
# ---------------------------------------------------------------------------
def _run_single_job(args: Tuple) -> Dict[str, Any]:
    """Worker: run one decoder on one (code, error_rate, erasure_rate) config.

    Returns a dict with enough metadata to reassemble results in the main
    process.
    """
    (code_name, code_loader_name, error_rate, erasure_rate,
     decoder_name, num_shots, tn_max_shots, random_seed) = args

    # Reload paths inside worker (forked process may not inherit sys.path)
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    pid = os.getpid()
    tag = f"[PID {pid}] {code_name} p={error_rate} e={erasure_rate} {decoder_name}"
    print(f"{tag} — starting", flush=True)
    t0 = time.time()

    try:
        # Load code inside worker
        code = CODE_LOADER_REGISTRY[code_loader_name]()

        if code.K == 0:
            return {
                'code_name': code_name,
                'error_rate': error_rate,
                'erasure_rate': erasure_rate,
                'decoder_name': decoder_name,
                'result': None,
                'error': 'code has 0 logical qubits',
                'code_params': {'N': int(code.N), 'K': 0, 'D': None},
            }

        # Generate syndrome data
        data = generate_syndrome_data(
            code=code,
            error_rate=error_rate,
            erasure_rate=erasure_rate,
            num_shots=num_shots,
            random_seed=random_seed
        )

        # Resolve decoder class
        decoder_class = DECODER_REGISTRY[decoder_name]
        max_shots = tn_max_shots if decoder_name == 'tensor_network_mld' else None

        result = decode_with_decoder(
            decoder_class, decoder_name, data,
            debug=False, max_shots=max_shots
        )

        elapsed = time.time() - t0
        ler = result['logical_error_rate']
        print(f"{tag} — done  LER={ler:.6f} ({elapsed:.1f}s)", flush=True)

        return {
            'code_name': code_name,
            'error_rate': error_rate,
            'erasure_rate': erasure_rate,
            'decoder_name': decoder_name,
            'result': result,
            'error': None,
            'code_params': {
                'N': int(code.N),
                'K': int(code.K),
                'D': int(code.D) if not np.isnan(code.D) else None,
            },
        }

    except Exception as exc:
        import traceback
        elapsed = time.time() - t0
        print(f"{tag} — FAILED ({elapsed:.1f}s): {exc}", flush=True)
        traceback.print_exc()
        return {
            'code_name': code_name,
            'error_rate': error_rate,
            'erasure_rate': erasure_rate,
            'decoder_name': decoder_name,
            'result': None,
            'error': str(exc),
            'code_params': None,
        }


# ---------------------------------------------------------------------------
# Load codes (single-process, for param reporting only)
# ---------------------------------------------------------------------------
def get_all_codes() -> List[Tuple[str, str]]:
    """Return (code_name, loader_key) pairs for codes that load successfully."""
    codes = []
    for name, loader in CODE_LOADER_REGISTRY.items():
        try:
            code = loader()
            print(f"  {name} OK  [[{code.N}, {code.K}, {code.D}]]")
            codes.append((name, name))  # (display_name, loader_key)
        except Exception as e:
            print(f"  {name} FAILED: {e}")
    return codes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run comprehensive benchmark using multiprocessing."""
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("COMPREHENSIVE qLDPC CODE BENCHMARK  (multi-process)")
    print("=" * 70)

    # Verify which codes load
    print("\nChecking codes...")
    all_codes = get_all_codes()
    if not all_codes:
        print("ERROR: No codes available for benchmarking")
        return

    print(f"\nCodes: {[c[0] for c in all_codes]}")
    print(f"Decoders: {list(DECODER_REGISTRY.keys())}")

    # Benchmark parameters
    error_rates = [0.001, 0.003, 0.005, 0.008, 0.01]
    erasure_rates = [0.0, 0.05, 0.1, 0.15]
    num_shots = 1_000_000
    tn_max_shots = 1_000_000
    random_seed = 42

    # Build job list
    jobs = []
    for code_name, loader_key in all_codes:
        for erasure_rate in erasure_rates:
            for error_rate in error_rates:
                for decoder_name in DECODER_REGISTRY:
                    jobs.append((
                        code_name, loader_key,
                        error_rate, erasure_rate,
                        decoder_name,
                        num_shots, tn_max_shots, random_seed
                    ))

    total_jobs = len(jobs)
    # Use all available CPUs but cap to avoid memory issues
    max_workers = min(mp.cpu_count(), total_jobs, 32)
    print(f"\nTotal jobs: {total_jobs}")
    print(f"Workers: {max_workers}  (CPUs: {mp.cpu_count()})")
    print(f"Shots per job: {num_shots}  (TN max: {tn_max_shots})")
    print(f"Starting...\n")

    # Collect results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'error_rates': error_rates,
        'erasure_rates': erasure_rates,
        'num_shots': num_shots,
        'tn_max_shots': tn_max_shots,
        'max_workers': max_workers,
        'codes': {}
    }

    completed = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_single_job, job): job for job in jobs}

        for future in as_completed(futures):
            completed += 1
            try:
                out = future.result()
            except Exception as exc:
                job = futures[future]
                print(f"[FATAL] Job {job[:5]} raised: {exc}")
                continue

            code_name = out['code_name']
            er = out['error_rate']
            erasure = out['erasure_rate']
            dname = out['decoder_name']

            # Ensure structure exists
            if code_name not in all_results['codes']:
                cp = out.get('code_params') or {}
                all_results['codes'][code_name] = {
                    'code_name': code_name,
                    'N': cp.get('N'),
                    'K': cp.get('K'),
                    'D': cp.get('D'),
                    'num_shots': num_shots,
                    'tn_max_shots': tn_max_shots,
                    'data': []
                }

            code_entry = all_results['codes'][code_name]

            # Find or create the config entry for (error_rate, erasure_rate)
            config_entry = None
            for d in code_entry['data']:
                if d['error_rate'] == er and d['erasure_rate'] == erasure:
                    config_entry = d
                    break
            if config_entry is None:
                config_entry = {
                    'error_rate': er,
                    'erasure_rate': erasure,
                    'decoders': {}
                }
                code_entry['data'].append(config_entry)

            if out['result'] is not None:
                config_entry['decoders'][dname] = out['result']
            else:
                config_entry['decoders'][dname] = {
                    'decoder': dname,
                    'logical_error_rate': float('nan'),
                    'error': out.get('error', 'unknown')
                }

            # Progress
            elapsed = time.time() - start_time
            eta = (elapsed / completed) * (total_jobs - completed) if completed else 0
            print(f"[{completed}/{total_jobs}] ETA {eta/60:.0f}min — "
                  f"last: {code_name} p={er} e={erasure} {dname}", flush=True)

            # Periodic save every 20 jobs
            if completed % 20 == 0:
                partial_file = os.path.join(results_dir, "comprehensive_benchmark_partial.json")
                with open(partial_file, 'w') as f:
                    json.dump(all_results, f, indent=2)

    # Final save
    total_time = time.time() - start_time
    all_results['total_time_seconds'] = total_time

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"comprehensive_benchmark_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"All done in {total_time/60:.1f} minutes")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}")

    # Print summary
    print("\n=== SUMMARY ===")
    for code_name, code_results in all_results['codes'].items():
        N = code_results.get('N', '?')
        K = code_results.get('K', '?')
        D = code_results.get('D', '?')
        print(f"\n{code_name} [[{N}, {K}, {D}]]:")
        for config in sorted(code_results['data'],
                             key=lambda c: (c['erasure_rate'], c['error_rate']))[:6]:
            er = config['error_rate']
            erasure = config['erasure_rate']
            print(f"  p={er:.4f}, erasure={erasure:.2f}:")
            for decoder_name, decoder_result in config['decoders'].items():
                ler = decoder_result.get('logical_error_rate', float('nan'))
                below = "< p" if ler < er else ">= p"
                print(f"    {decoder_name}: LER={ler:.6f} ({below})")

    return all_results


if __name__ == "__main__":
    # Use 'spawn' to avoid forking issues with numpy/quimb
    mp.set_start_method('spawn', force=True)
    main()
