#!/usr/bin/env python3
"""
Adaptive-shot TN MLD benchmark.

For each (code, error_rate, erasure_rate), keeps sampling until
`min_errors` logical errors are observed, giving reliable LER estimates
even for very strong codes (QCC_60, QCC_72) where fixed-shot runs
see zero errors.

Also runs BP-OSD with the same adaptive stopping for fair comparison.
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from codes.gnd_ldpc_codes import (
    qcc_18_4_4, qcc_60_8_4, qcc_72_12_6,
    qcc_90_8_10, qcc_108_8_10, qcc_144_12_12,
    ldpc_25_3_4, ldpc_30_6_4, tor_50_2_5,
)
from erasure_tensor_network_decoder import OptimizedErasureTNDecoder
from simple_decoders import LdpcBpOsdDecoder, LDPC_AVAILABLE

CODE_LOADERS = {
    'QCC_18_4_4':    qcc_18_4_4,
    'LDPC_25_3_4':   ldpc_25_3_4,
    'LDPC_30_6_4':   ldpc_30_6_4,
    'TOR_50_2_5':    tor_50_2_5,
    'QCC_60_8_4':    qcc_60_8_4,
    'QCC_72_12_6':   qcc_72_12_6,
}

# ── Parameters ──────────────────────────────────────────────────────────
MIN_ERRORS = 5          # stop after this many logical errors
MAX_SHOTS  = 2_000_000  # hard cap to avoid infinite loops
BATCH_SIZE = 1_000      # generate & decode in batches of this size
ERROR_RATES = [0.001, 0.003, 0.005, 0.008, 0.01]
ERASURE_RATES = [0.0, 0.05, 0.1, 0.15]


def _generate_batch(code, error_rate, erasure_rate, batch_size, rng):
    """Generate a batch of syndrome data using the given RNG."""
    H = code.hx.astype(np.float64)
    logical_obs = code.lz.astype(np.float64)
    num_checks, num_qubits = H.shape
    num_logicals = logical_obs.shape[0]

    syndromes = np.zeros((batch_size, num_checks), dtype=np.float64)
    actual_logicals = np.zeros((batch_size, num_logicals), dtype=bool)
    erasure_masks = np.zeros((batch_size, num_qubits), dtype=bool)

    for shot in range(batch_size):
        erasure_mask = rng.random(num_qubits) < erasure_rate
        errors = np.zeros(num_qubits, dtype=bool)
        for i in range(num_qubits):
            if erasure_mask[i]:
                errors[i] = rng.random() < 0.5
            else:
                errors[i] = rng.random() < error_rate
        syndrome = (H @ errors.astype(np.int32)) % 2
        syndromes[shot] = syndrome.astype(np.float64)
        actual_logicals[shot] = ((logical_obs @ errors.astype(np.int32)) % 2).astype(bool)
        erasure_masks[shot] = erasure_mask

    return syndromes, actual_logicals, erasure_masks


def _run_adaptive_job(args: Tuple) -> Dict[str, Any]:
    """Worker: run TN and BP-OSD with adaptive stopping."""
    code_name, error_rate, erasure_rate, decoder_name, seed = args

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    pid = os.getpid()
    tag = f"[PID {pid}] {code_name} p={error_rate} e={erasure_rate} {decoder_name}"
    print(f"{tag} — starting", flush=True)
    t0 = time.time()

    try:
        code = CODE_LOADERS[code_name]()
        H = code.hx.astype(np.float64)
        logical_obs = code.lz.astype(np.float64)
        error_probs = [error_rate] * int(code.N)
        rng = np.random.default_rng(seed)

        # Create decoder once
        if decoder_name == 'tensor_network_mld':
            decoder = OptimizedErasureTNDecoder(
                H=H, logical_obs=logical_obs[:1],
                error_probabilities=error_probs,
            )
        else:
            decoder = None  # BP-OSD is rebuilt per shot (erasure_mask changes)

        total_shots = 0
        total_errors = 0
        total_time_decode = 0.0

        while total_errors < MIN_ERRORS and total_shots < MAX_SHOTS:
            batch = min(BATCH_SIZE, MAX_SHOTS - total_shots)
            syndromes, actual_logicals, erasure_masks = _generate_batch(
                code, error_rate, erasure_rate, batch, rng
            )

            t_dec_start = time.time()
            for shot in range(batch):
                if decoder_name == 'tensor_network_mld':
                    result = decoder.decode(
                        syndromes[shot], erasure_mask=erasure_masks[shot]
                    )
                else:
                    bp_dec = LdpcBpOsdDecoder(
                        H=H, logical_obs=logical_obs[:1],
                        error_probabilities=error_probs,
                        erasure_mask=erasure_masks[shot],
                    )
                    result = bp_dec.decode(syndromes[shot].tolist())

                predicted = result['logical_error_prob'] > 0.5
                actual = actual_logicals[shot, 0]
                if predicted != actual:
                    total_errors += 1

                total_shots += 1
                if total_errors >= MIN_ERRORS:
                    break

            total_time_decode += time.time() - t_dec_start

            elapsed = time.time() - t0
            ler_est = total_errors / total_shots if total_shots > 0 else 0
            print(f"{tag} — {total_shots} shots, {total_errors} errors, "
                  f"LER~{ler_est:.2e} ({elapsed:.0f}s)", flush=True)

        elapsed = time.time() - t0
        ler = total_errors / total_shots if total_shots > 0 else float('nan')
        per_shot_ms = (total_time_decode / total_shots * 1000) if total_shots > 0 else 0

        print(f"{tag} — DONE: {total_shots} shots, {total_errors} errors, "
              f"LER={ler:.6e}, {per_shot_ms:.2f}ms/shot ({elapsed:.1f}s)", flush=True)

        return {
            'code_name': code_name,
            'error_rate': error_rate,
            'erasure_rate': erasure_rate,
            'decoder': decoder_name,
            'logical_error_rate': ler,
            'num_shots': total_shots,
            'num_errors': total_errors,
            'per_shot_ms': per_shot_ms,
            'time_seconds': elapsed,
            'converged': total_errors >= MIN_ERRORS,
            'code_params': {
                'N': int(code.N), 'K': int(code.K),
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
            'decoder': decoder_name,
            'logical_error_rate': float('nan'),
            'num_shots': 0,
            'num_errors': 0,
            'error': str(exc),
            'time_seconds': elapsed,
            'converged': False,
        }


def main():
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("ADAPTIVE-SHOT TN MLD + BP-OSD BENCHMARK")
    print(f"Min errors: {MIN_ERRORS}, Max shots: {MAX_SHOTS:,}")
    print("=" * 70)

    # Check codes
    print("\nChecking codes...")
    valid_codes = []
    for name, loader in CODE_LOADERS.items():
        try:
            code = loader()
            print(f"  {name} OK  [[{code.N}, {code.K}, {code.D}]]")
            valid_codes.append(name)
        except Exception as e:
            print(f"  {name} FAILED: {e}")

    decoders = ['tensor_network_mld']
    if LDPC_AVAILABLE:
        decoders.append('bp_osd')

    # Build job list
    jobs = []
    seed = 12345
    for code_name in valid_codes:
        for erasure_rate in ERASURE_RATES:
            for error_rate in ERROR_RATES:
                for dec in decoders:
                    jobs.append((code_name, error_rate, erasure_rate, dec, seed))
                    seed += 1

    total_jobs = len(jobs)
    max_workers = min(mp.cpu_count(), total_jobs, 16)

    print(f"\nTotal jobs: {total_jobs}")
    print(f"Workers: {max_workers}  (CPUs: {mp.cpu_count()})")
    print(f"Decoders: {decoders}")
    print(f"Starting...\n")

    all_results = []
    completed = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_adaptive_job, job): job for job in jobs}

        for future in as_completed(futures):
            completed += 1
            try:
                out = future.result()
                all_results.append(out)
            except Exception as exc:
                job = futures[future]
                print(f"[FATAL] Job {job[:4]} raised: {exc}")
                continue

            elapsed = time.time() - start_time
            eta = (elapsed / completed) * (total_jobs - completed) if completed else 0
            print(f"[{completed}/{total_jobs}] ETA {eta/60:.0f}min", flush=True)

            # Periodic save
            if completed % 10 == 0:
                partial_file = os.path.join(results_dir, "tn_adaptive_partial.json")
                with open(partial_file, 'w') as f:
                    json.dump(all_results, f, indent=2, default=str)

    # Final save
    total_time = time.time() - start_time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"tn_adaptive_{timestamp}.json")

    output = {
        'timestamp': datetime.now().isoformat(),
        'min_errors': MIN_ERRORS,
        'max_shots': MAX_SHOTS,
        'batch_size': BATCH_SIZE,
        'error_rates': ERROR_RATES,
        'erasure_rates': ERASURE_RATES,
        'total_time_seconds': total_time,
        'results': all_results,
    }
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"All done in {total_time/60:.1f} minutes")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}")

    # Summary
    print("\n=== SUMMARY ===")
    for code_name in valid_codes:
        code_results = [r for r in all_results if r['code_name'] == code_name]
        if not code_results:
            continue
        cp = code_results[0].get('code_params', {})
        print(f"\n{code_name} [[{cp.get('N','?')},{cp.get('K','?')},{cp.get('D','?')}]]:")
        for er in sorted(set(r['erasure_rate'] for r in code_results)):
            for p in sorted(set(r['error_rate'] for r in code_results)):
                for dec in decoders:
                    matches = [r for r in code_results
                               if r['error_rate'] == p and r['erasure_rate'] == er
                               and r['decoder'] == dec]
                    if matches:
                        r = matches[0]
                        ler = r['logical_error_rate']
                        shots = r['num_shots']
                        errs = r['num_errors']
                        conv = "Y" if r.get('converged') else "N"
                        ms = r.get('per_shot_ms', 0)
                        print(f"  p={p:.3f} e={er:.2f} {dec:>20s}: "
                              f"LER={ler:.2e} ({errs}/{shots} errs, "
                              f"conv={conv}, {ms:.1f}ms/shot)")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
