#!/usr/bin/env python3
"""
QCC_72 TN-MLD on H100 (256 cores, all threads, 1 worker).

MIN_ERRORS=2 for faster convergence. Single worker to maximize
per-shot throughput with all CPU threads available.
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from codes.gnd_ldpc_codes import qcc_72_12_6
from erasure_tensor_network_decoder import OptimizedErasureTNDecoder

MIN_ERRORS = 2
MAX_SHOTS = 10_000
BATCH_SIZE = 200
ERROR_RATES = [0.001, 0.003, 0.005, 0.008, 0.01]
ERASURE_RATES = [0.0, 0.05, 0.1, 0.15]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'qcc72_h100_results.json')


def _generate_batch(code, error_rate, erasure_rate, batch_size, rng):
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


def run_job(error_rate, erasure_rate, seed):
    tag = f"QCC_72 p={error_rate} e={erasure_rate}"
    print(f"{tag} — starting (max_shots={MAX_SHOTS:,}, min_errors={MIN_ERRORS})", flush=True)
    t0 = time.time()

    try:
        code = qcc_72_12_6()
        H = code.hx.astype(np.float64)
        logical_obs = code.lz.astype(np.float64)
        error_probs = [error_rate] * int(code.N)
        rng = np.random.default_rng(seed)

        decoder = OptimizedErasureTNDecoder(
            H=H, logical_obs=logical_obs[:1],
            error_probabilities=error_probs,
        )

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
                result = decoder.decode(
                    syndromes[shot], erasure_mask=erasure_masks[shot]
                )

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
            ms_per = (total_time_decode / total_shots * 1000) if total_shots > 0 else 0
            print(f"{tag} — {total_shots:,}/{MAX_SHOTS:,} shots, {total_errors} errors, "
                  f"LER~{ler_est:.2e}, {ms_per:.1f}ms/shot ({elapsed:.0f}s)", flush=True)

        elapsed = time.time() - t0
        ler = total_errors / total_shots if total_shots > 0 else float('nan')
        per_shot_ms = (total_time_decode / total_shots * 1000) if total_shots > 0 else 0

        print(f"{tag} — DONE: {total_shots:,} shots, {total_errors} errors, "
              f"LER={ler:.6e}, {per_shot_ms:.2f}ms/shot ({elapsed:.1f}s)", flush=True)

        return {
            'code_name': 'QCC_72_12_6',
            'error_rate': error_rate,
            'erasure_rate': erasure_rate,
            'decoder': 'tensor_network_mld',
            'logical_error_rate': ler,
            'num_shots': total_shots,
            'num_errors': total_errors,
            'per_shot_ms': per_shot_ms,
            'time_seconds': elapsed,
            'converged': total_errors >= MIN_ERRORS,
            'max_shots_used': MAX_SHOTS,
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
            'code_name': 'QCC_72_12_6',
            'error_rate': error_rate,
            'erasure_rate': erasure_rate,
            'decoder': 'tensor_network_mld',
            'logical_error_rate': float('nan'),
            'num_shots': 0,
            'num_errors': 0,
            'error': str(exc),
            'time_seconds': elapsed,
            'converged': False,
        }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Sort jobs: high-p first for fast early-stop
    jobs = []
    seed = 400000
    for error_rate in sorted(ERROR_RATES, reverse=True):
        for erasure_rate in sorted(ERASURE_RATES, reverse=True):
            jobs.append((error_rate, erasure_rate, seed))
            seed += 1

    print("=" * 60)
    print(f"QCC_72 TN-MLD on H100 — {len(jobs)} jobs, 1 worker (all threads)")
    print(f"MIN_ERRORS={MIN_ERRORS}, MAX_SHOTS={MAX_SHOTS:,}")
    print("=" * 60)

    # Quick code check
    code = qcc_72_12_6()
    print(f"QCC_72: [[{code.N}, {code.K}, {code.D}]]")
    print(f"CPU count: {mp.cpu_count()}")
    print(f"\nStarting {len(jobs)} jobs sequentially...\n")

    all_results = []
    start_time = time.time()

    for i, (error_rate, erasure_rate, seed) in enumerate(jobs):
        result = run_job(error_rate, erasure_rate, seed)
        all_results.append(result)

        elapsed = time.time() - start_time
        done = i + 1
        eta = (elapsed / done) * (len(jobs) - done) if done > 0 else 0
        print(f"[{done}/{len(jobs)}] ETA {eta/60:.0f}min\n", flush=True)

        # Save after each job
        with open(OUTPUT_FILE, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'min_errors': MIN_ERRORS,
                'max_shots': MAX_SHOTS,
                'results': all_results,
            }, f, indent=2, default=str)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"All done in {total_time/3600:.1f} hours")
    print(f"Results: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
