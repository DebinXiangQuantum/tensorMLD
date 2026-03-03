#!/usr/bin/env python3
"""
Phase 2: Run remaining QCC_60 TN, QCC_72 TN/BP-OSD jobs.

Uses fewer workers for TN (3) to avoid CPU contention that caused
20x slowdown in phase 1. BP-OSD jobs run separately with more workers.

Two-phase execution:
  Phase A: BP-OSD jobs (fast, 8 workers)
  Phase B: TN jobs (slow, 3 workers to avoid BLAS contention)
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from codes.gnd_ldpc_codes import (
    qcc_18_4_4, qcc_60_8_4, qcc_72_12_6,
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

MIN_ERRORS = 5
BATCH_SIZE = 200  # smaller batches for more frequent progress
ERROR_RATES = [0.001, 0.003, 0.005, 0.008, 0.01]
ERASURE_RATES = [0.0, 0.05, 0.1, 0.15]

MAX_SHOTS_MAP = {
    ('LDPC_30_6_4',  'tensor_network_mld'):  500_000,
    ('QCC_60_8_4',  'tensor_network_mld'):   50_000,
    ('QCC_72_12_6', 'tensor_network_mld'):   10_000,
}
DEFAULT_MAX_SHOTS = 2_000_000

TN_WORKERS = 3   # few workers to avoid BLAS/numpy CPU contention
BP_WORKERS = 8

PARTIAL_FILE = os.path.join(os.path.dirname(__file__), '..', 'results', 'tn_adaptive_partial.json')


def _get_max_shots(code_name, decoder_name):
    return MAX_SHOTS_MAP.get((code_name, decoder_name), DEFAULT_MAX_SHOTS)


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


def _run_adaptive_job(args: Tuple) -> Dict[str, Any]:
    code_name, error_rate, erasure_rate, decoder_name, seed = args

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    # Limit BLAS threads per worker to avoid contention
    for env in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[env] = '8'

    pid = os.getpid()
    max_shots = _get_max_shots(code_name, decoder_name)
    tag = f"[PID {pid}] {code_name} p={error_rate} e={erasure_rate} {decoder_name}"
    print(f"{tag} — starting (max_shots={max_shots:,})", flush=True)
    t0 = time.time()

    try:
        code = CODE_LOADERS[code_name]()
        H = code.hx.astype(np.float64)
        logical_obs = code.lz.astype(np.float64)
        error_probs = [error_rate] * int(code.N)
        rng = np.random.default_rng(seed)

        if decoder_name == 'tensor_network_mld':
            decoder = OptimizedErasureTNDecoder(
                H=H, logical_obs=logical_obs[:1],
                error_probabilities=error_probs,
            )
        else:
            decoder = None

        total_shots = 0
        total_errors = 0
        total_time_decode = 0.0

        while total_errors < MIN_ERRORS and total_shots < max_shots:
            batch = min(BATCH_SIZE, max_shots - total_shots)
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
            ms_per = (total_time_decode / total_shots * 1000) if total_shots > 0 else 0
            print(f"{tag} — {total_shots:,}/{max_shots:,} shots, {total_errors} errors, "
                  f"LER~{ler_est:.2e}, {ms_per:.1f}ms/shot ({elapsed:.0f}s)", flush=True)

        elapsed = time.time() - t0
        ler = total_errors / total_shots if total_shots > 0 else float('nan')
        per_shot_ms = (total_time_decode / total_shots * 1000) if total_shots > 0 else 0

        print(f"{tag} — DONE: {total_shots:,} shots, {total_errors} errors, "
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
            'max_shots_used': max_shots,
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


def _load_partial():
    if os.path.exists(PARTIAL_FILE):
        with open(PARTIAL_FILE) as f:
            return json.load(f)
    return []


def _completed_keys(results):
    return set(
        (r['code_name'], r['error_rate'], r['erasure_rate'], r['decoder'])
        for r in results
    )


def _run_phase(phase_name, jobs, max_workers, all_results):
    if not jobs:
        print(f"\n{phase_name}: no jobs to run")
        return

    print(f"\n{'='*60}")
    print(f"{phase_name}: {len(jobs)} jobs, {max_workers} workers")
    print(f"{'='*60}\n")

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
                print(f"[FATAL] {job[:4]}: {exc}")
                continue

            elapsed = time.time() - start_time
            eta = (elapsed / completed) * (len(jobs) - completed) if completed else 0
            print(f"[{phase_name} {completed}/{len(jobs)}] ETA {eta/60:.0f}min "
                  f"(total: {len(all_results)})", flush=True)

            # Save every 3 completions
            if completed % 3 == 0:
                with open(PARTIAL_FILE, 'w') as f:
                    json.dump(all_results, f, indent=2, default=str)
                print(f"  [saved {len(all_results)} results]", flush=True)

    # Save at end of phase
    with open(PARTIAL_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{phase_name} done. Total results: {len(all_results)}")


def main():
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    existing = _load_partial()
    done_keys = _completed_keys(existing)
    print(f"Loaded {len(existing)} existing results")

    # Build missing job lists
    bp_jobs = []
    tn_jobs = []
    seed = 200000

    for code_name in CODE_LOADERS:
        for erasure_rate in ERASURE_RATES:
            for error_rate in ERROR_RATES:
                for dec in ['tensor_network_mld', 'bp_osd']:
                    key = (code_name, error_rate, erasure_rate, dec)
                    if key not in done_keys:
                        job = (code_name, error_rate, erasure_rate, dec, seed)
                        if dec == 'bp_osd':
                            bp_jobs.append(job)
                        else:
                            tn_jobs.append(job)
                    seed += 1

    print(f"Missing BP-OSD jobs: {len(bp_jobs)}")
    print(f"Missing TN jobs: {len(tn_jobs)}")

    all_results = list(existing)

    # Phase A: BP-OSD (fast, many workers)
    if bp_jobs and LDPC_AVAILABLE:
        _run_phase("Phase A: BP-OSD", bp_jobs, BP_WORKERS, all_results)

    # Phase B: TN (slow, few workers to avoid BLAS contention)
    if tn_jobs:
        _run_phase("Phase B: TN-MLD", tn_jobs, TN_WORKERS, all_results)

    # Final save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"tn_adaptive_{timestamp}.json")
    output = {
        'timestamp': datetime.now().isoformat(),
        'min_errors': MIN_ERRORS,
        'results': all_results,
    }
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nFinal results: {len(all_results)}")
    print(f"Saved to: {results_file}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
