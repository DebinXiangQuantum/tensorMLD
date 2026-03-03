#!/usr/bin/env python3
"""
Phase 3: Run remaining QCC_60 + QCC_72 TN jobs with optimal thread settings.

Key changes from Phase 2:
  - OMP_NUM_THREADS=20 per worker (was 8, caused 6x slowdown on QCC_60)
  - 2 TN workers (was 3), so 2×20=40 threads on 64-core machine
  - Jobs sorted: high-p first (fast early-stop) to get results sooner
  - LDPC_30 runs first (fast single job)
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

CODE_LOADERS = {
    'QCC_18_4_4':    qcc_18_4_4,
    'LDPC_25_3_4':   ldpc_25_3_4,
    'LDPC_30_6_4':   ldpc_30_6_4,
    'TOR_50_2_5':    tor_50_2_5,
    'QCC_60_8_4':    qcc_60_8_4,
    'QCC_72_12_6':   qcc_72_12_6,
}

MIN_ERRORS = 5
BATCH_SIZE = 200
ERROR_RATES = [0.001, 0.003, 0.005, 0.008, 0.01]
ERASURE_RATES = [0.0, 0.05, 0.1, 0.15]

MAX_SHOTS_MAP = {
    ('LDPC_30_6_4',  'tensor_network_mld'):  500_000,
    ('QCC_60_8_4',  'tensor_network_mld'):   50_000,
    ('QCC_72_12_6', 'tensor_network_mld'):   10_000,
}
DEFAULT_MAX_SHOTS = 500_000

TN_WORKERS = 2
THREADS_PER_WORKER = 20  # 2 workers × 20 = 40 threads on 64-core

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

    # Set BLAS threads: 20 per worker, 2 workers = 40 threads total
    for env in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[env] = str(THREADS_PER_WORKER)

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

        decoder = OptimizedErasureTNDecoder(
            H=H, logical_obs=logical_obs[:1],
            error_probabilities=error_probs,
        )

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


def main():
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    existing = _load_partial()
    done_keys = _completed_keys(existing)
    print(f"Loaded {len(existing)} existing results")

    # Build TN-only missing jobs, sorted by error_rate DESC (high-p first for fast early-stop)
    tn_jobs = []
    seed = 300000

    for code_name in CODE_LOADERS:
        for erasure_rate in ERASURE_RATES:
            for error_rate in ERROR_RATES:
                key = (code_name, error_rate, erasure_rate, 'tensor_network_mld')
                if key not in done_keys:
                    tn_jobs.append((code_name, error_rate, erasure_rate, 'tensor_network_mld', seed))
                seed += 1

    # Sort: LDPC_30 first (fast), then high-p first within each code
    def sort_key(job):
        code, p, er, dec, s = job
        code_order = {'LDPC_30_6_4': 0, 'QCC_60_8_4': 1, 'QCC_72_12_6': 2}
        return (code_order.get(code, 0), -p, -er)

    tn_jobs.sort(key=sort_key)

    if not tn_jobs:
        print("All TN jobs done!")
        return

    print(f"\n{'='*60}")
    print(f"Phase 3: TN-MLD only, {len(tn_jobs)} jobs, {TN_WORKERS} workers")
    print(f"OMP_NUM_THREADS={THREADS_PER_WORKER} per worker")
    print(f"{'='*60}")

    from collections import Counter
    by_code = Counter(j[0] for j in tn_jobs)
    for code, cnt in sorted(by_code.items()):
        ms = _get_max_shots(code, 'tensor_network_mld')
        print(f"  {code:15s}: {cnt} jobs (max {ms:,} shots)")

    print(f"\nJob order (first 10):")
    for j in tn_jobs[:10]:
        print(f"  {j[0]} p={j[1]} e={j[2]}")

    print(f"\nStarting...\n")

    all_results = list(existing)
    completed = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=TN_WORKERS) as executor:
        futures = {executor.submit(_run_adaptive_job, job): job for job in tn_jobs}
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
            eta = (elapsed / completed) * (len(tn_jobs) - completed) if completed else 0
            print(f"[{completed}/{len(tn_jobs)}] ETA {eta/60:.0f}min "
                  f"(total: {len(all_results)})", flush=True)

            # Save every 2 completions
            if completed % 2 == 0:
                with open(PARTIAL_FILE, 'w') as f:
                    json.dump(all_results, f, indent=2, default=str)
                print(f"  [saved {len(all_results)} results]", flush=True)

    # Final save
    with open(PARTIAL_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"tn_adaptive_{timestamp}.json")
    output = {
        'timestamp': datetime.now().isoformat(),
        'min_errors': MIN_ERRORS,
        'results': all_results,
    }
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    total_time = time.time() - start_time
    print(f"\nPhase 3 done in {total_time/3600:.1f} hours")
    print(f"Total results: {len(all_results)}")
    print(f"Saved to: {results_file}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
