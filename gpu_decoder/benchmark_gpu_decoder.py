#!/usr/bin/env python3
"""
Benchmark GPU MPS decoder latency and throughput for all 9 ISCA codes.

Reports:
  - Single-decode latency (min/median/mean/max) in microseconds
  - Batch decode throughput for batch sizes 1, 10, 100, 1000
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mps_gpu_decoder import MpsGpuDecoder


def benchmark_code(code_name: str,
                   data_root: str,
                   num_warmup: int = 100,
                   num_bench: int = 1000,
                   batch_sizes: list = None,
                   seed: int = 42):
    """Benchmark one code's decode latency and throughput."""
    chain_dir = os.path.join(data_root, code_name, "chain")

    if not os.path.exists(chain_dir):
        print(f"  SKIP {code_name}: not found")
        return None

    gpu_dec = MpsGpuDecoder(chain_dir)
    K = gpu_dec.K
    M = gpu_dec.M
    N = gpu_dec.N
    max_chi = gpu_dec.max_chi

    if batch_sizes is None:
        batch_sizes = [1, 10, 100, 1000]

    rng = np.random.RandomState(seed)

    # Generate all test syndromes
    max_needed = max(num_warmup + num_bench, max(batch_sizes))
    all_syndromes = rng.randint(0, 2, size=(max_needed, M)).astype(np.int32)

    # --- Single decode benchmark ---
    print(f"  Single decode: warming up ({num_warmup} decodes)...")
    for i in range(num_warmup):
        gpu_dec.decode(all_syndromes[i])

    print(f"  Single decode: benchmarking ({num_bench} decodes)...")
    latencies = []
    for i in range(num_bench):
        _, _, lat = gpu_dec.decode(all_syndromes[num_warmup + i])
        latencies.append(lat)

    latencies = np.array(latencies)
    single_result = {
        "min_us": float(np.min(latencies)),
        "p25_us": float(np.percentile(latencies, 25)),
        "median_us": float(np.median(latencies)),
        "mean_us": float(np.mean(latencies)),
        "p75_us": float(np.percentile(latencies, 75)),
        "p99_us": float(np.percentile(latencies, 99)),
        "max_us": float(np.max(latencies)),
        "std_us": float(np.std(latencies)),
    }

    print(f"    min={single_result['min_us']:.1f}  median={single_result['median_us']:.1f}  "
          f"mean={single_result['mean_us']:.1f}  p99={single_result['p99_us']:.1f}  "
          f"max={single_result['max_us']:.1f} us")

    # --- Batch decode benchmark ---
    batch_results = {}
    for bs in batch_sizes:
        # Warmup
        batch_syn = all_syndromes[:bs]
        for _ in range(10):
            gpu_dec.decode_batch(batch_syn)

        # Benchmark
        batch_lats = []
        for _ in range(100):
            _, _, lat = gpu_dec.decode_batch(batch_syn)
            batch_lats.append(lat)

        batch_lats = np.array(batch_lats)
        throughput = bs / (np.mean(batch_lats) / 1e6)  # decodes/sec

        batch_results[bs] = {
            "batch_size": bs,
            "total_us_mean": float(np.mean(batch_lats)),
            "total_us_median": float(np.median(batch_lats)),
            "per_decode_us": float(np.mean(batch_lats) / bs),
            "throughput_per_sec": float(throughput),
        }

        print(f"    Batch {bs:5d}: total={np.mean(batch_lats):.1f} us, "
              f"per_decode={np.mean(batch_lats)/bs:.1f} us, "
              f"throughput={throughput:.0f} dec/s")

    gpu_dec.close()

    return {
        "code": code_name,
        "N": N,
        "K": K,
        "M": M,
        "max_chi": max_chi,
        "single_decode": single_result,
        "batch_decode": batch_results,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_root = str(project_root / "experiments" / "results" / "isca_revision_1d_mps_chi128")

    codes = [
        "QCC_18_4_4",
        "LDPC_25_3_4",
        "LDPC_30_6_4",
        "TOR_50_2_5",
        "QCC_60_8_4",
        "QCC_72_12_6",
        "QCC_90_8_10",
        "QCC_108_8_10",
        "QCC_144_12_12",
    ]

    print("=" * 70)
    print("GPU MPS Decoder Benchmark")
    print("=" * 70)

    all_results = []
    for code in codes:
        print(f"\n--- {code} ---")
        result = benchmark_code(code, data_root)
        if result:
            all_results.append(result)

    # Save results
    out_path = os.path.join(project_root, "gpu_decoder", "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Code':<18} {'N':>3} {'K':>3} {'chi':>4} | "
          f"{'Min':>7} {'Med':>7} {'Mean':>7} {'P99':>7} | "
          f"{'Batch1k':>10}")
    print("-" * 90)
    for r in all_results:
        s = r["single_decode"]
        b1k = r["batch_decode"].get(1000, {})
        tp = b1k.get("throughput_per_sec", 0)
        print(f"{r['code']:<18} {r['N']:>3} {r['K']:>3} {r['max_chi']:>4} | "
              f"{s['min_us']:>7.1f} {s['median_us']:>7.1f} {s['mean_us']:>7.1f} "
              f"{s['p99_us']:>7.1f} | "
              f"{tp:>10.0f}/s")
    print("=" * 90)


if __name__ == "__main__":
    main()
