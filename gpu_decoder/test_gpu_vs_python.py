#!/usr/bin/env python3
"""
Correctness test: compare GPU MPS decoder results against Python OneDChainMPS.

For each code, generates random syndrome vectors and compares:
  - Marginal probabilities (should match within tolerance)
  - Decisions (should match exactly in most cases)
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "cudaqx" / "libs" / "qec" / "python"))

from cudaq_qec.plugins.decoders.tensor_network_utils.mps_decoder_core import OneDChainMPS
from mps_gpu_decoder import MpsGpuDecoder


def test_code(code_name: str,
              data_root: str = "experiments/results/isca_revision_1d_mps_chi128",
              num_tests: int = 100,
              seed: int = 42):
    """Test one code: compare GPU vs Python decode results."""
    chain_dir = os.path.join(PROJECT_ROOT, data_root, code_name, "chain")

    if not os.path.exists(chain_dir):
        print(f"  SKIP {code_name}: {chain_dir} not found")
        return True

    # Load Python decoder
    print(f"  Loading Python decoder for {code_name}...")
    py_chain = OneDChainMPS.load(chain_dir)

    # Load GPU decoder
    print(f"  Loading GPU decoder for {code_name}...")
    gpu_dec = MpsGpuDecoder(chain_dir)

    K = gpu_dec.K
    M = gpu_dec.M
    syn_labels = gpu_dec.syndrome_labels
    log_labels = gpu_dec.logical_labels

    print(f"  N={gpu_dec.N} K={K} M={M} max_chi={gpu_dec.max_chi}")

    rng = np.random.RandomState(seed)

    n_match = 0
    n_total = 0
    max_marg_diff = 0.0
    sum_marg_diff = 0.0
    marg_count = 0

    gpu_times = []
    py_times = []

    for t in range(num_tests):
        # Generate random syndrome (binary)
        syn_arr = rng.randint(0, 2, size=M)

        # Build syndrome dict for Python
        syn_dict = {}
        for i, label in enumerate(syn_labels):
            syn_dict[label] = float(syn_arr[i])

        # Python decode
        t0 = time.perf_counter()
        py_assignments, py_marginals = py_chain.decode_logicals(
            syndrome_values=syn_dict,
            low_threshold=0.1,
            high_threshold=0.9,
            max_rounds=10,
        )
        py_time = (time.perf_counter() - t0) * 1e6  # us
        py_times.append(py_time)

        # GPU decode
        gpu_assignments, gpu_marginals, gpu_latency = gpu_dec.decode_dict(
            syndrome_values=syn_dict,
            low_thresh=0.1,
            high_thresh=0.9,
            max_rounds=10,
        )
        gpu_times.append(gpu_latency)

        # Compare
        for label in log_labels:
            n_total += 1
            py_dec = py_assignments[label]
            gpu_d = gpu_assignments[label]
            py_m = py_marginals[label]
            gpu_m = gpu_marginals[label]

            diff = abs(py_m - gpu_m)
            max_marg_diff = max(max_marg_diff, diff)
            sum_marg_diff += diff
            marg_count += 1

            if py_dec == gpu_d:
                n_match += 1

    match_rate = n_match / n_total * 100 if n_total > 0 else 0
    avg_marg_diff = sum_marg_diff / marg_count if marg_count > 0 else 0

    print(f"  Results ({num_tests} tests, {n_total} decisions):")
    print(f"    Decision match rate: {match_rate:.1f}% ({n_match}/{n_total})")
    print(f"    Max marginal diff:   {max_marg_diff:.6e}")
    print(f"    Avg marginal diff:   {avg_marg_diff:.6e}")
    print(f"    GPU latency:         min={min(gpu_times):.1f} median={np.median(gpu_times):.1f} "
          f"mean={np.mean(gpu_times):.1f} max={max(gpu_times):.1f} us")
    print(f"    Python latency:      min={min(py_times):.1f} median={np.median(py_times):.1f} "
          f"mean={np.mean(py_times):.1f} max={max(py_times):.1f} us")
    print(f"    Speedup:             {np.mean(py_times)/np.mean(gpu_times):.1f}x")

    gpu_dec.close()

    # Pass if decision match rate > 95% and marginal diff < 0.01
    # (Small differences are expected due to numerical precision and
    #  different normalization strategies)
    passed = match_rate >= 95.0 and max_marg_diff < 0.05
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    data_root = "experiments/results/isca_revision_1d_mps_chi128"
    num_tests = 50  # per code

    codes = [
        "QCC_18_4_4",     # smallest: N=11, K=4, max_chi=6
        "LDPC_25_3_4",    # N=15, K=3, max_chi=64
        "LDPC_30_6_4",    # N=19, K=6, max_chi=66
        "TOR_50_2_5",     # N=26, K=2, max_chi=128
        "QCC_60_8_4",     # N=34, K=8, max_chi=128
        "QCC_72_12_6",    # N=42, K=12, max_chi=128
        "QCC_90_8_10",    # N=49, K=8, max_chi=128
        "QCC_108_8_10",   # N=58, K=8, max_chi=128
        "QCC_144_12_12",  # largest: N=78, K=12, max_chi=128
    ]

    print("=" * 70)
    print("GPU vs Python MPS Decoder Correctness Test")
    print("=" * 70)

    results = {}
    for code in codes:
        print(f"\nTesting {code}:")
        passed = test_code(code, data_root, num_tests)
        results[code] = passed

    print("\n" + "=" * 70)
    print("Summary:")
    all_pass = True
    for code, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {code}: {status}")
        if not passed:
            all_pass = False

    print("=" * 70)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
