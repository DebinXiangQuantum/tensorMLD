#!/usr/bin/env python3
"""
FP32 vs FP64 accuracy validation for the GPU MPS decoder.

For each available code, decodes random syndromes with both FP64 and FP32
scan modes and compares:
  - Decision match rate (should be >99.9%)
  - Marginal max absolute difference
  - Marginal mean absolute difference

Usage:
    python3 gpu_decoder/test_fp32_accuracy.py [--num-syndromes 1000] [--chi 256]
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mps_gpu_decoder import MpsGpuDecoder


def find_chain_dirs(results_dir: str) -> list:
    """Find all chain directories with chain_meta.json."""
    chains = []
    results_path = Path(results_dir)
    if not results_path.exists():
        return chains
    for code_dir in sorted(results_path.iterdir()):
        if not code_dir.is_dir():
            continue
        # Check <code>/chain/ subdir (standard export format)
        chain_subdir = code_dir / "chain"
        if chain_subdir.exists() and (chain_subdir / "chain_meta.json").exists():
            chains.append((code_dir.name, str(chain_subdir)))
        # Check <code>/logical_0/ subdir (multi-logical format)
        elif (code_dir / "logical_0").exists() and (code_dir / "logical_0" / "chain_meta.json").exists():
            chains.append((code_dir.name, str(code_dir / "logical_0")))
        # Check <code>/chain_meta.json directly
        elif (code_dir / "chain_meta.json").exists():
            chains.append((code_dir.name, str(code_dir)))
    return chains


def random_syndrome(M: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random binary syndrome vector."""
    return rng.integers(0, 2, size=M, dtype=np.int32)


def run_accuracy_test(chain_dir: str, code_name: str, num_syndromes: int,
                      low_thresh: float = 0.1, high_thresh: float = 0.9,
                      max_rounds: int = 10, seed: int = 42):
    """Run FP64 vs FP32 comparison for a single code."""
    print(f"\n{'='*60}")
    print(f"Code: {code_name}")
    print(f"  Chain: {chain_dir}")

    # Create two decoders: FP64 and FP32
    dec_fp64 = MpsGpuDecoder(chain_dir, fp32=False)
    dec_fp32 = MpsGpuDecoder(chain_dir, fp32=True)

    print(f"  K={dec_fp64.K}, M={dec_fp64.M}, N={dec_fp64.N}, max_chi={dec_fp64.max_chi}")
    print(f"  FP32 mode: {dec_fp32.fp32}, FP64 mode: {dec_fp64.fp32}")

    rng = np.random.default_rng(seed)
    M = dec_fp64.M

    decision_matches = 0
    total_decisions = 0
    marginal_diffs = []
    fp32_faster = 0
    fp64_latencies = []
    fp32_latencies = []

    for i in range(num_syndromes):
        syn = random_syndrome(M, rng)

        # Decode with FP64
        dec64, marg64, lat64 = dec_fp64.decode(syn, low_thresh, high_thresh, max_rounds)
        # Decode with FP32
        dec32, marg32, lat32 = dec_fp32.decode(syn, low_thresh, high_thresh, max_rounds)

        # Compare decisions
        matches = np.sum(dec64 == dec32)
        decision_matches += matches
        total_decisions += len(dec64)

        # Compare marginals
        diff = np.abs(marg64 - marg32)
        marginal_diffs.append(diff)

        fp64_latencies.append(lat64)
        fp32_latencies.append(lat32)
        if lat32 < lat64:
            fp32_faster += 1

    marginal_diffs = np.array(marginal_diffs)
    match_rate = decision_matches / total_decisions if total_decisions > 0 else 1.0

    print(f"\n  Results ({num_syndromes} syndromes, {total_decisions} total decisions):")
    print(f"  Decision match rate:     {match_rate*100:.3f}%  ({decision_matches}/{total_decisions})")
    print(f"  Marginal max abs diff:   {marginal_diffs.max():.6e}")
    print(f"  Marginal mean abs diff:  {marginal_diffs.mean():.6e}")
    print(f"  Marginal median abs diff:{np.median(marginal_diffs):.6e}")
    print(f"  FP64 latency (median):   {np.median(fp64_latencies):.1f} us")
    print(f"  FP32 latency (median):   {np.median(fp32_latencies):.1f} us")
    speedup = np.median(fp64_latencies) / np.median(fp32_latencies) if np.median(fp32_latencies) > 0 else 0
    print(f"  Speedup:                 {speedup:.2f}x")
    print(f"  FP32 faster:             {fp32_faster}/{num_syndromes}")

    dec_fp64.close()
    dec_fp32.close()

    return {
        "code": code_name,
        "num_syndromes": num_syndromes,
        "K": dec_fp64.K,
        "M": dec_fp64.M,
        "max_chi": dec_fp64.max_chi,
        "decision_match_rate": match_rate,
        "marginal_max_diff": float(marginal_diffs.max()),
        "marginal_mean_diff": float(marginal_diffs.mean()),
        "fp64_median_us": float(np.median(fp64_latencies)),
        "fp32_median_us": float(np.median(fp32_latencies)),
        "speedup": float(speedup),
    }


def main():
    parser = argparse.ArgumentParser(description="FP32 vs FP64 accuracy validation")
    parser.add_argument("--num-syndromes", type=int, default=200,
                        help="Number of random syndromes per code (default: 200)")
    parser.add_argument("--chi", type=int, default=256,
                        help="Chi value to test (determines results directory)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results directory path")
    parser.add_argument("--code", type=str, default=None,
                        help="Test only this specific code name")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.results_dir:
        results_dir = args.results_dir
    else:
        base = Path(__file__).resolve().parent.parent / "experiments" / "results"
        results_dir = str(base / f"isca_revision_1d_mps_chi{args.chi}")

    print(f"Looking for chain data in: {results_dir}")
    chains = find_chain_dirs(results_dir)

    if not chains:
        print(f"No chain directories found in {results_dir}")
        print("Make sure MPS data has been exported first.")
        sys.exit(1)

    if args.code:
        chains = [(name, path) for name, path in chains if args.code in name]
        if not chains:
            print(f"No chain matching '{args.code}' found")
            sys.exit(1)

    print(f"Found {len(chains)} codes: {[name for name, _ in chains]}")

    all_results = []
    for code_name, chain_dir in chains:
        result = run_accuracy_test(
            chain_dir, code_name, args.num_syndromes, seed=args.seed)
        all_results.append(result)

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Code':<20} {'Match%':>8} {'MaxDiff':>12} {'MeanDiff':>12} "
          f"{'FP64(us)':>10} {'FP32(us)':>10} {'Speedup':>8}")
    print("-" * 80)

    all_match = True
    for r in all_results:
        print(f"{r['code']:<20} {r['decision_match_rate']*100:>7.3f}% "
              f"{r['marginal_max_diff']:>12.2e} {r['marginal_mean_diff']:>12.2e} "
              f"{r['fp64_median_us']:>10.1f} {r['fp32_median_us']:>10.1f} "
              f"{r['speedup']:>7.2f}x")
        if r['decision_match_rate'] < 0.999:
            all_match = False

    print("-" * 80)
    if all_match:
        print("PASS: All codes have >99.9% decision match rate")
    else:
        print("WARNING: Some codes have <99.9% decision match rate")

    # Save results
    output_path = Path(__file__).resolve().parent / "fp32_accuracy_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
