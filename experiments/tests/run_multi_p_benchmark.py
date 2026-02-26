#!/usr/bin/env python3
"""Run v2 benchmark at multiple noise rates for selected codes, in parallel.

Usage:
  .venv-mps/bin/python experiments/tests/run_multi_p_benchmark.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

# CRITICAL: Use 'spawn' to avoid CUDA fork issues
mp.set_start_method("spawn", force=True)

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent.parent


def _run_one(args: dict) -> list[dict]:
    """Worker function for a single (code, noise_p) combination.

    Imports are done inside the worker to ensure clean CUDA context
    in each spawned process.
    """
    import sys
    from pathlib import Path

    WORKSPACE = Path(args["workspace"])
    sys.path.insert(0, str(WORKSPACE))

    from experiments.tests.test_multi_decoder_benchmark_v2 import run_benchmark

    code_name = args["code_name"]
    noise_p = args["noise_p"]
    shots = args["shots"]
    chi = args["chi"]
    seed = args["seed"]
    chain_dir = Path(args["chain_dir"])

    print(f"[worker] Starting {code_name} p={noise_p} shots={shots}", flush=True)
    t0 = time.perf_counter()
    results = run_benchmark(
        shots=shots,
        noise_p=noise_p,
        chi=chi,
        seed=seed,
        code_names=[code_name],
        skip_missing=False,
        chain_dir=chain_dir,
        force_recompile=True,  # recompile for each noise_p
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"[worker] Done {code_name} p={noise_p} in {elapsed:.1f}s",
        flush=True,
    )
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-p benchmark for BB_18_4_4 and TB_25_3_4"
    )
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--chi", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--noise-ps",
        nargs="*",
        type=float,
        default=[0.001, 0.005, 0.01, 0.02, 0.05],
    )
    parser.add_argument(
        "--codes",
        nargs="*",
        default=["BB_18_4_4", "TB_25_3_4"],
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    chain_dir = WORKSPACE / "experiments" / "results" / "mps_chains"

    # Build task list: one task per (code, noise_p)
    tasks = []
    for code_name in args.codes:
        for noise_p in args.noise_ps:
            tasks.append(
                {
                    "code_name": code_name,
                    "noise_p": noise_p,
                    "shots": args.shots,
                    "chi": args.chi,
                    "seed": args.seed,
                    "chain_dir": str(chain_dir),
                    "workspace": str(WORKSPACE),
                }
            )

    print(f"Total tasks: {len(tasks)}, workers: {args.workers}")
    print(f"Codes: {args.codes}")
    print(f"Noise rates: {args.noise_ps}")
    print(f"Shots: {args.shots}, chi: {args.chi}")
    print()

    t_start = time.perf_counter()

    # Run in parallel
    with mp.Pool(processes=args.workers) as pool:
        all_results_nested = pool.map(_run_one, tasks)

    # Flatten results
    all_results = []
    for result_list in all_results_nested:
        all_results.extend(result_list)

    total_time = time.perf_counter() - t_start
    print(f"\nTotal time: {total_time:.1f}s")

    # Save results
    out_dir = WORKSPACE / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multi_p_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")

    # Summary table
    print(f"\n{'='*100}")
    print(
        f"{'Code':<15} {'p':>8} {'N':>4} {'K':>3} "
        f"{'MPS LER':>10} {'Exact LER':>10} {'BP-OSD LER':>10} "
        f"{'BP LER':>10} {'BP-LSD LER':>10}"
    )
    print(f"{'-'*100}")

    # Sort by code name then noise_p
    all_results.sort(key=lambda r: (r["code"], r["noise_p"]))

    for r in all_results:
        mps = r["decoders"].get("mps_tn", {})
        exact = r["decoders"].get("exact_tn", {})
        bposd = r["decoders"].get("bp_osd", {})
        bp = r["decoders"].get("bp", {})
        bplsd = r["decoders"].get("bp_lsd", {})

        def fmt_ler(d):
            if d.get("status") == "ok":
                return f"{d['overall_logical_error_rate']:.4f}"
            return d.get("status", "N/A")[:8]

        print(
            f"{r['code']:<15} {r['noise_p']:>8.3f} {r['N']:>4} {r['K']:>3} "
            f"{fmt_ler(mps):>10} {fmt_ler(exact):>10} {fmt_ler(bposd):>10} "
            f"{fmt_ler(bp):>10} {fmt_ler(bplsd):>10}"
        )

    # Also print per-logical details
    print(f"\n{'='*100}")
    print("Per-logical LER details:")
    print(f"{'Code':<15} {'p':>8} {'Decoder':<10} {'Per-logical LER'}")
    print(f"{'-'*100}")
    for r in all_results:
        for dec_name in ["mps_tn", "exact_tn", "bp_osd"]:
            d = r["decoders"].get(dec_name, {})
            if d.get("status") == "ok":
                print(
                    f"{r['code']:<15} {r['noise_p']:>8.3f} {dec_name:<10} "
                    f"{d['per_logical_ler']}"
                )


if __name__ == "__main__":
    main()
