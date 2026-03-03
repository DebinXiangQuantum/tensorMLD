#!/usr/bin/env python3
"""Ising machine / modified-QUBO decoder benchmark with BP/BP-OSD comparison.

Tests the Ising decoder on QLDPC codes from the ISCA revision benchmark suite
and compares against BP, BP-OSD, and BP-LSD decoders from the ldpc package.

Reports:
  - Logical error rate (block LER) across all K logicals
  - Decode latency (wall-clock per shot)
  - Ising machine hardware metrics (cycle counts, coupler counts, etc.)

Usage:
  python experiments/tests/test_ising_decoder.py
  python experiments/tests/test_ising_decoder.py --codes QCC_18_4_4 LDPC_25_3_4
  python experiments/tests/test_ising_decoder.py --noise_values 0.005,0.01,0.02,0.05
  python -m pytest experiments/tests/test_ising_decoder.py -v -s
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from experiments.tests._isca_code_registry import load_isca_code_cases
from experiments.isingdecoder.qubo_decoder import IsingDecoder
from experiments.isingdecoder.ising_machine_sim import (
    IsingMachineSim,
    ising_machine_metrics,
)

# ---------------------------------------------------------------------------
# BP decoders (ldpc package)
# ---------------------------------------------------------------------------
HAS_LDPC = False
BpDecoder = BpOsdDecoder = BpLsdDecoder = None
try:
    from ldpc import BpOsdDecoder, BpDecoder, BpLsdDecoder
    HAS_LDPC = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def _sample_errors_and_syndromes(
    H: np.ndarray, noise_p: float, shots: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample independent X errors and compute syndromes."""
    rng = np.random.default_rng(seed)
    n = H.shape[1]
    errors = (rng.random((shots, n)) < noise_p).astype(np.int8)
    syndromes = (errors @ H.T.astype(np.int8)) % 2
    return errors, syndromes


def _block_logical_error_rate(
    errors: np.ndarray, lz: np.ndarray, predictions: np.ndarray
) -> float:
    """Compute block logical error rate (shot fails if ANY logical is wrong)."""
    true_logicals = (errors @ lz.T.astype(np.int8)) % 2  # (shots, K)
    any_wrong = np.any(true_logicals != predictions, axis=1)  # (shots,)
    return float(np.mean(any_wrong))


# ---------------------------------------------------------------------------
# Code loading
# ---------------------------------------------------------------------------
@dataclass
class CodeCase:
    name: str
    code: Any


def _get_codes(
    code_names: Optional[list[str]] = None,
    skip_missing: bool = True,
) -> list[CodeCase]:
    loaded, skipped = load_isca_code_cases(
        code_names=code_names, skip_missing=skip_missing
    )
    if skipped:
        print(f"[codes] skipped {len(skipped)} code(s)")
    return [CodeCase(name=n, code=c) for n, c in loaded]


# ---------------------------------------------------------------------------
# Per-decoder runners
# ---------------------------------------------------------------------------
def _run_qubo_sa(
    H: np.ndarray, lz: np.ndarray,
    errors: np.ndarray, syndromes: np.ndarray,
    noise_p: float, seed: int,
    alpha, num_reads: int, num_steps_per_qubit: int, T_init: float,
    convergence_patience: int, bp_max_iter: int, cycle_time_ns: float,
    verbose: bool,
) -> dict[str, Any]:
    """Run modified-QUBO SA decoder with variable latency tracking."""
    K = int(lz.shape[0])
    num_errors = H.shape[1]
    shots = syndromes.shape[0]

    t0 = time.perf_counter()
    decoder = IsingDecoder(
        H=H, logical_obs=lz,
        noise_model=[noise_p] * num_errors,
        alpha=alpha, num_reads=num_reads,
        num_steps_per_qubit=num_steps_per_qubit,
        T_init=T_init, seed=seed,
        convergence_patience=convergence_patience,
        bp_max_iter=bp_max_iter,
    )
    actual_alpha = decoder.alpha
    init_ms = (time.perf_counter() - t0) * 1e3

    all_preds = np.zeros((shots, K), dtype=np.int8)
    total_violations = 0
    total_sa_steps = 0
    total_bp_init_iters = 0
    total_hw_cycles = 0
    t1 = time.perf_counter()
    for i in range(shots):
        res = decoder.decode(syndromes[i])
        for k in range(K):
            all_preds[i, k] = int(res.result[k] >= 0.5)
        total_violations += decoder.last_timing.parity_violations
        total_sa_steps += decoder.last_timing.actual_sa_steps
        total_bp_init_iters += decoder.last_timing.bp_init_iters
        # Hardware cycles per shot: BP init cycles + SA annealing cycles
        bp_cycles = decoder.last_timing.bp_init_iters * 2
        # SA steps map to cycles: each step ≈ 1 cycle in hardware
        sa_cycles = decoder.last_timing.actual_sa_steps
        total_hw_cycles += bp_cycles + sa_cycles
    decode_ms = (time.perf_counter() - t1) * 1e3

    ler = _block_logical_error_rate(errors, lz, all_preds)
    avg_hw_cycles = total_hw_cycles / shots
    avg_hw_latency_ns = avg_hw_cycles * cycle_time_ns
    result = {
        "status": "ok",
        "alpha_actual": round(actual_alpha, 2),
        "init_ms": round(init_ms, 2),
        "total_decode_ms": round(decode_ms, 2),
        "per_shot_ms": round(decode_ms / shots, 3),
        "logical_error_rate": round(ler, 6),
        "avg_parity_violations": round(total_violations / shots, 2),
        "avg_sa_steps": round(total_sa_steps / shots, 1),
        "avg_bp_init_iters": round(total_bp_init_iters / shots, 1),
        "avg_hw_cycles": round(avg_hw_cycles, 1),
        "avg_hw_latency_ns": round(avg_hw_latency_ns, 1),
    }
    if verbose:
        print(f"    QUBO-SA: LER={ler:.4f}, {decode_ms/shots:.2f} ms/shot, "
              f"violations={total_violations/shots:.1f}, alpha={actual_alpha:.1f}, "
              f"hw_latency={avg_hw_latency_ns:.0f}ns "
              f"(bp_init={total_bp_init_iters/shots:.0f}it + "
              f"sa={total_sa_steps/shots:.0f}steps)")
    return result


def _run_bp_decoder(
    dec_name: str, DecClass, dec_kwargs: dict,
    H: np.ndarray, lz: np.ndarray,
    errors: np.ndarray, syndromes: np.ndarray,
    noise_p: float, cycle_time_ns: float,
    verbose: bool,
) -> dict[str, Any]:
    """Run a BP-family decoder (BP, BP-OSD, BP-LSD) with iteration tracking."""
    K = int(lz.shape[0])
    num_errors = H.shape[1]
    shots = syndromes.shape[0]
    H_int = H.astype(np.uint8)
    channel_probs = np.full(num_errors, noise_p)
    max_iter = dec_kwargs.get("max_iter", 50)

    t0 = time.perf_counter()
    bp_dec = DecClass(
        H_int, error_rate=noise_p,
        channel_probs=channel_probs,
        **dec_kwargs,
    )
    init_ms = (time.perf_counter() - t0) * 1e3

    predictions_list = []
    total_bp_iters = 0
    t1 = time.perf_counter()
    for i in range(shots):
        correction = bp_dec.decode(syndromes[i].astype(np.uint8))
        logical_vals = (correction @ lz.T.astype(np.int8)) % 2
        predictions_list.append(logical_vals.tolist())
        # Track actual BP iterations (ldpc exposes .iter)
        bp_iters = getattr(bp_dec, "iter", max_iter)
        if bp_iters is None or bp_iters <= 0:
            bp_iters = max_iter
        total_bp_iters += int(bp_iters)
    decode_ms = (time.perf_counter() - t1) * 1e3

    preds = np.array(predictions_list, dtype=np.int8)
    ler = _block_logical_error_rate(errors, lz, preds)
    avg_iters = total_bp_iters / shots
    # BP hardware cycles: each iteration = 2 cycles (forward + backward pass)
    avg_hw_cycles = avg_iters * 2
    avg_hw_latency_ns = avg_hw_cycles * cycle_time_ns
    result = {
        "status": "ok",
        "init_ms": round(init_ms, 2),
        "total_decode_ms": round(decode_ms, 2),
        "per_shot_ms": round(decode_ms / shots, 3),
        "logical_error_rate": round(ler, 6),
        "avg_bp_iters": round(avg_iters, 1),
        "avg_hw_cycles": round(avg_hw_cycles, 1),
        "avg_hw_latency_ns": round(avg_hw_latency_ns, 1),
    }
    if verbose:
        print(f"    {dec_name}: LER={ler:.4f}, {decode_ms/shots:.2f} ms/shot, "
              f"hw_latency={avg_hw_latency_ns:.0f}ns ({avg_iters:.0f}it)")
    return result


def _run_ising_sim(
    H: np.ndarray, lz: np.ndarray,
    errors: np.ndarray, syndromes: np.ndarray,
    noise_p: float, seed: int,
    alpha: float, ising_sim_cycles: int,
    verbose: bool,
) -> dict[str, Any]:
    """Run Ising machine dynamics simulation."""
    K = int(lz.shape[0])
    num_errors = H.shape[1]
    sim_shots = min(syndromes.shape[0], 8)

    sim = IsingMachineSim(H=H, alpha=alpha)
    noise_p_arr = np.full(num_errors, noise_p)
    llr = np.log((1.0 - noise_p_arr) / noise_p_arr)

    sim_preds = np.zeros((sim_shots, K), dtype=np.int8)
    sim_violations = 0
    t0 = time.perf_counter()
    for i in range(sim_shots):
        sim_result = sim.simulate(
            syndrome=syndromes[i], llr=llr,
            num_cycles=ising_sim_cycles,
            seed=seed + i,
            trace_interval=max(1, ising_sim_cycles // 10),
        )
        logical_vals = (sim_result.errors @ lz.T.astype(np.int8)) % 2
        sim_preds[i] = logical_vals.astype(np.int8)
        sim_violations += sim_result.parity_violations
    sim_ms = (time.perf_counter() - t0) * 1e3
    sim_ler = _block_logical_error_rate(errors[:sim_shots], lz, sim_preds)

    result = {
        "status": "ok",
        "num_cycles": ising_sim_cycles,
        "sim_shots": sim_shots,
        "total_sim_ms": round(sim_ms, 2),
        "per_shot_ms": round(sim_ms / sim_shots, 3),
        "logical_error_rate": round(sim_ler, 6),
        "avg_parity_violations": round(sim_violations / sim_shots, 2),
    }
    if verbose:
        print(f"    Ising-Sim ({ising_sim_cycles}cyc): LER={sim_ler:.4f}, "
              f"{sim_shots} shots, violations={sim_violations/sim_shots:.1f}")
    return result


# ---------------------------------------------------------------------------
# Single code + single noise_p
# ---------------------------------------------------------------------------
def _run_single_code_single_p(
    cc: CodeCase,
    *,
    shots: int,
    sa_shots: Optional[int] = None,
    noise_p: float,
    seed: int,
    alpha: float,
    num_reads: int,
    num_steps_per_qubit: int,
    T_init: float,
    annealing_time_ns: float,
    cycle_time_ns: float,
    bp_max_iter: int,
    convergence_patience: int,
    ising_sim_cycles: int,
    verbose: bool,
) -> dict[str, Any]:
    H = np.ascontiguousarray(cc.code.hz.astype(np.float64))
    lz = np.ascontiguousarray(cc.code.lz.astype(np.float64))
    K = int(lz.shape[0])
    if sa_shots is None:
        sa_shots = shots

    if verbose:
        print(f"  p={noise_p}:")

    # Sample errors for both SA and BP (use max of the two shot counts)
    max_shots = max(shots, sa_shots)
    errors_all, syndromes_all = _sample_errors_and_syndromes(
        H, noise_p, max_shots, seed)

    record: dict[str, Any] = {
        "code": cc.name,
        "N": int(cc.code.N),
        "K": K,
        "D": float(cc.code.D),
        "noise_p": noise_p,
        "shots_bp": shots,
        "shots_sa": sa_shots,
        "decoders": {},
    }

    # --- QUBO-SA (uses sa_shots) ---
    try:
        record["decoders"]["qubo_sa"] = _run_qubo_sa(
            H, lz, errors_all[:sa_shots], syndromes_all[:sa_shots],
            noise_p, seed,
            alpha, num_reads, num_steps_per_qubit, T_init,
            convergence_patience, bp_max_iter, cycle_time_ns, verbose)
    except Exception as e:
        record["decoders"]["qubo_sa"] = {"status": "error", "error": str(e)}
        if verbose:
            print(f"    QUBO-SA: ERROR {e}")

    # --- BP decoders (uses full shots) ---
    if HAS_LDPC:
        bp_configs = [
            ("bp", BpDecoder, {"max_iter": 50, "bp_method": "ms"}),
            ("bp_osd", BpOsdDecoder, {
                "max_iter": 50, "bp_method": "ms",
                "osd_method": "osd_cs", "osd_order": 10}),
            ("bp_lsd", BpLsdDecoder, {
                "max_iter": 50, "bp_method": "ms",
                "lsd_order": 0}),
        ]
        for dec_name, DecClass, kwargs in bp_configs:
            try:
                record["decoders"][dec_name] = _run_bp_decoder(
                    dec_name, DecClass, kwargs,
                    H, lz, errors_all[:shots], syndromes_all[:shots],
                    noise_p, cycle_time_ns, verbose)
            except Exception as e:
                record["decoders"][dec_name] = {"status": "error", "error": str(e)}
                if verbose:
                    print(f"    {dec_name}: ERROR {e}")
    else:
        if verbose:
            print("    [BP decoders skipped: ldpc package not installed]")

    # --- Ising machine simulation ---
    if ising_sim_cycles > 0:
        try:
            record["decoders"]["ising_sim"] = _run_ising_sim(
                H, lz, errors_all, syndromes_all, noise_p, seed,
                alpha, ising_sim_cycles, verbose)
        except Exception as e:
            record["decoders"]["ising_sim"] = {"status": "error", "error": str(e)}
            if verbose:
                print(f"    Ising-Sim: ERROR {e}")

    return record


# ---------------------------------------------------------------------------
# Main benchmark: multi-code x multi-p
# ---------------------------------------------------------------------------
def run_benchmark(
    *,
    code_names: Optional[list[str]] = None,
    shots: int = 256,
    sa_shots: Optional[int] = None,
    noise_values: Optional[list[float]] = None,
    seed: int = 2026,
    alpha: float = 2.0,
    num_reads: int = 16,
    num_steps_per_qubit: int = 50,
    T_init: float = 2.0,
    annealing_time_ns: float = 320.0,
    cycle_time_ns: float = 4.0,
    bp_max_iter: int = 50,
    convergence_patience: int = 3,
    ising_sim_cycles: int = 500,
    verbose: bool = True,
    output_json: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Run multi-decoder benchmark across codes and noise levels.

    Args:
        shots: Number of shots for BP decoders (can be large, e.g. 1e6).
        sa_shots: Number of shots for QUBO-SA decoder (defaults to shots).
            SA is much slower, so use fewer shots if shots is very large.
    """
    if noise_values is None:
        noise_values = [0.001, 0.003, 0.005, 0.008, 0.01]
    if sa_shots is None:
        sa_shots = shots

    codes = _get_codes(code_names=code_names, skip_missing=True)
    if not codes:
        print("[ERROR] No codes loaded.")
        return []

    print(f"\nIsing Decoder Benchmark (with BP/BP-OSD comparison)")
    print(f"  Codes: {[c.name for c in codes]}")
    print(f"  noise_values: {noise_values}")
    print(f"  BP shots={shots}, SA shots={sa_shots}, seed={seed}")
    print(f"  SA: alpha={alpha}, reads={num_reads}, steps/qubit={num_steps_per_qubit}")
    print(f"  HW: annealing={annealing_time_ns}ns, cycle={cycle_time_ns}ns, bp_iter={bp_max_iter}")
    print(f"  BP available: {HAS_LDPC}")

    all_results: list[dict[str, Any]] = []

    for cc in codes:
        H = np.ascontiguousarray(cc.code.hz.astype(np.float64))
        hw = ising_machine_metrics(
            H, annealing_time_ns=annealing_time_ns,
            cycle_time_ns=cycle_time_ns, bp_max_iter=bp_max_iter,
        )

        if verbose:
            print(f"\n{'='*70}")
            print(f"Code: {cc.name}  [[{cc.code.N},{cc.code.K},{cc.code.D}]]  "
                  f"H={H.shape}, spins={hw['num_spins']}, "
                  f"couplers={hw['num_couplers_modified_qubo']}")
            print(f"  HW cycles: annealing={hw['annealing_cycles']}, "
                  f"bp_init={hw['bp_init_cycles']}, "
                  f"total={hw['total_cycles']}, "
                  f"latency={hw['hardware_latency_ns']}ns")

        for noise_p in noise_values:
            rec = _run_single_code_single_p(
                cc, shots=shots, sa_shots=sa_shots,
                noise_p=noise_p, seed=seed,
                alpha=alpha, num_reads=num_reads,
                num_steps_per_qubit=num_steps_per_qubit,
                T_init=T_init, annealing_time_ns=annealing_time_ns,
                cycle_time_ns=cycle_time_ns,
                bp_max_iter=bp_max_iter,
                convergence_patience=convergence_patience,
                ising_sim_cycles=ising_sim_cycles, verbose=verbose,
            )
            rec["ising_machine"] = hw
            all_results.append(rec)

    # --- Summary tables ---
    _print_summary(all_results, noise_values)

    # --- Save ---
    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    return all_results


def _print_summary(results: list[dict], noise_values: list[float]):
    """Print comparison summary tables."""
    # Collect all decoder names that appeared
    all_dec_names = set()
    for rec in results:
        all_dec_names.update(rec.get("decoders", {}).keys())
    # Order: qubo_sa first, then bp variants, then ising_sim
    dec_order = []
    for name in ["qubo_sa", "bp", "bp_osd", "bp_lsd", "ising_sim"]:
        if name in all_dec_names:
            dec_order.append(name)

    # --- LER comparison table ---
    print(f"\n{'='*90}")
    print("LOGICAL ERROR RATE COMPARISON")
    print(f"{'='*90}")
    dec_hdr = "  ".join(f"{d:>10}" for d in dec_order)
    print(f"{'Code':<16} {'p':>6}  {dec_hdr}")
    print("-" * (24 + 12 * len(dec_order)))

    for rec in results:
        decs = rec.get("decoders", {})
        vals = []
        for d in dec_order:
            info = decs.get(d, {})
            ler = info.get("logical_error_rate", None)
            if ler is not None:
                vals.append(f"{ler:>10.4f}")
            elif info.get("status") == "error":
                vals.append(f"{'ERR':>10}")
            else:
                vals.append(f"{'--':>10}")
        print(f"{rec['code']:<16} {rec['noise_p']:>6.3f}  {'  '.join(vals)}")

    # --- Latency comparison table ---
    print(f"\n{'='*90}")
    print("LATENCY (ms/shot) COMPARISON")
    print(f"{'='*90}")
    print(f"{'Code':<16} {'p':>6}  {dec_hdr}")
    print("-" * (24 + 12 * len(dec_order)))

    for rec in results:
        decs = rec.get("decoders", {})
        vals = []
        for d in dec_order:
            info = decs.get(d, {})
            ms = info.get("per_shot_ms", None)
            if ms is not None:
                vals.append(f"{ms:>10.2f}")
            else:
                vals.append(f"{'--':>10}")
        print(f"{rec['code']:<16} {rec['noise_p']:>6.3f}  {'  '.join(vals)}")

    # --- Ising machine hardware summary (once per code) ---
    seen_codes = set()
    hw_rows = []
    for rec in results:
        if rec["code"] not in seen_codes:
            seen_codes.add(rec["code"])
            hw_rows.append(rec)

    print(f"\n{'='*100}")
    print("ISING MACHINE HARDWARE METRICS")
    print(f"{'='*100}")
    print(f"{'Code':<16} {'N':>4} {'Spins':>6} {'Couplers':>9} {'XNOR':>5} "
          f"{'AnnealCyc':>10} {'BPinitCyc':>10} {'TotalCyc':>9} "
          f"{'Latency(ns)':>12}")
    print("-" * 100)
    for rec in hw_rows:
        hw = rec.get("ising_machine", {})
        print(f"{rec['code']:<16} {rec['N']:>4} "
              f"{hw.get('num_spins',0):>6} "
              f"{hw.get('num_couplers_modified_qubo',0):>9} "
              f"{hw.get('xnor_tree_depth',0):>5} "
              f"{hw.get('annealing_cycles',0):>10} "
              f"{hw.get('bp_init_cycles',0):>10} "
              f"{hw.get('total_cycles',0):>9} "
              f"{hw.get('hardware_latency_ns',0):>12.1f}")


# ---------------------------------------------------------------------------
# pytest entry
# ---------------------------------------------------------------------------
def test_ising_decoder_smoke():
    """Smoke test: 8 shots on 2 small codes, single p."""
    results = run_benchmark(
        code_names=["LDPC_25_3_4", "QCC_18_4_4"],
        shots=8,
        noise_values=[0.01],
        num_reads=4,
        num_steps_per_qubit=20,
        ising_sim_cycles=100,
        verbose=True,
    )
    assert len(results) > 0
    for rec in results:
        sa = rec.get("decoders", {}).get("qubo_sa", {})
        assert sa.get("status") == "ok", f"SA failed: {sa}"
        assert 0.0 <= sa["logical_error_rate"] <= 1.0


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ising / Modified-QUBO decoder benchmark with BP comparison"
    )
    parser.add_argument("--codes", nargs="*", default=None)
    parser.add_argument("--shots", type=int, default=1000000,
                        help="Shots for BP decoders (default: 1e6)")
    parser.add_argument("--sa_shots", type=int, default=None,
                        help="Shots for QUBO-SA (default: same as --shots)")
    parser.add_argument(
        "--noise_values", type=str, default="0.001,0.003,0.005,0.008,0.01",
        help="Comma-separated noise probabilities")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--alpha", type=str, default="auto",
                        help="Parity weight: 'auto' or a float (default: auto)")
    parser.add_argument("--num_reads", type=int, default=16)
    parser.add_argument("--num_steps_per_qubit", type=int, default=50)
    parser.add_argument("--T_init", type=float, default=2.0)
    parser.add_argument("--annealing_time_ns", type=float, default=320.0)
    parser.add_argument("--cycle_time_ns", type=float, default=4.0,
                        help="Hardware cycle time in ns (default: 4)")
    parser.add_argument("--bp_max_iter", type=int, default=50,
                        help="BP init max iterations (default: 50)")
    parser.add_argument("--convergence_patience", type=int, default=3,
                        help="SA early stop patience (0=disabled, default: 3)")
    parser.add_argument("--ising_sim_cycles", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    noise_values = [float(x) for x in args.noise_values.split(",")]
    alpha = "auto" if args.alpha == "auto" else float(args.alpha)

    run_benchmark(
        code_names=args.codes,
        shots=args.shots,
        sa_shots=args.sa_shots,
        noise_values=noise_values,
        seed=args.seed,
        alpha=alpha,
        num_reads=args.num_reads,
        num_steps_per_qubit=args.num_steps_per_qubit,
        T_init=args.T_init,
        annealing_time_ns=args.annealing_time_ns,
        cycle_time_ns=args.cycle_time_ns,
        bp_max_iter=args.bp_max_iter,
        convergence_patience=args.convergence_patience,
        ising_sim_cycles=args.ising_sim_cycles,
        verbose=not args.quiet,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
