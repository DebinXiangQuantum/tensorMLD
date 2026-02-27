#!/usr/bin/env python3
"""Chi sweep benchmark: sweep chi (virtual bond dimension) across noise rates.

Measures logical error rate and decode latency across chi values and noise
rates for multiple QLDPC codes.  Also runs BP-OSD as a baseline comparison
and records per-step truncation errors from the MPS compression pipeline.

Usage:
  python experiments/tests/test_chi_sweep.py
  python experiments/tests/test_chi_sweep.py --small   # small codes only
  python -m pytest experiments/tests/test_chi_sweep.py -v -s
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent.parent
QEC_PY = WORKSPACE / "cudaqx" / "libs" / "qec" / "python"
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
if str(QEC_PY) not in sys.path:
    sys.path.insert(0, str(QEC_PY))

# ---------------------------------------------------------------------------
# GPU / cudaq detection
# ---------------------------------------------------------------------------
HAS_GPU = False
try:
    import cupy
    HAS_GPU = cupy.cuda.is_available()
except Exception:
    pass

HAS_CUDAQ = False
try:
    import cudaq_qec as qec
    HAS_CUDAQ = True
except Exception:
    qec = None  # type: ignore

# ---------------------------------------------------------------------------
# BP decoders (ldpc package)
# ---------------------------------------------------------------------------
HAS_LDPC = False
try:
    from ldpc import BpOsdDecoder
    HAS_LDPC = True
except ImportError:
    BpOsdDecoder = None  # type: ignore

# ---------------------------------------------------------------------------
# Core module import
# ---------------------------------------------------------------------------
_TNU = QEC_PY / "cudaq_qec" / "plugins" / "decoders" / "tensor_network_utils"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


core = _load_module("mps_decoder_core", _TNU / "mps_decoder_core.py")
factory = _load_module("tensor_network_factory", _TNU / "tensor_network_factory.py")

from experiments.tests._isca_code_registry import (
    load_isca_code_cases,
    resolve_parallel_workers,
)
from quimb import oset
from quimb.tensor import Tensor, TensorNetwork

# ---------------------------------------------------------------------------
# Default sweep parameters
# ---------------------------------------------------------------------------
CHI_VALUES = [2, 4, 8, 16, 32, 64, 128, 256, 512]
NOISE_P_VALUES = [0.001, 0.005, 0.01]

# Small codes for quick validation runs
SMALL_CODES = ["QCC_18_4_4", "LDPC_25_3_4", "LDPC_30_6_4", "TOR_50_2_5"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_parameterised_tn(
    H: np.ndarray, logical_row: np.ndarray, noise_p: float
) -> tuple[TensorNetwork, list[str], list[str]]:
    """Build full parameterised TN (syndrome params as open legs)."""
    num_checks, num_errors = H.shape
    check_inds = [f"s_{i}" for i in range(num_checks)]
    error_inds = [f"e_{i}" for i in range(num_errors)]
    syn_param_inds = [f"syn_param_{i}" for i in range(num_checks)]
    logical_obs_inds = ["obs"]

    code_tn = factory.tensor_network_from_parity_check(
        H.astype(np.float64), col_inds=error_inds, row_inds=check_inds)

    logical_2d = logical_row.reshape(1, -1).astype(np.float64)
    logical_inds = ["l_0"]
    log_tn = factory.tensor_network_from_parity_check(
        logical_2d, col_inds=error_inds, row_inds=logical_inds, tags=["LOG_0"])
    log_tn = log_tn.combine(
        factory.tensor_network_from_logical_observable(
            logical_2d, logical_inds, logical_obs_inds, ["LOG_0"]),
        virtual=True)

    param_syn_tn = core.make_parametrized_syndrome_network(check_inds, syn_param_inds)

    noise_tensors = [Tensor(
        data=np.array([1.0 - noise_p, noise_p], dtype=np.float64),
        inds=(error_inds[j],), tags=oset([f"NOISE_{j}", "NOISE"]))
        for j in range(num_errors)]
    noise_tn = TensorNetwork(noise_tensors)

    full_tn = TensorNetwork()
    full_tn = full_tn.combine(code_tn, virtual=True)
    full_tn = full_tn.combine(log_tn, virtual=True)
    full_tn = full_tn.combine(param_syn_tn, virtual=True)
    full_tn = full_tn.combine(noise_tn, virtual=True)

    return full_tn, syn_param_inds, logical_obs_inds


def _compile_chain_cpu(
    full_tn: TensorNetwork,
    syn_param_inds: list[str],
    logical_obs_inds: list[str],
    chi: int,
) -> tuple[Any, Any, Any]:
    """Compile TN to 1D chain with given chi (CPU path).

    Returns (chain, compressed_tn, offline_stats).
    """
    preserve = set(syn_param_inds + logical_obs_inds)
    result = core.compile_to_1d_chain(
        tn=full_tn.copy(),
        preserve_inds=preserve,
        syndrome_inds=set(syn_param_inds),
        logical_inds=set(logical_obs_inds),
        chi=chi,
    )
    return result.chain, result.compressed_tn, result.offline_stats


def _decode_shots_cpu(
    chain, syndromes: np.ndarray, num_checks: int
) -> tuple[np.ndarray, float]:
    """Decode all shots via MPS chain, return (probs, total_ms)."""
    shots = syndromes.shape[0]
    probs = np.empty(shots)
    t0 = time.perf_counter()
    for i in range(shots):
        syn_vals = {f"syn_param_{j}": float(syndromes[i, j]) for j in range(num_checks)}
        _, marginals = chain.decode_logicals(syndrome_values=syn_vals)
        probs[i] = float(marginals.get("obs", 0.5))
    total_ms = (time.perf_counter() - t0) * 1e3
    return probs, total_ms


def _decode_shots_compressed_tn(
    compressed_tn, syndromes: np.ndarray, num_checks: int
) -> tuple[np.ndarray, float]:
    """Decode all shots via CompressedTNDecoder, return (probs, total_ms)."""
    shots = syndromes.shape[0]
    probs = np.empty(shots)
    t0 = time.perf_counter()
    for i in range(shots):
        syn_vals = {f"syn_param_{j}": float(syndromes[i, j]) for j in range(num_checks)}
        _, marginals = compressed_tn.decode_logicals(syndrome_values=syn_vals)
        probs[i] = float(marginals.get("obs", 0.5))
    total_ms = (time.perf_counter() - t0) * 1e3
    return probs, total_ms


def _decode_shots_gpu(
    H: np.ndarray, logical_row: np.ndarray, noise_p: float,
    chi: int, syndromes: np.ndarray, verbose: bool = False,
) -> tuple[np.ndarray, float, float, Any]:
    """GPU path: build decoder via cudaq_qec, decode shots.
    Returns (probs, init_ms, decode_ms, offline_stats).
    """
    num_checks, num_errors = H.shape
    t0 = time.perf_counter()
    dec = qec.get_decoder(
        "tensor_network_mps_decoder", H,
        logical_obs=logical_row.reshape(1, -1),
        noise_model=[noise_p] * num_errors,
        dtype="float32", device="cuda",
        bond_dim=chi, verbose=verbose)
    init_ms = (time.perf_counter() - t0) * 1e3

    shots = syndromes.shape[0]
    probs = np.empty(shots)
    t1 = time.perf_counter()
    for i in range(shots):
        res = dec.decode([float(x) for x in syndromes[i]])
        probs[i] = float(res.result[0])
    decode_ms = (time.perf_counter() - t1) * 1e3

    stats = getattr(dec, "offline_stats", None)
    return probs, init_ms, decode_ms, stats


def _decode_shots_bposd(
    H: np.ndarray, logical_row: np.ndarray, noise_p: float,
    syndromes_int: np.ndarray, true_logicals: np.ndarray,
) -> dict[str, Any]:
    """Run BP-OSD decoder and return result dict."""
    num_checks, num_errors = H.shape
    H_int = H.astype(np.uint8)
    channel_probs = np.full(num_errors, noise_p)

    t0 = time.perf_counter()
    bp_dec = BpOsdDecoder(
        H_int, error_rate=noise_p,
        channel_probs=channel_probs,
        max_iter=50, bp_method="ms",
        osd_method="osd_cs", osd_order=10)
    init_ms = (time.perf_counter() - t0) * 1e3

    shots = syndromes_int.shape[0]
    predictions = np.empty(shots, dtype=np.int8)
    t1 = time.perf_counter()
    for i in range(shots):
        correction = bp_dec.decode(syndromes_int[i])
        predictions[i] = int(correction @ logical_row.reshape(-1).astype(np.int8)) % 2
    decode_ms = (time.perf_counter() - t1) * 1e3

    ler = float(np.mean(true_logicals != predictions))
    return {
        "status": "ok",
        "init_ms": round(init_ms, 2),
        "total_decode_ms": round(decode_ms, 2),
        "per_shot_ms": round(decode_ms / shots, 3),
        "logical_error_rate": round(ler, 6),
    }


def _extract_step_errors(stats) -> list[dict]:
    """Extract per-step truncation errors from OfflineCompressionStats."""
    if not hasattr(stats, "step_errors") or not stats.step_errors:
        return []
    return [rec.to_dict() for rec in stats.step_errors]


def _sample(H, noise_p, shots, seed):
    rng = np.random.default_rng(seed)
    n = H.shape[1]
    errors = (rng.random((shots, n)) < noise_p).astype(np.int8)
    syndromes = (errors @ H.T.astype(np.int8)) % 2
    return errors, syndromes.astype(np.float64)


# ---------------------------------------------------------------------------
# Core sweep: one code, one noise_p, all chi values + BP-OSD
# ---------------------------------------------------------------------------
def _run_one_code_one_p(
    code_name: str,
    code: Any,
    *,
    chi_values: list[int],
    shots: int,
    noise_p: float,
    seed: int,
    verbose: bool,
    use_gpu: bool,
    run_bposd: bool = True,
) -> list[dict]:
    """Sweep chi values for one code at one noise rate. Also runs BP-OSD."""
    H = code.hz.astype(np.float64)
    logical_row = code.lz[0].astype(np.float64)
    num_checks, _num_errors = H.shape

    errors, syndromes = _sample(H, noise_p, shots, seed)
    syndromes_int = syndromes.astype(np.uint8)
    true_logicals = (errors @ logical_row.reshape(-1).astype(np.int8)) % 2

    if verbose:
        print(f"\n{'='*60}")
        print(f"Code: {code_name}  N={code.N} K={code.K}  p={noise_p}")
        print(f"  Backend: {'GPU' if use_gpu else 'CPU'}, BP-OSD: {run_bposd and HAS_LDPC}")

    # --- BP-OSD baseline (once per code/p, independent of chi) ---
    bposd_result: dict[str, Any] | None = None
    if run_bposd and HAS_LDPC:
        try:
            bposd_result = _decode_shots_bposd(
                H, logical_row, noise_p, syndromes_int, true_logicals)
            if verbose:
                print(f"  BP-OSD: LER={bposd_result['logical_error_rate']:.4f}, "
                      f"decode={bposd_result['total_decode_ms']:.1f}ms")
        except Exception as e:
            bposd_result = {"status": "error", "error": str(e)}
            if verbose:
                print(f"  BP-OSD: ERROR {e}")

    # Build TN once per (code, noise_p) for CPU path
    full_tn = None
    syn_param_inds = None
    logical_obs_inds = None
    if not use_gpu:
        full_tn, syn_param_inds, logical_obs_inds = _build_parameterised_tn(
            H, logical_row, noise_p)

    code_results: list[dict] = []
    for chi in chi_values:
        record: dict[str, Any] = {
            "code": code_name,
            "N": int(code.N),
            "K": int(code.K),
            "chi": chi,
            "noise_p": noise_p,
            "shots": shots,
            "backend": "gpu" if use_gpu else "cpu",
        }
        # Attach BP-OSD result to each chi record for easy comparison
        if bposd_result is not None:
            record["bposd"] = bposd_result

        try:
            if use_gpu:
                probs, init_ms, decode_ms, stats = _decode_shots_gpu(
                    H, logical_row, noise_p, chi, syndromes, verbose=verbose)
                record["init_ms"] = round(init_ms, 2)
                record["total_decode_ms"] = round(decode_ms, 2)
                record["per_shot_ms"] = round(decode_ms / shots, 3)
                if stats is not None:
                    record["truncation_error"] = float(stats.truncation_error)
                    record["max_bond_dim"] = int(stats.max_bond_dim)
                    record["offline_steps"] = int(stats.steps)
                    record["step_errors"] = _extract_step_errors(stats)
            else:
                t0 = time.perf_counter()
                chain, compressed_tn, stats = _compile_chain_cpu(
                    full_tn, syn_param_inds, logical_obs_inds, chi)
                compile_ms = (time.perf_counter() - t0) * 1e3
                record["compile_ms"] = round(compile_ms, 2)

                record["truncation_error"] = float(stats.truncation_error)
                record["max_bond_dim"] = int(stats.max_bond_dim)
                record["offline_steps"] = int(stats.steps)
                record["chi_requested"] = chi
                # Step-by-step truncation errors for analysis
                record["step_errors"] = _extract_step_errors(stats)

                if chain is not None:
                    probs, decode_ms = _decode_shots_cpu(chain, syndromes, num_checks)
                    record["decode_method"] = "mps_chain"
                elif compressed_tn is not None:
                    probs, decode_ms = _decode_shots_compressed_tn(
                        compressed_tn, syndromes, num_checks)
                    record["decode_method"] = "compressed_tn"

                record["total_decode_ms"] = round(decode_ms, 2)
                record["per_shot_ms"] = round(decode_ms / shots, 3)

            predictions = (probs >= 0.5).astype(np.int8)
            ler = float(np.mean(true_logicals != predictions))
            record["logical_error_rate"] = round(ler, 6)
            record["status"] = "ok"

            if verbose:
                trunc = record.get("truncation_error", "N/A")
                n_steps = len(record.get("step_errors", []))
                print(f"  chi={chi:4d}: LER={ler:.4f}, "
                      f"decode={record['total_decode_ms']:.1f}ms, "
                      f"trunc_err={trunc}, steps={n_steps}")

        except Exception as e:
            record["status"] = "error"
            record["error"] = str(e)
            record["logical_error_rate"] = None
            if verbose:
                print(f"  chi={chi:4d}: ERROR {e}")

        code_results.append(record)
    return code_results


# ---------------------------------------------------------------------------
# Top-level sweep: iterate over codes × noise rates
# ---------------------------------------------------------------------------
def run_chi_sweep(
    chi_values: list[int] | None = None,
    noise_p_values: list[float] | None = None,
    shots: int = 10000,
    seed: int = 2026,
    code_names: list[str] | None = None,
    skip_missing: bool = False,
    parallel_workers: int | None = None,
    run_bposd: bool = True,
    verbose: bool = False,
) -> list[dict]:
    if chi_values is None:
        chi_values = CHI_VALUES
    if noise_p_values is None:
        noise_p_values = NOISE_P_VALUES

    use_gpu = HAS_GPU and HAS_CUDAQ
    loaded_codes, skipped = load_isca_code_cases(
        code_names=code_names,
        skip_missing=skip_missing,
    )
    if verbose and skipped:
        print(f"[codes] skipped {len(skipped)} code(s):")
        for line in skipped:
            print(f"  - {line}")
    if not loaded_codes:
        raise RuntimeError("No codes were loaded for chi sweep.")

    workers = resolve_parallel_workers(
        use_gpu=use_gpu,
        parallel_workers=parallel_workers,
    )
    if verbose:
        print(f"[chi_sweep] codes={len(loaded_codes)}, noise_p={noise_p_values}, "
              f"chi={chi_values}")
        print(f"  workers={workers}, use_gpu={use_gpu}, bposd={run_bposd and HAS_LDPC}")

    # Build task list: (code_name, code, noise_p) triples
    tasks = [
        (code_name, code, p)
        for code_name, code in loaded_codes
        for p in noise_p_values
    ]

    if workers == 1 or len(tasks) == 1:
        results: list[dict] = []
        for code_name, code, p in tasks:
            results.extend(_run_one_code_one_p(
                code_name, code,
                chi_values=chi_values, shots=shots, noise_p=p,
                seed=seed, verbose=verbose, use_gpu=use_gpu,
                run_bposd=run_bposd,
            ))
        return results

    ordered: list[Optional[list[dict]]] = [None] * len(tasks)
    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _run_one_code_one_p,
                code_name, code,
                chi_values=chi_values, shots=shots, noise_p=p,
                seed=seed, verbose=verbose, use_gpu=use_gpu,
                run_bposd=run_bposd,
            ): idx for idx, (code_name, code, p) in enumerate(tasks)
        }
        for future in cf.as_completed(future_map):
            ordered[future_map[future]] = future.result()

    results = []
    for part in ordered:
        if part:
            results.extend(part)
    return results


def _print_summary(results: list[dict]) -> None:
    """Print summary table grouped by noise_p."""
    noise_ps = sorted(set(r["noise_p"] for r in results))
    for p in noise_ps:
        subset = [r for r in results if r["noise_p"] == p]
        print(f"\n--- noise_p = {p} ---")
        print(f"{'Code':<16} {'chi':>5} {'MPS_LER':>8} {'BP-OSD':>8} "
              f"{'decode_ms':>10} {'trunc_err':>12} {'steps':>6} {'status'}")
        print("-" * 80)
        for r in subset:
            ler_str = f"{r['logical_error_rate']:.4f}" if r.get('logical_error_rate') is not None else "N/A"
            bposd_ler = r.get("bposd", {}).get("logical_error_rate")
            bp_str = f"{bposd_ler:.4f}" if bposd_ler is not None else "N/A"
            dec_str = f"{r.get('total_decode_ms', 0):.1f}" if r.get('total_decode_ms') else "N/A"
            trunc_str = f"{r.get('truncation_error', 0):.2e}" if r.get('truncation_error') is not None else "N/A"
            n_steps = len(r.get("step_errors", []))
            print(f"{r['code']:<16} {r['chi']:>5} {ler_str:>8} {bp_str:>8} "
                  f"{dec_str:>10} {trunc_str:>12} {n_steps:>6} {r.get('status', '?')}")


# ---------------------------------------------------------------------------
# Pytest
# ---------------------------------------------------------------------------
def test_chi_sweep_smoke():
    """Smoke test: sweep 2 chi values on smallest code."""
    results = run_chi_sweep(
        chi_values=[4, 8],
        noise_p_values=[0.01],
        shots=8,
        seed=42,
        code_names=["QCC_18_4_4"],
        skip_missing=True,
        parallel_workers=1,
        run_bposd=True,
        verbose=True,
    )
    assert len(results) > 0
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    assert ok_count > 0, f"No chi sweep succeeded: {results}"
    # Check step_errors are recorded
    for r in results:
        if r.get("status") == "ok":
            assert "step_errors" in r, "step_errors not recorded"
    # Check BP-OSD result is attached
    if HAS_LDPC:
        for r in results:
            assert "bposd" in r, "BP-OSD result not attached"
    print(f"\nChi sweep smoke test: {ok_count}/{len(results)} ok")


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chi sweep benchmark")
    parser.add_argument("--small", action="store_true",
                        help="Run only small codes for quick validation")
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-bposd", action="store_true",
                        help="Skip BP-OSD baseline")
    parser.add_argument("--chi", type=int, nargs="+", default=None,
                        help="Override chi values")
    parser.add_argument("--noise-p", type=float, nargs="+", default=None,
                        help="Override noise_p values")
    parser.add_argument("--codes", type=str, nargs="+", default=None,
                        help="Override code names")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override parallel workers (default: auto)")
    args = parser.parse_args()

    print(f"GPU available: {HAS_GPU}")
    print(f"cudaq_qec available: {HAS_CUDAQ}")
    print(f"ldpc (BP-OSD) available: {HAS_LDPC}")
    print()

    code_names = args.codes
    if args.small and code_names is None:
        code_names = SMALL_CODES

    chi_values = args.chi if args.chi else CHI_VALUES
    noise_p_values = args.noise_p if args.noise_p else NOISE_P_VALUES

    out_dir = WORKSPACE / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chi_sweep_results.json"
    step_path = out_dir / "chi_sweep_step_errors.json"

    # Incremental save: save after EACH chi value to survive OOM crashes.
    all_results: list[dict] = []
    all_step_errors: list[dict] = []

    def _save_incremental():
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        if all_step_errors:
            with open(step_path, "w") as f:
                json.dump(all_step_errors, f, indent=2)

    use_gpu = HAS_GPU and HAS_CUDAQ
    loaded_codes, skipped = load_isca_code_cases(
        code_names=code_names, skip_missing=False)
    if skipped:
        print(f"[codes] skipped: {skipped}")

    print(f"[chi_sweep] codes={len(loaded_codes)}, noise_p={noise_p_values}, "
          f"chi={chi_values}, workers={args.workers or 'auto'}")

    for code_name, code in loaded_codes:
        for p in noise_p_values:
            print(f"\n>>> Running {code_name} p={p} ...")
            # Run ONE chi at a time so incremental save survives OOM
            batch_ok = 0
            batch_total = 0
            for chi in chi_values:
                batch = _run_one_code_one_p(
                    code_name, code,
                    chi_values=[chi], shots=args.shots, noise_p=p,
                    seed=args.seed, verbose=True, use_gpu=use_gpu,
                    run_bposd=(not args.no_bposd and chi == chi_values[0]),
                )
                for r in batch:
                    # Carry BP-OSD result to all chi records
                    if "bposd" not in r and all_results:
                        prev = [x for x in all_results
                                if x["code"] == code_name
                                and x["noise_p"] == p
                                and "bposd" in x]
                        if prev:
                            r["bposd"] = prev[0]["bposd"]
                    all_results.append(r)
                    if r.get("step_errors"):
                        all_step_errors.append({
                            "code": r["code"], "N": r["N"], "K": r["K"],
                            "chi": r["chi"], "noise_p": r["noise_p"],
                            "truncation_error": r.get("truncation_error"),
                            "max_bond_dim": r.get("max_bond_dim"),
                            "offline_steps": r.get("offline_steps"),
                            "step_errors": r["step_errors"],
                        })
                    batch_total += 1
                    if r.get("status") == "ok":
                        batch_ok += 1
                    # Save after EVERY chi value — survives OOM on next chi
                    _save_incremental()

            print(f"  >>> {code_name} p={p}: {batch_ok}/{batch_total} ok, "
                  f"{len(all_results)} total saved")

    print(f"\nFinal results saved to {out_path}")
    if all_step_errors:
        print(f"Step errors saved to {step_path}")
    _print_summary(all_results)
