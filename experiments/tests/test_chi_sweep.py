#!/usr/bin/env python3
"""Task 4: Sweep chi (virtual bond dimension) for MPS decoder.

Measures logical error rate and decode latency across chi values
for multiple QLDPC codes.

Usage:
  .venv-mps/bin/python experiments/tests/test_chi_sweep.py
  .venv-mps/bin/python -m pytest experiments/tests/test_chi_sweep.py -v -s
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import dataclass
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

from experiments.codes.codes import gen_BB_code, gen_HP_ring_code, get_benchmark_code
from quimb import oset
from quimb.tensor import Tensor, TensorNetwork

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


def _sample(H, noise_p, shots, seed):
    rng = np.random.default_rng(seed)
    n = H.shape[1]
    errors = (rng.random((shots, n)) < noise_p).astype(np.int8)
    syndromes = (errors @ H.T.astype(np.int8)) % 2
    return errors, syndromes.astype(np.float64)


# ---------------------------------------------------------------------------
# Chi sweep
# ---------------------------------------------------------------------------
CHI_VALUES = [2, 4, 8, 16, 32, 64]

CODES = [
    ("BB_72", lambda: gen_BB_code(72)),
    ("BB_144", lambda: gen_BB_code(144)),
    ("HP_162", lambda: gen_HP_ring_code(9, 9)),
]


def run_chi_sweep(
    chi_values: list[int] | None = None,
    shots: int = 64,
    noise_p: float = 0.01,
    seed: int = 2026,
    verbose: bool = False,
) -> list[dict]:
    if chi_values is None:
        chi_values = CHI_VALUES

    all_results: list[dict] = []
    use_gpu = HAS_GPU and HAS_CUDAQ

    for code_name, code_factory in CODES:
        code = code_factory()
        H = code.hz.astype(np.float64)
        logical_row = code.lz[0].astype(np.float64)
        num_checks, num_errors = H.shape

        errors, syndromes = _sample(H, noise_p, shots, seed)
        true_logicals = (errors @ logical_row.reshape(-1).astype(np.int8)) % 2

        if verbose:
            print(f"\n{'='*60}")
            print(f"Code: {code_name}  N={code.N} K={code.K}")
            print(f"  Backend: {'GPU' if use_gpu else 'CPU'}")

        # Build TN once for CPU path (reuse across chi values)
        full_tn = None
        syn_param_inds = None
        logical_obs_inds = None
        if not use_gpu:
            full_tn, syn_param_inds, logical_obs_inds = _build_parameterised_tn(
                H, logical_row, noise_p)

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
                else:
                    t0 = time.perf_counter()
                    chain, compressed_tn, stats = _compile_chain_cpu(
                        full_tn, syn_param_inds, logical_obs_inds, chi)
                    compile_ms = (time.perf_counter() - t0) * 1e3
                    record["compile_ms"] = round(compile_ms, 2)

                    record["truncation_error"] = float(stats.truncation_error)
                    record["max_bond_dim"] = int(stats.max_bond_dim)
                    record["offline_steps"] = int(stats.steps)

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
                    print(f"  chi={chi:4d}: LER={ler:.4f}, "
                          f"decode={record['total_decode_ms']:.1f}ms, "
                          f"trunc_err={trunc}")

            except Exception as e:
                record["status"] = "error"
                record["error"] = str(e)
                record["logical_error_rate"] = None
                if verbose:
                    print(f"  chi={chi:4d}: ERROR {e}")

            all_results.append(record)

    return all_results


# ---------------------------------------------------------------------------
# Pytest
# ---------------------------------------------------------------------------
def test_chi_sweep_smoke():
    """Smoke test: sweep 2 chi values on smallest code."""
    results = run_chi_sweep(
        chi_values=[4, 8], shots=8, noise_p=0.01, seed=42, verbose=True)
    assert len(results) > 0
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    assert ok_count > 0, f"No chi sweep succeeded: {results}"
    print(f"\nChi sweep smoke test: {ok_count}/{len(results)} ok")


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"GPU available: {HAS_GPU}")
    print(f"cudaq_qec available: {HAS_CUDAQ}")
    print()

    results = run_chi_sweep(
        chi_values=[2, 4, 8, 16, 32, 64],
        shots=10000, noise_p=0.01, seed=2026, verbose=True)

    out_dir = WORKSPACE / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chi_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'Code':<12} {'chi':>5} {'LER':>8} {'decode_ms':>10} {'trunc_err':>12} {'status'}")
    print("-" * 60)
    for r in results:
        ler_str = f"{r['logical_error_rate']:.4f}" if r.get('logical_error_rate') is not None else "N/A"
        dec_str = f"{r.get('total_decode_ms', 0):.1f}" if r.get('total_decode_ms') else "N/A"
        trunc_str = f"{r.get('truncation_error', 0):.2e}" if r.get('truncation_error') is not None else "N/A"
        print(f"{r['code']:<12} {r['chi']:>5} {ler_str:>8} {dec_str:>10} {trunc_str:>12} {r.get('status','?')}")
