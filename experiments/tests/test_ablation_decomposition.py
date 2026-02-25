#!/usr/bin/env python3
"""Ablation study: Compare decomposition methods for tensor network decoding.

Responds to reviewer concern: "The tensors in question appear highly structured,
making it potentially trivial to generate entries on the fly rather than reading
them from memory."

Three methods compared:
  Method A: Naive direct contraction (no decomposition, cotengra-optimized path)
  Method B: Quimb built-in compression (compress_all + contract)
  Method C: Our MPS decomposition (exact decomposition + offline contraction)

Metrics for each:
  - Estimated peak memory (contraction width, in log2 #floats)
  - Estimated FLOPs (contraction cost)
  - Actual wall time (offline + online)
  - Actual peak memory (tracemalloc)
  - Logical error rate (accuracy)

Usage:
  PYTHONPATH=cudaqx/libs/qec/python:. .venv-mps/bin/python experiments/tests/test_ablation_decomposition.py
  PYTHONPATH=cudaqx/libs/qec/python:. .venv-mps/bin/python -m pytest experiments/tests/test_ablation_decomposition.py -v -s
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
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


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


_TNU = QEC_PY / "cudaq_qec" / "plugins" / "decoders" / "tensor_network_utils"
core = _load_module("mps_decoder_core", _TNU / "mps_decoder_core.py")
factory = _load_module("tensor_network_factory", _TNU / "tensor_network_factory.py")

from experiments.codes.codes import gen_BB_code, gen_HP_ring_code
from quimb import oset
from quimb.tensor import Tensor, TensorNetwork

# ---------------------------------------------------------------------------
# Codes to test
# ---------------------------------------------------------------------------
CODES = [
    ("BB_72",  lambda: gen_BB_code(72)),
    ("BB_90",  lambda: gen_BB_code(90)),
    ("BB_144", lambda: gen_BB_code(144)),
    ("HP_162", lambda: gen_HP_ring_code(9, 9)),
]

# Smaller set for smoke test
CODES_SMOKE = [
    ("BB_72",  lambda: gen_BB_code(72)),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sample(H: np.ndarray, noise_p: float, shots: int, seed: int):
    """Generate random errors and syndromes."""
    rng = np.random.default_rng(seed)
    n = H.shape[1]
    errors = (rng.random((shots, n)) < noise_p).astype(np.int8)
    syndromes = (errors @ H.T.astype(np.int8)) % 2
    return errors, syndromes.astype(np.float64)


def _build_full_tn(
    H: np.ndarray, logical_row: np.ndarray, noise_p: float
) -> tuple[TensorNetwork, list[str], list[str], list[str]]:
    """Build the full parameterised TN (syndrome params as open legs).

    Returns (full_tn, check_inds, syn_param_inds, logical_obs_inds).
    """
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

    return full_tn, check_inds, syn_param_inds, logical_obs_inds


def _build_concrete_tn(
    H: np.ndarray, logical_row: np.ndarray, noise_p: float,
    syndrome: np.ndarray,
) -> tuple[TensorNetwork, list[str]]:
    """Build TN with a concrete syndrome substituted (for direct contraction).

    Returns (tn, logical_obs_inds).
    """
    num_checks, num_errors = H.shape
    check_inds = [f"s_{i}" for i in range(num_checks)]
    error_inds = [f"e_{i}" for i in range(num_errors)]
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

    # Concrete syndrome vectors: s=0 -> [1,1], s=1 -> [1,-1]
    syn_tensors = []
    for j in range(num_checks):
        s = float(syndrome[j])
        syn_tensors.append(Tensor(
            data=s * np.array([1.0, -1.0]) + (1.0 - s) * np.array([1.0, 1.0]),
            inds=(check_inds[j],), tags=oset([f"SYN_{j}"])))
    syn_tn = TensorNetwork(syn_tensors)

    noise_tensors = [Tensor(
        data=np.array([1.0 - noise_p, noise_p], dtype=np.float64),
        inds=(error_inds[j],), tags=oset([f"NOISE_{j}", "NOISE"]))
        for j in range(num_errors)]
    noise_tn = TensorNetwork(noise_tensors)

    full = TensorNetwork()
    full = full.combine(code_tn, virtual=True)
    full = full.combine(log_tn, virtual=True)
    full = full.combine(syn_tn, virtual=True)
    full = full.combine(noise_tn, virtual=True)

    return full, logical_obs_inds


def _tn_storage_bytes(tn: TensorNetwork) -> int:
    """Total bytes of all tensor data in the network."""
    total = 0
    for t in tn.tensors:
        total += t.data.nbytes
    return total


def _tn_total_elements(tn: TensorNetwork) -> int:
    """Total number of elements across all tensors."""
    total = 0
    for t in tn.tensors:
        total += t.data.size
    return total


# ---------------------------------------------------------------------------
# Method A: Naive direct contraction
# ---------------------------------------------------------------------------
def method_naive_contraction(
    H: np.ndarray, logical_row: np.ndarray, noise_p: float,
    syndromes: np.ndarray, true_logicals: np.ndarray,
    verbose: bool = False,
) -> dict[str, Any]:
    """Directly contract the full TN for each shot (no decomposition).

    Uses cotengra to find a good contraction path, then contracts.
    Measures estimated FLOPs, peak memory (contraction width), and actual timing.
    """
    num_checks = H.shape[0]
    shots = syndromes.shape[0]

    # Build one concrete TN to estimate contraction cost
    tn0, obs_inds = _build_concrete_tn(H, logical_row, noise_p, syndromes[0])
    output_inds = tuple(obs_inds)

    # Estimate contraction cost before actual computation
    try:
        import cotengra
        opt = cotengra.HyperOptimizer(
            minimize='combo', max_repeats=32, progbar=False)
        tree = tn0.contraction_tree(output_inds=output_inds, optimize=opt)
        est_flops = tree.total_flops()
        est_max_size = tree.max_size()  # Number of elements in largest intermediate
        est_peak_size_log2 = math.log2(est_max_size) if est_max_size > 0 else 0
        contraction_path = tree.get_path()
    except Exception:
        # Fallback to basic optimization
        info = tn0.contraction_info(output_inds=output_inds, optimize='auto')
        est_flops = info.opt_cost
        est_max_size = info.largest_intermediate
        est_peak_size_log2 = math.log2(est_max_size) if est_max_size > 0 else 0
        contraction_path = info.path

    storage_bytes = _tn_storage_bytes(tn0)
    num_tensors = len(list(tn0.tensors))
    total_elements = _tn_total_elements(tn0)

    if verbose:
        print(f"    [naive] tensors={num_tensors}, storage={storage_bytes}B, "
              f"elements={total_elements}")
        print(f"    [naive] est_flops={est_flops:.2e}, "
              f"est_peak_size={est_max_size} (log2={est_peak_size_log2:.1f})")

    # Actual contraction with memory tracking
    tracemalloc.start()
    tracemalloc.reset_peak()
    probs = np.empty(shots)
    t0 = time.perf_counter()
    for i in range(shots):
        tn_i, _ = _build_concrete_tn(H, logical_row, noise_p, syndromes[i])
        val = tn_i.contract(output_inds=output_inds, optimize=contraction_path)
        arr = val.data if hasattr(val, 'data') else np.asarray(val)
        arr = np.asarray(arr).ravel()
        if len(arr) < 2:
            probs[i] = 0.5
        else:
            total = float(arr[0] + arr[1])
            probs[i] = float(arr[1] / total) if total > 0 else 0.5
    total_ms = (time.perf_counter() - t0) * 1e3
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    predictions = (probs >= 0.5).astype(np.int8)
    ler = float(np.mean(true_logicals != predictions))

    return {
        "method": "naive_direct",
        "num_tensors": num_tensors,
        "storage_bytes": storage_bytes,
        "total_elements": total_elements,
        "est_flops_per_shot": float(est_flops),
        "est_flops_total": float(est_flops * shots),
        "est_peak_size_elements": int(est_max_size),
        "est_peak_size_log2": round(est_peak_size_log2, 2),
        "est_peak_memory_bytes": int(est_max_size * 8),  # float64
        "actual_total_ms": round(total_ms, 2),
        "actual_per_shot_ms": round(total_ms / shots, 3),
        "actual_peak_memory_bytes": int(peak_mem),
        "logical_error_rate": round(ler, 6),
        "shots": shots,
    }


# ---------------------------------------------------------------------------
# Method B: Quimb built-in compression + contraction
# ---------------------------------------------------------------------------
def method_quimb_compress(
    H: np.ndarray, logical_row: np.ndarray, noise_p: float,
    syndromes: np.ndarray, true_logicals: np.ndarray,
    chi: int = 32,
    verbose: bool = False,
) -> dict[str, Any]:
    """Compress TN using quimb's compress_all, then contract.

    This is the "alternative decomposition method" baseline.
    """
    num_checks = H.shape[0]
    shots = syndromes.shape[0]

    # Build a concrete TN and compress it
    tn0, obs_inds = _build_concrete_tn(H, logical_row, noise_p, syndromes[0])
    output_inds = tuple(obs_inds)

    # Pre-compression stats
    pre_storage = _tn_storage_bytes(tn0)
    pre_tensors = len(list(tn0.tensors))
    pre_elements = _tn_total_elements(tn0)

    # Compress
    t_compress_start = time.perf_counter()
    tn0_compressed = tn0.compress_all(
        max_bond=chi, cutoff=1e-12, inplace=False)
    compress_ms = (time.perf_counter() - t_compress_start) * 1e3

    post_storage = _tn_storage_bytes(tn0_compressed)
    post_tensors = len(list(tn0_compressed.tensors))
    post_elements = _tn_total_elements(tn0_compressed)

    # Estimate contraction cost of compressed TN
    try:
        import cotengra
        opt = cotengra.HyperOptimizer(
            minimize='combo', max_repeats=32, progbar=False)
        tree = tn0_compressed.contraction_tree(
            output_inds=output_inds, optimize=opt)
        est_flops = tree.total_flops()
        est_max_size = tree.max_size()
        est_peak_size_log2 = math.log2(est_max_size) if est_max_size > 0 else 0
        contraction_path = tree.get_path()
    except Exception:
        info = tn0_compressed.contraction_info(
            output_inds=output_inds, optimize='auto')
        est_flops = info.opt_cost
        est_max_size = info.largest_intermediate
        est_peak_size_log2 = math.log2(est_max_size) if est_max_size > 0 else 0
        contraction_path = info.path

    if verbose:
        print(f"    [quimb] pre: tensors={pre_tensors}, "
              f"storage={pre_storage}B, elements={pre_elements}")
        print(f"    [quimb] post: tensors={post_tensors}, "
              f"storage={post_storage}B, elements={post_elements}")
        print(f"    [quimb] compress_ms={compress_ms:.1f}, "
              f"est_flops={est_flops:.2e}, "
              f"est_peak_size={est_max_size} (log2={est_peak_size_log2:.1f})")

    # Actual contraction with memory tracking
    tracemalloc.start()
    tracemalloc.reset_peak()
    probs = np.empty(shots)
    t0 = time.perf_counter()
    for i in range(shots):
        tn_i, _ = _build_concrete_tn(H, logical_row, noise_p, syndromes[i])
        tn_i_c = tn_i.compress_all(max_bond=chi, cutoff=1e-12, inplace=False)
        val = tn_i_c.contract(output_inds=output_inds, optimize=contraction_path)
        arr = val.data if hasattr(val, 'data') else np.asarray(val)
        arr = np.asarray(arr).ravel()
        if len(arr) < 2:
            probs[i] = 0.5
        else:
            total = float(arr[0] + arr[1])
            probs[i] = float(arr[1] / total) if total > 0 else 0.5
    total_ms = (time.perf_counter() - t0) * 1e3
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    predictions = (probs >= 0.5).astype(np.int8)
    ler = float(np.mean(true_logicals != predictions))

    return {
        "method": "quimb_compress",
        "chi": chi,
        "pre_num_tensors": pre_tensors,
        "pre_storage_bytes": pre_storage,
        "pre_total_elements": pre_elements,
        "post_num_tensors": post_tensors,
        "post_storage_bytes": post_storage,
        "post_total_elements": post_elements,
        "compress_ms_per_shot": round(compress_ms, 2),
        "est_flops_per_shot": float(est_flops),
        "est_flops_total": float(est_flops * shots),
        "est_peak_size_elements": int(est_max_size),
        "est_peak_size_log2": round(est_peak_size_log2, 2),
        "est_peak_memory_bytes": int(est_max_size * 8),
        "actual_total_ms": round(total_ms, 2),
        "actual_per_shot_ms": round(total_ms / shots, 3),
        "actual_peak_memory_bytes": int(peak_mem),
        "logical_error_rate": round(ler, 6),
        "shots": shots,
    }


# ---------------------------------------------------------------------------
# Method C: Our MPS decomposition (offline compression + online decode)
# ---------------------------------------------------------------------------
def method_our_mps(
    H: np.ndarray, logical_row: np.ndarray, noise_p: float,
    syndromes: np.ndarray, true_logicals: np.ndarray,
    chi: int = 32,
    verbose: bool = False,
) -> dict[str, Any]:
    """Our method: exact MPS decomposition + offline contraction + online decode.

    Offline phase: build parameterised TN, compress to MPS (CompressedTNDecoder).
    Online phase: substitute syndrome values, contract 1D chain.
    """
    num_checks = H.shape[0]
    shots = syndromes.shape[0]

    # Build parameterised TN (for offline compression)
    full_tn, check_inds, syn_param_inds, logical_obs_inds = _build_full_tn(
        H, logical_row, noise_p)

    pre_storage = _tn_storage_bytes(full_tn)
    pre_tensors = len(list(full_tn.tensors))
    pre_elements = _tn_total_elements(full_tn)

    # Offline: compile to compressed decoder
    preserve = set(syn_param_inds + logical_obs_inds)
    tracemalloc.start()
    tracemalloc.reset_peak()
    t_offline_start = time.perf_counter()
    result = core.compile_to_1d_chain(
        tn=full_tn.copy(),
        preserve_inds=preserve,
        syndrome_inds=set(syn_param_inds),
        logical_inds=set(logical_obs_inds),
        chi=chi,
    )
    offline_ms = (time.perf_counter() - t_offline_start) * 1e3
    _, offline_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    chain = result.chain
    compressed_tn = result.compressed_tn
    stats = result.offline_stats
    decode_method = "1d_chain" if chain is not None else "compressed_tn"

    if verbose:
        print(f"    [ours] pre: tensors={pre_tensors}, "
              f"storage={pre_storage}B, elements={pre_elements}")
        print(f"    [ours] offline: {offline_ms:.1f}ms, "
              f"trunc_err={stats.truncation_error:.2e}, "
              f"max_bond={stats.max_bond_dim}, method={decode_method}")

    # Online: decode all shots
    tracemalloc.start()
    tracemalloc.reset_peak()
    probs = np.empty(shots)
    t_online_start = time.perf_counter()
    for i in range(shots):
        syn_vals = {f"syn_param_{j}": float(syndromes[i, j])
                    for j in range(num_checks)}
        if chain is not None:
            _, marginals = chain.decode_logicals(syndrome_values=syn_vals)
        else:
            _, marginals = compressed_tn.decode_logicals(syndrome_values=syn_vals)
        probs[i] = float(marginals.get("obs", 0.5))
    online_ms = (time.perf_counter() - t_online_start) * 1e3
    _, online_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    predictions = (probs >= 0.5).astype(np.int8)
    ler = float(np.mean(true_logicals != predictions))

    # Estimate the "equivalent" contraction cost for our online decode
    # For 1D chain: O(M * chi^2) per shot (M = num_checks)
    # For compressed TN: similar, multiply matrices along the chain
    mps_online_flops = num_checks * chi * chi * 2  # rough estimate per shot

    return {
        "method": "our_mps",
        "chi": chi,
        "decode_method": decode_method,
        "pre_num_tensors": pre_tensors,
        "pre_storage_bytes": pre_storage,
        "pre_total_elements": pre_elements,
        "offline_ms": round(offline_ms, 2),
        "offline_peak_memory_bytes": int(offline_peak_mem),
        "truncation_error": float(stats.truncation_error),
        "max_bond_dim": int(stats.max_bond_dim),
        "offline_steps": int(stats.steps),
        "est_online_flops_per_shot": mps_online_flops,
        "est_online_flops_total": mps_online_flops * shots,
        "online_total_ms": round(online_ms, 2),
        "online_per_shot_ms": round(online_ms / shots, 3),
        "online_peak_memory_bytes": int(online_peak_mem),
        "total_ms": round(offline_ms + online_ms, 2),
        "logical_error_rate": round(ler, 6),
        "shots": shots,
    }


# ---------------------------------------------------------------------------
# Run ablation study
# ---------------------------------------------------------------------------
def run_ablation(
    codes: list[tuple[str, Any]] | None = None,
    shots: int = 32,
    noise_p: float = 0.01,
    chi: int = 32,
    seed: int = 2026,
    verbose: bool = True,
    skip_naive_above: int = 200,
) -> list[dict]:
    """Run the full ablation study across all codes and methods.

    Args:
        skip_naive_above: Skip naive contraction for codes with N > this value
            (too slow/memory-intensive).
    """
    if codes is None:
        codes = CODES

    all_results: list[dict] = []

    for code_name, code_factory in codes:
        code = code_factory()
        H = code.hz.astype(np.float64)
        logical_row = code.lz[0].astype(np.float64)
        num_checks, num_errors = H.shape

        errors, syndromes = _sample(H, noise_p, shots, seed)
        true_logicals = (errors @ logical_row.reshape(-1).astype(np.int8)) % 2

        if verbose:
            print(f"\n{'='*70}")
            print(f"Code: {code_name}  N={code.N} K={code.K}  "
                  f"H=({num_checks}x{num_errors})  shots={shots}")

        # --- Method A: Naive direct contraction ---
        if num_errors <= skip_naive_above:
            if verbose:
                print(f"\n  Method A: Naive direct contraction")
            try:
                res_a = method_naive_contraction(
                    H, logical_row, noise_p, syndromes, true_logicals,
                    verbose=verbose)
                res_a["code"] = code_name
                res_a["N"] = int(code.N)
                res_a["K"] = int(code.K)
                res_a["noise_p"] = noise_p
                all_results.append(res_a)
                if verbose:
                    print(f"    LER={res_a['logical_error_rate']:.4f}, "
                          f"time={res_a['actual_total_ms']:.1f}ms, "
                          f"peak_mem={res_a['actual_peak_memory_bytes']/1024:.1f}KB, "
                          f"est_flops/shot={res_a['est_flops_per_shot']:.2e}")
            except Exception as e:
                if verbose:
                    print(f"    ERROR: {e}")
                all_results.append({
                    "method": "naive_direct", "code": code_name,
                    "N": int(code.N), "K": int(code.K),
                    "status": "error", "error": str(e),
                })
        else:
            if verbose:
                print(f"\n  Method A: SKIPPED (N={num_errors} > {skip_naive_above})")
            all_results.append({
                "method": "naive_direct", "code": code_name,
                "N": int(code.N), "K": int(code.K),
                "status": "skipped", "reason": f"N={num_errors} > {skip_naive_above}",
            })

        # --- Method B: Quimb compress_all + contract ---
        if verbose:
            print(f"\n  Method B: Quimb compress_all (chi={chi})")
        try:
            res_b = method_quimb_compress(
                H, logical_row, noise_p, syndromes, true_logicals,
                chi=chi, verbose=verbose)
            res_b["code"] = code_name
            res_b["N"] = int(code.N)
            res_b["K"] = int(code.K)
            res_b["noise_p"] = noise_p
            all_results.append(res_b)
            if verbose:
                print(f"    LER={res_b['logical_error_rate']:.4f}, "
                      f"time={res_b['actual_total_ms']:.1f}ms, "
                      f"peak_mem={res_b['actual_peak_memory_bytes']/1024:.1f}KB, "
                      f"est_flops/shot={res_b['est_flops_per_shot']:.2e}")
        except Exception as e:
            if verbose:
                print(f"    ERROR: {e}")
            all_results.append({
                "method": "quimb_compress", "code": code_name,
                "N": int(code.N), "K": int(code.K),
                "status": "error", "error": str(e),
            })

        # --- Method C: Our MPS decomposition ---
        if verbose:
            print(f"\n  Method C: Our MPS decomposition (chi={chi})")
        try:
            res_c = method_our_mps(
                H, logical_row, noise_p, syndromes, true_logicals,
                chi=chi, verbose=verbose)
            res_c["code"] = code_name
            res_c["N"] = int(code.N)
            res_c["K"] = int(code.K)
            res_c["noise_p"] = noise_p
            all_results.append(res_c)
            if verbose:
                print(f"    LER={res_c['logical_error_rate']:.4f}, "
                      f"offline={res_c['offline_ms']:.1f}ms, "
                      f"online={res_c['online_total_ms']:.1f}ms "
                      f"({res_c['online_per_shot_ms']:.3f}ms/shot), "
                      f"est_flops/shot={res_c['est_online_flops_per_shot']:.2e}")
        except Exception as e:
            if verbose:
                print(f"    ERROR: {e}")
            all_results.append({
                "method": "our_mps", "code": code_name,
                "N": int(code.N), "K": int(code.K),
                "status": "error", "error": str(e),
            })

    return all_results


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'='*100}")
    print(f"{'Code':<10} {'Method':<18} {'LER':>6} {'Total ms':>10} "
          f"{'Per shot':>10} {'Peak mem':>12} {'Est FLOPs':>14} {'Est peak':>12}")
    print(f"{'':<10} {'':<18} {'':<6} {'':<10} "
          f"{'(ms)':>10} {'(KB)':>12} {'(/shot)':>14} {'(log2)':>12}")
    print("-" * 100)

    for r in results:
        code = r.get("code", "?")
        method = r.get("method", "?")
        status = r.get("status", "ok")

        if status != "ok" and "logical_error_rate" not in r:
            ler_s = "N/A" if status == "skipped" else "ERR"
            print(f"{code:<10} {method:<18} {ler_s:>6} "
                  f"{'---':>10} {'---':>10} {'---':>12} {'---':>14} {'---':>12}")
            continue

        ler = r.get("logical_error_rate", -1)
        ler_s = f"{ler:.4f}" if ler >= 0 else "N/A"

        if method == "our_mps":
            total_ms = r.get("total_ms", 0)
            per_shot = r.get("online_per_shot_ms", 0)
            peak_mem = r.get("online_peak_memory_bytes", 0)
            est_flops = r.get("est_online_flops_per_shot", 0)
            est_peak_log2 = "N/A"
        else:
            total_ms = r.get("actual_total_ms", 0)
            per_shot = r.get("actual_per_shot_ms", 0)
            peak_mem = r.get("actual_peak_memory_bytes", 0)
            est_flops = r.get("est_flops_per_shot", 0)
            est_peak_log2 = str(r.get("est_peak_size_log2", "N/A"))

        print(f"{code:<10} {method:<18} {ler_s:>6} {total_ms:>10.1f} "
              f"{per_shot:>10.3f} {peak_mem/1024:>11.1f}K "
              f"{est_flops:>14.2e} {est_peak_log2:>12}")


# ---------------------------------------------------------------------------
# Pytest
# ---------------------------------------------------------------------------
def test_ablation_smoke():
    """Smoke test: run ablation on smallest codes with few shots."""
    results = run_ablation(
        codes=CODES_SMOKE,
        shots=8, noise_p=0.01, chi=8, seed=42,
        verbose=True, skip_naive_above=200)
    assert len(results) > 0
    ok_count = sum(1 for r in results
                   if r.get("logical_error_rate") is not None)
    assert ok_count > 0, f"No method succeeded: {results}"
    print_comparison_table(results)
    print(f"\nAblation smoke test: {ok_count} results with LER")


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Ablation Study: Decomposition Method Comparison")
    print("=" * 70)

    results = run_ablation(
        codes=CODES,
        shots=32, noise_p=0.01, chi=32, seed=2026,
        verbose=True, skip_naive_above=200)

    print_comparison_table(results)

    # Save results
    out_dir = WORKSPACE / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_decomposition_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
