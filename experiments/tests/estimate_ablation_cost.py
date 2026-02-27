#!/usr/bin/env python3
"""Estimate contraction cost for ablation study WITHOUT actually contracting.

Uses cotengra/quimb to estimate FLOPs, peak memory, contraction width
for all 3 methods across all codes. Very lightweight - no actual computation.

Usage:
  PYTHONPATH=cudaqx/libs/qec/python:. python3 experiments/tests/estimate_ablation_cost.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cotengra")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np

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
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_TNU = QEC_PY / "cudaq_qec" / "plugins" / "decoders" / "tensor_network_utils"
core = _load_module("mps_decoder_core", _TNU / "mps_decoder_core.py")
factory = _load_module("tensor_network_factory", _TNU / "tensor_network_factory.py")

from experiments.tests._isca_code_registry import load_isca_code_cases
from quimb import oset
from quimb.tensor import Tensor, TensorNetwork

# ---------------------------------------------------------------------------
CODES_DEFAULT = [
    "LDPC_25_3_4",
    "LDPC_30_6_4",
    "TOR_50_2_5",
    "QCC_18_4_4",
    "QCC_60_8_4",
    "QCC_72_12_6",
    "QCC_90_8_10",
    "QCC_108_8_10",
    "QCC_144_12_12",
]

NOISE_P = 0.01


# ---------------------------------------------------------------------------
# Build TNs (same as in test_ablation_decomposition.py)
# ---------------------------------------------------------------------------
def _build_concrete_tn(
    H: np.ndarray, logical_row: np.ndarray, noise_p: float,
    syndrome: np.ndarray,
) -> tuple[TensorNetwork, list[str]]:
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


def _build_full_tn(
    H: np.ndarray, logical_row: np.ndarray, noise_p: float,
) -> tuple[TensorNetwork, list[str], list[str], list[str]]:
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


def _tn_storage_bytes(tn: TensorNetwork) -> int:
    total = 0
    for t in tn.tensors:
        total += t.data.nbytes
    return total


def _tn_total_elements(tn: TensorNetwork) -> int:
    total = 0
    for t in tn.tensors:
        total += t.data.size
    return total


def _estimate_contraction(tn: TensorNetwork, output_inds: tuple,
                          label: str = "") -> dict[str, Any]:
    """Estimate contraction cost using cotengra WITHOUT actually contracting."""
    num_tensors = len(list(tn.tensors))
    storage_bytes = _tn_storage_bytes(tn)
    total_elements = _tn_total_elements(tn)

    # Get max tensor rank/size in the network
    max_tensor_size = max(t.data.size for t in tn.tensors)
    max_tensor_rank = max(len(t.inds) for t in tn.tensors)

    t0 = time.perf_counter()
    try:
        import cotengra
        # Use greedy optimizer - much faster than HyperOptimizer for estimation
        tree = tn.contraction_tree(output_inds=output_inds, optimize='greedy')
        est_flops = tree.total_flops()
        est_max_size = tree.max_size()
        est_write = tree.total_write()
        contraction_width = tree.contraction_width()
    except Exception as e:
        print(f"    [{label}] greedy failed: {e}, falling back to auto")
        try:
            info = tn.contraction_info(output_inds=output_inds, optimize='auto')
            est_flops = info.opt_cost
            est_max_size = info.largest_intermediate
            est_write = 0
            contraction_width = 0
        except Exception as e2:
            print(f"    [{label}] auto also failed: {e2}")
            return {
                "label": label, "status": "error", "error": str(e2),
                "num_tensors": num_tensors,
                "storage_bytes": storage_bytes,
            }
    estimate_ms = (time.perf_counter() - t0) * 1e3

    est_peak_log2 = math.log2(est_max_size) if est_max_size > 0 else 0
    est_peak_mem_bytes = est_max_size * 8  # float64

    return {
        "label": label,
        "status": "ok",
        "num_tensors": num_tensors,
        "storage_bytes": storage_bytes,
        "total_elements": total_elements,
        "max_tensor_size": max_tensor_size,
        "max_tensor_rank": max_tensor_rank,
        "est_flops": float(est_flops),
        "est_flops_log10": round(math.log10(est_flops), 2) if est_flops > 0 else 0,
        "est_peak_size_elements": int(est_max_size),
        "est_peak_size_log2": round(est_peak_log2, 2),
        "est_peak_memory_bytes": int(est_peak_mem_bytes),
        "est_peak_memory_MB": round(est_peak_mem_bytes / 1e6, 2),
        "est_peak_memory_GB": round(est_peak_mem_bytes / 1e9, 4),
        "est_total_write_elements": int(est_write) if est_write else 0,
        "contraction_width": round(contraction_width, 2) if contraction_width else 0,
        "estimate_time_ms": round(estimate_ms, 1),
    }


# ---------------------------------------------------------------------------
# Main estimation
# ---------------------------------------------------------------------------
def run_estimation(codes: list[str] | None = None, chi: int = 32):
    if codes is None:
        codes = CODES_DEFAULT

    loaded_codes, skipped = load_isca_code_cases(
        code_names=codes, skip_missing=True)
    if skipped:
        print(f"Skipped: {skipped}")

    all_results = []

    for code_name, code in loaded_codes:
        H = code.hz.astype(np.float64)
        logical_row = code.lz[0].astype(np.float64)
        num_checks, num_errors = H.shape

        # Dummy syndrome for concrete TN
        syndrome = np.zeros(num_checks, dtype=np.float64)

        print(f"\n{'='*70}")
        print(f"Code: {code_name}  N={code.N} K={code.K}  H=({num_checks}x{num_errors})")

        code_result: dict[str, Any] = {
            "code": code_name,
            "N": int(code.N),
            "K": int(code.K),
            "num_checks": num_checks,
            "num_errors": num_errors,
        }

        # --- Method A: Naive (concrete TN, no decomposition) ---
        print(f"\n  Method A: Naive direct contraction")
        tn_naive, obs_inds = _build_concrete_tn(H, logical_row, NOISE_P, syndrome)
        est_a = _estimate_contraction(tn_naive, tuple(obs_inds), label="naive")
        print(f"    tensors={est_a.get('num_tensors')}, "
              f"storage={est_a.get('storage_bytes')}B")
        print(f"    est_flops={est_a.get('est_flops', 0):.2e} "
              f"(10^{est_a.get('est_flops_log10', '?')}), "
              f"peak_mem={est_a.get('est_peak_memory_MB', '?')}MB "
              f"({est_a.get('est_peak_memory_GB', '?')}GB), "
              f"peak_log2={est_a.get('est_peak_size_log2', '?')}, "
              f"width={est_a.get('contraction_width', '?')}")
        code_result["naive"] = est_a

        # --- Method B: Quimb compress_all (compress then estimate) ---
        print(f"\n  Method B: Quimb compress_all (chi={chi})")
        t0 = time.perf_counter()
        tn_compressed = tn_naive.compress_all(
            max_bond=chi, cutoff=1e-12, inplace=False)
        compress_ms = (time.perf_counter() - t0) * 1e3
        est_b = _estimate_contraction(tn_compressed, tuple(obs_inds), label="quimb")
        est_b["compress_ms"] = round(compress_ms, 2)
        print(f"    compress={compress_ms:.1f}ms, "
              f"tensors={est_b.get('num_tensors')}, "
              f"storage={est_b.get('storage_bytes')}B")
        print(f"    est_flops={est_b.get('est_flops', 0):.2e} "
              f"(10^{est_b.get('est_flops_log10', '?')}), "
              f"peak_mem={est_b.get('est_peak_memory_MB', '?')}MB "
              f"({est_b.get('est_peak_memory_GB', '?')}GB), "
              f"peak_log2={est_b.get('est_peak_size_log2', '?')}, "
              f"width={est_b.get('contraction_width', '?')}")
        code_result["quimb_compress"] = est_b

        # --- Method C: Our MPS (estimate online cost analytically) ---
        print(f"\n  Method C: Our MPS decomposition (chi={chi})")
        # For MPS: online cost is O(M * chi^2 * 2) per shot
        # M = number of sites in the chain ~ num_checks
        mps_online_flops = num_checks * chi * chi * 2
        mps_online_memory = chi * chi * 8 * 2  # two chi x chi matrices in memory
        # Offline: build the parameterized TN and estimate its size
        full_tn, _, syn_param_inds, logical_obs_inds = _build_full_tn(
            H, logical_row, NOISE_P)
        full_storage = _tn_storage_bytes(full_tn)
        full_tensors = len(list(full_tn.tensors))
        full_elements = _tn_total_elements(full_tn)

        # After MPS compression, the chain has ~num_checks tensors of size chi x 2 x chi
        mps_tensor_size = chi * 2 * chi * 8  # bytes per tensor
        mps_total_storage = num_checks * mps_tensor_size

        est_c = {
            "label": "our_mps",
            "status": "ok",
            "pre_num_tensors": full_tensors,
            "pre_storage_bytes": full_storage,
            "pre_total_elements": full_elements,
            "chi": chi,
            "est_online_flops_per_shot": mps_online_flops,
            "est_online_flops_log10": round(math.log10(mps_online_flops), 2) if mps_online_flops > 0 else 0,
            "est_online_peak_memory_bytes": mps_online_memory,
            "est_online_peak_memory_MB": round(mps_online_memory / 1e6, 4),
            "est_mps_storage_bytes": mps_total_storage,
            "est_mps_storage_MB": round(mps_total_storage / 1e6, 4),
            "est_mps_num_tensors": num_checks,
            "est_mps_tensor_shape": f"({chi}, 2, {chi})",
        }
        print(f"    pre-TN: tensors={full_tensors}, storage={full_storage}B")
        print(f"    MPS chain: {num_checks} tensors of shape ({chi},2,{chi}), "
              f"total={mps_total_storage/1e6:.2f}MB")
        print(f"    online_flops/shot={mps_online_flops:.2e} "
              f"(10^{est_c['est_online_flops_log10']}), "
              f"online_peak_mem={mps_online_memory/1e6:.4f}MB")
        code_result["our_mps"] = est_c

        all_results.append(code_result)

    return all_results


def print_summary_table(results: list[dict]):
    print(f"\n{'='*130}")
    print(f"{'Code':<16} {'N':>4} {'K':>3} | "
          f"{'Naive FLOPs':>14} {'Peak(GB)':>10} {'log2':>6} | "
          f"{'Quimb FLOPs':>14} {'Peak(GB)':>10} {'log2':>6} | "
          f"{'MPS FLOPs/shot':>14} {'MPS stor(MB)':>12}")
    print("-" * 130)

    for r in results:
        code = r["code"]
        N = r["N"]
        K = r["K"]

        na = r.get("naive", {})
        nb = r.get("quimb_compress", {})
        nc = r.get("our_mps", {})

        a_flops = f"{na.get('est_flops', 0):.2e}" if na.get("status") == "ok" else "ERR"
        a_peak = f"{na.get('est_peak_memory_GB', 0):.4f}" if na.get("status") == "ok" else "ERR"
        a_log2 = f"{na.get('est_peak_size_log2', 0):.1f}" if na.get("status") == "ok" else "ERR"

        b_flops = f"{nb.get('est_flops', 0):.2e}" if nb.get("status") == "ok" else "ERR"
        b_peak = f"{nb.get('est_peak_memory_GB', 0):.4f}" if nb.get("status") == "ok" else "ERR"
        b_log2 = f"{nb.get('est_peak_size_log2', 0):.1f}" if nb.get("status") == "ok" else "ERR"

        c_flops = f"{nc.get('est_online_flops_per_shot', 0):.2e}"
        c_stor = f"{nc.get('est_mps_storage_MB', 0):.4f}"

        print(f"{code:<16} {N:>4} {K:>3} | "
              f"{a_flops:>14} {a_peak:>10} {a_log2:>6} | "
              f"{b_flops:>14} {b_peak:>10} {b_log2:>6} | "
              f"{c_flops:>14} {c_stor:>12}")

    # Highlight the scaling
    print(f"\n{'='*130}")
    print("Notes:")
    print("  - Naive/Quimb: FLOPs and Peak Memory are per-shot contraction estimates (cotengra)")
    print("  - MPS: FLOPs/shot is the online decode cost; MPS stor is the compressed chain size")
    print("  - Peak(GB) = est_peak_size_elements * 8 bytes (float64)")
    print("  - log2 = log2(peak intermediate tensor size in elements)")


if __name__ == "__main__":
    print("Ablation Cost Estimation (no actual contraction)")
    print("=" * 70)

    results = run_estimation(chi=32)
    print_summary_table(results)

    # Save
    out_dir = WORKSPACE / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_cost_estimates.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
