#!/usr/bin/env python3
"""Task 2: Multi-decoder benchmark on BB and HP codes.

Compares decoders:
  - MPS compressed TN decoder (CPU / GPU)
  - Exact TN decoder (CPU / GPU, small codes only)
  - BP-OSD  (ldpc package, CPU)
  - BP-LSD  (ldpc package, CPU)
  - BP      (ldpc package, CPU)

Codes:
  - ISCA revision 9-code suite:
    TB_25_3_4, TB_30_6_4, TB_48_4_8,
    BB_18_4_4, BB_60_8_4, BB_72_12_6,
    BB_90, BB_108, BB_144

Usage:
  .venv-mps/bin/python experiments/tests/test_multi_decoder_benchmark.py
  .venv-mps/bin/python -m pytest experiments/tests/test_multi_decoder_benchmark.py -v -s
"""
from __future__ import annotations

import concurrent.futures as cf
import importlib
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
# NOTE: Add WORKSPACE first (for experiments.codes etc), but defer QEC_PY
# until AFTER cudaq_qec import so the installed package (with compiled C
# extensions) takes precedence over the local source tree.
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

# ---------------------------------------------------------------------------
# GPU detection
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

# Now add local source path for tensor_network_utils (pure Python modules).
if str(QEC_PY) not in sys.path:
    sys.path.insert(0, str(QEC_PY))

# ---------------------------------------------------------------------------
# Core module import (always available, no cudaq dependency)
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

# ---------------------------------------------------------------------------
# BP decoders (ldpc package)
# ---------------------------------------------------------------------------
try:
    from ldpc import BpOsdDecoder, BpDecoder, BpLsdDecoder
    HAS_LDPC = True
except ImportError:
    HAS_LDPC = False
    BpOsdDecoder = BpDecoder = BpLsdDecoder = None

# ---------------------------------------------------------------------------
# Helper: build TN and compile MPS chain (CPU path, no cudaq)
# ---------------------------------------------------------------------------
from quimb import oset
from quimb.tensor import Tensor, TensorNetwork


def _build_tn_and_compile(
    H: np.ndarray,
    logical_row: np.ndarray,
    noise_p: float,
    chi: int = 32,
    verbose: bool = False,
) -> tuple[Any, Any, Any]:
    """Build parameterised TN and compile to 1D chain.

    Returns (chain, compressed_tn, offline_stats).
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
    logical_tn = factory.tensor_network_from_parity_check(
        logical_2d, col_inds=error_inds, row_inds=logical_inds, tags=["LOG_0"])
    logical_tn = logical_tn.combine(
        factory.tensor_network_from_logical_observable(
            logical_2d, logical_inds, logical_obs_inds, ["LOG_0"]),
        virtual=True)

    param_syn_tn = core.make_parametrized_syndrome_network(check_inds, syn_param_inds)

    # Noise model: independent depolarising
    noise_tensors = []
    for i, eind in enumerate(error_inds):
        noise_tensors.append(Tensor(
            data=np.array([1.0 - noise_p, noise_p], dtype=np.float64),
            inds=(eind,), tags=oset([f"NOISE_{i}", "NOISE"])))
    noise_tn = TensorNetwork(noise_tensors)

    full_tn = TensorNetwork()
    full_tn = full_tn.combine(code_tn, virtual=True)
    full_tn = full_tn.combine(logical_tn, virtual=True)
    full_tn = full_tn.combine(param_syn_tn, virtual=True)
    full_tn = full_tn.combine(noise_tn, virtual=True)

    preserve_inds = set(syn_param_inds + logical_obs_inds)
    result = core.compile_to_1d_chain(
        tn=full_tn.copy(),
        preserve_inds=preserve_inds,
        syndrome_inds=set(syn_param_inds),
        logical_inds=set(logical_obs_inds),
        chi=chi,
        verbose=verbose,
    )
    return result.chain, result.compressed_tn, result.offline_stats


def _decode_mps_cpu(
    chain, syndrome: np.ndarray, num_checks: int
) -> float:
    """Online decode single syndrome via MPS chain (CPU)."""
    syn_vals = {f"syn_param_{i}": float(syndrome[i]) for i in range(num_checks)}
    assignments, marginals = chain.decode_logicals(syndrome_values=syn_vals)
    return float(marginals.get("obs", 0.5))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def _sample_errors_and_syndromes(
    H: np.ndarray, noise_p: float, shots: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = H.shape[1]
    errors = (rng.random((shots, n)) < noise_p).astype(np.int8)
    syndromes = (errors @ H.T.astype(np.int8)) % 2
    return errors, syndromes


def _logical_error_rate(
    errors: np.ndarray, logical_row: np.ndarray, predictions: np.ndarray
) -> float:
    """Compute logical error rate from hard decisions."""
    true_logicals = (errors @ logical_row.reshape(-1).astype(np.int8)) % 2
    return float(np.mean(true_logicals != predictions))


# ---------------------------------------------------------------------------
# Code catalogue
# ---------------------------------------------------------------------------
@dataclass
class CodeCase:
    name: str
    code: Any
    skip_exact: bool = False  # skip exact TN for large codes


_EXACT_MAX_N = 72


def _get_codes(
    *,
    code_names: Optional[list[str]] = None,
    skip_missing: bool = False,
    verbose: bool = False,
) -> list[CodeCase]:
    loaded, skipped = load_isca_code_cases(
        code_names=code_names,
        skip_missing=skip_missing,
    )
    if verbose and skipped:
        print(f"[codes] skipped {len(skipped)} code(s):")
        for line in skipped:
            print(f"  - {line}")
    codes: list[CodeCase] = []
    for name, code in loaded:
        skip_exact = int(code.N) > _EXACT_MAX_N
        codes.append(CodeCase(name=name, code=code, skip_exact=skip_exact))
    return codes


def _run_single_code_case(
    cc: CodeCase,
    *,
    shots: int,
    noise_p: float,
    chi: int,
    seed: int,
    verbose: bool,
    use_gpu: bool,
) -> dict[str, Any]:
    H = np.ascontiguousarray(cc.code.hz.astype(np.float64))
    lz = np.ascontiguousarray(cc.code.lz.astype(np.float64))
    logical_row = np.ascontiguousarray(lz[0])  # first logical
    num_checks, num_errors = H.shape

    if verbose:
        print(f"\n{'='*60}")
        print(f"Code: {cc.name}  N={cc.code.N} K={cc.code.K} D={cc.code.D}")
        print(f"  H shape: {H.shape}, logical rows: {lz.shape[0]}")
        print(f"  noise_p={noise_p}, shots={shots}, chi={chi}")

    errors, syndromes = _sample_errors_and_syndromes(H, noise_p, shots, seed)

    record: dict[str, Any] = {
        "code": cc.name,
        "N": int(cc.code.N),
        "K": int(cc.code.K),
        "D": float(cc.code.D),
        "noise_p": noise_p,
        "shots": shots,
        "chi": chi,
        "decoders": {},
    }

    # --- MPS decoder (CPU or GPU) ---
    try:
        t0 = time.perf_counter()
        if use_gpu:
            # GPU path via cudaq_qec
            dec = qec.get_decoder(
                "tensor_network_mps_decoder", H,
                logical_obs=logical_row.reshape(1, -1),
                noise_model=[noise_p] * num_errors,
                dtype="float32", device="cuda",
                bond_dim=chi, verbose=verbose)
            init_ms = (time.perf_counter() - t0) * 1e3
            probs = []
            t1 = time.perf_counter()
            for i in range(shots):
                res = dec.decode([float(x) for x in syndromes[i]])
                probs.append(float(res.result[0]))
            decode_ms = (time.perf_counter() - t1) * 1e3
            backend = "gpu"
        else:
            # CPU path via core module
            chain, compressed_tn, _stats = _build_tn_and_compile(
                H, logical_row, noise_p, chi=chi)
            init_ms = (time.perf_counter() - t0) * 1e3
            if chain is not None:
                probs = []
                t1 = time.perf_counter()
                for i in range(shots):
                    probs.append(_decode_mps_cpu(chain, syndromes[i], num_checks))
                decode_ms = (time.perf_counter() - t1) * 1e3
            elif compressed_tn is not None:
                # Use CompressedTNDecoder when chain extraction fails
                probs = []
                t1 = time.perf_counter()
                for i in range(shots):
                    syn_vals = {f"syn_param_{j}": float(syndromes[i, j])
                                for j in range(num_checks)}
                    _, marginals = compressed_tn.decode_logicals(syndrome_values=syn_vals)
                    probs.append(float(marginals.get("obs", 0.5)))
                decode_ms = (time.perf_counter() - t1) * 1e3
            backend = "cpu"

        predictions = (np.array(probs) >= 0.5).astype(np.int8)
        ler = _logical_error_rate(errors, logical_row, predictions)
        record["decoders"]["mps_tn"] = {
            "status": "ok", "backend": backend,
            "init_ms": round(init_ms, 2),
            "total_decode_ms": round(decode_ms, 2),
            "per_shot_ms": round(decode_ms / shots, 3),
            "logical_error_rate": round(ler, 6),
        }
        if verbose:
            print(f"  MPS-TN ({backend}): LER={ler:.4f}, {decode_ms:.1f}ms total")
    except Exception as e:
        record["decoders"]["mps_tn"] = {"status": "error", "error": str(e)}
        if verbose:
            print(f"  MPS-TN: ERROR {e}")

    # --- Exact TN decoder (small codes only) ---
    if not cc.skip_exact:
        try:
            t0 = time.perf_counter()
            if use_gpu:
                dec = qec.get_decoder(
                    "tensor_network_decoder", H,
                    logical_obs=logical_row.reshape(1, -1),
                    noise_model=[noise_p] * num_errors,
                    dtype="float32", device="cuda")
                init_ms = (time.perf_counter() - t0) * 1e3
                probs = []
                t1 = time.perf_counter()
                for i in range(shots):
                    res = dec.decode([float(x) for x in syndromes[i]])
                    probs.append(float(res.result[0]))
                decode_ms = (time.perf_counter() - t1) * 1e3
                backend = "gpu"
            else:
                # CPU exact via quimb contraction
                check_inds = [f"s_{i}" for i in range(num_checks)]
                error_inds = [f"e_{i}" for i in range(num_errors)]
                logical_inds = ["l_0"]
                logical_obs_inds = ["obs"]

                code_tn = factory.tensor_network_from_parity_check(
                    H, col_inds=error_inds, row_inds=check_inds)
                logical_2d = logical_row.reshape(1, -1)
                log_tn = factory.tensor_network_from_parity_check(
                    logical_2d, col_inds=error_inds, row_inds=logical_inds,
                    tags=["LOG_0"])
                log_tn = log_tn.combine(
                    factory.tensor_network_from_logical_observable(
                        logical_2d, logical_inds, logical_obs_inds, ["LOG_0"]),
                    virtual=True)
                noise_ts = [Tensor(
                    data=np.array([1.0 - noise_p, noise_p]),
                    inds=(error_inds[j],), tags=oset([f"NOISE_{j}"]))
                    for j in range(num_errors)]
                noise_tn = TensorNetwork(noise_ts)

                init_ms = (time.perf_counter() - t0) * 1e3
                probs = []
                t1 = time.perf_counter()
                for i in range(shots):
                    syn_ts2 = []
                    for j in range(num_checks):
                        s = float(syndromes[i, j])
                        syn_ts2.append(Tensor(
                            data=s * np.array([1.0, -1.0]) + (1.0 - s) * np.array([1.0, 1.0]),
                            inds=(check_inds[j],), tags=oset([f"SYN_{j}"])))
                    syn_tn = TensorNetwork(syn_ts2)

                    full = TensorNetwork()
                    full = full.combine(code_tn, virtual=True)
                    full = full.combine(log_tn, virtual=True)
                    full = full.combine(syn_tn, virtual=True)
                    full = full.combine(noise_tn, virtual=True)
                    val = full.contract(output_inds=("obs",))
                    arr = val.data if hasattr(val, 'data') else np.asarray(val)
                    arr = np.asarray(arr).ravel()
                    if len(arr) < 2:
                        p1 = 0.5
                    else:
                        total = float(arr[0] + arr[1])
                        p1 = float(arr[1] / total) if total > 0 else 0.5
                    probs.append(p1)
                decode_ms = (time.perf_counter() - t1) * 1e3
                backend = "cpu"

            predictions = (np.array(probs) >= 0.5).astype(np.int8)
            ler = _logical_error_rate(errors, logical_row, predictions)
            record["decoders"]["exact_tn"] = {
                "status": "ok", "backend": backend,
                "init_ms": round(init_ms, 2),
                "total_decode_ms": round(decode_ms, 2),
                "per_shot_ms": round(decode_ms / shots, 3),
                "logical_error_rate": round(ler, 6),
            }
            if verbose:
                print(f"  Exact-TN ({backend}): LER={ler:.4f}, {decode_ms:.1f}ms total")
        except Exception as e:
            record["decoders"]["exact_tn"] = {"status": "error", "error": str(e)}
            if verbose:
                print(f"  Exact-TN: ERROR {e}")
    else:
        record["decoders"]["exact_tn"] = {"status": "skipped", "reason": "code too large"}

    # --- BP decoders (ldpc, CPU only) ---
    if HAS_LDPC:
        H_int = H.astype(np.uint8)
        channel_probs = np.full(num_errors, noise_p)

        for dec_name, DecClass, kwargs in [
            ("bp", BpDecoder, {"max_iter": 50, "bp_method": "ms"}),
            ("bp_osd", BpOsdDecoder, {
                "max_iter": 50, "bp_method": "ms",
                "osd_method": "osd_cs", "osd_order": 10}),
            ("bp_lsd", BpLsdDecoder, {
                "max_iter": 50, "bp_method": "ms",
                "lsd_order": 0}),
        ]:
            try:
                t0 = time.perf_counter()
                bp_dec = DecClass(
                    H_int, error_rate=noise_p,
                    channel_probs=channel_probs,
                    **kwargs)
                init_ms = (time.perf_counter() - t0) * 1e3
                predictions_list = []
                t1 = time.perf_counter()
                for i in range(shots):
                    correction = bp_dec.decode(syndromes[i].astype(np.uint8))
                    logical_val = int(correction @ logical_row.astype(np.int8)) % 2
                    predictions_list.append(logical_val)
                decode_ms = (time.perf_counter() - t1) * 1e3

                preds = np.array(predictions_list, dtype=np.int8)
                true_logicals = (errors @ logical_row.reshape(-1).astype(np.int8)) % 2
                ler = float(np.mean(true_logicals != preds))

                record["decoders"][dec_name] = {
                    "status": "ok", "backend": "cpu",
                    "init_ms": round(init_ms, 2),
                    "total_decode_ms": round(decode_ms, 2),
                    "per_shot_ms": round(decode_ms / shots, 3),
                    "logical_error_rate": round(ler, 6),
                }
                if verbose:
                    print(f"  {dec_name}: LER={ler:.4f}, {decode_ms:.1f}ms total")
            except Exception as e:
                record["decoders"][dec_name] = {"status": "error", "error": str(e)}
                if verbose:
                    print(f"  {dec_name}: ERROR {e}")

    return record


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(
    shots: int = 128,
    noise_p: float = 0.01,
    chi: int = 16,
    seed: int = 42,
    code_names: Optional[list[str]] = None,
    skip_missing: bool = False,
    parallel_workers: Optional[int] = None,
    verbose: bool = False,
) -> list[dict]:
    use_gpu = HAS_GPU and HAS_CUDAQ
    codes = _get_codes(
        code_names=code_names,
        skip_missing=skip_missing,
        verbose=verbose,
    )
    if not codes:
        raise RuntimeError("No codes were loaded for benchmark.")

    workers = resolve_parallel_workers(
        use_gpu=use_gpu,
        parallel_workers=parallel_workers,
    )
    if verbose:
        print(f"[benchmark] codes={len(codes)}, workers={workers}, use_gpu={use_gpu}")

    if workers == 1 or len(codes) == 1:
        return [
            _run_single_code_case(
                cc,
                shots=shots,
                noise_p=noise_p,
                chi=chi,
                seed=seed,
                verbose=verbose,
                use_gpu=use_gpu,
            ) for cc in codes
        ]

    ordered: list[Optional[dict[str, Any]]] = [None] * len(codes)
    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _run_single_code_case,
                cc,
                shots=shots,
                noise_p=noise_p,
                chi=chi,
                seed=seed,
                verbose=verbose,
                use_gpu=use_gpu,
            ): idx for idx, cc in enumerate(codes)
        }
        for future in cf.as_completed(future_map):
            idx = future_map[future]
            ordered[idx] = future.result()
    return [result for result in ordered if result is not None]


# ---------------------------------------------------------------------------
# Pytest interface
# ---------------------------------------------------------------------------
def test_multi_decoder_benchmark():
    """Smoke test: run benchmark with small shots."""
    results = run_benchmark(
        shots=8,
        noise_p=0.01,
        chi=8,
        seed=42,
        code_names=["BB_90", "BB_108"],
        skip_missing=True,
        parallel_workers=2,
        verbose=True,
    )
    assert len(results) > 0
    for r in results:
        assert r["code"] is not None
        # At least one decoder must succeed
        ok_count = sum(1 for d in r["decoders"].values()
                       if isinstance(d, dict) and d.get("status") == "ok")
        assert ok_count > 0, f"No decoder succeeded for {r['code']}: {r['decoders']}"
    print(f"\nBenchmark complete: {len(results)} codes tested")


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"GPU available: {HAS_GPU}")
    print(f"cudaq_qec available: {HAS_CUDAQ}")
    print(f"ldpc available: {HAS_LDPC}")
    print()

    results = run_benchmark(
        shots=20000,
        noise_p=0.01,
        chi=16,
        seed=2026,
        skip_missing=False,
        parallel_workers=None,
        verbose=True,
    )

    out_dir = WORKSPACE / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multi_decoder_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
