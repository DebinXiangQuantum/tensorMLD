#!/usr/bin/env python3
"""Task 5: GPU decode time, memory usage, and utilization metrics.

Profiles:
  - Online decode latency (cupy events on GPU, perf_counter on CPU)
  - GPU memory usage via pynvml (peak / avg during decode)
  - GPU core utilization via pynvml
  - MPS chain save/load + re-decode timing

Usage:
  .venv-mps/bin/python experiments/tests/test_gpu_metrics.py
  .venv-mps/bin/python -m pytest experiments/tests/test_gpu_metrics.py -v -s
"""
from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import json
import sys
import tempfile
import threading
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
# GPU / library detection
# ---------------------------------------------------------------------------
HAS_GPU = False
try:
    import cupy
    HAS_GPU = cupy.cuda.is_available()
except Exception:
    cupy = None

HAS_CUDAQ = False
try:
    import cudaq_qec as qec
    HAS_CUDAQ = True
except Exception:
    qec = None  # type: ignore

HAS_PYNVML = False
try:
    import pynvml
    HAS_PYNVML = True
except Exception:
    pynvml = None

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
# GPU profiler (pynvml-based, same design as run_decoder_comparison.py)
# ---------------------------------------------------------------------------
class GPUProfiler:
    """Background thread sampling GPU memory and utilization via pynvml."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.02):
        self.device_index = device_index
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not HAS_PYNVML:
            return
        self.samples.clear()
        self._stop.clear()

        def _loop():
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            except Exception:
                return
            while not self._stop.is_set():
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.samples.append({
                        "ts": time.perf_counter(),
                        "mem_used_mb": float(mem.used) / (1024 * 1024),
                        "mem_total_mb": float(mem.total) / (1024 * 1024),
                        "gpu_util": float(util.gpu),
                        "mem_util": float(util.memory),
                    })
                except Exception:
                    pass
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        if not HAS_PYNVML or self._thread is None:
            return {"enabled": False, "sample_count": 0}
        self._stop.set()
        self._thread.join(timeout=2.0)
        if not self.samples:
            return {"enabled": True, "sample_count": 0}
        mems = [s["mem_used_mb"] for s in self.samples]
        utils = [s["gpu_util"] for s in self.samples]
        mem_utils = [s["mem_util"] for s in self.samples]
        return {
            "enabled": True,
            "sample_count": len(self.samples),
            "mem_used_peak_mb": round(max(mems), 2),
            "mem_used_avg_mb": round(sum(mems) / len(mems), 2),
            "mem_total_mb": round(self.samples[0]["mem_total_mb"], 2),
            "gpu_util_peak": round(max(utils), 1),
            "gpu_util_avg": round(sum(utils) / len(utils), 1),
            "mem_util_peak": round(max(mem_utils), 1),
            "mem_util_avg": round(sum(mem_utils) / len(mem_utils), 1),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_and_compile_cpu(H, logical_row, noise_p, chi):
    """Build TN + compile to chain, CPU path."""
    num_checks, num_errors = H.shape
    check_inds = [f"s_{i}" for i in range(num_checks)]
    error_inds = [f"e_{i}" for i in range(num_errors)]
    syn_param_inds = [f"syn_param_{i}" for i in range(num_checks)]
    logical_obs_inds = ["obs"]

    code_tn = factory.tensor_network_from_parity_check(
        H.astype(np.float64), col_inds=error_inds, row_inds=check_inds)
    logical_2d = logical_row.reshape(1, -1).astype(np.float64)
    log_tn = factory.tensor_network_from_parity_check(
        logical_2d, col_inds=error_inds, row_inds=["l_0"], tags=["LOG_0"])
    log_tn = log_tn.combine(
        factory.tensor_network_from_logical_observable(
            logical_2d, ["l_0"], logical_obs_inds, ["LOG_0"]),
        virtual=True)
    param_syn = core.make_parametrized_syndrome_network(check_inds, syn_param_inds)
    noise_ts = [Tensor(
        data=np.array([1.0 - noise_p, noise_p]),
        inds=(error_inds[j],), tags=oset([f"NOISE_{j}"]))
        for j in range(num_errors)]
    noise_tn = TensorNetwork(noise_ts)

    full_tn = TensorNetwork()
    full_tn = full_tn.combine(code_tn, virtual=True)
    full_tn = full_tn.combine(log_tn, virtual=True)
    full_tn = full_tn.combine(param_syn, virtual=True)
    full_tn = full_tn.combine(noise_tn, virtual=True)

    preserve = set(syn_param_inds + logical_obs_inds)
    result = core.compile_to_1d_chain(
        tn=full_tn.copy(), preserve_inds=preserve,
        syndrome_inds=set(syn_param_inds),
        logical_inds=set(logical_obs_inds), chi=chi)
    return result.chain, result.compressed_tn, result.offline_stats


def _sample(H, noise_p, shots, seed):
    rng = np.random.default_rng(seed)
    n = H.shape[1]
    errors = (rng.random((shots, n)) < noise_p).astype(np.int8)
    syndromes = (errors @ H.T.astype(np.int8)) % 2
    return errors, syndromes.astype(np.float64)


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------
def _measure_latency_cpu(chain, syndromes, num_checks, warmup=5, repeats=20):
    """Measure per-shot decode latency on CPU."""
    syn_vals = {f"syn_param_{j}": float(syndromes[0, j]) for j in range(num_checks)}
    for _ in range(warmup):
        chain.decode_logicals(syndrome_values=syn_vals)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        chain.decode_logicals(syndrome_values=syn_vals)
        times.append((time.perf_counter() - t0) * 1e3)
    arr = np.array(times)
    return {
        "method": "perf_counter",
        "warmup": warmup,
        "repeats": repeats,
        "avg_ms": round(float(arr.mean()), 4),
        "min_ms": round(float(arr.min()), 4),
        "max_ms": round(float(arr.max()), 4),
        "std_ms": round(float(arr.std()), 4),
    }


def _measure_latency_gpu(dec, syndromes, warmup=10, repeats=50):
    """Measure per-shot decode latency on GPU with cupy events."""
    syndrome = [float(x) for x in syndromes[0]]
    for _ in range(warmup):
        dec.decode(syndrome)

    if cupy is not None and HAS_GPU:
        # GPU timing via cupy events
        times = []
        for _ in range(repeats):
            start = cupy.cuda.Event()
            end = cupy.cuda.Event()
            start.record()
            dec.decode(syndrome)
            end.record()
            end.synchronize()
            times.append(float(cupy.cuda.get_elapsed_time(start, end)))
        arr = np.array(times)
        return {
            "method": "cupy_event",
            "warmup": warmup,
            "repeats": repeats,
            "avg_ms": round(float(arr.mean()), 4),
            "min_ms": round(float(arr.min()), 4),
            "max_ms": round(float(arr.max()), 4),
            "std_ms": round(float(arr.std()), 4),
        }
    else:
        # Fallback to perf_counter
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            dec.decode(syndrome)
            times.append((time.perf_counter() - t0) * 1e3)
        arr = np.array(times)
        return {
            "method": "perf_counter",
            "warmup": warmup,
            "repeats": repeats,
            "avg_ms": round(float(arr.mean()), 4),
            "min_ms": round(float(arr.min()), 4),
            "max_ms": round(float(arr.max()), 4),
            "std_ms": round(float(arr.std()), 4),
        }


# ---------------------------------------------------------------------------
# Main profiling routine
# ---------------------------------------------------------------------------
def _profile_single_code(
    code_name: str,
    code: Any,
    *,
    chi: int = 16,
    shots: int,
    noise_p: float,
    seed: int,
    warmup: int,
    repeats: int,
    verbose: bool,
    use_gpu: bool,
) -> dict[str, Any]:
    H = code.hz.astype(np.float64)
    logical_row = code.lz[0].astype(np.float64)
    num_checks, num_errors = H.shape

    errors, syndromes = _sample(H, noise_p, shots, seed)
    true_logicals = (errors @ logical_row.reshape(-1).astype(np.int8)) % 2

    record: dict[str, Any] = {
        "code": code_name,
        "N": int(code.N),
        "K": int(code.K),
        "chi": chi,
        "noise_p": noise_p,
        "shots": shots,
        "backend": "gpu" if use_gpu else "cpu",
        "gpu_available": HAS_GPU,
        "pynvml_available": HAS_PYNVML,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Code: {code_name}  N={code.N}  backend={'GPU' if use_gpu else 'CPU'}")

    # --- Offline compilation + profiling ---
    profiler = GPUProfiler()
    profiler.start()
    t0 = time.perf_counter()

    if use_gpu:
        dec = qec.get_decoder(
            "tensor_network_mps_decoder", H,
            logical_obs=logical_row.reshape(1, -1),
            noise_model=[noise_p] * num_errors,
            dtype="float32", device="cuda",
            bond_dim=chi, verbose=verbose)
        chain = getattr(dec, "_compressed_chain", None)
        offline_stats = getattr(dec, "offline_stats", None)
        compressed_tn_decoder = None
    else:
        chain, compressed_tn_decoder, offline_stats = _build_and_compile_cpu(
            H, logical_row, noise_p, chi)
        dec = None

    compile_ms = (time.perf_counter() - t0) * 1e3
    compile_profile = profiler.stop()

    record["compile_ms"] = round(compile_ms, 2)
    record["compile_gpu_profile"] = compile_profile
    if offline_stats is not None:
        record["offline_stats"] = {
            "truncation_error": float(offline_stats.truncation_error),
            "max_bond_dim": int(offline_stats.max_bond_dim),
            "steps": int(offline_stats.steps),
            "initial_nodes": int(offline_stats.initial_nodes),
            "initial_edges": int(offline_stats.initial_edges),
        }
    if verbose:
        print(f"  Compile: {compile_ms:.1f}ms")
        print(f"  Compile GPU profile: {compile_profile}")

    # --- Online decode profiling ---
    has_chain = chain is not None
    has_compressed_tn = ((not use_gpu) and (not has_chain) and
                         (compressed_tn_decoder is not None))
    profiler2 = GPUProfiler()
    profiler2.start()

    if use_gpu:
        probs = np.empty(shots)
        t1 = time.perf_counter()
        for i in range(shots):
            res = dec.decode([float(x) for x in syndromes[i]])
            probs[i] = float(res.result[0])
        total_decode_ms = (time.perf_counter() - t1) * 1e3
        record["decode_method"] = "gpu"
    elif has_chain:
        probs = np.empty(shots)
        t1 = time.perf_counter()
        for i in range(shots):
            syn_vals = {f"syn_param_{j}": float(syndromes[i, j])
                        for j in range(num_checks)}
            _, marginals = chain.decode_logicals(syndrome_values=syn_vals)
            probs[i] = float(marginals.get("obs", 0.5))
        total_decode_ms = (time.perf_counter() - t1) * 1e3
        record["decode_method"] = "mps_chain"
    elif has_compressed_tn:
        # Use CompressedTNDecoder when chain extraction fails
        probs = np.empty(shots)
        t1 = time.perf_counter()
        for i in range(shots):
            syn_vals = {f"syn_param_{j}": float(syndromes[i, j])
                        for j in range(num_checks)}
            _, marginals = compressed_tn_decoder.decode_logicals(syndrome_values=syn_vals)
            probs[i] = float(marginals.get("obs", 0.5))
        total_decode_ms = (time.perf_counter() - t1) * 1e3
        record["decode_method"] = "compressed_tn"
        if verbose:
            print("  Chain extraction failed, using CompressedTNDecoder")
    else:
        raise RuntimeError("Neither 1D chain nor compressed_tn decoder is available.")

    decode_profile = profiler2.stop()

    predictions = (probs >= 0.5).astype(np.int8)
    ler = float(np.mean(true_logicals != predictions))

    record["total_decode_ms"] = round(total_decode_ms, 2)
    record["per_shot_ms"] = round(total_decode_ms / shots, 3)
    record["logical_error_rate"] = round(ler, 6)
    record["decode_gpu_profile"] = decode_profile

    if verbose:
        print(f"  Decode: {total_decode_ms:.1f}ms total, {total_decode_ms/shots:.3f}ms/shot")
        print(f"  Decode GPU profile: {decode_profile}")
        print(f"  LER: {ler:.4f}")

    # --- Single-shot latency benchmark (requires chain or GPU decoder) ---
    if use_gpu:
        latency = _measure_latency_gpu(dec, syndromes, warmup=warmup, repeats=repeats)
        record["single_shot_latency"] = latency
    elif has_chain:
        latency = _measure_latency_cpu(chain, syndromes, num_checks,
                                       warmup=warmup, repeats=repeats)
        record["single_shot_latency"] = latency
    else:
        record["single_shot_latency"] = {"method": "skipped", "reason": "no_chain"}
    if verbose:
        print(f"  Single-shot latency: {record['single_shot_latency']}")

    # --- Chain save/load + re-decode test (requires chain) ---
    if has_chain:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "chain"
            t_save = time.perf_counter()
            chain.save(save_path)
            save_ms = (time.perf_counter() - t_save) * 1e3

            t_load = time.perf_counter()
            loaded_chain = core.OneDChainMPS.load(save_path)
            load_ms = (time.perf_counter() - t_load) * 1e3

            # Re-decode one shot to verify correctness
            syn_vals = {f"syn_param_{j}": float(syndromes[0, j])
                        for j in range(num_checks)}
            _, marginals_orig = chain.decode_logicals(syndrome_values=syn_vals)
            _, marginals_loaded = loaded_chain.decode_logicals(syndrome_values=syn_vals)
            match = all(
                abs(marginals_orig.get(k, 0) - marginals_loaded.get(k, 0)) < 1e-10
                for k in marginals_orig)

            record["chain_io"] = {
                "save_ms": round(save_ms, 2),
                "load_ms": round(load_ms, 2),
                "reload_decode_match": match,
                "num_sites": len(chain.sites),
            }
        if verbose:
            print(f"  Chain I/O: save={save_ms:.1f}ms, load={load_ms:.1f}ms, match={match}")
    else:
        record["chain_io"] = {"status": "skipped", "reason": "no_chain"}

    # --- Truncation error JSON export ---
    if offline_stats is not None and hasattr(offline_stats, 'to_json'):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "stats.json"
            offline_stats.to_json(json_path)
            with open(json_path) as f:
                stats_data = json.load(f)
            record["truncation_json_exported"] = True
            record["truncation_step_count"] = len(stats_data.get("step_errors", []))
    else:
        record["truncation_json_exported"] = False

    record["status"] = "ok"
    return record


def run_gpu_metrics(
    chi: int = 16,
    shots: int = 32,
    noise_p: float = 0.01,
    seed: int = 2026,
    warmup: int = 5,
    repeats: int = 20,
    code_names: Optional[list[str]] = None,
    skip_missing: bool = False,
    parallel_workers: Optional[int] = None,
    verbose: bool = False,
) -> list[dict]:
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
        raise RuntimeError("No codes were loaded for GPU metrics.")

    workers = resolve_parallel_workers(
        use_gpu=use_gpu,
        parallel_workers=parallel_workers,
    )
    if verbose:
        print(f"[gpu_metrics] codes={len(loaded_codes)}, workers={workers}, use_gpu={use_gpu}")

    if workers == 1 or len(loaded_codes) == 1:
        return [
            _profile_single_code(
                code_name,
                code,
                chi=chi,
                shots=shots,
                noise_p=noise_p,
                seed=seed,
                warmup=warmup,
                repeats=repeats,
                verbose=verbose,
                use_gpu=use_gpu,
            ) for code_name, code in loaded_codes
        ]

    ordered: list[Optional[dict[str, Any]]] = [None] * len(loaded_codes)
    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                _profile_single_code,
                code_name,
                code,
                chi=chi,
                shots=shots,
                noise_p=noise_p,
                seed=seed,
                warmup=warmup,
                repeats=repeats,
                verbose=verbose,
                use_gpu=use_gpu,
            ): idx for idx, (code_name, code) in enumerate(loaded_codes)
        }
        for future in cf.as_completed(future_map):
            ordered[future_map[future]] = future.result()
    return [r for r in ordered if r is not None]


# ---------------------------------------------------------------------------
# Pytest
# ---------------------------------------------------------------------------
def test_gpu_metrics_smoke():
    """Smoke test: profile with small shots."""
    results = run_gpu_metrics(
        chi=8,
        shots=8,
        noise_p=0.01,
        seed=42,
        warmup=2,
        repeats=5,
        code_names=["BB_90"],
        skip_missing=True,
        parallel_workers=2,
        verbose=True,
    )
    assert len(results) > 0
    ok_count = 0
    for r in results:
        assert r.get("status") == "ok", f"Unexpected status: {r}"
        ok_count += 1
        if r.get("decode_method") not in ("exact_fallback", "compressed_tn"):
            assert r["single_shot_latency"].get("avg_ms", 0) > 0
            assert r["chain_io"]["reload_decode_match"] is True
        assert r.get("logical_error_rate") is not None
    assert ok_count > 0, "No codes completed successfully"
    print(f"\nGPU metrics smoke test passed: {ok_count} codes profiled")


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"GPU available: {HAS_GPU}")
    print(f"cudaq_qec available: {HAS_CUDAQ}")
    print(f"pynvml available: {HAS_PYNVML}")
    print()

    results = run_gpu_metrics(
        chi=16,
        shots=32,
        noise_p=0.01,
        seed=2026,
        warmup=5,
        repeats=20,
        code_names=None,
        skip_missing=False,
        parallel_workers=None,
        verbose=True,
    )

    out_dir = WORKSPACE / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gpu_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'Code':<10} {'Backend':<6} {'Compile':>10} {'Decode':>10} "
          f"{'Latency':>10} {'LER':>8} {'GPU_mem_peak':>13} {'GPU_util':>10}")
    print("-" * 90)
    for r in results:
        if r["status"] != "ok":
            print(f"{r['code']:<10} {r['backend']:<6} {'FAILED':>10}")
            continue
        gpu_mem = r.get("decode_gpu_profile", {}).get("mem_used_peak_mb", "N/A")
        gpu_util = r.get("decode_gpu_profile", {}).get("gpu_util_avg", "N/A")
        gpu_mem_str = f"{gpu_mem:.0f}MB" if isinstance(gpu_mem, (int, float)) else gpu_mem
        gpu_util_str = f"{gpu_util:.0f}%" if isinstance(gpu_util, (int, float)) else gpu_util
        print(f"{r['code']:<10} {r['backend']:<6} "
              f"{r['compile_ms']:>9.1f}ms "
              f"{r['total_decode_ms']:>9.1f}ms "
              f"{r['single_shot_latency']['avg_ms']:>9.3f}ms "
              f"{r['logical_error_rate']:>8.4f} "
              f"{gpu_mem_str:>13} {gpu_util_str:>10}")
