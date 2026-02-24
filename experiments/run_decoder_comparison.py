#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Compare compressed MPS decoder against exact tensor-network decoder.

Outputs:
1) detailed JSON report with per-case/per-noise metrics,
2) flat CSV summary for quick plotting.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    import pynvml
except Exception:  # pragma: no cover
    pynvml = None

try:
    import cudaq_qec as qec
    from cudaq_qec.plugins.decoders.tensor_network_mps_decoder import (
        set_global_mps_decoder_config,
    )
    _IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    qec = None  # type: ignore
    set_global_mps_decoder_config = None  # type: ignore
    _IMPORT_ERROR = exc

# Make workspace root importable when running as
# `python experiments/run_decoder_comparison.py`.
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for --config. Install with `pip install pyyaml`."
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must parse to a mapping: {path}")
    return data


def _read_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix in {".npy", ".npz"}:
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            if len(arr.files) != 1:
                raise ValueError(
                    f"{path} has multiple arrays, expected exactly one.")
            return np.asarray(arr[arr.files[0]], dtype=np.float64)
        return np.asarray(arr, dtype=np.float64)
    if path.suffix in {".csv"}:
        return np.loadtxt(path, delimiter=",", dtype=np.float64)
    return np.loadtxt(path, dtype=np.float64)


def _as_2d(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D array, got shape {arr.shape}.")
    return arr


def _infer_noise_vector(n: int,
                        case_cfg: dict[str, Any],
                        base_noise_p: float) -> list[float]:
    if "noise_vector" in case_cfg:
        vec = [float(x) for x in case_cfg["noise_vector"]]
        if len(vec) != n:
            raise ValueError(
                f"noise_vector length {len(vec)} != number of errors {n}.")
        return vec
    if "noise_path" in case_cfg:
        vec = _read_array(Path(str(case_cfg["noise_path"]))).reshape(-1)
        if vec.shape[0] != n:
            raise ValueError(f"noise_path length {vec.shape[0]} != {n}.")
        return [float(x) for x in vec.tolist()]
    return [float(case_cfg.get("noise_p", base_noise_p))] * n


def _try_call(obj: Any, names: list[str]) -> np.ndarray:
    errors = []
    for name in names:
        if not hasattr(obj, name):
            continue
        try:
            return np.asarray(getattr(obj, name)(), dtype=np.float64)
        except Exception as exc:  # pragma: no cover
            errors.append(f"{name}: {exc}")
    raise RuntimeError(f"Failed to get matrix via {names}, errors={errors}")


def _extract_code_matrices(code: Any,
                           logical_basis: str = "z") -> tuple[np.ndarray, np.ndarray]:
    basis = logical_basis.lower()
    if basis not in {"x", "z"}:
        raise ValueError(f"logical_basis must be 'x' or 'z', got {logical_basis}")

    if basis == "z":
        h = _try_call(code, ["get_parity_z", "get_parity"])
        logical = _try_call(code,
                            ["get_observables_z", "get_pauli_observables_matrix"])
    else:
        h = _try_call(code, ["get_parity_x", "get_parity"])
        logical = _try_call(code,
                            ["get_observables_x", "get_pauli_observables_matrix"])

    h = _as_2d(np.asarray(h, dtype=np.float64), "H")
    logical = _as_2d(np.asarray(logical, dtype=np.float64), "logical")
    if logical.shape[1] != h.shape[1]:
        raise ValueError(
            f"logical columns {logical.shape[1]} != H columns {h.shape[1]}.")
    return h, logical


def _extract_codegen_matrices(
        code_obj: Any,
        parity_attr: str = "hz",
        logical_attr: str = "lz") -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(code_obj, parity_attr):
        raise AttributeError(
            f"Generated code object has no parity attribute '{parity_attr}'.")
    if not hasattr(code_obj, logical_attr):
        raise AttributeError(
            f"Generated code object has no logical attribute '{logical_attr}'.")

    h = np.asarray(getattr(code_obj, parity_attr), dtype=np.float64)
    logical = np.asarray(getattr(code_obj, logical_attr), dtype=np.float64)
    h = _as_2d(h, f"codegen:{parity_attr}")
    logical = _as_2d(logical, f"codegen:{logical_attr}")
    if logical.shape[1] != h.shape[1]:
        raise ValueError(
            f"codegen logical columns {logical.shape[1]} != H columns {h.shape[1]}.")
    return h, logical


def _discover_matrix_cases(matrix_dir: Path) -> list[dict[str, Any]]:
    """Discover matrix cases from subfolders.

    Expected layout:
      matrix_dir/<case_name>/H.npy
      matrix_dir/<case_name>/logical.npy
      matrix_dir/<case_name>/noise.npy   (optional)
    """
    cases: list[dict[str, Any]] = []
    if not matrix_dir.exists():
        return cases
    for sub in sorted([p for p in matrix_dir.iterdir() if p.is_dir()]):
        h_path = sub / "H.npy"
        logical_path = sub / "logical.npy"
        if not (h_path.exists() and logical_path.exists()):
            continue
        case: dict[str, Any] = {
            "name": sub.name,
            "source": "matrix",
            "h_path": str(h_path),
            "logical_path": str(logical_path),
        }
        noise_path = sub / "noise.npy"
        if noise_path.exists():
            case["noise_path"] = str(noise_path)
        meta_path = sub / "meta.json"
        if meta_path.exists():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                if isinstance(meta, dict):
                    case.update(meta)
            except Exception:
                pass
        cases.append(case)
    return cases


def _sample_syndromes(h: np.ndarray,
                      noise_vec: list[float],
                      shots: int,
                      seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = h.shape[1]
    probs = np.asarray(noise_vec, dtype=np.float64).reshape(1, n)
    errors = (rng.random((shots, n)) < probs).astype(np.int8)
    syndromes = (errors @ h.T.astype(np.int8)) % 2
    return syndromes.astype(np.float64)


def _single_latency_ms(decoder: Any,
                       syndrome: list[float],
                       warmup: int,
                       repeats: int) -> dict[str, float]:
    for _ in range(max(0, warmup)):
        decoder.decode(syndrome)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        decoder.decode(syndrome)
        times.append((time.perf_counter() - t0) * 1e3)
    arr = np.asarray(times, dtype=np.float64)
    return {
        "avg_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def _decode_many(decoder: Any, syndromes: np.ndarray) -> tuple[np.ndarray, float]:
    probs = np.empty((syndromes.shape[0],), dtype=np.float64)
    t0 = time.perf_counter()
    for i in range(syndromes.shape[0]):
        out = decoder.decode([float(x) for x in syndromes[i]])
        probs[i] = float(out.result[0])
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    return probs, float(elapsed_ms)


@dataclass
class GPUSnapshot:
    mem_used_mb: Optional[float]
    mem_total_mb: Optional[float]
    gpu_util: Optional[float]
    mem_util: Optional[float]


def _gpu_snapshot(device_index: int) -> GPUSnapshot:
    if pynvml is None:
        return GPUSnapshot(None, None, None, None)
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return GPUSnapshot(
            mem_used_mb=float(mem.used) / (1024.0 * 1024.0),
            mem_total_mb=float(mem.total) / (1024.0 * 1024.0),
            gpu_util=float(util.gpu),
            mem_util=float(util.memory),
        )
    except Exception:
        return GPUSnapshot(None, None, None, None)


class GPUProfiler:
    def __init__(self, device_index: int = 0, interval_s: float = 0.05):
        self.device_index = int(device_index)
        self.interval_s = float(interval_s)
        self.samples: list[tuple[float, float, float, float]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._enabled = pynvml is not None

    def start(self) -> None:
        if not self._enabled:
            return
        self.samples.clear()
        self._stop_event.clear()

        def _loop():
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            except Exception:
                return
            while not self._stop_event.is_set():
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.samples.append((
                        float(mem.used) / (1024.0 * 1024.0),
                        float(mem.total) / (1024.0 * 1024.0),
                        float(util.gpu),
                        float(util.memory),
                    ))
                except Exception:
                    pass
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Optional[float]]:
        if not self._enabled:
            return {
                "sample_count": 0,
                "mem_used_peak_mb": None,
                "mem_used_avg_mb": None,
                "mem_total_mb": None,
                "gpu_util_peak": None,
                "gpu_util_avg": None,
                "mem_util_peak": None,
                "mem_util_avg": None,
            }
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self.samples:
            return {
                "sample_count": 0,
                "mem_used_peak_mb": None,
                "mem_used_avg_mb": None,
                "mem_total_mb": None,
                "gpu_util_peak": None,
                "gpu_util_avg": None,
                "mem_util_peak": None,
                "mem_util_avg": None,
            }
        arr = np.asarray(self.samples, dtype=np.float64)
        return {
            "sample_count": int(arr.shape[0]),
            "mem_used_peak_mb": float(arr[:, 0].max()),
            "mem_used_avg_mb": float(arr[:, 0].mean()),
            "mem_total_mb": float(arr[:, 1].max()),
            "gpu_util_peak": float(arr[:, 2].max()),
            "gpu_util_avg": float(arr[:, 2].mean()),
            "mem_util_peak": float(arr[:, 3].max()),
            "mem_util_avg": float(arr[:, 3].mean()),
        }


def _safe_get_decoder(name: str, allow_cpu_fallback: bool, **kwargs) -> tuple[Any, str]:
    h = kwargs.pop("H")
    try:
        return qec.get_decoder(name, h,
                               **kwargs), str(kwargs.get("device", "unknown"))
    except Exception:
        if not allow_cpu_fallback or kwargs.get("device") == "cpu":
            raise
        new_kwargs = dict(kwargs)
        new_kwargs["device"] = "cpu"
        return qec.get_decoder(name, h, **new_kwargs), "cpu"


def _expand_builtin_case(case_cfg: dict[str, Any],
                         default_noise_p: float) -> list[dict[str, Any]]:
    code_name = str(case_cfg["code_name"])
    code_options = dict(case_cfg.get("code_options", {}))
    logical_basis = str(case_cfg.get("logical_basis", "z"))
    code = qec.get_code(code_name, **code_options)
    h, logical_mat = _extract_code_matrices(code, logical_basis=logical_basis)
    max_logicals = int(case_cfg.get("max_logicals", 1))
    max_logicals = max(1, min(max_logicals, logical_mat.shape[0]))
    rows = list(range(max_logicals))
    if "logical_rows" in case_cfg:
        rows = [int(x) for x in case_cfg["logical_rows"]]
    out_cases: list[dict[str, Any]] = []
    for row in rows:
        if row < 0 or row >= logical_mat.shape[0]:
            continue
        new_cfg = dict(case_cfg)
        new_cfg["resolved_h"] = h
        new_cfg["resolved_logical"] = logical_mat[row:row + 1, :]
        new_cfg["resolved_noise"] = _infer_noise_vector(h.shape[1], case_cfg,
                                                        default_noise_p)
        new_cfg["resolved_case_id"] = f"{case_cfg['name']}::L{row}"
        new_cfg["resolved_source"] = "builtin"
        out_cases.append(new_cfg)
    return out_cases


def _expand_codegen_case(case_cfg: dict[str, Any],
                         default_noise_p: float) -> list[dict[str, Any]]:
    module_name = str(case_cfg.get("module", "experiments.codes.codes"))
    factory_name = str(case_cfg["factory"])
    factory_args = list(case_cfg.get("factory_args", []))
    factory_kwargs = dict(case_cfg.get("factory_kwargs", {}))
    skip_if_missing = bool(case_cfg.get("skip_if_missing", False))

    try:
        module = importlib.import_module(module_name)
        if not hasattr(module, factory_name):
            raise AttributeError(
                f"Module '{module_name}' has no factory '{factory_name}'.")
        factory = getattr(module, factory_name)
        code_obj = factory(*factory_args, **factory_kwargs)

        parity_attr = str(case_cfg.get("parity_attr", "hz"))
        logical_attr = str(case_cfg.get("logical_attr", "lz"))
        h, logical = _extract_codegen_matrices(
            code_obj, parity_attr=parity_attr, logical_attr=logical_attr)
    except Exception as exc:
        if skip_if_missing:
            print(
                f"[skip] codegen case '{case_cfg.get('name', 'unknown')}' "
                f"failed to resolve ({type(exc).__name__}): {exc}")
            return []
        raise

    max_logicals = int(case_cfg.get("max_logicals", 1))
    max_logicals = max(1, min(max_logicals, logical.shape[0]))
    rows = list(range(max_logicals))
    if "logical_rows" in case_cfg:
        rows = [int(x) for x in case_cfg["logical_rows"]]

    out_cases: list[dict[str, Any]] = []
    for row in rows:
        if row < 0 or row >= logical.shape[0]:
            continue
        new_cfg = dict(case_cfg)
        new_cfg["resolved_h"] = h
        new_cfg["resolved_logical"] = logical[row:row + 1, :]
        new_cfg["resolved_noise"] = _infer_noise_vector(h.shape[1], case_cfg,
                                                        default_noise_p)
        new_cfg["resolved_case_id"] = f"{case_cfg['name']}::L{row}"
        new_cfg["resolved_source"] = "codegen"
        new_cfg["resolved_codegen_meta"] = {
            "module": module_name,
            "factory": factory_name,
            "factory_args": factory_args,
            "factory_kwargs": factory_kwargs,
            "parity_attr": parity_attr,
            "logical_attr": logical_attr,
            "N": int(getattr(code_obj, "N", h.shape[1])),
            "K": int(getattr(code_obj, "K", logical.shape[0])),
            "D": float(getattr(code_obj, "D", math.nan)),
            "source_path": str(getattr(code_obj, "source_path", "")),
            "source_format": str(getattr(code_obj, "source_format", "")),
            "source_family": str(getattr(code_obj, "source_family", "")),
        }
        out_cases.append(new_cfg)
    return out_cases


def _expand_matrix_case(case_cfg: dict[str, Any],
                        default_noise_p: float) -> list[dict[str, Any]]:
    h_path = Path(str(case_cfg["h_path"]))
    logical_path = Path(str(case_cfg["logical_path"]))
    if bool(case_cfg.get("skip_if_missing", False)):
        if not h_path.exists() or not logical_path.exists():
            print(
                f"[skip] matrix case '{case_cfg.get('name', 'unknown')}' missing files: "
                f"H={h_path.exists()}, logical={logical_path.exists()}")
            return []

    h = _as_2d(_read_array(h_path), "H")
    logical = _as_2d(_read_array(logical_path), "logical")
    if logical.shape[1] != h.shape[1]:
        raise ValueError(
            f"case={case_cfg['name']}: logical columns {logical.shape[1]} != H columns {h.shape[1]}"
        )
    max_logicals = int(case_cfg.get("max_logicals", 1))
    max_logicals = max(1, min(max_logicals, logical.shape[0]))
    rows = list(range(max_logicals))
    if "logical_rows" in case_cfg:
        rows = [int(x) for x in case_cfg["logical_rows"]]

    out_cases: list[dict[str, Any]] = []
    for row in rows:
        if row < 0 or row >= logical.shape[0]:
            continue
        new_cfg = dict(case_cfg)
        new_cfg["resolved_h"] = h
        new_cfg["resolved_logical"] = logical[row:row + 1, :]
        new_cfg["resolved_noise"] = _infer_noise_vector(h.shape[1], case_cfg,
                                                        default_noise_p)
        new_cfg["resolved_case_id"] = f"{case_cfg['name']}::L{row}"
        new_cfg["resolved_source"] = "matrix"
        out_cases.append(new_cfg)
    return out_cases


def _flatten_cases(config_cases: list[dict[str, Any]],
                   default_noise_p: float) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for case_cfg in config_cases:
        source = str(case_cfg.get("source", "builtin")).lower()
        if source == "builtin":
            flat.extend(_expand_builtin_case(case_cfg, default_noise_p))
        elif source == "codegen":
            flat.extend(_expand_codegen_case(case_cfg, default_noise_p))
        elif source == "matrix":
            flat.extend(_expand_matrix_case(case_cfg, default_noise_p))
        else:
            raise ValueError(f"Unknown case source={source}: {case_cfg}")
    return flat


def _coerce_float_list(raw: str) -> list[float]:
    out = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    if not out:
        raise ValueError("Expected at least one numeric value.")
    return out


def _write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run decoder comparison sweep.")
    parser.add_argument("--config",
                        type=str,
                        default="experiments/configs/qldpc_six_codes.yaml")
    parser.add_argument("--output-dir",
                        type=str,
                        default="experiments/results")
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--noise-values", type=str, default="0.01,0.03")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--bond-dim", type=int, default=16)
    parser.add_argument("--low-threshold", type=float, default=0.4)
    parser.add_argument("--high-threshold", type=float, default=0.6)
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--offline-cutoff", type=float, default=1.0e-12)
    parser.add_argument("--offline-max-steps", type=int, default=100000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--latency-repeats", type=int, default=50)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-sample-interval-ms", type=float, default=50.0)
    parser.add_argument("--matrix-dir",
                        type=str,
                        default="experiments/data/cases")
    parser.add_argument("--case-filter",
                        type=str,
                        default="",
                        help="Regex on resolved case id.")
    parser.add_argument("--skip-exact-n-errors",
                        type=int,
                        default=256,
                        help="Skip exact decoder if number of error bits > value.")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if qec is None or set_global_mps_decoder_config is None:
        raise RuntimeError(
            "Failed to import cudaq_qec. Run setup first: "
            "`bash scripts/linux/setup_cudaq_mps_env.sh`. "
            f"Original error: {_IMPORT_ERROR}")

    config_path = Path(args.config)
    config = _load_yaml(config_path) if config_path.exists() else {}
    defaults = config.get("defaults", {})
    config_cases = list(config.get("cases", []))
    if not isinstance(config_cases, list):
        raise ValueError("config.cases must be a list.")

    matrix_dir = Path(args.matrix_dir)
    discovered_cases = _discover_matrix_cases(matrix_dir)
    existing_names = {
        str(c.get("name"))
        for c in config_cases if isinstance(c, dict) and "name" in c
    }
    for discovered in discovered_cases:
        name = str(discovered.get("name", ""))
        if name and name in existing_names:
            continue
        config_cases.append(discovered)
    if not config_cases:
        raise RuntimeError(
            f"No cases found. Please provide config cases or matrix cases under {matrix_dir}."
        )

    shots = int(defaults.get("shots", args.shots))
    default_noise_p = float(defaults.get("noise_p", 0.01))
    noise_values = _coerce_float_list(
        str(defaults.get("noise_values", args.noise_values)))
    seed = int(defaults.get("seed", args.seed))

    set_global_mps_decoder_config(
        bond_dim=args.bond_dim,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        max_conditional_rounds=args.max_rounds,
    )

    flat_cases = _flatten_cases(config_cases, default_noise_p=default_noise_p)
    if args.case_filter:
        pat = re.compile(args.case_filter)
        flat_cases = [
            c for c in flat_cases if pat.search(str(c["resolved_case_id"]))
        ]
    if not flat_cases:
        raise RuntimeError("No cases remain after filtering.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report: dict[str, Any] = {
        "timestamp_utc": stamp,
        "args": vars(args),
        "defaults": defaults,
        "noise_values": noise_values,
        "cases": [],
    }
    csv_rows: list[dict[str, Any]] = []

    for case_idx, case in enumerate(flat_cases):
        case_id = str(case["resolved_case_id"])
        h = np.asarray(case["resolved_h"], dtype=np.float64)
        logical_obs = np.asarray(case["resolved_logical"], dtype=np.float64)
        noise_base = list(case["resolved_noise"])

        if args.verbose:
            print(f"[{case_idx+1}/{len(flat_cases)}] case={case_id}, H={h.shape}")

        case_report: dict[str, Any] = {
            "case_id": case_id,
            "source": case.get("resolved_source", "unknown"),
            "codegen_meta": case.get("resolved_codegen_meta"),
            "shape": {
                "checks": int(h.shape[0]),
                "errors": int(h.shape[1]),
                "logicals": int(logical_obs.shape[0]),
            },
            "runs": [],
        }

        use_fixed_noise_case = (
            bool(case.get("fixed_noise", False))
            or ("noise_vector" in case)
            or ("noise_path" in case)
            or ("noise_p" in case))
        case_noise_values = noise_values
        if use_fixed_noise_case and not bool(case.get("sweep_noise", False)):
            case_noise_values = [float(case.get("noise_p", noise_values[0]))]

        for p_idx, noise_p in enumerate(case_noise_values):
            run_seed = seed + 1000 * case_idx + p_idx
            noise_vec = noise_base if use_fixed_noise_case else [
                float(noise_p)
            ] * h.shape[1]
            syndromes = _sample_syndromes(h, noise_vec, shots=shots, seed=run_seed)

            run_record: dict[str, Any] = {
                "noise_p": float(noise_p),
                "seed": int(run_seed),
                "shots": int(shots),
                "decoder_status": {},
            }

            snapshot_before = _gpu_snapshot(args.gpu_index)
            exact_decoder = None
            exact_device = args.device
            exact_error: Optional[str] = None
            if h.shape[1] > args.skip_exact_n_errors:
                exact_error = (
                    f"skipped_exact: n_errors={h.shape[1]} > skip_exact_n_errors={args.skip_exact_n_errors}"
                )
            else:
                try:
                    exact_decoder, exact_device = _safe_get_decoder(
                        "tensor_network_decoder",
                        allow_cpu_fallback=args.allow_cpu_fallback,
                        H=h,
                        logical_obs=logical_obs,
                        noise_model=noise_vec,
                        dtype=args.dtype,
                        device=args.device,
                    )
                except Exception as exc:
                    exact_error = f"init_failed: {exc}"
            snapshot_after_exact = _gpu_snapshot(args.gpu_index)

            mps_decoder = None
            mps_device = args.device
            mps_error: Optional[str] = None
            try:
                mps_decoder, mps_device = _safe_get_decoder(
                    "tensor_network_mps_decoder",
                    allow_cpu_fallback=args.allow_cpu_fallback,
                    H=h,
                    logical_obs=logical_obs,
                    noise_model=noise_vec,
                    dtype=args.dtype,
                    device=args.device,
                    bond_dim=args.bond_dim,
                    offline_cutoff=args.offline_cutoff,
                    offline_max_steps=args.offline_max_steps,
                    low_threshold=args.low_threshold,
                    high_threshold=args.high_threshold,
                    max_conditional_rounds=args.max_rounds,
                    verbose=args.verbose,
                )
            except Exception as exc:
                mps_error = f"init_failed: {exc}"
            snapshot_after_mps = _gpu_snapshot(args.gpu_index)

            run_record["gpu_init"] = {
                "before": snapshot_before.__dict__,
                "after_exact": snapshot_after_exact.__dict__,
                "after_mps": snapshot_after_mps.__dict__,
                "exact_init_mem_delta_mb":
                None
                if snapshot_before.mem_used_mb is None
                or snapshot_after_exact.mem_used_mb is None else
                float(snapshot_after_exact.mem_used_mb - snapshot_before.mem_used_mb),
                "mps_init_mem_delta_mb":
                None
                if snapshot_after_exact.mem_used_mb is None
                or snapshot_after_mps.mem_used_mb is None else
                float(snapshot_after_mps.mem_used_mb -
                      snapshot_after_exact.mem_used_mb),
            }

            exact_probs = None
            mps_probs = None

            if exact_decoder is not None:
                profiler = GPUProfiler(
                    device_index=args.gpu_index,
                    interval_s=max(1.0e-3, args.gpu_sample_interval_ms / 1000.0),
                )
                profiler.start()
                try:
                    exact_probs, exact_total_ms = _decode_many(
                        exact_decoder, syndromes)
                    exact_profile = profiler.stop()
                    exact_latency = _single_latency_ms(
                        exact_decoder,
                        syndrome=[float(x) for x in syndromes[0]],
                        warmup=args.warmup,
                        repeats=args.latency_repeats,
                    )
                    run_record["exact"] = {
                        "status": "ok",
                        "device": exact_device,
                        "total_ms": float(exact_total_ms),
                        "per_shot_ms": float(exact_total_ms / shots),
                        "latency_ms": exact_latency,
                        "gpu_profile": exact_profile,
                    }
                except Exception as exc:
                    exact_error = f"decode_failed: {exc}"
                    run_record["exact"] = {
                        "status": "error",
                        "device": exact_device,
                        "error": exact_error,
                    }
            else:
                run_record["exact"] = {
                    "status": "error",
                    "device": exact_device,
                    "error": exact_error,
                }

            if mps_decoder is not None:
                profiler = GPUProfiler(
                    device_index=args.gpu_index,
                    interval_s=max(1.0e-3, args.gpu_sample_interval_ms / 1000.0),
                )
                profiler.start()
                try:
                    mps_probs, mps_total_ms = _decode_many(mps_decoder, syndromes)
                    mps_profile = profiler.stop()
                    mps_latency = _single_latency_ms(
                        mps_decoder,
                        syndrome=[float(x) for x in syndromes[0]],
                        warmup=args.warmup,
                        repeats=args.latency_repeats,
                    )
                    run_record["mps"] = {
                        "status": "ok",
                        "device": mps_device,
                        "total_ms": float(mps_total_ms),
                        "per_shot_ms": float(mps_total_ms / shots),
                        "latency_ms": mps_latency,
                        "gpu_profile": mps_profile,
                        "offline_stats": getattr(mps_decoder, "offline_stats",
                                                 None).__dict__
                        if hasattr(mps_decoder, "offline_stats") else None,
                    }
                except Exception as exc:
                    mps_error = f"decode_failed: {exc}"
                    run_record["mps"] = {
                        "status": "error",
                        "device": mps_device,
                        "error": mps_error,
                    }
            else:
                run_record["mps"] = {
                    "status": "error",
                    "device": mps_device,
                    "error": mps_error,
                }

            if exact_probs is not None and mps_probs is not None:
                abs_diff = np.abs(exact_probs - mps_probs)
                decisions_exact = exact_probs >= 0.5
                decisions_mps = mps_probs >= 0.5
                agreement = float(np.mean(decisions_exact == decisions_mps))
                run_record["comparison"] = {
                    "mean_abs_prob_diff": float(np.mean(abs_diff)),
                    "max_abs_prob_diff": float(np.max(abs_diff)),
                    "decision_agreement": agreement,
                    "mps_speedup_over_exact":
                    None
                    if run_record["mps"]["total_ms"] <= 0 else
                    float(run_record["exact"]["total_ms"] /
                          run_record["mps"]["total_ms"]),
                }
            else:
                run_record["comparison"] = {
                    "mean_abs_prob_diff": None,
                    "max_abs_prob_diff": None,
                    "decision_agreement": None,
                    "mps_speedup_over_exact": None,
                }

            case_report["runs"].append(run_record)
            csv_rows.append({
                "case_id":
                case_id,
                "noise_p":
                float(noise_p),
                "shots":
                int(shots),
                "exact_status":
                run_record["exact"]["status"],
                "mps_status":
                run_record["mps"]["status"],
                "exact_total_ms":
                run_record["exact"].get("total_ms"),
                "mps_total_ms":
                run_record["mps"].get("total_ms"),
                "speedup_mps_over_exact":
                run_record["comparison"]["mps_speedup_over_exact"],
                "mean_abs_prob_diff":
                run_record["comparison"]["mean_abs_prob_diff"],
                "max_abs_prob_diff":
                run_record["comparison"]["max_abs_prob_diff"],
                "decision_agreement":
                run_record["comparison"]["decision_agreement"],
                "exact_gpu_mem_peak_mb":
                run_record["exact"].get("gpu_profile", {}).get("mem_used_peak_mb")
                if isinstance(run_record.get("exact"), dict) else None,
                "mps_gpu_mem_peak_mb":
                run_record["mps"].get("gpu_profile", {}).get("mem_used_peak_mb")
                if isinstance(run_record.get("mps"), dict) else None,
                "exact_gpu_util_avg":
                run_record["exact"].get("gpu_profile", {}).get("gpu_util_avg")
                if isinstance(run_record.get("exact"), dict) else None,
                "mps_gpu_util_avg":
                run_record["mps"].get("gpu_profile", {}).get("gpu_util_avg")
                if isinstance(run_record.get("mps"), dict) else None,
            })

        report["cases"].append(case_report)

    json_path = output_dir / f"decoder_comparison_{stamp}.json"
    csv_path = output_dir / f"decoder_comparison_{stamp}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    _write_summary_csv(csv_rows, csv_path)

    print(f"Saved JSON report: {json_path}")
    print(f"Saved CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
