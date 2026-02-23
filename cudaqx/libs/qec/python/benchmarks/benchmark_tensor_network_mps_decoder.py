#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Benchmark offline + online flow for `tensor_network_mps_decoder`.

Example:
  python benchmark_tensor_network_mps_decoder.py --repeats 100 --bond-dim 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import cudaq_qec as qec
from cudaq_qec.plugins.decoders.tensor_network_mps_decoder import (
    set_global_mps_decoder_config,
)


def _load_matrix(path: str) -> np.ndarray:
    file_path = Path(path)
    if file_path.suffix in {".npy", ".npz"}:
        data = np.load(file_path)
        if isinstance(data, np.lib.npyio.NpzFile):
            if len(data.files) != 1:
                raise ValueError(
                    f"{path} contains multiple arrays, expected one array.")
            return np.asarray(data[data.files[0]], dtype=np.float64)
        return np.asarray(data, dtype=np.float64)
    return np.loadtxt(file_path, dtype=np.float64)


def _toy_case(num_checks: int, num_errors: int, seed: int,
              noise_p: float) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    h = (rng.random((num_checks, num_errors)) < 0.33).astype(np.float64)
    # Ensure no all-zero rows/cols for a valid toy Tanner graph.
    for r in range(num_checks):
        if np.all(h[r] == 0):
            h[r, rng.integers(0, num_errors)] = 1.0
    for c in range(num_errors):
        if np.all(h[:, c] == 0):
            h[rng.integers(0, num_checks), c] = 1.0

    logical = np.zeros((1, num_errors), dtype=np.float64)
    logical[0, rng.choice(num_errors, size=max(1, num_errors // 4),
                          replace=False)] = 1.0
    noise = [float(noise_p)] * num_errors
    return h, logical, noise


def _parse_syndrome(raw: str, num_checks: int) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(values) != num_checks:
        raise ValueError(
            f"syndrome length {len(values)} != num_checks {num_checks}")
    return values


def _stats_to_dict(stats: Any) -> dict[str, Any]:
    return {
        "initial_nodes": int(stats.initial_nodes),
        "initial_edges": int(stats.initial_edges),
        "preserved_target": int(stats.preserved_target),
        "steps": int(stats.steps),
        "truncation_error": float(stats.truncation_error),
        "max_bond_dim": int(stats.max_bond_dim),
        "active_nodes": int(stats.active_nodes),
        "active_edges": int(stats.active_edges),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark tensor_network_mps_decoder.")
    parser.add_argument("--h-path", type=str, default=None)
    parser.add_argument("--logical-path", type=str, default=None)
    parser.add_argument("--noise-path", type=str, default=None)
    parser.add_argument("--num-checks", type=int, default=8)
    parser.add_argument("--num-errors", type=int, default=16)
    parser.add_argument("--noise-p", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bond-dim", type=int, default=16)
    parser.add_argument("--low-threshold", type=float, default=0.4)
    parser.add_argument("--high-threshold", type=float, default=0.6)
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--offline-cutoff", type=float, default=1.0e-12)
    parser.add_argument("--offline-max-steps", type=int, default=100000)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--syndrome", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if (args.h_path is None) != (args.logical_path is None):
        raise ValueError("--h-path and --logical-path must be provided together.")

    if args.h_path is not None:
        h = _load_matrix(args.h_path)
        logical = _load_matrix(args.logical_path)
        if logical.ndim == 1:
            logical = logical.reshape(1, -1)
        if h.ndim != 2 or logical.shape != (1, h.shape[1]):
            raise ValueError(
                "Invalid matrix shapes. Expected H:(m,n), logical:(1,n).")
        if args.noise_path is not None:
            noise_arr = _load_matrix(args.noise_path).reshape(-1)
            if noise_arr.shape[0] != h.shape[1]:
                raise ValueError(
                    "noise length must equal number of error indices.")
            noise = [float(x) for x in noise_arr.tolist()]
        else:
            noise = [float(args.noise_p)] * h.shape[1]
    else:
        h, logical, noise = _toy_case(
            num_checks=args.num_checks,
            num_errors=args.num_errors,
            seed=args.seed,
            noise_p=args.noise_p,
        )

    set_global_mps_decoder_config(
        bond_dim=args.bond_dim,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        max_conditional_rounds=args.max_rounds,
    )

    decoder = qec.get_decoder(
        "tensor_network_mps_decoder",
        h,
        logical_obs=logical,
        noise_model=noise,
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

    if args.syndrome is not None:
        syndrome = _parse_syndrome(args.syndrome, h.shape[0])
    else:
        rng = np.random.default_rng(args.seed + 1)
        syndrome = [float(x) for x in (rng.random(h.shape[0]) < 0.5)]

    result = decoder.decode(syndrome)
    latency = decoder.benchmark_decode_latency(
        syndrome,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    summary = {
        "shape": {
            "num_checks": int(h.shape[0]),
            "num_errors": int(h.shape[1]),
        },
        "config": {
            "bond_dim": int(args.bond_dim),
            "low_threshold": float(args.low_threshold),
            "high_threshold": float(args.high_threshold),
            "max_rounds": int(args.max_rounds),
            "device": args.device,
            "dtype": args.dtype,
        },
        "offline_stats": _stats_to_dict(decoder.offline_stats),
        "decode": {
            "syndrome": syndrome,
            "logical_probability_1": float(result.result[0]),
            "timing": decoder.last_decode_timing,
        },
        "latency_ms": latency,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
