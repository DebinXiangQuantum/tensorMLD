#!/usr/bin/env python3
"""Export offline-compressed 1D MPS chains for the 9 ISCA revision codes.

Uses a single multi-logical tensor network per code (all K logical observables
+ M syndrome parameters preserved), producing one chain with K+M sites.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from quimb import oset
from quimb.tensor import Tensor, TensorNetwork

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
core = _load_module("isca_revision_mps_core", _TNU / "mps_decoder_core.py")
factory = _load_module("isca_revision_mps_factory", _TNU / "tensor_network_factory.py")

from experiments.codes.isca_revision_codes import load_isca_revision_codes


def build_full_multi_logical_tn(
    H: np.ndarray,
    logical: np.ndarray,
    noise_p: float,
) -> tuple[TensorNetwork, list[str], list[str]]:
    """Build ONE parameterised TN with ALL K logical observables preserved.

    Returns (full_tn, syndrome_param_inds, logical_obs_inds) where
    syndrome_param_inds has M entries and logical_obs_inds has K entries.
    The preserve set for compression is their union (K+M indices).
    """
    num_checks, num_errors = H.shape
    K = logical.shape[0]
    check_inds = [f"s_{i}" for i in range(num_checks)]
    error_inds = [f"e_{i}" for i in range(num_errors)]
    syndrome_param_inds = [f"syn_param_{i}" for i in range(num_checks)]
    logical_obs_inds = [f"obs_{k}" for k in range(K)]
    logical_inds = [f"l_{k}" for k in range(K)]
    logical_tags = [f"LOG_{k}" for k in range(K)]

    # Code TN from parity check matrix
    code_tn = factory.tensor_network_from_parity_check(
        H.astype(np.float64),
        col_inds=error_inds,
        row_inds=check_inds,
    )

    # Logical TN: all K logical operators sharing the same error indices
    logical_2d = logical.astype(np.float64)
    logical_tn = factory.tensor_network_from_parity_check(
        logical_2d,
        col_inds=error_inds,
        row_inds=logical_inds,
        tags=logical_tags,
    )
    logical_tn = logical_tn.combine(
        factory.tensor_network_from_logical_observable(
            logical_2d,
            logical_inds,
            logical_obs_inds,
            logical_tags,
        ),
        virtual=True,
    )

    # Parametrized syndrome network
    param_syn_tn = core.make_parametrized_syndrome_network(
        check_inds,
        syndrome_param_inds,
    )

    # Independent noise model
    noise_tn = TensorNetwork([
        Tensor(
            data=np.array([1.0 - noise_p, noise_p], dtype=np.float64),
            inds=(error_inds[i],),
            tags=oset([f"NOISE_{i}", "NOISE"]),
        ) for i in range(num_errors)
    ])

    full_tn = TensorNetwork()
    full_tn = full_tn.combine(code_tn, virtual=True)
    full_tn = full_tn.combine(logical_tn, virtual=True)
    full_tn = full_tn.combine(param_syn_tn, virtual=True)
    full_tn = full_tn.combine(noise_tn, virtual=True)
    return full_tn, syndrome_param_inds, logical_obs_inds


def _assert_chain_only_has_syndrome_and_logical_inds(chain: Any) -> None:
    allowed = set(chain.syndrome_labels) | set(chain.logical_labels)
    site_labels = [site.label for site in chain.sites]
    unknown = [label for label in site_labels if label not in allowed]
    if unknown:
        raise RuntimeError(
            "1D chain contains non-syndrome/logical labels: "
            f"{sorted(set(str(x) for x in unknown))}"
        )


def _save_chain_numpy_bundle(chain: Any, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    chain.save(out_dir)

    site_labels = np.asarray([str(site.label) for site in chain.sites], dtype=np.str_)
    syndrome_labels = np.asarray(chain.syndrome_labels, dtype=np.str_)
    logical_labels = np.asarray(chain.logical_labels, dtype=np.str_)

    np.save(out_dir / "site_labels.npy", site_labels)
    np.save(out_dir / "syndrome_labels.npy", syndrome_labels)
    np.save(out_dir / "logical_labels.npy", logical_labels)


def export_isca_revision_1d_mps(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded, skipped = load_isca_revision_codes(
        seed=args.seed,
        code_dir=args.code_dir,
        strict_params=True,
        skip_missing=args.skip_missing,
    )

    manifest: dict[str, Any] = {
        "created_at_epoch_s": time.time(),
        "output_dir": str(output_dir),
        "bond_dim": int(args.bond_dim),
        "cutoff": float(args.cutoff),
        "max_steps": int(args.max_steps),
        "noise_p": float(args.noise_p),
        "seed": int(args.seed),
        "codes": [],
        "skipped": skipped,
    }

    for name, code in loaded:
        H = np.asarray(code.hz, dtype=np.float64)
        logical = np.asarray(code.lz, dtype=np.float64)
        if logical.ndim == 1:
            logical = logical.reshape(1, -1)
        if logical.ndim != 2 or logical.shape[1] != H.shape[1]:
            raise ValueError(
                f"{name}: logical shape {logical.shape} incompatible with H {H.shape}.")
        K = logical.shape[0]
        M = H.shape[0]
        if K <= 0:
            raise ValueError(f"{name}: no logical observables found.")

        if args.verbose:
            print(f"[compile] {name}: H={H.shape}, K={K}, M={M}, "
                  f"logical={logical.shape}, expected_sites={K+M}")

        # Build ONE multi-logical TN with K+M preserved indices
        full_tn, syndrome_param_inds, logical_obs_inds = build_full_multi_logical_tn(
            H=H,
            logical=logical,
            noise_p=args.noise_p,
        )

        if args.verbose:
            print(f"  Compiling single chain with {K} logical + {M} syndrome "
                  f"= {K+M} preserved sites...", flush=True)

        t0 = time.perf_counter()
        compile_result = core.compile_to_1d_chain(
            tn=full_tn.copy(),
            preserve_inds=set(syndrome_param_inds + logical_obs_inds),
            syndrome_inds=set(syndrome_param_inds),
            logical_inds=set(logical_obs_inds),
            chi=int(args.bond_dim),
            cutoff=float(args.cutoff),
            max_steps=int(args.max_steps),
            verbose=args.verbose,
        )
        compile_ms = (time.perf_counter() - t0) * 1e3

        code_out = output_dir / name
        code_out.mkdir(parents=True, exist_ok=True)

        if compile_result.chain is None:
            msg = f"{name}: failed to extract 1D chain from multi-logical TN."
            if args.skip_nonchain:
                manifest["skipped"].append(msg)
                if args.verbose:
                    print(f"  SKIP ({msg})")
                record = {
                    "name": name,
                    "N": int(code.N),
                    "K": K,
                    "D": float(code.D),
                    "status": "skipped",
                    "compile_ms": float(round(compile_ms, 3)),
                }
                manifest["codes"].append(record)
                continue
            raise RuntimeError(msg)

        chain = compile_result.chain
        _assert_chain_only_has_syndrome_and_logical_inds(chain)
        _save_chain_numpy_bundle(chain, code_out / "chain")

        stats_data = json.loads(compile_result.offline_stats.to_json())

        # Per-code manifest
        code_manifest = {
            "name": name,
            "K": K,
            "M": M,
            "num_sites": len(chain.sites),
            "syndrome_labels": chain.syndrome_labels,
            "logical_labels": chain.logical_labels,
            "site_labels": [s.label for s in chain.sites],
            "bond_dims": [s.tensor.shape[0] for s in chain.sites],
            "trunc_error": stats_data.get("truncation_error", 0.0),
            "compile_ms": float(round(compile_ms, 3)),
        }
        with open(code_out / "manifest.json", "w") as f:
            json.dump(code_manifest, f, indent=2)

        record = {
            "name": name,
            "N": int(code.N),
            "K": K,
            "M": M,
            "D": float(code.D),
            "source_path": getattr(code, "source_path", ""),
            "source_format": getattr(code, "source_format", ""),
            "source_family": getattr(code, "source_family", ""),
            "status": "ok",
            "compile_ms": float(round(compile_ms, 3)),
            "output_dir": str(code_out),
            "num_sites": len(chain.sites),
            "num_syndrome_sites": len(chain.syndrome_labels),
            "num_logical_sites": len(chain.logical_labels),
            "trunc_error": stats_data.get("truncation_error", 0.0),
            "offline_stats": stats_data,
        }
        manifest["codes"].append(record)
        if args.verbose:
            trunc = stats_data.get("truncation_error", 0.0)
            print(
                f"[saved] {name}: {len(chain.sites)} sites "
                f"({len(chain.syndrome_labels)} syn + {len(chain.logical_labels)} log), "
                f"trunc_err={trunc:.2e}, {compile_ms:.0f}ms")

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if args.verbose:
        print(f"[done] manifest={manifest_path}")
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile and export offline-compressed 1D MPS chains for the 9 ISCA "
            "revision benchmark codes. Uses a single multi-logical tensor network "
            "per code, producing one chain with K+M sites."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(WORKSPACE / "experiments" / "results" / "isca_revision_1d_mps"),
        help="Directory where per-code MPS chains and manifest are saved.",
    )
    parser.add_argument(
        "--bond-dim",
        type=int,
        default=32,
        help="Offline compression bond dimension (chi).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=1.0e-12,
        help="SVD cutoff used by offline compression.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100000,
        help="Maximum offline compression steps.",
    )
    parser.add_argument(
        "--noise-p",
        type=float,
        default=0.01,
        help="Independent physical noise probability for TN construction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used by code loaders that depend on seed.",
    )
    parser.add_argument(
        "--code-dir",
        type=str,
        default=None,
        help="Optional override for GND ldpc/qcc code file directory.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip codes with missing source files instead of failing.",
    )
    parser.add_argument(
        "--skip-nonchain",
        action="store_true",
        help="Skip codes that cannot be extracted to a 1D chain.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-code compile/export logs.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    export_isca_revision_1d_mps(parser.parse_args())
