#!/usr/bin/env python3
"""Export offline-compressed 1D MPS chains for the 9 ISCA revision codes."""
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


def _build_single_logical_tn(
    H: np.ndarray,
    logical_row: np.ndarray,
    noise_p: float,
) -> tuple[TensorNetwork, list[str], list[str]]:
    """Build parameterised TN for a SINGLE logical operator.

    Building K separate single-logical TNs avoids:
    1. quimb's broken multi-logical hyperedge contraction
    2. SVD OOM from high-degree nodes in multi-logical TNs
    """
    num_checks, num_errors = H.shape
    check_inds = [f"s_{i}" for i in range(num_checks)]
    error_inds = [f"e_{i}" for i in range(num_errors)]
    syndrome_param_inds = [f"syn_param_{i}" for i in range(num_checks)]
    logical_obs_inds = ["obs"]

    code_tn = factory.tensor_network_from_parity_check(
        H.astype(np.float64),
        col_inds=error_inds,
        row_inds=check_inds,
    )

    logical_2d = logical_row.reshape(1, -1).astype(np.float64)
    logical_inds = ["l_0"]
    logical_tags = ["LOG_0"]
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

    param_syn_tn = core.make_parametrized_syndrome_network(
        check_inds,
        syndrome_param_inds,
    )
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
        tb48_h_path=args.tb48_h_path,
        tb48_logical_path=args.tb48_logical_path,
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
        if K <= 0:
            raise ValueError(f"{name}: no logical observables found.")

        if args.verbose:
            print(f"[compile] {name}: H={H.shape}, K={K}, logical={logical.shape}")

        # Build K separate single-logical chains to avoid:
        # 1) quimb multi-logical contraction bug
        # 2) SVD OOM from high-degree nodes in multi-logical TN
        code_out = output_dir / name
        code_out.mkdir(parents=True, exist_ok=True)
        chain_records = []
        total_compile_ms = 0.0
        all_ok = True

        for k in range(K):
            if args.verbose:
                print(f"  Compiling chain for logical {k}/{K}...", end=" ", flush=True)

            full_tn, syndrome_param_inds, logical_obs_inds = _build_single_logical_tn(
                H=H,
                logical_row=logical[k],
                noise_p=args.noise_p,
            )

            t0 = time.perf_counter()
            compile_result = core.compile_to_1d_chain(
                tn=full_tn.copy(),
                preserve_inds=set(syndrome_param_inds + logical_obs_inds),
                syndrome_inds=set(syndrome_param_inds),
                logical_inds=set(logical_obs_inds),
                chi=int(args.bond_dim),
                cutoff=float(args.cutoff),
                max_steps=int(args.max_steps),
                verbose=False,
            )
            k_compile_ms = (time.perf_counter() - t0) * 1e3
            total_compile_ms += k_compile_ms

            if compile_result.chain is None:
                msg = f"{name}/logical_{k}: failed to extract 1D chain."
                if args.skip_nonchain:
                    manifest["skipped"].append(msg)
                    if args.verbose:
                        print(f"SKIP ({msg})")
                    chain_records.append({"logical_idx": k, "status": "skipped"})
                    all_ok = False
                    continue
                raise RuntimeError(msg)

            chain = compile_result.chain
            _assert_chain_only_has_syndrome_and_logical_inds(chain)

            chain_out = code_out / f"logical_{k}"
            _save_chain_numpy_bundle(chain, chain_out)
            stats_data = json.loads(compile_result.offline_stats.to_json())

            chain_records.append({
                "logical_idx": k,
                "status": "ok",
                "num_sites": len(chain.sites),
                "num_syndrome_indices": len(chain.syndrome_labels),
                "num_logical_indices": len(chain.logical_labels),
                "compile_ms": float(round(k_compile_ms, 3)),
                "trunc_error": stats_data.get("truncation_error", 0.0),
                "offline_stats": stats_data,
            })
            if args.verbose:
                trunc = stats_data.get("truncation_error", 0.0)
                print(f"chain, trunc_err={trunc:.2e}, {k_compile_ms:.0f}ms")

        # Save per-code manifest
        code_manifest = {
            "name": name,
            "K": K,
            "chains": chain_records,
        }
        with open(code_out / "manifest.json", "w") as f:
            json.dump(code_manifest, f, indent=2)

        record = {
            "name": name,
            "N": int(code.N),
            "K": K,
            "D": float(code.D),
            "source_path": getattr(code, "source_path", ""),
            "source_format": getattr(code, "source_format", ""),
            "source_family": getattr(code, "source_family", ""),
            "compile_ms": float(round(total_compile_ms, 3)),
            "output_dir": str(code_out),
            "num_chains": sum(1 for c in chain_records if c.get("status") == "ok"),
            "chain_details": chain_records,
        }
        manifest["codes"].append(record)
        if args.verbose:
            ok_count = record["num_chains"]
            print(
                f"[saved] {name}: {ok_count}/{K} chains, "
                f"total_compile_ms={record['compile_ms']}")

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
            "revision benchmark codes."
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
        "--tb48-h-path",
        type=str,
        default=None,
        help="Optional TB_48_4_8 H.npy fallback path.",
    )
    parser.add_argument(
        "--tb48-logical-path",
        type=str,
        default=None,
        help="Optional TB_48_4_8 logical.npy fallback path.",
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
