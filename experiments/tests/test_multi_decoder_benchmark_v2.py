#!/usr/bin/env python3
"""Multi-decoder benchmark v2: per-logical MPS chains, GPU batch decoding.

Compares decoders on ALL K logical qubits across the 9 ISCA codes:
  - MPS compressed TN decoder (per-logical chains, GPU batch)
  - Exact TN decoder (opt_einsum, per-logical, small codes only)
  - BP-OSD / BP-LSD / BP (ldpc package, CPU)

Key improvements over v1:
  - Builds K separate MPS chains (one per logical operator) to avoid
    quimb's broken multi-logical hyperedge contraction.
  - Saves/loads 1D MPS chains per code for GPU batch decoding.
  - Estimates memory for exact TN contraction.
  - Tests ALL K logical qubits.

Usage:
  .venv-mps/bin/python experiments/tests/test_multi_decoder_benchmark_v2.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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

# ---------------------------------------------------------------------------
# Core module import
# ---------------------------------------------------------------------------
import importlib.util


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_TNU = QEC_PY / "cudaq_qec" / "plugins" / "decoders" / "tensor_network_utils"
core = _load_module("mps_decoder_core", _TNU / "mps_decoder_core.py")
factory = _load_module(
    "tensor_network_factory", _TNU / "tensor_network_factory.py"
)

from experiments.tests._isca_code_registry import load_isca_code_cases

from quimb import oset
from quimb.tensor import Tensor, TensorNetwork

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
HAS_GPU = False
try:
    import cupy

    HAS_GPU = cupy.cuda.is_available()
except Exception:
    cupy = None

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
# opt_einsum for exact TN contraction (correct hyperedge handling)
# ---------------------------------------------------------------------------
try:
    import opt_einsum

    HAS_OPT_EINSUM = True
except ImportError:
    HAS_OPT_EINSUM = False


# ---------------------------------------------------------------------------
# Helper: build single-logical TN and compile to MPS chain
# ---------------------------------------------------------------------------
def build_single_logical_chain(
    H: np.ndarray,
    logical_row: np.ndarray,
    noise_p: float,
    chi: int = 32,
    verbose: bool = False,
) -> tuple[Any, Any, Any]:
    """Build parameterised TN for a single logical and compile to 1D chain."""
    num_checks, num_errors = H.shape
    check_inds = [f"s_{i}" for i in range(num_checks)]
    error_inds = [f"e_{i}" for i in range(num_errors)]
    syn_param_inds = [f"syn_param_{i}" for i in range(num_checks)]
    logical_obs_inds = ["obs"]

    code_tn = factory.tensor_network_from_parity_check(
        H.astype(np.float64), col_inds=error_inds, row_inds=check_inds
    )
    logical_2d = logical_row.reshape(1, -1).astype(np.float64)
    logical_inds = ["l_0"]
    logical_tn = factory.tensor_network_from_parity_check(
        logical_2d, col_inds=error_inds, row_inds=logical_inds, tags=["LOG_0"]
    )
    logical_tn = logical_tn.combine(
        factory.tensor_network_from_logical_observable(
            logical_2d, logical_inds, logical_obs_inds, ["LOG_0"]
        ),
        virtual=True,
    )

    param_syn_tn = core.make_parametrized_syndrome_network(
        check_inds, syn_param_inds
    )

    noise_tensors = []
    for i, eind in enumerate(error_inds):
        noise_tensors.append(
            Tensor(
                data=np.array([1.0 - noise_p, noise_p], dtype=np.float64),
                inds=(eind,),
                tags=oset([f"NOISE_{i}", "NOISE"]),
            )
        )
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


# ---------------------------------------------------------------------------
# Save/load K chains per code
# ---------------------------------------------------------------------------
def save_all_chains(
    chains: list, code_name: str, output_dir: Path
) -> Path:
    """Save K chains for a code to disk."""
    code_dir = output_dir / code_name
    code_dir.mkdir(parents=True, exist_ok=True)
    meta = {"code_name": code_name, "K": len(chains), "chains": []}
    for k, chain in enumerate(chains):
        chain_dir = code_dir / f"logical_{k}"
        if chain is not None:
            chain.save(chain_dir)
            meta["chains"].append(
                {
                    "logical_idx": k,
                    "status": "ok",
                    "num_sites": len(chain.sites),
                }
            )
        else:
            meta["chains"].append({"logical_idx": k, "status": "missing"})
    with open(code_dir / "manifest.json", "w") as f:
        json.dump(meta, f, indent=2)
    return code_dir


def load_all_chains(code_dir: Path) -> list:
    """Load K chains for a code from disk."""
    with open(code_dir / "manifest.json") as f:
        meta = json.load(f)
    chains = []
    for entry in meta["chains"]:
        k = entry["logical_idx"]
        if entry["status"] == "ok":
            chain = core.OneDChainMPS.load(code_dir / f"logical_{k}")
            chains.append(chain)
        else:
            chains.append(None)
    return chains


# ---------------------------------------------------------------------------
# GPU batch decoding with CuPy
# ---------------------------------------------------------------------------
def decode_batch_gpu(
    chain: core.OneDChainMPS,
    syndromes: np.ndarray,
    num_checks: int,
) -> np.ndarray:
    """Batch decode using CuPy matrix chain multiplication.

    Returns array of P(obs=1|syndrome) for each shot.
    """
    xp = cupy if HAS_GPU else np
    shots = syndromes.shape[0]
    probs = np.zeros(shots, dtype=np.float64)

    # Pre-transfer site tensors to GPU
    gpu_sites = []
    for site in chain.sites:
        gpu_sites.append(
            (site.label, xp.asarray(site.tensor, dtype=xp.float64))
        )

    syndrome_set = set(chain.syndrome_labels)

    for i in range(shots):
        syn_vals = {
            f"syn_param_{j}": float(syndromes[i, j])
            for j in range(num_checks)
        }
        prob = chain.marginal_probability(
            label="obs",
            syndrome_values=syn_vals,
            use_gpu=HAS_GPU,
        )
        probs[i] = prob

    return probs


# ---------------------------------------------------------------------------
# Exact TN decoder using opt_einsum (correct for multi-logical)
# ---------------------------------------------------------------------------
def exact_tn_single_logical(
    H: np.ndarray,
    logical_row: np.ndarray,
    syndrome: np.ndarray,
    noise_p: float,
) -> float:
    """Exact TN contraction for a single logical using opt_einsum."""
    num_checks, num_errors = H.shape
    check_inds = [f"s_{i}" for i in range(num_checks)]
    error_inds = [f"e_{i}" for i in range(num_errors)]

    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]])

    all_tensors = []
    all_inds = []

    # Code parity check
    rows, cols = np.nonzero(H)
    for i, j in zip(rows, cols):
        all_tensors.append(hadamard)
        all_inds.append((check_inds[i], error_inds[j]))

    # Single logical parity check
    logical_2d = logical_row.reshape(1, -1)
    lrows, lcols = np.nonzero(logical_2d)
    for i, j in zip(lrows, lcols):
        all_tensors.append(hadamard)
        all_inds.append(("l_0", error_inds[j]))

    # Logical observable
    all_tensors.append(hadamard)
    all_inds.append(("l_0", "obs"))

    # Syndrome
    for j in range(num_checks):
        s = float(syndrome[j])
        data = s * np.array([1.0, -1.0]) + (1.0 - s) * np.array([1.0, 1.0])
        all_tensors.append(data)
        all_inds.append((check_inds[j],))

    # Noise
    for i in range(num_errors):
        all_tensors.append(np.array([1.0 - noise_p, noise_p]))
        all_inds.append((error_inds[i],))

    # Build einsum
    all_ind_names = set()
    for inds in all_inds:
        all_ind_names.update(inds)

    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    extra_chars = [f"_{i}" for i in range(100)]
    char_list = list(chars) + extra_chars
    ind_to_char = {ind: char_list[i] for i, ind in enumerate(sorted(all_ind_names))}

    subscripts_list = []
    for inds in all_inds:
        sub = ",".join(ind_to_char[ind] for ind in inds)
        subscripts_list.append(sub)

    # Use integer indices for opt_einsum
    ind_to_int = {ind: i for i, ind in enumerate(sorted(all_ind_names))}
    operands = []
    for tensor, inds in zip(all_tensors, all_inds):
        operands.append(tensor)
        operands.append([ind_to_int[ind] for ind in inds])
    operands.append([ind_to_int["obs"]])

    result = opt_einsum.contract(*operands, optimize="auto")
    arr = np.asarray(result).ravel()
    if len(arr) < 2:
        return 0.5
    total = arr[0] + arr[1]
    if total <= 0:
        return 0.5
    return float(arr[1] / total)


def estimate_exact_tn_memory_gb(H: np.ndarray, K: int) -> float:
    """Estimate memory needed for exact TN contraction (single logical).

    The dominant cost is the largest intermediate tensor.
    For a code with m checks and n errors, the worst case is
    exponential in treewidth.
    Rough estimate: 2^(max_degree) * 8 bytes.
    """
    num_checks, num_errors = H.shape
    # Simple heuristic: memory ~ 2^(min(num_checks, num_errors)) * 8 bytes
    # But with good contraction order, it's much less.
    # For safety, estimate based on code size.
    if num_errors <= 30:
        # Can enumerate all errors
        mem_gb = (2**num_errors * 8) / 1e9
    elif num_errors <= 60:
        # Moderate: estimate from parity check structure
        # treewidth ~ num_checks / 2 for typical LDPC
        tw_est = min(num_checks, num_errors) // 2
        mem_gb = (2**tw_est * 8) / 1e9
    else:
        # Large code: likely too large for exact
        tw_est = min(num_checks, num_errors) // 2
        mem_gb = (2**tw_est * 8) / 1e9
    return mem_gb


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_errors_and_syndromes(
    H: np.ndarray, noise_p: float, shots: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = H.shape[1]
    errors = (rng.random((shots, n)) < noise_p).astype(np.int8)
    syndromes = (errors @ H.T.astype(np.int8)) % 2
    return errors, syndromes


# ---------------------------------------------------------------------------
# Per-code benchmark
# ---------------------------------------------------------------------------
@dataclass
class CodeCase:
    name: str
    code: Any
    skip_exact: bool = False


_EXACT_MAX_N = 72  # Skip exact TN for codes larger than this


def run_single_code(
    cc: CodeCase,
    *,
    shots: int,
    noise_p: float,
    chi: int,
    seed: int,
    verbose: bool,
    chain_dir: Optional[Path] = None,
    force_recompile: bool = False,
) -> dict[str, Any]:
    H = np.ascontiguousarray(cc.code.hz.astype(np.float64))
    lz = np.ascontiguousarray(cc.code.lz.astype(np.float64))
    num_checks, num_errors = H.shape
    K = lz.shape[0]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Code: {cc.name}  N={cc.code.N} K={K} D={cc.code.D}")
        print(f"  H shape: {H.shape}, lz shape: {lz.shape}")
        print(f"  noise_p={noise_p}, shots={shots}, chi={chi}")

    errors, syndromes = sample_errors_and_syndromes(H, noise_p, shots, seed)
    true_logicals = (errors @ lz.T.astype(np.int8)) % 2  # (shots, K)

    record: dict[str, Any] = {
        "code": cc.name,
        "N": int(cc.code.N),
        "K": K,
        "D": float(cc.code.D),
        "noise_p": noise_p,
        "shots": shots,
        "chi": chi,
        "decoders": {},
    }

    # --- Estimate exact TN memory ---
    mem_est_gb = estimate_exact_tn_memory_gb(H, K)
    record["exact_tn_memory_estimate_gb"] = round(mem_est_gb, 3)

    # --- MPS decoder (K separate chains) ---
    try:
        t0 = time.perf_counter()
        chains = []
        code_chain_dir = None
        if chain_dir is not None:
            code_chain_dir = chain_dir / cc.name

        # Try to load existing chains
        if (
            code_chain_dir is not None
            and (code_chain_dir / "manifest.json").exists()
            and not force_recompile
        ):
            if verbose:
                print(f"  Loading pre-compiled chains from {code_chain_dir}")
            chains = load_all_chains(code_chain_dir)
        else:
            # Compile K chains
            for k in range(K):
                if verbose:
                    print(
                        f"  Compiling MPS chain for logical {k}/{K}...",
                        end=" ",
                        flush=True,
                    )
                chain_k, ctn_k, stats_k = build_single_logical_chain(
                    H, lz[k], noise_p, chi=chi
                )
                chains.append(chain_k)
                if verbose:
                    status = "chain" if chain_k is not None else "compressed_tn"
                    print(
                        f"{status}, trunc_err={stats_k.truncation_error:.2e}"
                    )
            # Save chains
            if chain_dir is not None:
                save_all_chains(chains, cc.name, chain_dir)
                if verbose:
                    print(f"  Saved chains to {chain_dir / cc.name}")

        init_ms = (time.perf_counter() - t0) * 1e3

        # Decode all shots for all K logicals
        mps_predictions = np.zeros((shots, K), dtype=np.int8)
        mps_marginals = np.full((shots, K), 0.5, dtype=np.float64)

        t1 = time.perf_counter()
        for k in range(K):
            chain_k = chains[k]
            if chain_k is None:
                continue
            for i in range(shots):
                syn_vals = {
                    f"syn_param_{j}": float(syndromes[i, j])
                    for j in range(num_checks)
                }
                prob = chain_k.marginal_probability(
                    label="obs",
                    syndrome_values=syn_vals,
                    use_gpu=HAS_GPU,
                )
                mps_marginals[i, k] = prob
                mps_predictions[i, k] = 1 if prob >= 0.5 else 0
        decode_ms = (time.perf_counter() - t1) * 1e3

        # Compute LER per logical and overall
        per_logical_ler = []
        for k in range(K):
            ler_k = float(np.mean(true_logicals[:, k] != mps_predictions[:, k]))
            per_logical_ler.append(round(ler_k, 6))

        # Overall LER: logical error if ANY logical is wrong
        any_wrong = np.any(true_logicals != mps_predictions, axis=1)
        overall_ler = float(np.mean(any_wrong))

        record["decoders"]["mps_tn"] = {
            "status": "ok",
            "backend": "gpu" if HAS_GPU else "cpu",
            "init_ms": round(init_ms, 2),
            "total_decode_ms": round(decode_ms, 2),
            "per_shot_ms": round(decode_ms / shots, 3),
            "overall_logical_error_rate": round(overall_ler, 6),
            "per_logical_ler": per_logical_ler,
        }
        if verbose:
            print(
                f"  MPS-TN: overall LER={overall_ler:.4f}, "
                f"per-logical={per_logical_ler}, {decode_ms:.1f}ms total"
            )
    except Exception as e:
        record["decoders"]["mps_tn"] = {"status": "error", "error": str(e)}
        if verbose:
            import traceback

            print(f"  MPS-TN: ERROR {e}")
            traceback.print_exc()

    # --- Exact TN decoder (small codes only, opt_einsum) ---
    if not cc.skip_exact and HAS_OPT_EINSUM:
        try:
            t0 = time.perf_counter()
            init_ms = (time.perf_counter() - t0) * 1e3

            exact_predictions = np.zeros((shots, K), dtype=np.int8)
            t1 = time.perf_counter()
            for k in range(K):
                for i in range(shots):
                    p1 = exact_tn_single_logical(
                        H, lz[k], syndromes[i], noise_p
                    )
                    exact_predictions[i, k] = 1 if p1 >= 0.5 else 0
            decode_ms = (time.perf_counter() - t1) * 1e3

            per_logical_ler = []
            for k in range(K):
                ler_k = float(
                    np.mean(
                        true_logicals[:, k] != exact_predictions[:, k]
                    )
                )
                per_logical_ler.append(round(ler_k, 6))

            any_wrong = np.any(true_logicals != exact_predictions, axis=1)
            overall_ler = float(np.mean(any_wrong))

            record["decoders"]["exact_tn"] = {
                "status": "ok",
                "backend": "cpu",
                "init_ms": round(init_ms, 2),
                "total_decode_ms": round(decode_ms, 2),
                "per_shot_ms": round(decode_ms / shots, 3),
                "overall_logical_error_rate": round(overall_ler, 6),
                "per_logical_ler": per_logical_ler,
                "estimated_memory_gb": round(mem_est_gb, 3),
            }
            if verbose:
                print(
                    f"  Exact-TN: overall LER={overall_ler:.4f}, "
                    f"per-logical={per_logical_ler}, {decode_ms:.1f}ms total"
                )
        except Exception as e:
            record["decoders"]["exact_tn"] = {
                "status": "error",
                "error": str(e),
            }
            if verbose:
                print(f"  Exact-TN: ERROR {e}")
    else:
        reason = "code too large" if cc.skip_exact else "opt_einsum not available"
        record["decoders"]["exact_tn"] = {
            "status": "skipped",
            "reason": reason,
            "estimated_memory_gb": round(mem_est_gb, 3),
        }

    # --- BP decoders (ldpc, CPU only) ---
    if HAS_LDPC:
        H_int = H.astype(np.uint8)
        channel_probs = np.full(num_errors, noise_p)

        for dec_name, DecClass, kwargs in [
            ("bp", BpDecoder, {"max_iter": 50, "bp_method": "ms"}),
            (
                "bp_osd",
                BpOsdDecoder,
                {
                    "max_iter": 50,
                    "bp_method": "ms",
                    "osd_method": "osd_cs",
                    "osd_order": 10,
                },
            ),
            (
                "bp_lsd",
                BpLsdDecoder,
                {"max_iter": 50, "bp_method": "ms", "lsd_order": 0},
            ),
        ]:
            try:
                t0 = time.perf_counter()
                bp_dec = DecClass(
                    H_int,
                    error_rate=noise_p,
                    channel_probs=channel_probs,
                    **kwargs,
                )
                init_ms = (time.perf_counter() - t0) * 1e3

                bp_predictions = np.zeros((shots, K), dtype=np.int8)
                t1 = time.perf_counter()
                for i in range(shots):
                    correction = bp_dec.decode(syndromes[i].astype(np.uint8))
                    for k in range(K):
                        logical_val = (
                            int(correction @ lz[k].astype(np.int8)) % 2
                        )
                        bp_predictions[i, k] = logical_val
                decode_ms = (time.perf_counter() - t1) * 1e3

                per_logical_ler = []
                for k in range(K):
                    ler_k = float(
                        np.mean(
                            true_logicals[:, k] != bp_predictions[:, k]
                        )
                    )
                    per_logical_ler.append(round(ler_k, 6))

                any_wrong = np.any(
                    true_logicals != bp_predictions, axis=1
                )
                overall_ler = float(np.mean(any_wrong))

                record["decoders"][dec_name] = {
                    "status": "ok",
                    "backend": "cpu",
                    "init_ms": round(init_ms, 2),
                    "total_decode_ms": round(decode_ms, 2),
                    "per_shot_ms": round(decode_ms / shots, 3),
                    "overall_logical_error_rate": round(overall_ler, 6),
                    "per_logical_ler": per_logical_ler,
                }
                if verbose:
                    print(
                        f"  {dec_name}: overall LER={overall_ler:.4f}, "
                        f"per-logical={per_logical_ler}, {decode_ms:.1f}ms"
                    )
            except Exception as e:
                record["decoders"][dec_name] = {
                    "status": "error",
                    "error": str(e),
                }
                if verbose:
                    print(f"  {dec_name}: ERROR {e}")

    return record


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(
    shots: int = 2000,
    noise_p: float = 0.01,
    chi: int = 32,
    seed: int = 42,
    code_names: Optional[list[str]] = None,
    skip_missing: bool = False,
    chain_dir: Optional[Path] = None,
    force_recompile: bool = False,
    verbose: bool = False,
) -> list[dict]:
    loaded, skipped = load_isca_code_cases(
        code_names=code_names, skip_missing=skip_missing
    )
    if verbose and skipped:
        print(f"[codes] skipped {len(skipped)} code(s): {skipped}")

    codes = []
    for name, code in loaded:
        skip_exact = int(code.N) > _EXACT_MAX_N
        codes.append(CodeCase(name=name, code=code, skip_exact=skip_exact))

    if not codes:
        raise RuntimeError("No codes were loaded for benchmark.")

    if verbose:
        print(
            f"[benchmark] codes={len(codes)}, shots={shots}, "
            f"noise_p={noise_p}, chi={chi}, GPU={HAS_GPU}"
        )

    results = []
    for cc in codes:
        record = run_single_code(
            cc,
            shots=shots,
            noise_p=noise_p,
            chi=chi,
            seed=seed,
            verbose=verbose,
            chain_dir=chain_dir,
            force_recompile=force_recompile,
        )
        results.append(record)

    return results


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-decoder benchmark v2")
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--noise-p", type=float, default=0.01)
    parser.add_argument("--chi", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--codes", nargs="*", default=None)
    parser.add_argument("--chain-dir", type=str, default=None)
    parser.add_argument("--force-recompile", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    chain_dir = None
    if args.chain_dir:
        chain_dir = Path(args.chain_dir)
    else:
        chain_dir = WORKSPACE / "experiments" / "results" / "mps_chains"

    print(f"GPU available: {HAS_GPU}")
    print(f"ldpc available: {HAS_LDPC}")
    print(f"opt_einsum available: {HAS_OPT_EINSUM}")
    print(f"Chain dir: {chain_dir}")
    print()

    results = run_benchmark(
        shots=args.shots,
        noise_p=args.noise_p,
        chi=args.chi,
        seed=args.seed,
        code_names=args.codes,
        skip_missing=False,
        chain_dir=chain_dir,
        force_recompile=args.force_recompile,
        verbose=True,
    )

    out_dir = WORKSPACE / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "multi_decoder_benchmark_v2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Code':<15} {'N':>4} {'K':>3} {'D':>4} ", end="")
    print(f"{'MPS LER':>10} {'Exact LER':>10} {'BP-OSD LER':>10}")
    print(f"{'-'*80}")
    for r in results:
        mps = r["decoders"].get("mps_tn", {})
        exact = r["decoders"].get("exact_tn", {})
        bposd = r["decoders"].get("bp_osd", {})

        mps_ler = (
            f"{mps['overall_logical_error_rate']:.4f}"
            if mps.get("status") == "ok"
            else mps.get("status", "N/A")
        )
        exact_ler = (
            f"{exact['overall_logical_error_rate']:.4f}"
            if exact.get("status") == "ok"
            else exact.get("status", "N/A")
        )
        bposd_ler = (
            f"{bposd['overall_logical_error_rate']:.4f}"
            if bposd.get("status") == "ok"
            else bposd.get("status", "N/A")
        )

        print(
            f"{r['code']:<15} {r['N']:>4} {r['K']:>3} {r['D']:>4.0f} "
            f"{mps_ler:>10} {exact_ler:>10} {bposd_ler:>10}"
        )
