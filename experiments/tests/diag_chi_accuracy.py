#!/usr/bin/env python3
"""Diagnostic: compare MPS vs exact TN per-shot probabilities at different chi.

Goal: understand how chi affects MPS accuracy on LDPC_25_3_4 (smallest code).
"""
from __future__ import annotations
import sys, time, math
from pathlib import Path
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent.parent
QEC_PY = WORKSPACE / "cudaqx" / "libs" / "qec" / "python"
sys.path.insert(0, str(WORKSPACE))
if str(QEC_PY) not in sys.path:
    sys.path.insert(0, str(QEC_PY))

import importlib.util

_TNU = QEC_PY / "cudaq_qec" / "plugins" / "decoders" / "tensor_network_utils"

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

core = _load_module("mps_decoder_core", _TNU / "mps_decoder_core.py")
factory = _load_module("tensor_network_factory", _TNU / "tensor_network_factory.py")

from quimb import oset
from quimb.tensor import Tensor, TensorNetwork

from experiments.codes.gnd_ldpc_codes import ldpc_25_3_4

# ---- Load code ----
code = ldpc_25_3_4()
H = code.hz.astype(np.float64)
lz = code.lz.astype(np.float64)
num_checks, num_errors = H.shape
K = lz.shape[0]
print(f"Code: LDPC_25_3_4  [[{code.N},{code.K},{code.D}]]")
print(f"H shape: {H.shape}, K={K}")

# ---- Sample errors ----
noise_p = 0.01
shots = 50
seed = 2026
rng = np.random.default_rng(seed)
errors = (rng.random((shots, num_errors)) < noise_p).astype(np.int8)
syndromes = (errors @ H.T.astype(np.int8)) % 2
true_logicals = (errors @ lz.T.astype(np.int8)) % 2  # (shots, K)

check_inds = [f"s_{i}" for i in range(num_checks)]
error_inds = [f"e_{i}" for i in range(num_errors)]

# ---- Exact TN probabilities (per logical, per shot) ----
print("\n--- Computing exact TN probabilities ---")
code_tn = factory.tensor_network_from_parity_check(
    H, col_inds=error_inds, row_inds=check_inds)
noise_ts = [Tensor(
    data=np.array([1.0 - noise_p, noise_p]),
    inds=(error_inds[j],), tags=oset([f"NOISE_{j}"]))
    for j in range(num_errors)]
noise_tn = TensorNetwork(noise_ts)

# Build K logical TNs
log_tns = []
for k in range(K):
    lk = lz[k:k+1]
    l_inds = ["l_0"]
    obs_inds = ["obs"]
    lt = factory.tensor_network_from_parity_check(
        lk, col_inds=error_inds, row_inds=l_inds, tags=["LOG_0"])
    lt = lt.combine(
        factory.tensor_network_from_logical_observable(
            lk, l_inds, obs_inds, ["LOG_0"]),
        virtual=True)
    log_tns.append(lt)

exact_probs = np.zeros((shots, K))  # P(obs=1 | syndrome) per shot per logical
t0 = time.time()
for i in range(shots):
    syn_ts = []
    for j in range(num_checks):
        s = float(syndromes[i, j])
        syn_ts.append(Tensor(
            data=s * np.array([1.0, -1.0]) + (1.0 - s) * np.array([1.0, 1.0]),
            inds=(check_inds[j],), tags=oset([f"SYN_{j}"])))
    syn_tn = TensorNetwork(syn_ts)

    for k in range(K):
        full = TensorNetwork()
        full = full.combine(code_tn, virtual=True)
        full = full.combine(log_tns[k], virtual=True)
        full = full.combine(syn_tn, virtual=True)
        full = full.combine(noise_tn, virtual=True)
        val = full.contract(output_inds=("obs",))
        arr = np.asarray(val.data if hasattr(val, 'data') else val, dtype=np.float64).ravel()
        total = float(arr[0] + arr[1])
        p1 = float(arr[1] / total) if total > 0 else 0.5
        exact_probs[i, k] = p1
exact_time = time.time() - t0
print(f"Exact TN: {exact_time:.1f}s for {shots} shots x {K} logicals")

exact_preds = (exact_probs >= 0.5).astype(np.int8)
exact_wrong = np.any(exact_preds != true_logicals, axis=1)
exact_ler = float(np.mean(exact_wrong))
print(f"Exact TN block_LER: {exact_ler:.4f}")

# ---- MPS probabilities at different chi ----
chi_values = [16, 32, 64, 128, 256]

for chi in chi_values:
    print(f"\n--- MPS chi={chi} ---")

    # Build K chains (one per logical)
    chains = []
    t0 = time.time()
    for k in range(K):
        logical_row = lz[k]
        syn_param_inds = [f"syn_param_{j}" for j in range(num_checks)]
        logical_obs_inds = ["obs"]

        code_tn_k = factory.tensor_network_from_parity_check(
            H, col_inds=error_inds, row_inds=check_inds)
        logical_2d = logical_row.reshape(1, -1).astype(np.float64)
        l_inds_k = ["l_0"]
        logical_tn_k = factory.tensor_network_from_parity_check(
            logical_2d, col_inds=error_inds, row_inds=l_inds_k, tags=["LOG_0"])
        logical_tn_k = logical_tn_k.combine(
            factory.tensor_network_from_logical_observable(
                logical_2d, l_inds_k, logical_obs_inds, ["LOG_0"]),
            virtual=True)
        param_syn_tn = core.make_parametrized_syndrome_network(check_inds, syn_param_inds)
        noise_ts_k = [Tensor(
            data=np.array([1.0 - noise_p, noise_p], dtype=np.float64),
            inds=(error_inds[j],), tags=oset([f"NOISE_{j}", "NOISE"]))
            for j in range(num_errors)]
        noise_tn_k = TensorNetwork(noise_ts_k)

        full_tn = TensorNetwork()
        full_tn = full_tn.combine(code_tn_k, virtual=True)
        full_tn = full_tn.combine(logical_tn_k, virtual=True)
        full_tn = full_tn.combine(param_syn_tn, virtual=True)
        full_tn = full_tn.combine(noise_tn_k, virtual=True)

        preserve_inds = set(syn_param_inds + logical_obs_inds)
        result = core.compile_to_1d_chain(
            tn=full_tn.copy(),
            preserve_inds=preserve_inds,
            syndrome_inds=set(syn_param_inds),
            logical_inds=set(logical_obs_inds),
            chi=chi,
            verbose=(k == 0),
        )
        chains.append((result.chain, result.compressed_tn, result.offline_stats))
    compile_time = time.time() - t0
    print(f"  Compile time: {compile_time:.1f}s for {K} logicals")

    # Print offline stats for first logical
    st = chains[0][2]
    if hasattr(st, 'truncation_error'):
        print(f"  Truncation error (logical 0): {st.truncation_error:.6e}")
    if hasattr(st, 'max_bond_dim'):
        print(f"  Max bond dim: {st.max_bond_dim}")

    # Decode
    mps_probs = np.zeros((shots, K))
    t1 = time.time()
    for i in range(shots):
        for k in range(K):
            ch, ctn, _ = chains[k]
            syn_vals = {f"syn_param_{j}": float(syndromes[i, j]) for j in range(num_checks)}
            if ch is not None:
                _, marginals = ch.decode_logicals(syndrome_values=syn_vals, use_gpu=False)
                mps_probs[i, k] = float(marginals.get("obs", 0.5))
            elif ctn is not None:
                _, marginals = ctn.decode_logicals(syndrome_values=syn_vals)
                mps_probs[i, k] = float(marginals.get("obs", 0.5))
            else:
                mps_probs[i, k] = 0.5
    decode_time = time.time() - t1
    print(f"  Decode time: {decode_time:.1f}s for {shots} shots x {K} logicals")

    mps_preds = (mps_probs >= 0.5).astype(np.int8)
    mps_wrong = np.any(mps_preds != true_logicals, axis=1)
    mps_ler = float(np.mean(mps_wrong))
    print(f"  MPS block_LER: {mps_ler:.4f}")

    # Per-shot comparison with exact
    prob_diff = np.abs(mps_probs - exact_probs)
    print(f"  Probability diff vs exact: mean={prob_diff.mean():.6f}, max={prob_diff.max():.6f}")

    # Show first few shots detail
    disagree = np.where(np.any(mps_preds != exact_preds, axis=1))[0]
    print(f"  Shots where MPS disagrees with exact: {len(disagree)}/{shots}")
    if len(disagree) > 0:
        for idx in disagree[:5]:
            print(f"    shot {idx}: true={true_logicals[idx].tolist()}, "
                  f"exact_p={exact_probs[idx].round(4).tolist()}, "
                  f"mps_p={mps_probs[idx].round(4).tolist()}")

print("\n=== Summary ===")
print(f"{'chi':>6} | {'block_LER':>10} | {'mean_prob_diff':>15} | {'max_prob_diff':>15} | {'disagree':>10}")
print("-" * 70)
# Re-compute for summary
for chi_idx, chi in enumerate(chi_values):
    # We already computed - just re-derive from the output above
    pass
print("(See detailed output above)")
