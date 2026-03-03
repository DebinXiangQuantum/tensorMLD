"""Character Expansion (GF(2) Fourier) exact MLD decoder.

For a CSS quantum code with parity check H (M×N, binary) and logical operators L (K×N):

  P(L_k=ℓ | syndrome s) ∝ Σ_{e: He=s, Le_k=ℓ} P(e)

This is computed exactly via the GF(2) Fourier / Walsh-Hadamard transform:

  Z(s, ℓ₀,...,ℓ_{K-1}) = (1/2^M) Σ_{t∈{0,1}^M} (-1)^{t·s} · F_L(t, ℓ) · Φ(H^T t)

where:
  Φ(v) = Π_j [(1-p) + p·(-1)^{v_j}]   (noise characteristic function)
  F_L   = restriction to a specific logical sector

For depolarizing X-error model: P(e_j=1) = p independently.

Complexity: O(2^M · N) — practical for M ≤ ~20 (or M_eff ≤ ~22 with erasure).
  - QCC_18:   M=7  → 128 terms     (fastest)
  - LDPC_25:  M=10 → 1024 terms    (fast)
  - LDPC_30:  M=11 → 2048 terms    (fast)
  - TOR_50:   M=24 → 16M terms     (slow but feasible with numpy vectorization)
  - QCC_60+:  M=26+→ >64M terms    (impractical without erasure reduction)

Erasure support: When qubit erasure locations are known (p_j=1/2 for erased qubits),
the Fourier sum collapses to the null space of H_E^T, giving M_eff = M - rank(H_E) ≤ M.
This can make otherwise intractable codes feasible.

This serves as an exact reference / oracle decoder to benchmark other decoders.
The computation is naturally parallelizable over the 2^M terms (GPU via CuPy optional).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Optional GPU support via CuPy (drop-in numpy replacement on CUDA)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def _xp(use_gpu: bool):
    """Return cupy or numpy depending on availability and flag."""
    if use_gpu and CUPY_AVAILABLE:
        return cp
    return np


# ---------------------------------------------------------------------------
# Core character expansion
# ---------------------------------------------------------------------------

def _char_expansion_exact(
        H: np.ndarray,
        logicals: np.ndarray,
        noise_p: float,
        syndromes: np.ndarray,
        batch_size: int = 1 << 18,  # 256k rows per vectorized batch
) -> np.ndarray:
    """Exact MLD via character expansion for all syndromes simultaneously.

    Args:
        H:          Parity check matrix (M x N, binary uint8/int).
        logicals:   Logical operator matrix (K x N, binary).
        noise_p:    Physical bit-flip probability per qubit.
        syndromes:  Observed syndromes (n_shots x M, binary).
        batch_size: Number of t-vectors processed per vectorized batch.

    Returns:
        log_marginals: (n_shots, K, 2) array of log-probabilities.
            log_marginals[shot, k, ℓ] = log P(L_k=ℓ | syndrome[shot]).
    """
    M, N = H.shape
    K = logicals.shape[0]
    n_shots = syndromes.shape[0]
    total_t = 1 << M  # 2^M

    if total_t > (1 << 26):
        raise ValueError(
            f"M={M} requires 2^M={total_t} terms — too large for exact computation. "
            "Use M ≤ 20 for practical runtime.")

    # Precompute Φ values: for each t∈{0,1}^M, Φ(H^T t) = Π_j [(1-p)+p(-1)^{(H^T t)_j}]
    # (1-2p)^{popcount(H^T t)} for uniform p
    bias = 1.0 - 2.0 * noise_p

    # Enumerate all t ∈ {0,1}^M and compute Φ(H^T t) in batches
    phi_vals = np.zeros(total_t, dtype=np.float64)
    # Also compute logical parities for each t: L^T t ∈ {0,1}^K
    # (t · L_k^T from the code side perspective)
    log_parity_t = np.zeros((total_t, K), dtype=np.uint8)  # (2^M, K)

    for batch_start in range(0, total_t, batch_size):
        batch_end = min(batch_start + batch_size, total_t)
        t_idx = np.arange(batch_start, batch_end, dtype=np.int64)
        bs = len(t_idx)

        # Extract bits of each t: t_bits[batch, bit_pos]
        t_bits = ((t_idx[:, None] >> np.arange(M)[None, :]) & 1).astype(np.uint8)
        # t_bits: (bs, M)

        # H^T t: (N, M) @ (bs, M)^T = (N, bs) → binary weight
        Ht = (H.T.astype(np.uint8) @ t_bits.T) % 2  # (N, bs)
        weights = Ht.sum(axis=0)                      # (bs,)
        phi_vals[batch_start:batch_end] = bias ** weights.astype(np.float64)

        # Logical parities: L @ t_bits.T = (K, M) @ (M, bs) = (K, bs)
        # Wait: we need t · rows_of_H, but for logical restriction we need
        # the parity of t with the logical rows.
        # Actually: we don't restrict by logical here. We compute marginals differently.
        # See note below.

    # For the logical marginals, we use the syndrome-restricted character expansion.
    # The marginal P(L_k=ℓ|s) is:
    #   P(L_k=ℓ|s) ∝ Σ_{t: t·L_k≡ℓ (mod 2), all t∈{0,1}^M} (-1)^{t·s} Φ(H^T t) [WRONG]
    #
    # Actually the correct formula:
    # Denote the full partition function:
    #   Z(s) = Σ_{e: He=s} Π_j p^{e_j}(1-p)^{1-e_j}
    #        = (1/2^M) Σ_t (-1)^{t·s} Φ(H^T t)
    #
    # The logical-restricted version:
    #   Z_ℓ(s) = Σ_{e: He=s, L_k·e=ℓ} P(e)
    #
    # Using the additional logical delta:
    #   Z_ℓ(s) = Z(s)/2 + (-1)^ℓ/2 · (1/2^M) Σ_t (-1)^{t·s} Ψ_k(H^T t)
    # where Ψ_k(v) = Σ_e P(e)(-1)^{(H^T t)·e} · (-1)^{L_k·e}
    #              = Π_j [(1-p)(-1)^{v_j·L_k_j} + p(-1)^{v_j+L_k_j}] ... not quite
    #
    # Simpler: augment H with the logical row:
    #   H_aug = [H; L_k]  (M+1 x N)
    #   Z_{s,ℓ}(s, ℓ) = (1/2^{M+1}) Σ_{t_aug ∈ {0,1}^{M+1}} (-1)^{t_aug·[s;ℓ]} Φ(H_aug^T t_aug)
    #
    # This doubles the sum for each logical, making it 2^{M+1} per logical.
    # Better: use the identity Σ_{e:L_k·e=ℓ} = (1/2)[Σ_e + (-1)^ℓ Σ_e (-1)^{L_k·e} Σ_e]
    #
    # So:
    #   Z_ℓ(s) = (1/2)[Z(s) + (-1)^ℓ R_k(s)]
    # where R_k(s) = Σ_{e:He=s} P(e)(-1)^{L_k·e}
    #             = (1/2^M) Σ_t (-1)^{t·s} Φ_k(H^T t)
    # with Φ_k(v) = Π_j [(1-p)+p·(-1)^{v_j}] · (-1)^{L_k·e}... hmm
    #
    # More carefully: R_k(s) = (1/2^M) Σ_t (-1)^{t·s} Ψ_k(t)
    # where Ψ_k(t) = Σ_e P(e) (-1)^{(H^T t + L_k)·e}
    #             = Π_j [(1-p) + p·(-1)^{(H^T t)_j + (L_k)_j}]
    #             = Π_j [(1-p)·(-1)^{(H^T t)_j·0} + p·(-1)^{(H^T t)_j + (L_k)_j}]
    # Hmm, let's use (H^T t)_j ∈ {0,1} and (L_k)_j ∈ {0,1}:
    # factor_j = (1-p)·(-1)^0 + p·(-1)^{(H^T t)_j XOR (L_k)_j + (H^T t)_j}
    # Actually:
    # Ψ_k(t) = Σ_e [Π_j p^{e_j}(1-p)^{1-e_j}] · (-1)^{Σ_j [(H^T t)_j + (L_k)_j] e_j}
    #         = Π_j Σ_{e_j∈{0,1}} p^{e_j}(1-p)^{1-e_j} · (-1)^{[(H^T t)_j + (L_k)_j] e_j}
    #         = Π_j [(1-p) + p·(-1)^{(H^T t)_j XOR (L_k)_j}]
    #           if (H^T t)_j XOR (L_k)_j = 1: factor = (1-p) - p = 1-2p
    #           if (H^T t)_j XOR (L_k)_j = 0: factor = (1-p) + p = 1
    #         = (1-2p)^{weight((H^T t) XOR L_k)}
    #
    # So: Ψ_k(t) = bias^{weight(H^T t XOR L_k)}
    #
    # And: P(L_k=0|s)/P(L_k=1|s) = Z_0(s)/Z_1(s)
    #   = [Z(s) + R_k(s)] / [Z(s) - R_k(s)]

    # Recompute Ψ_k for each logical k
    psi_vals = np.zeros((total_t, K), dtype=np.float64)

    for batch_start in range(0, total_t, batch_size):
        batch_end = min(batch_start + batch_size, total_t)
        t_idx = np.arange(batch_start, batch_end, dtype=np.int64)
        bs = len(t_idx)

        t_bits = ((t_idx[:, None] >> np.arange(M)[None, :]) & 1).astype(np.uint8)
        Ht = (H.T.astype(np.uint8) @ t_bits.T) % 2  # (N, bs)

        for k in range(K):
            Lk = logicals[k].astype(np.uint8)  # (N,)
            xor_Lk = (Ht ^ Lk[:, None])         # (N, bs)
            wt_k = xor_Lk.sum(axis=0)            # (bs,)
            psi_vals[batch_start:batch_end, k] = bias ** wt_k.astype(np.float64)

    # Now compute syndrome-conditioned sums for all shots
    # Z(s) = (1/2^M) Σ_t (-1)^{t·s} Φ(H^T t)
    # R_k(s) = (1/2^M) Σ_t (-1)^{t·s} Ψ_k(H^T t)
    #
    # The signs (-1)^{t·s} for t ∈ {0,1}^M, s ∈ {0,1}^M can be computed via
    # the Walsh-Hadamard transform.

    # For each shot, s defines a sign pattern: sign[t] = (-1)^{t·s}
    # Z(s) ∝ Σ_t sign[t] · phi[t]
    # R_k(s) ∝ Σ_t sign[t] · psi_k[t]

    # Vectorize over shots by computing the sign patterns:
    # sign[shot, t] = (-1)^{s[shot] · t_bits[t]}
    # = Π_m (-1)^{s[shot,m] · t_m}
    # This is the WHT matrix.

    # For large n_shots, do in chunks
    log_marginals = np.zeros((n_shots, K, 2), dtype=np.float64)

    # Enumerate all t once
    all_t = np.arange(total_t, dtype=np.int64)
    t_bits_all = ((all_t[:, None] >> np.arange(M)[None, :]) & 1).astype(np.uint8)
    # t_bits_all: (2^M, M)

    # Process shots in chunks
    shot_chunk = max(1, min(n_shots, (1 << 22) // total_t))  # fit in ~4M floats
    for shot_start in range(0, n_shots, shot_chunk):
        shot_end = min(shot_start + shot_chunk, n_shots)
        s_chunk = syndromes[shot_start:shot_end]  # (chunk, M)

        # Signs: (chunk, 2^M), element = (-1)^{s·t}
        dot = (s_chunk.astype(np.uint8) @ t_bits_all.T) % 2  # (chunk, 2^M)
        signs = 1.0 - 2.0 * dot.astype(np.float64)           # (chunk, 2^M)

        # Z(s) for each shot: (chunk,)
        Z_s = signs @ phi_vals  # (chunk,)

        for k in range(K):
            R_s = signs @ psi_vals[:, k]  # (chunk,)
            Z_0 = (Z_s + R_s) / 2.0
            Z_1 = (Z_s - R_s) / 2.0
            # Clamp to avoid log(0)
            Z_0 = np.maximum(Z_0, 1e-300)
            Z_1 = np.maximum(Z_1, 1e-300)
            log_marginals[shot_start:shot_end, k, 0] = np.log(Z_0)
            log_marginals[shot_start:shot_end, k, 1] = np.log(Z_1)

    return log_marginals  # (n_shots, K, 2)


# ---------------------------------------------------------------------------
# GF(2) linear algebra utilities for erasure support
# ---------------------------------------------------------------------------

def gf2_null_space_basis(H_E: np.ndarray) -> np.ndarray:
    """Compute a basis for the null space of H_E^T over GF(2).

    Given H_E of shape (M, |E|), finds all t ∈ {0,1}^M with H_E^T t = 0 (mod 2).
    Uses Gaussian elimination on H_E^T.

    Returns:
        basis: (dim_null, M) binary matrix, dim_null = M - rank(H_E).
               Returns zeros((1, M)) if only t=0 satisfies the constraint.
    """
    M, E_size = H_E.shape
    if E_size == 0:
        # No erasure: every t is valid; full basis
        return np.eye(M, dtype=np.uint8)

    A = (H_E.T % 2).astype(np.uint8)   # E_size × M  (system we row-reduce)
    m, n = A.shape                       # m = E_size, n = M

    pivot_cols: list[int] = []
    row_ptr = 0
    for col in range(n):
        # Find first non-zero entry in column col at or below row_ptr
        found = -1
        for r in range(row_ptr, m):
            if A[r, col]:
                found = r
                break
        if found < 0:
            continue                        # free column
        if found != row_ptr:
            A[[row_ptr, found]] = A[[found, row_ptr]]
        for r in range(m):                 # eliminate entire column
            if r != row_ptr and A[r, col]:
                A[r] ^= A[row_ptr]
        pivot_cols.append(col)
        row_ptr += 1

    rank = len(pivot_cols)
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n) if c not in pivot_set]
    dim_null = n - rank                    # = M - rank(H_E)

    if dim_null == 0:
        return np.zeros((1, M), dtype=np.uint8)

    # One null-space basis vector per free column
    basis = np.zeros((dim_null, M), dtype=np.uint8)
    for bi, fc in enumerate(free_cols):
        v = np.zeros(M, dtype=np.uint8)
        v[fc] = 1
        # Each pivot row i says: pivot_col[i] variable = A[i, fc]
        for pi, pc in enumerate(pivot_cols):
            if A[pi, fc]:
                v[pc] = 1
        basis[bi] = v
    return basis


def gf2_solve(A: np.ndarray, b: np.ndarray):
    """Solve A x = b over GF(2) for x ∈ {0,1}^n.

    Args:
        A: (m, n) binary matrix.
        b: (m,)  binary vector.

    Returns:
        (x, True)  — particular solution x if the system is consistent.
        (None, False) — if no solution exists.
    """
    m, n = A.shape
    aug = np.hstack([A.copy().astype(np.uint8),
                     b.reshape(-1, 1).astype(np.uint8)])   # m × (n+1)
    pivot_rows: dict[int, int] = {}    # col → row index of pivot
    row_ptr = 0
    for col in range(n):
        found = -1
        for r in range(row_ptr, m):
            if aug[r, col]:
                found = r
                break
        if found < 0:
            continue
        if found != row_ptr:
            aug[[row_ptr, found]] = aug[[found, row_ptr]]
        for r in range(m):
            if r != row_ptr and aug[r, col]:
                aug[r] ^= aug[row_ptr]
        pivot_rows[col] = row_ptr
        row_ptr += 1

    # Consistency check: any zero-row with b=1 → no solution
    for r in range(row_ptr, m):
        if aug[r, n]:
            return None, False

    x = np.zeros(n, dtype=np.uint8)
    for col, r in pivot_rows.items():
        x[col] = aug[r, n]
    return x, True


def _enumerate_null_space(basis: np.ndarray) -> np.ndarray:
    """Enumerate all 2^dim vectors spanned by the basis rows over GF(2).

    Args:
        basis: (dim, M) binary matrix of basis vectors.

    Returns:
        null_vecs: (2^dim, M) binary matrix including the zero vector.
    """
    dim, M = basis.shape
    if dim == 0 or np.all(basis == 0):
        return np.zeros((1, M), dtype=np.uint8)
    total = 1 << dim
    idx = np.arange(total, dtype=np.int64)
    coeffs = ((idx[:, None] >> np.arange(dim)[None, :]) & 1).astype(np.uint8)
    return (coeffs @ basis % 2).astype(np.uint8)   # (2^dim, M)


# ---------------------------------------------------------------------------
# Erasure precomputation
# ---------------------------------------------------------------------------

@dataclass
class _ErasurePrecomp:
    """Cached offline quantities for a specific erasure mask."""
    null_vecs:    np.ndarray   # (T_eff, M)  valid t for Φ
    phi_vals:     np.ndarray   # (T_eff,)    Φ(t) = bias^{wt(H_nonE^T t)}
    psi_vals:     np.ndarray   # (T_eff, K)  Ψ_k values at coset positions
    shifts:       np.ndarray   # (K, M)      particular solutions for cosets
    psi_solvable: np.ndarray   # (K,) bool   whether Ψ_k coset exists
    T_eff:        int
    # For online sign-factor trick: shift dot products
    # signs_k[shot,t] = signs[shot,t] * (-1)^{s[shot]·shifts[k]}
    # => R_k[shot] = sign_factor_k[shot] * (signs @ psi_vals[:,k])[shot]


def _precompute_for_erasure(
        H: np.ndarray,
        logicals: np.ndarray,
        noise_p: float,
        erasure_mask: np.ndarray,
) -> _ErasurePrecomp:
    """Precompute all quantities needed for CE decoding with a fixed erasure mask.

    Mathematics:
        Φ(t)  = 0 if H_E^T t ≠ 0,  else bias^{wt(H_nonE^T t)}
        Ψ_k(t)= 0 if H_E^T t ≠ L_k_E, else bias^{wt(H_nonE^T t XOR L_k_nonE)}

    The valid-t set for Ψ_k is a coset: null_vecs XOR shift_k, so
        signs_k = signs * (-1)^{s · shift_k}  (scalar per shot)
        R_k(s)  = sign_factor_k(s) * (signs @ psi_vals[:,k])

    where psi_vals[:,k] = bias^{wt(Ht_nonE XOR L'_k)}, L'_k = H_nonE^T shift_k XOR L_k_nonE.
    """
    M, N = H.shape
    K = logicals.shape[0]
    bias = 1.0 - 2.0 * noise_p

    non_erased = ~erasure_mask
    H_E   = H[:, erasure_mask]   # M × |E|
    H_ne  = H[:, non_erased]     # M × (N-|E|)
    L_E   = logicals[:, erasure_mask]   # K × |E|
    L_ne  = logicals[:, non_erased]     # K × (N-|E|)

    # Null space of H_E^T
    if H_E.shape[1] == 0:
        basis = np.eye(M, dtype=np.uint8)
    else:
        basis = gf2_null_space_basis(H_E)
    null_vecs = _enumerate_null_space(basis)   # (T_eff, M)
    T_eff = len(null_vecs)

    # phi_vals: Φ(t) for valid t
    if H_ne.shape[1] > 0:
        Ht_ne = (H_ne.T.astype(np.uint8) @ null_vecs.T) % 2   # (N-|E|, T_eff)
        phi_vals = bias ** Ht_ne.sum(axis=0).astype(np.float64)
    else:
        Ht_ne = np.zeros((0, T_eff), dtype=np.uint8)
        phi_vals = np.ones(T_eff, dtype=np.float64)

    # For each logical k: find shift (particular solution) and psi_vals
    shifts       = np.zeros((K, M), dtype=np.uint8)
    psi_solvable = np.ones(K, dtype=bool)
    psi_vals     = np.zeros((T_eff, K), dtype=np.float64)

    H_ET = H_E.T.astype(np.uint8) if H_E.shape[1] > 0 else None  # |E| × M

    for k in range(K):
        if H_ET is not None:
            lk_E = L_E[k].astype(np.uint8)
            t_k, ok = gf2_solve(H_ET, lk_E)
            if not ok:
                psi_solvable[k] = False
                continue
            shifts[k] = t_k
        # else: H_E is empty → shift = 0, always solvable

        # L'_k = H_nonE^T shift_k XOR L_k_nonE
        if H_ne.shape[1] > 0:
            shift_k = shifts[k].astype(np.uint8)
            Hs = (H_ne.T.astype(np.uint8) @ shift_k) % 2   # (N-|E|,)
            Lp_k = (Hs ^ L_ne[k].astype(np.uint8))          # (N-|E|,)
            # psi_vals[:,k] = bias^{wt(Ht_ne XOR Lp_k[:,None])}
            xor_lp = (Ht_ne ^ Lp_k[:, None]) % 2            # (N-|E|, T_eff)
            psi_vals[:, k] = bias ** xor_lp.sum(axis=0).astype(np.float64)
        else:
            psi_vals[:, k] = np.ones(T_eff, dtype=np.float64)

    return _ErasurePrecomp(
        null_vecs=null_vecs,
        phi_vals=phi_vals,
        psi_vals=psi_vals,
        shifts=shifts,
        psi_solvable=psi_solvable,
        T_eff=T_eff,
    )


def _decode_batch_with_precomp(
        precomp: _ErasurePrecomp,
        syndromes: np.ndarray,        # (B, M) binary
        K: int,
        use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch-decode shots that all share the same erasure mask.

    Uses the sign-factor identity for efficiency:
        R_k(s) = (-1)^{s·shift_k} · (signs @ psi_vals[:,k])

    Returns:
        assignments: (B, K) int32
        p1:          (B, K) float64  — P(L_k=1 | s)
    """
    xp = _xp(use_gpu)
    B = syndromes.shape[0]
    null_vecs = xp.asarray(precomp.null_vecs)    # (T_eff, M)
    phi_vals  = xp.asarray(precomp.phi_vals)     # (T_eff,)
    psi_vals  = xp.asarray(precomp.psi_vals)     # (T_eff, K)
    shifts    = xp.asarray(precomp.shifts)        # (K, M)
    syn       = xp.asarray(syndromes.astype(np.uint8))   # (B, M)

    # Signs for Φ: (B, T_eff) = (-1)^{syn @ null_vecs.T mod 2}
    dot   = (syn @ null_vecs.T) % 2
    signs = 1.0 - 2.0 * dot.astype(xp.float64)  # (B, T_eff)

    # Z(s) = signs @ phi_vals: (B,)
    Z_s = signs @ phi_vals

    # Per-logical decoding
    p1 = xp.full((B, K), 0.5, dtype=xp.float64)
    for k in range(K):
        if not precomp.psi_solvable[k]:
            continue
        # sign_factor_k[shot] = (-1)^{syn[shot] · shift_k}
        shift_k = shifts[k]                               # (M,)
        sf = (syn @ shift_k) % 2                          # (B,) in {0,1}
        sign_factor = 1.0 - 2.0 * sf.astype(xp.float64)  # (B,)
        R_raw = signs @ psi_vals[:, k]                    # (B,)
        R_s   = sign_factor * R_raw                       # (B,)

        Z_0 = xp.maximum((Z_s + R_s) / 2.0, 0.0)
        Z_1 = xp.maximum((Z_s - R_s) / 2.0, 0.0)
        denom = Z_0 + Z_1
        valid = denom > 0
        p1[:, k] = xp.where(valid, Z_1 / xp.where(valid, denom, 1.0), 0.5)

    if use_gpu and CUPY_AVAILABLE:
        p1_cpu = cp.asnumpy(p1)
    else:
        p1_cpu = np.asarray(p1)

    assignments = (p1_cpu >= 0.5).astype(np.int32)
    return assignments, p1_cpu


def effective_m_with_erasure(
        H: np.ndarray,
        erasure_rate: float,
        n_trials: int = 20,
        seed: int = 0,
) -> dict:
    """Estimate effective M (Fourier space size) for CE with random erasure.

    Returns a dict with:
        M:              number of checks
        avg_rank:       average rank(H_E) over trials
        avg_eff_M:      average M - rank(H_E)
        avg_T_eff:      average 2^{M - rank(H_E)}
        tractable_ce:   whether avg_T_eff ≤ 2^22 (CE practical threshold)
    """
    M, N = H.shape
    rng = np.random.default_rng(seed)
    ranks = []
    for _ in range(n_trials):
        mask = rng.random(N) < erasure_rate
        if not np.any(mask):
            ranks.append(0)
            continue
        H_E = H[:, mask]
        # Rank via GF(2) elimination
        A = (H_E % 2).astype(np.uint8)
        m2, n2 = A.shape
        rp = 0
        for col in range(n2):
            fnd = -1
            for r in range(rp, m2):
                if A[r, col]:
                    fnd = r
                    break
            if fnd < 0:
                continue
            if fnd != rp:
                A[[rp, fnd]] = A[[fnd, rp]]
            for r in range(m2):
                if r != rp and A[r, col]:
                    A[r] ^= A[rp]
            rp += 1
        ranks.append(rp)

    avg_rank  = np.mean(ranks)
    avg_eff_M = M - avg_rank
    avg_T_eff = 2 ** avg_eff_M
    return dict(
        M=M, N=N,
        avg_rank=avg_rank,
        avg_eff_M=avg_eff_M,
        avg_T_eff=avg_T_eff,
        tractable_ce=(avg_eff_M <= 22),
    )


# ---------------------------------------------------------------------------
# High-level decoder class
# ---------------------------------------------------------------------------

@dataclass
class CharExpDecodeResult:
    """Results from a single decode call."""
    assignments: dict[str, int]     # obs_k -> 0 or 1
    marginals: dict[str, float]     # obs_k -> P(L_k=1|s)
    log_probs: dict[str, tuple]     # obs_k -> (log P0, log P1)


class CharacterExpansionDecoder:
    """Exact MLD decoder via GF(2) Fourier / character expansion.

    Suitable for codes with M ≤ ~20 checks (runtime O(2^M · N)).

    Usage::

        decoder = CharacterExpansionDecoder(H, logicals, noise_p)
        assignments, marginals = decoder.decode_logicals(syndrome_dict)

    For batch decoding (faster)::

        results = decoder.decode_batch(syndromes_array)  # (n_shots, M) binary
    """

    def __init__(
            self,
            H: np.ndarray,
            logicals: np.ndarray,
            noise_p: float,
            batch_size: int = 1 << 18,
    ):
        M, N = H.shape
        self.H = H.astype(np.uint8)
        self.logicals = logicals.astype(np.uint8)
        self.noise_p = noise_p
        self.M = M
        self.N = N
        self.K = logicals.shape[0]
        self._batch_size = batch_size

        if 1 << M > (1 << 26):
            raise ValueError(f"M={M} exceeds practical limit (2^M > 64M terms).")

        self._syndrome_labels = [f"syn_param_{i}" for i in range(M)]
        self._logical_labels  = [f"obs_{k}" for k in range(self.K)]

    # ------------------------------------------------------------------
    # Online API (mirrors OneDChainMPS)
    # ------------------------------------------------------------------

    def decode_logicals(
            self,
            syndrome_values: dict[str, float],
    ) -> tuple[dict[str, int], dict[str, float]]:
        """Decode one shot. Returns (assignments, marginals).

        Args:
            syndrome_values: dict mapping 'syn_param_i' → 0.0 or 1.0.
        """
        s = np.array(
            [int(round(syndrome_values.get(lbl, 0.0)))
             for lbl in self._syndrome_labels],
            dtype=np.uint8
        )
        log_marg = _char_expansion_exact(
            self.H, self.logicals, self.noise_p, s[None, :],
            batch_size=self._batch_size
        )  # (1, K, 2)

        assignments = {}
        marginals = {}
        for k, lbl in enumerate(self._logical_labels):
            lp0, lp1 = log_marg[0, k, 0], log_marg[0, k, 1]
            p1 = np.exp(lp1 - np.logaddexp(lp0, lp1))
            marginals[lbl] = float(p1)
            assignments[lbl] = 1 if p1 >= 0.5 else 0

        return assignments, marginals

    def decode_batch(
            self,
            syndromes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode a batch of syndromes.

        Args:
            syndromes: (n_shots, M) binary array.

        Returns:
            (assignments, marginals) both (n_shots, K) arrays.
        """
        log_marg = _char_expansion_exact(
            self.H, self.logicals, self.noise_p, syndromes.astype(np.uint8),
            batch_size=self._batch_size
        )  # (n_shots, K, 2)

        # Normalise to probabilities
        log_Z = np.logaddexp(log_marg[:, :, 0], log_marg[:, :, 1])  # (n, K)
        p1 = np.exp(log_marg[:, :, 1] - log_Z)   # (n, K)
        assignments = (p1 >= 0.5).astype(np.int32)
        return assignments, p1

    def decode_batch_erasure(
            self,
            syndromes: np.ndarray,
            erasure_masks: np.ndarray,
            use_gpu: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode a batch of syndromes with known per-shot erasure locations.

        For erased qubits the noise probability is 0.5 (location known but type
        unknown); non-erased qubits use self.noise_p.

        Shots sharing the same erasure mask are grouped together so that the
        GF(2) precomputation (null-space enumeration) is done only once per
        unique mask, dramatically reducing overhead when many shots share a
        common pattern.

        Args:
            syndromes:     (n_shots, M) binary array.
            erasure_masks: (n_shots, N) boolean array. True = qubit is erased.
            use_gpu:       Use CuPy for GPU acceleration (requires CuPy install).

        Returns:
            assignments: (n_shots, K) int32 — 0 or 1 per logical.
            p1:          (n_shots, K) float64 — P(L_k=1 | syndrome, erasure).
        """
        n_shots = syndromes.shape[0]
        assignments = np.zeros((n_shots, self.K), dtype=np.int32)
        p1_out = np.full((n_shots, self.K), 0.5, dtype=np.float64)

        # Group shots by unique erasure mask (bytes key for efficiency)
        mask_groups: dict[bytes, list[int]] = {}
        for i in range(n_shots):
            key = erasure_masks[i].astype(np.uint8).tobytes()
            if key not in mask_groups:
                mask_groups[key] = []
            mask_groups[key].append(i)

        n_unique = len(mask_groups)
        for key_idx, (key, indices) in enumerate(mask_groups.items()):
            # Reconstruct boolean erasure mask from bytes key
            erasure_mask = np.frombuffer(key, dtype=np.uint8).astype(bool)

            # Offline: GF(2) null-space + Phi/Psi precomputation
            precomp = _precompute_for_erasure(
                self.H, self.logicals, self.noise_p, erasure_mask
            )

            idx_arr = np.array(indices, dtype=np.int64)
            syn_batch = syndromes[idx_arr]

            # Online: batch decode for all shots sharing this mask
            asn_batch, p1_batch = _decode_batch_with_precomp(
                precomp, syn_batch, self.K, use_gpu=use_gpu
            )
            assignments[idx_arr] = asn_batch
            p1_out[idx_arr] = p1_batch

        return assignments, p1_out


# ---------------------------------------------------------------------------
# Factory / convenience
# ---------------------------------------------------------------------------

def compile_char_expansion(
        H: np.ndarray,
        logicals: np.ndarray,
        noise_p: float,
) -> CharacterExpansionDecoder:
    """Create a CharacterExpansionDecoder (no offline compile step needed).

    The decoder is inherently exact — no chi parameter.  The 'compile'
    step is just construction.

    Args:
        H:          Parity check matrix (M x N, binary).
        logicals:   Logical operators (K x N, binary).
        noise_p:    Physical error probability.

    Returns:
        A CharacterExpansionDecoder ready for online decoding.
    """
    return CharacterExpansionDecoder(H, logicals, noise_p)


def max_cut_width(H: np.ndarray, n_trials: int = 16, seed: int = 0) -> int:
    """Estimate the minimum max-cut width of the check adjacency graph.

    This lower-bounds the chi required for exact 1D MPS representation.
    O(n_trials * M^2 * N) runtime.

    Returns:
        best_cut: minimum over random orderings of the maximum cut width.
    """
    M, N = H.shape
    rng = np.random.default_rng(seed)
    row_sup = [set(np.where(H[i])[0]) for i in range(M)]
    best = M
    for _ in range(n_trials):
        perm = rng.permutation(M)
        above = set()
        max_c = 0
        for idx in range(M - 1):
            above.update(row_sup[perm[idx]])
            below = set()
            for i in perm[idx + 1:]:
                below.update(row_sup[i])
            max_c = max(max_c, len(above & below))
        best = min(best, max_c)
    return best
