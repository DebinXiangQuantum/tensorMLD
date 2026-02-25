# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Core utilities for compressed tensor-network decoding.

This module is intentionally independent from `cudaq_qec` extension bindings so
its logic can be unit tested directly:
1) exact tensor decomposition to MPS (with structured exact shortcuts),
2) offline contraction-swap-merge compression (adapted from `catn/`),
3) online decoding on a 1D-chain MPS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np
from quimb import oset
from quimb.tensor import Tensor, TensorNetwork

try:
    import cupy
except Exception:  # pragma: no cover - optional dependency
    cupy = None

OUT_PREFIX = "__OUT__::"
DELTA_TAG = "DELTA_INDEX"

# Global defaults (can be changed via `set_global_decoder_config`).
GLOBAL_BOND_DIM = 32
GLOBAL_LOW_THRESHOLD = 0.4
GLOBAL_HIGH_THRESHOLD = 0.6
GLOBAL_MAX_CONDITIONAL_ROUNDS = 16


def _validate_threshold_window(low: float, high: float) -> None:
    if not (0.0 <= low <= 1.0):
        raise ValueError("low_threshold must be in [0, 1].")
    if not (0.0 <= high <= 1.0):
        raise ValueError("high_threshold must be in [0, 1].")
    if low > high:
        raise ValueError("low_threshold must be <= high_threshold.")


def set_global_decoder_config(
        *,
        bond_dim: Optional[int] = None,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        max_conditional_rounds: Optional[int] = None) -> None:
    """Set module-level defaults used by compressed MPS decoding."""
    global GLOBAL_BOND_DIM
    global GLOBAL_LOW_THRESHOLD
    global GLOBAL_HIGH_THRESHOLD
    global GLOBAL_MAX_CONDITIONAL_ROUNDS

    next_low = (GLOBAL_LOW_THRESHOLD if low_threshold is None else
                float(low_threshold))
    next_high = (GLOBAL_HIGH_THRESHOLD if high_threshold is None else
                 float(high_threshold))
    _validate_threshold_window(next_low, next_high)

    if bond_dim is not None:
        if bond_dim <= 0:
            raise ValueError("bond_dim must be > 0.")
        GLOBAL_BOND_DIM = int(bond_dim)
    if low_threshold is not None:
        GLOBAL_LOW_THRESHOLD = next_low
    if high_threshold is not None:
        GLOBAL_HIGH_THRESHOLD = next_high
    if max_conditional_rounds is not None:
        if max_conditional_rounds <= 0:
            raise ValueError("max_conditional_rounds must be > 0.")
        GLOBAL_MAX_CONDITIONAL_ROUNDS = int(max_conditional_rounds)


try:
    from .mps_node_np import MPSNode
except ImportError:
    # Fallback for standalone loading (e.g. importlib.util.spec_from_file_location)
    import importlib.util as _ilu
    _mps_spec = _ilu.spec_from_file_location(
        "mps_node_np", Path(__file__).parent / "mps_node_np.py")
    _mps_mod = _ilu.module_from_spec(_mps_spec)  # type: ignore[arg-type]
    # Pre-load npsvd into the module's namespace so its relative import resolves
    _svd_spec = _ilu.spec_from_file_location(
        "npsvd", Path(__file__).parent / "npsvd.py")
    _svd_mod = _ilu.module_from_spec(_svd_spec)  # type: ignore[arg-type]
    _svd_spec.loader.exec_module(_svd_mod)  # type: ignore[union-attr]
    import sys as _sys
    _sys.modules["npsvd"] = _svd_mod
    _mps_spec.loader.exec_module(_mps_mod)  # type: ignore[union-attr]
    MPSNode = _mps_mod.MPSNode  # type: ignore[misc]


def make_parametrized_syndrome_network(
        check_inds: list[str],
        syndrome_param_inds: list[str]) -> TensorNetwork:
    """Create syndrome tensors with an explicit parameter index.

    Each syndrome tensor is a Hadamard-like matrix `[[1, 1], [1, -1]]` with
    indices `(check_i, syn_param_i)`. During online decoding, we contract the
    `syn_param_i` index with `[1-s_i, s_i]`.
    """
    assert len(check_inds) == len(syndrome_param_inds)
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64)
    return TensorNetwork([
        Tensor(
            data=hadamard,
            inds=(c_ind, s_ind),
            tags=oset([f"SYN_{i}", "SYNDROME", "SYNDROME_PARAM"]),
        ) for i, (c_ind, s_ind) in enumerate(zip(check_inds,
                                                 syndrome_param_inds))
    ])


def make_delta_tensor(dim: int, order: int) -> np.ndarray:
    """Construct a Kronecker delta tensor with shape `(dim,) * order`."""
    data = np.zeros((dim,) * order, dtype=np.float64)
    for value in range(dim):
        data[(value,) * order] = 1.0
    return data


def _is_delta_tensor(data: np.ndarray, atol: float = 1e-12) -> bool:
    """Check whether `data` matches a Kronecker-delta tensor exactly."""
    if data.ndim == 0:
        return True
    if len(set(data.shape)) != 1:
        return False
    dim = data.shape[0]
    expected = make_delta_tensor(dim=dim, order=data.ndim)
    return np.allclose(data, expected, atol=atol, rtol=0.0)


def _delta_to_mps(data: np.ndarray) -> list[np.ndarray]:
    """Exact MPS decomposition for Kronecker-delta tensors."""
    if data.ndim == 0:
        return []
    if data.ndim == 1:
        return [data.reshape(1, data.shape[0], 1)]

    dim = data.shape[0]
    order = data.ndim
    cores: list[np.ndarray] = []

    first = np.zeros((1, dim, dim), dtype=np.float64)
    for a in range(dim):
        first[0, a, a] = 1.0
    cores.append(first)

    for _ in range(order - 2):
        core = np.zeros((dim, dim, dim), dtype=np.float64)
        for state in range(dim):
            core[state, state, state] = 1.0
        cores.append(core)

    last = np.zeros((dim, dim, 1), dtype=np.float64)
    for a in range(dim):
        last[a, a, 0] = 1.0
    cores.append(last)
    return cores


def _parity_values(data: np.ndarray,
                   atol: float = 1e-12) -> Optional[tuple[float, float]]:
    """Detect parity-structured tensor: value depends only on parity of indices.

    Returns `(even_value, odd_value)` if structured, else `None`.
    """
    if data.ndim == 0:
        return float(data), float(data)
    if any(dim != 2 for dim in data.shape):
        return None

    even_val: Optional[float] = None
    odd_val: Optional[float] = None
    for idx in np.ndindex(data.shape):
        parity = sum(idx) & 1
        val = float(data[idx])
        if parity == 0:
            if even_val is None:
                even_val = val
            elif not math.isclose(val, even_val, abs_tol=atol, rel_tol=0.0):
                return None
        else:
            if odd_val is None:
                odd_val = val
            elif not math.isclose(val, odd_val, abs_tol=atol, rel_tol=0.0):
                return None

    if even_val is None:
        even_val = 0.0
    if odd_val is None:
        odd_val = even_val
    return even_val, odd_val


def _parity_to_mps(data: np.ndarray) -> Optional[list[np.ndarray]]:
    """Exact bond-2 MPS for parity-structured tensors."""
    values = _parity_values(data)
    if values is None:
        return None
    even_val, odd_val = values

    if data.ndim == 0:
        return []
    if data.ndim == 1:
        core = np.array([[even_val], [odd_val]], dtype=np.float64).reshape(
            1, 2, 1)
        return [core]

    order = data.ndim
    cores: list[np.ndarray] = []

    first = np.zeros((1, 2, 2), dtype=np.float64)
    first[0, 0, 0] = 1.0
    first[0, 1, 1] = 1.0
    cores.append(first)

    for _ in range(order - 2):
        core = np.zeros((2, 2, 2), dtype=np.float64)
        core[0, 0, 0] = 1.0
        core[0, 1, 1] = 1.0
        core[1, 0, 1] = 1.0
        core[1, 1, 0] = 1.0
        cores.append(core)

    last = np.zeros((2, 2, 1), dtype=np.float64)
    for prev_parity in (0, 1):
        for bit in (0, 1):
            final_parity = prev_parity ^ bit
            last[prev_parity, bit, 0] = even_val if final_parity == 0 else odd_val
    cores.append(last)
    return cores


def _svd_to_mps(data: np.ndarray,
                chi: int,
                cutoff: float = 1e-12) -> list[np.ndarray]:
    """Generic sequential SVD decomposition into an MPS."""
    if data.ndim == 0:
        return []
    if data.ndim == 1:
        return [data.reshape(1, data.shape[0], 1)]

    dims = list(data.shape)
    work = data.reshape(1, -1)
    left_dim = 1
    cores: list[np.ndarray] = []

    for physical_dim in dims[:-1]:
        work = work.reshape(left_dim * physical_dim, -1)
        u, s, vh = np.linalg.svd(work, full_matrices=False)

        keep = np.where(s > cutoff)[0]
        if keep.size == 0:
            keep = np.array([0], dtype=np.int64)
        if chi > 0:
            keep = keep[:chi]

        rank = int(keep.size)
        u = u[:, keep]
        s = s[keep]
        vh = vh[keep, :]

        cores.append(u.reshape(left_dim, physical_dim, rank))
        work = (s[:, None] * vh)
        left_dim = rank

    cores.append(work.reshape(left_dim, dims[-1], 1))
    return cores


def decompose_tensor_to_mps(data: np.ndarray,
                            chi: int,
                            cutoff: float = 1e-12) -> list[np.ndarray]:
    """Decompose a tensor into MPS.

    Priority:
    1) exact Kronecker-delta decomposition,
    2) exact parity decomposition,
    3) generic SVD decomposition.
    """
    arr = np.asarray(data, dtype=np.float64)

    if _is_delta_tensor(arr, atol=cutoff):
        return _delta_to_mps(arr)

    parity_mps = _parity_to_mps(arr)
    if parity_mps is not None:
        return parity_mps

    return _svd_to_mps(arr, chi=chi, cutoff=cutoff)


def mps_to_raw(mps: list[np.ndarray]) -> np.ndarray:
    """Reconstruct a raw tensor from MPS cores."""
    if not mps:
        return np.array(1.0, dtype=np.float64)
    if len(mps) == 1:
        core = mps[0]
        return core.reshape(core.shape[1])

    shape = [mps[0].shape[1]]
    mat = mps[0].reshape(mps[0].shape[1], mps[0].shape[2])
    for core in mps[1:]:
        shape.append(core.shape[1])
        mat = np.einsum("ij,jkl->ikl", mat, core).reshape(mat.shape[0] *
                                                          core.shape[1],
                                                          core.shape[2])
    return mat.reshape(shape)


@dataclass
class PairwiseGraphData:
    node_tensors: dict[int, np.ndarray]
    node_neighbors: dict[int, np.ndarray]
    node_tags: dict[int, set[str]]
    preserved_nodes: set[int]
    initial_internal_edges: int


def _out_token(index_name: str) -> str:
    return f"{OUT_PREFIX}{index_name}"


def out_token_to_index(token: str) -> str:
    if not token.startswith(OUT_PREFIX):
        return token
    return token[len(OUT_PREFIX):]


def build_pairwise_graph_from_tn(tn: TensorNetwork,
                                 preserve_inds: set[str]) -> PairwiseGraphData:
    """Convert a (hyper) tensor network into a pairwise graph.

    For each index with degree > 2, we add an explicit delta node so that each
    index is represented by pairwise edges between nodes.
    """
    tensors = list(tn.tensors)

    node_tensors: dict[int, np.ndarray] = {}
    node_neighbors: dict[int, np.ndarray] = {}
    node_tags: dict[int, set[str]] = {}
    preserved_nodes: set[int] = set()

    for node_id, tensor in enumerate(tensors):
        data = np.asarray(tensor.data, dtype=np.float64)
        node_tensors[node_id] = data
        node_neighbors[node_id] = np.empty(data.ndim, dtype=object)
        node_tags[node_id] = set(tensor.tags)

    index_to_legs: dict[str, list[tuple[int, int, int]]] = {}
    for node_id, tensor in enumerate(tensors):
        for axis, ind in enumerate(tensor.inds):
            dim = int(tensor.shape[axis])
            index_to_legs.setdefault(ind, []).append((node_id, axis, dim))

    next_node_id = len(tensors)
    for ind, legs in index_to_legs.items():
        degree = len(legs)
        if degree == 1:
            node_id, axis, _ = legs[0]
            node_neighbors[node_id][axis] = _out_token(ind)
            if ind in preserve_inds:
                preserved_nodes.add(node_id)
            continue

        if degree == 2:
            (a, axis_a, _), (b, axis_b, _) = legs
            if a != b:
                node_neighbors[a][axis_a] = b
                node_neighbors[b][axis_b] = a
                continue
            # Repeated index on the same tensor: promote to explicit delta.

        dim = int(legs[0][2])
        delta_node = next_node_id
        next_node_id += 1

        node_tensors[delta_node] = make_delta_tensor(dim=dim, order=degree)
        node_neighbors[delta_node] = np.empty(degree, dtype=object)
        node_tags[delta_node] = {DELTA_TAG, f"DELTA_{ind}"}

        for delta_axis, (orig_node, orig_axis, _) in enumerate(legs):
            node_neighbors[orig_node][orig_axis] = delta_node
            node_neighbors[delta_node][delta_axis] = orig_node

    for node_id, neighbors in node_neighbors.items():
        if neighbors.size and np.any(neighbors == None):  # noqa: E711
            missing_axes = np.where(neighbors == None)[0]  # noqa: E711
            raise RuntimeError(
                f"Incomplete pairwise graph for node {node_id}, "
                f"missing neighbor mapping on axes {missing_axes.tolist()}.")

    edge_count = count_internal_edges(set(node_tensors.keys()), node_neighbors)
    return PairwiseGraphData(
        node_tensors=node_tensors,
        node_neighbors=node_neighbors,
        node_tags=node_tags,
        preserved_nodes=preserved_nodes,
        initial_internal_edges=edge_count,
    )


def count_internal_edges(active_nodes: set[int],
                         node_neighbors: dict[int,
                                              np.ndarray]) -> int:
    edges: set[tuple[int, int]] = set()
    for i in active_nodes:
        for nb in node_neighbors[i]:
            if isinstance(nb, int) and nb in active_nodes:
                a, b = (i, nb) if i < nb else (nb, i)
                edges.add((a, b))
    return len(edges)


def build_mps_nodes(pairwise: PairwiseGraphData, chi: int,
                    cutoff: float) -> dict[int, Any]:
    """Instantiate catn-style MPS nodes with exact-structure decomposition."""
    nodes: dict[int, Any] = {}
    for node_id, raw_tensor in pairwise.node_tensors.items():
        neighbors = np.array(pairwise.node_neighbors[node_id], dtype=object)
        node = MPSNode(
            np.asarray(raw_tensor, dtype=np.float64),
            node_id,
            neighbors,
            chi=chi,
            cutoff=cutoff,
            norm_method=1,
            svdopt=True,
            swapopt=True,
            verbose=0,
        )
        node.neighbor = neighbors
        node.mps = decompose_tensor_to_mps(raw_tensor, chi=chi, cutoff=cutoff)
        node.cano = len(node.mps) - 1 if node.mps else 0
        nodes[node_id] = node
    return nodes


def _edge_cost(nodes: dict[int, Any], i: int, j: int) -> float:
    idx = nodes[i].find_neighbor(j)
    if idx < 0:
        return float("inf")
    return float(nodes[i].logdim() + nodes[j].logdim() -
                 2.0 * nodes[i].logdim(idx))


def _collect_internal_edges(active_nodes: set[int],
                            nodes: dict[int, Any]) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i in active_nodes:
        for nb in nodes[i].neighbor:
            if isinstance(nb, int) and nb in active_nodes:
                a, b = (i, nb) if i < nb else (nb, i)
                edges.add((a, b))
    return sorted(edges)


@dataclass
class StepTruncationRecord:
    """Per-step truncation error breakdown during offline compression."""
    step: int
    edge: tuple[int, int]
    eat_error: float
    merge_error: float
    compress_error: float
    total_error: float
    active_nodes_after: int

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "edge": list(self.edge),
            "eat_error": self.eat_error,
            "merge_error": self.merge_error,
            "compress_error": self.compress_error,
            "total_error": self.total_error,
            "active_nodes_after": self.active_nodes_after,
        }


@dataclass
class OfflineCompressionStats:
    initial_nodes: int
    initial_edges: int
    preserved_target: int
    steps: int
    truncation_error: float
    max_bond_dim: int
    active_nodes: int
    active_edges: int
    chi: int = 0
    step_errors: list[StepTruncationRecord] = field(default_factory=list)

    def to_json(self, path: str | Path | None = None) -> str:
        """Export compression stats to JSON, optionally writing to a file."""
        data = {
            "initial_nodes": self.initial_nodes,
            "initial_edges": self.initial_edges,
            "preserved_target": self.preserved_target,
            "steps": self.steps,
            "truncation_error": self.truncation_error,
            "max_bond_dim": self.max_bond_dim,
            "active_nodes": self.active_nodes,
            "active_edges": self.active_edges,
            "chi": self.chi,
            "step_errors": [r.to_dict() for r in self.step_errors],
        }
        text = json.dumps(data, indent=2)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        return text


@dataclass
class OfflineCompressionResult:
    nodes: dict[int, Any]
    active_nodes: set[int]
    preserved_nodes: set[int]
    stats: OfflineCompressionStats


def _cut_bondim_opt(
        nodes: dict[int, Any],
        i: int,
        idx_k_in_i: int,
        Dmax: int,
        cutoff: float = 1e-15) -> float:
    """Truncate the physical bond dimension at a specific edge.

    After merging duplicate edges, the bond dimension between nodes i and k
    can grow large.  This function performs QR + SVD to reduce it back to
    at most *Dmax*, returning the total truncation error.

    This mirrors ``cut_bondim_opt`` from ``catn/tensor_network_np.py``.
    """
    k = nodes[i].neighbor[idx_k_in_i]
    idx_i_in_k = int(nodes[k].find_neighbor(i))
    if idx_i_in_k < 0:
        return 0.0

    # Canonicalise both nodes towards the connecting edge
    nodes[i].cano_to(idx_k_in_i)
    nodes[k].cano_to(idx_i_in_k)

    core_i = nodes[i].mps[idx_k_in_i]
    core_k = nodes[k].mps[idx_i_in_k]

    da_l, d, da_r = core_i.shape
    db_l, d2, db_r = core_k.shape
    assert d == d2, f"bond dim mismatch: {d} vs {d2}"

    mati = core_i.transpose(0, 2, 1).reshape(da_l * da_r, d)
    matk = core_k.transpose(0, 2, 1).reshape(db_l * db_r, d)

    # Optional QR to reduce SVD size
    flag_left = mati.shape[0] > mati.shape[1]
    if flag_left:
        qi, ri = np.linalg.qr(mati)
    else:
        ri = mati

    flag_right = matk.shape[0] > matk.shape[1]
    if flag_right:
        qk, rk = np.linalg.qr(matk)
    else:
        rk = matk

    merged = ri @ rk.T
    U, s, Vt = np.linalg.svd(merged, full_matrices=False)
    V = Vt.T

    s_eff = s[s > cutoff]
    if len(s_eff) == 0:
        s_eff = s[:1]

    error = float(s[len(s_eff):].sum())
    myd = min(len(s_eff), Dmax) if Dmax > 0 else len(s_eff)
    if myd == 0:
        myd = 1

    error += float(s_eff[myd:].sum())
    s_eff = s_eff[:myd]
    s_diag = np.diag(np.sqrt(s_eff))
    U = U[:, :myd]
    V = V[:, :myd]
    new_mati = U @ s_diag
    new_matk = (s_diag @ V.T).T

    if flag_left:
        new_mati = qi @ new_mati
    if flag_right:
        new_matk = qk @ new_matk

    nodes[i].mps[idx_k_in_i] = new_mati.reshape(
        da_l, da_r, myd).transpose(0, 2, 1)
    nodes[k].mps[idx_i_in_k] = new_matk.reshape(
        db_l, db_r, myd).transpose(0, 2, 1)
    return error


def compress_until_preserved(
        nodes: dict[int, Any],
        preserved_nodes: set[int],
        max_steps: int = 100000,
        reverse: bool = True,
        compress_each_step: bool = True,
        verbose: bool = False,
        chi: int = 0) -> OfflineCompressionResult:
    """Run contraction-swap-merge until only preserved nodes remain active."""
    active_nodes = set(nodes.keys())
    target_preserved = len(preserved_nodes)
    initial_edges = count_internal_edges(active_nodes,
                                         {k: v.neighbor
                                          for k, v in nodes.items()})
    truncation_error = 0.0
    steps = 0
    step_errors: list[StepTruncationRecord] = []

    while any(node_id not in preserved_nodes for node_id in active_nodes):
        edges = _collect_internal_edges(active_nodes, nodes)
        # Disallow contracting edges fully inside the preserved set.
        candidates = [e for e in edges if not (e[0] in preserved_nodes
                                               and e[1] in preserved_nodes)]
        if not candidates:
            if verbose:
                print("No valid contraction edge left before reaching target set.")
            break

        i, j = min(candidates, key=lambda edge: _edge_cost(nodes, edge[0],
                                                           edge[1]))
        if nodes[j].order() > nodes[i].order():
            i, j = j, i

        idx_j_in_i = int(nodes[i].find_neighbor(j))
        idx_i_in_j = int(nodes[j].find_neighbor(i))
        if idx_j_in_i < 0 or idx_i_in_j < 0:
            raise RuntimeError(f"Broken adjacency while contracting edge ({i}, {j}).")

        if reverse:
            if idx_j_in_i < len(nodes[i].neighbor) // 2:
                nodes[i].reverse()
                idx_j_in_i = int(nodes[i].find_neighbor(j))
            if idx_i_in_j >= len(nodes[j].neighbor) // 2:
                nodes[j].reverse()
                idx_i_in_j = int(nodes[j].find_neighbor(i))

        nodes[i].delete_neighbor(j)
        neighbors_j = list(nodes[j].neighbor)
        duplicate_neighbors: list[int] = []

        step_merge_err = 0.0
        for pos, k in enumerate(neighbors_j):
            if pos == idx_i_in_j:
                continue
            nodes[i].add_neighbor(k)
            if isinstance(k, int) and k in active_nodes:
                idx_i_in_k = int(nodes[k].find_neighbor(i))
                idx_j_in_k = int(nodes[k].delete_neighbor(j))
                nodes[k].add_neighbor(i, idx_j_in_k)
                if idx_i_in_k > -1:
                    duplicate_neighbors.append(k)
                    m_err = float(
                        nodes[k].merge(i, cross=idx_i_in_k > idx_j_in_k))
                    step_merge_err += m_err
                    truncation_error += m_err

        _, err, _ = nodes[i].eat(nodes[j], idx_j_in_i, idx_i_in_j)
        step_eat_err = abs(float(err))
        truncation_error += step_eat_err

        for k in duplicate_neighbors:
            m_err = float(nodes[i].merge(k, cross=False))
            step_merge_err += m_err
            truncation_error += m_err
            # Truncate the merged bond dimension (critical for 1D chain)
            idx_k_in_i = int(nodes[i].find_neighbor(k))
            if idx_k_in_i >= 0 and chi > 0:
                cut_err = _cut_bondim_opt(nodes, i, idx_k_in_i, Dmax=chi)
                step_merge_err += cut_err
                truncation_error += cut_err

        step_compress_err = 0.0
        if compress_each_step:
            step_compress_err = float(nodes[i].compress_opt())
            truncation_error += step_compress_err

        nodes[j].clear()
        active_nodes.remove(j)

        if j in preserved_nodes:
            preserved_nodes.remove(j)
            preserved_nodes.add(i)

        steps += 1
        step_total = step_eat_err + step_merge_err + step_compress_err
        step_errors.append(StepTruncationRecord(
            step=steps,
            edge=(i, j),
            eat_error=step_eat_err,
            merge_error=step_merge_err,
            compress_error=step_compress_err,
            total_error=step_total,
            active_nodes_after=len(active_nodes),
        ))

        if verbose and (steps % 100 == 0):
            edges_left = count_internal_edges(
                active_nodes, {k: v.neighbor
                               for k, v in nodes.items()})
            print(f"[offline] step={steps}, active={len(active_nodes)}, edges={edges_left}")
        if steps >= max_steps:
            if verbose:
                print(f"Reached max_steps={max_steps}, stopping offline compression.")
            break

    max_bond = 1
    for node_id in active_nodes:
        for core in nodes[node_id].mps:
            max_bond = max(max_bond, int(core.shape[0]), int(core.shape[2]))

    active_edges = count_internal_edges(active_nodes,
                                        {k: v.neighbor
                                         for k, v in nodes.items()})
    stats = OfflineCompressionStats(
        initial_nodes=len(nodes),
        initial_edges=initial_edges,
        preserved_target=target_preserved,
        steps=steps,
        truncation_error=float(truncation_error),
        max_bond_dim=max_bond,
        active_nodes=len(active_nodes),
        active_edges=active_edges,
        chi=chi,
        step_errors=step_errors,
    )
    return OfflineCompressionResult(
        nodes=nodes,
        active_nodes=active_nodes,
        preserved_nodes=preserved_nodes,
        stats=stats,
    )


def merge_preserved_to_path(
        nodes: dict[int, Any],
        active_nodes: set[int],
        chi: int = 0,
        max_steps: int = 100000,
        verbose: bool = False) -> float:
    """Merge preserved nodes until the internal adjacency graph is a 1D path.

    After ``compress_until_preserved`` the residual graph may still have nodes
    with degree > 2 (i.e. not a simple path).  This function iteratively
    contracts edges between preserved nodes, choosing edges that maximally
    reduce the *maximum* degree, until every node has degree <= 2.

    Returns the cumulative truncation error introduced by the merging.
    """
    truncation_error = 0.0
    steps = 0

    while steps < max_steps:
        # Build current adjacency and check if already a path.
        adjacency = _build_internal_adjacency(active_nodes, nodes)
        if _path_order_from_adjacency(adjacency) is not None:
            if verbose:
                print(f"[merge_preserved] path achieved after {steps} merging steps")
            break

        # Find edges where at least one endpoint has degree > 2.
        edges = _collect_internal_edges(active_nodes, nodes)
        if not edges:
            break

        # Heuristic: pick the edge (i, j) that results in the lowest max
        # degree after merging. When i eats j, the merged node's degree is:
        #   deg(i) + deg(j) - 2 - 2*|common_neighbors(i,j)|
        # (−2 for the contracted edge, −2 per shared neighbor that gets merged).
        # We want to minimize the *resulting* degree of the merged node, then
        # break ties by contraction cost.
        def _merge_priority(edge):
            a, b = edge
            neigh_a = adjacency.get(a, set())
            neigh_b = adjacency.get(b, set())
            common = len(neigh_a & neigh_b)
            result_deg = len(neigh_a) + len(neigh_b) - 2 - 2 * common
            cost = _edge_cost(nodes, a, b)
            return (result_deg, cost)

        i, j = min(edges, key=_merge_priority)

        # Ensure larger node eats the smaller one.
        if nodes[j].order() > nodes[i].order():
            i, j = j, i

        idx_j_in_i = int(nodes[i].find_neighbor(j))
        idx_i_in_j = int(nodes[j].find_neighbor(i))
        if idx_j_in_i < 0 or idx_i_in_j < 0:
            raise RuntimeError(
                f"[merge_preserved] Broken adjacency while merging ({i}, {j}).")

        # Optional: reverse MPS to put contraction edge near the boundary
        # for cheaper swap chains.
        if idx_j_in_i < len(nodes[i].neighbor) // 2:
            nodes[i].reverse()
            idx_j_in_i = int(nodes[i].find_neighbor(j))
        if idx_i_in_j >= len(nodes[j].neighbor) // 2:
            nodes[j].reverse()
            idx_i_in_j = int(nodes[j].find_neighbor(i))

        # --- Standard eat+merge pattern (same as compress_until_preserved) ---
        nodes[i].delete_neighbor(j)
        neighbors_j = list(nodes[j].neighbor)
        duplicate_neighbors: list[int] = []

        step_merge_err = 0.0
        for pos, k in enumerate(neighbors_j):
            if pos == idx_i_in_j:
                continue
            nodes[i].add_neighbor(k)
            if isinstance(k, int) and k in active_nodes:
                idx_i_in_k = int(nodes[k].find_neighbor(i))
                idx_j_in_k = int(nodes[k].delete_neighbor(j))
                nodes[k].add_neighbor(i, idx_j_in_k)
                if idx_i_in_k > -1:
                    duplicate_neighbors.append(k)
                    m_err = float(
                        nodes[k].merge(i, cross=idx_i_in_k > idx_j_in_k))
                    step_merge_err += m_err
                    truncation_error += m_err
            # String tokens (open indices) are simply inherited.

        _, err, _ = nodes[i].eat(nodes[j], idx_j_in_i, idx_i_in_j)
        truncation_error += abs(float(err))

        for k in duplicate_neighbors:
            m_err = float(nodes[i].merge(k, cross=False))
            step_merge_err += m_err
            truncation_error += m_err
            idx_k_in_i = int(nodes[i].find_neighbor(k))
            if idx_k_in_i >= 0 and chi > 0:
                cut_err = _cut_bondim_opt(nodes, i, idx_k_in_i, Dmax=chi)
                truncation_error += cut_err

        # Compress the merged node.
        comp_err = float(nodes[i].compress_opt())
        truncation_error += comp_err

        nodes[j].clear()
        active_nodes.discard(j)

        steps += 1
        if verbose and (steps % 10 == 0):
            adj = _build_internal_adjacency(active_nodes, nodes)
            max_deg = max((len(v) for v in adj.values()), default=0)
            print(f"[merge_preserved] step={steps}, active={len(active_nodes)}, "
                  f"max_degree={max_deg}")

    return truncation_error


@dataclass
class ChainSite:
    label: Optional[str]
    tensor: np.ndarray  # shape: [Dl, physical_dim, Dr]


@dataclass
class OneDChainMPS:
    sites: list[ChainSite]
    syndrome_labels: list[str]
    logical_labels: list[str]

    def save(self, path: str | Path) -> None:
        """Save the 1D chain MPS to a directory (NumPy arrays + JSON metadata)."""
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        meta = {
            "syndrome_labels": self.syndrome_labels,
            "logical_labels": self.logical_labels,
            "num_sites": len(self.sites),
            "site_labels": [s.label for s in self.sites],
        }
        for i, site in enumerate(self.sites):
            np.save(out / f"site_{i}.npy", site.tensor)
        with open(out / "chain_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "OneDChainMPS":
        """Load a 1D chain MPS from a directory saved by `save()`."""
        src = Path(path)
        with open(src / "chain_meta.json") as f:
            meta = json.load(f)
        sites: list[ChainSite] = []
        for i in range(meta["num_sites"]):
            tensor = np.load(src / f"site_{i}.npy")
            sites.append(ChainSite(label=meta["site_labels"][i], tensor=tensor))
        return cls(
            sites=sites,
            syndrome_labels=meta["syndrome_labels"],
            logical_labels=meta["logical_labels"],
        )

    def _logical_set(self) -> set[str]:
        return set(self.logical_labels)

    def _prepare_matrix_chain(
            self,
            syndrome_values: dict[str, float],
            fixed_logicals: dict[str, int],
            target_label: Optional[str] = None,
            target_value: Optional[int] = None,
            use_gpu: bool = False) -> list[Any]:
        xp = cupy if (use_gpu and cupy is not None) else np
        syndrome_set = set(self.syndrome_labels)
        logical_set = self._logical_set()
        mats: list[Any] = []

        for site in self.sites:
            data = xp.asarray(site.tensor)
            label = site.label
            if label in syndrome_set:
                if label not in syndrome_values:
                    raise KeyError(
                        f"Missing syndrome value for index '{label}'.")
                p = float(syndrome_values[label])
                vec = xp.asarray([1.0 - p, p], dtype=data.dtype)
                mats.append(xp.tensordot(data, vec, axes=(1, 0)))
                continue

            if label in logical_set:
                if label == target_label and target_value is not None:
                    mats.append(data[:, int(target_value), :])
                    continue
                if label in fixed_logicals:
                    mats.append(data[:, int(fixed_logicals[label]), :])
                    continue
                mats.append(data.sum(axis=1))
                continue

            mats.append(data.sum(axis=1))
        return mats

    @staticmethod
    def _logabs_contract(mats: list[Any], use_gpu: bool) -> float:
        xp = cupy if (use_gpu and cupy is not None) else np
        if not mats:
            return 0.0

        acc = mats[0]
        if acc.ndim == 0:
            return float(np.log(abs(float(acc)))) if acc != 0 else -math.inf
        if acc.ndim == 1:
            acc = acc.reshape(1, -1)

        log_scale = 0.0
        for mat in mats[1:]:
            if mat.ndim == 1:
                mat = mat.reshape(mat.shape[0], 1)
            acc = acc @ mat
            norm = float(xp.max(xp.abs(acc)))
            if norm > 0.0:
                acc = acc / norm
                log_scale += math.log(norm)

        if acc.ndim != 2:
            acc = acc.reshape(1, 1)
        value = xp.trace(acc) if acc.shape[0] == acc.shape[1] else xp.sum(acc)
        abs_value = float(xp.abs(value))
        if abs_value <= 0.0:
            return -math.inf
        return log_scale + math.log(abs_value)

    def marginal_probability(self,
                             label: str,
                             syndrome_values: dict[str, float],
                             fixed_logicals: Optional[dict[str, int]] = None,
                             use_gpu: bool = False) -> float:
        fixed_logicals = fixed_logicals or {}
        mats0 = self._prepare_matrix_chain(
            syndrome_values=syndrome_values,
            fixed_logicals=fixed_logicals,
            target_label=label,
            target_value=0,
            use_gpu=use_gpu,
        )
        mats1 = self._prepare_matrix_chain(
            syndrome_values=syndrome_values,
            fixed_logicals=fixed_logicals,
            target_label=label,
            target_value=1,
            use_gpu=use_gpu,
        )
        log0 = self._logabs_contract(mats0, use_gpu=use_gpu)
        log1 = self._logabs_contract(mats1, use_gpu=use_gpu)
        if math.isinf(log0) and math.isinf(log1):
            return 0.5
        if log1 > log0:
            return 1.0 / (1.0 + math.exp(log0 - log1))
        return math.exp(log1 - log0) / (1.0 + math.exp(log1 - log0))

    def decode_logicals(
            self,
            syndrome_values: dict[str, float],
            low_threshold: Optional[float] = None,
            high_threshold: Optional[float] = None,
            max_rounds: Optional[int] = None,
            use_gpu: bool = False) -> tuple[dict[str, int], dict[str, float]]:
        if low_threshold is None:
            low_threshold = GLOBAL_LOW_THRESHOLD
        if high_threshold is None:
            high_threshold = GLOBAL_HIGH_THRESHOLD
        _validate_threshold_window(float(low_threshold), float(high_threshold))
        if max_rounds is None:
            max_rounds = GLOBAL_MAX_CONDITIONAL_ROUNDS
        if max_rounds <= 0:
            raise ValueError("max_rounds must be > 0.")
        assignments: dict[str, int] = {}
        marginals: dict[str, float] = {}
        unknown = set(self.logical_labels)

        rounds = 0
        while unknown and rounds < max_rounds:
            rounds += 1
            round_updates: dict[str, int] = {}
            for label in list(unknown):
                prob = self.marginal_probability(
                    label=label,
                    syndrome_values=syndrome_values,
                    fixed_logicals=assignments,
                    use_gpu=use_gpu,
                )
                marginals[label] = prob
                if prob > high_threshold:
                    round_updates[label] = 1
                elif prob < low_threshold:
                    round_updates[label] = 0

            if not round_updates:
                break
            assignments.update(round_updates)
            unknown.difference_update(round_updates.keys())

        for label in list(unknown):
            prob = self.marginal_probability(
                label=label,
                syndrome_values=syndrome_values,
                fixed_logicals=assignments,
                use_gpu=use_gpu,
            )
            marginals[label] = prob
            assignments[label] = 1 if prob >= 0.5 else 0
            unknown.remove(label)

        return assignments, marginals


def _build_internal_adjacency(active_nodes: set[int],
                              nodes: dict[int, Any]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {n: set() for n in active_nodes}
    for i in active_nodes:
        for nb in nodes[i].neighbor:
            if isinstance(nb, int) and nb in active_nodes:
                adjacency[i].add(nb)
                adjacency[nb].add(i)
    return adjacency


def _path_order_from_adjacency(
        adjacency: dict[int, set[int]]) -> Optional[list[int]]:
    if not adjacency:
        return []
    if len(adjacency) == 1:
        return [next(iter(adjacency))]

    if any(len(neigh) > 2 for neigh in adjacency.values()):
        return None

    endpoints = [node for node, neigh in adjacency.items() if len(neigh) == 1]
    if len(endpoints) < 2:
        return None

    start = min(endpoints)
    order = [start]
    prev = None
    current = start
    while True:
        nxt = [n for n in adjacency[current] if n != prev]
        if not nxt:
            break
        prev, current = current, nxt[0]
        order.append(current)
        if len(order) > len(adjacency):
            return None

    if len(order) != len(adjacency):
        return None
    return order


def _node_to_site_tensor(raw: np.ndarray,
                         left_pos: Optional[int],
                         open_pos: Optional[int],
                         right_pos: Optional[int]) -> np.ndarray:
    if raw.ndim == 0:
        return raw.reshape(1, 1, 1)

    selected = [p for p in (left_pos, open_pos, right_pos) if p is not None]
    remaining = [axis for axis in range(raw.ndim) if axis not in selected]
    perm = selected + remaining
    arr = raw.transpose(perm) if perm != list(range(raw.ndim)) else raw

    cursor = 0
    if left_pos is not None:
        dl = int(arr.shape[cursor])
        cursor += 1
    else:
        dl = 1

    if open_pos is not None:
        dp = int(arr.shape[cursor])
        cursor += 1
    else:
        dp = 1

    if right_pos is not None:
        dr = int(arr.shape[cursor])
        cursor += 1
    else:
        dr = 1

    if cursor < arr.ndim:
        dp *= int(np.prod(arr.shape[cursor:], dtype=np.int64))
    return arr.reshape(dl, dp, dr)


def extract_1d_chain(active_nodes: set[int],
                     nodes: dict[int, Any],
                     syndrome_label_set: set[str],
                     logical_label_set: set[str]) -> Optional[OneDChainMPS]:
    """Extract a path-ordered 1D chain from active preserved nodes.

    After ``merge_preserved_to_path``, each node may carry *multiple* open
    tokens (syndrome/logical labels).  We decompose such multi-label nodes
    into multiple ChainSite entries by rearranging the MPS so that:
      - the left-bond position is at the MPS head,
      - the right-bond position is at the MPS tail,
      - all open-token positions are in between (each becoming one site).
    """
    adjacency = _build_internal_adjacency(active_nodes, nodes)
    order = _path_order_from_adjacency(adjacency)
    if order is None:
        return None

    sites: list[ChainSite] = []
    syndrome_labels: list[str] = []
    logical_labels: list[str] = []
    expected_open_labels = syndrome_label_set | logical_label_set

    for idx, node_id in enumerate(order):
        prev_node = order[idx - 1] if idx > 0 else None
        next_node = order[idx + 1] if idx < len(order) - 1 else None

        node = nodes[node_id]

        # Identify left-bond, right-bond, and open-token MPS positions.
        left_mps_pos: Optional[int] = None
        right_mps_pos: Optional[int] = None
        if prev_node is not None:
            pos = int(node.find_neighbor(prev_node))
            left_mps_pos = pos if pos >= 0 else None
        if next_node is not None:
            pos = int(node.find_neighbor(next_node))
            right_mps_pos = pos if pos >= 0 else None

        # Collect open tokens with their MPS positions.
        open_entries: list[tuple[int, str, str]] = []  # (mps_pos, token, label)
        for mps_pos_iter, nb in enumerate(node.neighbor):
            if isinstance(nb, int) and nb in active_nodes:
                continue
            if not isinstance(nb, str):
                continue
            label = out_token_to_index(nb)
            if label in expected_open_labels:
                open_entries.append((mps_pos_iter, nb, label))

        if len(open_entries) == 0:
            return None

        # --- Single open token: original fast path (no rearrangement) ---
        if len(open_entries) == 1:
            mps_pos_open, _tok, label = open_entries[0]
            raw = mps_to_raw(node.mps)
            site_tensor = _node_to_site_tensor(
                raw, left_mps_pos, mps_pos_open, right_mps_pos)
            if site_tensor.shape[1] != 2:
                return None
            if label in syndrome_label_set:
                syndrome_labels.append(label)
            if label in logical_label_set:
                logical_labels.append(label)
            sites.append(ChainSite(label=label, tensor=site_tensor))
            continue

        # --- Multi-label node: rearrange MPS then extract per-token sites ---
        # Target MPS order:
        #   [left_bond] [open_0] [open_1] ... [open_k] [right_bond]
        # We use the node's move() method which performs swap chains correctly.
        import copy
        work_node = copy.deepcopy(node)

        # Build desired target layout: which original positions go where.
        desired_orig: list[int] = []
        if left_mps_pos is not None:
            desired_orig.append(left_mps_pos)
        for (mp, _t, _l) in open_entries:
            desired_orig.append(mp)
        if right_mps_pos is not None:
            desired_orig.append(right_mps_pos)

        n_mps = len(work_node.mps)
        # pos_map[orig] = current MPS position of the core that was originally at `orig`.
        pos_map = list(range(n_mps))
        # inv_map[cur] = which original position is currently at MPS slot `cur`.
        inv_map = list(range(n_mps))

        for target_slot, orig in enumerate(desired_orig):
            cur = pos_map[orig]
            if cur == target_slot:
                continue
            # Use move() which does consecutive swaps.
            work_node.move(cur, target_slot)
            # Update tracking for all displaced positions.
            if cur > target_slot:
                # Elements in [target_slot, cur-1] shifted right by 1.
                moved_orig = inv_map[cur]
                for s in range(cur, target_slot, -1):
                    inv_map[s] = inv_map[s - 1]
                    pos_map[inv_map[s]] = s
                inv_map[target_slot] = moved_orig
                pos_map[moved_orig] = target_slot
            else:
                # Elements in [cur+1, target_slot] shifted left by 1.
                moved_orig = inv_map[cur]
                for s in range(cur, target_slot):
                    inv_map[s] = inv_map[s + 1]
                    pos_map[inv_map[s]] = s
                inv_map[target_slot] = moved_orig
                pos_map[moved_orig] = target_slot

        # Now the MPS is rearranged. Extract chain sites.
        # Layout: [left_bond_core?] [open_0] ... [open_k] [right_bond_core?]
        mps_cores = list(work_node.mps)

        core_start = 0
        # Absorb the left-bond core into the first open core.
        if left_mps_pos is not None:
            left_core = mps_cores[core_start]  # (Dl, d_bond, Dr)
            next_core = mps_cores[core_start + 1]  # (Dr, d_open, Dr2)
            merged = np.einsum("ijk,kab->ijab", left_core, next_core)
            # (Dl, d_bond, d_open, Dr2) → (Dl*d_bond, d_open, Dr2)
            merged = merged.reshape(
                left_core.shape[0] * left_core.shape[1],
                next_core.shape[1], next_core.shape[2])
            mps_cores[core_start + 1] = merged
            core_start += 1

        core_end = len(mps_cores)
        # Absorb the right-bond core into the last open core.
        if right_mps_pos is not None:
            right_core = mps_cores[core_end - 1]  # (Dl, d_bond, Dr)
            prev_core = mps_cores[core_end - 2]    # (Dl2, d_open, Dr_prev)
            merged = np.einsum("ijk,kab->ijab", prev_core, right_core)
            # (Dl2, d_open, d_bond, Dr) → (Dl2, d_open, d_bond*Dr)
            merged = merged.reshape(
                prev_core.shape[0], prev_core.shape[1],
                right_core.shape[1] * right_core.shape[2])
            mps_cores[core_end - 2] = merged
            core_end -= 1

        open_cores = mps_cores[core_start:core_end]
        if len(open_cores) != len(open_entries):
            return None

        for ci, (_, _tok, label) in enumerate(open_entries):
            core = open_cores[ci]  # shape (Dl, d_phys, Dr)
            if core.shape[1] != 2:
                return None
            if label in syndrome_label_set:
                syndrome_labels.append(label)
            if label in logical_label_set:
                logical_labels.append(label)
            sites.append(ChainSite(label=label, tensor=core))

    return OneDChainMPS(
        sites=sites,
        syndrome_labels=syndrome_labels,
        logical_labels=logical_labels,
    )


class CompressedTNDecoder:
    """Online decoder that substitutes syndrome values FIRST, then contracts.

    After offline contraction-swap-merge (phase 1) removes non-preserved
    nodes, we store the residual graph of preserved nodes (syndrome +
    logical).  At online decode time:

    1. Substitute concrete syndrome values (0/1) into each node's MPS,
       collapsing the syndrome physical index (dim 2 → scalar matrix).
    2. Contract the now-simplified residual graph to a single MPS.
    3. Read the logical marginal from the final 1-site MPS.

    This avoids the exponential bond growth that occurs when trying to
    contract all preserved nodes *before* knowing the syndrome.
    """

    def __init__(
        self,
        active_nodes: set[int],
        nodes: dict[int, Any],
        syndrome_label_set: set[str],
        logical_label_set: set[str],
        chi: int = 0,  # online chi for residual graph contraction
    ):
        import copy
        # Store deep copies of the preserved graph for online reuse.
        # Each decode call will work on a fresh copy.
        self._active_ids = set(active_nodes)
        self._chi = chi
        self._node_snapshots: dict[int, Any] = {}
        for nid in active_nodes:
            self._node_snapshots[nid] = copy.deepcopy(nodes[nid])

        self._syndrome_label_set = set(syndrome_label_set)
        self._logical_label_set = set(logical_label_set)
        self.phase2_truncation_error = 0.0

        # Build label → (node_id, position) mapping for fast substitution
        self.syndrome_labels: list[str] = []
        self.logical_labels: list[str] = []
        self._syndrome_node_pos: dict[str, tuple[int, int]] = {}
        self._logical_node_pos: dict[str, tuple[int, int]] = {}

        for nid in active_nodes:
            node = self._node_snapshots[nid]
            for pos, nb in enumerate(node.neighbor):
                if isinstance(nb, str):
                    label = out_token_to_index(nb)
                    if label in syndrome_label_set:
                        self.syndrome_labels.append(label)
                        self._syndrome_node_pos[label] = (nid, pos)
                    elif label in logical_label_set:
                        self.logical_labels.append(label)
                        self._logical_node_pos[label] = (nid, pos)

    def decode_logicals(
        self,
        syndrome_values: dict[str, float],
        use_gpu: bool = False,
    ) -> tuple[dict[str, int], dict[str, float]]:
        """Decode by substituting syndrome, then contracting residual graph."""
        marginals: dict[str, float] = {}
        assignments: dict[str, int] = {}

        for log_label in self.logical_labels:
            prob = self._marginal_for(log_label, syndrome_values)
            marginals[log_label] = prob
            assignments[log_label] = 1 if prob >= 0.5 else 0

        return assignments, marginals

    def _marginal_for(
        self, logical_label: str, syndrome_values: dict[str, float]
    ) -> float:
        """Compute P(logical=1 | syndrome) by substituting then contracting."""
        log0 = self._substitute_and_contract(logical_label, 0, syndrome_values)
        log1 = self._substitute_and_contract(logical_label, 1, syndrome_values)

        if math.isinf(log0) and math.isinf(log1):
            return 0.5
        if log1 > log0:
            return 1.0 / (1.0 + math.exp(log0 - log1))
        return math.exp(log1 - log0) / (1.0 + math.exp(log1 - log0))

    def _substitute_and_contract(
        self,
        logical_label: str,
        logical_value: int,
        syndrome_values: dict[str, float],
    ) -> float:
        """Substitute syndrome+logical values, contract residual graph in MPS form.

        Steps:
        1. Deep-copy the stored node graph.
        2. For each syndrome site, contract phys index with [1-s, s] → 2D core.
        3. For the target logical, select phys index = logical_value → 2D core.
        4. For other logicals, sum over phys index (marginalise) → 2D core.
        5. Collapse all 2D cores by absorbing into adjacent 3D cores.
        6. Contract the residual graph using MPS eat operations.
        7. Return log|value| of the final scalar.
        """
        import copy
        nodes = {nid: copy.deepcopy(n) for nid, n in self._node_snapshots.items()}
        remaining = set(self._active_ids)

        # Step 1-4: Substitute concrete values into MPS cores.
        _COLLAPSED = "__COLLAPSED__"
        for nid in list(remaining):
            node = nodes[nid]
            for pos in range(len(node.neighbor)):
                nb = node.neighbor[pos]
                if not isinstance(nb, str):
                    continue
                label = out_token_to_index(nb)
                if label in self._syndrome_label_set:
                    p = float(syndrome_values.get(label, 0.0))
                    vec = np.array([1.0 - p, p], dtype=node.mps[pos].dtype)
                    node.mps[pos] = np.tensordot(
                        node.mps[pos], vec, axes=(1, 0))
                    node.neighbor[pos] = _COLLAPSED
                elif label == logical_label:
                    node.mps[pos] = node.mps[pos][:, logical_value, :]
                    node.neighbor[pos] = _COLLAPSED
                elif label in self._logical_label_set:
                    node.mps[pos] = node.mps[pos].sum(axis=1)
                    node.neighbor[pos] = _COLLAPSED

        # Step 5: Collapse 2D cores by absorbing into adjacent cores.
        # This avoids exponential blowup from reconstructing raw tensors.
        log_scale = 0.0
        for nid in list(remaining):
            node = nodes[nid]
            # Process from right to left: absorb each 2D core into its left neighbor.
            i = len(node.mps) - 1
            while i >= 0:
                if node.mps[i].ndim == 2:
                    if i > 0:
                        left = node.mps[i - 1]
                        right = node.mps[i]
                        if left.ndim == 3:
                            node.mps[i - 1] = np.einsum(
                                'ijk,kl->ijl', left, right)
                        else:
                            node.mps[i - 1] = left @ right
                        node.mps.pop(i)
                        node.neighbor = np.delete(node.neighbor, i)
                    elif len(node.mps) > 1:
                        left = node.mps[i]
                        right_core = node.mps[i + 1]
                        if right_core.ndim == 3:
                            node.mps[i + 1] = np.einsum(
                                'ij,jkl->ikl', left, right_core)
                        else:
                            node.mps[i + 1] = left @ right_core
                        node.mps.pop(i)
                        node.neighbor = np.delete(node.neighbor, i)
                        continue  # re-check position i (now different core)
                    else:
                        # Single 2D core: (1, 1) — scalar node
                        val = float(np.abs(node.mps[0].ravel()[0]))
                        if val > 0:
                            log_scale += math.log(val)
                        else:
                            return -math.inf
                        node.mps = []
                        node.neighbor = np.array([], dtype=node.neighbor.dtype)
                        break
                i -= 1

            # If node has no MPS cores, it's been fully contracted to a scalar.
            if len(node.mps) == 0:
                remaining.discard(nid)
                continue

            # Normalize to prevent numerical overflow/underflow.
            node.cano = min(node.cano, len(node.mps) - 1)
            node.cano = max(node.cano, 0)
            node.left_canonical()
            norm = np.linalg.norm(node.mps[node.cano])
            if norm > 0:
                log_scale += math.log(norm)
                node.mps[node.cano] = node.mps[node.cano] / norm
            else:
                return -math.inf

        if len(remaining) == 0:
            return log_scale

        # Step 6: Contract the residual graph using MPS eat operations.
        # Same pattern as compress_until_preserved but for the residual graph.
        while len(remaining) > 1:
            # Find an edge: a pair of nodes that share an internal neighbor.
            pair_found = False
            for nid_i in list(remaining):
                node_i = nodes[nid_i]
                for pos_i, nb in enumerate(node_i.neighbor):
                    if isinstance(nb, (int, np.integer)) and int(nb) in remaining and int(nb) != nid_i:
                        nid_j = int(nb)
                        node_j = nodes[nid_j]
                        pos_j = int(node_j.find_neighbor(nid_i))
                        if pos_j < 0:
                            continue

                        # Update neighbor bookkeeping (mirrors compress_until_preserved)
                        node_i.delete_neighbor(nid_j)
                        neighbors_j = list(node_j.neighbor)
                        duplicate_neighbors: list[int] = []

                        for p, k in enumerate(neighbors_j):
                            if p == pos_j:
                                continue
                            node_i.add_neighbor(k)
                            if isinstance(k, (int, np.integer)) and int(k) in remaining:
                                k = int(k)
                                idx_i_in_k = int(nodes[k].find_neighbor(nid_i))
                                idx_j_in_k = int(nodes[k].delete_neighbor(nid_j))
                                nodes[k].add_neighbor(nid_i, idx_j_in_k)
                                if idx_i_in_k > -1:
                                    duplicate_neighbors.append(k)
                                    node_err = float(
                                        nodes[k].merge(nid_i,
                                                       cross=idx_i_in_k > idx_j_in_k))

                        # Use ORIGINAL MPS positions saved before neighbor
                        # modifications.  delete_neighbor / add_neighbor only
                        # touch the neighbor array, not the MPS cores, so the
                        # original positions still index the correct cores.
                        lognorm, err, sign = node_i.eat(
                            node_j, pos_i, pos_j)
                        log_scale += lognorm

                        for k in duplicate_neighbors:
                            node_i.merge(k, cross=False)

                        node_i.compress_opt()
                        # Re-normalize after contraction.
                        if len(node_i.mps) > 0:
                            node_i.left_canonical()
                            norm = np.linalg.norm(node_i.mps[node_i.cano])
                            if norm > 0:
                                log_scale += math.log(norm)
                                node_i.mps[node_i.cano] = node_i.mps[node_i.cano] / norm

                        node_j.clear()
                        remaining.remove(nid_j)
                        pair_found = True
                        break
                if pair_found:
                    break
            if not pair_found:
                break

        # Step 7: Compute log|value| of the final scalar.
        final_id = next(iter(remaining))
        final_node = nodes[final_id]
        if len(final_node.mps) == 0:
            return log_scale
        lognorm, sign = final_node.lognorm()
        return log_scale + lognorm


@dataclass
class CompileResult:
    chain: Optional[OneDChainMPS]
    compressed_tn: Optional[CompressedTNDecoder]
    offline_stats: OfflineCompressionStats


def compile_to_1d_chain(
        tn: TensorNetwork,
        preserve_inds: set[str],
        syndrome_inds: set[str],
        logical_inds: set[str],
        chi: Optional[int] = None,
        cutoff: float = 1e-12,
        max_steps: int = 100000,
        verbose: bool = False) -> CompileResult:
    if chi is None:
        chi = GLOBAL_BOND_DIM
    pairwise = build_pairwise_graph_from_tn(tn, preserve_inds=preserve_inds)
    nodes = build_mps_nodes(pairwise, chi=chi, cutoff=cutoff)

    offline = compress_until_preserved(
        nodes=nodes,
        preserved_nodes=set(pairwise.preserved_nodes),
        max_steps=max_steps,
        reverse=True,
        compress_each_step=True,
        verbose=verbose,
        chi=chi,
    )

    # Phase 1.5: try direct extraction first; if the residual graph is
    # already a path (degree <= 2 everywhere) we skip the merging step.
    chain = extract_1d_chain(
        active_nodes=offline.active_nodes,
        nodes=offline.nodes,
        syndrome_label_set=syndrome_inds,
        logical_label_set=logical_inds,
    )

    if chain is None:
        # Phase 2: merge preserved nodes until the graph becomes a 1D path.
        if verbose:
            adj = _build_internal_adjacency(offline.active_nodes, offline.nodes)
            max_deg = max((len(v) for v in adj.values()), default=0)
            print(f"[compile] chain extraction failed (max_degree={max_deg}), "
                  f"starting preserved-node merging ...")

        merge_err = merge_preserved_to_path(
            nodes=offline.nodes,
            active_nodes=offline.active_nodes,
            chi=chi,
            max_steps=max_steps,
            verbose=verbose,
        )
        offline.stats.truncation_error += merge_err

        # Try extraction again after merging.
        chain = extract_1d_chain(
            active_nodes=offline.active_nodes,
            nodes=offline.nodes,
            syndrome_label_set=syndrome_inds,
            logical_label_set=logical_inds,
        )

    # Build compressed TN decoder as fallback when 1D chain isn't available
    compressed_tn: Optional[CompressedTNDecoder] = None
    if chain is None:
        if verbose:
            print("[compile] chain extraction still failed after merging, "
                  "falling back to CompressedTNDecoder")
        compressed_tn = CompressedTNDecoder(
            active_nodes=offline.active_nodes,
            nodes=offline.nodes,
            syndrome_label_set=syndrome_inds,
            logical_label_set=logical_inds,
        )

    return CompileResult(
        chain=chain,
        compressed_tn=compressed_tn,
        offline_stats=offline.stats,
    )


def timed_chain_decode(chain: OneDChainMPS,
                       syndrome_values: dict[str, float],
                       use_gpu: bool = False,
                       low_threshold: Optional[float] = None,
                       high_threshold: Optional[float] = None,
                       max_rounds: Optional[int] = None) -> tuple[
                           dict[str, int], dict[str, float], float]:
    """Decode logicals and return `(assignments, marginals, latency_ms)`."""
    if use_gpu and cupy is not None:
        start = cupy.cuda.Event()
        end = cupy.cuda.Event()
        start.record()
        assignments, marginals = chain.decode_logicals(
            syndrome_values=syndrome_values,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            max_rounds=max_rounds,
            use_gpu=True,
        )
        end.record()
        end.synchronize()
        latency_ms = float(cupy.cuda.get_elapsed_time(start, end))
        return assignments, marginals, latency_ms

    t0 = time.perf_counter()
    assignments, marginals = chain.decode_logicals(
        syndrome_values=syndrome_values,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        max_rounds=max_rounds,
        use_gpu=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1e3
    return assignments, marginals, float(latency_ms)
