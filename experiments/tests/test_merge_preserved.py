#!/usr/bin/env python3
"""Quick smoke test for merge_preserved_to_path and multi-label extract_1d_chain."""

from __future__ import annotations

import sys
from pathlib import Path

# Add workspace source to path.
WORKSPACE = Path(__file__).resolve().parents[2]
QEC_PY = WORKSPACE / "cudaqx" / "libs" / "qec" / "python"
sys.path.insert(0, str(QEC_PY))

import numpy as np
from quimb.tensor import Tensor, TensorNetwork

from cudaq_qec.plugins.decoders.tensor_network_utils.mps_decoder_core import (
    OUT_PREFIX,
    _build_internal_adjacency,
    _path_order_from_adjacency,
    build_pairwise_graph_from_tn,
    build_mps_nodes,
    compile_to_1d_chain,
    compress_until_preserved,
    count_internal_edges,
    extract_1d_chain,
    merge_preserved_to_path,
)


def _make_small_tn_with_dense_preserved():
    """Create a small TN where preserved nodes form a dense (non-path) graph.

    Structure: 4 syndrome params + 1 logical, connected through 5 error nodes.
    This should create a situation where after compress_until_preserved, the
    preserved nodes have degree > 2.
    """
    # Simple repetition-code-like structure but dense enough.
    # 3 checks, 4 errors, 1 logical
    H = np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
    ], dtype=np.float64)

    logical = np.array([[1, 0, 0, 1]], dtype=np.float64)

    # Build TN manually.
    check_inds = [f"s_{i}" for i in range(3)]
    error_inds = [f"e_{i}" for i in range(4)]
    logical_inds = ["l_0"]
    logical_obs_inds = ["obs"]
    syn_param_inds = [f"syn_param_{i}" for i in range(3)]

    tensors = []

    # Parity check tensors
    for row_idx in range(3):
        cols = np.where(H[row_idx] == 1)[0]
        inds = [check_inds[row_idx]] + [error_inds[c] for c in cols]
        n = len(inds)
        data = np.zeros((2,) * n, dtype=np.float64)
        for idx_tuple in np.ndindex(*([2] * n)):
            if sum(idx_tuple) % 2 == 0:
                data[idx_tuple] = 1.0
        tensors.append(Tensor(data=data, inds=inds, tags=[f"CHECK_{row_idx}"]))

    # Logical tensor
    log_cols = np.where(logical[0] == 1)[0]
    log_inds = [logical_inds[0]] + [error_inds[c] for c in log_cols]
    n = len(log_inds)
    data = np.zeros((2,) * n, dtype=np.float64)
    for idx_tuple in np.ndindex(*([2] * n)):
        if sum(idx_tuple) % 2 == 0:
            data[idx_tuple] = 1.0
    tensors.append(Tensor(data=data, inds=log_inds, tags=["LOG_0"]))

    # Logical observable connection
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64)
    tensors.append(Tensor(data=hadamard, inds=(logical_inds[0], logical_obs_inds[0]),
                          tags=["LOG_OBS_0"]))

    # Syndrome parameters
    for i in range(3):
        tensors.append(Tensor(data=hadamard, inds=(check_inds[i], syn_param_inds[i]),
                              tags=[f"SYN_{i}", "SYNDROME_PARAM"]))

    # Noise model (independent bit-flip p=0.1)
    p = 0.1
    for i in range(4):
        noise_tensor = np.array([1.0 - p, p], dtype=np.float64)
        tensors.append(Tensor(data=noise_tensor, inds=(error_inds[i],),
                              tags=[f"NOISE_{i}"]))

    tn = TensorNetwork(tensors)
    preserve_inds = set(syn_param_inds + logical_obs_inds)
    syndrome_inds = set(syn_param_inds)
    logical_ind_set = set(logical_obs_inds)

    return tn, preserve_inds, syndrome_inds, logical_ind_set


def test_compile_produces_chain():
    """Test that compile_to_1d_chain with merging produces a valid chain."""
    tn, preserve_inds, syndrome_inds, logical_inds = _make_small_tn_with_dense_preserved()

    result = compile_to_1d_chain(
        tn=tn.copy(),
        preserve_inds=preserve_inds,
        syndrome_inds=syndrome_inds,
        logical_inds=logical_inds,
        chi=32,
        verbose=True,
    )

    print(f"\nchain_ready = {result.chain is not None}")
    print(f"compressed_tn_ready = {result.compressed_tn is not None}")
    print(f"offline_stats: nodes={result.offline_stats.active_nodes}, "
          f"edges={result.offline_stats.active_edges}, "
          f"trunc_err={result.offline_stats.truncation_error:.3e}")

    if result.chain is not None:
        chain = result.chain
        print(f"chain sites: {len(chain.sites)}")
        print(f"syndrome_labels: {chain.syndrome_labels}")
        print(f"logical_labels: {chain.logical_labels}")
        for i, site in enumerate(chain.sites):
            print(f"  site[{i}]: label={site.label}, "
                  f"tensor.shape={site.tensor.shape}")

        # Decode with all-zero syndrome.
        syndrome_values = {label: 0.0 for label in chain.syndrome_labels}
        assignments, marginals = chain.decode_logicals(
            syndrome_values=syndrome_values)
        print(f"\nall-zero syndrome:")
        print(f"  assignments: {assignments}")
        print(f"  marginals: {marginals}")
    else:
        print("Chain extraction failed!")
        if result.compressed_tn is not None:
            print("Using compressed TN decoder instead...")
            syndrome_values = {label: 0.0 for label in syndrome_inds}
            assignments, marginals = result.compressed_tn.decode_logicals(
                syndrome_values=syndrome_values)
            print(f"  assignments: {assignments}")
            print(f"  marginals: {marginals}")


def test_merge_preserved_directly():
    """Test merge_preserved_to_path on the residual graph after compression."""
    tn, preserve_inds, syndrome_inds, logical_inds = _make_small_tn_with_dense_preserved()

    pairwise = build_pairwise_graph_from_tn(tn.copy(), preserve_inds=preserve_inds)
    nodes = build_mps_nodes(pairwise, chi=32, cutoff=1e-12)

    print(f"Initial: {len(nodes)} nodes")

    preserved = set(pairwise.preserved_nodes)
    offline = compress_until_preserved(
        nodes=nodes,
        preserved_nodes=preserved,
        chi=32,
        verbose=True,
    )

    print(f"\nAfter phase 1: {len(offline.active_nodes)} active nodes")
    adj = _build_internal_adjacency(offline.active_nodes, offline.nodes)
    for nid, neigh in sorted(adj.items()):
        node = offline.nodes[nid]
        open_tokens = [nb for nb in node.neighbor if isinstance(nb, str)]
        print(f"  node {nid}: degree={len(neigh)}, "
              f"open_tokens={len(open_tokens)}, mps_order={node.order()}")

    path_order = _path_order_from_adjacency(adj)
    print(f"Path order before merge: {path_order}")

    # Now merge.
    err = merge_preserved_to_path(
        nodes=offline.nodes,
        active_nodes=offline.active_nodes,
        chi=32,
        verbose=True,
    )
    print(f"\nAfter merge: {len(offline.active_nodes)} active nodes, "
          f"trunc_err={err:.3e}")

    adj2 = _build_internal_adjacency(offline.active_nodes, offline.nodes)
    for nid, neigh in sorted(adj2.items()):
        node = offline.nodes[nid]
        open_tokens = [nb for nb in node.neighbor if isinstance(nb, str)]
        print(f"  node {nid}: degree={len(neigh)}, "
              f"open_tokens={len(open_tokens)}, mps_order={node.order()}")

    path_order2 = _path_order_from_adjacency(adj2)
    print(f"Path order after merge: {path_order2}")

    # Try chain extraction.
    chain = extract_1d_chain(
        active_nodes=offline.active_nodes,
        nodes=offline.nodes,
        syndrome_label_set=syndrome_inds,
        logical_label_set=logical_inds,
    )
    print(f"\nChain extracted: {chain is not None}")
    if chain is not None:
        for i, site in enumerate(chain.sites):
            print(f"  site[{i}]: label={site.label}, shape={site.tensor.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: merge_preserved_to_path (direct)")
    print("=" * 60)
    test_merge_preserved_directly()

    print("\n" + "=" * 60)
    print("TEST 2: compile_to_1d_chain (end-to-end)")
    print("=" * 60)
    test_compile_produces_chain()
