# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import sys
import numpy as np
import pytest

import cudaq_qec as qec

pytestmark = pytest.mark.skipif(sys.version_info < (3, 11),
                                reason="Requires Python >= 3.11")


def _gpu_available():
    import cupy
    try:
        return cupy.cuda.is_available()
    except cupy.cuda.runtime.CUDARuntimeError:
        return False


def _simple_case():
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
    logical = np.array([[1, 0, 1]], dtype=np.float64)
    noise = [0.1, 0.2, 0.3]
    return H, logical, noise


def test_mps_decoder_init_smoke():
    H, logical, noise = _simple_case()
    from cudaq_qec.plugins.decoders.tensor_network_mps_decoder import (
        set_global_mps_decoder_config, GLOBAL_BOND_DIM)
    set_global_mps_decoder_config(bond_dim=GLOBAL_BOND_DIM)
    decoder = qec.get_decoder("tensor_network_mps_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)
    assert hasattr(decoder, "offline_stats")
    assert hasattr(decoder, "syndrome_param_inds")
    assert len(decoder.syndrome_param_inds) == H.shape[0]

    if _gpu_available():
        assert "cuda" in decoder.contractor_config.device
    else:
        assert decoder.contractor_config.device == "cpu"


def test_mps_decoder_decode_single():
    H, logical, noise = _simple_case()
    decoder = qec.get_decoder("tensor_network_mps_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)
    out = decoder.decode([1.0, 0.0])
    assert out.converged
    assert isinstance(out.result, list)
    assert 0.0 <= out.result[0] <= 1.0


def test_mps_decoder_decode_batch():
    H, logical, noise = _simple_case()
    decoder = qec.get_decoder("tensor_network_mps_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)
    batch = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    out = decoder.decode_batch(batch)
    assert len(out) == batch.shape[0]
    assert all(result.converged for result in out)
    assert all(0.0 <= result.result[0] <= 1.0 for result in out)


def test_mps_decoder_latency_api():
    H, logical, noise = _simple_case()
    decoder = qec.get_decoder("tensor_network_mps_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise)
    stats = decoder.benchmark_decode_latency([1.0, 0.0], warmup=1, repeats=3)
    assert {"avg_ms", "min_ms", "max_ms"} <= set(stats.keys())
    assert stats["avg_ms"] >= 0.0


def test_mps_decoder_explicit_bond_dim():
    H, logical, noise = _simple_case()
    decoder = qec.get_decoder("tensor_network_mps_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise,
                              bond_dim=24)
    assert decoder.bond_dim == 24


def test_mps_decoder_bond_dim_priority_over_offline_chi():
    H, logical, noise = _simple_case()
    decoder = qec.get_decoder("tensor_network_mps_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise,
                              bond_dim=24,
                              offline_chi=8)
    assert decoder.bond_dim == 24
    assert decoder.offline_chi == 24


def test_mps_decoder_matches_exact_simple():
    """Verify MPS decoder gives same probabilities as exact decoder on a small code."""
    H, logical, noise = _simple_case()
    exact_dec = qec.get_decoder("tensor_network_decoder",
                                H,
                                logical_obs=logical,
                                noise_model=noise)
    mps_dec = qec.get_decoder("tensor_network_mps_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise,
                              bond_dim=32,
                              verbose=False)
    syndromes = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    for syn in syndromes:
        exact_out = exact_dec.decode(syn)
        mps_out = mps_dec.decode(syn)
        assert abs(exact_out.result[0] - mps_out.result[0]) < 1e-6, (
            f"syndrome={syn}: exact={exact_out.result[0]:.8f}, "
            f"mps={mps_out.result[0]:.8f}")


def test_mps_decoder_matches_exact_steane():
    """Verify MPS decoder matches exact on a Steane-like code (higher check count)."""
    # Steane [[7,1,3]] code parity check matrix (Hz)
    H = np.array([
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
    ], dtype=np.float64)
    logical = np.array([[1, 1, 1, 0, 0, 0, 0]], dtype=np.float64)
    noise = [0.05] * 7

    exact_dec = qec.get_decoder("tensor_network_decoder",
                                H,
                                logical_obs=logical,
                                noise_model=noise)
    mps_dec = qec.get_decoder("tensor_network_mps_decoder",
                              H,
                              logical_obs=logical,
                              noise_model=noise,
                              bond_dim=64,
                              verbose=False)

    rng = np.random.default_rng(42)
    for _ in range(10):
        error = (rng.random(7) < 0.05).astype(np.int8)
        syn = ((error @ H.T.astype(np.int8)) % 2).astype(np.float64).tolist()
        exact_out = exact_dec.decode(syn)
        mps_out = mps_dec.decode(syn)
        assert abs(exact_out.result[0] - mps_out.result[0]) < 1e-4, (
            f"syndrome={syn}: exact={exact_out.result[0]:.8f}, "
            f"mps={mps_out.result[0]:.8f}")


def test_mps_decoder_global_threshold_validation():
    from cudaq_qec.plugins.decoders.tensor_network_mps_decoder import (
        set_global_mps_decoder_config, )

    with pytest.raises(ValueError):
        set_global_mps_decoder_config(low_threshold=0.7, high_threshold=0.6)
