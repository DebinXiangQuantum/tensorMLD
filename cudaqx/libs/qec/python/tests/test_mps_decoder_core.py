# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest

pytest.importorskip("quimb")
pytest.importorskip("scipy")


def _load_core_module():
    root = Path(__file__).resolve().parents[1]
    core_path = root / "cudaq_qec/plugins/decoders/tensor_network_utils/mps_decoder_core.py"
    spec = importlib.util.spec_from_file_location("mps_decoder_core", core_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = _load_core_module()


def test_parity_exact_decomposition_roundtrip():
    omega = 0.73
    data = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for idx in np.ndindex(data.shape):
        parity = sum(idx) & 1
        data[idx] = np.exp(omega if parity == 0 else -omega)

    mps = core.decompose_tensor_to_mps(data, chi=32, cutoff=1.0e-12)
    rebuilt = core.mps_to_raw(mps)
    np.testing.assert_allclose(rebuilt, data, rtol=0.0, atol=1.0e-10)


def test_delta_exact_decomposition_roundtrip():
    data = core.make_delta_tensor(dim=2, order=6)
    mps = core.decompose_tensor_to_mps(data, chi=32, cutoff=1.0e-12)
    rebuilt = core.mps_to_raw(mps)
    np.testing.assert_allclose(rebuilt, data, rtol=0.0, atol=1.0e-12)


def test_chain_online_decode_marginal_probability():
    # Site-0: syndrome selector, maps syn=0 -> bond state 0, syn=1 -> bond state 1.
    syn_site = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    # Site-1: logical site.
    logical_site = np.array(
        [
            [[0.8], [0.2]],  # if incoming bond is 0
            [[0.1], [0.9]],  # if incoming bond is 1
        ],
        dtype=np.float64,
    )
    chain = core.OneDChainMPS(
        sites=[
            core.ChainSite(label="syn_param_0", tensor=syn_site),
            core.ChainSite(label="obs", tensor=logical_site),
        ],
        syndrome_labels=["syn_param_0"],
        logical_labels=["obs"],
    )

    p_obs_when_syn0 = chain.marginal_probability(
        label="obs",
        syndrome_values={"syn_param_0": 0.0},
        fixed_logicals={},
        use_gpu=False,
    )
    p_obs_when_syn1 = chain.marginal_probability(
        label="obs",
        syndrome_values={"syn_param_0": 1.0},
        fixed_logicals={},
        use_gpu=False,
    )

    assert abs(p_obs_when_syn0 - 0.2) < 1.0e-12
    assert abs(p_obs_when_syn1 - 0.9) < 1.0e-12


def test_global_threshold_validation():
    with pytest.raises(ValueError):
        core.set_global_decoder_config(low_threshold=0.8, high_threshold=0.7)
    with pytest.raises(ValueError):
        core.set_global_decoder_config(low_threshold=-0.1, high_threshold=0.7)
    with pytest.raises(ValueError):
        core.set_global_decoder_config(low_threshold=0.1, high_threshold=1.1)


def test_decode_logicals_parallel_round_update():
    chain = core.OneDChainMPS(
        sites=[],
        syndrome_labels=[],
        logical_labels=["l0", "l1", "l2"],
    )
    seen_fixed: list[dict[str, int]] = []

    def _fake_marginal(self, label, syndrome_values, fixed_logicals, use_gpu):
        del self, label, syndrome_values, use_gpu
        snapshot = dict(fixed_logicals)
        seen_fixed.append(snapshot)
        # If updates leak inside the same round this becomes < low_threshold.
        return 0.99 if not snapshot else 0.01

    chain.marginal_probability = types.MethodType(_fake_marginal, chain)

    assignments, _ = chain.decode_logicals(
        syndrome_values={},
        low_threshold=0.4,
        high_threshold=0.6,
        max_rounds=1,
        use_gpu=False,
    )
    assert len(seen_fixed) == 3
    assert all(snapshot == {} for snapshot in seen_fixed)
    assert assignments == {"l0": 1, "l1": 1, "l2": 1}
