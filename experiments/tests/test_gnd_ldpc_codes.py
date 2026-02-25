from __future__ import annotations

import numpy as np
import pytest

from experiments.codes.gnd_ldpc_codes import (
    _resolve_ldpc_file_path,
    load_ldpc_css_code,
    split_gnd_stabilizer_to_hx_hz,
)


def test_split_gnd_stabilizer_to_hx_hz() -> None:
    # 0:I, 1:X, 2:Z, 3:Y.
    g = np.asarray([
        [1, 0, 1, 0],  # X-row -> Hx
        [0, 2, 0, 2],  # Z-row -> Hz
        [3, 1, 0, 0],  # contains X/Y -> Hx
        [0, 0, 0, 0],  # no X-part -> Hz
    ],
                   dtype=np.int64)

    hx, hz = split_gnd_stabilizer_to_hx_hz(g)
    np.testing.assert_array_equal(
        hx,
        np.asarray([
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        ],
                   dtype=np.int64),
    )
    np.testing.assert_array_equal(
        hz,
        np.asarray([
            [0, 1, 0, 1],
            [0, 0, 0, 0],
        ],
                   dtype=np.int64),
    )


def test_resolve_ldpc_path() -> None:
    path = _resolve_ldpc_file_path(
        n=25, d=4, k=3, seed=0, code_dir="GND/code", c_type="ldpc")
    assert str(path).endswith("GND/code/ldpc_n25_d4_k3_seed0")


def test_resolve_ldpc_path_prefers_local_data_dir() -> None:
    path = _resolve_ldpc_file_path(
        n=25, d=4, k=3, seed=0, c_type="ldpc")
    normalized = str(path).replace("\\", "/")
    assert normalized.endswith("experiments/codes/gnd_data/ldpc_n25_d4_k3_seed0")


def test_load_ldpc_css_code_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_ldpc_css_code(
            n=999,
            k=1,
            d=3,
            seed=0,
            code_dir="experiments/data/does_not_exist",
            strict_params=False,
        )
