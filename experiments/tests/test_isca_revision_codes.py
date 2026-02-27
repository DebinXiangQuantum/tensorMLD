from __future__ import annotations

import pytest

from experiments.codes.isca_revision_codes import (
    ISCA_REVISION_CODE_ORDER,
    load_isca_revision_code,
    load_isca_revision_codes,
)


def test_isca_revision_code_order() -> None:
    assert len(ISCA_REVISION_CODE_ORDER) == 9
    assert ISCA_REVISION_CODE_ORDER[0] == "LDPC_25_3_4"
    assert ISCA_REVISION_CODE_ORDER[-1] == "QCC_144_12_12"


def test_load_isca_revision_codes_skip_missing() -> None:
    loaded, skipped = load_isca_revision_codes(skip_missing=True)
    loaded_names = [name for name, _ in loaded]
    assert "QCC_90_8_10" in loaded_names
    assert "QCC_108_8_10" in loaded_names
    assert "QCC_144_12_12" in loaded_names
    assert len(skipped) >= 0


def test_load_isca_revision_code_unknown_name() -> None:
    with pytest.raises(ValueError):
        load_isca_revision_code("UNKNOWN_CODE")
