from __future__ import annotations

from pathlib import Path

import numpy as np

from .codes import css_code, gen_BB_code
from .gnd_ldpc_codes import (
    bb_18_4_4,
    bb_60_8_4,
    bb_72_12_6,
    tb_25_3_4,
    tb_30_6_4,
    tb_48_4_8,
)


ISCA_REVISION_CODE_ORDER: tuple[str, ...] = (
    "TB_25_3_4",
    "TB_30_6_4",
    "TB_48_4_8",
    "BB_18_4_4",
    "BB_60_8_4",
    "BB_72_12_6",
    "BB_90",
    "BB_108",
    "BB_144",
)

_DEFAULT_TB48_H_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "cases" / "tb_48_4_8" / "H.npy"
)
_DEFAULT_TB48_LOGICAL_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "cases" / "tb_48_4_8" / "logical.npy"
)


def _load_tb48_from_matrix(
    *,
    h_path: str | Path | None = None,
    logical_path: str | Path | None = None,
) -> css_code:
    resolved_h = _DEFAULT_TB48_H_PATH if h_path is None else Path(h_path)
    resolved_logical = (
        _DEFAULT_TB48_LOGICAL_PATH if logical_path is None else Path(logical_path)
    )

    if not resolved_h.exists():
        raise FileNotFoundError(resolved_h)
    if not resolved_logical.exists():
        raise FileNotFoundError(resolved_logical)

    hz = np.asarray(np.load(resolved_h), dtype=np.int64)
    logical = np.asarray(np.load(resolved_logical), dtype=np.int64)

    if hz.ndim != 2:
        raise ValueError(f"TB_48_4_8 H matrix must be 2D, got shape={hz.shape}.")
    if logical.ndim == 1:
        logical = logical.reshape(1, -1)
    if logical.ndim != 2:
        raise ValueError(
            f"TB_48_4_8 logical matrix must be 1D/2D, got shape={logical.shape}.")
    if logical.shape[1] != hz.shape[1]:
        raise ValueError(
            "TB_48_4_8 matrix mismatch: "
            f"logical cols={logical.shape[1]}, H cols={hz.shape[1]}.")

    n = int(hz.shape[1])
    code = css_code(
        hx=np.zeros((0, n), dtype=np.int64),
        hz=hz % 2,
        code_distance=8.0,
        name="TB_48_4_8",
        name_prefix="TB",
        check_css=False,
    )
    code.lz = logical % 2
    code.K = int(code.lz.shape[0])
    code.source_path = f"{resolved_h}|{resolved_logical}"
    code.source_format = "matrix_npy"
    code.source_family = "tb"
    return code


def load_isca_revision_code(
    name: str,
    *,
    seed: int = 0,
    code_dir: str | Path | None = None,
    strict_params: bool = True,
    tb48_h_path: str | Path | None = None,
    tb48_logical_path: str | Path | None = None,
) -> css_code:
    key = name.upper()

    if key == "TB_25_3_4":
        return tb_25_3_4(
            seed=seed,
            code_dir=code_dir,
            strict_params=strict_params,
        )
    if key == "TB_30_6_4":
        return tb_30_6_4(
            seed=seed,
            code_dir=code_dir,
            strict_params=strict_params,
        )
    if key == "TB_48_4_8":
        try:
            return tb_48_4_8(
                seed=seed,
                code_dir=code_dir,
                strict_params=strict_params,
            )
        except (FileNotFoundError, ImportError):
            return _load_tb48_from_matrix(
                h_path=tb48_h_path,
                logical_path=tb48_logical_path,
            )
    if key == "BB_18_4_4":
        return bb_18_4_4(
            seed=seed,
            code_dir=code_dir,
            strict_params=strict_params,
        )
    if key == "BB_60_8_4":
        return bb_60_8_4(
            seed=seed,
            code_dir=code_dir,
            strict_params=strict_params,
        )
    if key == "BB_72_12_6":
        return bb_72_12_6(
            seed=seed,
            code_dir=code_dir,
            strict_params=strict_params,
        )
    if key == "BB_90":
        code = gen_BB_code(90)
        code.name = "BB_90"
        return code
    if key == "BB_108":
        code = gen_BB_code(108)
        code.name = "BB_108"
        return code
    if key == "BB_144":
        code = gen_BB_code(144)
        code.name = "BB_144"
        return code

    raise ValueError(
        f"Unsupported ISCA revision benchmark code '{name}'. "
        f"Supported: {', '.join(ISCA_REVISION_CODE_ORDER)}")


def load_isca_revision_codes(
    *,
    seed: int = 0,
    code_dir: str | Path | None = None,
    strict_params: bool = True,
    skip_missing: bool = False,
    tb48_h_path: str | Path | None = None,
    tb48_logical_path: str | Path | None = None,
) -> tuple[list[tuple[str, css_code]], list[str]]:
    loaded: list[tuple[str, css_code]] = []
    skipped: list[str] = []

    for name in ISCA_REVISION_CODE_ORDER:
        try:
            code = load_isca_revision_code(
                name=name,
                seed=seed,
                code_dir=code_dir,
                strict_params=strict_params,
                tb48_h_path=tb48_h_path,
                tb48_logical_path=tb48_logical_path,
            )
        except (FileNotFoundError, ImportError) as exc:
            if not skip_missing:
                raise
            skipped.append(f"{name}: {exc}")
            continue
        loaded.append((name, code))

    return loaded, skipped
