"""Unified ISCA revision benchmark code registry.

All 9 codes load exclusively from GND data files in experiments/codes/gnd_data/.
Names follow the format {c_type}_{n}_{k}_{d} matching [[n,k,d]] notation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .codes import css_code
from .gnd_ldpc_codes import (
    load_ldpc_css_code,
    ldpc_25_3_4,
    ldpc_30_6_4,
    tor_50_2_5,
    qcc_18_4_4,
    qcc_60_8_4,
    qcc_72_12_6,
    qcc_90_8_10,
    qcc_108_8_10,
    qcc_144_12_12,
)


# Canonical code order for benchmarks.
# Format: {c_type}_{n}_{k}_{d}  matching [[n,k,d]] notation.
ISCA_REVISION_CODE_ORDER: tuple[str, ...] = (
    "LDPC_25_3_4",     # ldpc_n25_d4_k3_seed0
    "LDPC_30_6_4",     # ldpc_n30_d4_k6_seed0
    "TOR_50_2_5",      # tor_n50_d5_k2_seed0
    "QCC_18_4_4",      # qcc_n18_d4_k4_seed0
    "QCC_60_8_4",      # qcc_n60_d4_k8_seed0
    "QCC_72_12_6",     # qcc_n72_d6_k12_seed0
    "QCC_90_8_10",     # qcc_n90_d10_k8_seed0
    "QCC_108_8_10",    # qcc_n108_d10_k8_seed0
    "QCC_144_12_12",   # qcc_n144_d12_k12_seed0
)

# Mapping from code name to loader function.
_CODE_LOADERS = {
    "LDPC_25_3_4":   ldpc_25_3_4,
    "LDPC_30_6_4":   ldpc_30_6_4,
    "TOR_50_2_5":    tor_50_2_5,
    "QCC_18_4_4":    qcc_18_4_4,
    "QCC_60_8_4":    qcc_60_8_4,
    "QCC_72_12_6":   qcc_72_12_6,
    "QCC_90_8_10":   qcc_90_8_10,
    "QCC_108_8_10":  qcc_108_8_10,
    "QCC_144_12_12": qcc_144_12_12,
}


def load_isca_revision_code(
    name: str,
    *,
    seed: int = 0,
    code_dir: str | Path | None = None,
    strict_params: bool = True,
    **kwargs,
) -> css_code:
    key = name.upper()
    loader = _CODE_LOADERS.get(key)
    if loader is None:
        raise ValueError(
            f"Unsupported ISCA revision benchmark code '{name}'. "
            f"Supported: {', '.join(ISCA_REVISION_CODE_ORDER)}")
    return loader(seed=seed, code_dir=code_dir, strict_params=strict_params)


def load_isca_revision_codes(
    *,
    seed: int = 0,
    code_dir: str | Path | None = None,
    strict_params: bool = True,
    skip_missing: bool = False,
    **kwargs,
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
            )
        except (FileNotFoundError, ImportError) as exc:
            if not skip_missing:
                raise
            skipped.append(f"{name}: {exc}")
            continue
        loaded.append((name, code))

    return loaded, skipped
