# ============================================================================ #
# qLDPC Benchmark Case Loading Utilities                                       #
# ============================================================================ #
"""Utilities for loading the six qLDPC benchmark cases.

Expected on-disk layout:

data/cases/<case_name>/
  H.npy
  logical.npy
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class QLDPCBenchmarkCaseSpec:
    """Static descriptor of a benchmark case."""

    name: str
    family: str
    n: int
    k: int
    d: int


@dataclass
class QLDPCBenchmarkCase:
    """Loaded benchmark matrices and metadata."""

    spec: QLDPCBenchmarkCaseSpec
    h: np.ndarray
    logical: np.ndarray


DEFAULT_QLDPC_CASE_SPECS: tuple[QLDPCBenchmarkCaseSpec, ...] = (
    QLDPCBenchmarkCaseSpec("bb_18_4_4", "BB", 18, 4, 4),
    QLDPCBenchmarkCaseSpec("bb_60_8_4", "BB", 60, 8, 4),
    QLDPCBenchmarkCaseSpec("bb_72_12_6", "BB", 72, 12, 6),
    QLDPCBenchmarkCaseSpec("tb_25_3_4", "TB", 25, 3, 4),
    QLDPCBenchmarkCaseSpec("tb_30_6_4", "TB", 30, 6, 4),
    QLDPCBenchmarkCaseSpec("tb_48_4_8", "TB", 48, 4, 8),
)


def project_root() -> Path:
    """Return repository root path."""
    return Path(__file__).resolve().parents[1]


def default_cases_root() -> Path:
    """Return default case directory."""
    return project_root() / "data" / "cases"


def _as_2d(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape={arr.shape}")
    return arr


def resolve_case_paths(spec: QLDPCBenchmarkCaseSpec,
                       cases_root: Optional[Path] = None) -> tuple[Path, Path]:
    """Resolve H/logical paths for one benchmark case."""
    root = default_cases_root() if cases_root is None else Path(cases_root)
    case_dir = root / spec.name
    return case_dir / "H.npy", case_dir / "logical.npy"


def load_case(spec: QLDPCBenchmarkCaseSpec,
              cases_root: Optional[Path] = None,
              max_logicals: int = 1) -> QLDPCBenchmarkCase:
    """Load one benchmark case from disk."""
    h_path, logical_path = resolve_case_paths(spec, cases_root)
    if not h_path.exists():
        raise FileNotFoundError(f"Missing parity-check matrix: {h_path}")
    if not logical_path.exists():
        raise FileNotFoundError(f"Missing logical matrix: {logical_path}")

    h = _as_2d(np.asarray(np.load(h_path), dtype=np.float64), "H")
    logical = _as_2d(np.asarray(np.load(logical_path), dtype=np.float64), "logical")

    if logical.shape[1] != h.shape[1]:
        raise ValueError(
            f"{spec.name}: logical columns={logical.shape[1]} != H columns={h.shape[1]}"
        )

    max_rows = max(1, min(int(max_logicals), logical.shape[0]))
    logical = logical[:max_rows, :]
    return QLDPCBenchmarkCase(spec=spec, h=h, logical=logical)


def load_qldpc_cases(
    cases_root: Optional[Path] = None,
    case_filter: str = "",
    skip_missing: bool = False,
    max_logicals: int = 1,
) -> tuple[list[QLDPCBenchmarkCase], list[str]]:
    """Load all (or filtered) qLDPC benchmark cases.

    Returns:
        (loaded_cases, skipped_or_missing_messages)
    """
    loaded: list[QLDPCBenchmarkCase] = []
    skipped: list[str] = []
    pattern = re.compile(case_filter) if case_filter else None

    for spec in DEFAULT_QLDPC_CASE_SPECS:
        if pattern is not None and not pattern.search(spec.name):
            continue
        try:
            loaded.append(
                load_case(
                    spec=spec,
                    cases_root=cases_root,
                    max_logicals=max_logicals,
                )
            )
        except FileNotFoundError as exc:
            if skip_missing:
                skipped.append(str(exc))
                continue
            raise

    return loaded, skipped
