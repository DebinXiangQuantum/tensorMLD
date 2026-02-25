from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .codes import css_code


_DEFAULT_GND_CODE_DIR = Path(__file__).resolve().parent / "gnd_data"
_LEGACY_GND_CODE_DIR = Path(__file__).resolve().parents[2] / "GND" / "code"


def _to_numpy_int64(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=np.int64)


def _load_torch_payload(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Loading GND ldpc_*/qcc_* code files requires torch. "
            "Install torch in your experiment environment."
        ) from exc

    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": True}
    try:
        payload = torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        payload = torch.load(path, **kwargs)

    if not isinstance(payload, (tuple, list)) or len(payload) < 3:
        raise ValueError(
            f"Unexpected payload format in {path}: expected tuple/list of length >= 3."
        )

    g_stabilizer = _to_numpy_int64(payload[0])
    logical_opt = _to_numpy_int64(payload[1])
    pure_es = _to_numpy_int64(payload[2])
    return g_stabilizer, logical_opt, pure_es


def split_gnd_stabilizer_to_hx_hz(
        g_stabilizer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert GND's Pauli stabilizer representation to CSS (Hx, Hz).

    GND encodes Pauli operators as integers per qubit:
      0: I, 1: X, 2: Z, 3: Y.
    """
    g = np.asarray(g_stabilizer, dtype=np.int64)
    if g.ndim != 2:
        raise ValueError(
            f"g_stabilizer must be a 2D matrix, got shape={g.shape}.")

    x_rows = np.any((g % 2) != 0, axis=1)
    z_rows = ~x_rows

    n = g.shape[1]
    hx = (g[x_rows] % 2).astype(np.int64, copy=False)
    hz = ((g[z_rows] // 2) % 2).astype(np.int64, copy=False)
    if hx.size == 0:
        hx = np.zeros((0, n), dtype=np.int64)
    if hz.size == 0:
        hz = np.zeros((0, n), dtype=np.int64)
    return hx, hz


def _resolve_ldpc_file_path(
    n: int,
    d: int,
    k: int,
    seed: int = 0,
    code_dir: str | Path | None = None,
    c_type: str = "ldpc",
) -> Path:
    filename = f"{c_type}_n{int(n)}_d{int(d)}_k{int(k)}_seed{int(seed)}"

    if code_dir is not None:
        base = Path(code_dir)
        return base / filename

    candidates = [_DEFAULT_GND_CODE_DIR]
    if _LEGACY_GND_CODE_DIR != _DEFAULT_GND_CODE_DIR:
        candidates.append(_LEGACY_GND_CODE_DIR)

    for base in candidates:
        path = base / filename
        if path.exists():
            return path
    return candidates[0] / filename


def load_ldpc_css_code(
    n: int,
    k: int,
    d: int,
    seed: int = 0,
    code_dir: str | Path | None = None,
    c_type: str = "ldpc",
    name: str | None = None,
    strict_params: bool = True,
) -> css_code:
    """Load a GND ldpc_* file and expose it as `css_code`.

    This is used by benchmark configs via `source: codegen`.
    """
    path = _resolve_ldpc_file_path(
        n=n, d=d, k=k, seed=seed, code_dir=code_dir, c_type=c_type)
    if not path.exists():
        raise FileNotFoundError(path)

    g_stabilizer, _, _ = _load_torch_payload(path)
    hx, hz = split_gnd_stabilizer_to_hx_hz(g_stabilizer)

    code_name = (
        name if name is not None else
        f"{c_type.upper()}_n{int(n)}_k{int(k)}_d{int(d)}_seed{int(seed)}")
    code = css_code(
        hx=hx,
        hz=hz,
        code_distance=float(d),
        name=code_name,
        name_prefix=c_type.upper(),
        check_css=True,
    )

    if strict_params:
        if int(code.N) != int(n):
            raise ValueError(f"N mismatch: requested {n}, loaded {code.N}")
        if int(code.K) != int(k):
            raise ValueError(f"K mismatch: requested {k}, loaded {code.K}")

    # Attach metadata for benchmark reports/debug.
    code.source_path = str(path)
    code.source_format = "gnd_torch"
    code.source_family = c_type
    return code


def tb_25_3_4(seed: int = 0,
              code_dir: str | Path | None = None,
              strict_params: bool = True) -> css_code:
    return load_ldpc_css_code(
        n=25,
        k=3,
        d=4,
        seed=seed,
        code_dir=code_dir,
        strict_params=strict_params,
    )


def tb_30_6_4(seed: int = 0,
              code_dir: str | Path | None = None,
              strict_params: bool = True) -> css_code:
    return load_ldpc_css_code(
        n=30,
        k=6,
        d=4,
        seed=seed,
        code_dir=code_dir,
        strict_params=strict_params,
    )


def tb_48_4_8(seed: int = 0,
              code_dir: str | Path | None = None,
              strict_params: bool = True) -> css_code:
    return load_ldpc_css_code(
        n=48,
        k=4,
        d=8,
        seed=seed,
        code_dir=code_dir,
        strict_params=strict_params,
    )


def bb_18_4_4(seed: int = 0,
              code_dir: str | Path | None = None,
              strict_params: bool = True) -> css_code:
    """Load BB [[18,4,4]] from GND qcc file."""
    return load_ldpc_css_code(
        n=18,
        k=4,
        d=4,
        seed=seed,
        code_dir=code_dir,
        c_type="qcc",
        name="BB_18_4_4",
        strict_params=strict_params,
    )


def bb_60_8_4(seed: int = 0,
              code_dir: str | Path | None = None,
              strict_params: bool = True) -> css_code:
    """Load BB [[60,8,4]] from GND qcc file."""
    return load_ldpc_css_code(
        n=60,
        k=8,
        d=4,
        seed=seed,
        code_dir=code_dir,
        c_type="qcc",
        name="BB_60_8_4",
        strict_params=strict_params,
    )


def bb_72_12_6(seed: int = 0,
               code_dir: str | Path | None = None,
               strict_params: bool = True) -> css_code:
    """Load BB [[72,12,6]] from GND qcc file."""
    return load_ldpc_css_code(
        n=72,
        k=12,
        d=6,
        seed=seed,
        code_dir=code_dir,
        c_type="qcc",
        name="BB_72_12_6",
        strict_params=strict_params,
    )
