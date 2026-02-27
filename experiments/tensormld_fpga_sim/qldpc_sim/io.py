from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .sparse_matrix import EPS, SparseMatrix


def _parse_syndrome_value(value: Any, expected_m: int, source: str) -> List[int]:
    if isinstance(value, str):
        bits = "".join(ch for ch in value if not ch.isspace())
        if len(bits) != expected_m:
            raise ValueError(f"{source}: syndrome length must be {expected_m}, got {len(bits)}")
        out: List[int] = []
        for ch in bits:
            if ch not in ("0", "1"):
                raise ValueError(f"{source}: syndrome string must contain only 0/1")
            out.append(int(ch))
        return out

    if isinstance(value, list):
        if len(value) != expected_m:
            raise ValueError(f"{source}: syndrome length must be {expected_m}, got {len(value)}")
        out: List[int] = []
        for idx, item in enumerate(value):
            if isinstance(item, bool):
                bit = int(item)
            elif isinstance(item, int):
                bit = item
            else:
                raise ValueError(f"{source}: syndrome[{idx}] must be int/bool")
            if bit not in (0, 1):
                raise ValueError(f"{source}: syndrome[{idx}] must be 0 or 1")
            out.append(bit)
        return out

    raise ValueError(f"{source}: unsupported syndrome format")


def parse_syndrome_string(value: str, expected_m: int) -> List[int]:
    return _parse_syndrome_value(value, expected_m, "--syndrome")


def load_syndrome_file(path: str, expected_m: int) -> List[int]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _parse_syndrome_value(text, expected_m, f"{path}")

    if isinstance(payload, dict):
        if "syndrome" in payload:
            return _parse_syndrome_value(payload["syndrome"], expected_m, f"{path}.syndrome")
        if "bitstring" in payload:
            return _parse_syndrome_value(payload["bitstring"], expected_m, f"{path}.bitstring")
        raise ValueError(f"{path}: JSON must contain 'syndrome' or 'bitstring'")

    # Plain text like 1011 can be parsed by json as int; fall back to raw text.
    if isinstance(payload, (int, float, bool)):
        return _parse_syndrome_value(text, expected_m, path)

    return _parse_syndrome_value(payload, expected_m, path)


def _matrix_from_entries(entries: Any, chi: int, source: str) -> SparseMatrix:
    if isinstance(entries, dict):
        if "entries" in entries:
            entries = entries["entries"]
        elif "coo" in entries:
            entries = entries["coo"]
        else:
            raise ValueError(f"{source}: matrix dict must contain 'entries' or 'coo'")

    if not isinstance(entries, list):
        raise ValueError(f"{source}: matrix must be a list of [row,col,val]")

    rows: List[Dict[int, float]] = [{} for _ in range(chi)]
    for idx, item in enumerate(entries):
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError(f"{source}: entry #{idx} must be [row, col, val]")
        r, c, v = item
        if not isinstance(r, int) or not isinstance(c, int):
            raise ValueError(f"{source}: entry #{idx} row/col must be int")
        if r < 0 or r >= chi or c < 0 or c >= chi:
            raise ValueError(f"{source}: entry #{idx} index out of range for chi={chi}")

        val = float(v)
        if abs(val) <= EPS:
            continue
        rows[r][c] = rows[r].get(c, 0.0) + val
        if abs(rows[r][c]) <= EPS:
            del rows[r][c]

    return SparseMatrix(chi, rows)


def _logical_pair_to_mats(obj: Any, chi: int, source: str) -> Tuple[SparseMatrix, SparseMatrix]:
    if isinstance(obj, dict):
        zero = obj.get("zero", obj.get("t0", obj.get("0")))
        one = obj.get("one", obj.get("t1", obj.get("1")))
        if zero is None or one is None:
            raise ValueError(f"{source}: logical tensor dict must contain zero/t0/0 and one/t1/1")
        return _matrix_from_entries(zero, chi, source + ".zero"), _matrix_from_entries(one, chi, source + ".one")

    if isinstance(obj, list) and len(obj) == 2:
        return _matrix_from_entries(obj[0], chi, source + "[0]"), _matrix_from_entries(obj[1], chi, source + "[1]")

    raise ValueError(f"{source}: logical tensor must be dict or [mat0, mat1]")


def load_matrix_file(
    path: str,
    expected_m: int,
    expected_k: int,
    expected_chi: int,
) -> Tuple[List[SparseMatrix], Optional[List[Tuple[SparseMatrix, SparseMatrix]]]]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: matrix file must be a JSON object")

    chi = payload.get("chi", expected_chi)
    if not isinstance(chi, int) or chi <= 0:
        raise ValueError(f"{path}: 'chi' must be positive int")
    if chi != expected_chi:
        raise ValueError(f"{path}: chi mismatch, file={chi}, cli={expected_chi}")

    matrix_list_raw = payload.get("matrix_list", payload.get("matrices"))
    if not isinstance(matrix_list_raw, list):
        raise ValueError(f"{path}: require 'matrix_list' as list")
    if len(matrix_list_raw) != expected_m:
        raise ValueError(f"{path}: matrix_list length must be {expected_m}, got {len(matrix_list_raw)}")

    matrix_list: List[SparseMatrix] = []
    for idx, mat_raw in enumerate(matrix_list_raw):
        matrix_list.append(_matrix_from_entries(mat_raw, chi, f"{path}.matrix_list[{idx}]"))

    logical_raw = payload.get("logical_tensors")
    if logical_raw is None:
        return matrix_list, None

    if not isinstance(logical_raw, list):
        raise ValueError(f"{path}: logical_tensors must be a list")
    if len(logical_raw) != expected_k:
        raise ValueError(f"{path}: logical_tensors length must be {expected_k}, got {len(logical_raw)}")

    logical_tensors: List[Tuple[SparseMatrix, SparseMatrix]] = []
    for idx, pair_raw in enumerate(logical_raw):
        logical_tensors.append(_logical_pair_to_mats(pair_raw, chi, f"{path}.logical_tensors[{idx}]"))

    return matrix_list, logical_tensors
