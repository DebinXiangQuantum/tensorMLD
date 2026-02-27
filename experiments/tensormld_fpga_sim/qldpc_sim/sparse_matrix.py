from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

EPS = 1e-12


class SparseMatrix:
    """Sparse chi x chi matrix with row dictionaries."""

    def __init__(self, chi: int, rows: Optional[List[Dict[int, float]]] = None):
        self.chi = chi
        if rows is None:
            self.rows = [{} for _ in range(chi)]
        else:
            if len(rows) != chi:
                raise ValueError("rows length must equal chi")
            self.rows = rows

    @staticmethod
    def random(chi: int, density: float, rng: random.Random) -> "SparseMatrix":
        density = max(0.0, min(1.0, density))
        rows: List[Dict[int, float]] = []
        for _ in range(chi):
            row: Dict[int, float] = {}
            for c in range(chi):
                if rng.random() < density:
                    v = rng.uniform(-1.0, 1.0)
                    if abs(v) > EPS:
                        row[c] = v
            rows.append(row)
        return SparseMatrix(chi, rows)

    @staticmethod
    def identity(chi: int) -> "SparseMatrix":
        rows: List[Dict[int, float]] = []
        for i in range(chi):
            rows.append({i: 1.0})
        return SparseMatrix(chi, rows)

    def clone(self) -> "SparseMatrix":
        return SparseMatrix(self.chi, [dict(r) for r in self.rows])

    @property
    def nnz(self) -> int:
        return sum(len(r) for r in self.rows)

    def add(self, other: "SparseMatrix") -> Tuple["SparseMatrix", int]:
        if self.chi != other.chi:
            raise ValueError("shape mismatch")
        out_rows: List[Dict[int, float]] = []
        ops = 0
        for r in range(self.chi):
            out = dict(self.rows[r])
            for c, v in other.rows[r].items():
                out[c] = out.get(c, 0.0) + v
                ops += 1
            tiny = [c for c, val in out.items() if abs(val) <= EPS]
            for c in tiny:
                del out[c]
            out_rows.append(out)
        return SparseMatrix(self.chi, out_rows), max(ops, self.nnz + other.nnz)

    def matmul(self, other: "SparseMatrix") -> Tuple["SparseMatrix", int]:
        if self.chi != other.chi:
            raise ValueError("shape mismatch")
        out_rows: List[Dict[int, float]] = []
        macs = 0
        for r in range(self.chi):
            acc: Dict[int, float] = {}
            for k, av in self.rows[r].items():
                b_row = other.rows[k]
                for c, bv in b_row.items():
                    acc[c] = acc.get(c, 0.0) + av * bv
                    macs += 1
            tiny = [c for c, val in acc.items() if abs(val) <= EPS]
            for c in tiny:
                del acc[c]
            out_rows.append(acc)
        return SparseMatrix(self.chi, out_rows), macs

    def trace_abs(self) -> float:
        t = 0.0
        for i in range(self.chi):
            t += self.rows[i].get(i, 0.0)
        return abs(t)
