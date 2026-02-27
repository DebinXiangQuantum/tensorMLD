from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from .scheduler import schedule_durations, schedule_jobs
from .sparse_matrix import EPS, SparseMatrix
from .types import ArchConfig, CycleBreakdown, DeviceBudget, ResourceModelConfig, SimResult
from .utils import ceil_div, judge_probability


class CycleAccurateDecoderSim:
    def __init__(
        self,
        m: int,
        k: int,
        chi: int,
        matrix_density: float,
        logical_density: float,
        code_seed: int,
        arch: ArchConfig,
        matrix_list: Optional[List[SparseMatrix]] = None,
        logical_tensors: Optional[List[Tuple[SparseMatrix, SparseMatrix]]] = None,
    ):
        self.m = m
        self.k = k
        self.chi = chi
        self.arch = arch
        rng = random.Random(code_seed)

        if matrix_list is None:
            self.matrix_list = [SparseMatrix.random(chi, matrix_density, rng) for _ in range(m)]
        else:
            if len(matrix_list) != m:
                raise ValueError(f"matrix_list length must be {m}")
            for idx, mat in enumerate(matrix_list):
                if mat.chi != chi:
                    raise ValueError(f"matrix_list[{idx}] chi mismatch: got {mat.chi}, expect {chi}")
            self.matrix_list = matrix_list

        if logical_tensors is None:
            self.logical_tensors = [
                (SparseMatrix.random(chi, logical_density, rng), SparseMatrix.random(chi, logical_density, rng))
                for _ in range(k)
            ]
        else:
            if len(logical_tensors) != k:
                raise ValueError(f"logical_tensors length must be {k}")
            for idx, (t0, t1) in enumerate(logical_tensors):
                if t0.chi != chi or t1.chi != chi:
                    raise ValueError(
                        f"logical_tensors[{idx}] chi mismatch: got ({t0.chi}, {t1.chi}), expect ({chi}, {chi})"
                    )
            self.logical_tensors = logical_tensors

    def _csr_bits(self, mat: SparseMatrix, data_bits: int) -> int:
        nnz = mat.nnz
        col_bits = max(1, (self.chi - 1).bit_length())
        ptr_bits = max(1, (nnz + 1).bit_length())
        return nnz * (data_bits + col_bits) + (self.chi + 1) * ptr_bits

    def _spmm_cycles(self, a: SparseMatrix, b: SparseMatrix, c: SparseMatrix, macs: int) -> int:
        comp = ceil_div(max(1, macs), self.arch.spmm_pe)
        read = ceil_div(a.nnz + b.nnz, self.arch.spmm_read_bw)
        write = ceil_div(max(1, c.nnz), self.arch.spmm_write_bw)
        return max(comp, read, write) + self.arch.spmm_pipeline

    def _spmm(self, a: SparseMatrix, b: SparseMatrix, breakdown: CycleBreakdown) -> Tuple[SparseMatrix, int]:
        c, macs = a.matmul(b)
        cyc = self._spmm_cycles(a, b, c, macs)
        breakdown.spmm_calls += 1
        breakdown.spmm_macs += macs
        return c, cyc

    def _contract_tree(self, mats: List[SparseMatrix], breakdown: CycleBreakdown) -> Tuple[SparseMatrix, int]:
        if not mats:
            return SparseMatrix.identity(self.chi), 0
        if len(mats) == 1:
            return mats[0], 0

        total_cycles = 0
        cur = mats
        while len(cur) > 1:
            nxt: List[SparseMatrix] = []
            level_cycles: List[int] = []
            for i in range(0, len(cur), 2):
                if i + 1 < len(cur):
                    out, cyc = self._spmm(cur[i], cur[i + 1], breakdown)
                    nxt.append(out)
                    level_cycles.append(cyc)
                else:
                    nxt.append(cur[i])
            total_cycles += schedule_durations(level_cycles, self.arch.spmm_engines)
            cur = nxt
        return cur[0], total_cycles

    def _select_matrices_by_syndrome(self, syndrome: List[int], breakdown: CycleBreakdown) -> List[SparseMatrix]:
        scan_cycles = ceil_div(self.m, self.arch.selector_width)
        breakdown.selection_scan_cycles += scan_cycles

        selected = [self.matrix_list[i] for i, b in enumerate(syndrome) if b == 1]
        fetch_nnz = sum(mat.nnz for mat in selected)
        fetch_cycles = ceil_div(max(1, fetch_nnz), self.arch.spmm_read_bw) if selected else 0
        breakdown.selection_fetch_cycles += fetch_cycles
        return selected

    def estimate_resources(self, syndrome: List[int], model: ResourceModelConfig, budget: DeviceBudget) -> Dict[str, Any]:
        if len(syndrome) != self.m:
            raise ValueError(f"syndrome length must be {self.m}")

        selected = [self.matrix_list[i] for i, b in enumerate(syndrome) if b == 1]

        mac_lanes = self.arch.spmm_pe * self.arch.spmm_engines
        lut_spmm = mac_lanes * model.lut_per_mac + self.arch.spmm_engines * model.lut_per_spmm_engine
        ff_spmm = mac_lanes * model.ff_per_mac + self.arch.spmm_engines * model.ff_per_spmm_engine
        dsp = mac_lanes * model.dsp_per_mac

        lut_add = self.arch.add_engines * model.lut_per_add_engine
        ff_add = self.arch.add_engines * model.ff_per_add_engine

        lut_ctrl = (
            model.lut_ctrl_base
            + self.arch.parallel_bits * model.lut_per_parallel_bit
            + self.arch.contract_engines * model.lut_per_contract_engine
            + self.arch.selector_width * 8
            + self.arch.chain_select_bw * 12
            + self.arch.trace_bw * 10
        )
        ff_ctrl = (
            model.ff_ctrl_base
            + self.arch.parallel_bits * model.ff_per_parallel_bit
            + self.arch.contract_engines * model.ff_per_contract_engine
            + self.arch.selector_width * 10
            + self.arch.chain_select_bw * 16
            + self.arch.trace_bw * 12
        )

        lut_total = lut_spmm + lut_add + lut_ctrl
        ff_total = ff_spmm + ff_add + ff_ctrl

        scratch_bits = (
            self.arch.spmm_engines
            * model.scratch_mats_per_spmm_engine
            * self.chi
            * self.chi
            * model.data_bits
        )
        scratch_bits += self.arch.parallel_bits * self.chi * self.chi * model.data_bits
        bram36 = ceil_div(scratch_bits, model.bram36_bits)
        uram = 0

        code_bits = 0
        for mat in self.matrix_list:
            code_bits += self._csr_bits(mat, model.data_bits)
        for t0, t1 in self.logical_tensors:
            code_bits += self._csr_bits(t0, model.data_bits)
            code_bits += self._csr_bits(t1, model.data_bits)

        active_bits = 0
        for mat in selected:
            active_bits += self._csr_bits(mat, model.data_bits)
        for t0, t1 in self.logical_tensors:
            active_bits += self._csr_bits(t0, model.data_bits)
            active_bits += self._csr_bits(t1, model.data_bits)

        util = {
            "lut_pct": 100.0 * lut_total / max(1, budget.lut),
            "ff_pct": 100.0 * ff_total / max(1, budget.ff),
            "dsp_pct": 100.0 * dsp / max(1, budget.dsp),
            "bram36_pct": 100.0 * bram36 / max(1, budget.bram36),
            "uram_pct": 100.0 * uram / max(1, budget.uram),
        }
        fit = (
            lut_total <= budget.lut
            and ff_total <= budget.ff
            and dsp <= budget.dsp
            and bram36 <= budget.bram36
            and uram <= budget.uram
        )

        return {
            "estimate": {
                "lut": int(lut_total),
                "ff": int(ff_total),
                "dsp": int(dsp),
                "bram36": int(bram36),
                "uram": int(uram),
                "hbm_code_bytes": ceil_div(code_bits, 8),
                "hbm_active_bytes": ceil_div(active_bits, 8),
            },
            "u50_dynamic_budget": asdict(budget),
            "utilization_pct": {k: round(v, 3) for k, v in util.items()},
            "fits_u50_dynamic_region": fit,
            "model_assumptions": {
                "data_bits": model.data_bits,
                "dsp_per_mac": model.dsp_per_mac,
                "lut_per_mac": model.lut_per_mac,
                "ff_per_mac": model.ff_per_mac,
                "notes": "Early architectural estimate; needs post-HLS/RTL report calibration.",
            },
        }

    def simulate(self, syndrome: List[int]) -> SimResult:
        if len(syndrome) != self.m:
            raise ValueError(f"syndrome length must be {self.m}")
        if any(b not in (0, 1) for b in syndrome):
            raise ValueError("syndrome must contain only 0/1")

        cycles = 0
        breakdown = CycleBreakdown()

        selected = self._select_matrices_by_syndrome(syndrome, breakdown)
        cycles += breakdown.selection_scan_cycles + breakdown.selection_fetch_cycles

        if selected:
            t0, c1 = self._contract_tree(selected, breakdown)
            breakdown.step1_contract_cycles += c1
            cycles += c1
        else:
            t0 = SparseMatrix.identity(self.chi)

        runtime_tensors: List[Tuple[SparseMatrix, SparseMatrix]] = [
            (a.clone(), b.clone()) for (a, b) in self.logical_tensors
        ]

        t00, c00 = self._spmm(t0, runtime_tensors[0][0], breakdown)
        t01, c01 = self._spmm(t0, runtime_tensors[0][1], breakdown)
        runtime_tensors[0] = (t00, t01)
        c_attach = schedule_durations([c00, c01], self.arch.spmm_engines)
        breakdown.step1_attach_cycles += c_attach
        cycles += c_attach

        logical: List[Optional[int]] = [None for _ in range(self.k)]
        probs: List[float] = [0.5 for _ in range(self.k)]

        rounds = 0
        while any(v is None for v in logical):
            rounds += 1
            unknown = [i for i, v in enumerate(logical) if v is None]

            add_jobs: List[Tuple[Tuple[int, int], int]] = []
            sum_cache: Dict[int, SparseMatrix] = {}
            for kidx in unknown:
                sm, add_ops = runtime_tensors[kidx][0].add(runtime_tensors[kidx][1])
                sum_cache[kidx] = sm
                add_cycles = ceil_div(max(1, add_ops), self.arch.add_bw)
                add_jobs.append(((kidx, 0), add_cycles))
            add_makespan, _ = schedule_jobs(add_jobs, self.arch.add_engines)
            breakdown.step2_sum_cycles += add_makespan
            cycles += add_makespan

            progress = False
            batch_size = max(1, self.arch.parallel_bits)
            for st in range(0, len(unknown), batch_size):
                batch = unknown[st : st + batch_size]

                job_cycles: List[Tuple[Tuple[int, int], int]] = []
                fvals: Dict[Tuple[int, int], float] = {}

                for bit_i in batch:
                    for lam in (0, 1):
                        chain: List[SparseMatrix] = []
                        for kk in range(self.k):
                            if kk == bit_i:
                                chain.append(runtime_tensors[kk][lam])
                            elif logical[kk] is None:
                                chain.append(sum_cache[kk])
                            else:
                                chain.append(runtime_tensors[kk][logical[kk]])

                        psi, c_contract = self._contract_tree(chain, breakdown)
                        fvals[(bit_i, lam)] = psi.trace_abs()

                        c_mux = ceil_div(self.k, self.arch.chain_select_bw)
                        c_trace = ceil_div(self.chi, self.arch.trace_bw)
                        total_job_cycles = c_mux + c_contract + c_trace
                        job_cycles.append(((bit_i, lam), total_job_cycles))

                batch_makespan, finish = schedule_jobs(job_cycles, self.arch.contract_engines)

                decision_end = 0
                for bit_i in batch:
                    done_t = max(finish[(bit_i, 0)], finish[(bit_i, 1)])
                    done_t += self.arch.prob_latency + self.arch.judge_latency
                    decision_end = max(decision_end, done_t)

                    f0 = fvals[(bit_i, 0)]
                    f1 = fvals[(bit_i, 1)]
                    p = 0.5 if f0 + f1 <= EPS else f1 / (f1 + f0)
                    probs[bit_i] = p
                    d = judge_probability(p)
                    if d is not None and logical[bit_i] is None:
                        logical[bit_i] = d
                        progress = True

                c_batch = max(batch_makespan, decision_end)
                breakdown.step2_batch_cycles += c_batch
                cycles += c_batch

            if not progress and any(v is None for v in logical):
                remain = [i for i, v in enumerate(logical) if v is None]
                for ridx in remain:
                    logical[ridx] = 1 if probs[ridx] >= 0.5 else 0
                force_cycles = ceil_div(len(remain), max(1, self.arch.parallel_bits)) * self.arch.judge_latency
                breakdown.step2_force_cycles += force_cycles
                cycles += force_cycles

        return SimResult(
            total_cycles=cycles,
            rounds=rounds,
            selected_matrix_count=sum(syndrome),
            syndrome_weight=sum(syndrome),
            logical_bits=[int(x) for x in logical],
            logical_probabilities=probs,
            breakdown={
                "selection_scan_cycles": breakdown.selection_scan_cycles,
                "selection_fetch_cycles": breakdown.selection_fetch_cycles,
                "step1_contract_cycles": breakdown.step1_contract_cycles,
                "step1_attach_cycles": breakdown.step1_attach_cycles,
                "step2_sum_cycles": breakdown.step2_sum_cycles,
                "step2_batch_cycles": breakdown.step2_batch_cycles,
                "step2_force_cycles": breakdown.step2_force_cycles,
                "spmm_calls": breakdown.spmm_calls,
                "spmm_macs": breakdown.spmm_macs,
            },
            arch=asdict(self.arch),
        )
