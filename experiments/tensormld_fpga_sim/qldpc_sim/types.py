from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ArchConfig:
    selector_width: int = 64
    spmm_pe: int = 64
    spmm_read_bw: int = 128
    spmm_write_bw: int = 64
    spmm_pipeline: int = 4
    spmm_engines: int = 8
    contract_engines: int = 4
    add_bw: int = 128
    add_engines: int = 8
    chain_select_bw: int = 32
    trace_bw: int = 32
    prob_latency: int = 2
    judge_latency: int = 1
    parallel_bits: int = 4


@dataclass
class CycleBreakdown:
    selection_scan_cycles: int = 0
    selection_fetch_cycles: int = 0
    step1_contract_cycles: int = 0
    step1_attach_cycles: int = 0
    step2_sum_cycles: int = 0
    step2_batch_cycles: int = 0
    step2_force_cycles: int = 0
    spmm_calls: int = 0
    spmm_macs: int = 0


@dataclass
class ResourceModelConfig:
    data_bits: int = 16
    dsp_per_mac: int = 1
    lut_per_mac: int = 90
    ff_per_mac: int = 140
    lut_per_spmm_engine: int = 1800
    ff_per_spmm_engine: int = 2600
    lut_per_add_engine: int = 700
    ff_per_add_engine: int = 900
    lut_ctrl_base: int = 8000
    ff_ctrl_base: int = 12000
    lut_per_parallel_bit: int = 220
    ff_per_parallel_bit: int = 300
    lut_per_contract_engine: int = 180
    ff_per_contract_engine: int = 250
    scratch_mats_per_spmm_engine: int = 3
    bram36_bits: int = 36 * 1024
    uram_bits: int = 288 * 1024


@dataclass
class DeviceBudget:
    # UG1120 (v1.5), Table 8, xilinx_u50_gen3x16_xdma_201920_3 dynamic region.
    lut: int = 731_000
    ff: int = 1_462_000
    dsp: int = 5_340
    bram36: int = 1_128
    uram: int = 608


@dataclass
class SimResult:
    total_cycles: int
    rounds: int
    selected_matrix_count: int
    syndrome_weight: int
    logical_bits: List[int]
    logical_probabilities: List[float]
    breakdown: Dict[str, int]
    arch: Dict[str, int]
    resource_estimate: Optional[Dict[str, Any]] = None
