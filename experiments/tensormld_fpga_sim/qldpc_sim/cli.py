from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from typing import List, Optional, Tuple

from .decoder import CycleAccurateDecoderSim
from .io import load_matrix_file, load_syndrome_file, parse_syndrome_string
from .sparse_matrix import SparseMatrix
from .types import ArchConfig, DeviceBudget, ResourceModelConfig, SimResult


def random_syndrome(m: int, one_prob: float, seed: int) -> List[int]:
    rng = random.Random(seed)
    return [1 if rng.random() < one_prob else 0 for _ in range(m)]


def run_once(
    args: argparse.Namespace,
    code_seed: int,
    syndrome: List[int],
    matrix_list: Optional[List[SparseMatrix]] = None,
    logical_tensors: Optional[List[Tuple[SparseMatrix, SparseMatrix]]] = None,
) -> SimResult:
    arch = ArchConfig(
        selector_width=args.selector_width,
        spmm_pe=args.spmm_pe,
        spmm_read_bw=args.spmm_read_bw,
        spmm_write_bw=args.spmm_write_bw,
        spmm_pipeline=args.spmm_pipeline,
        spmm_engines=args.spmm_engines,
        contract_engines=args.contract_engines,
        add_bw=args.add_bw,
        add_engines=args.add_engines,
        chain_select_bw=args.chain_select_bw,
        trace_bw=args.trace_bw,
        prob_latency=args.prob_latency,
        judge_latency=args.judge_latency,
        parallel_bits=args.parallel_bits,
    )
    sim = CycleAccurateDecoderSim(
        m=args.m,
        k=args.k,
        chi=args.chi,
        matrix_density=args.matrix_density,
        logical_density=args.logical_density,
        code_seed=code_seed,
        arch=arch,
        matrix_list=matrix_list,
        logical_tensors=logical_tensors,
    )
    out = sim.simulate(syndrome)
    if args.estimate_resource:
        rmodel = ResourceModelConfig(
            data_bits=args.data_bits,
            dsp_per_mac=args.dsp_per_mac,
            lut_per_mac=args.lut_per_mac,
            ff_per_mac=args.ff_per_mac,
        )
        out.resource_estimate = sim.estimate_resources(syndrome, rmodel, DeviceBudget())
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FPGA cycle-accurate style simulator for QLDPC decode flow")
    p.add_argument("--m", type=int, default=128, help="syndrome length M")
    p.add_argument("--k", type=int, default=32, help="logical length K")
    p.add_argument("--chi", type=int, default=16, help="bond dimension chi")
    p.add_argument("--matrix-density", type=float, default=0.08, help="density of syndrome-side matrices")
    p.add_argument("--logical-density", type=float, default=0.10, help="density of logical tensors")

    p.add_argument("--seed", type=int, default=1, help="base seed")
    p.add_argument("--num-codes", type=int, default=1, help="number of random code instances")
    p.add_argument("--syndrome", type=str, default="", help="optional fixed syndrome bitstring")
    p.add_argument("--syndrome-file", type=str, default="", help="path to syndrome file (json/txt)")
    p.add_argument("--matrix-file", type=str, default="", help="path to matrix JSON file")
    p.add_argument("--syndrome-one-prob", type=float, default=0.2, help="if random syndrome, probability of 1")

    p.add_argument("--parallel-bits", type=int, default=4, help="N unknown bits in parallel")

    p.add_argument("--selector-width", type=int, default=64)
    p.add_argument("--spmm-pe", type=int, default=64)
    p.add_argument("--spmm-read-bw", type=int, default=128)
    p.add_argument("--spmm-write-bw", type=int, default=64)
    p.add_argument("--spmm-pipeline", type=int, default=4)
    p.add_argument("--spmm-engines", type=int, default=8)
    p.add_argument("--contract-engines", type=int, default=4)
    p.add_argument("--add-bw", type=int, default=128)
    p.add_argument("--add-engines", type=int, default=8)
    p.add_argument("--chain-select-bw", type=int, default=32)
    p.add_argument("--trace-bw", type=int, default=32)
    p.add_argument("--prob-latency", type=int, default=2)
    p.add_argument("--judge-latency", type=int, default=1)

    p.add_argument("--json", action="store_true", help="print full json")
    p.add_argument("--estimate-resource", action="store_true", help="estimate LUT/FF/DSP/BRAM for U50")
    p.add_argument("--data-bits", type=int, default=16, help="value bit width in sparse storage/model")
    p.add_argument("--dsp-per-mac", type=int, default=1, help="DSPs used per MAC lane")
    p.add_argument("--lut-per-mac", type=int, default=90, help="LUTs used per MAC lane")
    p.add_argument("--ff-per-mac", type=int, default=140, help="FFs used per MAC lane")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.m <= 0 or args.k <= 0 or args.chi <= 0:
        raise ValueError("m/k/chi must be > 0")
    if args.num_codes <= 0:
        raise ValueError("num-codes must be > 0")
    if args.data_bits <= 0 or args.dsp_per_mac <= 0 or args.lut_per_mac <= 0 or args.ff_per_mac <= 0:
        raise ValueError("resource model knobs must be > 0")

    matrix_list: Optional[List[SparseMatrix]] = None
    logical_tensors: Optional[List[Tuple[SparseMatrix, SparseMatrix]]] = None
    if args.matrix_file:
        matrix_list, logical_tensors = load_matrix_file(
            args.matrix_file,
            expected_m=args.m,
            expected_k=args.k,
            expected_chi=args.chi,
        )

    if args.syndrome and args.syndrome_file:
        raise ValueError("use only one of --syndrome and --syndrome-file")
    if args.syndrome_file:
        syndrome = load_syndrome_file(args.syndrome_file, args.m)
    elif args.syndrome:
        syndrome = parse_syndrome_string(args.syndrome, args.m)
    else:
        syndrome = random_syndrome(args.m, args.syndrome_one_prob, args.seed + 9973)

    results: List[SimResult] = []
    for idx in range(args.num_codes):
        code_seed = args.seed + idx * 1009
        results.append(
            run_once(
                args,
                code_seed,
                syndrome,
                matrix_list=matrix_list,
                logical_tensors=logical_tensors,
            )
        )

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
        return

    print("=== QLDPC FPGA Cycle Simulator ===")
    print(f"M={args.m}, K={args.k}, chi={args.chi}, parallel_bits={args.parallel_bits}")
    print(f"syndrome_weight={sum(syndrome)} / {len(syndrome)}")
    print("")

    total = 0
    for i, res in enumerate(results):
        total += res.total_cycles
        print(
            f"code#{i:02d}: cycles={res.total_cycles}, rounds={res.rounds}, "
            f"selected_mats={res.selected_matrix_count}, spmm_calls={res.breakdown['spmm_calls']}"
        )

    avg = total / len(results)
    print("")
    print(f"average_cycles={avg:.2f} over {len(results)} code(s)")

    if args.estimate_resource and results:
        est = results[0].resource_estimate
        if est is not None:
            e = est["estimate"]
            u = est["utilization_pct"]
            print("")
            print("resource_estimate (U50 dynamic region check):")
            print(
                f"LUT={e['lut']} ({u['lut_pct']}%), FF={e['ff']} ({u['ff_pct']}%), "
                f"DSP={e['dsp']} ({u['dsp_pct']}%)"
            )
            print(
                f"BRAM36={e['bram36']} ({u['bram36_pct']}%), URAM={e['uram']} ({u['uram_pct']}%), "
                f"fit={est['fits_u50_dynamic_region']}"
            )
            print(f"HBM(code)={e['hbm_code_bytes']} bytes, HBM(active)={e['hbm_active_bytes']} bytes")
