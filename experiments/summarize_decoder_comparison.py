#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

"""Summarize decoder comparison JSON into Markdown and CSV-friendly output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt(x: Any, ndigits: int = 4) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.{ndigits}f}"
    return str(x)


def _flatten_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in report.get("cases", []):
        case_id = case.get("case_id", "unknown")
        for run in case.get("runs", []):
            exact = run.get("exact", {})
            mps = run.get("mps", {})
            cmp_metrics = run.get("comparison", {})
            rows.append({
                "case_id":
                case_id,
                "noise_p":
                run.get("noise_p"),
                "exact_status":
                exact.get("status"),
                "mps_status":
                mps.get("status"),
                "exact_total_ms":
                exact.get("total_ms"),
                "mps_total_ms":
                mps.get("total_ms"),
                "speedup_mps_over_exact":
                cmp_metrics.get("mps_speedup_over_exact"),
                "mean_abs_prob_diff":
                cmp_metrics.get("mean_abs_prob_diff"),
                "max_abs_prob_diff":
                cmp_metrics.get("max_abs_prob_diff"),
                "decision_agreement":
                cmp_metrics.get("decision_agreement"),
                "exact_gpu_mem_peak_mb":
                exact.get("gpu_profile", {}).get("mem_used_peak_mb"),
                "mps_gpu_mem_peak_mb":
                mps.get("gpu_profile", {}).get("mem_used_peak_mb"),
                "exact_gpu_util_avg":
                exact.get("gpu_profile", {}).get("gpu_util_avg"),
                "mps_gpu_util_avg":
                mps.get("gpu_profile", {}).get("gpu_util_avg"),
            })
    return rows


def _build_markdown(rows: list[dict[str, Any]]) -> str:
    header = [
        "| case | p | exact(ms) | mps(ms) | speedup | mean|diff| | decision agree | exact mem(MB) | mps mem(MB) | exact util(%) | mps util(%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    body = []
    for r in rows:
        body.append(
            "| {case} | {p} | {exact} | {mps} | {speedup} | {mad} | {agree} | {emem} | {mmem} | {eutil} | {mutil} |"
            .format(
                case=r["case_id"],
                p=_fmt(r["noise_p"], 4),
                exact=_fmt(r["exact_total_ms"], 3),
                mps=_fmt(r["mps_total_ms"], 3),
                speedup=_fmt(r["speedup_mps_over_exact"], 3),
                mad=_fmt(r["mean_abs_prob_diff"], 6),
                agree=_fmt(r["decision_agreement"], 4),
                emem=_fmt(r["exact_gpu_mem_peak_mb"], 2),
                mmem=_fmt(r["mps_gpu_mem_peak_mb"], 2),
                eutil=_fmt(r["exact_gpu_util_avg"], 2),
                mutil=_fmt(r["mps_gpu_util_avg"], 2),
            ))
    if not body:
        body = ["| NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |"]
    return "\n".join(header + body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize decoder comparison.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-md",
                        type=str,
                        default="experiments/results/decoder_comparison_summary.md")
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    rows = _flatten_rows(report)
    summary_md = _build_markdown(rows)

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_md.open("w", encoding="utf-8") as f:
        f.write("# Decoder Comparison Summary\n\n")
        f.write(f"- Source: `{input_path}`\n")
        f.write(f"- Cases: {len(report.get('cases', []))}\n")
        f.write(f"- Rows: {len(rows)}\n\n")
        f.write(summary_md)
        f.write("\n")

    print(f"Summary written to: {output_md}")
    print(summary_md)


if __name__ == "__main__":
    main()
