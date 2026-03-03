#!/usr/bin/env python3
"""
Per-phase latency profiling for the GPU MPS decoder.

For each code, runs profiled decode and reports latency breakdown:
  - prepare_base: Prepare base matrices from syndrome (N parallel blocks)
  - forward_scan: Left-to-right prefix product (1 block, sequential)
  - backward_scan: Right-to-left suffix product (1 block, sequential)
  - marginals: Compute marginal probabilities (K parallel blocks)
  - refinement: Conditional refinement rounds (repeat scan+marginals)
  - total: End-to-end decode latency

Also collects GPU metrics: memory usage, utilization, and power draw.

Reports median values across many decode runs.
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mps_gpu_decoder import MpsGpuDecoder


# ---------------------------------------------------------------------------
# GPU metrics collection via pynvml
# ---------------------------------------------------------------------------

_nvml_initialized = False
_nvml_handle = None


def _init_nvml():
    """Initialize NVML for GPU metrics. Returns True on success."""
    global _nvml_initialized, _nvml_handle
    if _nvml_initialized:
        return _nvml_handle is not None
    _nvml_initialized = True
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return True
    except Exception as e:
        print(f"  Warning: NVML init failed ({e}), GPU metrics will be unavailable")
        _nvml_handle = None
        return False


def query_gpu_metrics():
    """Query current GPU memory, utilization, power, temperature, and clocks.

    Returns a dict or None if NVML is unavailable.
    """
    if not _init_nvml():
        return None
    try:
        import pynvml
        h = _nvml_handle
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(h)           # milliwatts
        power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(h)
        temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        clock_sm = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        clock_mem = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
        return {
            "memory_used_mb": round(mem.used / (1024 * 1024), 1),
            "memory_total_mb": round(mem.total / (1024 * 1024), 1),
            "memory_used_pct": round(mem.used / mem.total * 100, 2),
            "gpu_utilization_pct": util.gpu,
            "memory_utilization_pct": util.memory,
            "power_draw_w": round(power_mw / 1000.0, 1),
            "power_limit_w": round(power_limit_mw / 1000.0, 1),
            "temperature_c": temp,
            "clock_sm_mhz": clock_sm,
            "clock_mem_mhz": clock_mem,
        }
    except Exception as e:
        return {"error": str(e)}


def query_gpu_device_info():
    """Query static GPU device properties (name, driver, etc.). Returns a dict."""
    if not _init_nvml():
        return None
    try:
        import pynvml
        h = _nvml_handle
        name = pynvml.nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("utf-8")
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(h)
        return {
            "gpu_name": name,
            "driver_version": driver,
            "memory_total_mb": round(mem.total / (1024 * 1024), 1),
            "power_limit_w": round(power_limit_mw / 1000.0, 1),
        }
    except Exception as e:
        return {"error": str(e)}


def sample_gpu_metrics_during(func, interval_s=0.05):
    """Run func() while sampling GPU metrics in a background thread.

    Returns (func_result, metrics_summary) where metrics_summary has
    min/mean/max/samples for utilization, power, temperature, etc.
    """
    import threading

    samples = []
    stop_event = threading.Event()

    def _sampler():
        while not stop_event.is_set():
            m = query_gpu_metrics()
            if m and "error" not in m:
                samples.append(m)
            stop_event.wait(interval_s)

    t = threading.Thread(target=_sampler, daemon=True)
    t.start()
    result = func()
    stop_event.set()
    t.join(timeout=1.0)

    if not samples:
        return result, None

    summary = {}
    for key in ["gpu_utilization_pct", "memory_utilization_pct",
                "power_draw_w", "temperature_c", "clock_sm_mhz",
                "clock_mem_mhz", "memory_used_mb"]:
        vals = [s[key] for s in samples if key in s]
        if vals:
            summary[key] = {
                "min": round(min(vals), 1),
                "mean": round(sum(vals) / len(vals), 1),
                "max": round(max(vals), 1),
                "samples": len(vals),
            }

    return result, summary


# ---------------------------------------------------------------------------
# Per-code profiling
# ---------------------------------------------------------------------------

def profile_code(code_name: str,
                 data_root: str,
                 num_warmup: int = 200,
                 num_bench: int = 1000,
                 seed: int = 42,
                 fp32: bool = True):
    """Profile one code's per-phase decode latency and GPU metrics."""
    chain_dir = os.path.join(data_root, code_name, "chain")

    if not os.path.exists(chain_dir):
        print(f"  SKIP {code_name}: not found")
        return None

    mode_str = "FP32" if fp32 else "FP64"
    gpu_dec = MpsGpuDecoder(chain_dir, fp32=fp32)
    K = gpu_dec.K
    M = gpu_dec.M
    N = gpu_dec.N
    max_chi = gpu_dec.max_chi

    print(f"  [{mode_str}] N={N} K={K} M={M} max_chi={max_chi} fp32={gpu_dec.fp32}")

    # Snapshot GPU memory after loading the decoder (before any decoding)
    gpu_after_load = query_gpu_metrics()
    if gpu_after_load and "error" not in gpu_after_load:
        print(f"  GPU memory after load: {gpu_after_load['memory_used_mb']:.0f} MB")

    rng = np.random.RandomState(seed)
    all_syndromes = rng.randint(0, 2, size=(num_warmup + num_bench, M)).astype(np.int32)

    # Warmup
    print(f"  Warming up ({num_warmup} decodes)...")
    for i in range(num_warmup):
        gpu_dec.decode(all_syndromes[i])

    # Snapshot GPU memory after warmup (all buffers allocated)
    gpu_after_warmup = query_gpu_metrics()
    if gpu_after_warmup and "error" not in gpu_after_warmup:
        print(f"  GPU memory after warmup: {gpu_after_warmup['memory_used_mb']:.0f} MB")

    # Profiled benchmark — sample GPU metrics concurrently
    print(f"  Profiling ({num_bench} decodes)...")
    phase_keys = ["prepare_base_us", "forward_scan_us", "backward_scan_us",
                  "marginals_us", "refinement_us", "total_us"]
    all_phases = {k: [] for k in phase_keys}

    def run_benchmark():
        for i in range(num_bench):
            _, _, phases = gpu_dec.decode_profiled(all_syndromes[num_warmup + i])
            for k in phase_keys:
                all_phases[k].append(phases[k])

    _, gpu_metrics_summary = sample_gpu_metrics_during(run_benchmark)

    # Compute statistics
    result = {
        "code": code_name,
        "N": N, "K": K, "M": M, "max_chi": max_chi,
        "num_bench": num_bench,
        "fp32": fp32,
        "phases": {},
    }

    # Add GPU metrics
    if gpu_after_warmup and "error" not in gpu_after_warmup:
        result["gpu_memory_used_mb"] = gpu_after_warmup["memory_used_mb"]
    if gpu_metrics_summary:
        result["gpu_metrics"] = gpu_metrics_summary
        # Print key GPU metrics
        for key, label in [("gpu_utilization_pct", "GPU util"),
                           ("memory_utilization_pct", "Mem util"),
                           ("power_draw_w", "Power"),
                           ("temperature_c", "Temp"),
                           ("clock_sm_mhz", "SM clock")]:
            if key in gpu_metrics_summary:
                m = gpu_metrics_summary[key]
                unit = {"gpu_utilization_pct": "%", "memory_utilization_pct": "%",
                        "power_draw_w": "W", "temperature_c": "C",
                        "clock_sm_mhz": "MHz"}.get(key, "")
                print(f"  {label}: min={m['min']}{unit} mean={m['mean']}{unit} max={m['max']}{unit} ({m['samples']} samples)")

    print(f"  {'Phase':<20} {'Min':>8} {'P25':>8} {'Median':>8} {'Mean':>8} {'P75':>8} {'P99':>8} {'Max':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for k in phase_keys:
        arr = np.array(all_phases[k])
        stats = {
            "min": float(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
        }
        result["phases"][k] = stats

        name = k.replace("_us", "")
        print(f"  {name:<20} {stats['min']:>8.1f} {stats['p25']:>8.1f} {stats['median']:>8.1f} "
              f"{stats['mean']:>8.1f} {stats['p75']:>8.1f} {stats['p99']:>8.1f} {stats['max']:>8.1f}")

    # Compute percentage breakdown (based on median)
    total_med = result["phases"]["total_us"]["median"]
    if total_med > 0:
        print(f"\n  Phase breakdown (median):")
        for k in phase_keys[:-1]:  # skip total
            pct = result["phases"][k]["median"] / total_med * 100
            name = k.replace("_us", "")
            print(f"    {name:<20} {result['phases'][k]['median']:>8.1f} us  ({pct:>5.1f}%)")
        print(f"    {'total':<20} {total_med:>8.1f} us  (100.0%)")

    gpu_dec.close()
    return result


def print_summary_table(dir_name: str, dir_results: list):
    """Print a summary table for one data directory."""
    print(f"\n{'='*120}")
    print(f"Summary: {dir_name}")
    print(f"{'Code':<18} {'chi':>4} {'Mode':>4} {'N':>3} {'K':>3} | "
          f"{'Prepare':>8} {'FwdScan':>8} {'BwdScan':>8} {'Marginal':>8} {'Refine':>8} | "
          f"{'Total':>8} | {'Mem':>6} {'Power':>6}")
    print(f"{'-'*120}")
    for r in dir_results:
        p = r["phases"]
        mode = "F32" if r.get("fp32", False) else "F64"
        mem_str = f"{r.get('gpu_memory_used_mb', 0):.0f}"
        pwr = r.get("gpu_metrics", {}).get("power_draw_w", {})
        pwr_str = f"{pwr.get('mean', 0):.0f}" if isinstance(pwr, dict) else ""
        print(f"{r['code']:<18} {r['max_chi']:>4} {mode:>4} {r['N']:>3} {r['K']:>3} | "
              f"{p['prepare_base_us']['median']:>8.1f} "
              f"{p['forward_scan_us']['median']:>8.1f} "
              f"{p['backward_scan_us']['median']:>8.1f} "
              f"{p['marginals_us']['median']:>8.1f} "
              f"{p['refinement_us']['median']:>8.1f} | "
              f"{p['total_us']['median']:>8.1f} | "
              f"{mem_str:>5}M {pwr_str:>5}W")
    print(f"{'='*120}")


def print_comparison_table(dir_name: str, fp64_results: list, fp32_results: list):
    """Print FP32 vs FP64 comparison for a data directory."""
    fp64_by_code = {r["code"]: r for r in fp64_results}
    fp32_by_code = {r["code"]: r for r in fp32_results}
    common_codes = [c for c in fp64_by_code if c in fp32_by_code]
    if not common_codes:
        return

    print(f"\n{'='*130}")
    print(f"FP32 vs FP64 Comparison: {dir_name}")
    print(f"{'Code':<18} {'chi':>4} | "
          f"{'FP64 Total':>10} {'FP32 Total':>10} {'Speedup':>8} | "
          f"{'FP64 Scan':>10} {'FP32 Scan':>10} {'ScanSpd':>8} | "
          f"{'FP64 Prep':>10} {'FP32 Prep':>10} {'PrepSpd':>8}")
    print(f"{'-'*130}")
    for code in common_codes:
        r64 = fp64_by_code[code]
        r32 = fp32_by_code[code]
        p64 = r64["phases"]
        p32 = r32["phases"]

        t64 = p64["total_us"]["median"]
        t32 = p32["total_us"]["median"]
        speedup = t64 / t32 if t32 > 0 else 0

        s64 = p64["forward_scan_us"]["median"]
        s32 = p32["forward_scan_us"]["median"]
        scan_spd = s64 / s32 if s32 > 0 else 0

        pr64 = p64["prepare_base_us"]["median"]
        pr32 = p32["prepare_base_us"]["median"]
        prep_spd = pr64 / pr32 if pr32 > 0 else 0

        print(f"{code:<18} {r64['max_chi']:>4} | "
              f"{t64:>10.1f} {t32:>10.1f} {speedup:>7.2f}x | "
              f"{s64:>10.1f} {s32:>10.1f} {scan_spd:>7.2f}x | "
              f"{pr64:>10.1f} {pr32:>10.1f} {prep_spd:>7.2f}x")
    print(f"{'='*130}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Per-phase latency profiling")
    parser.add_argument("data_dirs", nargs="*", help="Override data directories")
    parser.add_argument("--fp64-only", action="store_true",
                        help="Only benchmark FP64 mode")
    parser.add_argument("--fp32-only", action="store_true",
                        help="Only benchmark FP32 mode")
    parser.add_argument("--num-warmup", type=int, default=200)
    parser.add_argument("--num-bench", type=int, default=1000)
    parser.add_argument("--code", type=str, default=None,
                        help="Profile only this specific code")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # Default: profile chi128 and chi256
    if args.data_dirs:
        data_dirs = []
        for arg in args.data_dirs:
            name = os.path.basename(arg.rstrip("/")).replace("isca_revision_1d_mps_", "")
            data_dirs.append((name, arg))
    else:
        data_dirs = [
            ("chi128", str(project_root / "experiments" / "results" / "isca_revision_1d_mps_chi128")),
            ("chi256", str(project_root / "experiments" / "results" / "isca_revision_1d_mps_chi256")),
        ]

    # Determine which modes to benchmark
    modes = []
    if not args.fp32_only:
        modes.append(False)  # FP64
    if not args.fp64_only:
        modes.append(True)   # FP32

    codes = [
        "QCC_18_4_4",
        "LDPC_25_3_4",
        "LDPC_30_6_4",
        "TOR_50_2_5",
        "QCC_60_8_4",
        "QCC_72_12_6",
        "QCC_90_8_10",
        "QCC_108_8_10",
        "QCC_144_12_12",
    ]

    if args.code:
        codes = [c for c in codes if args.code in c]
        if not codes:
            print(f"No code matching '{args.code}'")
            sys.exit(1)

    # Collect static GPU device info
    device_info = query_gpu_device_info()
    if device_info and "error" not in device_info:
        print(f"\nGPU: {device_info['gpu_name']}, "
              f"Memory: {device_info['memory_total_mb']:.0f} MB, "
              f"Power limit: {device_info['power_limit_w']:.0f} W, "
              f"Driver: {device_info['driver_version']}")

    # Baseline GPU metrics (before any decoder is loaded)
    gpu_baseline = query_gpu_metrics()

    all_results = {}

    for dir_name, data_root in data_dirs:
        if not os.path.isdir(data_root):
            print(f"\nSKIP {dir_name}: directory not found")
            continue

        fp64_results = []
        fp32_results = []

        for fp32_mode in modes:
            mode_str = "FP32" if fp32_mode else "FP64"
            print(f"\n{'='*80}")
            print(f"Profiling: {dir_name} [{mode_str}]")
            print(f"{'='*80}")

            dir_results = []
            for code in codes:
                print(f"\n--- {code} [{mode_str}] ---")
                result = profile_code(code, data_root,
                                      num_warmup=args.num_warmup,
                                      num_bench=args.num_bench,
                                      fp32=fp32_mode)
                if result:
                    dir_results.append(result)

            if fp32_mode:
                fp32_results = dir_results
            else:
                fp64_results = dir_results

            print_summary_table(f"{dir_name} [{mode_str}]", dir_results)

        # Store results keyed by dir_name
        all_results[dir_name] = {
            "fp64": fp64_results,
            "fp32": fp32_results,
        }

        # Print comparison if both modes were run
        if fp64_results and fp32_results:
            print_comparison_table(dir_name, fp64_results, fp32_results)

    # Add metadata to results
    all_results["_metadata"] = {
        "device_info": device_info,
        "gpu_baseline": gpu_baseline,
        "modes": ["FP64" if not m else "FP32" for m in modes],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Save results
    out_path = os.path.join(project_root, "gpu_decoder", "phase_latency_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
