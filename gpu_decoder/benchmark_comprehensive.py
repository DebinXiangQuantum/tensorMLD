#!/usr/bin/env python3
"""
Comprehensive GPU MPS Decoder Benchmark.

Measures for each code in each MPS data directory:
  - GPU memory usage (model loading + per-decode working memory)
  - Single-decode latency (min/median/mean/p99/max) in microseconds
  - Batch decode throughput for batch sizes 1, 10, 100, 1000
  - GPU utilization (%) during decode
  - GPU power consumption (W) during decode

Results are saved as gpu_benchmark_results.json under each data directory.

Usage:
    python benchmark_comprehensive.py [--data-dirs dir1 dir2 ...] [--num-bench N]
"""

import sys
import os
import time
import json
import subprocess
import threading
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mps_gpu_decoder import MpsGpuDecoder


# ──────────────────────────────────────────────────────────────────
# GPU metrics collection via nvidia-smi
# ──────────────────────────────────────────────────────────────────

def query_gpu_metrics():
    """Query current GPU metrics via nvidia-smi. Returns dict or None."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,memory.free,"
             "utilization.gpu,utilization.memory,power.draw,power.limit,"
             "temperature.gpu,clocks.sm,clocks.mem",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None

        parts = [x.strip() for x in result.stdout.strip().split("\n")[0].split(",")]
        if len(parts) < 10:
            return None

        return {
            "memory_used_mb": float(parts[0]),
            "memory_total_mb": float(parts[1]),
            "memory_free_mb": float(parts[2]),
            "gpu_utilization_pct": float(parts[3]),
            "mem_utilization_pct": float(parts[4]),
            "power_draw_w": float(parts[5]),
            "power_limit_w": float(parts[6]),
            "temperature_c": float(parts[7]),
            "sm_clock_mhz": float(parts[8]),
            "mem_clock_mhz": float(parts[9]),
        }
    except Exception as e:
        print(f"  WARNING: nvidia-smi query failed: {e}")
        return None


class GpuMetricsSampler:
    """Background thread that samples GPU metrics at regular intervals."""

    def __init__(self, interval_s=0.1):
        self.interval = interval_s
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            m = query_gpu_metrics()
            if m:
                m["timestamp"] = time.time()
                self.samples.append(m)
            self._stop.wait(self.interval)

    def summarize(self):
        if not self.samples:
            return {}
        gpu_util = [s["gpu_utilization_pct"] for s in self.samples]
        mem_util = [s["mem_utilization_pct"] for s in self.samples]
        power = [s["power_draw_w"] for s in self.samples]
        temp = [s["temperature_c"] for s in self.samples]
        mem_used = [s["memory_used_mb"] for s in self.samples]
        sm_clk = [s["sm_clock_mhz"] for s in self.samples]

        return {
            "num_samples": len(self.samples),
            "duration_s": self.samples[-1]["timestamp"] - self.samples[0]["timestamp"]
                if len(self.samples) > 1 else 0,
            "gpu_utilization_pct": {
                "min": float(np.min(gpu_util)),
                "mean": float(np.mean(gpu_util)),
                "max": float(np.max(gpu_util)),
            },
            "mem_utilization_pct": {
                "min": float(np.min(mem_util)),
                "mean": float(np.mean(mem_util)),
                "max": float(np.max(mem_util)),
            },
            "power_draw_w": {
                "min": float(np.min(power)),
                "mean": float(np.mean(power)),
                "max": float(np.max(power)),
            },
            "temperature_c": {
                "min": float(np.min(temp)),
                "mean": float(np.mean(temp)),
                "max": float(np.max(temp)),
            },
            "memory_used_mb": {
                "min": float(np.min(mem_used)),
                "mean": float(np.mean(mem_used)),
                "max": float(np.max(mem_used)),
            },
            "sm_clock_mhz": {
                "min": float(np.min(sm_clk)),
                "mean": float(np.mean(sm_clk)),
                "max": float(np.max(sm_clk)),
            },
        }


# ──────────────────────────────────────────────────────────────────
# Benchmark logic
# ──────────────────────────────────────────────────────────────────

def benchmark_code(code_name: str,
                   data_root: str,
                   num_warmup: int = 100,
                   num_bench: int = 1000,
                   batch_sizes: list = None,
                   seed: int = 42):
    """Benchmark one code with comprehensive GPU metrics."""
    chain_dir = os.path.join(data_root, code_name, "chain")

    if not os.path.exists(chain_dir):
        print(f"  SKIP {code_name}: chain dir not found at {chain_dir}")
        return None

    # Baseline GPU memory before loading
    baseline = query_gpu_metrics()
    baseline_mem = baseline["memory_used_mb"] if baseline else 0

    # Load decoder and measure memory
    print(f"  Loading decoder...")
    t_load_start = time.perf_counter()
    try:
        gpu_dec = MpsGpuDecoder(chain_dir)
    except Exception as e:
        print(f"  ERROR loading {code_name}: {e}")
        return None
    t_load = time.perf_counter() - t_load_start

    after_load = query_gpu_metrics()
    loaded_mem = after_load["memory_used_mb"] if after_load else 0
    model_mem_mb = loaded_mem - baseline_mem

    K = gpu_dec.K
    M = gpu_dec.M
    N = gpu_dec.N
    max_chi = gpu_dec.max_chi

    print(f"  N={N} K={K} M={M} max_chi={max_chi}  load_time={t_load:.3f}s  model_mem={model_mem_mb:.1f}MB")

    if batch_sizes is None:
        batch_sizes = [1, 10, 100, 1000]

    rng = np.random.RandomState(seed)
    max_needed = max(num_warmup + num_bench, max(batch_sizes))
    all_syndromes = rng.randint(0, 2, size=(max_needed, M)).astype(np.int32)

    # ── Single decode benchmark with GPU metrics sampling ──
    print(f"  Warming up ({num_warmup} decodes)...")
    for i in range(num_warmup):
        gpu_dec.decode(all_syndromes[i])

    print(f"  Benchmarking single decode ({num_bench} decodes)...")
    sampler = GpuMetricsSampler(interval_s=0.05)
    sampler.start()

    latencies = []
    for i in range(num_bench):
        _, _, lat = gpu_dec.decode(all_syndromes[num_warmup + i])
        latencies.append(lat)

    sampler.stop()
    single_gpu_metrics = sampler.summarize()

    latencies = np.array(latencies)
    single_result = {
        "num_decodes": num_bench,
        "min_us": float(np.min(latencies)),
        "p10_us": float(np.percentile(latencies, 10)),
        "p25_us": float(np.percentile(latencies, 25)),
        "median_us": float(np.median(latencies)),
        "mean_us": float(np.mean(latencies)),
        "p75_us": float(np.percentile(latencies, 75)),
        "p90_us": float(np.percentile(latencies, 90)),
        "p99_us": float(np.percentile(latencies, 99)),
        "max_us": float(np.max(latencies)),
        "std_us": float(np.std(latencies)),
    }

    print(f"    min={single_result['min_us']:.1f}  median={single_result['median_us']:.1f}  "
          f"mean={single_result['mean_us']:.1f}  p99={single_result['p99_us']:.1f}  "
          f"max={single_result['max_us']:.1f} us")

    # ── Batch decode benchmark with GPU metrics sampling ──
    batch_results = {}
    for bs in batch_sizes:
        try:
            batch_syn = all_syndromes[:bs]

            # Warmup
            for _ in range(10):
                gpu_dec.decode_batch(batch_syn)

            # Benchmark with metrics
            sampler_b = GpuMetricsSampler(interval_s=0.05)
            sampler_b.start()

            batch_lats = []
            for _ in range(100):
                _, _, lat = gpu_dec.decode_batch(batch_syn)
                batch_lats.append(lat)

            sampler_b.stop()
            batch_gpu_metrics = sampler_b.summarize()

            batch_lats = np.array(batch_lats)
            throughput = bs / (np.mean(batch_lats) / 1e6)  # decodes/sec

            batch_results[str(bs)] = {
                "batch_size": bs,
                "total_us_min": float(np.min(batch_lats)),
                "total_us_mean": float(np.mean(batch_lats)),
                "total_us_median": float(np.median(batch_lats)),
                "total_us_p99": float(np.percentile(batch_lats, 99)),
                "total_us_max": float(np.max(batch_lats)),
                "per_decode_us": float(np.mean(batch_lats) / bs),
                "throughput_per_sec": float(throughput),
                "gpu_metrics": batch_gpu_metrics,
            }

            print(f"    Batch {bs:5d}: total={np.mean(batch_lats):.1f} us, "
                  f"per_decode={np.mean(batch_lats)/bs:.1f} us, "
                  f"throughput={throughput:.0f} dec/s")
        except Exception as e:
            print(f"    Batch {bs:5d}: ERROR - {e}")
            batch_results[str(bs)] = {"batch_size": bs, "error": str(e)}

    # Memory after all decoding
    after_decode = query_gpu_metrics()
    decode_mem = after_decode["memory_used_mb"] if after_decode else 0

    gpu_dec.close()

    # Memory after cleanup
    after_close = query_gpu_metrics()
    freed_mem = after_close["memory_used_mb"] if after_close else 0

    memory_info = {
        "baseline_gpu_mem_mb": baseline_mem,
        "after_load_gpu_mem_mb": loaded_mem,
        "model_mem_mb": model_mem_mb,
        "during_decode_gpu_mem_mb": decode_mem,
        "after_close_gpu_mem_mb": freed_mem,
        "load_time_s": t_load,
    }

    return {
        "code": code_name,
        "N": N,
        "K": K,
        "M": M,
        "max_chi": max_chi,
        "memory": memory_info,
        "single_decode": single_result,
        "single_decode_gpu_metrics": single_gpu_metrics,
        "batch_decode": batch_results,
    }


def benchmark_directory(data_root: str, codes: list, num_warmup: int, num_bench: int):
    """Benchmark all codes in a data directory, saving incrementally."""
    dir_name = os.path.basename(data_root)
    print(f"\n{'='*70}")
    print(f"Benchmarking: {dir_name}")
    print(f"  Path: {data_root}")
    print(f"{'='*70}")

    if not os.path.exists(data_root):
        print(f"  SKIP: directory not found")
        return None

    # Check which codes exist
    available = []
    for code in codes:
        chain_dir = os.path.join(data_root, code, "chain")
        if os.path.exists(chain_dir):
            available.append(code)
    print(f"  Available codes: {len(available)}/{len(codes)}")

    # Load existing results (for incremental/resume support)
    out_path = os.path.join(data_root, "gpu_benchmark_results.json")
    existing_codes = set()
    all_results = []
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                prev = json.load(f)
            if prev.get("num_bench") == num_bench:
                all_results = prev.get("codes", [])
                existing_codes = {r["code"] for r in all_results}
                print(f"  Resuming: {len(existing_codes)} codes already done")
        except Exception:
            pass

    # Get initial GPU state
    initial_metrics = query_gpu_metrics()

    for code in available:
        if code in existing_codes:
            print(f"\n--- {code} --- (already done, skipping)")
            continue
        print(f"\n--- {code} ---")
        try:
            result = benchmark_code(code, data_root, num_warmup, num_bench)
        except Exception as e:
            print(f"  ERROR: {e}")
            result = None
        if result:
            all_results.append(result)

            # Save incrementally after each code
            output = {
                "data_directory": dir_name,
                "data_path": data_root,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_warmup": num_warmup,
                "num_bench": num_bench,
                "initial_gpu_state": initial_metrics,
                "codes": all_results,
            }
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)
            sys.stdout.flush()

    if not all_results:
        return None

    # Final save
    output = {
        "data_directory": dir_name,
        "data_path": data_root,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_warmup": num_warmup,
        "num_bench": num_bench,
        "initial_gpu_state": initial_metrics,
        "codes": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*100}")
    print(f"Summary: {dir_name}")
    print(f"{'Code':<18} {'N':>3} {'K':>3} {'chi':>4} | "
          f"{'ModelMB':>8} | "
          f"{'Min':>7} {'Med':>7} {'Mean':>7} {'P99':>7} | "
          f"{'Batch1k':>10}")
    print("-" * 100)
    for r in all_results:
        s = r["single_decode"]
        b1k = r["batch_decode"].get("1000", {})
        tp = b1k.get("throughput_per_sec", 0)
        mem = r["memory"]["model_mem_mb"]
        print(f"{r['code']:<18} {r['N']:>3} {r['K']:>3} {r['max_chi']:>4} | "
              f"{mem:>7.1f}M | "
              f"{s['min_us']:>7.1f} {s['median_us']:>7.1f} {s['mean_us']:>7.1f} "
              f"{s['p99_us']:>7.1f} | "
              f"{tp:>10.0f}/s")
    print("=" * 100)

    return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive GPU MPS Decoder Benchmark")
    parser.add_argument("--data-dirs", nargs="+", default=None,
                        help="Data directories to benchmark (relative to experiments/results/)")
    parser.add_argument("--num-warmup", type=int, default=100,
                        help="Number of warmup decodes (default: 100)")
    parser.add_argument("--num-bench", type=int, default=1000,
                        help="Number of benchmark decodes (default: 1000)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    results_root = project_root / "experiments" / "results"

    default_dirs = [
        "isca_revision_1d_mps_adaptive",
        "isca_revision_1d_mps_chi128",
        "isca_revision_1d_mps_chi256",
        "isca_revision_1d_mps_chi512",
        "isca_revision_1d_mps_multi",
    ]

    data_dirs = args.data_dirs if args.data_dirs else default_dirs

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

    print("=" * 70)
    print("Comprehensive GPU MPS Decoder Benchmark")
    print(f"  Warmup: {args.num_warmup} decodes")
    print(f"  Bench:  {args.num_bench} decodes")
    print(f"  Dirs:   {len(data_dirs)}")
    print("=" * 70)

    # Initial GPU info
    gpu_info = query_gpu_metrics()
    if gpu_info:
        print(f"\nGPU: {gpu_info['memory_total_mb']:.0f}MB total, "
              f"{gpu_info['memory_free_mb']:.0f}MB free, "
              f"power limit {gpu_info['power_limit_w']:.0f}W")

    all_dir_results = {}
    for d in data_dirs:
        data_root = str(results_root / d)
        result = benchmark_directory(data_root, codes, args.num_warmup, args.num_bench)
        if result:
            all_dir_results[d] = result

    # Save combined results
    combined_path = str(project_root / "gpu_decoder" / "benchmark_comprehensive_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_dir_results, f, indent=2)
    print(f"\nCombined results saved to {combined_path}")

    # Final cross-directory comparison
    print(f"\n{'='*120}")
    print("Cross-directory comparison (median single-decode latency, us)")
    print(f"{'Code':<18}", end="")
    for d in data_dirs:
        label = d.replace("isca_revision_1d_mps_", "")
        print(f" | {label:>12}", end="")
    print()
    print("-" * 120)

    for code in codes:
        print(f"{code:<18}", end="")
        for d in data_dirs:
            if d in all_dir_results:
                found = False
                for r in all_dir_results[d]["codes"]:
                    if r["code"] == code:
                        print(f" | {r['single_decode']['median_us']:>10.1f}us", end="")
                        found = True
                        break
                if not found:
                    print(f" | {'SKIP':>12}", end="")
            else:
                print(f" | {'N/A':>12}", end="")
        print()
    print("=" * 120)


if __name__ == "__main__":
    main()
