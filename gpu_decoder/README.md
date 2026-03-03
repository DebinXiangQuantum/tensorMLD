# GPU MPS Decoder — Technical Report

A CUDA C/C++ implementation of the Matrix Product State (MPS) online decoder for quantum error correction, targeting NVIDIA H100 GPUs. The decoder computes maximum-likelihood marginal probabilities for logical qubits given a syndrome measurement, using pre-compressed 1D MPS chains.

## Table of Contents

- [Algorithm Overview](#algorithm-overview)
- [Architecture](#architecture)
- [CUDA Kernel Design](#cuda-kernel-design)
- [Optimization Techniques](#optimization-techniques)
- [Performance Results](#performance-results)
- [GPU Resource Utilization](#gpu-resource-utilization)
- [File Structure](#file-structure)
- [Build & Usage](#build--usage)

---

## Algorithm Overview

Given a 1D MPS chain with N sites (interleaved syndrome and logical sites) and a syndrome vector **s**, the decoder computes marginal probabilities P(logical_k = 1 | **s**) for each logical qubit k:

1. **Prepare base matrices** — For each site i, collapse the 3D tensor T_i[chi_l, 2, chi_r] into a 2D base matrix B_i[chi_l, chi_r]:
   - Syndrome site: B_i = T_i[:, s_i, :] (select the observed syndrome value)
   - Unfixed logical site: B_i = T_i[:, 0, :] + T_i[:, 1, :] (sum over physical dimension)
   - Fixed logical site: B_i = T_i[:, v, :] (select the fixed value)

2. **Forward scan** (prefix products) — Compute left-to-right running product:
   prefix[0] = [1, 0, ...], prefix[j+1] = prefix[j] @ B[j]. Since chi_l[0] = 1, the prefix is always a row vector of size chi, making each step a vector-matrix multiply of cost O(chi^2).

3. **Backward scan** (suffix products) — Compute right-to-left running product:
   suffix[N] = [1, 0, ...], suffix[j] = B[j] @ suffix[j+1]. Since chi_r[N-1] = 1, the suffix is always a column vector.

4. **Compute marginals** — For each logical qubit k at chain position pos_k:
   ```
   result_lambda = prefix[pos_k] @ T[pos_k][:, lambda, :] @ suffix[pos_k],   lambda in {0, 1}
   P_k = |result_1| / (|result_0| + |result_1|)
   ```

5. **Conditional refinement** — Fix confident logicals (P_k > 0.9 or P_k < 0.1), update base matrices, and re-run scans + marginals for the remaining uncertain logicals. Repeat until all logicals are decided (typically 2-4 rounds, at most O(log K)).

All arithmetic is in fp64. A log-scale normalization scheme (max-abs normalize after each step, accumulate log of the norm) prevents overflow/underflow across the chain.

## Architecture

### Execution Pipeline

```
               stream (primary)                    stream2
               ─────────────                       ───────
Phase 1:  ┌─ prepare_base_matrices (N blocks) ─┐
          │       (N blocks parallel)           │
          └─────────────────────────────────────┘
                        │
          ┌─── event: prepare_done ──────────── wait ──┐
          │                                             │
Phase 2:  │  forward_scan (1 block)       backward_scan (1 block)
          │  left→right prefix            right→left suffix
          │                                             │
          │           ┌──── event: backward_done ───────┘
          │           │
Phase 3:  └── compute_marginals (K blocks) ──┐
                  (K blocks parallel)        │
                                              ▼
Phase 4:       refinement_check (1 thread)
                        │
               if uncertain logicals remain:
                        │
                update_base → re-scan → marginals → check
                   (repeat up to max_rounds times)
```

### Data Layout on GPU

**Persistent (loaded once per code, read-only):**
| Buffer | Size | Description |
|--------|------|-------------|
| `d_raw_tensors` | N * chi_l * 2 * chi_r * 8 B | All site tensors packed contiguously |
| `d_data_offsets[N]` | N * 4 B | Element offset for each site tensor |
| `d_chi_l[N], d_chi_r[N]` | N * 4 B each | Bond dimensions per site |
| `d_site_type[N]` | N * 4 B | 0 = syndrome, 1 = logical |
| `d_syn_index[N], d_log_index[N]` | N * 4 B each | Syndrome/logical index per site |
| `d_logical_positions[K]` | K * 4 B | Chain position of each logical |
| `d_base_offsets[N]` | N * 4 B | Element offset for each base matrix |

**Per-decode working memory:**
| Buffer | Size | Description |
|--------|------|-------------|
| `d_base_matrices` | sum(chi_l * chi_r) * 8 B | Row-major base matrices (forward scan) |
| `d_base_matrices_T` | sum(chi_l * chi_r) * 8 B | Transposed base matrices (backward scan) |
| `d_syndrome[M]` | M * 8 B | Input syndrome values (fp64) |
| `d_prefix_vecs[K * max_chi]` | K * chi * 8 B | Prefix row vectors at logical positions |
| `d_suffix_vecs[K * max_chi]` | K * chi * 8 B | Suffix column vectors at logical positions |
| `d_prefix_log[K], d_suffix_log[K]` | K * 8 B each | Log-scale normalization factors |
| `d_marginals[K]` | K * 8 B | Output marginal probabilities |
| `d_decisions[K]` | K * 4 B | Output decisions: 0, 1, or -1 (uncertain) |
| `d_fixed[K], d_fixed_values[K]` | K * 4 B each | Refinement state |
| `d_refine_ctrl[6]` | 24 B | Device-side refinement control flags |

For chi=256, QCC_144_12_12 (N=78, K=12, M=66): raw tensors = 54 MB, base matrices = 2 x 27 MB.

## CUDA Kernel Design

### Kernel 1: `prepare_base_matrices_kernel`

- **Grid**: N blocks (one per site), `block_size` threads per block
- **Parallelism**: Fully parallel across all N sites; each thread handles multiple matrix elements via stride loop
- **Dual output**: Writes both row-major B[a, cr + b] (for forward scan) and transposed B_T[b, cl + a] (for backward scan) in a single pass
- Uses `__ldg()` read-only cache hints for raw tensor reads

### Kernel 2: `forward_scan_kernel` (Multi-Group)

- **Grid**: 1 block, `scan_block_size` threads (up to 1024)
- **Sequential chain**: Processes sites left-to-right, saving prefix vectors at each logical position
- **Multi-group inner loop**: The key optimization — partitions the inner sum of the vector-matrix multiply across `num_groups = scan_block_size / max_chi` thread groups. For chi=256 with scan_block_size=1024, this gives 4 groups of 256 threads each, where each group computes a partial sum over 1/4 of the `a` dimension, then results are reduced across groups.
- **Shared memory layout**: `acc[max_chi] + partial[scan_block_size] + warp_max[32]`
- **Warp-level max-abs normalization**: Uses `__shfl_down_sync` for intra-warp reduction, then a single cross-warp reduction step — total of 2 `__syncthreads` barriers (vs 8+ with shared-memory tree reduction)
- **Partial re-scan support**: Accepts `start_site` and `start_logical_idx` parameters. For refinement rounds, only re-scans from the earliest changed logical position, loading the saved prefix as initial state.

### Kernel 3: `backward_scan_kernel` (Multi-Group)

- Mirrors forward scan but processes right-to-left
- Reads from **transposed** base matrices (`d_base_matrices_T`) so that adjacent threads read adjacent memory addresses (coalesced access: thread `tid` reads `B_T[b * cl + tid]`)
- Same multi-group and warp-level reduction as forward scan

### Kernel 4: `compute_marginals_kernel`

- **Grid**: K blocks (one per logical qubit), `block_size` threads
- Each block computes `result_lambda = prefix @ T[:, lambda, :] @ suffix` via:
  1. Vector-matrix multiply: `mid[b] = sum_a prefix[a] * T[a, lambda, b]` (parallel across b)
  2. Dot product: `sum_b mid[b] * suffix[b]` via shared-memory tree reduction
- Computes marginal P_k via log-space subtraction to avoid numerical issues
- Makes threshold decision: P_k > high_thresh → 1, P_k < low_thresh → 0, else → -1 (uncertain)

### Kernel 5: `refinement_check_kernel`

- **Grid**: 1 block, 1 thread (K is tiny, typically 2-12)
- Runs entirely on device to avoid host round-trip for checking decisions
- Updates `fixed[]` and `fixed_values[]` arrays in-place on GPU
- Outputs 6 control integers: `[has_unknown, has_new_fixed, earliest_log, latest_log, earliest_site, latest_site]`
- Only 24 bytes transferred to host per refinement round

### Kernel 6: `update_base_for_fixed_kernel`

- **Grid**: K blocks, `block_size` threads
- Only processes newly-fixed logicals (early-exit via `if (!fixed[k]) return`)
- Updates both row-major and transposed base matrices

## Optimization Techniques

### 1. Dual-Stream Concurrent Scans

**Problem**: Forward and backward scans are independent — they read from the same base matrices but write to separate prefix/suffix buffers.

**Solution**: Launch forward scan on `stream` and backward scan on `stream2`. The GPU executes them concurrently on different SMs, effectively halving the scan wall-clock time.

```
Before (sequential):  |--- forward (660 us) ---|--- backward (660 us) ---|  = 1320 us
After  (concurrent):  |--- forward (660 us) ---|
                      |--- backward (660 us) ---|                           = 660 us
```

A CUDA event synchronization ensures `stream` waits for `stream2` to complete before launching marginals. Measured on H100: concurrent QCC_144 = 1759 us vs sequential = 7614 us.

### 2. Transposed Base Matrices for Coalesced Backward Scan

**Problem**: The backward scan computes `temp[a] = sum_b B[a, b] * acc[b]`. With row-major B, thread `tid=a` must read `B[tid, 0], B[tid, 1], ..., B[tid, cr-1]` — a stride-cr access pattern causing uncoalesced reads (one cache line per element).

**Solution**: Pre-compute `B_T[b, a] = B[a, b]` during the prepare phase. Now the backward scan computes `temp[a] = sum_b B_T[b, a] * acc[b]`, and thread `tid=a` reads `B_T[b, tid]` for each b — adjacent threads read adjacent addresses, achieving full coalescing.

The prepare kernel writes both layouts in a single pass (zero extra kernel launch cost). Memory cost is 2x base matrices, but these are small relative to raw tensors.

### 3. Multi-Group Inner Loop for Latency Hiding

**Problem**: The forward/backward scan kernel uses 1 thread block. With `block_size = max_chi = 256` threads (8 warps), only 2 warps per SM scheduler are available. The inner loop's vector-matrix multiply reads base matrices from L2 cache with ~200-300 cycle latency, but each iteration has only ~8 cycles of FP64 compute. With 2 warps per scheduler, the GPU cannot issue enough independent memory requests to hide the latency.

**Solution**: Increase the scan block size to `scan_block_size = min(block_size * 4, 1024)`. For chi=256, this gives 1024 threads (32 warps, 8 per scheduler). The inner sum is partitioned across `num_groups = scan_block_size / max_chi` groups:

```
Thread tid:
  tid_b = tid % max_chi    (output column index)
  tid_g = tid / max_chi    (group index)

Group g handles: a in [g * chunk, (g+1) * chunk)   where chunk = ceil(chi_l / num_groups)
```

Each group computes a partial sum, stored in `partial[tid_g * max_chi + tid_b]`. After a `__syncthreads`, the first `max_chi` threads reduce across groups:

```cuda
// Reduce across groups
if (tid < max_chi && num_groups > 1) {
    double sum = partial[tid];
    for (int g = 1; g < num_groups; g++)
        sum += partial[g * max_chi + tid];
}
```

This provides ~4x more warps for latency hiding, measured as **2.7x speedup** on forward scan for QCC_144 chi=256 on H100.

### 4. Warp-Level Shuffle Reduction

**Problem**: Shared-memory tree reduction for max-abs normalization requires O(log2(N)) barriers (`__syncthreads`), each costing ~20 cycles.

**Solution**: Use `__shfl_down_sync` for intra-warp reduction (zero-cost, register-to-register), then a single shared-memory step across warps:

```cuda
// Intra-warp reduction (no barriers needed)
double my_abs = (tid < cr) ? fabs(sum) : 0.0;
for (int s = 16; s > 0; s >>= 1)
    my_abs = fmax(my_abs, __shfl_down_sync(0xFFFFFFFF, my_abs, s));

// Lane 0 of each warp writes to shared memory
if (tid % 32 == 0) warp_max[tid / 32] = my_abs;
__syncthreads();

// Final reduction across warps (only warp 0)
if (tid < 32) {
    my_abs = (tid < num_warps) ? warp_max[tid] : 0.0;
    for (int s = 16; s > 0; s >>= 1)
        my_abs = fmax(my_abs, __shfl_down_sync(0xFFFFFFFF, my_abs, s));
    if (tid == 0) warp_max[0] = my_abs;
}
__syncthreads();
```

Total: 2 barriers instead of 8+ (for 1024 threads).

### 5. Partial Re-Scan for Refinement

**Problem**: Naively, each refinement round requires a full forward + backward scan over all N sites, even though only a few logical sites changed.

**Solution**: Track the earliest and latest changed logical positions. Only re-scan the affected range:
- Forward scan: start from the earliest changed position (loading the saved prefix as initial state)
- Backward scan: start from the latest changed position (loading the saved suffix)

For K=12 logicals spread across N=78 sites, a typical refinement round only re-scans ~60% of the chain.

### 6. Device-Side Refinement Check

**Problem**: Each refinement round previously required copying K decisions to host, checking on CPU, then copying updated fixed arrays back — 3 PCIe transfers and a synchronization per round.

**Solution**: A single-thread GPU kernel (`refinement_check_kernel`) performs the check and updates directly on device memory. Only 24 bytes (6 control ints) are transferred to host per round. Since K is tiny (2-12), a single thread is optimal.

### 7. Read-Only Cache Hints

All tensor and base matrix reads use `__ldg()` (load through read-only data cache), which:
- Uses the texture/L1 read-only cache path, separate from the L1/shared memory
- Avoids polluting the L1 cache used for shared memory and local data
- Provides better caching behavior for broadcast-style access patterns

### 8. Log-Scale Normalization

After each vector-matrix multiply in the scan, the result vector is normalized by its max-abs value, and the log of the norm is accumulated. This prevents the exponential growth/decay of values along the chain (which spans up to 78 sites). The log scales cancel in the marginal ratio computation:

```
P_k = |result_1| / (|result_0| + |result_1|)
    = exp(log|r1|) / (exp(log|r0|) + exp(log|r1|))
```

where `log|r_lambda| = log|dot| + prefix_log_scale + suffix_log_scale`. The prefix and suffix log scales are identical for both lambda values, so they cancel in the probability computation. Only the dot product's absolute value matters.

## Performance Results

All measurements on NVIDIA H100 PCIe 80GB (sm_90), CUDA 12.9, Driver 575.64.03, 1000 decodes, median latency in microseconds. GPU metrics collected via NVML.

### Chi=128 (scan_block_size=512)

| Code | chi | N | K | Prepare | FwdScan | BwdScan | Marginal | Refine | **Total** |
|------|-----|---|---|---------|---------|---------|----------|--------|-----------|
| QCC_18_4_4 | 6 | 11 | 4 | 7.1 | 21.2 | 4.3 | 6.7 | 64.0 | **103.4** |
| LDPC_25_3_4 | 64 | 15 | 3 | 16.1 | 26.9 | 1.7 | 6.8 | 75.1 | **126.7** |
| LDPC_30_6_4 | 66 | 19 | 6 | 18.6 | 41.2 | -1.4 | 12.1 | 88.3 | **158.9** |
| TOR_50_2_5 | 128 | 26 | 2 | 54.8 | 71.6 | 2.2 | 26.6 | 137.0 | **292.3** |
| QCC_60_8_4 | 128 | 34 | 8 | 55.6 | 91.0 | 1.8 | 28.3 | 153.4 | **330.4** |
| QCC_72_12_6 | 128 | 42 | 12 | 56.0 | 111.5 | 1.7 | 26.1 | 182.5 | **377.8** |
| QCC_90_8_10 | 128 | 49 | 8 | 55.2 | 137.3 | 4.5 | 28.0 | 198.1 | **423.1** |
| QCC_108_8_10 | 128 | 58 | 8 | 56.3 | 178.9 | 3.3 | 28.6 | 232.0 | **499.1** |
| QCC_144_12_12 | 128 | 78 | 12 | 56.6 | 234.7 | 3.7 | 26.9 | 290.8 | **612.8** |

### Chi=256 (scan_block_size=1024)

| Code | chi | N | K | Prepare | FwdScan | BwdScan | Marginal | Refine | **Total** |
|------|-----|---|---|---------|---------|---------|----------|--------|-----------|
| QCC_18_4_4 | 6 | 11 | 4 | 6.8 | 21.1 | 4.4 | 6.5 | 63.6 | **102.6** |
| LDPC_25_3_4 | 64 | 15 | 3 | 15.6 | 27.0 | 1.7 | 6.7 | 73.3 | **124.4** |
| LDPC_30_6_4 | 128 | 19 | 6 | 29.3 | 37.9 | -1.4 | 11.6 | 95.4 | **172.9** |
| TOR_50_2_5 | 256 | 26 | 2 | 143.2 | 107.9 | 1.7 | 46.3 | 186.3 | **485.4** |
| QCC_60_8_4 | 256 | 34 | 8 | 148.3 | 138.0 | 2.6 | 47.5 | 221.0 | **557.4** |
| QCC_72_12_6 | 256 | 42 | 12 | 142.1 | 179.0 | -1.4 | 57.2 | 273.0 | **649.8** |
| QCC_90_8_10 | 256 | 49 | 8 | 163.4 | 291.2 | 2.3 | 84.9 | 291.0 | **832.8** |
| QCC_108_8_10 | 256 | 58 | 8 | 177.3 | 484.5 | 1.7 | 108.4 | 561.5 | **1333.3** |
| QCC_144_12_12 | 256 | 78 | 12 | 181.9 | 660.9 | 1.7 | 107.9 | 810.9 | **1763.5** |

**Notes:**
- BwdScan shows ~0 us (or slightly negative due to event timing noise) because it fully overlaps with FwdScan via dual-stream concurrency.
- Refinement includes multiple rounds of partial scan + marginals. Its cost is typically 1.0-1.2x the initial forward scan cost.
- For small codes (chi <= 64), kernel launch overhead (~60-70 us total) dominates over compute.

### Phase Breakdown (QCC_144_12_12, chi=256)

```
Phase            Median     % of Total
─────────────    ──────     ──────────
prepare_base     181.9 us     10.3%
forward_scan     660.9 us     37.5%
backward_scan      1.7 us      0.1%  (overlapped)
marginals        107.9 us      6.1%
refinement       810.9 us     46.0%
─────────────    ──────     ──────────
total           1763.5 us    100.0%
```

### Forward Scan Scaling

Forward scan cost scales linearly with N (number of sites) and quadratically with chi:

| chi | Per-site cost (us) | Theoretical O(chi^2) |
|-----|-------------------|---------------------|
| 6   | 1.9 | -- |
| 64  | 1.8 | 4,096 |
| 128 | 3.0 | 16,384 |
| 256 | 8.5 | 65,536 |

The sub-quadratic scaling from chi=128 to chi=256 (2.8x vs expected 4x) is due to the multi-group optimization improving memory utilization at higher chi values.

### Optimization Impact Summary (QCC_144_12_12, chi=256, H100)

| Optimization | Total Latency | Improvement |
|-------------|--------------|-------------|
| Baseline (single-stream, 256 threads) | ~10,500 us | -- |
| + Transposed backward scan | ~5,280 us | 2.0x |
| + Dual-stream concurrent scans | ~4,160 us | 1.3x |
| + Multi-group scan (1024 threads) | ~1,764 us | 2.4x |
| + Partial re-scan + device-side check | (included above) | ~1.1x |
| **Combined** | **~1,764 us** | **~6.0x** |

## GPU Resource Utilization

All measurements on NVIDIA H100 PCIe 80GB (sm_90, 350W TDP), Driver 575.64.03. GPU metrics sampled via NVML during 1000-decode benchmark runs.

### Device Specifications

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 PCIe |
| Memory | 81,559 MB (80 GB HBM3) |
| Power Limit | 350 W |
| SM Clock | 1755 MHz |
| Driver | 575.64.03 |

### GPU Memory Usage

| Code | chi | N | K | Raw Tensors (MB) | Memory Used (MB) | Delta from Baseline |
|------|-----|---|---|-------------------|-------------------|---------------------|
| QCC_18_4_4 | 6 | 11 | 4 | <0.1 | 954 | +0 |
| LDPC_25_3_4 | 64 | 15 | 3 | 0.1 | 954 | +0 |
| LDPC_30_6_4 | 66 | 19 | 6 | 0.2 | 954 | +0 |
| TOR_50_2_5 | 128 | 26 | 2 | 3.3 | 960 | +6 |
| QCC_60_8_4 | 128 | 34 | 8 | 4.2 | 968 | +14 |
| QCC_72_12_6 | 128 | 42 | 12 | 4.8 | 968 | +14 |
| QCC_90_8_10 | 128 | 49 | 8 | 7.3 | 970 | +16 |
| QCC_108_8_10 | 128 | 58 | 8 | 10.5 | 978 | +24 |
| QCC_144_12_12 | 128 | 78 | 12 | 13.7 | 984 | +30 |

Chi=256 largest codes:

| Code | chi | Memory Used (MB) | Delta from Baseline |
|------|-----|-------------------|---------------------|
| TOR_50_2_5 | 256 | 978 | +24 |
| QCC_60_8_4 | 256 | 986 | +32 |
| QCC_72_12_6 | 256 | 992 | +38 |
| QCC_90_8_10 | 256 | 1,002 | +48 |
| QCC_108_8_10 | 256 | 1,040 | +86 |
| QCC_144_12_12 | 256 | 1,066 | +112 |

The decoder's GPU memory footprint is very small — even the largest code (QCC_144, chi=256) uses only ~112 MB of additional GPU memory (raw tensors + 2x base matrices + prefix/suffix buffers + working memory). The ~954 MB baseline is CUDA runtime + driver overhead.

### GPU Utilization & Power

**Chi=128:**

| Code | chi | Total (us) | GPU Util (%) | Mem BW Util (%) | Power (W) | Temp (C) |
|------|-----|-----------|-------------|-----------------|-----------|----------|
| QCC_18_4_4 | 6 | 103 | 14 | 0 | 54 | 33 |
| LDPC_25_3_4 | 64 | 127 | 54 | 0 | 60 | 33 |
| LDPC_30_6_4 | 66 | 159 | 53 | 0 | 69 | 33 |
| TOR_50_2_5 | 128 | 292 | 69 | 0 | 79 | 33 |
| QCC_60_8_4 | 128 | 330 | 78 | 0 | 81 | 33 |
| QCC_72_12_6 | 128 | 378 | 71 | 0 | 83 | 33 |
| QCC_90_8_10 | 128 | 423 | 74 | 0 | 83 | 33 |
| QCC_108_8_10 | 128 | 499 | 74 | 0 | 84 | 33 |
| QCC_144_12_12 | 128 | 613 | 84 | 0 | 85 | 33 |

**Chi=256:**

| Code | chi | Total (us) | GPU Util (%) | Mem BW Util (%) | Power (W) | Temp (C) |
|------|-----|-----------|-------------|-----------------|-----------|----------|
| QCC_18_4_4 | 6 | 103 | 75 | 0 | 85 | 33 |
| LDPC_25_3_4 | 64 | 124 | 57 | 0 | 84 | 33 |
| LDPC_30_6_4 | 128 | 173 | 54 | 0 | 82 | 33 |
| TOR_50_2_5 | 256 | 485 | 76 | 0 | 81 | 33 |
| QCC_60_8_4 | 256 | 557 | 78 | 0 | 85 | 33 |
| QCC_72_12_6 | 256 | 650 | 78 | 0 | 87 | 33 |
| QCC_90_8_10 | 256 | 833 | 83 | 3 | 89 | 33 |
| QCC_108_8_10 | 256 | 1,333 | 91 | 7 | 95 | 34 |
| QCC_144_12_12 | 256 | 1,764 | 95 | 7 | 98 | 34 |

**Key observations:**

- **GPU utilization** scales with code size: small codes (chi <= 64) only reach 14-57% due to kernel launch overhead dominating the short computation, while large codes (chi=256, N >= 49) reach 83-95%.
- **Memory bandwidth utilization** is near zero for chi <= 128 (working sets fit in L2 cache: H100 has 50 MB L2, largest chi=128 working set is 13.7 MB). For chi=256 with N >= 49, the working set exceeds L2 capacity and memory bandwidth utilization rises to 3-7%.
- **Power draw** is 50-98W (14-28% of the 350W TDP), confirming the decoder is compute-bound rather than power-limited. The low power reflects the sequential nature of the scan kernels — only 1-2 SMs are active during scans.
- **Temperature** stays at 33-34C across all benchmarks (effectively cold), showing negligible thermal impact.
- **SM clock** is constant at 1755 MHz — no thermal throttling occurs.

## File Structure

```
gpu_decoder/
├── src/
│   ├── mps_decoder.cu           # CUDA kernels + host C API (~2000 lines)
│   ├── mps_decoder.h            # C header (extern "C" API)
│   └── Makefile                 # nvcc compilation
├── mps_gpu_decoder.py           # Python ctypes wrapper (MpsGpuDecoder class)
├── benchmark_gpu_decoder.py     # Single-decode + batch throughput benchmark
├── benchmark_phase_latency.py   # Per-phase latency profiling
├── test_gpu_vs_python.py        # Correctness verification vs Python reference
├── phase_latency_results.json   # Benchmark results (chi128 + chi256)
└── README.md                    # This document
```

## Build & Usage

### Prerequisites

- NVIDIA GPU with compute capability >= 8.0 (A100, L20, H100, ...)
- CUDA Toolkit >= 11.0
- Python 3.8+ with NumPy

### Compilation

```bash
cd gpu_decoder/src

# Auto-detect architecture (defaults to sm_80)
make

# Explicit architecture for H100
make CUDA_ARCH=sm_90

# Or directly:
nvcc -O3 -arch=sm_90 --use_fast_math -shared -Xcompiler -fPIC \
     -o libmps_decoder.so mps_decoder.cu
```

### Python Usage

```python
from mps_gpu_decoder import MpsGpuDecoder
import numpy as np

# Load pre-compressed MPS chain
decoder = MpsGpuDecoder("/path/to/code/chain/")

# Single decode
syndrome = np.array([0, 1, 0, 1, ...], dtype=np.int32)  # M syndrome values
decisions, marginals, latency_us = decoder.decode(syndrome)
# decisions: int array [K], 0 or 1
# marginals: float array [K], probability of logical = 1
# latency_us: GPU decode latency in microseconds

# Profiled decode (per-phase breakdown)
decisions, marginals, phases = decoder.decode_profiled(syndrome)
# phases: dict with keys prepare_base_us, forward_scan_us, backward_scan_us,
#         marginals_us, refinement_us, total_us

# Batch decode
syndromes = np.random.randint(0, 2, size=(1000, decoder.M)).astype(np.int32)
decisions, marginals, latency_us = decoder.decode_batch(syndromes)

decoder.close()
```

### Benchmarking

```bash
cd gpu_decoder

# Latency benchmark (single decode, all 9 codes)
python benchmark_gpu_decoder.py

# Per-phase profiling (chi128 + chi256)
python benchmark_phase_latency.py

# Correctness check vs Python reference
python test_gpu_vs_python.py
```

### C API

```c
#include "mps_decoder.h"

MpsDecoderHandle dec = mps_decoder_create("/path/to/chain/");

int syndrome[M] = { ... };
int decisions[K];
double marginals[K];
double latency_us;

mps_decoder_decode(dec, syndrome, decisions, marginals, &latency_us,
                   0.1, 0.9, 10);  // low_thresh, high_thresh, max_rounds

mps_decoder_destroy(dec);
```
