/**
 * mps_decoder.cu — CUDA kernels + host C API for MPS-based online decoding.
 *
 * Algorithm: For a 1D MPS chain with N sites (interleaved syndrome + logical),
 * compute marginal probabilities for each logical qubit via forward/backward
 * prefix/suffix products, then apply conditional refinement.
 *
 * All tensors are fp64. Bond dimensions vary per site (up to max_chi).
 *
 * IMPORTANT: In global memory, prefix/suffix vectors are stored with stride
 * = max_chi (the actual maximum bond dimension). In shared memory, arrays
 * use blockDim.x as spacing (which is >= max_chi, rounded to power of 2).
 */

#include "mps_decoder.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

// ---- Minimal .npy reader (for float64 arrays) ----

struct NpyArray {
    std::vector<int> shape;
    std::vector<double> data;
};

static bool read_npy(const char* path, NpyArray& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    // .npy format: 6 magic bytes + 2 version bytes + 2 header_len bytes + header
    unsigned char preamble[10];
    if (fread(preamble, 1, 10, f) != 10) { fclose(f); return false; }

    // Header length is at bytes 8-9 (little-endian) for v1.0
    uint16_t header_len = (uint16_t)preamble[8] | ((uint16_t)preamble[9] << 8);

    // Read header string
    std::vector<char> header(header_len + 1);
    if (fread(header.data(), 1, header_len, f) != header_len) {
        fclose(f); return false;
    }
    header[header_len] = '\0';
    std::string hdr(header.data());

    // Parse shape from header: ...'shape': (d0, d1, d2), ...
    size_t shape_pos = hdr.find("'shape':");
    if (shape_pos == std::string::npos) shape_pos = hdr.find("\"shape\":");
    if (shape_pos == std::string::npos) { fclose(f); return false; }

    size_t paren_start = hdr.find('(', shape_pos);
    size_t paren_end = hdr.find(')', paren_start);
    if (paren_start == std::string::npos || paren_end == std::string::npos) {
        fclose(f); return false;
    }
    std::string shape_str = hdr.substr(paren_start + 1, paren_end - paren_start - 1);

    out.shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ','))
            pos++;
        if (pos >= shape_str.size()) break;
        int val = atoi(shape_str.c_str() + pos);
        out.shape.push_back(val);
        while (pos < shape_str.size() && shape_str[pos] != ',' && shape_str[pos] != ')')
            pos++;
    }

    int total = 1;
    for (int d : out.shape) total *= d;

    out.data.resize(total);
    size_t read_count = fread(out.data.data(), sizeof(double), total, f);
    fclose(f);
    return (int)read_count == total;
}

// ---- Minimal JSON parser for chain_meta.json ----

static std::string read_file_to_string(const char* path) {
    std::ifstream ifs(path);
    if (!ifs) return "";
    return std::string((std::istreambuf_iterator<char>(ifs)),
                       std::istreambuf_iterator<char>());
}

static std::vector<std::string> parse_json_string_array(const std::string& json,
                                                         const std::string& key) {
    std::vector<std::string> result;
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return result;

    size_t bracket = json.find('[', pos);
    size_t end_bracket = json.find(']', bracket);
    if (bracket == std::string::npos || end_bracket == std::string::npos) return result;

    std::string arr = json.substr(bracket + 1, end_bracket - bracket - 1);
    size_t p = 0;
    while (p < arr.size()) {
        size_t q1 = arr.find('"', p);
        if (q1 == std::string::npos) break;
        size_t q2 = arr.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        result.push_back(arr.substr(q1 + 1, q2 - q1 - 1));
        p = q2 + 1;
    }
    return result;
}

static int parse_json_int(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return -1;
    size_t colon = json.find(':', pos);
    if (colon == std::string::npos) return -1;
    return atoi(json.c_str() + colon + 1);
}

// ---- Decoder internal state ----

struct MpsDecoder {
    // Metadata
    int N;          // number of sites
    int K;          // number of logical qubits
    int M;          // number of syndrome sites
    int max_chi;    // max bond dimension across all sites
    int block_size; // thread block size (>= max_chi, power of 2, min 32)
    int scan_block_size; // scan kernel block size (>= block_size, up to 1024, for latency hiding)

    std::vector<std::string> site_labels;
    std::vector<std::string> syndrome_labels;
    std::vector<std::string> logical_labels;
    std::vector<int> logical_positions;  // chain position for each logical (indexed by log_index)

    // GPU memory: raw tensor data (persistent, read-only after init)
    double* d_raw_tensors;     // packed tensor data
    int*    d_data_offsets;    // byte offset for each site
    int*    d_chi_l;           // left bond dim per site
    int*    d_chi_r;           // right bond dim per site
    int*    d_site_type;       // 0=syndrome, 1=logical
    int*    d_syn_index;       // syndrome index per site (-1 if logical)
    int*    d_log_index;       // logical index per site (-1 if syndrome)
    int*    d_logical_positions;
    int*    d_base_offsets;    // offset for each site's base matrix

    // GPU memory: per-decode working buffers (single decode)
    double* d_base_matrices;   // row-major base matrices (for forward scan)
    double* d_base_matrices_T; // transposed base matrices (for backward scan, coalesced reads)
    double* d_syndrome;        // M doubles
    double* d_prefix_vecs;     // K * max_chi doubles
    double* d_suffix_vecs;     // K * max_chi doubles
    double* d_prefix_log;      // K doubles
    double* d_suffix_log;      // K doubles
    double* d_marginals;       // K doubles
    int*    d_decisions;       // K ints
    int*    d_fixed;           // K ints
    int*    d_fixed_values;    // K ints
    int*    d_refine_ctrl;     // 6 ints: [has_unknown, has_new_fixed, earliest_log, latest_log, earliest_site, latest_site]

    // Batch buffers (allocated on demand)
    int batch_alloc;
    double* d_batch_syndrome;
    double* d_batch_base;
    double* d_batch_base_T;    // transposed batch base matrices
    double* d_batch_prefix;
    double* d_batch_suffix;
    double* d_batch_prefix_log;
    double* d_batch_suffix_log;
    double* d_batch_marginals;
    int*    d_batch_decisions;
    int*    d_batch_fixed;
    int*    d_batch_fixed_values;

    // Host-side data
    int total_raw_elements;
    int total_base_elements;
    std::vector<int> h_base_offsets;

    cudaStream_t stream;
    cudaStream_t stream2;  // second stream for backward scan (dual-stream overlap)

    // Adaptive scan mode: sequential when working set > L2 cache
    int l2_cache_size;         // device L2 cache in bytes
    bool use_sequential_scan;  // true = both scans on same stream (avoids L2 thrashing)

    // ---- FP32 mixed-precision scan ----
    bool use_fp32_scan;            // enabled by default for chi >= 128
    int scan_block_size_f32;       // scan block size for FP32 mode (typically max_chi*2)

    // FP32 working buffers (single decode)
    float* d_base_matrices_f32;    // FP32 row-major base matrices
    float* d_base_matrices_T_f32;  // FP32 transposed base matrices
    float* d_prefix_vecs_f32;      // K * max_chi floats
    float* d_suffix_vecs_f32;      // K * max_chi floats
    float* d_prefix_log_f32;       // K floats (log-scale per logical)
    float* d_suffix_log_f32;       // K floats

    // FP32 batch buffers (allocated on demand alongside FP64 batch buffers)
    float* d_batch_base_f32;
    float* d_batch_base_T_f32;
    float* d_batch_prefix_f32;
    float* d_batch_suffix_f32;
    float* d_batch_prefix_log_f32;
    float* d_batch_suffix_log_f32;

    // Device-side refinement (eliminates host↔device sync in refinement loop)
    int*   d_refine_skip;   // 1 int: once set to 1, all subsequent refinement rounds skip
    int*   d_final_skip;    // 1 int: if 1, final forced pass is skipped
    int*   d_scan_range;    // 4 ints: [fwd_start_site, fwd_start_log, bwd_end_site, bwd_end_log]
};

// ---- CUDA error checking ----

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

// ============================================================================
// CUDA Kernels
// ============================================================================

// ---- Kernel 1: Prepare base matrices ----
// Syndrome site: base = T[:, s, :]  (s = 0 or 1 from syndrome)
// Logical unfixed: base = T[:, 0, :] + T[:, 1, :]
// Logical fixed:   base = T[:, v, :]

__global__ void prepare_base_matrices_kernel(
    const double* __restrict__ raw_tensors,
    const int* __restrict__ data_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ site_type,
    const int* __restrict__ syn_index,
    const int* __restrict__ log_index,
    const double* __restrict__ syndrome,
    const int* __restrict__ fixed,
    const int* __restrict__ fixed_values,
    double* __restrict__ base_matrices,
    double* __restrict__ base_matrices_T,  // transposed output for backward scan
    const int* __restrict__ base_offsets,
    int N)
{
    int site = blockIdx.x;
    if (site >= N) return;

    int cl = chi_l[site];
    int cr = chi_r[site];
    int offset_raw = data_offsets[site];
    int offset_base = base_offsets[site];
    int mat_size = cl * cr;

    // Raw tensor: [chi_l, 2, chi_r] row-major → element (a, p, b) at a*2*cr + p*cr + b
    // Base matrix: [chi_l, chi_r] row-major → element (a, b) at a*cr + b
    // Transposed:  [chi_r, chi_l] row-major → element (b, a) at b*cl + a

    if (site_type[site] == 0) {
        // Syndrome site
        int si = syn_index[site];
        int s = (int)(syndrome[si] + 0.5);  // round to 0 or 1
        for (int idx = threadIdx.x; idx < mat_size; idx += blockDim.x) {
            int a = idx / cr;
            int b = idx % cr;
            double val = __ldg(&raw_tensors[offset_raw + a * 2 * cr + s * cr + b]);
            base_matrices[offset_base + a * cr + b] = val;
            base_matrices_T[offset_base + b * cl + a] = val;
        }
    } else {
        // Logical site
        int li = log_index[site];
        if (fixed[li]) {
            int v = fixed_values[li];
            for (int idx = threadIdx.x; idx < mat_size; idx += blockDim.x) {
                int a = idx / cr;
                int b = idx % cr;
                double val = __ldg(&raw_tensors[offset_raw + a * 2 * cr + v * cr + b]);
                base_matrices[offset_base + a * cr + b] = val;
                base_matrices_T[offset_base + b * cl + a] = val;
            }
        } else {
            // Sum over physical dim
            for (int idx = threadIdx.x; idx < mat_size; idx += blockDim.x) {
                int a = idx / cr;
                int b = idx % cr;
                double v0 = __ldg(&raw_tensors[offset_raw + a * 2 * cr + 0 * cr + b]);
                double v1 = __ldg(&raw_tensors[offset_raw + a * 2 * cr + 1 * cr + b]);
                double val = v0 + v1;
                base_matrices[offset_base + a * cr + b] = val;
                base_matrices_T[offset_base + b * cl + a] = val;
            }
        }
    }
}

// Batch version: grid (N, batch_size)
__global__ void prepare_base_matrices_batch_kernel(
    const double* __restrict__ raw_tensors,
    const int* __restrict__ data_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ site_type,
    const int* __restrict__ syn_index,
    const int* __restrict__ log_index,
    const double* __restrict__ syndrome,     // batch * M
    const int* __restrict__ fixed,           // batch * K
    const int* __restrict__ fixed_values,    // batch * K
    double* __restrict__ base_matrices,      // batch * total_base
    double* __restrict__ base_matrices_T,    // transposed, batch * total_base
    const int* __restrict__ base_offsets,
    int N, int M, int K, int total_base_elements)
{
    int site = blockIdx.x;
    int batch = blockIdx.y;
    if (site >= N) return;

    int cl = chi_l[site];
    int cr = chi_r[site];
    int offset_raw = data_offsets[site];
    long long offset_base = (long long)batch * total_base_elements + base_offsets[site];
    int mat_size = cl * cr;

    if (site_type[site] == 0) {
        int si = syn_index[site];
        int s = (int)(syndrome[batch * M + si] + 0.5);
        for (int idx = threadIdx.x; idx < mat_size; idx += blockDim.x) {
            int a = idx / cr;
            int b = idx % cr;
            double val = __ldg(&raw_tensors[offset_raw + a * 2 * cr + s * cr + b]);
            base_matrices[offset_base + a * cr + b] = val;
            base_matrices_T[offset_base + b * cl + a] = val;
        }
    } else {
        int li = log_index[site];
        if (fixed[batch * K + li]) {
            int v = fixed_values[batch * K + li];
            for (int idx = threadIdx.x; idx < mat_size; idx += blockDim.x) {
                int a = idx / cr;
                int b = idx % cr;
                double val = __ldg(&raw_tensors[offset_raw + a * 2 * cr + v * cr + b]);
                base_matrices[offset_base + a * cr + b] = val;
                base_matrices_T[offset_base + b * cl + a] = val;
            }
        } else {
            for (int idx = threadIdx.x; idx < mat_size; idx += blockDim.x) {
                int a = idx / cr;
                int b = idx % cr;
                double v0 = __ldg(&raw_tensors[offset_raw + a * 2 * cr + 0 * cr + b]);
                double v1 = __ldg(&raw_tensors[offset_raw + a * 2 * cr + 1 * cr + b]);
                double val = v0 + v1;
                base_matrices[offset_base + a * cr + b] = val;
                base_matrices_T[offset_base + b * cl + a] = val;
            }
        }
    }
}

// ---- Kernel: Refinement check (device-side) ----
//
// Checks decisions, updates fixed/fixed_values on device.
// Writes control flags: [has_unknown, has_new_fixed, earliest_log, latest_log, earliest_site, latest_site]
// Also sets refine_skip flag and scan_range for fully device-side refinement.
// Single-thread kernel since K is tiny (<=12).

__global__ void refinement_check_kernel(
    const int* __restrict__ decisions,
    int* __restrict__ fixed,
    int* __restrict__ fixed_values,
    const int* __restrict__ logical_positions,
    int* __restrict__ control,  // 6 ints output
    int K,
    int* __restrict__ refine_skip,   // if non-null: set to 1 when refinement should stop
    int* __restrict__ scan_range)    // if non-null: [fwd_start_site, fwd_start_log, bwd_end_site, bwd_end_log]
{
    // If refinement was already stopped by a previous round, skip entirely
    if (refine_skip && *refine_skip) return;

    int has_unknown = 0, has_new_fixed = 0;
    int earliest_log = -1, latest_log = -1;
    int earliest_site = 999999, latest_site = -1;

    for (int k = 0; k < K; k++) {
        if (fixed[k]) continue;
        if (decisions[k] != -1) {
            fixed[k] = 1;
            fixed_values[k] = decisions[k];
            has_new_fixed = 1;
            int pos = logical_positions[k];
            if (pos < earliest_site) { earliest_site = pos; earliest_log = k; }
            if (pos > latest_site) { latest_site = pos; latest_log = k; }
        } else {
            has_unknown = 1;
        }
    }

    control[0] = has_unknown;
    control[1] = has_new_fixed;
    control[2] = earliest_log;
    control[3] = latest_log;
    control[4] = earliest_site;
    control[5] = latest_site;

    // Set skip flag: stop if no unknowns or no new fixes
    if (refine_skip && (!has_unknown || !has_new_fixed)) {
        *refine_skip = 1;
    }

    // Set scan range for next round's partial scan
    if (scan_range) {
        scan_range[0] = earliest_site;   // fwd_start_site
        scan_range[1] = earliest_log;    // fwd_start_log
        scan_range[2] = latest_site;     // bwd_end_site
        scan_range[3] = latest_log;      // bwd_end_log
    }
}

// ---- Kernel: Final forced check ----
// Runs after all refinement rounds. If unknowns remain, force-fix them
// and set final_skip=0 (meaning the final scan+marginals should run).
// If no unknowns, set final_skip=1 (skip final pass).

__global__ void final_force_check_kernel(
    const double* __restrict__ marginals,
    int* __restrict__ fixed,
    int* __restrict__ fixed_values,
    const int* __restrict__ logical_positions,
    int* __restrict__ final_skip,
    int* __restrict__ scan_range,
    int K, int N)
{
    int has_unknown = 0;
    int earliest_site = 999999, latest_site = -1;
    int earliest_log = -1, latest_log = -1;

    for (int k = 0; k < K; k++) {
        if (fixed[k]) continue;
        has_unknown = 1;
        // Force-fix: decide based on marginal
        fixed[k] = 1;
        fixed_values[k] = (marginals[k] >= 0.5) ? 1 : 0;
        int pos = logical_positions[k];
        if (pos < earliest_site) { earliest_site = pos; earliest_log = k; }
        if (pos > latest_site) { latest_site = pos; latest_log = k; }
    }

    if (!has_unknown) {
        *final_skip = 1;
    } else {
        *final_skip = 0;
        if (scan_range) {
            scan_range[0] = earliest_site;
            scan_range[1] = earliest_log;
            scan_range[2] = latest_site;
            scan_range[3] = latest_log;
        }
    }
}

// ---- Kernel 2: Forward scan (prefix products) ----
//
// Computes running product left-to-right: acc = acc @ base[site]
// Since chi_l[0] = 1, acc is always a row vector.
// Saves prefix at each logical site position (BEFORE multiplying by that site).
//
// Shared memory layout (using blockDim.x as spacing):
//   smem[0 .. BD-1]     = acc (running accumulator)
//   smem[BD .. 2*BD-1]  = temp (intermediate result)
//   smem[2*BD .. 3*BD-1] = reduce buffer for max-abs normalization
// where BD = blockDim.x

// Forward scan with multi-group inner loop for latency hiding.
// Block size can be larger than max_chi: extra threads participate in parallel
// partial-sum groups, increasing warp count to hide L2 cache latency.
// With 1024 threads (32 warps) vs 256 (8 warps), latency hiding improves ~4x.
__global__ void forward_scan_kernel(
    const double* __restrict__ base_matrices,
    const int* __restrict__ base_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ site_type,
    const int* __restrict__ log_index,
    double* __restrict__ prefix_vecs,        // K * max_chi
    double* __restrict__ prefix_log_scales,  // K
    int N, int K, int max_chi,
    int start_site,          // 0 for full scan, >0 for partial re-scan
    int start_logical_idx,   // -1 for full scan (init acc=[1,0,...]), >=0 loads saved prefix
    const int* __restrict__ skip_flag,       // if non-null and *skip_flag != 0, skip
    const int* __restrict__ scan_range)      // if non-null, override start_site/start_logical_idx
{
    // Device-side skip check (all threads check same flag, no divergence)
    if (skip_flag) {
        __shared__ int s_skip;
        if (threadIdx.x == 0) s_skip = *skip_flag;
        __syncthreads();
        if (s_skip) return;
    }
    // Override scan range from device memory if provided
    if (scan_range) {
        start_site = scan_range[0];
        start_logical_idx = scan_range[1];
    }

    extern __shared__ double smem[];
    int BD = blockDim.x;
    // Shared memory layout: acc[max_chi] + partial[BD] + warp_max[32]
    double* acc      = smem;              // max_chi doubles
    double* partial  = smem + max_chi;    // BD doubles (partial sums from all thread groups)
    double* warp_max = smem + max_chi + BD; // 32 doubles (for warp-level reduction)

    int tid = threadIdx.x;
    int num_groups = BD / max_chi;  // how many parallel groups (1 if BD == max_chi)
    int tid_b = tid % max_chi;     // output column index
    int tid_g = tid / max_chi;     // group index

    double log_scale;
    int acc_dim;

    if (start_logical_idx >= 0) {
        if (tid < max_chi)
            acc[tid] = prefix_vecs[start_logical_idx * max_chi + tid];
        log_scale = prefix_log_scales[start_logical_idx];
        acc_dim = chi_l[start_site];
    } else {
        if (tid < max_chi)
            acc[tid] = (tid == 0) ? 1.0 : 0.0;
        log_scale = 0.0;
        acc_dim = 1;
    }
    __syncthreads();

    for (int site = start_site; site < N; site++) {
        int cl = chi_l[site];
        int cr = chi_r[site];

        // If logical site, save prefix BEFORE processing
        if (site_type[site] == 1) {
            int li = log_index[site];
            if (tid < max_chi) {
                prefix_vecs[li * max_chi + tid] = (tid < acc_dim) ? acc[tid] : 0.0;
            }
            if (tid == 0) prefix_log_scales[li] = log_scale;
        }
        __syncthreads();

        // Multi-group vec-mat multiply: partition the inner sum across groups
        // Each group handles a chunk of the 'a' dimension
        int offset = base_offsets[site];
        double val = 0.0;
        if (tid_b < cr && num_groups > 1) {
            int chunk = (cl + num_groups - 1) / num_groups;
            int a_start = tid_g * chunk;
            int a_end = a_start + chunk;
            if (a_end > cl) a_end = cl;
            for (int a = a_start; a < a_end; a++) {
                val += acc[a] * __ldg(&base_matrices[offset + a * cr + tid_b]);
            }
        } else if (tid_b < cr) {
            // Single group fallback (BD == max_chi)
            for (int a = 0; a < cl; a++) {
                val += acc[a] * __ldg(&base_matrices[offset + a * cr + tid_b]);
            }
        }

        // Store partial sums: partial[tid_g * max_chi + tid_b]
        partial[tid] = val;
        __syncthreads();

        // Reduce partial sums across groups (first max_chi threads)
        double sum;
        if (tid < max_chi && num_groups > 1) {
            sum = partial[tid];  // group 0
            for (int g = 1; g < num_groups; g++) {
                sum += partial[g * max_chi + tid];
            }
        } else if (tid < max_chi) {
            sum = partial[tid];
        } else {
            sum = 0.0;
        }

        // Warp-level max-abs reduction (replaces shared-memory tree reduction)
        // First: each thread has its contribution to max-abs
        double my_abs = (tid < cr) ? fabs(sum) : 0.0;
        // Intra-warp reduction via shuffle
        for (int s = 16; s > 0; s >>= 1) {
            my_abs = fmax(my_abs, __shfl_down_sync(0xFFFFFFFF, my_abs, s));
        }
        // Lane 0 of each warp writes to warp_max
        if (tid % 32 == 0) warp_max[tid / 32] = my_abs;
        __syncthreads();
        // Final reduction across warps (warp 0)
        if (tid < 32) {
            my_abs = (tid < (BD + 31) / 32) ? warp_max[tid] : 0.0;
            for (int s = 16; s > 0; s >>= 1) {
                my_abs = fmax(my_abs, __shfl_down_sync(0xFFFFFFFF, my_abs, s));
            }
            if (tid == 0) warp_max[0] = my_abs;
        }
        __syncthreads();
        double norm = warp_max[0];

        if (tid < max_chi) {
            if (norm > 0.0) {
                acc[tid] = (tid < cr) ? sum / norm : 0.0;
            } else {
                acc[tid] = (tid < cr) ? sum : 0.0;
            }
        }
        if (tid == 0 && norm > 0.0) log_scale += log(norm);
        __syncthreads();
        acc_dim = cr;
    }
}

// Batch forward scan: one block per batch element
__global__ void forward_scan_batch_kernel(
    const double* __restrict__ base_matrices,
    const int* __restrict__ base_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ site_type,
    const int* __restrict__ log_index,
    double* __restrict__ prefix_vecs,
    double* __restrict__ prefix_log_scales,
    int N, int K, int max_chi, int total_base_elements)
{
    int batch = blockIdx.x;
    extern __shared__ double smem[];
    int BD = blockDim.x;
    double* acc    = smem;
    double* temp   = smem + BD;
    double* reduce = smem + 2 * BD;
    int tid = threadIdx.x;

    acc[tid] = (tid == 0) ? 1.0 : 0.0;
    __syncthreads();

    double log_scale = 0.0;
    int acc_dim = 1;
    long long bbo = (long long)batch * total_base_elements;
    long long bpo = (long long)batch * K * max_chi;
    int blo = batch * K;

    for (int site = 0; site < N; site++) {
        int cl = chi_l[site];
        int cr = chi_r[site];

        if (site_type[site] == 1) {
            int li = log_index[site];
            if (tid < max_chi)
                prefix_vecs[bpo + li * max_chi + tid] = (tid < acc_dim) ? acc[tid] : 0.0;
            if (tid == 0) prefix_log_scales[blo + li] = log_scale;
        }
        __syncthreads();

        long long offset = bbo + base_offsets[site];
        double val = 0.0;
        if (tid < cr) {
            for (int a = 0; a < cl; a++)
                val += acc[a] * __ldg(&base_matrices[offset + a * cr + tid]);
        }
        __syncthreads();
        temp[tid] = (tid < cr) ? val : 0.0;
        __syncthreads();

        reduce[tid] = (tid < cr) ? fabs(temp[tid]) : 0.0;
        __syncthreads();
        for (int s = BD / 2; s > 0; s >>= 1) {
            if (tid < s) reduce[tid] = fmax(reduce[tid], reduce[tid + s]);
            __syncthreads();
        }
        double norm = reduce[0];
        if (norm > 0.0) {
            acc[tid] = (tid < cr) ? temp[tid] / norm : 0.0;
            if (tid == 0) log_scale += log(norm);
        } else {
            acc[tid] = temp[tid];
        }
        __syncthreads();
        acc_dim = cr;
    }
}

// ---- Kernel 3: Backward scan (suffix products) ----
//
// Computes running product right-to-left: acc = base[site] @ acc
// Since chi_r[N-1] = 1, acc is always a column vector.
// Saves suffix at each logical site position (BEFORE multiplying by that site from the right).

// Backward scan with multi-group inner loop for latency hiding (mirrors forward_scan_kernel).
__global__ void backward_scan_kernel(
    const double* __restrict__ base_matrices_T,  // transposed base matrices for coalesced reads
    const int* __restrict__ base_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ site_type,
    const int* __restrict__ log_index,
    double* __restrict__ suffix_vecs,        // K * max_chi
    double* __restrict__ suffix_log_scales,  // K
    int N, int K, int max_chi,
    int end_site,            // N-1 for full scan, <N-1 for partial re-scan
    int end_logical_idx,     // -1 for full scan (init acc=[1,0,...]), >=0 loads saved suffix
    const int* __restrict__ skip_flag,
    const int* __restrict__ scan_range)
{
    if (skip_flag) {
        __shared__ int s_skip;
        if (threadIdx.x == 0) s_skip = *skip_flag;
        __syncthreads();
        if (s_skip) return;
    }
    if (scan_range) {
        end_site = scan_range[2];
        end_logical_idx = scan_range[3];
    }

    extern __shared__ double smem[];
    int BD = blockDim.x;
    double* acc      = smem;
    double* partial  = smem + max_chi;
    double* warp_max = smem + max_chi + BD;
    int tid = threadIdx.x;
    int num_groups = BD / max_chi;
    int tid_b = tid % max_chi;  // output row index (a)
    int tid_g = tid / max_chi;  // group index

    double log_scale;
    int acc_dim;

    if (end_logical_idx >= 0) {
        if (tid < max_chi)
            acc[tid] = suffix_vecs[end_logical_idx * max_chi + tid];
        log_scale = suffix_log_scales[end_logical_idx];
        acc_dim = chi_r[end_site];
    } else {
        if (tid < max_chi)
            acc[tid] = (tid == 0) ? 1.0 : 0.0;
        log_scale = 0.0;
        acc_dim = 1;
    }
    __syncthreads();

    for (int site = end_site; site >= 0; site--) {
        int cl = chi_l[site];
        int cr = chi_r[site];

        // If logical site, save suffix BEFORE processing (from right)
        if (site_type[site] == 1) {
            int li = log_index[site];
            if (tid < max_chi) {
                suffix_vecs[li * max_chi + tid] = (tid < acc_dim) ? acc[tid] : 0.0;
            }
            if (tid == 0) suffix_log_scales[li] = log_scale;
        }
        __syncthreads();

        // Multi-group mat-vec multiply: temp[a] = sum_b M_T[b*cl + a] * acc[b]
        // Using transposed matrix for coalesced reads: threads with adjacent tid_b read adjacent M_T addresses
        int offset = base_offsets[site];
        double val = 0.0;
        if (tid_b < cl && num_groups > 1) {
            int chunk = (cr + num_groups - 1) / num_groups;
            int b_start = tid_g * chunk;
            int b_end = b_start + chunk;
            if (b_end > cr) b_end = cr;
            for (int b = b_start; b < b_end; b++) {
                val += __ldg(&base_matrices_T[offset + b * cl + tid_b]) * acc[b];
            }
        } else if (tid_b < cl) {
            for (int b = 0; b < cr; b++) {
                val += __ldg(&base_matrices_T[offset + b * cl + tid_b]) * acc[b];
            }
        }

        partial[tid] = val;
        __syncthreads();

        // Reduce partial sums across groups
        double sum;
        if (tid < max_chi && num_groups > 1) {
            sum = partial[tid];
            for (int g = 1; g < num_groups; g++) {
                sum += partial[g * max_chi + tid];
            }
        } else if (tid < max_chi) {
            sum = partial[tid];
        } else {
            sum = 0.0;
        }

        // Warp-level max-abs reduction
        double my_abs = (tid < cl) ? fabs(sum) : 0.0;
        for (int s = 16; s > 0; s >>= 1) {
            my_abs = fmax(my_abs, __shfl_down_sync(0xFFFFFFFF, my_abs, s));
        }
        if (tid % 32 == 0) warp_max[tid / 32] = my_abs;
        __syncthreads();
        if (tid < 32) {
            my_abs = (tid < (BD + 31) / 32) ? warp_max[tid] : 0.0;
            for (int s = 16; s > 0; s >>= 1) {
                my_abs = fmax(my_abs, __shfl_down_sync(0xFFFFFFFF, my_abs, s));
            }
            if (tid == 0) warp_max[0] = my_abs;
        }
        __syncthreads();
        double norm = warp_max[0];

        if (tid < max_chi) {
            if (norm > 0.0) {
                acc[tid] = (tid < cl) ? sum / norm : 0.0;
            } else {
                acc[tid] = (tid < cl) ? sum : 0.0;
            }
        }
        if (tid == 0 && norm > 0.0) log_scale += log(norm);
        __syncthreads();
        acc_dim = cl;
    }
}

// Batch backward scan (using transposed base matrices for coalesced reads)
__global__ void backward_scan_batch_kernel(
    const double* __restrict__ base_matrices_T,  // transposed
    const int* __restrict__ base_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ site_type,
    const int* __restrict__ log_index,
    double* __restrict__ suffix_vecs,
    double* __restrict__ suffix_log_scales,
    int N, int K, int max_chi, int total_base_elements)
{
    int batch = blockIdx.x;
    extern __shared__ double smem[];
    int BD = blockDim.x;
    double* acc    = smem;
    double* temp   = smem + BD;
    double* reduce = smem + 2 * BD;
    int tid = threadIdx.x;

    acc[tid] = (tid == 0) ? 1.0 : 0.0;
    __syncthreads();

    double log_scale = 0.0;
    int acc_dim = 1;
    long long bbo = (long long)batch * total_base_elements;
    long long bso = (long long)batch * K * max_chi;
    int blo = batch * K;

    for (int site = N - 1; site >= 0; site--) {
        int cl = chi_l[site];
        int cr = chi_r[site];

        if (site_type[site] == 1) {
            int li = log_index[site];
            if (tid < max_chi)
                suffix_vecs[bso + li * max_chi + tid] = (tid < acc_dim) ? acc[tid] : 0.0;
            if (tid == 0) suffix_log_scales[blo + li] = log_scale;
        }
        __syncthreads();

        // Using transposed: M_T[b*cl + a], thread tid reads M_T[b*cl + tid] — coalesced
        long long offset = bbo + base_offsets[site];
        double val = 0.0;
        if (tid < cl) {
            for (int b = 0; b < cr; b++)
                val += __ldg(&base_matrices_T[offset + b * cl + tid]) * acc[b];
        }
        __syncthreads();
        temp[tid] = (tid < cl) ? val : 0.0;
        __syncthreads();

        reduce[tid] = (tid < cl) ? fabs(temp[tid]) : 0.0;
        __syncthreads();
        for (int s = BD / 2; s > 0; s >>= 1) {
            if (tid < s) reduce[tid] = fmax(reduce[tid], reduce[tid + s]);
            __syncthreads();
        }
        double norm = reduce[0];
        if (norm > 0.0) {
            acc[tid] = (tid < cl) ? temp[tid] / norm : 0.0;
            if (tid == 0) log_scale += log(norm);
        } else {
            acc[tid] = temp[tid];
        }
        __syncthreads();
        acc_dim = cl;
    }
}

// ---- Kernel 4: Compute marginals ----
//
// For each logical k at chain position pos_k:
//   result_lambda = prefix[k] @ T[pos_k][:, lambda, :] @ suffix[k]
//   P_k = |result_1| / (|result_0| + |result_1|)
//
// Shared memory: mid[BD] + reduce[BD]

__global__ void compute_marginals_kernel(
    const double* __restrict__ raw_tensors,
    const int* __restrict__ data_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ logical_positions,
    const double* __restrict__ prefix_vecs,
    const double* __restrict__ prefix_log_scales,
    const double* __restrict__ suffix_vecs,
    const double* __restrict__ suffix_log_scales,
    const int* __restrict__ fixed,
    double* __restrict__ marginals,
    int* __restrict__ decisions,
    int K, int max_chi,
    double low_thresh, double high_thresh,
    const int* __restrict__ skip_flag)
{
    if (skip_flag && *skip_flag) return;
    int k = blockIdx.x;
    if (k >= K) return;
    if (fixed[k]) return;

    int pos = logical_positions[k];
    int cl = chi_l[pos];
    int cr = chi_r[pos];
    int tid = threadIdx.x;
    int BD = blockDim.x;

    extern __shared__ double smem[];
    double* mid    = smem;
    double* reduce = smem + BD;

    int offset_raw = data_offsets[pos];

    double log_result[2];

    for (int lam = 0; lam < 2; lam++) {
        // Step 1: mid[b] = sum_a prefix[k][a] * T[pos][a, lam, b]
        double val = 0.0;
        if (tid < cr) {
            for (int a = 0; a < cl; a++) {
                val += prefix_vecs[k * max_chi + a] *
                       __ldg(&raw_tensors[offset_raw + a * 2 * cr + lam * cr + tid]);
            }
        }
        mid[tid] = (tid < cr) ? val : 0.0;
        __syncthreads();

        // Step 2: dot product = sum_b mid[b] * suffix[k][b]
        double prod = 0.0;
        if (tid < cr) {
            prod = mid[tid] * suffix_vecs[k * max_chi + tid];
        }

        // Parallel reduction for sum
        reduce[tid] = prod;
        __syncthreads();
        for (int s = BD / 2; s > 0; s >>= 1) {
            if (tid < s) reduce[tid] += reduce[tid + s];
            __syncthreads();
        }

        if (tid == 0) {
            double result = reduce[0];
            double abs_r = fabs(result);
            log_result[lam] = (abs_r > 0.0) ?
                log(abs_r) + prefix_log_scales[k] + suffix_log_scales[k] : -1e300;
        }
        __syncthreads();
    }

    if (tid == 0) {
        double log0 = log_result[0];
        double log1 = log_result[1];

        double prob;
        if (log0 <= -1e299 && log1 <= -1e299) {
            prob = 0.5;
        } else if (log1 > log0) {
            prob = 1.0 / (1.0 + exp(log0 - log1));
        } else {
            prob = exp(log1 - log0) / (1.0 + exp(log1 - log0));
        }

        marginals[k] = prob;
        if (prob > high_thresh)       decisions[k] = 1;
        else if (prob < low_thresh)   decisions[k] = 0;
        else                          decisions[k] = -1;
    }
}

// Batch marginals: grid (K, batch_size)
__global__ void compute_marginals_batch_kernel(
    const double* __restrict__ raw_tensors,
    const int* __restrict__ data_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ logical_positions,
    const double* __restrict__ prefix_vecs,
    const double* __restrict__ prefix_log_scales,
    const double* __restrict__ suffix_vecs,
    const double* __restrict__ suffix_log_scales,
    const int* __restrict__ fixed,
    double* __restrict__ marginals,
    int* __restrict__ decisions,
    int K, int max_chi,
    double low_thresh, double high_thresh)
{
    int k = blockIdx.x;
    int batch = blockIdx.y;
    if (k >= K) return;
    if (fixed[batch * K + k]) return;

    int pos = logical_positions[k];
    int cl = chi_l[pos];
    int cr = chi_r[pos];
    int tid = threadIdx.x;
    int BD = blockDim.x;

    extern __shared__ double smem[];
    double* mid    = smem;
    double* reduce = smem + BD;

    int offset_raw = data_offsets[pos];
    long long bpo = (long long)batch * K * max_chi;
    int blo = batch * K;

    double log_result[2];

    for (int lam = 0; lam < 2; lam++) {
        double val = 0.0;
        if (tid < cr) {
            for (int a = 0; a < cl; a++) {
                val += prefix_vecs[bpo + k * max_chi + a] *
                       __ldg(&raw_tensors[offset_raw + a * 2 * cr + lam * cr + tid]);
            }
        }
        mid[tid] = (tid < cr) ? val : 0.0;
        __syncthreads();

        double prod = (tid < cr) ? mid[tid] * suffix_vecs[bpo + k * max_chi + tid] : 0.0;
        reduce[tid] = prod;
        __syncthreads();
        for (int s = BD / 2; s > 0; s >>= 1) {
            if (tid < s) reduce[tid] += reduce[tid + s];
            __syncthreads();
        }

        if (tid == 0) {
            double result = reduce[0];
            double abs_r = fabs(result);
            log_result[lam] = (abs_r > 0.0) ?
                log(abs_r) + prefix_log_scales[blo + k] + suffix_log_scales[blo + k]
                : -1e300;
        }
        __syncthreads();
    }

    if (tid == 0) {
        double log0 = log_result[0];
        double log1 = log_result[1];
        double prob;
        if (log0 <= -1e299 && log1 <= -1e299)  prob = 0.5;
        else if (log1 > log0)  prob = 1.0 / (1.0 + exp(log0 - log1));
        else  prob = exp(log1 - log0) / (1.0 + exp(log1 - log0));

        marginals[batch * K + k] = prob;
        if (prob > high_thresh)       decisions[batch * K + k] = 1;
        else if (prob < low_thresh)   decisions[batch * K + k] = 0;
        else                          decisions[batch * K + k] = -1;
    }
}

// ---- Kernel 5: Update base matrices for newly-fixed logicals ----

__global__ void update_base_for_fixed_kernel(
    const double* __restrict__ raw_tensors,
    const int* __restrict__ data_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ logical_positions,
    const int* __restrict__ fixed,
    const int* __restrict__ fixed_values,
    double* __restrict__ base_matrices,
    double* __restrict__ base_matrices_T,
    const int* __restrict__ base_offsets,
    int K,
    const int* __restrict__ skip_flag)
{
    if (skip_flag && *skip_flag) return;
    int k = blockIdx.x;
    if (k >= K || !fixed[k]) return;

    int pos = logical_positions[k];
    int cl = chi_l[pos];
    int cr = chi_r[pos];
    int offset_raw = data_offsets[pos];
    int offset_base = base_offsets[pos];
    int mat_size = cl * cr;
    int v = fixed_values[k];

    for (int idx = threadIdx.x; idx < mat_size; idx += blockDim.x) {
        int a = idx / cr;
        int b = idx % cr;
        double val = __ldg(&raw_tensors[offset_raw + a * 2 * cr + v * cr + b]);
        base_matrices[offset_base + a * cr + b] = val;
        base_matrices_T[offset_base + b * cl + a] = val;
    }
}

// Batch version
__global__ void update_base_for_fixed_batch_kernel(
    const double* __restrict__ raw_tensors,
    const int* __restrict__ data_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ logical_positions,
    const int* __restrict__ fixed,
    const int* __restrict__ fixed_values,
    double* __restrict__ base_matrices,
    double* __restrict__ base_matrices_T,
    const int* __restrict__ base_offsets,
    int K, int total_base_elements)
{
    int k = blockIdx.x;
    int batch = blockIdx.y;
    if (k >= K || !fixed[batch * K + k]) return;

    int pos = logical_positions[k];
    int cl = chi_l[pos];
    int cr = chi_r[pos];
    int offset_raw = data_offsets[pos];
    long long offset_base = (long long)batch * total_base_elements + base_offsets[pos];
    int mat_size = cl * cr;
    int v = fixed_values[batch * K + k];

    for (int idx = threadIdx.x; idx < mat_size; idx += blockDim.x) {
        int a = idx / cr;
        int b = idx % cr;
        double val = __ldg(&raw_tensors[offset_raw + a * 2 * cr + v * cr + b]);
        base_matrices[offset_base + a * cr + b] = val;
        base_matrices_T[offset_base + b * cl + a] = val;
    }
}

// ============================================================================
// FP32 Mixed-Precision Kernels
// ============================================================================

// ---- FP32 Kernel: Prepare base matrices (FP64 raw → FP32 base) ----
__global__ void prepare_base_matrices_fp32_kernel(
    const double* __restrict__ raw_tensors,
    const int* __restrict__ data_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ site_type,
    const int* __restrict__ syn_index,
    const int* __restrict__ log_index,
    const double* __restrict__ syndrome,
    const int* __restrict__ fixed,
    const int* __restrict__ fixed_values,
    float* __restrict__ base_matrices_f32,
    float* __restrict__ base_matrices_T_f32,
    const int* __restrict__ base_offsets,
    int N)
{
    int site = blockIdx.x;
    if (site >= N) return;

    int cl = chi_l[site];
    int cr = chi_r[site];
    int offset_raw = data_offsets[site];
    int offset_base = base_offsets[site];
    int mat_size = cl * cr;

    if (site_type[site] == 0) {
        int si = syn_index[site];
        int s = (int)(syndrome[si] + 0.5);
        for (int idx = threadIdx.x; idx < mat_size; idx += blockDim.x) {
            int a = idx / cr;
            int b = idx % cr;
            float val = (float)__ldg(&raw_tensors[offset_raw + a * 2 * cr + s * cr + b]);
            base_matrices_f32[offset_base + a * cr + b] = val;
            base_matrices_T_f32[offset_base + b * cl + a] = val;
        }
    } else {
        int li = log_index[site];
        if (fixed[li]) {
            int v = fixed_values[li];
            for (int idx = threadIdx.x; idx < mat_size; idx += blockDim.x) {
                int a = idx / cr;
                int b = idx % cr;
                float val = (float)__ldg(&raw_tensors[offset_raw + a * 2 * cr + v * cr + b]);
                base_matrices_f32[offset_base + a * cr + b] = val;
                base_matrices_T_f32[offset_base + b * cl + a] = val;
            }
        } else {
            for (int idx = threadIdx.x; idx < mat_size; idx += blockDim.x) {
                int a = idx / cr;
                int b = idx % cr;
                double v0 = __ldg(&raw_tensors[offset_raw + a * 2 * cr + 0 * cr + b]);
                double v1 = __ldg(&raw_tensors[offset_raw + a * 2 * cr + 1 * cr + b]);
                float val = (float)(v0 + v1);
                base_matrices_f32[offset_base + a * cr + b] = val;
                base_matrices_T_f32[offset_base + b * cl + a] = val;
            }
        }
    }
}

// ---- FP32 Kernel: Batch prepare base matrices ----
__global__ void prepare_base_matrices_fp32_batch_kernel(
    const double* __restrict__ raw_tensors,
    const int* __restrict__ data_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ site_type,
    const int* __restrict__ syn_index,
    const int* __restrict__ log_index,
    const double* __restrict__ syndrome,
    const int* __restrict__ fixed,
    const int* __restrict__ fixed_values,
    float* __restrict__ base_matrices_f32,
    float* __restrict__ base_matrices_T_f32,
    const int* __restrict__ base_offsets,
    int N, int M, int K, int total_base_elements)
{
    int site = blockIdx.x;
    int batch = blockIdx.y;
    if (site >= N) return;

    int cl = chi_l[site];
    int cr = chi_r[site];
    int offset_raw = data_offsets[site];
    long long offset_base = (long long)batch * total_base_elements + base_offsets[site];
    int mat_size = cl * cr;

    if (site_type[site] == 0) {
        int si = syn_index[site];
        int s = (int)(syndrome[batch * M + si] + 0.5);
        for (int idx = threadIdx.x; idx < mat_size; idx += blockDim.x) {
            int a = idx / cr;
            int b = idx % cr;
            float val = (float)__ldg(&raw_tensors[offset_raw + a * 2 * cr + s * cr + b]);
            base_matrices_f32[offset_base + a * cr + b] = val;
            base_matrices_T_f32[offset_base + b * cl + a] = val;
        }
    } else {
        int li = log_index[site];
        if (fixed[batch * K + li]) {
            int v = fixed_values[batch * K + li];
            for (int idx = threadIdx.x; idx < mat_size; idx += blockDim.x) {
                int a = idx / cr;
                int b = idx % cr;
                float val = (float)__ldg(&raw_tensors[offset_raw + a * 2 * cr + v * cr + b]);
                base_matrices_f32[offset_base + a * cr + b] = val;
                base_matrices_T_f32[offset_base + b * cl + a] = val;
            }
        } else {
            for (int idx = threadIdx.x; idx < mat_size; idx += blockDim.x) {
                int a = idx / cr;
                int b = idx % cr;
                double v0 = __ldg(&raw_tensors[offset_raw + a * 2 * cr + 0 * cr + b]);
                double v1 = __ldg(&raw_tensors[offset_raw + a * 2 * cr + 1 * cr + b]);
                float val = (float)(v0 + v1);
                base_matrices_f32[offset_base + a * cr + b] = val;
                base_matrices_T_f32[offset_base + b * cl + a] = val;
            }
        }
    }
}

// ---- FP32 Forward scan kernel ----
// Same algorithm as FP64 forward_scan_kernel but all working data is float.
// Halves L2 bandwidth per step → ~2x speedup on memory-bound scan.
__global__ void forward_scan_fp32_kernel(
    const float* __restrict__ base_matrices_f32,
    const int* __restrict__ base_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ site_type,
    const int* __restrict__ log_index,
    float* __restrict__ prefix_vecs_f32,
    float* __restrict__ prefix_log_f32,
    int N, int K, int max_chi,
    int start_site, int start_logical_idx,
    const int* __restrict__ skip_flag,
    const int* __restrict__ scan_range)
{
    if (skip_flag) {
        __shared__ int s_skip;
        if (threadIdx.x == 0) s_skip = *skip_flag;
        __syncthreads();
        if (s_skip) return;
    }
    if (scan_range) {
        start_site = scan_range[0];
        start_logical_idx = scan_range[1];
    }

    extern __shared__ float smem_f[];
    int BD = blockDim.x;
    float* acc      = smem_f;
    float* partial  = smem_f + max_chi;
    float* warp_max = smem_f + max_chi + BD;

    int tid = threadIdx.x;
    int num_groups = BD / max_chi;
    int tid_b = tid % max_chi;
    int tid_g = tid / max_chi;

    float log_scale;
    int acc_dim;

    if (start_logical_idx >= 0) {
        if (tid < max_chi)
            acc[tid] = prefix_vecs_f32[start_logical_idx * max_chi + tid];
        log_scale = prefix_log_f32[start_logical_idx];
        acc_dim = chi_l[start_site];
    } else {
        if (tid < max_chi)
            acc[tid] = (tid == 0) ? 1.0f : 0.0f;
        log_scale = 0.0f;
        acc_dim = 1;
    }
    __syncthreads();

    for (int site = start_site; site < N; site++) {
        int cl = chi_l[site];
        int cr = chi_r[site];

        if (site_type[site] == 1) {
            int li = log_index[site];
            if (tid < max_chi) {
                prefix_vecs_f32[li * max_chi + tid] = (tid < acc_dim) ? acc[tid] : 0.0f;
            }
            if (tid == 0) prefix_log_f32[li] = log_scale;
        }
        __syncthreads();

        int offset = base_offsets[site];
        float val = 0.0f;
        if (tid_b < cr && num_groups > 1) {
            int chunk = (cl + num_groups - 1) / num_groups;
            int a_start = tid_g * chunk;
            int a_end = a_start + chunk;
            if (a_end > cl) a_end = cl;
            for (int a = a_start; a < a_end; a++) {
                val += acc[a] * __ldg(&base_matrices_f32[offset + a * cr + tid_b]);
            }
        } else if (tid_b < cr) {
            for (int a = 0; a < cl; a++) {
                val += acc[a] * __ldg(&base_matrices_f32[offset + a * cr + tid_b]);
            }
        }

        partial[tid] = val;
        __syncthreads();

        float sum;
        if (tid < max_chi && num_groups > 1) {
            sum = partial[tid];
            for (int g = 1; g < num_groups; g++) {
                sum += partial[g * max_chi + tid];
            }
        } else if (tid < max_chi) {
            sum = partial[tid];
        } else {
            sum = 0.0f;
        }

        // Warp-level max-abs reduction (float)
        float my_abs = (tid < cr) ? fabsf(sum) : 0.0f;
        for (int s = 16; s > 0; s >>= 1) {
            my_abs = fmaxf(my_abs, __shfl_down_sync(0xFFFFFFFF, my_abs, s));
        }
        if (tid % 32 == 0) warp_max[tid / 32] = my_abs;
        __syncthreads();
        if (tid < 32) {
            my_abs = (tid < (BD + 31) / 32) ? warp_max[tid] : 0.0f;
            for (int s = 16; s > 0; s >>= 1) {
                my_abs = fmaxf(my_abs, __shfl_down_sync(0xFFFFFFFF, my_abs, s));
            }
            if (tid == 0) warp_max[0] = my_abs;
        }
        __syncthreads();
        float norm = warp_max[0];

        if (tid < max_chi) {
            if (norm > 0.0f) {
                acc[tid] = (tid < cr) ? sum / norm : 0.0f;
            } else {
                acc[tid] = (tid < cr) ? sum : 0.0f;
            }
        }
        if (tid == 0 && norm > 0.0f) log_scale += logf(norm);
        __syncthreads();
        acc_dim = cr;
    }
}

// ---- FP32 Backward scan kernel ----
__global__ void backward_scan_fp32_kernel(
    const float* __restrict__ base_matrices_T_f32,
    const int* __restrict__ base_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ site_type,
    const int* __restrict__ log_index,
    float* __restrict__ suffix_vecs_f32,
    float* __restrict__ suffix_log_f32,
    int N, int K, int max_chi,
    int end_site, int end_logical_idx,
    const int* __restrict__ skip_flag,
    const int* __restrict__ scan_range)
{
    if (skip_flag) {
        __shared__ int s_skip;
        if (threadIdx.x == 0) s_skip = *skip_flag;
        __syncthreads();
        if (s_skip) return;
    }
    if (scan_range) {
        end_site = scan_range[2];
        end_logical_idx = scan_range[3];
    }

    extern __shared__ float smem_f[];
    int BD = blockDim.x;
    float* acc      = smem_f;
    float* partial  = smem_f + max_chi;
    float* warp_max = smem_f + max_chi + BD;
    int tid = threadIdx.x;
    int num_groups = BD / max_chi;
    int tid_b = tid % max_chi;
    int tid_g = tid / max_chi;

    float log_scale;
    int acc_dim;

    if (end_logical_idx >= 0) {
        if (tid < max_chi)
            acc[tid] = suffix_vecs_f32[end_logical_idx * max_chi + tid];
        log_scale = suffix_log_f32[end_logical_idx];
        acc_dim = chi_r[end_site];
    } else {
        if (tid < max_chi)
            acc[tid] = (tid == 0) ? 1.0f : 0.0f;
        log_scale = 0.0f;
        acc_dim = 1;
    }
    __syncthreads();

    for (int site = end_site; site >= 0; site--) {
        int cl = chi_l[site];
        int cr = chi_r[site];

        if (site_type[site] == 1) {
            int li = log_index[site];
            if (tid < max_chi) {
                suffix_vecs_f32[li * max_chi + tid] = (tid < acc_dim) ? acc[tid] : 0.0f;
            }
            if (tid == 0) suffix_log_f32[li] = log_scale;
        }
        __syncthreads();

        int offset = base_offsets[site];
        float val = 0.0f;
        if (tid_b < cl && num_groups > 1) {
            int chunk = (cr + num_groups - 1) / num_groups;
            int b_start = tid_g * chunk;
            int b_end = b_start + chunk;
            if (b_end > cr) b_end = cr;
            for (int b = b_start; b < b_end; b++) {
                val += __ldg(&base_matrices_T_f32[offset + b * cl + tid_b]) * acc[b];
            }
        } else if (tid_b < cl) {
            for (int b = 0; b < cr; b++) {
                val += __ldg(&base_matrices_T_f32[offset + b * cl + tid_b]) * acc[b];
            }
        }

        partial[tid] = val;
        __syncthreads();

        float sum;
        if (tid < max_chi && num_groups > 1) {
            sum = partial[tid];
            for (int g = 1; g < num_groups; g++) {
                sum += partial[g * max_chi + tid];
            }
        } else if (tid < max_chi) {
            sum = partial[tid];
        } else {
            sum = 0.0f;
        }

        float my_abs = (tid < cl) ? fabsf(sum) : 0.0f;
        for (int s = 16; s > 0; s >>= 1) {
            my_abs = fmaxf(my_abs, __shfl_down_sync(0xFFFFFFFF, my_abs, s));
        }
        if (tid % 32 == 0) warp_max[tid / 32] = my_abs;
        __syncthreads();
        if (tid < 32) {
            my_abs = (tid < (BD + 31) / 32) ? warp_max[tid] : 0.0f;
            for (int s = 16; s > 0; s >>= 1) {
                my_abs = fmaxf(my_abs, __shfl_down_sync(0xFFFFFFFF, my_abs, s));
            }
            if (tid == 0) warp_max[0] = my_abs;
        }
        __syncthreads();
        float norm = warp_max[0];

        if (tid < max_chi) {
            if (norm > 0.0f) {
                acc[tid] = (tid < cl) ? sum / norm : 0.0f;
            } else {
                acc[tid] = (tid < cl) ? sum : 0.0f;
            }
        }
        if (tid == 0 && norm > 0.0f) log_scale += logf(norm);
        __syncthreads();
        acc_dim = cl;
    }
}

// ---- FP32 Batch forward scan ----
__global__ void forward_scan_fp32_batch_kernel(
    const float* __restrict__ base_matrices_f32,
    const int* __restrict__ base_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ site_type,
    const int* __restrict__ log_index,
    float* __restrict__ prefix_vecs_f32,
    float* __restrict__ prefix_log_f32,
    int N, int K, int max_chi, int total_base_elements)
{
    int batch = blockIdx.x;
    extern __shared__ float smem_f[];
    int BD = blockDim.x;
    float* acc    = smem_f;
    float* temp   = smem_f + BD;
    float* reduce = smem_f + 2 * BD;
    int tid = threadIdx.x;

    acc[tid] = (tid == 0) ? 1.0f : 0.0f;
    __syncthreads();

    float log_scale = 0.0f;
    int acc_dim = 1;
    long long bbo = (long long)batch * total_base_elements;
    long long bpo = (long long)batch * K * max_chi;
    int blo = batch * K;

    for (int site = 0; site < N; site++) {
        int cl = chi_l[site];
        int cr = chi_r[site];

        if (site_type[site] == 1) {
            int li = log_index[site];
            if (tid < max_chi)
                prefix_vecs_f32[bpo + li * max_chi + tid] = (tid < acc_dim) ? acc[tid] : 0.0f;
            if (tid == 0) prefix_log_f32[blo + li] = log_scale;
        }
        __syncthreads();

        long long offset = bbo + base_offsets[site];
        float val = 0.0f;
        if (tid < cr) {
            for (int a = 0; a < cl; a++)
                val += acc[a] * __ldg(&base_matrices_f32[offset + a * cr + tid]);
        }
        __syncthreads();
        temp[tid] = (tid < cr) ? val : 0.0f;
        __syncthreads();

        reduce[tid] = (tid < cr) ? fabsf(temp[tid]) : 0.0f;
        __syncthreads();
        for (int s = BD / 2; s > 0; s >>= 1) {
            if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
            __syncthreads();
        }
        float norm = reduce[0];
        if (norm > 0.0f) {
            acc[tid] = (tid < cr) ? temp[tid] / norm : 0.0f;
            if (tid == 0) log_scale += logf(norm);
        } else {
            acc[tid] = temp[tid];
        }
        __syncthreads();
        acc_dim = cr;
    }
}

// ---- FP32 Batch backward scan ----
__global__ void backward_scan_fp32_batch_kernel(
    const float* __restrict__ base_matrices_T_f32,
    const int* __restrict__ base_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ site_type,
    const int* __restrict__ log_index,
    float* __restrict__ suffix_vecs_f32,
    float* __restrict__ suffix_log_f32,
    int N, int K, int max_chi, int total_base_elements)
{
    int batch = blockIdx.x;
    extern __shared__ float smem_f[];
    int BD = blockDim.x;
    float* acc    = smem_f;
    float* temp   = smem_f + BD;
    float* reduce = smem_f + 2 * BD;
    int tid = threadIdx.x;

    acc[tid] = (tid == 0) ? 1.0f : 0.0f;
    __syncthreads();

    float log_scale = 0.0f;
    int acc_dim = 1;
    long long bbo = (long long)batch * total_base_elements;
    long long bso = (long long)batch * K * max_chi;
    int blo = batch * K;

    for (int site = N - 1; site >= 0; site--) {
        int cl = chi_l[site];
        int cr = chi_r[site];

        if (site_type[site] == 1) {
            int li = log_index[site];
            if (tid < max_chi)
                suffix_vecs_f32[bso + li * max_chi + tid] = (tid < acc_dim) ? acc[tid] : 0.0f;
            if (tid == 0) suffix_log_f32[blo + li] = log_scale;
        }
        __syncthreads();

        long long offset = bbo + base_offsets[site];
        float val = 0.0f;
        if (tid < cl) {
            for (int b = 0; b < cr; b++)
                val += __ldg(&base_matrices_T_f32[offset + b * cl + tid]) * acc[b];
        }
        __syncthreads();
        temp[tid] = (tid < cl) ? val : 0.0f;
        __syncthreads();

        reduce[tid] = (tid < cl) ? fabsf(temp[tid]) : 0.0f;
        __syncthreads();
        for (int s = BD / 2; s > 0; s >>= 1) {
            if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
            __syncthreads();
        }
        float norm = reduce[0];
        if (norm > 0.0f) {
            acc[tid] = (tid < cl) ? temp[tid] / norm : 0.0f;
            if (tid == 0) log_scale += logf(norm);
        } else {
            acc[tid] = temp[tid];
        }
        __syncthreads();
        acc_dim = cl;
    }
}

// ---- FP32 → FP64 conversion kernel for prefix/suffix vectors ----
// Converts FP32 prefix/suffix vecs and log scales to FP64 buffers so the
// standard FP64 marginals kernel can be reused without mixed-precision overhead.
__global__ void convert_vecs_f32_to_f64_kernel(
    const float* __restrict__ src_vecs_f32,
    const float* __restrict__ src_log_f32,
    double* __restrict__ dst_vecs_f64,
    double* __restrict__ dst_log_f64,
    int K, int max_chi,
    const int* __restrict__ skip_flag)
{
    if (skip_flag && *skip_flag) return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < K * max_chi) {
        dst_vecs_f64[tid] = (double)src_vecs_f32[tid];
    }
    if (tid < K) {
        dst_log_f64[tid] = (double)src_log_f32[tid];
    }
}

// ---- Mixed-precision marginals: FP32 prefix/suffix, FP64 raw tensors ----
// Loads FP32 prefix/suffix and casts to FP64 for the final computation.
// Uses FP64 raw tensors directly for maximum precision in the probability ratio.
__global__ void compute_marginals_mixed_kernel(
    const double* __restrict__ raw_tensors,
    const int* __restrict__ data_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ logical_positions,
    const float* __restrict__ prefix_vecs_f32,
    const float* __restrict__ prefix_log_f32,
    const float* __restrict__ suffix_vecs_f32,
    const float* __restrict__ suffix_log_f32,
    const int* __restrict__ fixed,
    double* __restrict__ marginals,
    int* __restrict__ decisions,
    int K, int max_chi,
    double low_thresh, double high_thresh)
{
    int k = blockIdx.x;
    if (k >= K) return;
    if (fixed[k]) return;

    int pos = logical_positions[k];
    int cl = chi_l[pos];
    int cr = chi_r[pos];
    int tid = threadIdx.x;
    int BD = blockDim.x;

    extern __shared__ double smem[];
    double* mid    = smem;
    double* reduce = smem + BD;

    int offset_raw = data_offsets[pos];

    double log_result[2];

    for (int lam = 0; lam < 2; lam++) {
        // mid[b] = sum_a (double)prefix_f32[k][a] * T_fp64[pos][a, lam, b]
        double val = 0.0;
        if (tid < cr) {
            for (int a = 0; a < cl; a++) {
                val += (double)prefix_vecs_f32[k * max_chi + a] *
                       __ldg(&raw_tensors[offset_raw + a * 2 * cr + lam * cr + tid]);
            }
        }
        mid[tid] = (tid < cr) ? val : 0.0;
        __syncthreads();

        // dot = sum_b mid[b] * (double)suffix_f32[k][b]
        double prod = 0.0;
        if (tid < cr) {
            prod = mid[tid] * (double)suffix_vecs_f32[k * max_chi + tid];
        }

        reduce[tid] = prod;
        __syncthreads();
        for (int s = BD / 2; s > 0; s >>= 1) {
            if (tid < s) reduce[tid] += reduce[tid + s];
            __syncthreads();
        }

        if (tid == 0) {
            double result = reduce[0];
            double abs_r = fabs(result);
            // Use FP32 log scales cast to FP64
            log_result[lam] = (abs_r > 0.0) ?
                log(abs_r) + (double)prefix_log_f32[k] + (double)suffix_log_f32[k] : -1e300;
        }
        __syncthreads();
    }

    if (tid == 0) {
        double log0 = log_result[0];
        double log1 = log_result[1];

        double prob;
        if (log0 <= -1e299 && log1 <= -1e299) {
            prob = 0.5;
        } else if (log1 > log0) {
            prob = 1.0 / (1.0 + exp(log0 - log1));
        } else {
            prob = exp(log1 - log0) / (1.0 + exp(log1 - log0));
        }

        marginals[k] = prob;
        if (prob > high_thresh)       decisions[k] = 1;
        else if (prob < low_thresh)   decisions[k] = 0;
        else                          decisions[k] = -1;
    }
}

// ---- Mixed-precision batch marginals ----
__global__ void compute_marginals_mixed_batch_kernel(
    const double* __restrict__ raw_tensors,
    const int* __restrict__ data_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ logical_positions,
    const float* __restrict__ prefix_vecs_f32,
    const float* __restrict__ prefix_log_f32,
    const float* __restrict__ suffix_vecs_f32,
    const float* __restrict__ suffix_log_f32,
    const int* __restrict__ fixed,
    double* __restrict__ marginals,
    int* __restrict__ decisions,
    int K, int max_chi,
    double low_thresh, double high_thresh)
{
    int k = blockIdx.x;
    int batch = blockIdx.y;
    if (k >= K) return;
    if (fixed[batch * K + k]) return;

    int pos = logical_positions[k];
    int cl = chi_l[pos];
    int cr = chi_r[pos];
    int tid = threadIdx.x;
    int BD = blockDim.x;

    extern __shared__ double smem[];
    double* mid    = smem;
    double* reduce = smem + BD;

    int offset_raw = data_offsets[pos];
    long long bpo = (long long)batch * K * max_chi;
    int blo = batch * K;

    double log_result[2];

    for (int lam = 0; lam < 2; lam++) {
        double val = 0.0;
        if (tid < cr) {
            for (int a = 0; a < cl; a++) {
                val += (double)prefix_vecs_f32[bpo + k * max_chi + a] *
                       __ldg(&raw_tensors[offset_raw + a * 2 * cr + lam * cr + tid]);
            }
        }
        mid[tid] = (tid < cr) ? val : 0.0;
        __syncthreads();

        double prod = (tid < cr) ? mid[tid] * (double)suffix_vecs_f32[bpo + k * max_chi + tid] : 0.0;
        reduce[tid] = prod;
        __syncthreads();
        for (int s = BD / 2; s > 0; s >>= 1) {
            if (tid < s) reduce[tid] += reduce[tid + s];
            __syncthreads();
        }

        if (tid == 0) {
            double result = reduce[0];
            double abs_r = fabs(result);
            log_result[lam] = (abs_r > 0.0) ?
                log(abs_r) + (double)prefix_log_f32[blo + k] + (double)suffix_log_f32[blo + k]
                : -1e300;
        }
        __syncthreads();
    }

    if (tid == 0) {
        double log0 = log_result[0];
        double log1 = log_result[1];
        double prob;
        if (log0 <= -1e299 && log1 <= -1e299)  prob = 0.5;
        else if (log1 > log0)  prob = 1.0 / (1.0 + exp(log0 - log1));
        else  prob = exp(log1 - log0) / (1.0 + exp(log1 - log0));

        marginals[batch * K + k] = prob;
        if (prob > high_thresh)       decisions[batch * K + k] = 1;
        else if (prob < low_thresh)   decisions[batch * K + k] = 0;
        else                          decisions[batch * K + k] = -1;
    }
}

// ---- FP32 update base for fixed logicals ----
__global__ void update_base_for_fixed_fp32_kernel(
    const double* __restrict__ raw_tensors,
    const int* __restrict__ data_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ logical_positions,
    const int* __restrict__ fixed,
    const int* __restrict__ fixed_values,
    float* __restrict__ base_matrices_f32,
    float* __restrict__ base_matrices_T_f32,
    const int* __restrict__ base_offsets,
    int K,
    const int* __restrict__ skip_flag)
{
    if (skip_flag && *skip_flag) return;
    int k = blockIdx.x;
    if (k >= K || !fixed[k]) return;

    int pos = logical_positions[k];
    int cl = chi_l[pos];
    int cr = chi_r[pos];
    int offset_raw = data_offsets[pos];
    int offset_base = base_offsets[pos];
    int mat_size = cl * cr;
    int v = fixed_values[k];

    for (int idx = threadIdx.x; idx < mat_size; idx += blockDim.x) {
        int a = idx / cr;
        int b = idx % cr;
        float val = (float)__ldg(&raw_tensors[offset_raw + a * 2 * cr + v * cr + b]);
        base_matrices_f32[offset_base + a * cr + b] = val;
        base_matrices_T_f32[offset_base + b * cl + a] = val;
    }
}

// ---- FP32 batch update base for fixed ----
__global__ void update_base_for_fixed_fp32_batch_kernel(
    const double* __restrict__ raw_tensors,
    const int* __restrict__ data_offsets,
    const int* __restrict__ chi_l,
    const int* __restrict__ chi_r,
    const int* __restrict__ logical_positions,
    const int* __restrict__ fixed,
    const int* __restrict__ fixed_values,
    float* __restrict__ base_matrices_f32,
    float* __restrict__ base_matrices_T_f32,
    const int* __restrict__ base_offsets,
    int K, int total_base_elements)
{
    int k = blockIdx.x;
    int batch = blockIdx.y;
    if (k >= K || !fixed[batch * K + k]) return;

    int pos = logical_positions[k];
    int cl = chi_l[pos];
    int cr = chi_r[pos];
    int offset_raw = data_offsets[pos];
    long long offset_base = (long long)batch * total_base_elements + base_offsets[pos];
    int mat_size = cl * cr;
    int v = fixed_values[batch * K + k];

    for (int idx = threadIdx.x; idx < mat_size; idx += blockDim.x) {
        int a = idx / cr;
        int b = idx % cr;
        float val = (float)__ldg(&raw_tensors[offset_raw + a * 2 * cr + v * cr + b]);
        base_matrices_f32[offset_base + a * cr + b] = val;
        base_matrices_T_f32[offset_base + b * cl + a] = val;
    }
}

// ============================================================================
// Host: single decode
// ============================================================================

// Helper: launch forward + backward scans, adaptive mode (sequential or concurrent).
// Dispatches to FP32 or FP64 kernels based on dec->use_fp32_scan.
// skip_flag/scan_range: for device-side refinement (nullptr = host-side control).
static void launch_scans(MpsDecoder* dec,
                         int fwd_start_site = 0, int fwd_start_log = -1,
                         int bwd_end_site = -1, int bwd_end_log = -1,
                         const int* skip_flag = nullptr,
                         const int* scan_range = nullptr) {
    int N = dec->N, K = dec->K, max_chi = dec->max_chi;
    if (bwd_end_site < 0) bwd_end_site = N - 1;

    if (dec->use_fp32_scan) {
        // ---- FP32 scan path ----
        int SBS = dec->scan_block_size_f32;
        size_t smem_scan = (max_chi + SBS + 32) * sizeof(float);

        forward_scan_fp32_kernel<<<1, SBS, smem_scan, dec->stream>>>(
            dec->d_base_matrices_f32, dec->d_base_offsets,
            dec->d_chi_l, dec->d_chi_r,
            dec->d_site_type, dec->d_log_index,
            dec->d_prefix_vecs_f32, dec->d_prefix_log_f32,
            N, K, max_chi,
            fwd_start_site, fwd_start_log,
            skip_flag, scan_range);

        if (dec->use_sequential_scan) {
            backward_scan_fp32_kernel<<<1, SBS, smem_scan, dec->stream>>>(
                dec->d_base_matrices_T_f32, dec->d_base_offsets,
                dec->d_chi_l, dec->d_chi_r,
                dec->d_site_type, dec->d_log_index,
                dec->d_suffix_vecs_f32, dec->d_suffix_log_f32,
                N, K, max_chi,
                bwd_end_site, bwd_end_log,
                skip_flag, scan_range);
        } else {
            // For concurrent backward scan on stream2, stream2 must wait for stream
            // (ensure update_base or refinement_check finished on stream)
            if (skip_flag) {
                cudaEvent_t ev_sync;
                cudaEventCreate(&ev_sync);
                cudaEventRecord(ev_sync, dec->stream);
                cudaStreamWaitEvent(dec->stream2, ev_sync, 0);
                cudaEventDestroy(ev_sync);
            }
            backward_scan_fp32_kernel<<<1, SBS, smem_scan, dec->stream2>>>(
                dec->d_base_matrices_T_f32, dec->d_base_offsets,
                dec->d_chi_l, dec->d_chi_r,
                dec->d_site_type, dec->d_log_index,
                dec->d_suffix_vecs_f32, dec->d_suffix_log_f32,
                N, K, max_chi,
                bwd_end_site, bwd_end_log,
                skip_flag, scan_range);

            cudaEvent_t ev_bwd_done;
            cudaEventCreate(&ev_bwd_done);
            cudaEventRecord(ev_bwd_done, dec->stream2);
            cudaStreamWaitEvent(dec->stream, ev_bwd_done, 0);
            cudaEventDestroy(ev_bwd_done);
        }
    } else {
        // ---- FP64 scan path (original) ----
        int SBS = dec->scan_block_size;
        size_t smem_scan = (max_chi + SBS + 32) * sizeof(double);

        forward_scan_kernel<<<1, SBS, smem_scan, dec->stream>>>(
            dec->d_base_matrices, dec->d_base_offsets,
            dec->d_chi_l, dec->d_chi_r,
            dec->d_site_type, dec->d_log_index,
            dec->d_prefix_vecs, dec->d_prefix_log,
            N, K, max_chi,
            fwd_start_site, fwd_start_log,
            skip_flag, scan_range);

        if (dec->use_sequential_scan) {
            backward_scan_kernel<<<1, SBS, smem_scan, dec->stream>>>(
                dec->d_base_matrices_T, dec->d_base_offsets,
                dec->d_chi_l, dec->d_chi_r,
                dec->d_site_type, dec->d_log_index,
                dec->d_suffix_vecs, dec->d_suffix_log,
                N, K, max_chi,
                bwd_end_site, bwd_end_log,
                skip_flag, scan_range);
        } else {
            if (skip_flag) {
                cudaEvent_t ev_sync;
                cudaEventCreate(&ev_sync);
                cudaEventRecord(ev_sync, dec->stream);
                cudaStreamWaitEvent(dec->stream2, ev_sync, 0);
                cudaEventDestroy(ev_sync);
            }
            backward_scan_kernel<<<1, SBS, smem_scan, dec->stream2>>>(
                dec->d_base_matrices_T, dec->d_base_offsets,
                dec->d_chi_l, dec->d_chi_r,
                dec->d_site_type, dec->d_log_index,
                dec->d_suffix_vecs, dec->d_suffix_log,
                N, K, max_chi,
                bwd_end_site, bwd_end_log,
                skip_flag, scan_range);

            cudaEvent_t ev_bwd_done;
            cudaEventCreate(&ev_bwd_done);
            cudaEventRecord(ev_bwd_done, dec->stream2);
            cudaStreamWaitEvent(dec->stream, ev_bwd_done, 0);
            cudaEventDestroy(ev_bwd_done);
        }
    }
}

// Helper macros for dispatch to avoid deep nesting
#define LAUNCH_PREPARE_BASE(dec, N, BS) do { \
    if ((dec)->use_fp32_scan) { \
        prepare_base_matrices_fp32_kernel<<<N, BS, 0, (dec)->stream>>>( \
            (dec)->d_raw_tensors, (dec)->d_data_offsets, \
            (dec)->d_chi_l, (dec)->d_chi_r, \
            (dec)->d_site_type, (dec)->d_syn_index, (dec)->d_log_index, \
            (dec)->d_syndrome, \
            (dec)->d_fixed, (dec)->d_fixed_values, \
            (dec)->d_base_matrices_f32, (dec)->d_base_matrices_T_f32, (dec)->d_base_offsets, N); \
    } else { \
        prepare_base_matrices_kernel<<<N, BS, 0, (dec)->stream>>>( \
            (dec)->d_raw_tensors, (dec)->d_data_offsets, \
            (dec)->d_chi_l, (dec)->d_chi_r, \
            (dec)->d_site_type, (dec)->d_syn_index, (dec)->d_log_index, \
            (dec)->d_syndrome, \
            (dec)->d_fixed, (dec)->d_fixed_values, \
            (dec)->d_base_matrices, (dec)->d_base_matrices_T, (dec)->d_base_offsets, N); \
    } \
} while(0)

#define LAUNCH_MARGINALS(dec, K, BS, smem, lo, hi, skip) do { \
    if ((dec)->use_fp32_scan) { \
        /* Convert FP32 prefix/suffix to FP64, then use standard FP64 marginals. */ \
        /* This avoids the mixed-precision kernel's compiler overhead. */ \
        int _cvt_n = K * (dec)->max_chi; \
        int _cvt_bs = 256; \
        int _cvt_grid = (_cvt_n + _cvt_bs - 1) / _cvt_bs; \
        convert_vecs_f32_to_f64_kernel<<<_cvt_grid, _cvt_bs, 0, (dec)->stream>>>( \
            (dec)->d_prefix_vecs_f32, (dec)->d_prefix_log_f32, \
            (dec)->d_prefix_vecs, (dec)->d_prefix_log, \
            K, (dec)->max_chi, (skip)); \
        convert_vecs_f32_to_f64_kernel<<<_cvt_grid, _cvt_bs, 0, (dec)->stream>>>( \
            (dec)->d_suffix_vecs_f32, (dec)->d_suffix_log_f32, \
            (dec)->d_suffix_vecs, (dec)->d_suffix_log, \
            K, (dec)->max_chi, (skip)); \
        compute_marginals_kernel<<<K, BS, smem, (dec)->stream>>>( \
            (dec)->d_raw_tensors, (dec)->d_data_offsets, \
            (dec)->d_chi_l, (dec)->d_chi_r, \
            (dec)->d_logical_positions, \
            (dec)->d_prefix_vecs, (dec)->d_prefix_log, \
            (dec)->d_suffix_vecs, (dec)->d_suffix_log, \
            (dec)->d_fixed, \
            (dec)->d_marginals, (dec)->d_decisions, \
            K, (dec)->max_chi, lo, hi, (skip)); \
    } else { \
        compute_marginals_kernel<<<K, BS, smem, (dec)->stream>>>( \
            (dec)->d_raw_tensors, (dec)->d_data_offsets, \
            (dec)->d_chi_l, (dec)->d_chi_r, \
            (dec)->d_logical_positions, \
            (dec)->d_prefix_vecs, (dec)->d_prefix_log, \
            (dec)->d_suffix_vecs, (dec)->d_suffix_log, \
            (dec)->d_fixed, \
            (dec)->d_marginals, (dec)->d_decisions, \
            K, (dec)->max_chi, lo, hi, (skip)); \
    } \
} while(0)

#define LAUNCH_UPDATE_BASE_FIXED(dec, K, BS, skip) do { \
    if ((dec)->use_fp32_scan) { \
        update_base_for_fixed_fp32_kernel<<<K, BS, 0, (dec)->stream>>>( \
            (dec)->d_raw_tensors, (dec)->d_data_offsets, \
            (dec)->d_chi_l, (dec)->d_chi_r, \
            (dec)->d_logical_positions, \
            (dec)->d_fixed, (dec)->d_fixed_values, \
            (dec)->d_base_matrices_f32, (dec)->d_base_matrices_T_f32, (dec)->d_base_offsets, K, (skip)); \
    } else { \
        update_base_for_fixed_kernel<<<K, BS, 0, (dec)->stream>>>( \
            (dec)->d_raw_tensors, (dec)->d_data_offsets, \
            (dec)->d_chi_l, (dec)->d_chi_r, \
            (dec)->d_logical_positions, \
            (dec)->d_fixed, (dec)->d_fixed_values, \
            (dec)->d_base_matrices, (dec)->d_base_matrices_T, (dec)->d_base_offsets, K, (skip)); \
    } \
} while(0)

static int run_single_decode(MpsDecoder* dec, double low_thresh, double high_thresh, int max_rounds) {
    int N = dec->N;
    int K = dec->K;
    int BS = dec->block_size;

    // Marginals always use FP64 shared memory
    size_t smem_marg = 2 * BS * sizeof(double);

    // Phase 1: Prepare base matrices
    LAUNCH_PREPARE_BASE(dec, N, BS);

    // If concurrent mode, stream2 must wait for prepare to finish
    if (!dec->use_sequential_scan) {
        cudaEvent_t ev_prepare_done;
        cudaEventCreate(&ev_prepare_done);
        cudaEventRecord(ev_prepare_done, dec->stream);
        cudaStreamWaitEvent(dec->stream2, ev_prepare_done, 0);
        cudaEventDestroy(ev_prepare_done);
    }

    // Phase 2+3: Forward + Backward scan
    launch_scans(dec);

    // Phase 4: Compute marginals
    LAUNCH_MARGINALS(dec, K, BS, smem_marg, low_thresh, high_thresh, nullptr);

    // ---- Conditional refinement loop (host-sync) ----
    int h_ctrl[6];

    for (int round = 0; round < max_rounds; round++) {
        refinement_check_kernel<<<1, 1, 0, dec->stream>>>(
            dec->d_decisions, dec->d_fixed, dec->d_fixed_values,
            dec->d_logical_positions, dec->d_refine_ctrl, K,
            nullptr, nullptr);

        cudaMemcpyAsync(h_ctrl, dec->d_refine_ctrl, 6 * sizeof(int),
                        cudaMemcpyDeviceToHost, dec->stream);
        cudaStreamSynchronize(dec->stream);

        if (!h_ctrl[0] || !h_ctrl[1]) break;

        int earliest_log = h_ctrl[2];
        int latest_log = h_ctrl[3];
        int earliest_site = h_ctrl[4];
        int latest_site = h_ctrl[5];

        LAUNCH_UPDATE_BASE_FIXED(dec, K, BS, nullptr);

        if (!dec->use_sequential_scan) {
            cudaEvent_t ev_update;
            cudaEventCreate(&ev_update);
            cudaEventRecord(ev_update, dec->stream);
            cudaStreamWaitEvent(dec->stream2, ev_update, 0);
            cudaEventDestroy(ev_update);
        }

        launch_scans(dec,
                     earliest_site, earliest_log,
                     latest_site, latest_log);

        LAUNCH_MARGINALS(dec, K, BS, smem_marg, low_thresh, high_thresh, nullptr);
    }

    // Final pass: force-decide remaining unknowns
    {
        refinement_check_kernel<<<1, 1, 0, dec->stream>>>(
            dec->d_decisions, dec->d_fixed, dec->d_fixed_values,
            dec->d_logical_positions, dec->d_refine_ctrl, K,
            nullptr, nullptr);

        cudaMemcpyAsync(h_ctrl, dec->d_refine_ctrl, 6 * sizeof(int),
                        cudaMemcpyDeviceToHost, dec->stream);
        cudaStreamSynchronize(dec->stream);

        bool need_final = (h_ctrl[0] != 0);

        if (need_final) {
            // Force-fix remaining unknowns on host
            std::vector<int> h_fix(K), h_fv(K);
            std::vector<double> h_marg(K);
            cudaMemcpy(h_fix.data(), dec->d_fixed, K * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_fv.data(), dec->d_fixed_values, K * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_marg.data(), dec->d_marginals, K * sizeof(double), cudaMemcpyDeviceToHost);

            std::vector<int> h_logpos(K);
            cudaMemcpy(h_logpos.data(), dec->d_logical_positions, K * sizeof(int), cudaMemcpyDeviceToHost);

            int earliest_site = N, latest_site = -1;
            int earliest_log = -1, latest_log = -1;
            for (int k = 0; k < K; k++) {
                if (h_fix[k]) continue;
                h_fix[k] = 1;
                h_fv[k] = (h_marg[k] >= 0.5) ? 1 : 0;
                int pos = h_logpos[k];
                if (pos < earliest_site) { earliest_site = pos; earliest_log = k; }
                if (pos > latest_site) { latest_site = pos; latest_log = k; }
            }

            cudaMemcpyAsync(dec->d_fixed, h_fix.data(), K * sizeof(int),
                            cudaMemcpyHostToDevice, dec->stream);
            cudaMemcpyAsync(dec->d_fixed_values, h_fv.data(), K * sizeof(int),
                            cudaMemcpyHostToDevice, dec->stream);

            LAUNCH_UPDATE_BASE_FIXED(dec, K, BS, nullptr);

            if (!dec->use_sequential_scan) {
                cudaEvent_t ev_update;
                cudaEventCreate(&ev_update);
                cudaEventRecord(ev_update, dec->stream);
                cudaStreamWaitEvent(dec->stream2, ev_update, 0);
                cudaEventDestroy(ev_update);
            }

            if (earliest_site >= 0 && latest_site >= 0) {
                launch_scans(dec,
                             earliest_site, earliest_log,
                             latest_site, latest_log);
            } else {
                launch_scans(dec);
            }

            LAUNCH_MARGINALS(dec, K, BS, smem_marg, 0.5, 0.5, nullptr);
        }
    }

    return 0;
}

// ============================================================================
// Public API
// ============================================================================

extern "C" {

MpsDecoderHandle mps_decoder_create(const char* data_dir) {
    MpsDecoder* dec = new MpsDecoder();
    memset(dec, 0, sizeof(MpsDecoder));
    dec->batch_alloc = 0;

    std::string meta_path = std::string(data_dir) + "/chain_meta.json";
    std::string json = read_file_to_string(meta_path.c_str());
    if (json.empty()) {
        fprintf(stderr, "Failed to read %s\n", meta_path.c_str());
        delete dec; return nullptr;
    }

    dec->N = parse_json_int(json, "num_sites");
    dec->syndrome_labels = parse_json_string_array(json, "syndrome_labels");
    dec->logical_labels = parse_json_string_array(json, "logical_labels");
    dec->site_labels = parse_json_string_array(json, "site_labels");
    dec->K = (int)dec->logical_labels.size();
    dec->M = (int)dec->syndrome_labels.size();

    if (dec->N <= 0 || dec->K <= 0 || dec->M <= 0) {
        fprintf(stderr, "Invalid metadata: N=%d K=%d M=%d\n", dec->N, dec->K, dec->M);
        delete dec; return nullptr;
    }

    // Load site tensors
    dec->max_chi = 0;
    dec->total_raw_elements = 0;
    dec->total_base_elements = 0;

    std::vector<double> all_raw;
    std::vector<int> h_offsets(dec->N), h_cl(dec->N), h_cr(dec->N);
    std::vector<int> h_stype(dec->N), h_si(dec->N), h_li(dec->N);
    dec->h_base_offsets.resize(dec->N);

    for (int i = 0; i < dec->N; i++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/site_%d.npy", data_dir, i);

        NpyArray arr;
        if (!read_npy(path, arr) || arr.shape.size() != 3 || arr.shape[1] != 2) {
            fprintf(stderr, "Failed to read or bad shape: %s\n", path);
            delete dec; return nullptr;
        }

        int cl = arr.shape[0], cr = arr.shape[2];
        dec->max_chi = std::max(dec->max_chi, std::max(cl, cr));

        h_offsets[i] = dec->total_raw_elements;
        h_cl[i] = cl;
        h_cr[i] = cr;

        // Determine site type
        std::string& label = dec->site_labels[i];
        int syn_idx = -1, log_idx = -1;
        for (int s = 0; s < dec->M; s++) {
            if (dec->syndrome_labels[s] == label) { syn_idx = s; break; }
        }
        if (syn_idx < 0) {
            for (int l = 0; l < dec->K; l++) {
                if (dec->logical_labels[l] == label) { log_idx = l; break; }
            }
        }

        h_stype[i] = (log_idx >= 0) ? 1 : 0;
        h_si[i] = syn_idx;
        h_li[i] = log_idx;

        all_raw.insert(all_raw.end(), arr.data.begin(), arr.data.end());
        dec->total_raw_elements += (int)arr.data.size();
        dec->h_base_offsets[i] = dec->total_base_elements;
        dec->total_base_elements += cl * cr;
    }

    // Build logical_positions indexed by logical index
    dec->logical_positions.assign(dec->K, -1);
    for (int i = 0; i < dec->N; i++) {
        if (h_li[i] >= 0) dec->logical_positions[h_li[i]] = i;
    }

    // Compute block_size: next power of 2 >= max_chi, minimum 32
    dec->block_size = 32;
    while (dec->block_size < dec->max_chi) dec->block_size *= 2;
    if (dec->block_size > 256) dec->block_size = 256;

    // FP64 scan: use more threads for latency hiding (up to 4x block_size, capped at 1024)
    dec->scan_block_size = dec->block_size;
    if (dec->max_chi >= 32) {
        dec->scan_block_size = dec->block_size * 4;
        if (dec->scan_block_size > 1024) dec->scan_block_size = 1024;
    }

    // FP32 scan: same thread count as FP64 for latency hiding on single SM.
    // Halved data size (float vs double) gives bandwidth savings at same occupancy.
    dec->scan_block_size_f32 = dec->block_size;
    if (dec->max_chi >= 32) {
        dec->scan_block_size_f32 = dec->block_size * 4;
        if (dec->scan_block_size_f32 > 1024) dec->scan_block_size_f32 = 1024;
    }

    // Enable FP32 scan by default (significant speedup, minimal accuracy impact)
    dec->use_fp32_scan = true;

    // ---- GPU allocation ----
    cudaMalloc(&dec->d_raw_tensors, dec->total_raw_elements * sizeof(double));
    cudaMemcpy(dec->d_raw_tensors, all_raw.data(), dec->total_raw_elements * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&dec->d_data_offsets, dec->N * sizeof(int));
    cudaMalloc(&dec->d_chi_l, dec->N * sizeof(int));
    cudaMalloc(&dec->d_chi_r, dec->N * sizeof(int));
    cudaMalloc(&dec->d_site_type, dec->N * sizeof(int));
    cudaMalloc(&dec->d_syn_index, dec->N * sizeof(int));
    cudaMalloc(&dec->d_log_index, dec->N * sizeof(int));
    cudaMalloc(&dec->d_logical_positions, dec->K * sizeof(int));
    cudaMalloc(&dec->d_base_offsets, dec->N * sizeof(int));

    cudaMemcpy(dec->d_data_offsets, h_offsets.data(), dec->N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dec->d_chi_l, h_cl.data(), dec->N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dec->d_chi_r, h_cr.data(), dec->N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dec->d_site_type, h_stype.data(), dec->N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dec->d_syn_index, h_si.data(), dec->N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dec->d_log_index, h_li.data(), dec->N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dec->d_logical_positions, dec->logical_positions.data(), dec->K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dec->d_base_offsets, dec->h_base_offsets.data(), dec->N * sizeof(int), cudaMemcpyHostToDevice);

    // Working buffers — stride = max_chi for prefix/suffix vectors
    cudaMalloc(&dec->d_base_matrices, dec->total_base_elements * sizeof(double));
    cudaMalloc(&dec->d_base_matrices_T, dec->total_base_elements * sizeof(double));  // transposed
    cudaMalloc(&dec->d_syndrome, dec->M * sizeof(double));
    cudaMalloc(&dec->d_prefix_vecs, (size_t)dec->K * dec->max_chi * sizeof(double));
    cudaMalloc(&dec->d_suffix_vecs, (size_t)dec->K * dec->max_chi * sizeof(double));
    cudaMalloc(&dec->d_prefix_log, dec->K * sizeof(double));
    cudaMalloc(&dec->d_suffix_log, dec->K * sizeof(double));
    cudaMalloc(&dec->d_marginals, dec->K * sizeof(double));
    cudaMalloc(&dec->d_decisions, dec->K * sizeof(int));
    cudaMalloc(&dec->d_fixed, dec->K * sizeof(int));
    cudaMalloc(&dec->d_fixed_values, dec->K * sizeof(int));
    cudaMalloc(&dec->d_refine_ctrl, 6 * sizeof(int));

    // Device-side refinement flags
    cudaMalloc(&dec->d_refine_skip, sizeof(int));
    cudaMalloc(&dec->d_final_skip, sizeof(int));
    cudaMalloc(&dec->d_scan_range, 4 * sizeof(int));

    // FP32 working buffers (always allocated; cheap compared to FP64 buffers)
    cudaMalloc(&dec->d_base_matrices_f32, dec->total_base_elements * sizeof(float));
    cudaMalloc(&dec->d_base_matrices_T_f32, dec->total_base_elements * sizeof(float));
    cudaMalloc(&dec->d_prefix_vecs_f32, (size_t)dec->K * dec->max_chi * sizeof(float));
    cudaMalloc(&dec->d_suffix_vecs_f32, (size_t)dec->K * dec->max_chi * sizeof(float));
    cudaMalloc(&dec->d_prefix_log_f32, dec->K * sizeof(float));
    cudaMalloc(&dec->d_suffix_log_f32, dec->K * sizeof(float));

    // Initialize FP32 batch pointers to null
    dec->d_batch_base_f32 = nullptr;
    dec->d_batch_base_T_f32 = nullptr;
    dec->d_batch_prefix_f32 = nullptr;
    dec->d_batch_suffix_f32 = nullptr;
    dec->d_batch_prefix_log_f32 = nullptr;
    dec->d_batch_suffix_log_f32 = nullptr;

    cudaStreamCreate(&dec->stream);
    cudaStreamCreate(&dec->stream2);

    // Detect L2 cache size (for diagnostics)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        dec->l2_cache_size = prop.l2CacheSize;
        size_t working_set = 2 * (size_t)dec->total_base_elements * sizeof(double);
        // Always use concurrent dual-stream: overlap benefit far outweighs L2 contention.
        // Measured on H100: concurrent QCC_144 = 4176 us vs sequential = 7614 us.
        // The ~5-8% L2 penalty on forward scan is dwarfed by the ~50% overlap saving.
        dec->use_sequential_scan = false;
        fprintf(stderr, "L2 cache: %d MB, scan working set: %.1f MB -> CONCURRENT mode\n",
                dec->l2_cache_size / (1024 * 1024),
                (double)working_set / (1024 * 1024));
    }

    fprintf(stderr, "MPS decoder created: N=%d K=%d M=%d max_chi=%d block_size=%d "
            "scan_block_size=%d(fp64)/%d(fp32) fp32_scan=%s "
            "raw_elems=%d base_elems=%d\n",
            dec->N, dec->K, dec->M, dec->max_chi, dec->block_size,
            dec->scan_block_size, dec->scan_block_size_f32,
            dec->use_fp32_scan ? "ON" : "OFF",
            dec->total_raw_elements, dec->total_base_elements);

    return (MpsDecoderHandle)dec;
}

void mps_decoder_destroy(MpsDecoderHandle handle) {
    if (!handle) return;
    MpsDecoder* dec = (MpsDecoder*)handle;

    cudaFree(dec->d_raw_tensors);
    cudaFree(dec->d_data_offsets);
    cudaFree(dec->d_chi_l);
    cudaFree(dec->d_chi_r);
    cudaFree(dec->d_site_type);
    cudaFree(dec->d_syn_index);
    cudaFree(dec->d_log_index);
    cudaFree(dec->d_logical_positions);
    cudaFree(dec->d_base_offsets);
    cudaFree(dec->d_base_matrices);
    cudaFree(dec->d_base_matrices_T);
    cudaFree(dec->d_syndrome);
    cudaFree(dec->d_prefix_vecs);
    cudaFree(dec->d_suffix_vecs);
    cudaFree(dec->d_prefix_log);
    cudaFree(dec->d_suffix_log);
    cudaFree(dec->d_marginals);
    cudaFree(dec->d_decisions);
    cudaFree(dec->d_fixed);
    cudaFree(dec->d_fixed_values);
    cudaFree(dec->d_refine_ctrl);
    cudaFree(dec->d_refine_skip);
    cudaFree(dec->d_final_skip);
    cudaFree(dec->d_scan_range);

    // Free FP32 buffers
    cudaFree(dec->d_base_matrices_f32);
    cudaFree(dec->d_base_matrices_T_f32);
    cudaFree(dec->d_prefix_vecs_f32);
    cudaFree(dec->d_suffix_vecs_f32);
    cudaFree(dec->d_prefix_log_f32);
    cudaFree(dec->d_suffix_log_f32);

    if (dec->batch_alloc > 0) {
        cudaFree(dec->d_batch_syndrome);
        cudaFree(dec->d_batch_base);
        cudaFree(dec->d_batch_base_T);
        cudaFree(dec->d_batch_prefix);
        cudaFree(dec->d_batch_suffix);
        cudaFree(dec->d_batch_prefix_log);
        cudaFree(dec->d_batch_suffix_log);
        cudaFree(dec->d_batch_marginals);
        cudaFree(dec->d_batch_decisions);
        cudaFree(dec->d_batch_fixed);
        cudaFree(dec->d_batch_fixed_values);

        // Free FP32 batch buffers
        cudaFree(dec->d_batch_base_f32);
        cudaFree(dec->d_batch_base_T_f32);
        cudaFree(dec->d_batch_prefix_f32);
        cudaFree(dec->d_batch_suffix_f32);
        cudaFree(dec->d_batch_prefix_log_f32);
        cudaFree(dec->d_batch_suffix_log_f32);
    }

    cudaStreamDestroy(dec->stream);
    cudaStreamDestroy(dec->stream2);
    delete dec;
}

int mps_decoder_decode(MpsDecoderHandle handle,
                       const int* syndrome,
                       int* decisions,
                       double* marginals,
                       double* latency_us,
                       double low_thresh,
                       double high_thresh,
                       int max_rounds) {
    if (!handle) return -1;
    MpsDecoder* dec = (MpsDecoder*)handle;

    std::vector<double> syn_d(dec->M);
    for (int i = 0; i < dec->M; i++) syn_d[i] = (double)syndrome[i];

    CUDA_CHECK(cudaMemcpyAsync(dec->d_syndrome, syn_d.data(),
                               dec->M * sizeof(double), cudaMemcpyHostToDevice, dec->stream));
    CUDA_CHECK(cudaMemsetAsync(dec->d_fixed, 0, dec->K * sizeof(int), dec->stream));
    CUDA_CHECK(cudaMemsetAsync(dec->d_fixed_values, 0, dec->K * sizeof(int), dec->stream));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, dec->stream);

    int ret = run_single_decode(dec, low_thresh, high_thresh, max_rounds);

    cudaEventRecord(stop, dec->stream);
    cudaStreamSynchronize(dec->stream);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    *latency_us = (double)elapsed_ms * 1000.0;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (ret != 0) return ret;

    // Download results
    std::vector<int> h_dec(dec->K);
    std::vector<double> h_marg(dec->K);
    cudaMemcpy(h_dec.data(), dec->d_decisions, dec->K * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_marg.data(), dec->d_marginals, dec->K * sizeof(double), cudaMemcpyDeviceToHost);

    for (int k = 0; k < dec->K; k++) {
        marginals[k] = h_marg[k];
        decisions[k] = (h_dec[k] >= 0) ? h_dec[k] : (h_marg[k] >= 0.5 ? 1 : 0);
    }

    return 0;
}

int mps_decoder_decode_batch(MpsDecoderHandle handle,
                             const int* syndromes,
                             int batch_size,
                             int* decisions,
                             double* marginals,
                             double* latency_us,
                             double low_thresh,
                             double high_thresh,
                             int max_rounds) {
    if (!handle) return -1;
    MpsDecoder* dec = (MpsDecoder*)handle;
    int K = dec->K, M = dec->M, N = dec->N;
    int max_chi = dec->max_chi;
    int BS = dec->block_size;

    // Allocate batch buffers if needed
    if (batch_size > dec->batch_alloc) {
        if (dec->batch_alloc > 0) {
            cudaFree(dec->d_batch_syndrome);
            cudaFree(dec->d_batch_base);
            cudaFree(dec->d_batch_base_T);
            cudaFree(dec->d_batch_prefix);
            cudaFree(dec->d_batch_suffix);
            cudaFree(dec->d_batch_prefix_log);
            cudaFree(dec->d_batch_suffix_log);
            cudaFree(dec->d_batch_marginals);
            cudaFree(dec->d_batch_decisions);
            cudaFree(dec->d_batch_fixed);
            cudaFree(dec->d_batch_fixed_values);
            cudaFree(dec->d_batch_base_f32);
            cudaFree(dec->d_batch_base_T_f32);
            cudaFree(dec->d_batch_prefix_f32);
            cudaFree(dec->d_batch_suffix_f32);
            cudaFree(dec->d_batch_prefix_log_f32);
            cudaFree(dec->d_batch_suffix_log_f32);
            dec->batch_alloc = 0;
        }
        cudaError_t err;
        size_t total_needed =
            (size_t)batch_size * M * sizeof(double) +
            (size_t)batch_size * dec->total_base_elements * sizeof(double) * 2 +
            (size_t)batch_size * dec->total_base_elements * sizeof(float) * 2 +   // FP32 base
            (size_t)batch_size * K * max_chi * sizeof(double) * 2 +
            (size_t)batch_size * K * max_chi * sizeof(float) * 2 +                // FP32 prefix/suffix
            (size_t)batch_size * K * sizeof(double) * 2 +
            (size_t)batch_size * K * sizeof(float) * 2 +                          // FP32 log scales
            (size_t)batch_size * K * sizeof(double) +
            (size_t)batch_size * K * sizeof(int) * 3;
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        if (total_needed > free_mem * 9 / 10) {
            fprintf(stderr, "Batch %d needs %.1f GB but only %.1f GB free\n",
                    batch_size, total_needed / 1e9, free_mem / 1e9);
            *latency_us = 0;
            return -2;
        }
        err = cudaMalloc(&dec->d_batch_syndrome, (size_t)batch_size * M * sizeof(double));
        if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc batch_syndrome failed: %s\n", cudaGetErrorString(err)); return -2; }
        err = cudaMalloc(&dec->d_batch_base, (size_t)batch_size * dec->total_base_elements * sizeof(double));
        if (err != cudaSuccess) { cudaFree(dec->d_batch_syndrome); return -2; }
        err = cudaMalloc(&dec->d_batch_base_T, (size_t)batch_size * dec->total_base_elements * sizeof(double));
        if (err != cudaSuccess) { cudaFree(dec->d_batch_syndrome); cudaFree(dec->d_batch_base); return -2; }
        cudaMalloc(&dec->d_batch_prefix, (size_t)batch_size * K * max_chi * sizeof(double));
        cudaMalloc(&dec->d_batch_suffix, (size_t)batch_size * K * max_chi * sizeof(double));
        cudaMalloc(&dec->d_batch_prefix_log, (size_t)batch_size * K * sizeof(double));
        cudaMalloc(&dec->d_batch_suffix_log, (size_t)batch_size * K * sizeof(double));
        cudaMalloc(&dec->d_batch_marginals, (size_t)batch_size * K * sizeof(double));
        cudaMalloc(&dec->d_batch_decisions, (size_t)batch_size * K * sizeof(int));
        cudaMalloc(&dec->d_batch_fixed, (size_t)batch_size * K * sizeof(int));
        cudaMalloc(&dec->d_batch_fixed_values, (size_t)batch_size * K * sizeof(int));
        // FP32 batch buffers
        cudaMalloc(&dec->d_batch_base_f32, (size_t)batch_size * dec->total_base_elements * sizeof(float));
        cudaMalloc(&dec->d_batch_base_T_f32, (size_t)batch_size * dec->total_base_elements * sizeof(float));
        cudaMalloc(&dec->d_batch_prefix_f32, (size_t)batch_size * K * max_chi * sizeof(float));
        cudaMalloc(&dec->d_batch_suffix_f32, (size_t)batch_size * K * max_chi * sizeof(float));
        cudaMalloc(&dec->d_batch_prefix_log_f32, (size_t)batch_size * K * sizeof(float));
        cudaMalloc(&dec->d_batch_suffix_log_f32, (size_t)batch_size * K * sizeof(float));
        dec->batch_alloc = batch_size;
    }

    std::vector<double> syn_d((size_t)batch_size * M);
    for (int i = 0; i < batch_size * M; i++) syn_d[i] = (double)syndromes[i];
    cudaMemcpyAsync(dec->d_batch_syndrome, syn_d.data(),
                    (size_t)batch_size * M * sizeof(double), cudaMemcpyHostToDevice, dec->stream);
    cudaMemsetAsync(dec->d_batch_fixed, 0, (size_t)batch_size * K * sizeof(int), dec->stream);
    cudaMemsetAsync(dec->d_batch_fixed_values, 0, (size_t)batch_size * K * sizeof(int), dec->stream);

    bool fp32 = dec->use_fp32_scan;
    size_t smem_scan = fp32 ? (3 * BS * sizeof(float)) : (3 * BS * sizeof(double));
    size_t smem_marg = 2 * BS * sizeof(double);  // marginals always FP64

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, dec->stream);

    // Phase 1: Prepare base matrices
    dim3 grid_base(N, batch_size);
    if (fp32) {
        prepare_base_matrices_fp32_batch_kernel<<<grid_base, BS, 0, dec->stream>>>(
            dec->d_raw_tensors, dec->d_data_offsets,
            dec->d_chi_l, dec->d_chi_r,
            dec->d_site_type, dec->d_syn_index, dec->d_log_index,
            dec->d_batch_syndrome,
            dec->d_batch_fixed, dec->d_batch_fixed_values,
            dec->d_batch_base_f32, dec->d_batch_base_T_f32, dec->d_base_offsets,
            N, M, K, dec->total_base_elements);
    } else {
        prepare_base_matrices_batch_kernel<<<grid_base, BS, 0, dec->stream>>>(
            dec->d_raw_tensors, dec->d_data_offsets,
            dec->d_chi_l, dec->d_chi_r,
            dec->d_site_type, dec->d_syn_index, dec->d_log_index,
            dec->d_batch_syndrome,
            dec->d_batch_fixed, dec->d_batch_fixed_values,
            dec->d_batch_base, dec->d_batch_base_T, dec->d_base_offsets,
            N, M, K, dec->total_base_elements);
    }

    // Phase 2+3: Forward + backward scan
    if (fp32) {
        forward_scan_fp32_batch_kernel<<<batch_size, BS, smem_scan, dec->stream>>>(
            dec->d_batch_base_f32, dec->d_base_offsets,
            dec->d_chi_l, dec->d_chi_r,
            dec->d_site_type, dec->d_log_index,
            dec->d_batch_prefix_f32, dec->d_batch_prefix_log_f32,
            N, K, max_chi, dec->total_base_elements);
        backward_scan_fp32_batch_kernel<<<batch_size, BS, smem_scan, dec->stream>>>(
            dec->d_batch_base_T_f32, dec->d_base_offsets,
            dec->d_chi_l, dec->d_chi_r,
            dec->d_site_type, dec->d_log_index,
            dec->d_batch_suffix_f32, dec->d_batch_suffix_log_f32,
            N, K, max_chi, dec->total_base_elements);
    } else {
        forward_scan_batch_kernel<<<batch_size, BS, smem_scan, dec->stream>>>(
            dec->d_batch_base, dec->d_base_offsets,
            dec->d_chi_l, dec->d_chi_r,
            dec->d_site_type, dec->d_log_index,
            dec->d_batch_prefix, dec->d_batch_prefix_log,
            N, K, max_chi, dec->total_base_elements);
        backward_scan_batch_kernel<<<batch_size, BS, smem_scan, dec->stream>>>(
            dec->d_batch_base_T, dec->d_base_offsets,
            dec->d_chi_l, dec->d_chi_r,
            dec->d_site_type, dec->d_log_index,
            dec->d_batch_suffix, dec->d_batch_suffix_log,
            N, K, max_chi, dec->total_base_elements);
    }

    // Phase 4: Marginals
    dim3 grid_marg(K, batch_size);
    if (fp32) {
        compute_marginals_mixed_batch_kernel<<<grid_marg, BS, smem_marg, dec->stream>>>(
            dec->d_raw_tensors, dec->d_data_offsets,
            dec->d_chi_l, dec->d_chi_r,
            dec->d_logical_positions,
            dec->d_batch_prefix_f32, dec->d_batch_prefix_log_f32,
            dec->d_batch_suffix_f32, dec->d_batch_suffix_log_f32,
            dec->d_batch_fixed,
            dec->d_batch_marginals, dec->d_batch_decisions,
            K, max_chi, low_thresh, high_thresh);
    } else {
        compute_marginals_batch_kernel<<<grid_marg, BS, smem_marg, dec->stream>>>(
            dec->d_raw_tensors, dec->d_data_offsets,
            dec->d_chi_l, dec->d_chi_r,
            dec->d_logical_positions,
            dec->d_batch_prefix, dec->d_batch_prefix_log,
            dec->d_batch_suffix, dec->d_batch_suffix_log,
            dec->d_batch_fixed,
            dec->d_batch_marginals, dec->d_batch_decisions,
            K, max_chi, low_thresh, high_thresh);
    }

    // Refinement
    std::vector<int> h_dec_all((size_t)batch_size * K);
    std::vector<int> h_fix_all((size_t)batch_size * K, 0);
    std::vector<int> h_fval_all((size_t)batch_size * K, 0);

    for (int round = 0; round < max_rounds; round++) {
        cudaMemcpyAsync(h_dec_all.data(), dec->d_batch_decisions,
                        (size_t)batch_size * K * sizeof(int), cudaMemcpyDeviceToHost, dec->stream);
        cudaStreamSynchronize(dec->stream);

        bool has_unk = false, has_new = false;
        for (int idx = 0; idx < batch_size * K; idx++) {
            if (h_fix_all[idx]) continue;
            if (h_dec_all[idx] != -1) {
                h_fix_all[idx] = 1;
                h_fval_all[idx] = h_dec_all[idx];
                has_new = true;
            } else {
                has_unk = true;
            }
        }
        if (!has_unk || !has_new) break;

        cudaMemcpyAsync(dec->d_batch_fixed, h_fix_all.data(),
                        (size_t)batch_size * K * sizeof(int), cudaMemcpyHostToDevice, dec->stream);
        cudaMemcpyAsync(dec->d_batch_fixed_values, h_fval_all.data(),
                        (size_t)batch_size * K * sizeof(int), cudaMemcpyHostToDevice, dec->stream);

        dim3 grid_fix(K, batch_size);
        if (fp32) {
            update_base_for_fixed_fp32_batch_kernel<<<grid_fix, BS, 0, dec->stream>>>(
                dec->d_raw_tensors, dec->d_data_offsets,
                dec->d_chi_l, dec->d_chi_r,
                dec->d_logical_positions,
                dec->d_batch_fixed, dec->d_batch_fixed_values,
                dec->d_batch_base_f32, dec->d_batch_base_T_f32, dec->d_base_offsets,
                K, dec->total_base_elements);
            forward_scan_fp32_batch_kernel<<<batch_size, BS, smem_scan, dec->stream>>>(
                dec->d_batch_base_f32, dec->d_base_offsets,
                dec->d_chi_l, dec->d_chi_r,
                dec->d_site_type, dec->d_log_index,
                dec->d_batch_prefix_f32, dec->d_batch_prefix_log_f32,
                N, K, max_chi, dec->total_base_elements);
            backward_scan_fp32_batch_kernel<<<batch_size, BS, smem_scan, dec->stream>>>(
                dec->d_batch_base_T_f32, dec->d_base_offsets,
                dec->d_chi_l, dec->d_chi_r,
                dec->d_site_type, dec->d_log_index,
                dec->d_batch_suffix_f32, dec->d_batch_suffix_log_f32,
                N, K, max_chi, dec->total_base_elements);
            { /* Convert FP32 batch prefix/suffix to FP64, then use standard batch marginals */
            int _cvt_n = batch_size * K * max_chi;
            int _cvt_bs = 256;
            int _cvt_grid = (_cvt_n + _cvt_bs - 1) / _cvt_bs;
            convert_vecs_f32_to_f64_kernel<<<_cvt_grid, _cvt_bs, 0, dec->stream>>>(
                dec->d_batch_prefix_f32, dec->d_batch_prefix_log_f32,
                dec->d_batch_prefix, dec->d_batch_prefix_log,
                batch_size * K, max_chi, nullptr);
            convert_vecs_f32_to_f64_kernel<<<_cvt_grid, _cvt_bs, 0, dec->stream>>>(
                dec->d_batch_suffix_f32, dec->d_batch_suffix_log_f32,
                dec->d_batch_suffix, dec->d_batch_suffix_log,
                batch_size * K, max_chi, nullptr);
        }
        compute_marginals_batch_kernel<<<grid_marg, BS, smem_marg, dec->stream>>>(
                dec->d_raw_tensors, dec->d_data_offsets,
                dec->d_chi_l, dec->d_chi_r,
                dec->d_logical_positions,
                dec->d_batch_prefix, dec->d_batch_prefix_log,
                dec->d_batch_suffix, dec->d_batch_suffix_log,
                dec->d_batch_fixed,
                dec->d_batch_marginals, dec->d_batch_decisions,
                K, max_chi, low_thresh, high_thresh);
        } else {
            update_base_for_fixed_batch_kernel<<<grid_fix, BS, 0, dec->stream>>>(
                dec->d_raw_tensors, dec->d_data_offsets,
                dec->d_chi_l, dec->d_chi_r,
                dec->d_logical_positions,
                dec->d_batch_fixed, dec->d_batch_fixed_values,
                dec->d_batch_base, dec->d_batch_base_T, dec->d_base_offsets,
                K, dec->total_base_elements);
            forward_scan_batch_kernel<<<batch_size, BS, smem_scan, dec->stream>>>(
                dec->d_batch_base, dec->d_base_offsets,
                dec->d_chi_l, dec->d_chi_r,
                dec->d_site_type, dec->d_log_index,
                dec->d_batch_prefix, dec->d_batch_prefix_log,
                N, K, max_chi, dec->total_base_elements);
            backward_scan_batch_kernel<<<batch_size, BS, smem_scan, dec->stream>>>(
                dec->d_batch_base_T, dec->d_base_offsets,
                dec->d_chi_l, dec->d_chi_r,
                dec->d_site_type, dec->d_log_index,
                dec->d_batch_suffix, dec->d_batch_suffix_log,
                N, K, max_chi, dec->total_base_elements);
            compute_marginals_batch_kernel<<<grid_marg, BS, smem_marg, dec->stream>>>(
                dec->d_raw_tensors, dec->d_data_offsets,
                dec->d_chi_l, dec->d_chi_r,
                dec->d_logical_positions,
                dec->d_batch_prefix, dec->d_batch_prefix_log,
                dec->d_batch_suffix, dec->d_batch_suffix_log,
                dec->d_batch_fixed,
                dec->d_batch_marginals, dec->d_batch_decisions,
                K, max_chi, low_thresh, high_thresh);
        }
    }

    // Final pass
    {
        cudaMemcpyAsync(h_dec_all.data(), dec->d_batch_decisions,
                        (size_t)batch_size * K * sizeof(int), cudaMemcpyDeviceToHost, dec->stream);
        cudaStreamSynchronize(dec->stream);

        bool need_final = false;
        for (int idx = 0; idx < batch_size * K; idx++) {
            if (h_fix_all[idx]) continue;
            if (h_dec_all[idx] == -1) need_final = true;
            else { h_fix_all[idx] = 1; h_fval_all[idx] = h_dec_all[idx]; }
        }

        if (need_final) {
            cudaMemcpyAsync(dec->d_batch_fixed, h_fix_all.data(),
                            (size_t)batch_size * K * sizeof(int), cudaMemcpyHostToDevice, dec->stream);
            cudaMemcpyAsync(dec->d_batch_fixed_values, h_fval_all.data(),
                            (size_t)batch_size * K * sizeof(int), cudaMemcpyHostToDevice, dec->stream);

            dim3 grid_fix(K, batch_size);
            if (fp32) {
                update_base_for_fixed_fp32_batch_kernel<<<grid_fix, BS, 0, dec->stream>>>(
                    dec->d_raw_tensors, dec->d_data_offsets,
                    dec->d_chi_l, dec->d_chi_r,
                    dec->d_logical_positions,
                    dec->d_batch_fixed, dec->d_batch_fixed_values,
                    dec->d_batch_base_f32, dec->d_batch_base_T_f32, dec->d_base_offsets,
                    K, dec->total_base_elements);
                forward_scan_fp32_batch_kernel<<<batch_size, BS, smem_scan, dec->stream>>>(
                    dec->d_batch_base_f32, dec->d_base_offsets,
                    dec->d_chi_l, dec->d_chi_r,
                    dec->d_site_type, dec->d_log_index,
                    dec->d_batch_prefix_f32, dec->d_batch_prefix_log_f32,
                    N, K, max_chi, dec->total_base_elements);
                backward_scan_fp32_batch_kernel<<<batch_size, BS, smem_scan, dec->stream>>>(
                    dec->d_batch_base_T_f32, dec->d_base_offsets,
                    dec->d_chi_l, dec->d_chi_r,
                    dec->d_site_type, dec->d_log_index,
                    dec->d_batch_suffix_f32, dec->d_batch_suffix_log_f32,
                    N, K, max_chi, dec->total_base_elements);
                { /* Convert FP32 batch prefix/suffix to FP64 */
                    int _cvt_n = batch_size * K * max_chi;
                    int _cvt_bs = 256;
                    int _cvt_grid = (_cvt_n + _cvt_bs - 1) / _cvt_bs;
                    convert_vecs_f32_to_f64_kernel<<<_cvt_grid, _cvt_bs, 0, dec->stream>>>(
                        dec->d_batch_prefix_f32, dec->d_batch_prefix_log_f32,
                        dec->d_batch_prefix, dec->d_batch_prefix_log,
                        batch_size * K, max_chi, nullptr);
                    convert_vecs_f32_to_f64_kernel<<<_cvt_grid, _cvt_bs, 0, dec->stream>>>(
                        dec->d_batch_suffix_f32, dec->d_batch_suffix_log_f32,
                        dec->d_batch_suffix, dec->d_batch_suffix_log,
                        batch_size * K, max_chi, nullptr);
                }
                compute_marginals_batch_kernel<<<grid_marg, BS, smem_marg, dec->stream>>>(
                    dec->d_raw_tensors, dec->d_data_offsets,
                    dec->d_chi_l, dec->d_chi_r,
                    dec->d_logical_positions,
                    dec->d_batch_prefix, dec->d_batch_prefix_log,
                    dec->d_batch_suffix, dec->d_batch_suffix_log,
                    dec->d_batch_fixed,
                    dec->d_batch_marginals, dec->d_batch_decisions,
                    K, max_chi, 0.5, 0.5);
            } else {
                update_base_for_fixed_batch_kernel<<<grid_fix, BS, 0, dec->stream>>>(
                    dec->d_raw_tensors, dec->d_data_offsets,
                    dec->d_chi_l, dec->d_chi_r,
                    dec->d_logical_positions,
                    dec->d_batch_fixed, dec->d_batch_fixed_values,
                    dec->d_batch_base, dec->d_batch_base_T, dec->d_base_offsets,
                    K, dec->total_base_elements);
                forward_scan_batch_kernel<<<batch_size, BS, smem_scan, dec->stream>>>(
                    dec->d_batch_base, dec->d_base_offsets,
                    dec->d_chi_l, dec->d_chi_r,
                    dec->d_site_type, dec->d_log_index,
                    dec->d_batch_prefix, dec->d_batch_prefix_log,
                    N, K, max_chi, dec->total_base_elements);
                backward_scan_batch_kernel<<<batch_size, BS, smem_scan, dec->stream>>>(
                    dec->d_batch_base_T, dec->d_base_offsets,
                    dec->d_chi_l, dec->d_chi_r,
                    dec->d_site_type, dec->d_log_index,
                    dec->d_batch_suffix, dec->d_batch_suffix_log,
                    N, K, max_chi, dec->total_base_elements);
                compute_marginals_batch_kernel<<<grid_marg, BS, smem_marg, dec->stream>>>(
                    dec->d_raw_tensors, dec->d_data_offsets,
                    dec->d_chi_l, dec->d_chi_r,
                    dec->d_logical_positions,
                    dec->d_batch_prefix, dec->d_batch_prefix_log,
                    dec->d_batch_suffix, dec->d_batch_suffix_log,
                    dec->d_batch_fixed,
                    dec->d_batch_marginals, dec->d_batch_decisions,
                    K, max_chi, 0.5, 0.5);
            }
        }
    }

    cudaEventRecord(stop, dec->stream);
    cudaStreamSynchronize(dec->stream);
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    *latency_us = (double)elapsed_ms * 1000.0;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Download
    std::vector<int> h_d((size_t)batch_size * K);
    std::vector<double> h_m((size_t)batch_size * K);
    cudaMemcpy(h_d.data(), dec->d_batch_decisions, (size_t)batch_size * K * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_m.data(), dec->d_batch_marginals, (size_t)batch_size * K * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch_size * K; i++) {
        marginals[i] = h_m[i];
        decisions[i] = (h_d[i] >= 0) ? h_d[i] : (h_m[i] >= 0.5 ? 1 : 0);
    }

    return 0;
}

int mps_decoder_decode_profiled(MpsDecoderHandle handle,
                                const int* syndrome,
                                int* decisions,
                                double* marginals,
                                double* phase_us,
                                double low_thresh,
                                double high_thresh,
                                int max_rounds) {
    if (!handle) return -1;
    MpsDecoder* dec = (MpsDecoder*)handle;
    int N = dec->N, K = dec->K, M = dec->M;
    int max_chi = dec->max_chi, BS = dec->block_size;

    std::vector<double> syn_d(M);
    for (int i = 0; i < M; i++) syn_d[i] = (double)syndrome[i];

    CUDA_CHECK(cudaMemcpyAsync(dec->d_syndrome, syn_d.data(),
                               M * sizeof(double), cudaMemcpyHostToDevice, dec->stream));
    CUDA_CHECK(cudaMemsetAsync(dec->d_fixed, 0, K * sizeof(int), dec->stream));
    CUDA_CHECK(cudaMemsetAsync(dec->d_fixed_values, 0, K * sizeof(int), dec->stream));

    // Dispatch FP32 or FP64 scan block size
    int SBS = dec->use_fp32_scan ? dec->scan_block_size_f32 : dec->scan_block_size;
    size_t smem_scan = dec->use_fp32_scan ?
        (max_chi + SBS + 32) * sizeof(float) :
        (max_chi + SBS + 32) * sizeof(double);
    size_t smem_marg = 2 * BS * sizeof(double);

    // Create events for phase timing
    cudaEvent_t ev_start, ev_after_prepare, ev_after_forward, ev_after_backward,
                ev_after_marginals, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_after_prepare);
    cudaEventCreate(&ev_after_forward);
    cudaEventCreate(&ev_after_backward);
    cudaEventCreate(&ev_after_marginals);
    cudaEventCreate(&ev_end);

    cudaEventRecord(ev_start, dec->stream);

    // Phase 1: Prepare base matrices
    LAUNCH_PREPARE_BASE(dec, N, BS);
    cudaEventRecord(ev_after_prepare, dec->stream);

    if (!dec->use_sequential_scan) {
        cudaStreamWaitEvent(dec->stream2, ev_after_prepare, 0);
    }

    // Phase 2: Forward scan
    if (dec->use_fp32_scan) {
        forward_scan_fp32_kernel<<<1, SBS, smem_scan, dec->stream>>>(
            dec->d_base_matrices_f32, dec->d_base_offsets,
            dec->d_chi_l, dec->d_chi_r,
            dec->d_site_type, dec->d_log_index,
            dec->d_prefix_vecs_f32, dec->d_prefix_log_f32,
            N, K, max_chi, 0, -1,
            nullptr, nullptr);
    } else {
        forward_scan_kernel<<<1, SBS, smem_scan, dec->stream>>>(
            dec->d_base_matrices, dec->d_base_offsets,
            dec->d_chi_l, dec->d_chi_r,
            dec->d_site_type, dec->d_log_index,
            dec->d_prefix_vecs, dec->d_prefix_log,
            N, K, max_chi, 0, -1,
            nullptr, nullptr);
    }
    cudaEventRecord(ev_after_forward, dec->stream);

    // Phase 3: Backward scan
    if (dec->use_fp32_scan) {
        if (dec->use_sequential_scan) {
            backward_scan_fp32_kernel<<<1, SBS, smem_scan, dec->stream>>>(
                dec->d_base_matrices_T_f32, dec->d_base_offsets,
                dec->d_chi_l, dec->d_chi_r,
                dec->d_site_type, dec->d_log_index,
                dec->d_suffix_vecs_f32, dec->d_suffix_log_f32,
                N, K, max_chi, N - 1, -1,
                nullptr, nullptr);
            cudaEventRecord(ev_after_backward, dec->stream);
        } else {
            backward_scan_fp32_kernel<<<1, SBS, smem_scan, dec->stream2>>>(
                dec->d_base_matrices_T_f32, dec->d_base_offsets,
                dec->d_chi_l, dec->d_chi_r,
                dec->d_site_type, dec->d_log_index,
                dec->d_suffix_vecs_f32, dec->d_suffix_log_f32,
                N, K, max_chi, N - 1, -1,
                nullptr, nullptr);
            cudaEventRecord(ev_after_backward, dec->stream2);
            cudaStreamWaitEvent(dec->stream, ev_after_backward, 0);
        }
    } else {
        if (dec->use_sequential_scan) {
            backward_scan_kernel<<<1, SBS, smem_scan, dec->stream>>>(
                dec->d_base_matrices_T, dec->d_base_offsets,
                dec->d_chi_l, dec->d_chi_r,
                dec->d_site_type, dec->d_log_index,
                dec->d_suffix_vecs, dec->d_suffix_log,
                N, K, max_chi, N - 1, -1,
                nullptr, nullptr);
            cudaEventRecord(ev_after_backward, dec->stream);
        } else {
            backward_scan_kernel<<<1, SBS, smem_scan, dec->stream2>>>(
                dec->d_base_matrices_T, dec->d_base_offsets,
                dec->d_chi_l, dec->d_chi_r,
                dec->d_site_type, dec->d_log_index,
                dec->d_suffix_vecs, dec->d_suffix_log,
                N, K, max_chi, N - 1, -1,
                nullptr, nullptr);
            cudaEventRecord(ev_after_backward, dec->stream2);
            cudaStreamWaitEvent(dec->stream, ev_after_backward, 0);
        }
    }

    // Phase 4: Compute marginals
    LAUNCH_MARGINALS(dec, K, BS, smem_marg, low_thresh, high_thresh, nullptr);
    cudaEventRecord(ev_after_marginals, dec->stream);

    // ---- Conditional refinement loop (profiled: keep host sync for timing) ----
    int h_ctrl[6];

    for (int round = 0; round < max_rounds; round++) {
        refinement_check_kernel<<<1, 1, 0, dec->stream>>>(
            dec->d_decisions, dec->d_fixed, dec->d_fixed_values,
            dec->d_logical_positions, dec->d_refine_ctrl, K,
            nullptr, nullptr);

        cudaMemcpyAsync(h_ctrl, dec->d_refine_ctrl, 6 * sizeof(int),
                        cudaMemcpyDeviceToHost, dec->stream);
        cudaStreamSynchronize(dec->stream);

        if (!h_ctrl[0] || !h_ctrl[1]) break;

        int earliest_log = h_ctrl[2];
        int latest_log = h_ctrl[3];
        int earliest_site = h_ctrl[4];
        int latest_site = h_ctrl[5];

        LAUNCH_UPDATE_BASE_FIXED(dec, K, BS, nullptr);

        if (!dec->use_sequential_scan) {
            cudaEvent_t ev_update;
            cudaEventCreate(&ev_update);
            cudaEventRecord(ev_update, dec->stream);
            cudaStreamWaitEvent(dec->stream2, ev_update, 0);
            cudaEventDestroy(ev_update);
        }

        launch_scans(dec,
                     earliest_site, earliest_log,
                     latest_site, latest_log);

        LAUNCH_MARGINALS(dec, K, BS, smem_marg, low_thresh, high_thresh, nullptr);
    }

    // Final pass: force-decide remaining unknowns
    {
        refinement_check_kernel<<<1, 1, 0, dec->stream>>>(
            dec->d_decisions, dec->d_fixed, dec->d_fixed_values,
            dec->d_logical_positions, dec->d_refine_ctrl, K,
            nullptr, nullptr);

        cudaMemcpyAsync(h_ctrl, dec->d_refine_ctrl, 6 * sizeof(int),
                        cudaMemcpyDeviceToHost, dec->stream);
        cudaStreamSynchronize(dec->stream);

        bool need_final = (h_ctrl[0] != 0);

        if (need_final) {
            int earliest_site = h_ctrl[4];
            int earliest_log = h_ctrl[2];
            int latest_site = h_ctrl[5];
            int latest_log = h_ctrl[3];

            LAUNCH_UPDATE_BASE_FIXED(dec, K, BS, nullptr);

            if (!dec->use_sequential_scan) {
                cudaEvent_t ev_update;
                cudaEventCreate(&ev_update);
                cudaEventRecord(ev_update, dec->stream);
                cudaStreamWaitEvent(dec->stream2, ev_update, 0);
                cudaEventDestroy(ev_update);
            }

            if (earliest_site >= 0 && latest_site >= 0) {
                launch_scans(dec,
                             earliest_site, earliest_log,
                             latest_site, latest_log);
            } else {
                launch_scans(dec);
            }

            LAUNCH_MARGINALS(dec, K, BS, smem_marg, 0.5, 0.5, nullptr);
        }
    }

    cudaEventRecord(ev_end, dec->stream);
    cudaStreamSynchronize(dec->stream);
    if (!dec->use_sequential_scan) {
        cudaStreamSynchronize(dec->stream2);
    }

    // Compute per-phase times
    float ms;
    cudaEventElapsedTime(&ms, ev_start, ev_after_prepare);
    phase_us[0] = (double)ms * 1000.0;  // prepare_base
    cudaEventElapsedTime(&ms, ev_after_prepare, ev_after_forward);
    phase_us[1] = (double)ms * 1000.0;  // forward_scan
    cudaEventElapsedTime(&ms, ev_after_forward, ev_after_backward);
    phase_us[2] = (double)ms * 1000.0;  // backward_scan
    cudaEventElapsedTime(&ms, ev_after_backward, ev_after_marginals);
    phase_us[3] = (double)ms * 1000.0;  // marginals
    cudaEventElapsedTime(&ms, ev_after_marginals, ev_end);
    phase_us[4] = (double)ms * 1000.0;  // refinement_total
    cudaEventElapsedTime(&ms, ev_start, ev_end);
    phase_us[5] = (double)ms * 1000.0;  // total

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_after_prepare);
    cudaEventDestroy(ev_after_forward);
    cudaEventDestroy(ev_after_backward);
    cudaEventDestroy(ev_after_marginals);
    cudaEventDestroy(ev_end);

    // Download results
    std::vector<int> h_dec(K);
    std::vector<double> h_marg(K);
    cudaMemcpy(h_dec.data(), dec->d_decisions, K * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_marg.data(), dec->d_marginals, K * sizeof(double), cudaMemcpyDeviceToHost);

    for (int k = 0; k < K; k++) {
        marginals[k] = h_marg[k];
        decisions[k] = (h_dec[k] >= 0) ? h_dec[k] : (h_marg[k] >= 0.5 ? 1 : 0);
    }

    return 0;
}

int mps_decoder_get_num_logicals(MpsDecoderHandle handle) {
    return handle ? ((MpsDecoder*)handle)->K : 0;
}
int mps_decoder_get_num_syndromes(MpsDecoderHandle handle) {
    return handle ? ((MpsDecoder*)handle)->M : 0;
}
int mps_decoder_get_num_sites(MpsDecoderHandle handle) {
    return handle ? ((MpsDecoder*)handle)->N : 0;
}
int mps_decoder_get_max_chi(MpsDecoderHandle handle) {
    return handle ? ((MpsDecoder*)handle)->max_chi : 0;
}

void mps_decoder_set_fp32(MpsDecoderHandle handle, int enable) {
    if (!handle) return;
    MpsDecoder* dec = (MpsDecoder*)handle;
    dec->use_fp32_scan = (enable != 0);
    fprintf(stderr, "FP32 scan %s\n", dec->use_fp32_scan ? "ENABLED" : "DISABLED");
}

int mps_decoder_get_fp32(MpsDecoderHandle handle) {
    if (!handle) return 0;
    return ((MpsDecoder*)handle)->use_fp32_scan ? 1 : 0;
}

} // extern "C"
