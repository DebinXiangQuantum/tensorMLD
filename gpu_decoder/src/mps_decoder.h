#ifndef MPS_DECODER_H
#define MPS_DECODER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle returned by mps_decoder_create(). */
typedef void* MpsDecoderHandle;

/**
 * Create a decoder from pre-compressed MPS chain data on disk.
 *
 * @param data_dir  Path to chain/ directory containing chain_meta.json,
 *                  site_*.npy, site_labels.npy, etc.
 * @return Opaque handle (NULL on failure).
 */
MpsDecoderHandle mps_decoder_create(const char* data_dir);

/**
 * Free all GPU and host memory associated with the decoder.
 */
void mps_decoder_destroy(MpsDecoderHandle handle);

/**
 * Decode a single syndrome vector.
 *
 * @param handle      Decoder handle from mps_decoder_create().
 * @param syndrome    Array of M integer syndrome values (0 or 1).
 * @param decisions   Output: array of K integers (0 or 1).
 * @param marginals   Output: array of K doubles (probabilities for value 1).
 * @param latency_us  Output: decode latency in microseconds (CUDA event timing).
 * @param low_thresh  Threshold below which a marginal is "confident 0" (e.g. 0.1).
 * @param high_thresh Threshold above which a marginal is "confident 1" (e.g. 0.9).
 * @param max_rounds  Maximum conditional refinement rounds.
 * @return 0 on success, nonzero on error.
 */
int mps_decoder_decode(MpsDecoderHandle handle,
                       const int* syndrome,
                       int* decisions,
                       double* marginals,
                       double* latency_us,
                       double low_thresh,
                       double high_thresh,
                       int max_rounds);

/**
 * Decode a batch of syndrome vectors.
 *
 * @param handle      Decoder handle.
 * @param syndromes   Row-major array of batch_size * M integers.
 * @param batch_size  Number of syndromes.
 * @param decisions   Output: row-major batch_size * K integers.
 * @param marginals   Output: row-major batch_size * K doubles.
 * @param latency_us  Output: total batch latency in microseconds.
 * @param low_thresh  Threshold for confident 0.
 * @param high_thresh Threshold for confident 1.
 * @param max_rounds  Maximum conditional refinement rounds.
 * @return 0 on success, nonzero on error.
 */
int mps_decoder_decode_batch(MpsDecoderHandle handle,
                             const int* syndromes,
                             int batch_size,
                             int* decisions,
                             double* marginals,
                             double* latency_us,
                             double low_thresh,
                             double high_thresh,
                             int max_rounds);

/**
 * Query decoder metadata.
 */
int mps_decoder_get_num_logicals(MpsDecoderHandle handle);
int mps_decoder_get_num_syndromes(MpsDecoderHandle handle);
int mps_decoder_get_num_sites(MpsDecoderHandle handle);
int mps_decoder_get_max_chi(MpsDecoderHandle handle);

/**
 * Decode with per-phase latency breakdown (for profiling).
 * phase_us: output array of 6 doubles:
 *   [0] prepare_base, [1] forward_scan, [2] backward_scan,
 *   [3] marginals, [4] refinement_total, [5] total
 */
int mps_decoder_decode_profiled(MpsDecoderHandle handle,
                                const int* syndrome,
                                int* decisions,
                                double* marginals,
                                double* phase_us,
                                double low_thresh,
                                double high_thresh,
                                int max_rounds);

/**
 * Enable or disable FP32 mixed-precision scan mode.
 * FP32 scan halves L2 bandwidth requirements, giving ~2x speedup on
 * memory-bound scan kernels. Marginals are still computed in FP64 using
 * the original raw tensors. Enabled by default.
 *
 * @param handle  Decoder handle.
 * @param enable  1 to enable FP32, 0 to disable (use FP64).
 */
void mps_decoder_set_fp32(MpsDecoderHandle handle, int enable);

/**
 * Query whether FP32 scan mode is enabled.
 * @return 1 if enabled, 0 if disabled.
 */
int mps_decoder_get_fp32(MpsDecoderHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* MPS_DECODER_H */
