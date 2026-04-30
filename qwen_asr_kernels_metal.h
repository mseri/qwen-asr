/*
 * qwen_asr_kernels_metal.h - Metal GPU kernels for Qwen3-ASR (Apple Silicon)
 *
 * C-callable interface; safe to include from plain C translation units.
 * Implementation is in qwen_asr_kernels_metal.m (Objective-C + MPS).
 */

#ifndef QWEN_ASR_KERNELS_METAL_H
#define QWEN_ASR_KERNELS_METAL_H

/* =========================================================================
 * Lifecycle
 * ========================================================================= */

void qwen_metal_init(void);
void qwen_metal_free(void);
int  qwen_metal_available(void);
int  qwen_metal_experimental_enabled(void);

/* =========================================================================
 * Dispatch policy
 * ========================================================================= */

int qwen_metal_should_offload(int M, int K, int N);

/* =========================================================================
 * GEMM (MPSMatrixMultiplication)
 * ========================================================================= */

void qwen_metal_gemm(float *C, const float *A, const float *B,
                     int M, int K, int N, int transpose_b);

/* =========================================================================
 * Custom compute kernels — availability queries
 * ========================================================================= */

int qwen_metal_has_gelu(void);
int qwen_metal_has_layer_norm(void);
int qwen_metal_has_rms_norm(void);
int qwen_metal_has_rms_norm_per_head(void);
int qwen_metal_has_swiglu(void);
int qwen_metal_has_bidir_attn(void);
int qwen_metal_has_causal_attn(void);
int qwen_metal_has_add_bias(void);
int qwen_metal_has_add_inplace(void);

/* =========================================================================
 * GELU activation (in-place)
 * ========================================================================= */

void qwen_metal_gelu(float *x, int n);

/* =========================================================================
 * LayerNorm: out[s,h] = weight[h] * (x[s,h] - mean) / sqrt(var + eps) + bias[h]
 * ========================================================================= */

void qwen_metal_layer_norm(float *out, const float *x, const float *weight,
                           const float *bias, int seq_len, int hidden, float eps);

/* =========================================================================
 * RMSNorm: out[s,h] = weight[h] * x[s,h] / sqrt(sum_sq/hidden + eps)
 * ========================================================================= */

void qwen_metal_rms_norm(float *out, const float *x, const float *weight,
                         int seq_len, int hidden, float eps);

/* =========================================================================
 * RMSNorm per head (in-place): each head_dim segment normalized independently
 * ========================================================================= */

void qwen_metal_rms_norm_per_head(float *x, const float *weight,
                                  int seq_len, int n_heads, int head_dim, float eps);

/* =========================================================================
 * SwiGLU: out[s,i] = SiLU(gate[s,i]) * up[s,i], interleaved layout
 * ========================================================================= */

void qwen_metal_swiglu(float *out, const float *gate_up, int seq_len, int intermediate);

/* =========================================================================
 * Windowed bidirectional attention (encoder)
 * ========================================================================= */

void qwen_metal_bidirectional_attention(float *out, const float *Q, const float *K,
                                        const float *V, int seq, int n_heads,
                                        int head_dim, float scale,
                                        const int *window_starts, int n_windows);

/* =========================================================================
 * Causal attention with GQA (decoder prefill)
 * ========================================================================= */

void qwen_metal_causal_attention(float *out, const float *Q, const float *K,
                                 const float *V, int seq_q, int seq_k,
                                 int n_heads, int n_kv_heads, int head_dim,
                                 float scale, int q_offset);

/* =========================================================================
 * Bias addition: y[s,d] += bias[d]
 * ========================================================================= */

void qwen_metal_add_bias(float *y, const float *bias, int seq_len, int dim);

/* =========================================================================
 * Residual addition (in-place): a[i] += b[i]
 * ========================================================================= */

void qwen_metal_add_inplace(float *a, const float *b, int n);

#endif /* QWEN_ASR_KERNELS_METAL_H */
