/*
 * qwen_asr_kernels.metal - Custom Metal compute shaders for Qwen3-ASR
 *
 * All kernels operate on float32. Layout conventions:
 *   - Sequences: [seq_len, hidden]  (row-major)
 *   - QKV:       [seq_len, n_heads * head_dim]
 *   - SwiGLU:    [seq_len, 2 * intermediate]  (interleaved gate/up)
 */

#include <metal_stdlib>
using namespace metal;

/* =========================================================================
 * GELU activation (in-place)
 *   x[i] = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 *   One thread per element.
 * ========================================================================= */

kernel void gelu(device float *x [[buffer(0)]],
                 constant int &n [[buffer(1)]],
                 uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(n)) return;
    float v = x[tid];
    float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    x[tid] = 0.5f * v * (1.0f + tanh(inner));
}

/* =========================================================================
 * LayerNorm — one threadgroup per sequence row
 *   out[s,h] = weight[h] * (x[s,h] - mean) / sqrt(var + eps) + bias[h]
 *
 * Uses threadgroup reduction for mean and variance.
 * ========================================================================= */

kernel void layer_norm(device float *out [[buffer(0)]],
                       device const float *x [[buffer(1)]],
                       device const float *weight [[buffer(2)]],
                       device const float *bias [[buffer(3)]],
                       constant int &hidden [[buffer(4)]],
                       constant float &eps [[buffer(5)]],
                       uint gid [[threadgroup_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint tpg [[threads_per_threadgroup]]) {
    threadgroup float shared_sum[1024];
    threadgroup float shared_var[1024];

    uint row_off = gid * uint(hidden);

    /* Pass 1: compute mean via reduction */
    float local_sum = 0.0f;
    for (uint i = lid; i < uint(hidden); i += tpg) {
        local_sum += x[row_off + i];
    }
    shared_sum[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        if (lid < stride) shared_sum[lid] += shared_sum[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared_sum[0] / float(hidden);

    /* Pass 2: compute variance */
    float local_var = 0.0f;
    for (uint i = lid; i < uint(hidden); i += tpg) {
        float d = x[row_off + i] - mean;
        local_var += d * d;
    }
    shared_var[lid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        if (lid < stride) shared_var[lid] += shared_var[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(shared_var[0] / float(hidden) + eps);

    /* Pass 3: normalize and scale+shift */
    for (uint i = lid; i < uint(hidden); i += tpg) {
        float normed = (x[row_off + i] - mean) * inv_std;
        out[row_off + i] = weight[i] * normed + bias[i];
    }
}

/* =========================================================================
 * RMSNorm — one threadgroup per sequence row
 *   out[s,h] = weight[h] * x[s,h] * rsqrt(sum(x²)/hidden + eps)
 * ========================================================================= */

kernel void rms_norm(device float *out [[buffer(0)]],
                     device const float *x [[buffer(1)]],
                     device const float *weight [[buffer(2)]],
                     constant int &hidden [[buffer(3)]],
                     constant float &eps [[buffer(4)]],
                     uint gid [[threadgroup_position_in_grid]],
                     uint lid [[thread_position_in_threadgroup]],
                     uint tpg [[threads_per_threadgroup]]) {
    threadgroup float shared[1024];

    uint row_off = gid * uint(hidden);

    float local_sq = 0.0f;
    for (uint i = lid; i < uint(hidden); i += tpg) {
        float v = x[row_off + i];
        local_sq += v * v;
    }
    shared[lid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        if (lid < stride) shared[lid] += shared[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms_inv = rsqrt(shared[0] / float(hidden) + eps);

    for (uint i = lid; i < uint(hidden); i += tpg) {
        out[row_off + i] = weight[i] * x[row_off + i] * rms_inv;
    }
}

/* =========================================================================
 * RMSNorm per head — one threadgroup per (seq, head) pair
 *   In-place: x[s, h*head_dim + d] *= weight[d] * rsqrt(...)
 * ========================================================================= */

kernel void rms_norm_per_head(device float *x [[buffer(0)]],
                              device const float *weight [[buffer(1)]],
                              constant int &n_heads [[buffer(2)]],
                              constant int &head_dim [[buffer(3)]],
                              constant float &eps [[buffer(4)]],
                              uint gid [[threadgroup_position_in_grid]],
                              uint lid [[thread_position_in_threadgroup]],
                              uint tpg [[threads_per_threadgroup]]) {
    threadgroup float shared[1024];

    uint hidden = uint(n_heads) * uint(head_dim);
    uint s = gid / uint(n_heads);
    uint h = gid % uint(n_heads);
    uint base = s * hidden + h * uint(head_dim);

    float local_sq = 0.0f;
    for (uint d = lid; d < uint(head_dim); d += tpg) {
        float v = x[base + d];
        local_sq += v * v;
    }
    shared[lid] = local_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        if (lid < stride) shared[lid] += shared[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms_inv = rsqrt(shared[0] / float(head_dim) + eps);

    for (uint d = lid; d < uint(head_dim); d += tpg) {
        x[base + d] = weight[d] * x[base + d] * rms_inv;
    }
}

/* =========================================================================
 * SwiGLU — one thread per output element
 *   gate_up layout: [seq, 2*intermediate] interleaved as [g0,u0,g1,u1,...]
 *   out[s,i] = silu(gate_up[s, 2*i]) * gate_up[s, 2*i+1]
 *   where silu(x) = x / (1 + exp(-x))
 * ========================================================================= */

kernel void swiglu(device float *out [[buffer(0)]],
                   device const float *gate_up [[buffer(1)]],
                   constant int &intermediate [[buffer(2)]],
                   uint tid [[thread_position_in_grid]]) {
    uint s = tid / uint(intermediate);
    uint i = tid % uint(intermediate);
    uint gu_base = s * 2u * uint(intermediate);
    uint gu_idx = gu_base + 2u * i;
    float gate = gate_up[gu_idx];
    float up   = gate_up[gu_idx + 1u];
    float silu_gate = gate / (1.0f + exp(-gate));
    out[tid] = silu_gate * up;
}

/* =========================================================================
 * Windowed bidirectional attention (encoder)
 *   One thread per (query_position, head) pair.
 *   Uses online softmax algorithm matching the CPU implementation.
 *   Q,K,V layout: [seq, n_heads * head_dim]
 * ========================================================================= */

kernel void bidirectional_attention(
    device float *out [[buffer(0)]],
    device const float *Q [[buffer(1)]],
    device const float *K [[buffer(2)]],
    device const float *V [[buffer(3)]],
    device const int *window_starts [[buffer(4)]],
    constant int &n_heads [[buffer(5)]],
    constant int &head_dim [[buffer(6)]],
    constant float &scale [[buffer(7)]],
    constant int &n_windows [[buffer(8)]],
    constant int &seq [[buffer(9)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint qi = pos.x;  /* query position */
    uint h  = pos.y;  /* head index */

    if (qi >= uint(seq) || h >= uint(n_heads)) return;

    uint hidden = uint(n_heads) * uint(head_dim);

    /* Find which window this query belongs to */
    int ws = 0, we = seq;
    for (int w = 0; w < n_windows; w++) {
        if (int(qi) >= window_starts[w] && int(qi) < window_starts[w + 1]) {
            ws = window_starts[w];
            we = window_starts[w + 1];
            break;
        }
    }

    /* Online softmax attention */
    float max_score = -1e30f;
    float sum_exp = 0.0f;

    /* Accumulate output in registers (head_dim typically 64 or 128) */
    uint q_off = qi * hidden + h * uint(head_dim);

    /* Zero output */
    for (uint d = 0; d < uint(head_dim); d++) {
        out[q_off + d] = 0.0f;
    }

    for (int j = ws; j < we; j++) {
        uint k_off = uint(j) * hidden + h * uint(head_dim);

        /* dot(q, k) */
        float score = 0.0f;
        for (uint d = 0; d < uint(head_dim); d++) {
            score += Q[q_off + d] * K[k_off + d];
        }
        score *= scale;

        if (score > max_score) {
            float correction = exp(max_score - score);
            sum_exp = sum_exp * correction + 1.0f;
            /* out = out * correction + v */
            for (uint d = 0; d < uint(head_dim); d++) {
                out[q_off + d] = out[q_off + d] * correction + V[k_off + d];
            }
            max_score = score;
        } else {
            float wt = exp(score - max_score);
            sum_exp += wt;
            for (uint d = 0; d < uint(head_dim); d++) {
                out[q_off + d] += wt * V[k_off + d];
            }
        }
    }

    /* Normalize */
    if (sum_exp > 0.0f) {
        float inv_sum = 1.0f / sum_exp;
        for (uint d = 0; d < uint(head_dim); d++) {
            out[q_off + d] *= inv_sum;
        }
    }
}

/* =========================================================================
 * Causal attention with GQA (decoder prefill)
 *   One thread per (query_position, head) pair.
 *   Online softmax, causal mask via q_offset.
 *   Q layout: [seq_q, n_heads * head_dim]
 *   K,V layout: [seq_k, n_kv_heads * head_dim]
 * ========================================================================= */

kernel void causal_attention(
    device float *out [[buffer(0)]],
    device const float *Q [[buffer(1)]],
    device const float *K [[buffer(2)]],
    device const float *V [[buffer(3)]],
    constant int &seq_q [[buffer(4)]],
    constant int &seq_k [[buffer(5)]],
    constant int &n_heads [[buffer(6)]],
    constant int &n_kv_heads [[buffer(7)]],
    constant int &head_dim [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    constant int &q_offset [[buffer(10)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint qi = pos.x;
    uint h  = pos.y;

    if (qi >= uint(seq_q) || h >= uint(n_heads)) return;

    int heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / uint(heads_per_kv);
    uint q_hidden = uint(n_heads) * uint(head_dim);
    uint kv_hidden = uint(n_kv_heads) * uint(head_dim);

    int global_pos = q_offset + int(qi);
    int k_end = global_pos + 1;
    if (k_end > seq_k) k_end = seq_k;

    uint q_off = qi * q_hidden + h * uint(head_dim);

    /* Zero output */
    for (uint d = 0; d < uint(head_dim); d++) {
        out[q_off + d] = 0.0f;
    }

    float max_score = -1e30f;
    float sum_exp = 0.0f;

    for (int j = 0; j < k_end; j++) {
        uint k_off = uint(j) * kv_hidden + kv_h * uint(head_dim);

        float score = 0.0f;
        for (uint d = 0; d < uint(head_dim); d++) {
            score += Q[q_off + d] * K[k_off + d];
        }
        score *= scale;

        if (score > max_score) {
            float correction = exp(max_score - score);
            sum_exp = sum_exp * correction + 1.0f;
            for (uint d = 0; d < uint(head_dim); d++) {
                out[q_off + d] = out[q_off + d] * correction + V[k_off + d];
            }
            max_score = score;
        } else {
            float wt = exp(score - max_score);
            sum_exp += wt;
            for (uint d = 0; d < uint(head_dim); d++) {
                out[q_off + d] += wt * V[k_off + d];
            }
        }
    }

    if (sum_exp > 0.0f) {
        float inv_sum = 1.0f / sum_exp;
        for (uint d = 0; d < uint(head_dim); d++) {
            out[q_off + d] *= inv_sum;
        }
    }
}

/* =========================================================================
 * Bias addition: y[s*dim + d] += bias[d]
 *   One thread per element.
 * ========================================================================= */

kernel void add_bias(device float *y [[buffer(0)]],
                     device const float *bias [[buffer(1)]],
                     constant int &dim [[buffer(2)]],
                     uint tid [[thread_position_in_grid]]) {
    y[tid] += bias[tid % uint(dim)];
}

/* =========================================================================
 * Element-wise in-place addition: a[i] += b[i]
 *   One thread per element.
 * ========================================================================= */

kernel void add_inplace(device float *a [[buffer(0)]],
                        device const float *b [[buffer(1)]],
                        uint tid [[thread_position_in_grid]]) {
    a[tid] += b[tid];
}
