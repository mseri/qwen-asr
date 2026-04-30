/*
 * qwen_asr_kernels_metal.m - Metal GPU kernels for Qwen3-ASR (Apple Silicon)
 *
 * Phase 1: MPSMatrixMultiplication for large f32 GEMM operations.
 * Phase 2: Custom Metal compute kernels for fused operations:
 *   - GELU activation
 *   - LayerNorm (with affine weight+bias)
 *   - RMSNorm (full-sequence and per-head variants)
 *   - SwiGLU activation (fused gate·up with SiLU)
 *   - Windowed bidirectional attention (encoder)
 *   - Causal attention with GQA (decoder prefill)
 *   - Bias addition (fused with GEMM dispatch)
 *   - Residual addition
 *
 * Buffer strategy: growable MTLBuffer pool backed by shared (CPU+GPU) memory.
 * On Apple Silicon, shared memory is unified — no PCIe copies, just cache
 * coherency. Buffers grow but never shrink to avoid allocation churn.
 *
 * All GPU dispatch is synchronous (commit + waitUntilCompleted) since the
 * engine is single-threaded. This keeps the caller's semantics unchanged.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "qwen_asr_kernels_metal.h"

/* =========================================================================
 * Global Metal state
 * ========================================================================= */

static id<MTLDevice>       g_device       = nil;
static id<MTLCommandQueue> g_queue        = nil;
static int                 g_available    = 0;
static int                 g_experimental = -1;

/* Compute pipeline states for custom kernels */
static id<MTLLibrary>      g_library      = nil;
static id<MTLComputePipelineState> g_pipe_gelu          = nil;
static id<MTLComputePipelineState> g_pipe_layer_norm    = nil;
static id<MTLComputePipelineState> g_pipe_rms_norm      = nil;
static id<MTLComputePipelineState> g_pipe_rms_norm_head = nil;
static id<MTLComputePipelineState> g_pipe_swiglu        = nil;
static id<MTLComputePipelineState> g_pipe_bidir_attn    = nil;
static id<MTLComputePipelineState> g_pipe_causal_attn   = nil;
static id<MTLComputePipelineState> g_pipe_add_bias      = nil;
static id<MTLComputePipelineState> g_pipe_add_inplace   = nil;

/* Growable buffer pool. Slots:
 * 0 = A (left matrix / input)
 * 1 = B (right matrix / weights)
 * 2 = C (output matrix / result)
 * 3-7 = scratch (Q, K, V, attention output, misc) */
#define METAL_BUF_SLOTS 8
static id<MTLBuffer> g_buf[METAL_BUF_SLOTS];
static size_t        g_buf_cap[METAL_BUF_SLOTS];

static dispatch_once_t g_init_once;

/* =========================================================================
 * Internal helpers
 * ========================================================================= */

static id<MTLBuffer> metal_wrap_pointer(void *ptr, size_t bytes) {
    uintptr_t addr = (uintptr_t)ptr;
    size_t page = (size_t)getpagesize();
    size_t aligned_bytes = (bytes + page - 1) & ~(page - 1);
    if ((addr & (page - 1)) != 0) return nil;
    if (aligned_bytes > (size_t)[g_device maxBufferLength]) return nil;
    return [g_device newBufferWithBytesNoCopy:ptr
                                       length:aligned_bytes
                                      options:MTLResourceStorageModeShared
                                  deallocator:nil];
}

static id<MTLBuffer> metal_ensure_buf(int idx, size_t bytes) {
    if (g_buf_cap[idx] >= bytes) return g_buf[idx];

    static const size_t GRAIN = 4u * 1024u * 1024u;
    size_t new_cap = (bytes + GRAIN - 1) & ~(GRAIN - 1);
    if (new_cap < GRAIN) new_cap = GRAIN;

    id<MTLBuffer> buf = [g_device newBufferWithLength:new_cap
                                              options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "[metal] buffer allocation failed (%zu bytes)\n", new_cap);
        return nil;
    }
    g_buf[idx]     = buf;
    g_buf_cap[idx] = new_cap;
    return buf;
}

/* Wrap caller's pointer in a no-copy shared MTLBuffer.
 * The caller's memory must be page-aligned for newBufferWithBytesNoCopy;
 * if not, fall back to copy into a pool buffer. */
static id<MTLBuffer> metal_wrap_or_copy(const float *ptr, size_t bytes, int pool_slot) {
    id<MTLBuffer> buf = metal_wrap_pointer((void *)ptr, bytes);
    if (buf) return buf;
    /* Fallback: copy into growable pool buffer */
    buf = metal_ensure_buf(pool_slot, bytes);
    if (buf) memcpy([buf contents], ptr, bytes);
    return buf;
}

/* Create a compute pipeline for a named kernel function. Returns nil on failure. */
static id<MTLComputePipelineState> metal_make_pipeline(NSString *name) {
    id<MTLFunction> fn = [g_library newFunctionWithName:name];
    if (!fn) {
        fprintf(stderr, "[metal] kernel function '%s' not found\n", [name UTF8String]);
        return nil;
    }
    NSError *err = nil;
    id<MTLComputePipelineState> pso = [g_device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "[metal] pipeline '%s' failed: %s\n",
                [name UTF8String], [[err localizedDescription] UTF8String]);
    }
    return pso;
}

/* Dispatch a 1D compute kernel over `count` threads. */
static inline void metal_dispatch_1d(id<MTLComputeCommandEncoder> enc,
                                     id<MTLComputePipelineState> pso,
                                     NSUInteger count) {
    NSUInteger tpg = pso.maxTotalThreadsPerThreadgroup;
    if (tpg > count) tpg = count;
    [enc setComputePipelineState:pso];
    [enc dispatchThreads:MTLSizeMake(count, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
}

/* =========================================================================
 * Public API — lifecycle
 * ========================================================================= */

void qwen_metal_init(void) {
    dispatch_once(&g_init_once, ^{
        for (int i = 0; i < METAL_BUF_SLOTS; i++) {
            g_buf[i]     = nil;
            g_buf_cap[i] = 0;
        }

        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "[metal] no Metal device available\n");
            return;
        }

        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            fprintf(stderr, "[metal] failed to create command queue\n");
            g_device = nil;
            return;
        }

        g_available = 1;
        fprintf(stderr, "[metal] using device: %s\n",
                [[g_device name] UTF8String]);

        /* Load compiled metallib from same directory as executable */
        NSString *path = [[NSBundle mainBundle] pathForResource:@"qwen_asr_kernels"
                                                        ofType:@"metallib"];
        if (!path) {
            /* Try current directory */
            path = @"qwen_asr_kernels.metallib";
        }
        NSError *err = nil;
        g_library = [g_device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];

        if (!g_library) {
            /* Try compiling from .metal source at runtime as fallback */
            NSString *srcPath = @"qwen_asr_kernels.metal";
            NSString *src = [NSString stringWithContentsOfFile:srcPath
                                                     encoding:NSUTF8StringEncoding error:nil];
            if (src) {
                MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
                opts.mathMode = MTLMathModeRelaxed;
#else
                opts.fastMathEnabled = YES;
#endif
                g_library = [g_device newLibraryWithSource:src options:opts error:&err];
            }
        }

        if (g_library) {
            g_pipe_gelu          = metal_make_pipeline(@"gelu");
            g_pipe_layer_norm    = metal_make_pipeline(@"layer_norm");
            g_pipe_rms_norm      = metal_make_pipeline(@"rms_norm");
            g_pipe_rms_norm_head = metal_make_pipeline(@"rms_norm_per_head");
            g_pipe_swiglu        = metal_make_pipeline(@"swiglu");
            g_pipe_bidir_attn    = metal_make_pipeline(@"bidirectional_attention");
            g_pipe_causal_attn   = metal_make_pipeline(@"causal_attention");
            g_pipe_add_bias      = metal_make_pipeline(@"add_bias");
            g_pipe_add_inplace   = metal_make_pipeline(@"add_inplace");
            if (g_pipe_gelu)
                fprintf(stderr, "[metal] custom compute kernels loaded\n");
        } else {
            fprintf(stderr, "[metal] no metallib found — using MPS only\n");
        }

        /* Pre-warm MPS GEMM kernel (JIT compilation cost) */
        static const int WM = 64, WK = 256, WN = 256;
        float *wa = (float *)calloc((size_t)WM * WK, sizeof(float));
        if (wa) {
            float *wb = (float *)calloc((size_t)WN * WK, sizeof(float));
            float *wc = (float *)calloc((size_t)WM * WN, sizeof(float));
            if (wb && wc) {
                qwen_metal_gemm(wc, wa, wb, WM, WK, WN, 1);
                qwen_metal_gemm(wc, wa, wb, WM, WK, WN, 0);
            }
            free(wb); free(wc); free(wa);
        }
    });
}

void qwen_metal_free(void) {
    if (!g_available) return;
    for (int i = 0; i < METAL_BUF_SLOTS; i++) {
        g_buf[i]     = nil;
        g_buf_cap[i] = 0;
    }
    g_pipe_gelu = nil; g_pipe_layer_norm = nil; g_pipe_rms_norm = nil;
    g_pipe_rms_norm_head = nil; g_pipe_swiglu = nil;
    g_pipe_bidir_attn = nil; g_pipe_causal_attn = nil;
    g_pipe_add_bias = nil; g_pipe_add_inplace = nil;
    g_library   = nil;
    g_queue     = nil;
    g_device    = nil;
    g_available = 0;
    g_experimental = -1;
}

int qwen_metal_available(void) {
    return g_available;
}

int qwen_metal_experimental_enabled(void) {
    if (g_experimental < 0) {
        const char *env = getenv("QWEN_MPS_EXPERIMENTAL");
        g_experimental = (env && env[0] && strcmp(env, "0") != 0) ? 1 : 0;
    }
    return g_experimental;
}

/* =========================================================================
 * Offload policy
 * ========================================================================= */

int qwen_metal_should_offload(int M, int K, int N) {
    if (!g_available)   return 0;
    if (M <= 1)         return 0;
    return (long long)M * K * N * 2 >= 2000000LL;
}

/* =========================================================================
 * GEMM via MPS
 * ========================================================================= */

void qwen_metal_gemm(float *C, const float *A, const float *B,
                     int M, int K, int N, int transpose_b) {
    if (!g_available) return;

    const size_t sA = (size_t)M * K * sizeof(float);
    const size_t sB = transpose_b ? (size_t)N * K * sizeof(float)
                                  : (size_t)K * N * sizeof(float);
    const size_t sC = (size_t)M * N * sizeof(float);

    id<MTLBuffer> bufA = metal_wrap_or_copy(A, sA, 0);
    id<MTLBuffer> bufB = metal_wrap_or_copy(B, sB, 1);
    id<MTLBuffer> bufC = metal_ensure_buf(2, sC);
    if (!bufA || !bufB || !bufC) return;

    MPSMatrixDescriptor *descA =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)M
                                              columns:(NSUInteger)K
                                             rowBytes:(NSUInteger)K * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    NSUInteger bRows = transpose_b ? (NSUInteger)N : (NSUInteger)K;
    NSUInteger bCols = transpose_b ? (NSUInteger)K : (NSUInteger)N;
    MPSMatrixDescriptor *descB =
        [MPSMatrixDescriptor matrixDescriptorWithRows:bRows
                                              columns:bCols
                                             rowBytes:bCols * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)M
                                              columns:(NSUInteger)N
                                             rowBytes:(NSUInteger)N * sizeof(float)
                                             dataType:MPSDataTypeFloat32];

    MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
    MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
    MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

    MPSMatrixMultiplication *mm =
        [[MPSMatrixMultiplication alloc]
            initWithDevice:g_device
             transposeLeft:NO
            transposeRight:(transpose_b ? YES : NO)
               resultRows:(NSUInteger)M
            resultColumns:(NSUInteger)N
          interiorColumns:(NSUInteger)K
                    alpha:1.0
                     beta:0.0];

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    [mm encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(C, [bufC contents], sC);
}

/* =========================================================================
 * Custom compute kernels — GELU
 * ========================================================================= */

int qwen_metal_has_gelu(void) { return g_pipe_gelu != nil; }

void qwen_metal_gelu(float *x, int n) {
    if (!g_pipe_gelu || !g_available) return;
    size_t bytes = (size_t)n * sizeof(float);
    id<MTLBuffer> buf = metal_wrap_or_copy(x, bytes, 3);
    if (!buf) return;
    /* If we copied (not no-copy), need to know for writeback */
    int is_nocopy = ([buf contents] == (void *)x);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBytes:&n length:sizeof(int) atIndex:1];
    metal_dispatch_1d(enc, g_pipe_gelu, (NSUInteger)n);
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (!is_nocopy) memcpy(x, [buf contents], bytes);
}

/* =========================================================================
 * Custom compute kernels — LayerNorm
 * ========================================================================= */

int qwen_metal_has_layer_norm(void) { return g_pipe_layer_norm != nil; }

void qwen_metal_layer_norm(float *out, const float *x, const float *weight,
                           const float *bias, int seq_len, int hidden, float eps) {
    if (!g_pipe_layer_norm || !g_available) return;
    size_t sz = (size_t)seq_len * hidden * sizeof(float);
    size_t wsz = (size_t)hidden * sizeof(float);

    id<MTLBuffer> buf_x = metal_wrap_or_copy(x, sz, 3);
    id<MTLBuffer> buf_o = metal_ensure_buf(4, sz);
    id<MTLBuffer> buf_w = metal_wrap_or_copy(weight, wsz, 5);
    id<MTLBuffer> buf_b = metal_wrap_or_copy(bias, wsz, 6);
    if (!buf_x || !buf_o || !buf_w || !buf_b) return;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_pipe_layer_norm];
    [enc setBuffer:buf_o offset:0 atIndex:0];
    [enc setBuffer:buf_x offset:0 atIndex:1];
    [enc setBuffer:buf_w offset:0 atIndex:2];
    [enc setBuffer:buf_b offset:0 atIndex:3];
    [enc setBytes:&hidden length:sizeof(int) atIndex:4];
    [enc setBytes:&eps length:sizeof(float) atIndex:5];
    /* One threadgroup per sequence row; threadgroup size = min(hidden, max_tpg) */
    NSUInteger tpg = g_pipe_layer_norm.maxTotalThreadsPerThreadgroup;
    if (tpg > (NSUInteger)hidden) tpg = (NSUInteger)hidden;
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)seq_len, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(out, [buf_o contents], sz);
}

/* =========================================================================
 * Custom compute kernels — RMSNorm
 * ========================================================================= */

int qwen_metal_has_rms_norm(void) { return g_pipe_rms_norm != nil; }

void qwen_metal_rms_norm(float *out, const float *x, const float *weight,
                         int seq_len, int hidden, float eps) {
    if (!g_pipe_rms_norm || !g_available) return;
    size_t sz = (size_t)seq_len * hidden * sizeof(float);
    size_t wsz = (size_t)hidden * sizeof(float);

    id<MTLBuffer> buf_x = metal_wrap_or_copy(x, sz, 3);
    id<MTLBuffer> buf_o = metal_ensure_buf(4, sz);
    id<MTLBuffer> buf_w = metal_wrap_or_copy(weight, wsz, 5);
    if (!buf_x || !buf_o || !buf_w) return;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_pipe_rms_norm];
    [enc setBuffer:buf_o offset:0 atIndex:0];
    [enc setBuffer:buf_x offset:0 atIndex:1];
    [enc setBuffer:buf_w offset:0 atIndex:2];
    [enc setBytes:&hidden length:sizeof(int) atIndex:3];
    [enc setBytes:&eps length:sizeof(float) atIndex:4];
    NSUInteger tpg = g_pipe_rms_norm.maxTotalThreadsPerThreadgroup;
    if (tpg > (NSUInteger)hidden) tpg = (NSUInteger)hidden;
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)seq_len, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(out, [buf_o contents], sz);
}

/* =========================================================================
 * Custom compute kernels — RMSNorm per head
 * ========================================================================= */

int qwen_metal_has_rms_norm_per_head(void) { return g_pipe_rms_norm_head != nil; }

void qwen_metal_rms_norm_per_head(float *x, const float *weight,
                                  int seq_len, int n_heads, int head_dim, float eps) {
    if (!g_pipe_rms_norm_head || !g_available) return;
    int hidden = n_heads * head_dim;
    size_t sz = (size_t)seq_len * hidden * sizeof(float);
    size_t wsz = (size_t)head_dim * sizeof(float);

    id<MTLBuffer> buf_x = metal_wrap_or_copy(x, sz, 3);
    id<MTLBuffer> buf_w = metal_wrap_or_copy(weight, wsz, 5);
    int is_nocopy = ([buf_x contents] == (void *)x);
    if (!buf_x || !buf_w) return;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_pipe_rms_norm_head];
    [enc setBuffer:buf_x offset:0 atIndex:0];
    [enc setBuffer:buf_w offset:0 atIndex:1];
    [enc setBytes:&n_heads length:sizeof(int) atIndex:2];
    [enc setBytes:&head_dim length:sizeof(int) atIndex:3];
    [enc setBytes:&eps length:sizeof(float) atIndex:4];
    /* One threadgroup per (seq_pos, head) pair */
    NSUInteger total_heads = (NSUInteger)seq_len * (NSUInteger)n_heads;
    NSUInteger tpg = g_pipe_rms_norm_head.maxTotalThreadsPerThreadgroup;
    if (tpg > (NSUInteger)head_dim) tpg = (NSUInteger)head_dim;
    [enc dispatchThreadgroups:MTLSizeMake(total_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (!is_nocopy) memcpy(x, [buf_x contents], sz);
}

/* =========================================================================
 * Custom compute kernels — SwiGLU
 * ========================================================================= */

int qwen_metal_has_swiglu(void) { return g_pipe_swiglu != nil; }

void qwen_metal_swiglu(float *out, const float *gate_up, int seq_len, int intermediate) {
    if (!g_pipe_swiglu || !g_available) return;
    size_t in_sz = (size_t)seq_len * 2 * intermediate * sizeof(float);
    size_t out_sz = (size_t)seq_len * intermediate * sizeof(float);

    id<MTLBuffer> buf_gu = metal_wrap_or_copy(gate_up, in_sz, 3);
    id<MTLBuffer> buf_o  = metal_ensure_buf(4, out_sz);
    if (!buf_gu || !buf_o) return;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:buf_o offset:0 atIndex:0];
    [enc setBuffer:buf_gu offset:0 atIndex:1];
    [enc setBytes:&intermediate length:sizeof(int) atIndex:2];
    metal_dispatch_1d(enc, g_pipe_swiglu, (NSUInteger)seq_len * (NSUInteger)intermediate);
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(out, [buf_o contents], out_sz);
}

/* =========================================================================
 * Custom compute kernels — Windowed bidirectional attention (encoder)
 * ========================================================================= */

int qwen_metal_has_bidir_attn(void) { return g_pipe_bidir_attn != nil; }

void qwen_metal_bidirectional_attention(float *out, const float *Q, const float *K,
                                        const float *V, int seq, int n_heads,
                                        int head_dim, float scale,
                                        const int *window_starts, int n_windows) {
    if (!g_pipe_bidir_attn || !g_available) return;
    int hidden = n_heads * head_dim;
    size_t qkv_sz = (size_t)seq * hidden * sizeof(float);
    size_t win_sz = (size_t)(n_windows + 1) * sizeof(int);

    id<MTLBuffer> buf_q   = metal_wrap_or_copy(Q, qkv_sz, 0);
    id<MTLBuffer> buf_k   = metal_wrap_or_copy(K, qkv_sz, 1);
    id<MTLBuffer> buf_v   = metal_wrap_or_copy(V, qkv_sz, 3);
    id<MTLBuffer> buf_o   = metal_ensure_buf(2, qkv_sz);
    id<MTLBuffer> buf_win = metal_ensure_buf(4, win_sz);
    if (!buf_q || !buf_k || !buf_v || !buf_o || !buf_win) return;
    memcpy([buf_win contents], window_starts, win_sz);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_pipe_bidir_attn];
    [enc setBuffer:buf_o   offset:0 atIndex:0];
    [enc setBuffer:buf_q   offset:0 atIndex:1];
    [enc setBuffer:buf_k   offset:0 atIndex:2];
    [enc setBuffer:buf_v   offset:0 atIndex:3];
    [enc setBuffer:buf_win offset:0 atIndex:4];
    [enc setBytes:&n_heads    length:sizeof(int) atIndex:5];
    [enc setBytes:&head_dim   length:sizeof(int) atIndex:6];
    [enc setBytes:&scale      length:sizeof(float) atIndex:7];
    [enc setBytes:&n_windows  length:sizeof(int) atIndex:8];
    [enc setBytes:&seq        length:sizeof(int) atIndex:9];
    /* Dispatch one thread per (query_position, head) pair */
    [enc dispatchThreads:MTLSizeMake((NSUInteger)seq, (NSUInteger)n_heads, 1)
   threadsPerThreadgroup:MTLSizeMake(1, (NSUInteger)(n_heads < 16 ? n_heads : 16), 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(out, [buf_o contents], qkv_sz);
}

/* =========================================================================
 * Custom compute kernels — Causal attention with GQA (decoder prefill)
 * ========================================================================= */

int qwen_metal_has_causal_attn(void) { return g_pipe_causal_attn != nil; }

void qwen_metal_causal_attention(float *out, const float *Q, const float *K,
                                 const float *V, int seq_q, int seq_k,
                                 int n_heads, int n_kv_heads, int head_dim,
                                 float scale, int q_offset) {
    if (!g_pipe_causal_attn || !g_available) return;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;
    size_t q_sz = (size_t)seq_q * q_hidden * sizeof(float);
    size_t k_sz = (size_t)seq_k * kv_hidden * sizeof(float);
    size_t v_sz = k_sz;

    id<MTLBuffer> buf_q = metal_wrap_or_copy(Q, q_sz, 0);
    id<MTLBuffer> buf_k = metal_wrap_or_copy(K, k_sz, 1);
    id<MTLBuffer> buf_v = metal_wrap_or_copy(V, v_sz, 3);
    id<MTLBuffer> buf_o = metal_ensure_buf(2, q_sz);
    if (!buf_q || !buf_k || !buf_v || !buf_o) return;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_pipe_causal_attn];
    [enc setBuffer:buf_o offset:0 atIndex:0];
    [enc setBuffer:buf_q offset:0 atIndex:1];
    [enc setBuffer:buf_k offset:0 atIndex:2];
    [enc setBuffer:buf_v offset:0 atIndex:3];
    [enc setBytes:&seq_q      length:sizeof(int) atIndex:4];
    [enc setBytes:&seq_k      length:sizeof(int) atIndex:5];
    [enc setBytes:&n_heads    length:sizeof(int) atIndex:6];
    [enc setBytes:&n_kv_heads length:sizeof(int) atIndex:7];
    [enc setBytes:&head_dim   length:sizeof(int) atIndex:8];
    [enc setBytes:&scale      length:sizeof(float) atIndex:9];
    [enc setBytes:&q_offset   length:sizeof(int) atIndex:10];
    /* Dispatch one thread per (query_position, head) pair */
    [enc dispatchThreads:MTLSizeMake((NSUInteger)seq_q, (NSUInteger)n_heads, 1)
   threadsPerThreadgroup:MTLSizeMake(1, (NSUInteger)(n_heads < 16 ? n_heads : 16), 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    memcpy(out, [buf_o contents], q_sz);
}

/* =========================================================================
 * Custom compute kernels — Bias addition
 * ========================================================================= */

int qwen_metal_has_add_bias(void) { return g_pipe_add_bias != nil; }

void qwen_metal_add_bias(float *y, const float *bias, int seq_len, int dim) {
    if (!g_pipe_add_bias || !g_available) return;
    size_t sz = (size_t)seq_len * dim * sizeof(float);
    size_t bsz = (size_t)dim * sizeof(float);
    id<MTLBuffer> buf_y = metal_wrap_or_copy(y, sz, 3);
    id<MTLBuffer> buf_b = metal_wrap_or_copy(bias, bsz, 5);
    int is_nocopy = ([buf_y contents] == (void *)y);
    if (!buf_y || !buf_b) return;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:buf_y offset:0 atIndex:0];
    [enc setBuffer:buf_b offset:0 atIndex:1];
    [enc setBytes:&dim length:sizeof(int) atIndex:2];
    metal_dispatch_1d(enc, g_pipe_add_bias, (NSUInteger)seq_len * (NSUInteger)dim);
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (!is_nocopy) memcpy(y, [buf_y contents], sz);
}

/* =========================================================================
 * Custom compute kernels — Residual addition (in-place a += b)
 * ========================================================================= */

int qwen_metal_has_add_inplace(void) { return g_pipe_add_inplace != nil; }

void qwen_metal_add_inplace(float *a, const float *b, int n) {
    if (!g_pipe_add_inplace || !g_available) return;
    size_t sz = (size_t)n * sizeof(float);
    id<MTLBuffer> buf_a = metal_wrap_or_copy(a, sz, 3);
    id<MTLBuffer> buf_b = metal_wrap_or_copy(b, sz, 4);
    int is_nocopy = ([buf_a contents] == (void *)a);
    if (!buf_a || !buf_b) return;

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setBuffer:buf_a offset:0 atIndex:0];
    [enc setBuffer:buf_b offset:0 atIndex:1];
    metal_dispatch_1d(enc, g_pipe_add_inplace, (NSUInteger)n);
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (!is_nocopy) memcpy(a, [buf_a contents], sz);
}
