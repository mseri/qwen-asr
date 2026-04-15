/*
 * qwen_asr_kernels_metal.m - Metal/MPS matmul kernels (Apple Silicon)
 *
 * Uses MPSMatrixMultiplication from the MetalPerformanceShaders framework to
 * accelerate large float32 GEMM operations. Single-row matvecs are left on
 * the CPU (NEON) because GPU dispatch overhead exceeds the compute time.
 *
 * Buffer strategy: three growable MTLBuffer slots (A, B, C) backed by shared
 * (CPU+GPU) memory. Buffers grow to the largest size seen; they are never
 * shrunk at runtime to avoid allocation churn. All calls come from a single
 * thread so no locking is needed for the buffer pool.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "qwen_asr_kernels_metal.h"

/* -------------------------------------------------------------------------
 * Global Metal state
 * ---------------------------------------------------------------------- */

static id<MTLDevice>       g_device       = nil;
static id<MTLCommandQueue> g_queue        = nil;
static int                 g_available    = 0;

/* Three-slot growable buffer pool: 0=A, 1=B, 2=C */
#define METAL_BUF_SLOTS 3
static id<MTLBuffer> g_buf[METAL_BUF_SLOTS];
static size_t        g_buf_cap[METAL_BUF_SLOTS];

/* dispatch_once token for thread-safe lazy init */
static dispatch_once_t g_init_once;

/* -------------------------------------------------------------------------
 * Internal helpers
 * ---------------------------------------------------------------------- */

/* Ensure slot idx can hold at least `bytes` bytes. Returns the buffer or
 * nil on allocation failure. */
static id<MTLBuffer> metal_ensure_buf(int idx, size_t bytes) {
    if (g_buf_cap[idx] >= bytes) return g_buf[idx];

    /* Round up to 4 MiB boundary to reduce reallocation frequency. */
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

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

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

        /* Pre-warm the MPS matrix multiply kernel so JIT compilation is
         * not charged to the first inference call. Uses a small matmul
         * at a size that passes the offload threshold (>= 2M FLOPs). */
        static const int WM = 64, WK = 256, WN = 256;  /* 2*64*256*256 = 8M FLOPs */
        float *wa = (float *)calloc((size_t)WM * WK, sizeof(float));
        if (wa) {
            float *wb = (float *)calloc((size_t)WN * WK, sizeof(float));
            float *wc = (float *)calloc((size_t)WM * WN, sizeof(float));
            if (wb && wc) {
                qwen_metal_gemm(wc, wa, wb, WM, WK, WN, /*transpose_b=*/1);
                qwen_metal_gemm(wc, wa, wb, WM, WK, WN, /*transpose_b=*/0);
            }
            free(wb);
            free(wc);
            free(wa);
        }
    });
}

void qwen_metal_free(void) {
    if (!g_available) return;
    for (int i = 0; i < METAL_BUF_SLOTS; i++) {
        g_buf[i]     = nil;
        g_buf_cap[i] = 0;
    }
    g_queue     = nil;
    g_device    = nil;
    g_available = 0;
}

int qwen_metal_available(void) {
    return g_available;
}

/*
 * Offload threshold: skip dispatch for single-row matvecs (decoder step,
 * seq_len == 1) and any op below ~2M FLOPs to avoid PCIe-equivalent overhead
 * even on unified memory.
 */
int qwen_metal_should_offload(int M, int K, int N) {
    if (!g_available)        return 0;
    if (M <= 1)              return 0;  /* decoder step → NEON */
    return (long long)M * K * N * 2 >= 2000000LL;
}

/*
 * qwen_metal_gemm - blocking GPU matmul
 *
 * transpose_b == 0: C[M,N] = A[M,K] @ B[K,N]
 * transpose_b == 1: C[M,N] = A[M,K] @ B[N,K]^T
 *
 * Copies A and B into shared MTLBuffers, dispatches the MPS kernel, waits
 * for completion, then copies C back to the caller's float*.
 */
void qwen_metal_gemm(float *C, const float *A, const float *B,
                     int M, int K, int N, int transpose_b) {
    if (!g_available) return;

    const size_t sA = (size_t)M * K * sizeof(float);
    const size_t sB = transpose_b
                      ? (size_t)N * K * sizeof(float)   /* B is [N,K] */
                      : (size_t)K * N * sizeof(float);  /* B is [K,N] */
    const size_t sC = (size_t)M * N * sizeof(float);

    id<MTLBuffer> bufA = metal_ensure_buf(0, sA);
    id<MTLBuffer> bufB = metal_ensure_buf(1, sB);
    id<MTLBuffer> bufC = metal_ensure_buf(2, sC);
    if (!bufA || !bufB || !bufC) return;  /* allocation failed; caller keeps CPU result */

    /* Copy inputs into shared buffers */
    memcpy([bufA contents], A, sA);
    memcpy([bufB contents], B, sB);

    /* Build MPS matrix descriptors (rowBytes must be multiple of 4 = sizeof float) */
    MPSMatrixDescriptor *descA =
        [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)M
                                              columns:(NSUInteger)K
                                             rowBytes:(NSUInteger)K * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    /* For transpose_b=1: B stored as [N,K]; MPS will read it transposed.
     * For transpose_b=0: B stored as [K,N]; MPS reads it normally.          */
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

    /* Copy result back */
    memcpy(C, [bufC contents], sC);
}
