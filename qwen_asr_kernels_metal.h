/*
 * qwen_asr_kernels_metal.h - Metal/MPS accelerated math kernels (Apple Silicon)
 *
 * C-callable interface; safe to include from plain C translation units.
 * Implementation is in qwen_asr_kernels_metal.m (Objective-C + MPS).
 *
 * Currently MPSMatrixMultiplication replaces large BLAS sgemm calls.
 * For possible future use qwen_asr_kernels.metal provides custom shader stubs.
 */

#ifndef QWEN_ASR_KERNELS_METAL_H
#define QWEN_ASR_KERNELS_METAL_H

/* -------------------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------------- */

/* Initialize Metal device and command queue. Idempotent; thread-safe.
 * Called automatically on first use but may be invoked early for clean
 * startup messaging. */
void qwen_metal_init(void);

/* Release Metal resources. Safe to call even if init was never called. */
void qwen_metal_free(void);

/* Returns 1 if Metal is available and initialized, 0 otherwise. */
int qwen_metal_available(void);

/* -------------------------------------------------------------------------
 * Dispatch policy
 * ---------------------------------------------------------------------- */

/* Returns 1 when it is worth dispatching a (M x K) x (K x N) matmul to the
 * GPU.  Returns 0 for single-row matvecs (decoder step) and tiny matrices
 * where CPU overhead would dominate. */
int qwen_metal_should_offload(int M, int K, int N);

/* -------------------------------------------------------------------------
 * GEMM
 * ---------------------------------------------------------------------- */

/*
 * General matrix multiply via MPSMatrixMultiplication.
 *
 * transpose_b == 0:  C[M,N] = A[M,K] @ B[K,N]   (NoTrans × NoTrans)
 * transpose_b == 1:  C[M,N] = A[M,K] @ B[N,K]^T (NoTrans × Trans)
 *
 * All matrices are row-major float32.
 */
void qwen_metal_gemm(float *C, const float *A, const float *B,
                     int M, int K, int N, int transpose_b);

#endif /* QWEN_ASR_KERNELS_METAL_H */
