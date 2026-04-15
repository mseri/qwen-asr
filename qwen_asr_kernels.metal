/*
 * qwen_asr_kernels.metal - Custom Metal shaders (Phase 2 extension point)
 *
 * This file is reserved for future custom Metal compute kernels, such as:
 *   - Fused bf16 GEMM with quantized weights
 *   - Batched multi-head attention with causal masking
 *   - RMS norm + gating fusion
 *
 * Phase 1 uses MPSMatrixMultiplication (see qwen_asr_kernels_metal.m).
 * Add custom kernels here when profiling reveals further bottlenecks.
 */

#include <metal_stdlib>
using namespace metal;

/* placeholder kernel to keep the file valid Metal source */
kernel void qwen_noop(uint tid [[thread_position_in_grid]]) {
    (void)tid;
}
