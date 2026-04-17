/*
 * qwen_asr_kernels_avx.c - x86 SIMD hot kernels (AVX2+FMA, with AVX-512 when available)
 *
 * bf16→f32 conversion: load 16 uint16 → zero-extend to 32-bit → shift left 16.
 * Uses AVX-512F+BW for 16-wide bf16 matvec/argmax (dominant for seq_len==1), and
 * AVX2+FMA / AVX-512F for f32 attention helpers (which operate on cache-resident data).
 *
 * The bf16 matvec processes 4 output rows simultaneously to reduce instruction
 * overhead and improve out-of-order execution on memory-bound workloads.
 */

#include "qwen_asr_kernels_impl.h"

#if defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>
#include <string.h>

/* =====================================================================
 * BF16 matvec - uses AVX-512F+BW when available, AVX2 fallback otherwise
 * ===================================================================== */

#if defined(__AVX512F__) && defined(__AVX512BW__)

void qwen_bf16_matvec_fused_avx(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim) {
    int o = 0;

    /* Process 4 output rows at a time */
    for (; o + 3 < out_dim; o += 4) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = w0 + in_dim;
        const uint16_t *w2 = w1 + in_dim;
        const uint16_t *w3 = w2 + in_dim;

        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        __m512 a4 = _mm512_setzero_ps(), a5 = _mm512_setzero_ps();
        __m512 a6 = _mm512_setzero_ps(), a7 = _mm512_setzero_ps();
        int k = 0;

        for (; k + 32 <= in_dim; k += 32) {
            __m512 xv0 = _mm512_loadu_ps(x + k);
            __m512 xv1 = _mm512_loadu_ps(x + k + 16);

            /* Row 0 */
            __m256i r0a = _mm256_loadu_si256((const __m256i *)(w0 + k));
            __m256i r0b = _mm256_loadu_si256((const __m256i *)(w0 + k + 16));
            a0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0a), 16)), xv0, a0);
            a1 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0b), 16)), xv1, a1);

            /* Row 1 */
            __m256i r1a = _mm256_loadu_si256((const __m256i *)(w1 + k));
            __m256i r1b = _mm256_loadu_si256((const __m256i *)(w1 + k + 16));
            a2 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1a), 16)), xv0, a2);
            a3 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1b), 16)), xv1, a3);

            /* Row 2 */
            __m256i r2a = _mm256_loadu_si256((const __m256i *)(w2 + k));
            __m256i r2b = _mm256_loadu_si256((const __m256i *)(w2 + k + 16));
            a4 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r2a), 16)), xv0, a4);
            a5 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r2b), 16)), xv1, a5);

            /* Row 3 */
            __m256i r3a = _mm256_loadu_si256((const __m256i *)(w3 + k));
            __m256i r3b = _mm256_loadu_si256((const __m256i *)(w3 + k + 16));
            a6 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r3a), 16)), xv0, a6);
            a7 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r3b), 16)), xv1, a7);
        }

        for (; k + 16 <= in_dim; k += 16) {
            __m512 xv = _mm512_loadu_ps(x + k);
            __m256i r0 = _mm256_loadu_si256((const __m256i *)(w0 + k));
            __m256i r1 = _mm256_loadu_si256((const __m256i *)(w1 + k));
            __m256i r2 = _mm256_loadu_si256((const __m256i *)(w2 + k));
            __m256i r3 = _mm256_loadu_si256((const __m256i *)(w3 + k));
            a0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0), 16)), xv, a0);
            a2 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1), 16)), xv, a2);
            a4 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r2), 16)), xv, a4);
            a6 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r3), 16)), xv, a6);
        }

        float s0 = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1)) + (bias ? bias[o]   : 0.0f);
        float s1 = _mm512_reduce_add_ps(_mm512_add_ps(a2, a3)) + (bias ? bias[o+1] : 0.0f);
        float s2 = _mm512_reduce_add_ps(_mm512_add_ps(a4, a5)) + (bias ? bias[o+2] : 0.0f);
        float s3 = _mm512_reduce_add_ps(_mm512_add_ps(a6, a7)) + (bias ? bias[o+3] : 0.0f);

        for (; k < in_dim; k++) {
            float xk = x[k];
            uint32_t b0=((uint32_t)w0[k])<<16, b1=((uint32_t)w1[k])<<16;
            uint32_t b2=((uint32_t)w2[k])<<16, b3=((uint32_t)w3[k])<<16;
            float v0,v1,v2,v3;
            memcpy(&v0,&b0,4); memcpy(&v1,&b1,4); memcpy(&v2,&b2,4); memcpy(&v3,&b3,4);
            s0+=v0*xk; s1+=v1*xk; s2+=v2*xk; s3+=v3*xk;
        }
        y[o]=s0; y[o+1]=s1; y[o+2]=s2; y[o+3]=s3;
    }

    /* Handle 2-row remainder */
    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = w0 + in_dim;
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        int k = 0;

        for (; k + 32 <= in_dim; k += 32) {
            __m512 xv0 = _mm512_loadu_ps(x + k);
            __m512 xv1 = _mm512_loadu_ps(x + k + 16);
            __m256i r0a = _mm256_loadu_si256((const __m256i *)(w0 + k));
            __m256i r0b = _mm256_loadu_si256((const __m256i *)(w0 + k + 16));
            __m256i r1a = _mm256_loadu_si256((const __m256i *)(w1 + k));
            __m256i r1b = _mm256_loadu_si256((const __m256i *)(w1 + k + 16));
            a0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0a), 16)), xv0, a0);
            a1 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0b), 16)), xv1, a1);
            a2 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1a), 16)), xv0, a2);
            a3 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1b), 16)), xv1, a3);
        }
        for (; k + 16 <= in_dim; k += 16) {
            __m512 xv = _mm512_loadu_ps(x + k);
            __m256i r0 = _mm256_loadu_si256((const __m256i *)(w0 + k));
            __m256i r1 = _mm256_loadu_si256((const __m256i *)(w1 + k));
            a0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0), 16)), xv, a0);
            a2 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1), 16)), xv, a2);
        }

        float s0 = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1)) + (bias ? bias[o]   : 0.0f);
        float s1 = _mm512_reduce_add_ps(_mm512_add_ps(a2, a3)) + (bias ? bias[o+1] : 0.0f);

        for (; k < in_dim; k++) {
            uint32_t b0=((uint32_t)w0[k])<<16, b1=((uint32_t)w1[k])<<16;
            float v0,v1; memcpy(&v0,&b0,4); memcpy(&v1,&b1,4);
            s0+=v0*x[k]; s1+=v1*x[k];
        }
        y[o]=s0; y[o+1]=s1;
    }

    /* Single remaining row */
    for (; o < out_dim; o++) {
        const uint16_t *w = W_bf16 + (size_t)o * in_dim;
        __m512 acc = _mm512_setzero_ps();
        int k = 0;
        for (; k + 16 <= in_dim; k += 16) {
            __m256i raw = _mm256_loadu_si256((const __m256i *)(w + k));
            acc = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16)),
                                  _mm512_loadu_ps(x + k), acc);
        }
        float sum = _mm512_reduce_add_ps(acc) + (bias ? bias[o] : 0.0f);
        for (; k < in_dim; k++) {
            uint32_t bits=((uint32_t)w[k])<<16; float wv; memcpy(&wv,&bits,4);
            sum += wv*x[k];
        }
        y[o] = sum;
    }
}

void qwen_argmax_bf16_range_avx(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out) {
    int best = start;
    float best_val = -1e30f;
    int o = start;

    /* Process 4 rows at a time */
    for (; o + 3 < end; o += 4) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = w0 + in_dim;
        const uint16_t *w2 = w1 + in_dim;
        const uint16_t *w3 = w2 + in_dim;

        __m512 a0=_mm512_setzero_ps(), a1=_mm512_setzero_ps();
        __m512 a2=_mm512_setzero_ps(), a3=_mm512_setzero_ps();
        __m512 a4=_mm512_setzero_ps(), a5=_mm512_setzero_ps();
        __m512 a6=_mm512_setzero_ps(), a7=_mm512_setzero_ps();
        int k = 0;

        for (; k + 32 <= in_dim; k += 32) {
            __m512 xv0 = _mm512_loadu_ps(x + k);
            __m512 xv1 = _mm512_loadu_ps(x + k + 16);
            __m256i r0a=_mm256_loadu_si256((const __m256i*)(w0+k));
            __m256i r0b=_mm256_loadu_si256((const __m256i*)(w0+k+16));
            a0=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0a),16)),xv0,a0);
            a1=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0b),16)),xv1,a1);
            __m256i r1a=_mm256_loadu_si256((const __m256i*)(w1+k));
            __m256i r1b=_mm256_loadu_si256((const __m256i*)(w1+k+16));
            a2=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1a),16)),xv0,a2);
            a3=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1b),16)),xv1,a3);
            __m256i r2a=_mm256_loadu_si256((const __m256i*)(w2+k));
            __m256i r2b=_mm256_loadu_si256((const __m256i*)(w2+k+16));
            a4=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r2a),16)),xv0,a4);
            a5=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r2b),16)),xv1,a5);
            __m256i r3a=_mm256_loadu_si256((const __m256i*)(w3+k));
            __m256i r3b=_mm256_loadu_si256((const __m256i*)(w3+k+16));
            a6=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r3a),16)),xv0,a6);
            a7=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r3b),16)),xv1,a7);
        }
        for (; k + 16 <= in_dim; k += 16) {
            __m512 xv = _mm512_loadu_ps(x + k);
            __m256i r0=_mm256_loadu_si256((const __m256i*)(w0+k));
            __m256i r1=_mm256_loadu_si256((const __m256i*)(w1+k));
            __m256i r2=_mm256_loadu_si256((const __m256i*)(w2+k));
            __m256i r3=_mm256_loadu_si256((const __m256i*)(w3+k));
            a0=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r0),16)),xv,a0);
            a2=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r1),16)),xv,a2);
            a4=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r2),16)),xv,a4);
            a6=_mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(r3),16)),xv,a6);
        }

        float s0=_mm512_reduce_add_ps(_mm512_add_ps(a0,a1));
        float s1=_mm512_reduce_add_ps(_mm512_add_ps(a2,a3));
        float s2=_mm512_reduce_add_ps(_mm512_add_ps(a4,a5));
        float s3=_mm512_reduce_add_ps(_mm512_add_ps(a6,a7));

        for (; k < in_dim; k++) {
            float xk=x[k];
            uint32_t b0=((uint32_t)w0[k])<<16, b1=((uint32_t)w1[k])<<16;
            uint32_t b2=((uint32_t)w2[k])<<16, b3=((uint32_t)w3[k])<<16;
            float v0,v1,v2,v3;
            memcpy(&v0,&b0,4); memcpy(&v1,&b1,4); memcpy(&v2,&b2,4); memcpy(&v3,&b3,4);
            s0+=v0*xk; s1+=v1*xk; s2+=v2*xk; s3+=v3*xk;
        }

        if (s0>best_val){best_val=s0;best=o;}
        if (s1>best_val){best_val=s1;best=o+1;}
        if (s2>best_val){best_val=s2;best=o+2;}
        if (s3>best_val){best_val=s3;best=o+3;}
    }

    /* Remaining 1-3 rows */
    for (; o < end; o++) {
        const uint16_t *w = W_bf16 + (size_t)o * in_dim;
        __m512 acc = _mm512_setzero_ps();
        int k = 0;
        for (; k + 16 <= in_dim; k += 16) {
            __m256i raw = _mm256_loadu_si256((const __m256i *)(w + k));
            acc = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(raw), 16)),
                                  _mm512_loadu_ps(x + k), acc);
        }
        float sum = _mm512_reduce_add_ps(acc);
        for (; k < in_dim; k++) {
            uint32_t bits=((uint32_t)w[k])<<16; float wv; memcpy(&wv,&bits,4);
            sum += wv*x[k];
        }
        if (sum > best_val) { best_val = sum; best = o; }
    }

    *best_out = best;
    *best_val_out = best_val;
}

#else /* AVX2 only (or AVX-512F without BW) */

/* Helper: bf16→f32 for 8 elements */
static inline __m256 bf16x8_to_f32(__m128i raw) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
}

static inline void bf16x16_to_f32(const uint16_t *src, __m256 *lo, __m256 *hi) {
    __m256i raw = _mm256_loadu_si256((const __m256i *)src);
    *lo = bf16x8_to_f32(_mm256_castsi256_si128(raw));
    *hi = bf16x8_to_f32(_mm256_extracti128_si256(raw, 1));
}

void qwen_bf16_matvec_fused_avx(float *y, const float *x, const uint16_t *W_bf16,
                                 const float *bias, int in_dim, int out_dim) {
    int o = 0;

    for (; o + 3 < out_dim; o += 4) {
        const uint16_t *w0 = W_bf16 + (size_t)(o + 0) * in_dim;
        const uint16_t *w1 = W_bf16 + (size_t)(o + 1) * in_dim;
        const uint16_t *w2 = W_bf16 + (size_t)(o + 2) * in_dim;
        const uint16_t *w3 = W_bf16 + (size_t)(o + 3) * in_dim;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
        __m256 a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();
        int k = 0;
        for (; k + 16 <= in_dim; k += 16) {
            __m256 xlo = _mm256_loadu_ps(x + k);
            __m256 xhi = _mm256_loadu_ps(x + k + 8);
            __m256 wlo, whi;
            bf16x16_to_f32(w0 + k, &wlo, &whi);
            a0 = _mm256_fmadd_ps(wlo, xlo, a0);
            a1 = _mm256_fmadd_ps(whi, xhi, a1);
            bf16x16_to_f32(w1 + k, &wlo, &whi);
            a2 = _mm256_fmadd_ps(wlo, xlo, a2);
            a3 = _mm256_fmadd_ps(whi, xhi, a3);
            bf16x16_to_f32(w2 + k, &wlo, &whi);
            a4 = _mm256_fmadd_ps(wlo, xlo, a4);
            a5 = _mm256_fmadd_ps(whi, xhi, a5);
            bf16x16_to_f32(w3 + k, &wlo, &whi);
            a6 = _mm256_fmadd_ps(wlo, xlo, a6);
            a7 = _mm256_fmadd_ps(whi, xhi, a7);
        }
        a0 = _mm256_add_ps(a0, a1);
        a2 = _mm256_add_ps(a2, a3);
        a4 = _mm256_add_ps(a4, a5);
        a6 = _mm256_add_ps(a6, a7);
        __m128 s0 = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
        __m128 s1 = _mm_add_ps(_mm256_castps256_ps128(a2), _mm256_extractf128_ps(a2, 1));
        __m128 s2 = _mm_add_ps(_mm256_castps256_ps128(a4), _mm256_extractf128_ps(a4, 1));
        __m128 s3 = _mm_add_ps(_mm256_castps256_ps128(a6), _mm256_extractf128_ps(a6, 1));
        __m128 h01 = _mm_hadd_ps(s0, s1);
        __m128 h23 = _mm_hadd_ps(s2, s3);
        __m128 sums = _mm_hadd_ps(h01, h23);
        float r0 = _mm_cvtss_f32(sums) + (bias ? bias[o + 0] : 0.0f);
        float r1 = _mm_cvtss_f32(_mm_shuffle_ps(sums, sums, 0x55)) + (bias ? bias[o + 1] : 0.0f);
        float r2 = _mm_cvtss_f32(_mm_shuffle_ps(sums, sums, 0xAA)) + (bias ? bias[o + 2] : 0.0f);
        float r3 = _mm_cvtss_f32(_mm_shuffle_ps(sums, sums, 0xFF)) + (bias ? bias[o + 3] : 0.0f);
        for (; k < in_dim; k++) {
            float xk = x[k];
            uint32_t b0 = ((uint32_t)w0[k]) << 16;
            uint32_t b1 = ((uint32_t)w1[k]) << 16;
            uint32_t b2 = ((uint32_t)w2[k]) << 16;
            uint32_t b3 = ((uint32_t)w3[k]) << 16;
            float v0, v1, v2, v3;
            memcpy(&v0, &b0, 4);
            memcpy(&v1, &b1, 4);
            memcpy(&v2, &b2, 4);
            memcpy(&v3, &b3, 4);
            r0 += v0 * xk;
            r1 += v1 * xk;
            r2 += v2 * xk;
            r3 += v3 * xk;
        }
        y[o + 0] = r0;
        y[o + 1] = r1;
        y[o + 2] = r2;
        y[o + 3] = r3;
    }

    for (; o + 1 < out_dim; o += 2) {
        const uint16_t *w0 = W_bf16 + (size_t)o * in_dim;
        const uint16_t *w1 = w0 + in_dim;
        __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps();
        __m256 a2=_mm256_setzero_ps(), a3=_mm256_setzero_ps();
        int k = 0;
        for (; k + 16 <= in_dim; k += 16) {
            __m256 xlo = _mm256_loadu_ps(x + k);
            __m256 xhi = _mm256_loadu_ps(x + k + 8);
            __m256 wlo, whi;
            bf16x16_to_f32(w0 + k, &wlo, &whi);
            a0 = _mm256_fmadd_ps(wlo, xlo, a0);
            a1 = _mm256_fmadd_ps(whi, xhi, a1);
            bf16x16_to_f32(w1 + k, &wlo, &whi);
            a2 = _mm256_fmadd_ps(wlo, xlo, a2);
            a3 = _mm256_fmadd_ps(whi, xhi, a3);
        }
        a0 = _mm256_add_ps(a0, a1); a2 = _mm256_add_ps(a2, a3);
        __m128 r0 = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
        __m128 r1 = _mm_add_ps(_mm256_castps256_ps128(a2), _mm256_extractf128_ps(a2, 1));
        r0 = _mm_hadd_ps(r0, r1); r0 = _mm_hadd_ps(r0, r0);
        float s0 = _mm_cvtss_f32(r0) + (bias ? bias[o] : 0.0f);
        float s1 = _mm_cvtss_f32(_mm_shuffle_ps(r0,r0,1)) + (bias ? bias[o+1] : 0.0f);
        for (; k < in_dim; k++) {
            uint32_t b0=((uint32_t)w0[k])<<16, b1=((uint32_t)w1[k])<<16;
            float v0,v1; memcpy(&v0,&b0,4); memcpy(&v1,&b1,4);
            s0+=v0*x[k]; s1+=v1*x[k];
        }
        y[o]=s0; y[o+1]=s1;
    }
    for (; o < out_dim; o++) {
        const uint16_t *w = W_bf16 + (size_t)o * in_dim;
        __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps();
        int k = 0;
        for (; k + 16 <= in_dim; k += 16) {
            __m256 wlo, whi;
            bf16x16_to_f32(w + k, &wlo, &whi);
            a0 = _mm256_fmadd_ps(wlo, _mm256_loadu_ps(x+k), a0);
            a1 = _mm256_fmadd_ps(whi, _mm256_loadu_ps(x+k+8), a1);
        }
        a0 = _mm256_add_ps(a0, a1);
        __m128 r = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
        r = _mm_hadd_ps(r, r); r = _mm_hadd_ps(r, r);
        float sum = _mm_cvtss_f32(r) + (bias ? bias[o] : 0.0f);
        for (; k < in_dim; k++) {
            uint32_t bits=((uint32_t)w[k])<<16; float wv; memcpy(&wv,&bits,4);
            sum+=wv*x[k];
        }
        y[o] = sum;
    }
}

void qwen_argmax_bf16_range_avx(const float *x, const uint16_t *W_bf16,
                                 int in_dim, int start, int end,
                                 int *best_out, float *best_val_out) {
    int best = start;
    float best_val = -1e30f;

    for (int o = start; o < end; o++) {
        const uint16_t *w = W_bf16 + (size_t)o * in_dim;
        __m256 a0=_mm256_setzero_ps(), a1=_mm256_setzero_ps();
        int k = 0;
        for (; k + 16 <= in_dim; k += 16) {
            __m256 wlo, whi;
            bf16x16_to_f32(w + k, &wlo, &whi);
            a0 = _mm256_fmadd_ps(wlo, _mm256_loadu_ps(x+k), a0);
            a1 = _mm256_fmadd_ps(whi, _mm256_loadu_ps(x+k+8), a1);
        }
        a0 = _mm256_add_ps(a0, a1);
        __m128 r = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
        r = _mm_hadd_ps(r, r); r = _mm_hadd_ps(r, r);
        float sum = _mm_cvtss_f32(r);
        for (; k < in_dim; k++) {
            uint32_t bits=((uint32_t)w[k])<<16; float wv; memcpy(&wv,&bits,4);
            sum+=wv*x[k];
        }
        if (sum > best_val) { best_val = sum; best = o; }
    }
    *best_out = best;
    *best_val_out = best_val;
}

#endif /* AVX-512F+BW vs AVX2 for bf16 */

static inline __m256 bf16x8_to_f32_any(__m128i raw) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
}

static inline void bf16x16_to_f32_any(const uint16_t *src, __m256 *lo, __m256 *hi) {
    __m256i raw = _mm256_loadu_si256((const __m256i *)src);
    *lo = bf16x8_to_f32_any(_mm256_castsi256_si128(raw));
    *hi = bf16x8_to_f32_any(_mm256_extracti128_si256(raw, 1));
}

static inline float hsum256_f32(__m256 v) {
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    return _mm_cvtss_f32(r);
}

void qwen_bf16_qkv_fused_matvec_avx(float *q, float *k, float *v, const float *x,
                                    const uint16_t *Wqkv_bf16, int in_dim, int kv_dim) {
    for (int g = 0; g < kv_dim; g++) {
        const uint16_t *w0 = Wqkv_bf16 + (size_t)(4 * g + 0) * in_dim;
        const uint16_t *w1 = Wqkv_bf16 + (size_t)(4 * g + 1) * in_dim;
        const uint16_t *w2 = Wqkv_bf16 + (size_t)(4 * g + 2) * in_dim;
        const uint16_t *w3 = Wqkv_bf16 + (size_t)(4 * g + 3) * in_dim;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
        __m256 a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();
        int i = 0;
        for (; i + 16 <= in_dim; i += 16) {
            __m256 xlo = _mm256_loadu_ps(x + i);
            __m256 xhi = _mm256_loadu_ps(x + i + 8);
            __m256 wlo, whi;
            bf16x16_to_f32_any(w0 + i, &wlo, &whi);
            a0 = _mm256_fmadd_ps(wlo, xlo, a0);
            a1 = _mm256_fmadd_ps(whi, xhi, a1);
            bf16x16_to_f32_any(w1 + i, &wlo, &whi);
            a2 = _mm256_fmadd_ps(wlo, xlo, a2);
            a3 = _mm256_fmadd_ps(whi, xhi, a3);
            bf16x16_to_f32_any(w2 + i, &wlo, &whi);
            a4 = _mm256_fmadd_ps(wlo, xlo, a4);
            a5 = _mm256_fmadd_ps(whi, xhi, a5);
            bf16x16_to_f32_any(w3 + i, &wlo, &whi);
            a6 = _mm256_fmadd_ps(wlo, xlo, a6);
            a7 = _mm256_fmadd_ps(whi, xhi, a7);
        }
        float s0 = hsum256_f32(_mm256_add_ps(a0, a1));
        float s1 = hsum256_f32(_mm256_add_ps(a2, a3));
        float s2 = hsum256_f32(_mm256_add_ps(a4, a5));
        float s3 = hsum256_f32(_mm256_add_ps(a6, a7));
        for (; i < in_dim; i++) {
            float xi = x[i];
            uint32_t b0 = ((uint32_t)w0[i]) << 16;
            uint32_t b1 = ((uint32_t)w1[i]) << 16;
            uint32_t b2 = ((uint32_t)w2[i]) << 16;
            uint32_t b3 = ((uint32_t)w3[i]) << 16;
            float f0, f1, f2, f3;
            memcpy(&f0, &b0, 4);
            memcpy(&f1, &b1, 4);
            memcpy(&f2, &b2, 4);
            memcpy(&f3, &b3, 4);
            s0 += f0 * xi;
            s1 += f1 * xi;
            s2 += f2 * xi;
            s3 += f3 * xi;
        }
        q[2 * g + 0] = s0;
        q[2 * g + 1] = s1;
        k[g] = s2;
        v[g] = s3;
    }
}

/* =====================================================================
 * f32 attention helpers - AVX2+FMA, with AVX-512F when available
 * (operates on L1-resident head vectors)
 * ===================================================================== */

float qwen_dot_f32_avx(const float *a, const float *b, int n) {
#if defined(__AVX512F__)
    int i = 0;
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    for (; i + 64 <= n; i += 64) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i),      acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
        acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), acc2);
        acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), acc3);
    }
    __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
    for (; i + 16 <= n; i += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#else
    int i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i),    acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8),  acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24), acc3);
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    for (; i + 8 <= n; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i), acc0);
    }
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    float sum = _mm_cvtss_f32(r);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#endif
}

void qwen_vec_scale_inplace_avx(float *dst, float scale, int n) {
#if defined(__AVX512F__)
    int i = 0;
    __m512 s = _mm512_set1_ps(scale);
    for (; i + 64 <= n; i += 64) {
        _mm512_storeu_ps(dst + i,      _mm512_mul_ps(_mm512_loadu_ps(dst + i),      s));
        _mm512_storeu_ps(dst + i + 16, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 16), s));
        _mm512_storeu_ps(dst + i + 32, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 32), s));
        _mm512_storeu_ps(dst + i + 48, _mm512_mul_ps(_mm512_loadu_ps(dst + i + 48), s));
    }
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(dst + i, _mm512_mul_ps(_mm512_loadu_ps(dst + i), s));
    }
    for (; i < n; i++) dst[i] *= scale;
#else
    int i = 0;
    __m256 s = _mm256_set1_ps(scale);
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_ps(dst+i,    _mm256_mul_ps(_mm256_loadu_ps(dst+i),    s));
        _mm256_storeu_ps(dst+i+8,  _mm256_mul_ps(_mm256_loadu_ps(dst+i+8),  s));
        _mm256_storeu_ps(dst+i+16, _mm256_mul_ps(_mm256_loadu_ps(dst+i+16), s));
        _mm256_storeu_ps(dst+i+24, _mm256_mul_ps(_mm256_loadu_ps(dst+i+24), s));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst+i, _mm256_mul_ps(_mm256_loadu_ps(dst+i), s));
    }
    for (; i < n; i++) dst[i] *= scale;
#endif
}

void qwen_vec_axpy_inplace_avx(float *dst, const float *src, float alpha, int n) {
#if defined(__AVX512F__)
    int i = 0;
    __m512 a = _mm512_set1_ps(alpha);
    for (; i + 64 <= n; i += 64) {
        _mm512_storeu_ps(dst + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(src + i), a, _mm512_loadu_ps(dst + i)));
        _mm512_storeu_ps(dst + i + 16,
                         _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 16), a, _mm512_loadu_ps(dst + i + 16)));
        _mm512_storeu_ps(dst + i + 32,
                         _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 32), a, _mm512_loadu_ps(dst + i + 32)));
        _mm512_storeu_ps(dst + i + 48,
                         _mm512_fmadd_ps(_mm512_loadu_ps(src + i + 48), a, _mm512_loadu_ps(dst + i + 48)));
    }
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(dst + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(src + i), a, _mm512_loadu_ps(dst + i)));
    }
    for (; i < n; i++) dst[i] += alpha * src[i];
#else
    int i = 0;
    __m256 a = _mm256_set1_ps(alpha);
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_ps(dst+i,    _mm256_fmadd_ps(_mm256_loadu_ps(src+i),    a, _mm256_loadu_ps(dst+i)));
        _mm256_storeu_ps(dst+i+8,  _mm256_fmadd_ps(_mm256_loadu_ps(src+i+8),  a, _mm256_loadu_ps(dst+i+8)));
        _mm256_storeu_ps(dst+i+16, _mm256_fmadd_ps(_mm256_loadu_ps(src+i+16), a, _mm256_loadu_ps(dst+i+16)));
        _mm256_storeu_ps(dst+i+24, _mm256_fmadd_ps(_mm256_loadu_ps(src+i+24), a, _mm256_loadu_ps(dst+i+24)));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst+i, _mm256_fmadd_ps(_mm256_loadu_ps(src+i), a, _mm256_loadu_ps(dst+i)));
    }
    for (; i < n; i++) dst[i] += alpha * src[i];
#endif
}

void qwen_vec_scale_add_avx(float *dst, const float *src, float correction, int n) {
#if defined(__AVX512F__)
    int i = 0;
    __m512 c = _mm512_set1_ps(correction);
    for (; i + 64 <= n; i += 64) {
        _mm512_storeu_ps(dst + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(dst + i), c, _mm512_loadu_ps(src + i)));
        _mm512_storeu_ps(dst + i + 16,
                         _mm512_fmadd_ps(_mm512_loadu_ps(dst + i + 16), c, _mm512_loadu_ps(src + i + 16)));
        _mm512_storeu_ps(dst + i + 32,
                         _mm512_fmadd_ps(_mm512_loadu_ps(dst + i + 32), c, _mm512_loadu_ps(src + i + 32)));
        _mm512_storeu_ps(dst + i + 48,
                         _mm512_fmadd_ps(_mm512_loadu_ps(dst + i + 48), c, _mm512_loadu_ps(src + i + 48)));
    }
    for (; i + 16 <= n; i += 16) {
        _mm512_storeu_ps(dst + i,
                         _mm512_fmadd_ps(_mm512_loadu_ps(dst + i), c, _mm512_loadu_ps(src + i)));
    }
    for (; i < n; i++) dst[i] = dst[i] * correction + src[i];
#else
    int i = 0;
    __m256 c = _mm256_set1_ps(correction);
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_ps(dst+i,    _mm256_fmadd_ps(_mm256_loadu_ps(dst+i),    c, _mm256_loadu_ps(src+i)));
        _mm256_storeu_ps(dst+i+8,  _mm256_fmadd_ps(_mm256_loadu_ps(dst+i+8),  c, _mm256_loadu_ps(src+i+8)));
        _mm256_storeu_ps(dst+i+16, _mm256_fmadd_ps(_mm256_loadu_ps(dst+i+16), c, _mm256_loadu_ps(src+i+16)));
        _mm256_storeu_ps(dst+i+24, _mm256_fmadd_ps(_mm256_loadu_ps(dst+i+24), c, _mm256_loadu_ps(src+i+24)));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(dst+i, _mm256_fmadd_ps(_mm256_loadu_ps(dst+i), c, _mm256_loadu_ps(src+i)));
    }
    for (; i < n; i++) dst[i] = dst[i] * correction + src[i];
#endif
}

#endif /* __AVX2__ && __FMA__ */
