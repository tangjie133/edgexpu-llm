#include "edgexpu/gguf_quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 按 GGUF 块格式反量化。布局与 llama.cpp 一致，便于后续数值对齐。 */

#if defined(EDGEXPU_AVX2) || defined(EDGEXPU_F16C)
#include <immintrin.h>
#endif
#if defined(EDGEXPU_NEON)
#include <arm_neon.h>
#endif

#define QK8_0 32
#define QK5_0 32
#define QK4_0 32
#define QK_K 256
#define Q8_0_BLOCK_BYTES 34
#define Q5_0_BLOCK_BYTES 22
#define Q4_0_BLOCK_BYTES 18
#define Q4_K_BLOCK_BYTES 144
#define Q5_K_BLOCK_BYTES 176
#define Q6_K_BLOCK_BYTES 210

static void set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
    }
}

#if defined(EDGEXPU_F16C)
static float fp16_to_fp32(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}
#else
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h >> 15) & 1u) << 31;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    float value;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127u - 15u + 1u;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }

    memcpy(&value, &bits, sizeof(value));
    return value;
}
#endif

size_t edgexpu_gguf_type_block_size(uint32_t type) {
    switch (type) {
        case EDGEXPU_GGUF_TYPE_F32:
        case EDGEXPU_GGUF_TYPE_F16:
            return 1;
        case EDGEXPU_GGUF_TYPE_Q4_0:
        case EDGEXPU_GGUF_TYPE_Q5_0:
        case EDGEXPU_GGUF_TYPE_Q8_0:
            return QK8_0;
        case EDGEXPU_GGUF_TYPE_Q4_K:
        case EDGEXPU_GGUF_TYPE_Q5_K:
        case EDGEXPU_GGUF_TYPE_Q6_K:
            return QK_K;
        default:
            return 0;
    }
}

size_t edgexpu_gguf_type_block_bytes(uint32_t type) {
    switch (type) {
        case EDGEXPU_GGUF_TYPE_F32:
            return 4;
        case EDGEXPU_GGUF_TYPE_F16:
            return 2;
        case EDGEXPU_GGUF_TYPE_Q4_0:
            return Q4_0_BLOCK_BYTES;
        case EDGEXPU_GGUF_TYPE_Q5_0:
            return Q5_0_BLOCK_BYTES;
        case EDGEXPU_GGUF_TYPE_Q8_0:
            return Q8_0_BLOCK_BYTES;
        case EDGEXPU_GGUF_TYPE_Q4_K:
            return Q4_K_BLOCK_BYTES;
        case EDGEXPU_GGUF_TYPE_Q5_K:
            return Q5_K_BLOCK_BYTES;
        case EDGEXPU_GGUF_TYPE_Q6_K:
            return Q6_K_BLOCK_BYTES;
        default:
            return 0;
    }
}

uint64_t edgexpu_gguf_tensor_elements(const edgexpu_gguf_tensor *tensor) {
    uint64_t n = 1;
    uint32_t i;
    if (tensor == NULL) {
        return 0;
    }
    for (i = 0; i < tensor->n_dims; i++) {
        n *= tensor->dims[i];
    }
    return n;
}

size_t edgexpu_gguf_tensor_nbytes(const edgexpu_gguf_tensor *tensor) {
    uint64_t n;
    size_t block_size;
    size_t block_bytes;
    if (tensor == NULL) {
        return 0;
    }
    n = edgexpu_gguf_tensor_elements(tensor);
    block_size = edgexpu_gguf_type_block_size(tensor->type);
    block_bytes = edgexpu_gguf_type_block_bytes(tensor->type);
    if (block_size == 0 || block_bytes == 0 || n % block_size != 0) {
        return 0;
    }
    return (size_t)(n / block_size) * block_bytes;
}

/* 块格式：fp16 scale + 32 个 int8。 */
static void dequant_q8_0(const uint8_t *src, uint64_t n, float *dst) {
    uint64_t nb = n / QK8_0;
    uint64_t i;
    uint64_t j;
    for (i = 0; i < nb; i++) {
        const uint8_t *block = src + i * Q8_0_BLOCK_BYTES;
        uint16_t d_bits;
        float d;
        const int8_t *qs;
        memcpy(&d_bits, block, 2);
        d = fp16_to_fp32(d_bits);
        qs = (const int8_t *)(block + 2);
        for (j = 0; j < QK8_0; j++) {
            dst[i * QK8_0 + j] = d * (float)qs[j];
        }
    }
}

static void dequant_q5_0(const uint8_t *src, uint64_t n, float *dst) {
    uint64_t nb = n / QK5_0;
    uint64_t i;
    int j;
    for (i = 0; i < nb; i++) {
        const uint8_t *block = src + i * Q5_0_BLOCK_BYTES;
        uint16_t d_bits;
        uint32_t qh;
        float d;
        const uint8_t *qs;
        memcpy(&d_bits, block, 2);
        memcpy(&qh, block + 2, 4);
        qs = block + 6;
        d = fp16_to_fp32(d_bits);
        for (j = 0; j < QK5_0 / 2; j++) {
            uint8_t xh_0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
            uint8_t xh_1 = (uint8_t)((qh >> (j + 12)) & 0x10);
            int32_t x0 = (int32_t)((qs[j] & 0x0F) | xh_0) - 16;
            int32_t x1 = (int32_t)((qs[j] >> 4) | xh_1) - 16;
            dst[i * QK5_0 + j] = (float)x0 * d;
            dst[i * QK5_0 + j + QK5_0 / 2] = (float)x1 * d;
        }
    }
}

/* F16：逐元素 IEEE half。 */
static void dequant_f16(const uint8_t *src, uint64_t n, float *dst) {
    uint64_t i;
    for (i = 0; i < n; i++) {
        uint16_t bits;
        memcpy(&bits, src + i * 2u, 2);
        dst[i] = fp16_to_fp32(bits);
    }
}

/* Q4_0：32 元素，fp16 d + 16 字节 nibble，量化值偏移 8。 */
static void dequant_q4_0(const uint8_t *src, uint64_t n, float *dst) {
    uint64_t nb = n / QK4_0;
    uint64_t i;
    int j;
    for (i = 0; i < nb; i++) {
        const uint8_t *block = src + i * Q4_0_BLOCK_BYTES;
        uint16_t d_bits;
        float d;
        const uint8_t *qs;
        memcpy(&d_bits, block, 2);
        d = fp16_to_fp32(d_bits);
        qs = block + 2;
        for (j = 0; j < QK4_0 / 2; j++) {
            dst[i * QK4_0 + j] = ((float)(qs[j] & 0x0F) - 8.0f) * d;
            dst[i * QK4_0 + j + QK4_0 / 2] = ((float)(qs[j] >> 4) - 8.0f) * d;
        }
    }
}

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

/* Q4_K super-block：256 元素，含 d/dmin 与 4-bit qs。 */
static void dequant_q4_k(const uint8_t *src, uint64_t n, float *dst) {
    uint64_t nb = n / QK_K;
    uint64_t i;
    for (i = 0; i < nb; i++) {
        const uint8_t *block = src + i * Q4_K_BLOCK_BYTES;
        uint16_t d_bits;
        uint16_t min_bits;
        float d;
        float min;
        const uint8_t *scales = block + 4;
        const uint8_t *q = block + 16;
        float *y = dst + i * QK_K;
        int is = 0;
        int nout = 0;
        int j;
        memcpy(&d_bits, block, 2);
        memcpy(&min_bits, block + 2, 2);
        d = fp16_to_fp32(d_bits);
        min = fp16_to_fp32(min_bits);
        for (j = 0; j < QK_K; j += 64) {
            uint8_t sc;
            uint8_t m;
            float d1;
            float m1;
            float d2;
            float m2;
            int l;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            d1 = d * (float)sc;
            m1 = min * (float)m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            d2 = d * (float)sc;
            m2 = min * (float)m;
            for (l = 0; l < 32; l++) {
                y[nout++] = d1 * (float)(q[l] & 0xF) - m1;
            }
            for (l = 0; l < 32; l++) {
                y[nout++] = d2 * (float)(q[l] >> 4) - m2;
            }
            q += 32;
            is += 2;
        }
    }
}

/* Q6_K super-block：256 元素。 */
static void dequant_q6_k(const uint8_t *src, uint64_t n, float *dst) {
    uint64_t nb = n / QK_K;
    uint64_t i;
    for (i = 0; i < nb; i++) {
        const uint8_t *block = src + i * Q6_K_BLOCK_BYTES;
        const uint8_t *ql = block;
        const uint8_t *qh = block + QK_K / 2;
        const int8_t *sc = (const int8_t *)(block + QK_K / 2 + QK_K / 4);
        uint16_t d_bits;
        float d;
        float *y = dst + i * QK_K;
        int nblock;
        memcpy(&d_bits, block + Q6_K_BLOCK_BYTES - 2, 2);
        d = fp16_to_fp32(d_bits);
        for (nblock = 0; nblock < QK_K; nblock += 128) {
            int l;
            for (l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/* Q5_K super-block：256 元素，4-bit qs + 1-bit qh，带 dmin。 */
static void dequant_q5_k(const uint8_t *src, uint64_t n, float *dst) {
    uint64_t nb = n / QK_K;
    uint64_t i;
    for (i = 0; i < nb; i++) {
        const uint8_t *block = src + i * Q5_K_BLOCK_BYTES;
        uint16_t d_bits;
        uint16_t min_bits;
        float d;
        float minv;
        const uint8_t *scales = block + 4;
        const uint8_t *qh = block + 16;
        const uint8_t *ql = block + 48;
        float *y = dst + i * QK_K;
        int is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;
        int j;
        memcpy(&d_bits, block, 2);
        memcpy(&min_bits, block + 2, 2);
        d = fp16_to_fp32(d_bits);
        minv = fp16_to_fp32(min_bits);
        for (j = 0; j < QK_K; j += 64) {
            uint8_t sc;
            uint8_t m;
            float d1;
            float m1;
            float d2;
            float m2;
            int l;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            d1 = d * (float)sc;
            m1 = minv * (float)m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            d2 = d * (float)sc;
            m2 = minv * (float)m;
            for (l = 0; l < 32; l++) {
                y[l] = d1 * (float)((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
            }
            for (l = 0; l < 32; l++) {
                y[l + 32] = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            }
            y += 64;
            ql += 32;
            is += 2;
            u1 = (uint8_t)(u1 << 2);
            u2 = (uint8_t)(u2 << 2);
        }
    }
}

int edgexpu_gguf_dequantize(
    const uint8_t *src,
    uint32_t type,
    uint64_t n_elements,
    float *dst,
    char *error,
    size_t error_size
) {
    size_t block_size;

    if (src == NULL || dst == NULL || n_elements == 0) {
        set_error(error, error_size, "dequantize 参数为空");
        return 0;
    }

    block_size = edgexpu_gguf_type_block_size(type);
    if (block_size == 0 || n_elements % block_size != 0) {
        set_error(error, error_size, "不支持的 GGUF 量化类型或元素数不对齐");
        return 0;
    }

    if (type == EDGEXPU_GGUF_TYPE_F32) {
        memcpy(dst, src, (size_t)n_elements * sizeof(float));
        return 1;
    }
    if (type == EDGEXPU_GGUF_TYPE_F16) {
        dequant_f16(src, n_elements, dst);
        return 1;
    }
    if (type == EDGEXPU_GGUF_TYPE_Q4_0) {
        dequant_q4_0(src, n_elements, dst);
        return 1;
    }
    if (type == EDGEXPU_GGUF_TYPE_Q8_0) {
        dequant_q8_0(src, n_elements, dst);
        return 1;
    }
    if (type == EDGEXPU_GGUF_TYPE_Q5_0) {
        dequant_q5_0(src, n_elements, dst);
        return 1;
    }
    if (type == EDGEXPU_GGUF_TYPE_Q4_K) {
        dequant_q4_k(src, n_elements, dst);
        return 1;
    }
    if (type == EDGEXPU_GGUF_TYPE_Q5_K) {
        dequant_q5_k(src, n_elements, dst);
        return 1;
    }
    if (type == EDGEXPU_GGUF_TYPE_Q6_K) {
        dequant_q6_k(src, n_elements, dst);
        return 1;
    }

    set_error(error, error_size, "当前 native loader 不支持该量化类型");
    return 0;
}

int edgexpu_gguf_dequantize_q8_0_rows(
    const uint8_t *src,
    uint64_t row,
    int n_embd,
    float *dst
) {
    size_t blocks;
    if (src == NULL || dst == NULL || n_embd <= 0 || n_embd % QK8_0 != 0) {
        return 0;
    }
    blocks = (size_t)n_embd / QK8_0;
    dequant_q8_0(src + row * blocks * Q8_0_BLOCK_BYTES, (uint64_t)n_embd, dst);
    return 1;
}

int edgexpu_gguf_dequantize_row(
    const uint8_t *src,
    const edgexpu_gguf_tensor *tensor,
    uint64_t row,
    int n_embd,
    float *dst,
    char *error,
    size_t error_size
) {
    uint64_t n;
    uint64_t rows;
    size_t nbytes;
    size_t row_bytes;

    if (src == NULL || tensor == NULL || dst == NULL || n_embd <= 0) {
        set_error(error, error_size, "dequantize_row 参数为空");
        return 0;
    }
    n = edgexpu_gguf_tensor_elements(tensor);
    if (n == 0 || n % (uint64_t)n_embd != 0) {
        set_error(error, error_size, "tensor 行宽与 embedding_length 不匹配");
        return 0;
    }
    rows = n / (uint64_t)n_embd;
    if (row >= rows) {
        set_error(error, error_size, "tensor 行越界");
        return 0;
    }
    if (tensor->type == EDGEXPU_GGUF_TYPE_Q8_0) {
        return edgexpu_gguf_dequantize_q8_0_rows(src, row, n_embd, dst);
    }
    nbytes = edgexpu_gguf_tensor_nbytes(tensor);
    if (nbytes == 0 || nbytes % rows != 0) {
        set_error(error, error_size, "tensor 字节数无法按行切分");
        return 0;
    }
    row_bytes = nbytes / (size_t)rows;
    return edgexpu_gguf_dequantize(src + row * row_bytes, tensor->type, (uint64_t)n_embd, dst, error, error_size);
}

size_t edgexpu_gguf_row_bytes(uint32_t type, int n) {
    size_t block_size;
    size_t block_bytes;
    if (n <= 0) {
        return 0;
    }
    block_size = edgexpu_gguf_type_block_size(type);
    block_bytes = edgexpu_gguf_type_block_bytes(type);
    if (block_size == 0 || block_bytes == 0 || (size_t)n % block_size != 0) {
        return 0;
    }
    return ((size_t)n / block_size) * block_bytes;
}

#if defined(EDGEXPU_AVX2)
#if defined(EDGEXPU_FMA)
#define EDGEXPU_FMADD256(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#else
#define EDGEXPU_FMADD256(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#endif
static float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehdup_ps(s));
    s = _mm_add_ss(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(s);
}
#endif

#if defined(EDGEXPU_NEON)
static float hsum128(float32x4_t v) {
    float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    s = vpadd_f32(s, s);
    return vget_lane_f32(s, 0);
}
#endif

static float dot_f32_row(const float *w, const float *x, int n) {
    int i = 0;
    double acc = 0.0;
#if defined(EDGEXPU_AVX2)
    {
        __m256 accv = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) {
            __m256 wv = _mm256_loadu_ps(w + i);
            __m256 xv = _mm256_loadu_ps(x + i);
#if defined(EDGEXPU_FMA)
            accv = _mm256_fmadd_ps(wv, xv, accv);
#else
            accv = _mm256_add_ps(accv, _mm256_mul_ps(wv, xv));
#endif
        }
        acc += (double)hsum256(accv);
    }
#elif defined(EDGEXPU_NEON)
    {
        float32x4_t accv = vdupq_n_f32(0.0f);
        for (; i + 4 <= n; i += 4) {
            accv = vfmaq_f32(accv, vld1q_f32(w + i), vld1q_f32(x + i));
        }
        acc += (double)hsum128(accv);
    }
#endif
    for (; i < n; i++) {
        acc += (double)w[i] * (double)x[i];
    }
    return (float)acc;
}

/* 32 个 4-bit 与 f32 点积，同时累加 x 供 Q4_K 的 min 项。shift=0 低 nibble，4 高 nibble。 */
static void dot_u4_f32_32(const uint8_t *q, const float *x, int shift, float *sum_qx, float *sum_x) {
    int i;
    float qx = 0.0f;
    float sx = 0.0f;
#if defined(EDGEXPU_AVX2)
    {
        __m256 vqx = _mm256_setzero_ps();
        __m256 vx = _mm256_setzero_ps();
        __m128i mask = _mm_set1_epi8(0x0F);
        for (i = 0; i < 32; i += 8) {
            uint64_t packed = 0;
            __m128i bytes;
            __m256 xf;
            memcpy(&packed, q + i, 8);
            bytes = _mm_cvtsi64_si128((int64_t)packed);
            if (shift) {
                bytes = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
            } else {
                bytes = _mm_and_si128(bytes, mask);
            }
            xf = _mm256_loadu_ps(x + i);
            vqx = EDGEXPU_FMADD256(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes)), xf, vqx);
            vx = _mm256_add_ps(vx, xf);
        }
        qx = hsum256(vqx);
        sx = hsum256(vx);
    }
#elif defined(EDGEXPU_NEON)
    {
        float32x4_t vqx = vdupq_n_f32(0.0f);
        float32x4_t vx = vdupq_n_f32(0.0f);
        for (i = 0; i < 32; i += 8) {
            uint8x8_t b = vld1_u8(q + i);
            uint16x8_t w;
            float32x4_t x0;
            float32x4_t x1;
            float32x4_t q0;
            float32x4_t q1;
            if (shift) {
                b = vshr_n_u8(b, 4);
            } else {
                b = vand_u8(b, vdup_n_u8(0x0F));
            }
            w = vmovl_u8(b);
            q0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w)));
            q1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w)));
            x0 = vld1q_f32(x + i);
            x1 = vld1q_f32(x + i + 4);
            vqx = vfmaq_f32(vqx, q0, x0);
            vqx = vfmaq_f32(vqx, q1, x1);
            vx = vaddq_f32(vx, x0);
            vx = vaddq_f32(vx, x1);
        }
        qx = hsum128(vqx);
        sx = hsum128(vx);
    }
#else
    for (i = 0; i < 32; i++) {
        float qv = (float)((q[i] >> shift) & 0x0F);
        qx += qv * x[i];
        sx += x[i];
    }
#endif
    *sum_qx = qx;
    *sum_x = sx;
}

static float dot_q8_0(const uint8_t *row, const float *x, int n) {
    size_t blocks;
    const uint8_t *block;
    float acc = 0.0f;
    size_t b;

    if (row == NULL || x == NULL || n <= 0 || n % QK8_0 != 0) {
        return 0.0f;
    }
    blocks = (size_t)n / QK8_0;
    block = row;
    for (b = 0; b < blocks; b++) {
        uint16_t d_bits;
        float d;
        const int8_t *qs;
        memcpy(&d_bits, block, 2);
        d = fp16_to_fp32(d_bits);
        qs = (const int8_t *)(block + 2);
#if defined(EDGEXPU_AVX2)
        {
            __m256 accv = _mm256_setzero_ps();
            int off;
            for (off = 0; off < QK8_0; off += 8) {
                int64_t packed = 0;
                __m128i q8;
                memcpy(&packed, qs + off, 8);
                q8 = _mm_cvtsi64_si128(packed);
                accv = EDGEXPU_FMADD256(
                    _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8)),
                    _mm256_loadu_ps(x + off),
                    accv
                );
            }
            acc += d * hsum256(accv);
        }
#elif defined(EDGEXPU_NEON)
        {
            float32x4_t accv = vdupq_n_f32(0.0f);
            int off;
            for (off = 0; off < QK8_0; off += 16) {
                int8x16_t q = vld1q_s8(qs + off);
                int16x8_t lo = vmovl_s8(vget_low_s8(q));
                int16x8_t hi = vmovl_s8(vget_high_s8(q));
                accv = vfmaq_f32(accv, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vld1q_f32(x + off));
                accv = vfmaq_f32(accv, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vld1q_f32(x + off + 4));
                accv = vfmaq_f32(accv, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), vld1q_f32(x + off + 8));
                accv = vfmaq_f32(accv, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vld1q_f32(x + off + 12));
            }
            acc += d * hsum128(accv);
        }
#else
        {
            float s = 0.0f;
            int j;
            for (j = 0; j < QK8_0; j++) {
                s += x[j] * (float)qs[j];
            }
            acc += d * s;
        }
#endif
        x += QK8_0;
        block += Q8_0_BLOCK_BYTES;
    }
    return acc;
}

static float dot_q5_0(const uint8_t *row, const float *x, int n) {
    uint64_t nb;
    uint64_t i;
    float acc = 0.0f;

    if (row == NULL || x == NULL || n <= 0 || n % QK5_0 != 0) {
        return 0.0f;
    }
    nb = (uint64_t)n / QK5_0;
    for (i = 0; i < nb; i++) {
        const uint8_t *block = row + i * Q5_0_BLOCK_BYTES;
        uint16_t d_bits;
        uint32_t qh;
        float d;
        const uint8_t *qs;
        const float *xx = x + i * QK5_0;
        int j;
        memcpy(&d_bits, block, 2);
        memcpy(&qh, block + 2, 4);
        qs = block + 6;
        d = fp16_to_fp32(d_bits);
        for (j = 0; j < QK5_0 / 2; j++) {
            uint8_t xh_0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
            uint8_t xh_1 = (uint8_t)((qh >> (j + 12)) & 0x10);
            int32_t x0 = (int32_t)((qs[j] & 0x0F) | xh_0) - 16;
            int32_t x1 = (int32_t)((qs[j] >> 4) | xh_1) - 16;
            acc += d * ((float)x0 * xx[j] + (float)x1 * xx[j + QK5_0 / 2]);
        }
    }
    return acc;
}

static float dot_q4_k(const uint8_t *row, const float *x, int n) {
    uint64_t nb;
    uint64_t i;
    float acc = 0.0f;

    if (row == NULL || x == NULL || n <= 0 || n % QK_K != 0) {
        return 0.0f;
    }
    nb = (uint64_t)n / QK_K;
    for (i = 0; i < nb; i++) {
        const uint8_t *block = row + i * Q4_K_BLOCK_BYTES;
        uint16_t d_bits;
        uint16_t min_bits;
        float d;
        float minv;
        const uint8_t *scales = block + 4;
        const uint8_t *q = block + 16;
        const float *xx = x + i * QK_K;
        int is = 0;
        int j;
        memcpy(&d_bits, block, 2);
        memcpy(&min_bits, block + 2, 2);
        d = fp16_to_fp32(d_bits);
        minv = fp16_to_fp32(min_bits);
        for (j = 0; j < QK_K; j += 64) {
            uint8_t sc;
            uint8_t m;
            float d1;
            float m1;
            float d2;
            float m2;
            float qx;
            float sx;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            d1 = d * (float)sc;
            m1 = minv * (float)m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            d2 = d * (float)sc;
            m2 = minv * (float)m;
            dot_u4_f32_32(q, xx + j, 0, &qx, &sx);
            acc += d1 * qx - m1 * sx;
            dot_u4_f32_32(q, xx + j + 32, 4, &qx, &sx);
            acc += d2 * qx - m2 * sx;
            q += 32;
            is += 2;
        }
    }
    return acc;
}

static float dot_q6_k(const uint8_t *row, const float *x, int n) {
    uint64_t nb;
    uint64_t i;
    float acc = 0.0f;

    if (row == NULL || x == NULL || n <= 0 || n % QK_K != 0) {
        return 0.0f;
    }
    nb = (uint64_t)n / QK_K;
    for (i = 0; i < nb; i++) {
        const uint8_t *block = row + i * Q6_K_BLOCK_BYTES;
        const uint8_t *ql = block;
        const uint8_t *qh = block + QK_K / 2;
        const int8_t *sc = (const int8_t *)(block + QK_K / 2 + QK_K / 4);
        uint16_t d_bits;
        float d;
        const float *xx = x + i * QK_K;
        int nblock;
        memcpy(&d_bits, block + Q6_K_BLOCK_BYTES - 2, 2);
        d = fp16_to_fp32(d_bits);
        for (nblock = 0; nblock < QK_K; nblock += 128) {
            int l;
            for (l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                acc += d * (float)sc[is + 0] * (float)q1 * xx[l + 0];
                acc += d * (float)sc[is + 2] * (float)q2 * xx[l + 32];
                acc += d * (float)sc[is + 4] * (float)q3 * xx[l + 64];
                acc += d * (float)sc[is + 6] * (float)q4 * xx[l + 96];
            }
            xx += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return acc;
}

static float dot_f16(const uint8_t *row, const float *x, int n) {
    int i;
    double acc = 0.0;
    if (row == NULL || x == NULL || n <= 0) {
        return 0.0f;
    }
    for (i = 0; i < n; i++) {
        uint16_t bits;
        memcpy(&bits, row + (size_t)i * 2u, 2);
        acc += (double)fp16_to_fp32(bits) * (double)x[i];
    }
    return (float)acc;
}

static float dot_q4_0(const uint8_t *row, const float *x, int n) {
    uint64_t nb;
    uint64_t i;
    float acc = 0.0f;

    if (row == NULL || x == NULL || n <= 0 || n % QK4_0 != 0) {
        return 0.0f;
    }
    nb = (uint64_t)n / QK4_0;
    for (i = 0; i < nb; i++) {
        const uint8_t *block = row + i * Q4_0_BLOCK_BYTES;
        uint16_t d_bits;
        float d;
        const uint8_t *qs;
        const float *xx = x + i * QK4_0;
        int j;
        memcpy(&d_bits, block, 2);
        d = fp16_to_fp32(d_bits);
        qs = block + 2;
        for (j = 0; j < QK4_0 / 2; j++) {
            acc += d * ((float)(qs[j] & 0x0F) - 8.0f) * xx[j];
            acc += d * ((float)(qs[j] >> 4) - 8.0f) * xx[j + QK4_0 / 2];
        }
    }
    return acc;
}

static float dot_q5_k(const uint8_t *row, const float *x, int n) {
    uint64_t nb;
    uint64_t i;
    float acc = 0.0f;

    if (row == NULL || x == NULL || n <= 0 || n % QK_K != 0) {
        return 0.0f;
    }
    nb = (uint64_t)n / QK_K;
    for (i = 0; i < nb; i++) {
        const uint8_t *block = row + i * Q5_K_BLOCK_BYTES;
        uint16_t d_bits;
        uint16_t min_bits;
        float d;
        float minv;
        const uint8_t *scales = block + 4;
        const uint8_t *qh = block + 16;
        const uint8_t *ql = block + 48;
        const float *xx = x + i * QK_K;
        int is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;
        int j;
        memcpy(&d_bits, block, 2);
        memcpy(&min_bits, block + 2, 2);
        d = fp16_to_fp32(d_bits);
        minv = fp16_to_fp32(min_bits);
        for (j = 0; j < QK_K; j += 64) {
            uint8_t sc;
            uint8_t m;
            float d1;
            float m1;
            float d2;
            float m2;
            int l;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            d1 = d * (float)sc;
            m1 = minv * (float)m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            d2 = d * (float)sc;
            m2 = minv * (float)m;
            for (l = 0; l < 32; l++) {
                float q0 = (float)((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0));
                float q1 = (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0));
                acc += (d1 * q0 - m1) * xx[l];
                acc += (d2 * q1 - m2) * xx[l + 32];
            }
            xx += 64;
            ql += 32;
            is += 2;
            u1 = (uint8_t)(u1 << 2);
            u2 = (uint8_t)(u2 << 2);
        }
    }
    return acc;
}

float edgexpu_gguf_dot_quant(uint32_t type, const uint8_t *row, const float *x, int n) {
    if (row == NULL || x == NULL || n <= 0) {
        return 0.0f;
    }
    switch (type) {
        case EDGEXPU_GGUF_TYPE_F32:
            return dot_f32_row((const float *)row, x, n);
        case EDGEXPU_GGUF_TYPE_F16:
            return dot_f16(row, x, n);
        case EDGEXPU_GGUF_TYPE_Q4_0:
            return dot_q4_0(row, x, n);
        case EDGEXPU_GGUF_TYPE_Q8_0:
            return dot_q8_0(row, x, n);
        case EDGEXPU_GGUF_TYPE_Q5_0:
            return dot_q5_0(row, x, n);
        case EDGEXPU_GGUF_TYPE_Q4_K:
            return dot_q4_k(row, x, n);
        case EDGEXPU_GGUF_TYPE_Q5_K:
            return dot_q5_k(row, x, n);
        case EDGEXPU_GGUF_TYPE_Q6_K:
            return dot_q6_k(row, x, n);
        default:
            return 0.0f;
    }
}

float edgexpu_gguf_q8_0_dot_row(
    const uint8_t *src,
    uint64_t row,
    int n_embd,
    const float *x
) {
    size_t row_bytes;
    if (src == NULL || x == NULL || n_embd <= 0 || n_embd % QK8_0 != 0) {
        return 0.0f;
    }
    row_bytes = edgexpu_gguf_row_bytes(EDGEXPU_GGUF_TYPE_Q8_0, n_embd);
    return dot_q8_0(src + row * row_bytes, x, n_embd);
}

int edgexpu_gguf_quant_selftest(char *error, size_t error_size) {
    uint8_t block[Q4_K_BLOCK_BYTES];
    float x[QK_K];
    float y[QK_K];
    float fused;
    float expanded;
    uint16_t one = 0x3C00; /* fp16 1.0 */
    int i;

    memset(block, 0, sizeof(block));
    memset(x, 0, sizeof(x));
    memcpy(block, &one, 2);
    block[4] = 1;     /* first group scale = 1 */
    block[16] = 0x02; /* low nibble 2 at element 0 */
    x[0] = 3.0f;
    dequant_q4_k(block, QK_K, y);
    expanded = 0.0f;
    for (i = 0; i < QK_K; i++) {
        expanded += y[i] * x[i];
    }
    fused = edgexpu_gguf_dot_quant(EDGEXPU_GGUF_TYPE_Q4_K, block, x, QK_K);
    if (fabsf(fused - expanded) > 1e-4f || fabsf(fused - 6.0f) > 1e-3f) {
        snprintf(error, error_size, "q4_k fused dot selftest failed fused=%.6f expanded=%.6f", fused, expanded);
        return 0;
    }

    {
        uint8_t q40[Q4_0_BLOCK_BYTES];
        float x0[QK4_0];
        float y0[QK4_0];
        memset(q40, 0, sizeof(q40));
        memset(x0, 0, sizeof(x0));
        memcpy(q40, &one, 2);
        q40[2] = 0x09; /* low nibble 9 → (9-8)*d = 1 */
        x0[0] = 3.0f;
        dequant_q4_0(q40, QK4_0, y0);
        expanded = 0.0f;
        for (i = 0; i < QK4_0; i++) {
            expanded += y0[i] * x0[i];
        }
        fused = edgexpu_gguf_dot_quant(EDGEXPU_GGUF_TYPE_Q4_0, q40, x0, QK4_0);
        if (fabsf(fused - expanded) > 1e-4f || fabsf(fused - 3.0f) > 1e-3f) {
            snprintf(error, error_size, "q4_0 fused dot selftest failed fused=%.6f expanded=%.6f", fused, expanded);
            return 0;
        }
    }

    {
        uint8_t h[4];
        float xh[2];
        uint16_t two = 0x4000; /* fp16 2.0 */
        memcpy(h, &one, 2);
        memcpy(h + 2, &two, 2);
        xh[0] = 2.0f;
        xh[1] = 3.0f;
        fused = edgexpu_gguf_dot_quant(EDGEXPU_GGUF_TYPE_F16, h, xh, 2);
        if (fabsf(fused - 8.0f) > 1e-3f) {
            snprintf(error, error_size, "f16 fused dot selftest failed fused=%.6f", fused);
            return 0;
        }
    }

    {
        uint8_t q5[Q5_K_BLOCK_BYTES];
        memset(q5, 0, sizeof(q5));
        memset(x, 0, sizeof(x));
        memcpy(q5, &one, 2);
        q5[4] = 1;
        q5[48] = 0x02;
        x[0] = 3.0f;
        dequant_q5_k(q5, QK_K, y);
        expanded = 0.0f;
        for (i = 0; i < QK_K; i++) {
            expanded += y[i] * x[i];
        }
        fused = edgexpu_gguf_dot_quant(EDGEXPU_GGUF_TYPE_Q5_K, q5, x, QK_K);
        if (fabsf(fused - expanded) > 1e-4f || fabsf(fused - 6.0f) > 1e-3f) {
            snprintf(error, error_size, "q5_k fused dot selftest failed fused=%.6f expanded=%.6f", fused, expanded);
            return 0;
        }
    }

    {
        uint8_t w[34];
        uint8_t xq[34];
        float xv[32];
        uint16_t one = 0x3C00;
        float fused_q8;
        memset(w, 0, sizeof(w));
        memcpy(w, &one, 2);
        for (i = 0; i < 32; i++) {
            w[2 + i] = 1;
            xv[i] = 1.0f;
        }
        if (!edgexpu_gguf_quantize_q8_0(xv, 32, xq)) {
            snprintf(error, error_size, "q8_0 quantize selftest failed");
            return 0;
        }
        fused_q8 = edgexpu_gguf_dot_quant_q8(EDGEXPU_GGUF_TYPE_Q8_0, w, xq, 32);
        if (fabsf(fused_q8 - 32.0f) > 0.5f) {
            snprintf(error, error_size, "q8_0 integer dot selftest failed fused=%.6f", fused_q8);
            return 0;
        }
    }

    {
        uint8_t w[22];
        uint8_t xq[34];
        float xv[32];
        uint16_t one = 0x3C00;
        float fused_q8;
        memset(w, 0, sizeof(w));
        memcpy(w, &one, 2);
        for (i = 0; i < 32; i++) {
            xv[i] = 1.0f;
        }
        if (!edgexpu_gguf_quantize_q8_0(xv, 32, xq)) {
            snprintf(error, error_size, "q5_0 q8 quantize selftest failed");
            return 0;
        }
        fused_q8 = edgexpu_gguf_dot_quant_q8(EDGEXPU_GGUF_TYPE_Q5_0, w, xq, 32);
        if (fabsf(fused_q8 + 512.0f) > 1.0f) {
            snprintf(error, error_size, "q5_0 integer dot selftest failed fused=%.6f", fused_q8);
            return 0;
        }
    }
    return 1;
}
