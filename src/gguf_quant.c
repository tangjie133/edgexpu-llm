#include "edgexpu/gguf_quant.h"

#include <stdio.h>
#include <string.h>

#if defined(EDGEXPU_AVX2)
#include <immintrin.h>
#endif

#define QK8_0 32
#define QK5_0 32
#define QK_K 256
#define Q8_0_BLOCK_BYTES 34
#define Q5_0_BLOCK_BYTES 22
#define Q4_K_BLOCK_BYTES 144
#define Q6_K_BLOCK_BYTES 210

static void set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
    }
}

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

size_t edgexpu_gguf_type_block_size(uint32_t type) {
    switch (type) {
        case EDGEXPU_GGUF_TYPE_F32:
            return 1;
        case EDGEXPU_GGUF_TYPE_Q5_0:
        case EDGEXPU_GGUF_TYPE_Q8_0:
            return QK8_0;
        case EDGEXPU_GGUF_TYPE_Q4_K:
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
        case EDGEXPU_GGUF_TYPE_Q5_0:
            return Q5_0_BLOCK_BYTES;
        case EDGEXPU_GGUF_TYPE_Q8_0:
            return Q8_0_BLOCK_BYTES;
        case EDGEXPU_GGUF_TYPE_Q4_K:
            return Q4_K_BLOCK_BYTES;
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

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

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

#if defined(EDGEXPU_AVX2)
static float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehdup_ps(s));
    s = _mm_add_ss(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(s);
}
#endif

float edgexpu_gguf_q8_0_dot_row(
    const uint8_t *src,
    uint64_t row,
    int n_embd,
    const float *x
) {
    size_t blocks;
    const uint8_t *block;
    float acc = 0.0f;
    size_t b;

    if (src == NULL || x == NULL || n_embd <= 0 || n_embd % QK8_0 != 0) {
        return 0.0f;
    }

    blocks = (size_t)n_embd / QK8_0;
    block = src + row * blocks * Q8_0_BLOCK_BYTES;
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
                __m256 qf;
                __m256 xf;
                memcpy(&packed, qs + off, 8);
                q8 = _mm_cvtsi64_si128(packed);
                qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
                xf = _mm256_loadu_ps(x + off);
#if defined(EDGEXPU_FMA)
                accv = _mm256_fmadd_ps(qf, xf, accv);
#else
                accv = _mm256_add_ps(accv, _mm256_mul_ps(qf, xf));
#endif
            }
            acc += d * hsum256(accv);
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
