#include "edgexpu/gguf_quant.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* llama.cpp 同款：激活量化一次，权重保持压缩，寄存器内整数 SIMD 点积。
 * 不要先 unpack 到 int8[32] 再点积——那会比 fused f32 更慢。
 */

#if defined(EDGEXPU_AVX2) || defined(EDGEXPU_F16C)
#include <immintrin.h>
#endif
#if defined(EDGEXPU_NEON)
#include <arm_neon.h>
#endif

#define QK8_0 32
#define QK_K 256
#define Q8_0_BLOCK_BYTES 34
#define Q8_K_BLOCK_BYTES 292
#define Q5_0_BLOCK_BYTES 22
#define Q4_0_BLOCK_BYTES 18
#define Q4_K_BLOCK_BYTES 144
#define Q5_K_BLOCK_BYTES 176
#define Q6_K_BLOCK_BYTES 210

#if defined(EDGEXPU_F16C)
static float fp16_to_fp32(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}

static uint16_t fp32_to_fp16(float f) {
    return (uint16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0);
}
#else
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h >> 15) & 1u) << 31;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    float value;

    if (exp == 0) {
        bits = mant == 0 ? sign : 0;
        if (mant != 0) {
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

static uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    uint32_t sign;
    int32_t exp;
    uint32_t mant;

    memcpy(&x, &f, 4);
    sign = (x >> 16) & 0x8000u;
    exp = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        return (uint16_t)sign;
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}
#endif

static float load_fp16(const uint8_t *p) {
    uint16_t bits;
    memcpy(&bits, p, 2);
    return fp16_to_fp32(bits);
}

static float load_q8_d(const uint8_t *block) {
    return load_fp16(block);
}

static const int8_t *q8_0_qs(const uint8_t *block) {
    return (const int8_t *)(block + 2);
}

static float q8k_d(const uint8_t *block) {
    float d;
    memcpy(&d, block, 4);
    return d;
}

static const int8_t *q8k_qs(const uint8_t *block) {
    return (const int8_t *)(block + 4);
}

static const int16_t *q8k_bsums(const uint8_t *block) {
    return (const int16_t *)(block + 4 + QK_K);
}

static void unpack_k4_scales(const uint8_t *scales12, uint32_t utmp[4]) {
    static const uint32_t kmask1 = 0x3f3f3f3fu;
    static const uint32_t kmask2 = 0x0f0f0f0fu;
    static const uint32_t kmask3 = 0x03030303u;
    uint32_t uaux;

    memcpy(utmp, scales12, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
}

#if defined(EDGEXPU_AVX2)
#define MM256_SET_M128I(hi, lo) _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1)

static inline float hsum_float_8(__m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float(dot);
#endif
}

static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_float(ax, sy);
}

static inline __m256i bytes_from_bits_32(const uint8_t *x) {
    uint32_t x32;
    __m256i bytes;
    const __m256i shuf_mask = _mm256_set_epi64x(
        0x0303030303030303LL,
        0x0202020202020202LL,
        0x0101010101010101LL,
        0x0000000000000000LL
    );
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL);
    memcpy(&x32, x, sizeof(x32));
    bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int)x32), shuf_mask);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

static inline __m256i bytes_from_nibbles_32(const uint8_t *rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i low_mask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(low_mask, bytes);
}

static inline void acc_fmadd(__m256 *acc, __m256 d, __m256 q) {
#if defined(__FMA__) || defined(EDGEXPU_FMA)
    *acc = _mm256_fmadd_ps(d, q, *acc);
#else
    *acc = _mm256_add_ps(_mm256_mul_ps(d, q), *acc);
#endif
}

static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
        6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15
    };
    return _mm256_loadu_si256((const __m256i *)k_shuffle + i);
}

static inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15
    };
    return _mm_loadu_si128((const __m128i *)k_shuffle + i);
}
#endif

#if defined(EDGEXPU_NEON)
static inline int32x4_t i8dot_acc(int32x4_t acc, int8x16_t a, int8x16_t b) {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, a, b);
#else
    {
        int16x8_t lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
        int16x8_t hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
        return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(lo), vpaddlq_s16(hi)));
    }
#endif
}

static uint64_t table_b2b_1[256];
static int table_b2b_ready;

static void ensure_b2b_table(void) {
    int i;
    int b;
    if (table_b2b_ready) {
        return;
    }
    for (i = 0; i < 256; i++) {
        uint64_t v = 0;
        for (b = 0; b < 8; b++) {
            if ((i & (1 << b)) == 0) {
                v |= (uint64_t)0x10u << (8 * b);
            }
        }
        table_b2b_1[i] = v;
    }
    table_b2b_ready = 1;
}
#endif

size_t edgexpu_gguf_q8_0_nbytes(int n) {
    if (n <= 0 || n % QK8_0 != 0) {
        return 0;
    }
    return ((size_t)n / QK8_0) * Q8_0_BLOCK_BYTES;
}

size_t edgexpu_gguf_q8_k_nbytes(int n) {
    if (n <= 0 || n % QK_K != 0) {
        return 0;
    }
    return ((size_t)n / QK_K) * Q8_K_BLOCK_BYTES;
}

int edgexpu_gguf_quantize_q8_0(const float *x, int n, uint8_t *dst) {
    int nb;
    int b;
    if (x == NULL || dst == NULL || n <= 0 || n % QK8_0 != 0) {
        return 0;
    }
    nb = n / QK8_0;
#if defined(EDGEXPU_AVX2)
    for (b = 0; b < nb; b++) {
        const float *xx = x + b * QK8_0;
        uint8_t *block = dst + (size_t)b * Q8_0_BLOCK_BYTES;
        __m256 v0 = _mm256_loadu_ps(xx);
        __m256 v1 = _mm256_loadu_ps(xx + 8);
        __m256 v2 = _mm256_loadu_ps(xx + 16);
        __m256 v3 = _mm256_loadu_ps(xx + 24);
        const __m256 sign_bit = _mm256_set1_ps(-0.0f);
        __m256 max_abs = _mm256_andnot_ps(sign_bit, v0);
        __m128 max4;
        float max_scalar;
        float d;
        float id;
        uint16_t dh;
        __m256 mul;
        __m256i i0;
        __m256i i1;
        __m256i i2;
        __m256i i3;
        const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

        max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v1));
        max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v2));
        max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v3));
        max4 = _mm_max_ps(_mm256_extractf128_ps(max_abs, 1), _mm256_castps256_ps128(max_abs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        max_scalar = _mm_cvtss_f32(max4);
        d = max_scalar / 127.0f;
        id = max_scalar != 0.0f ? 127.0f / max_scalar : 0.0f;
        dh = fp32_to_fp16(d);
        memcpy(block, &dh, 2);
        mul = _mm256_set1_ps(id);
        v0 = _mm256_round_ps(_mm256_mul_ps(v0, mul), _MM_FROUND_TO_NEAREST_INT);
        v1 = _mm256_round_ps(_mm256_mul_ps(v1, mul), _MM_FROUND_TO_NEAREST_INT);
        v2 = _mm256_round_ps(_mm256_mul_ps(v2, mul), _MM_FROUND_TO_NEAREST_INT);
        v3 = _mm256_round_ps(_mm256_mul_ps(v3, mul), _MM_FROUND_TO_NEAREST_INT);
        i0 = _mm256_cvtps_epi32(v0);
        i1 = _mm256_cvtps_epi32(v1);
        i2 = _mm256_cvtps_epi32(v2);
        i3 = _mm256_cvtps_epi32(v3);
        i0 = _mm256_packs_epi32(i0, i1);
        i2 = _mm256_packs_epi32(i2, i3);
        i0 = _mm256_packs_epi16(i0, i2);
        i0 = _mm256_permutevar8x32_epi32(i0, perm);
        _mm256_storeu_si256((__m256i *)(block + 2), i0);
    }
    return 1;
#else
    for (b = 0; b < nb; b++) {
        const float *xx = x + b * QK8_0;
        uint8_t *block = dst + (size_t)b * Q8_0_BLOCK_BYTES;
        float amax = 0.0f;
        float d;
        float id;
        uint16_t dh;
        int8_t *qs = (int8_t *)(block + 2);
        int i;
        for (i = 0; i < QK8_0; i++) {
            float ax = fabsf(xx[i]);
            if (ax > amax) {
                amax = ax;
            }
        }
        d = amax / 127.0f;
        id = d > 0.0f ? 1.0f / d : 0.0f;
        dh = fp32_to_fp16(d);
        memcpy(block, &dh, 2);
        for (i = 0; i < QK8_0; i++) {
            int v = (int)lrintf(xx[i] * id);
            if (v > 127) {
                v = 127;
            }
            if (v < -127) {
                v = -127;
            }
            qs[i] = (int8_t)v;
        }
    }
    return 1;
#endif
}

int edgexpu_gguf_quantize_q8_k(const float *x, int n, uint8_t *dst) {
    int nb;
    int i;
    if (x == NULL || dst == NULL || n <= 0 || n % QK_K != 0) {
        return 0;
    }
    nb = n / QK_K;
    for (i = 0; i < nb; i++) {
        const float *xx = x + i * QK_K;
        uint8_t *block = dst + (size_t)i * Q8_K_BLOCK_BYTES;
        int8_t *qs = (int8_t *)(block + 4);
        int16_t bsums[16];
        float amax = 0.0f;
        float maxv = 0.0f;
        float iscale;
        float d;
        int j;
        for (j = 0; j < QK_K; j++) {
            float ax = fabsf(xx[j]);
            if (ax > amax) {
                amax = ax;
                maxv = xx[j];
            }
        }
        if (amax == 0.0f) {
            memset(block, 0, Q8_K_BLOCK_BYTES);
            continue;
        }
        iscale = -127.0f / maxv;
        for (j = 0; j < QK_K; j++) {
            int v = (int)lrintf(iscale * xx[j]);
            if (v > 127) {
                v = 127;
            }
            if (v < -127) {
                v = -127;
            }
            qs[j] = (int8_t)v;
        }
        for (j = 0; j < 16; j++) {
            int sum = 0;
            int ii;
            for (ii = 0; ii < 16; ii++) {
                sum += (int)qs[j * 16 + ii];
            }
            bsums[j] = (int16_t)sum;
        }
        d = 1.0f / iscale;
        memcpy(block, &d, 4);
        memcpy(block + 4 + QK_K, bsums, sizeof(bsums));
    }
    return 1;
}

int edgexpu_gguf_can_dot_q8(uint32_t type) {
    switch (type) {
        case EDGEXPU_GGUF_TYPE_Q4_0:
        case EDGEXPU_GGUF_TYPE_Q5_0:
        case EDGEXPU_GGUF_TYPE_Q8_0:
            return 1;
        default:
            return 0;
    }
}

int edgexpu_gguf_can_dot_q8k(uint32_t type) {
    switch (type) {
        case EDGEXPU_GGUF_TYPE_Q4_K:
        case EDGEXPU_GGUF_TYPE_Q5_K:
        case EDGEXPU_GGUF_TYPE_Q6_K:
            return 1;
        default:
            return 0;
    }
}

static float dot_q8_0_q8_0(const uint8_t *wrow, const uint8_t *x_q8, int n) {
    int nb = n / QK8_0;
    int b = 0;
    float acc = 0.0f;
#if defined(EDGEXPU_AVX2)
    {
        __m256 vacc = _mm256_setzero_ps();
        for (; b < nb; b++) {
            const uint8_t *wb = wrow + (size_t)b * Q8_0_BLOCK_BYTES;
            const uint8_t *xb = x_q8 + (size_t)b * Q8_0_BLOCK_BYTES;
            const __m256 d = _mm256_set1_ps(load_q8_d(wb) * load_q8_d(xb));
            const __m256i qx = _mm256_loadu_si256((const __m256i *)q8_0_qs(wb));
            const __m256i qy = _mm256_loadu_si256((const __m256i *)q8_0_qs(xb));
            acc_fmadd(&vacc, d, mul_sum_i8_pairs_float(qx, qy));
        }
        return hsum_float_8(vacc);
    }
#elif defined(EDGEXPU_NEON)
    for (; b < nb; b++) {
        const uint8_t *wb = wrow + (size_t)b * Q8_0_BLOCK_BYTES;
        const uint8_t *xb = x_q8 + (size_t)b * Q8_0_BLOCK_BYTES;
        int8x16_t a0 = vld1q_s8(q8_0_qs(wb));
        int8x16_t a1 = vld1q_s8(q8_0_qs(wb) + 16);
        int8x16_t y0 = vld1q_s8(q8_0_qs(xb));
        int8x16_t y1 = vld1q_s8(q8_0_qs(xb) + 16);
        int32x4_t p = i8dot_acc(vdupq_n_s32(0), a0, y0);
        p = i8dot_acc(p, a1, y1);
        acc += load_q8_d(wb) * load_q8_d(xb) * (float)vaddvq_s32(p);
    }
    return acc;
#else
    for (; b < nb; b++) {
        const int8_t *wq = q8_0_qs(wrow + (size_t)b * Q8_0_BLOCK_BYTES);
        const int8_t *xq = q8_0_qs(x_q8 + (size_t)b * Q8_0_BLOCK_BYTES);
        int32_t ip = 0;
        int j;
        for (j = 0; j < QK8_0; j++) {
            ip += (int32_t)wq[j] * (int32_t)xq[j];
        }
        acc += load_q8_d(wrow + (size_t)b * Q8_0_BLOCK_BYTES) *
            load_q8_d(x_q8 + (size_t)b * Q8_0_BLOCK_BYTES) * (float)ip;
    }
    return acc;
#endif
}

static float dot_q5_0_q8_0(const uint8_t *wrow, const uint8_t *x_q8, int n) {
    int nb = n / QK8_0;
    int b = 0;
    float acc = 0.0f;
#if defined(EDGEXPU_AVX2)
    {
        __m256 vacc = _mm256_setzero_ps();
        const __m256i high_nibble = _mm256_set1_epi8((char)0xF0);
        for (; b < nb; b++) {
            const uint8_t *wb = wrow + (size_t)b * Q5_0_BLOCK_BYTES;
            const uint8_t *xb = x_q8 + (size_t)b * Q8_0_BLOCK_BYTES;
            const __m256 d = _mm256_set1_ps(load_fp16(wb) * load_q8_d(xb));
            __m256i qx = bytes_from_nibbles_32(wb + 6);
            __m256i bxhi = bytes_from_bits_32(wb + 2);
            bxhi = _mm256_andnot_si256(bxhi, high_nibble);
            qx = _mm256_or_si256(qx, bxhi);
            acc_fmadd(&vacc, d, mul_sum_i8_pairs_float(qx, _mm256_loadu_si256((const __m256i *)q8_0_qs(xb))));
        }
        return hsum_float_8(vacc);
    }
#elif defined(EDGEXPU_NEON)
    ensure_b2b_table();
    for (; b < nb; b++) {
        const uint8_t *wb = wrow + (size_t)b * Q5_0_BLOCK_BYTES;
        const uint8_t *xb = x_q8 + (size_t)b * Q8_0_BLOCK_BYTES;
        uint32_t qh;
        uint64_t tmp[4];
        uint8x16_t qs;
        int8x16_t lo;
        int8x16_t hi;
        int32x4_t p;
        memcpy(&qh, wb + 2, 4);
        tmp[0] = table_b2b_1[(qh >> 0) & 0xFFu];
        tmp[1] = table_b2b_1[(qh >> 8) & 0xFFu];
        tmp[2] = table_b2b_1[(qh >> 16) & 0xFFu];
        tmp[3] = table_b2b_1[qh >> 24];
        qs = vld1q_u8(wb + 6);
        lo = vreinterpretq_s8_u8(vandq_u8(qs, vdupq_n_u8(0x0F)));
        hi = vreinterpretq_s8_u8(vshrq_n_u8(qs, 4));
        lo = vsubq_s8(lo, vld1q_s8((const int8_t *)(tmp + 0)));
        hi = vsubq_s8(hi, vld1q_s8((const int8_t *)(tmp + 2)));
        p = i8dot_acc(vdupq_n_s32(0), lo, vld1q_s8(q8_0_qs(xb)));
        p = i8dot_acc(p, hi, vld1q_s8(q8_0_qs(xb) + 16));
        acc += load_fp16(wb) * load_q8_d(xb) * (float)vaddvq_s32(p);
    }
    return acc;
#else
    for (; b < nb; b++) {
        const uint8_t *wb = wrow + (size_t)b * Q5_0_BLOCK_BYTES;
        const int8_t *xq = q8_0_qs(x_q8 + (size_t)b * Q8_0_BLOCK_BYTES);
        const uint8_t *qs = wb + 6;
        uint32_t qh;
        int32_t sumi0 = 0;
        int32_t sumi1 = 0;
        int j;
        memcpy(&qh, wb + 2, 4);
        for (j = 0; j < 16; j++) {
            uint8_t xh0 = (uint8_t)(((qh & (1u << (unsigned)j)) >> (unsigned)j) << 4);
            uint8_t xh1 = (uint8_t)((qh >> (unsigned)(j + 12)) & 0x10u);
            int32_t x0 = (int8_t)(((qs[j] & 0x0F) | xh0) - 16);
            int32_t x1 = (int8_t)(((qs[j] >> 4) | xh1) - 16);
            sumi0 += x0 * (int32_t)xq[j];
            sumi1 += x1 * (int32_t)xq[j + 16];
        }
        acc += load_fp16(wb) * load_q8_d(x_q8 + (size_t)b * Q8_0_BLOCK_BYTES) * (float)(sumi0 + sumi1);
    }
    return acc;
#endif
}

static float dot_q4_0_q8_0(const uint8_t *wrow, const uint8_t *x_q8, int n) {
    int nb = n / QK8_0;
    int b = 0;
    float acc = 0.0f;
#if defined(EDGEXPU_AVX2)
    {
        __m256 vacc = _mm256_setzero_ps();
        const __m256i off = _mm256_set1_epi8(8);
        for (; b < nb; b++) {
            const uint8_t *wb = wrow + (size_t)b * Q4_0_BLOCK_BYTES;
            const uint8_t *xb = x_q8 + (size_t)b * Q8_0_BLOCK_BYTES;
            const __m256 d = _mm256_set1_ps(load_fp16(wb) * load_q8_d(xb));
            __m256i qx = bytes_from_nibbles_32(wb + 2);
            qx = _mm256_sub_epi8(qx, off);
            acc_fmadd(&vacc, d, mul_sum_i8_pairs_float(qx, _mm256_loadu_si256((const __m256i *)q8_0_qs(xb))));
        }
        return hsum_float_8(vacc);
    }
#elif defined(EDGEXPU_NEON)
    for (; b < nb; b++) {
        const uint8_t *wb = wrow + (size_t)b * Q4_0_BLOCK_BYTES;
        const uint8_t *xb = x_q8 + (size_t)b * Q8_0_BLOCK_BYTES;
        uint8x16_t qs = vld1q_u8(wb + 2);
        int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(qs, vdupq_n_u8(0x0F)));
        int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(qs, 4));
        int32x4_t p;
        lo = vsubq_s8(lo, vdupq_n_s8(8));
        hi = vsubq_s8(hi, vdupq_n_s8(8));
        p = i8dot_acc(vdupq_n_s32(0), lo, vld1q_s8(q8_0_qs(xb)));
        p = i8dot_acc(p, hi, vld1q_s8(q8_0_qs(xb) + 16));
        acc += load_fp16(wb) * load_q8_d(xb) * (float)vaddvq_s32(p);
    }
    return acc;
#else
    for (; b < nb; b++) {
        const uint8_t *wb = wrow + (size_t)b * Q4_0_BLOCK_BYTES;
        const int8_t *xq = q8_0_qs(x_q8 + (size_t)b * Q8_0_BLOCK_BYTES);
        const uint8_t *qs = wb + 2;
        int32_t sumi0 = 0;
        int32_t sumi1 = 0;
        int j;
        for (j = 0; j < 16; j++) {
            sumi0 += ((int32_t)(qs[j] & 0x0F) - 8) * (int32_t)xq[j];
            sumi1 += ((int32_t)(qs[j] >> 4) - 8) * (int32_t)xq[j + 16];
        }
        acc += load_fp16(wb) * load_q8_d(x_q8 + (size_t)b * Q8_0_BLOCK_BYTES) * (float)(sumi0 + sumi1);
    }
    return acc;
#endif
}

static float dot_q4_k_q8_k(const uint8_t *wrow, const uint8_t *x_q8k, int n) {
    int nb;
    int i;
    float acc = 0.0f;
    if (n % QK_K != 0) {
        return 0.0f;
    }
    nb = n / QK_K;
#if defined(EDGEXPU_AVX2)
    {
        const __m256i m4 = _mm256_set1_epi8(0xF);
        __m256 vacc = _mm256_setzero_ps();
        __m128 acc_m = _mm_setzero_ps();
        for (i = 0; i < nb; i++) {
            const uint8_t *block = wrow + (size_t)i * Q4_K_BLOCK_BYTES;
            const uint8_t *yb = x_q8k + (size_t)i * Q8_K_BLOCK_BYTES;
            const float d = q8k_d(yb) * load_fp16(block);
            const float dmin = -q8k_d(yb) * load_fp16(block + 2);
            uint32_t utmp[4];
            const uint8_t *q4;
            const int8_t *q8;
            __m256i mins_and_scales;
            __m256i q8sums;
            __m128i q8s;
            __m128i prod;
            __m128i sc128;
            __m256i scales;
            __m256i sumi;
            int j;
            unpack_k4_scales(block + 4, utmp);
            q4 = block + 16;
            q8 = q8k_qs(yb);
            mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));
            q8sums = _mm256_loadu_si256((const __m256i *)q8k_bsums(yb));
            q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
#if defined(__FMA__) || defined(EDGEXPU_FMA)
            acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);
#else
            acc_m = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)), acc_m);
#endif
            sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            scales = MM256_SET_M128I(sc128, sc128);
            sumi = _mm256_setzero_si256();
            for (j = 0; j < QK_K / 64; j++) {
                const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
                const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
                const __m256i q4bits = _mm256_loadu_si256((const __m256i *)q4);
                const __m256i q4l = _mm256_and_si256(q4bits, m4);
                const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                const __m256i q8l = _mm256_loadu_si256((const __m256i *)q8);
                const __m256i q8h = _mm256_loadu_si256((const __m256i *)(q8 + 32));
                __m256i p16l = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l));
                __m256i p16h = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
                q4 += 32;
                q8 += 64;
            }
            acc_fmadd(&vacc, _mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi));
        }
        acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
        acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
        return hsum_float_8(vacc) + _mm_cvtss_f32(acc_m);
    }
#else
    for (i = 0; i < nb; i++) {
        const uint8_t *block = wrow + (size_t)i * Q4_K_BLOCK_BYTES;
        const uint8_t *yb = x_q8k + (size_t)i * Q8_K_BLOCK_BYTES;
        const int8_t *q8 = q8k_qs(yb);
        const int16_t *bsums = q8k_bsums(yb);
        const uint8_t *q4 = block + 16;
        uint32_t utmp[4];
        const uint8_t *scales;
        const uint8_t *mins;
        float d;
        float dmin;
        int8_t aux8[QK_K];
        int8_t *a;
        int j;
        int l;
        int sumi = 0;
        unpack_k4_scales(block + 4, utmp);
        scales = (const uint8_t *)&utmp[0];
        mins = (const uint8_t *)&utmp[2];
        d = load_fp16(block) * q8k_d(yb);
        dmin = load_fp16(block + 2) * q8k_d(yb);
        a = aux8;
        for (j = 0; j < QK_K / 64; j++) {
            for (l = 0; l < 32; l++) {
                a[l] = (int8_t)(q4[l] & 0x0F);
            }
            a += 32;
            for (l = 0; l < 32; l++) {
                a[l] = (int8_t)(q4[l] >> 4);
            }
            a += 32;
            q4 += 32;
        }
        for (j = 0; j < QK_K / 16; j++) {
            sumi += (int)bsums[j] * (int)mins[j / 2];
        }
        acc -= dmin * (float)sumi;
        a = aux8;
        for (j = 0; j < QK_K / 32; j++) {
            int32_t scale = (int32_t)scales[j];
            int32_t ip = 0;
            for (l = 0; l < 32; l++) {
                ip += (int32_t)q8[l] * (int32_t)a[l];
            }
            acc += d * (float)(scale * ip);
            q8 += 32;
            a += 32;
        }
    }
    return acc;
#endif
}

static float dot_q5_k_q8_k(const uint8_t *wrow, const uint8_t *x_q8k, int n) {
    int nb;
    int i;
    float acc = 0.0f;
    if (n % QK_K != 0) {
        return 0.0f;
    }
    nb = n / QK_K;
#if defined(EDGEXPU_AVX2)
    {
        const __m256i m4 = _mm256_set1_epi8(0xF);
        const __m128i mzero = _mm_setzero_si128();
        const __m256i mone = _mm256_set1_epi8(1);
        __m256 vacc = _mm256_setzero_ps();
        float summs = 0.0f;
        for (i = 0; i < nb; i++) {
            const uint8_t *block = wrow + (size_t)i * Q5_K_BLOCK_BYTES;
            const uint8_t *yb = x_q8k + (size_t)i * Q8_K_BLOCK_BYTES;
            const uint8_t *q5 = block + 48;
            const int8_t *q8 = q8k_qs(yb);
            const float d = q8k_d(yb) * load_fp16(block);
            const float dmin = -q8k_d(yb) * load_fp16(block + 2);
            uint32_t utmp[4];
            __m256i mins_and_scales;
            __m256i q8sums;
            __m128i q8s;
            __m128i prod;
            __m128i hsum;
            __m128i sc128;
            __m256i scales;
            __m256i hbits;
            __m256i hmask;
            __m256i sumi;
            int bit = 0;
            int j;
            unpack_k4_scales(block + 4, utmp);
            mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32((int)utmp[3], (int)utmp[2], (int)utmp[1], (int)utmp[0]));
            q8sums = _mm256_loadu_si256((const __m256i *)q8k_bsums(yb));
            q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
            hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
            summs += dmin * (float)_mm_extract_epi32(hsum, 0);
            sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            scales = MM256_SET_M128I(sc128, sc128);
            hbits = _mm256_loadu_si256((const __m256i *)(block + 16));
            hmask = mone;
            sumi = _mm256_setzero_si256();
            for (j = 0; j < QK_K / 64; j++) {
                const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
                const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
                const __m256i q5bits = _mm256_loadu_si256((const __m256i *)q5);
                const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
                const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit), 4);
                const __m256i q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
                __m256i q5l_1;
                __m256i q5h_1;
                __m256i q5_1;
                __m256i p16_0;
                __m256i p16_1;
                bit++;
                hmask = _mm256_slli_epi16(hmask, 1);
                q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
                q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit), 4);
                q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
                bit++;
                hmask = _mm256_slli_epi16(hmask, 1);
                p16_0 = _mm256_madd_epi16(scale_0, _mm256_maddubs_epi16(q5_0, _mm256_loadu_si256((const __m256i *)q8)));
                p16_1 = _mm256_madd_epi16(scale_1, _mm256_maddubs_epi16(q5_1, _mm256_loadu_si256((const __m256i *)(q8 + 32))));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                q5 += 32;
                q8 += 64;
            }
            acc_fmadd(&vacc, _mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi));
        }
        return hsum_float_8(vacc) + summs;
    }
#else
    for (i = 0; i < nb; i++) {
        const uint8_t *block = wrow + (size_t)i * Q5_K_BLOCK_BYTES;
        const uint8_t *yb = x_q8k + (size_t)i * Q8_K_BLOCK_BYTES;
        const uint8_t *qh = block + 16;
        const uint8_t *ql = block + 48;
        const int8_t *q8 = q8k_qs(yb);
        const int16_t *bsums = q8k_bsums(yb);
        uint32_t utmp[4];
        const uint8_t *scales;
        const uint8_t *mins;
        float d;
        float dmin;
        int8_t aux8[QK_K];
        int8_t *a = aux8;
        uint8_t m = 1;
        int j;
        int l;
        int sumi = 0;
        unpack_k4_scales(block + 4, utmp);
        scales = (const uint8_t *)&utmp[0];
        mins = (const uint8_t *)&utmp[2];
        d = load_fp16(block) * q8k_d(yb);
        dmin = load_fp16(block + 2) * q8k_d(yb);
        for (j = 0; j < QK_K / 64; j++) {
            for (l = 0; l < 32; l++) {
                a[l] = (int8_t)((ql[l] & 0x0F) + ((qh[l] & m) ? 16 : 0));
            }
            a += 32;
            m = (uint8_t)(m << 1);
            for (l = 0; l < 32; l++) {
                a[l] = (int8_t)((ql[l] >> 4) + ((qh[l] & m) ? 16 : 0));
            }
            a += 32;
            m = (uint8_t)(m << 1);
            ql += 32;
        }
        for (j = 0; j < QK_K / 16; j++) {
            sumi += (int)bsums[j] * (int)mins[j / 2];
        }
        acc -= dmin * (float)sumi;
        a = aux8;
        for (j = 0; j < QK_K / 32; j++) {
            int32_t scale = (int32_t)scales[j];
            int32_t ip = 0;
            for (l = 0; l < 32; l++) {
                ip += (int32_t)q8[l] * (int32_t)a[l];
            }
            acc += d * (float)(scale * ip);
            q8 += 32;
            a += 32;
        }
    }
    return acc;
#endif
}

static float dot_q6_k_q8_k(const uint8_t *wrow, const uint8_t *x_q8k, int n) {
    int nb;
    int i;
    float acc = 0.0f;
    if (n % QK_K != 0) {
        return 0.0f;
    }
    nb = n / QK_K;
#if defined(EDGEXPU_AVX2)
    {
        const __m256i m3 = _mm256_set1_epi8(3);
        const __m256i m15 = _mm256_set1_epi8(15);
        __m256 vacc = _mm256_setzero_ps();
        for (i = 0; i < nb; i++) {
            const uint8_t *block = wrow + (size_t)i * Q6_K_BLOCK_BYTES;
            const uint8_t *yb = x_q8k + (size_t)i * Q8_K_BLOCK_BYTES;
            const uint8_t *q4 = block;
            const uint8_t *qh = block + QK_K / 2;
            const int8_t *q8 = q8k_qs(yb);
            const float d = q8k_d(yb) * load_fp16(block + Q6_K_BLOCK_BYTES - 2);
            const __m256i q8sums = _mm256_loadu_si256((const __m256i *)q8k_bsums(yb));
            const __m128i scales = _mm_loadu_si128((const __m128i *)(block + QK_K / 2 + QK_K / 4));
            const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
            const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);
            __m256i sumi = _mm256_setzero_si256();
            int is = 0;
            int j;
            for (j = 0; j < QK_K / 128; j++) {
                const __m256i q4bits1 = _mm256_loadu_si256((const __m256i *)q4);
                const __m256i q4bits2 = _mm256_loadu_si256((const __m256i *)(q4 + 32));
                const __m256i q4bitsH = _mm256_loadu_si256((const __m256i *)qh);
                const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
                const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
                const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
                const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8((char)-64)), 2);
                const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
                const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
                const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
                const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8);
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)(q8 + 32));
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)(q8 + 64));
                const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)(q8 + 96));
                __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
                __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
                __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
                __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);
                const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
                const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
                const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
                const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
                is += 4;
                p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
                p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
                p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
                p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
                q4 += 64;
                qh += 32;
                q8 += 128;
            }
            sumi = _mm256_sub_epi32(sumi, q8sclsub);
            acc_fmadd(&vacc, _mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi));
        }
        return hsum_float_8(vacc);
    }
#else
    for (i = 0; i < nb; i++) {
        const uint8_t *block = wrow + (size_t)i * Q6_K_BLOCK_BYTES;
        const uint8_t *yb = x_q8k + (size_t)i * Q8_K_BLOCK_BYTES;
        const uint8_t *ql = block;
        const uint8_t *qh = block + QK_K / 2;
        const int8_t *sc = (const int8_t *)(block + QK_K / 2 + QK_K / 4);
        const int8_t *q8 = q8k_qs(yb);
        float d = load_fp16(block + Q6_K_BLOCK_BYTES - 2) * q8k_d(yb);
        int8_t aux8[QK_K];
        int8_t *a = aux8;
        int j;
        int l;
        for (j = 0; j < QK_K; j += 128) {
            for (l = 0; l < 32; l++) {
                a[l + 0] = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a += 128;
            ql += 64;
            qh += 32;
        }
        a = aux8;
        for (j = 0; j < QK_K / 16; j++) {
            int32_t scale = (int32_t)sc[j];
            int32_t ip = 0;
            for (l = 0; l < 16; l++) {
                ip += (int32_t)q8[l] * (int32_t)a[l];
            }
            acc += d * (float)(scale * ip);
            q8 += 16;
            a += 16;
        }
    }
    return acc;
#endif
}

float edgexpu_gguf_dot_quant_q8(
    uint32_t type,
    const uint8_t *wrow,
    const uint8_t *x_q8,
    int n
) {
    if (wrow == NULL || x_q8 == NULL || n <= 0 || n % QK8_0 != 0) {
        return 0.0f;
    }
    switch (type) {
        case EDGEXPU_GGUF_TYPE_Q8_0:
            return dot_q8_0_q8_0(wrow, x_q8, n);
        case EDGEXPU_GGUF_TYPE_Q5_0:
            return dot_q5_0_q8_0(wrow, x_q8, n);
        case EDGEXPU_GGUF_TYPE_Q4_0:
            return dot_q4_0_q8_0(wrow, x_q8, n);
        default:
            return 0.0f;
    }
}

float edgexpu_gguf_dot_quant_q8k(
    uint32_t type,
    const uint8_t *wrow,
    const uint8_t *x_q8k,
    int n
) {
    if (wrow == NULL || x_q8k == NULL || n <= 0 || n % QK_K != 0) {
        return 0.0f;
    }
    switch (type) {
        case EDGEXPU_GGUF_TYPE_Q4_K:
            return dot_q4_k_q8_k(wrow, x_q8k, n);
        case EDGEXPU_GGUF_TYPE_Q5_K:
            return dot_q5_k_q8_k(wrow, x_q8k, n);
        case EDGEXPU_GGUF_TYPE_Q6_K:
            return dot_q6_k_q8_k(wrow, x_q8k, n);
        default:
            return 0.0f;
    }
}
