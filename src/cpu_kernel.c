#include "edgexpu/cpu_kernel.h"
#include "edgexpu/gguf_quant.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* f32 算子实现。linear / 点积走 AVX2 或 NEON；RoPE 分 Neox 与 NORM。 */

#if defined(EDGEXPU_AVX2)
#include <immintrin.h>
#endif
#if defined(EDGEXPU_NEON)
#include <arm_neon.h>
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

static int linear_thread_count(int n_out);

/* 0=未测 1=探测中 2=已选定。探测时禁止再进入 tune，避免递归。 */
static int g_thread_tune_state = 0;
static volatile float g_probe_sink;

static void tune_openmp_threads(void) {
#if defined(_OPENMP)
    const int probe_n_out = 4096;
    const int probe_n_in = 896; /* Qwen n_embd，Q5_0 层 GEMV 是 decode 主路径 */
    const int repeats = 5;
    const int max_cand = 16;
    int max_t;
    int best_t;
    int t;
    int n_cand = 0;
    int c;
    float *x = NULL;
    uint8_t *w = NULL;
    float *y = NULL;
    size_t row_bytes;
    double times[16];
    int counts[16];
    double best_seconds;
    int i;

    if (g_thread_tune_state != 0) {
        return;
    }
    g_thread_tune_state = 1;

    /* 用户显式设了 OMP_NUM_THREADS 就尊重它，不再探测。 */
    if (getenv("OMP_NUM_THREADS") != NULL) {
        g_thread_tune_state = 2;
        return;
    }

    max_t = omp_get_max_threads();
    if (max_t <= 1) {
        g_thread_tune_state = 2;
        return;
    }

    row_bytes = edgexpu_gguf_row_bytes(EDGEXPU_GGUF_TYPE_Q5_0, probe_n_in);
    x = (float *)malloc((size_t)probe_n_in * sizeof(float));
    y = (float *)malloc((size_t)probe_n_out * sizeof(float));
    w = row_bytes > 0 ? (uint8_t *)calloc((size_t)probe_n_out * row_bytes, 1) : NULL;
    if (x == NULL || y == NULL || w == NULL || row_bytes == 0) {
        free(x);
        free(y);
        free(w);
        g_thread_tune_state = 2;
        return;
    }
    for (i = 0; i < probe_n_in; i++) {
        x[i] = 0.01f * (float)((i % 8) + 1);
    }

    t = 1;
    while (t <= max_t && n_cand < max_cand) {
        counts[n_cand] = t;
        n_cand++;
        if (t == max_t) {
            break;
        }
        t = t * 2;
        if (t > max_t) {
            t = max_t;
        }
    }

    omp_set_num_threads(max_t);
    for (i = 0; i < 4; i++) {
        edgexpu_cpu_linear_quant_batch(
            y,
            x,
            w,
            EDGEXPU_GGUF_TYPE_Q5_0,
            NULL,
            1,
            probe_n_out,
            probe_n_in
        );
        g_probe_sink += y[0];
    }

    for (c = n_cand - 1; c >= 0; c--) {
        int r;
        int a;
        int b;
        double samples[8];
        double fastest;
        t = counts[c];
        omp_set_num_threads(t);
        edgexpu_cpu_linear_quant_batch(
            y,
            x,
            w,
            EDGEXPU_GGUF_TYPE_Q5_0,
            NULL,
            1,
            probe_n_out,
            probe_n_in
        );
        for (r = 0; r < repeats; r++) {
            double t0 = omp_get_wtime();
            edgexpu_cpu_linear_quant_batch(
                y,
                x,
                w,
                EDGEXPU_GGUF_TYPE_Q5_0,
                NULL,
                1,
                probe_n_out,
                probe_n_in
            );
            samples[r] = omp_get_wtime() - t0;
            g_probe_sink += y[0];
        }
        for (a = 0; a < repeats; a++) {
            for (b = a + 1; b < repeats; b++) {
                if (samples[b] < samples[a]) {
                    double tmp = samples[a];
                    samples[a] = samples[b];
                    samples[b] = tmp;
                }
            }
        }
        fastest = samples[repeats / 2];
        times[c] = fastest;
        if (getenv("EDGEXPU_DEBUG_THREADS") != NULL) {
            fprintf(stderr, "edgexpu thread probe: t=%d median=%.3fms\n", t, fastest * 1000.0);
        }
    }

    best_t = counts[0];
    best_seconds = times[0];
    for (c = 1; c < n_cand; c++) {
        if (times[c] < best_seconds) {
            best_seconds = times[c];
            best_t = counts[c];
        }
    }

    omp_set_num_threads(best_t);
    if (getenv("EDGEXPU_DEBUG_THREADS") != NULL) {
        fprintf(stderr, "edgexpu thread probe: selected %d\n", best_t);
    }
    free(x);
    free(w);
    free(y);
    g_thread_tune_state = 2;
#else
    (void)0;
#endif
}

static int linear_thread_count(int n_out) {
#if defined(_OPENMP)
    int max_t = omp_get_max_threads();
    int t;
    if (n_out < 32 || max_t <= 1) {
        return 1;
    }
    t = n_out / 16;
    if (t < 2) {
        return 1;
    }
    if (t > max_t) {
        t = max_t;
    }
    return t;
#else
    (void)n_out;
    return 1;
#endif
}

void edgexpu_cpu_add(float *out, const float *a, const float *b, int n) {
    int i;
    if (out == NULL || a == NULL || b == NULL) {
        return;
    }
    for (i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void edgexpu_cpu_mul(float *out, const float *a, const float *b, int n) {
    int i;
    if (out == NULL || a == NULL || b == NULL) {
        return;
    }
    for (i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

void edgexpu_cpu_rmsnorm(float *out, const float *x, const float *weight, int n, float eps) {
    int i;
    double ss = 0.0;
    float mean;
    float scale;

    if (out == NULL || x == NULL || weight == NULL || n <= 0) {
        return;
    }

    /* 与 ggml_compute_forward_rms_norm_f32 一致：double 累加平方和，再 scale，再乘 weight。 */
    for (i = 0; i < n; i++) {
        ss += (double)x[i] * (double)x[i];
    }
    mean = (float)(ss / (double)n);
    scale = 1.0f / sqrtf(mean + eps);
    for (i = 0; i < n; i++) {
        out[i] = x[i] * scale;
    }
    for (i = 0; i < n; i++) {
        out[i] *= weight[i];
    }
}

void edgexpu_cpu_silu(float *out, const float *x, int n) {
    int i;
    if (out == NULL || x == NULL) {
        return;
    }
    for (i = 0; i < n; i++) {
        out[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

void edgexpu_cpu_sigmoid(float *io, int n) {
    int i;
    if (io == NULL) {
        return;
    }
    for (i = 0; i < n; i++) {
        io[i] = 1.0f / (1.0f + expf(-io[i]));
    }
}

void edgexpu_cpu_softplus(float *io, int n) {
    int i;
    if (io == NULL) {
        return;
    }
    for (i = 0; i < n; i++) {
        float x = io[i];
        io[i] = x > 20.0f ? x : logf(1.0f + expf(x));
    }
}

void edgexpu_cpu_l2_normalize(float *x, int n, float eps) {
    int i;
    double ss = 0.0;
    float scale;
    if (x == NULL || n <= 0) {
        return;
    }
    for (i = 0; i < n; i++) {
        ss += (double)x[i] * (double)x[i];
    }
    scale = 1.0f / sqrtf((float)ss + eps);
    for (i = 0; i < n; i++) {
        x[i] *= scale;
    }
}

void edgexpu_cpu_gated_delta_step(
    float *state,
    const float *q,
    const float *k,
    const float *v,
    float g_log,
    float beta,
    float *out,
    int dk,
    int dv
) {
    int i;
    int col;
    float g;
    float scale;
    if (state == NULL || q == NULL || k == NULL || v == NULL || out == NULL || dk <= 0 || dv <= 0) {
        return;
    }
    g = expf(g_log);
    scale = 1.0f / sqrtf((float)dv);
    for (col = 0; col < dv; col++) {
        float kv = 0.0f;
        float delta;
        float attn = 0.0f;
        for (i = 0; i < dk; i++) {
            kv += state[i * dv + col] * k[i];
        }
        delta = (v[col] - g * kv) * beta;
        for (i = 0; i < dk; i++) {
            float s = g * state[i * dv + col] + k[i] * delta;
            state[i * dv + col] = s;
            attn += s * q[i];
        }
        out[col] = attn * scale;
    }
}

void edgexpu_cpu_softmax(float *io, int n) {
    int i;
    float max_value;
    double sum = 0.0;

    if (io == NULL || n <= 0) {
        return;
    }

    max_value = io[0];
    for (i = 1; i < n; i++) {
        if (io[i] > max_value) {
            max_value = io[i];
        }
    }
    for (i = 0; i < n; i++) {
        io[i] = expf(io[i] - max_value);
        sum += (double)io[i];
    }
    if (sum <= 0.0) {
        return;
    }
    for (i = 0; i < n; i++) {
        io[i] = (float)((double)io[i] / sum);
    }
}

void edgexpu_cpu_matmul_f32(
    float *out,
    const float *a,
    const float *b,
    int m,
    int k,
    int n
) {
    int row;
    int col;
    int inner;

    if (out == NULL || a == NULL || b == NULL) {
        return;
    }

    for (row = 0; row < m; row++) {
        for (col = 0; col < n; col++) {
            float acc = 0.0f;
            for (inner = 0; inner < k; inner++) {
                acc += a[row * k + inner] * b[inner * n + col];
            }
            out[row * n + col] = acc;
        }
    }
}

float edgexpu_cpu_dot(const float *a, const float *b, int n) {
    int i = 0;
    float acc = 0.0f;
    if (a == NULL || b == NULL || n <= 0) {
        return 0.0f;
    }
#if defined(EDGEXPU_AVX2)
    {
        __m256 accv = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) {
            __m256 av = _mm256_loadu_ps(a + i);
            __m256 bv = _mm256_loadu_ps(b + i);
#if defined(EDGEXPU_FMA)
            accv = _mm256_fmadd_ps(av, bv, accv);
#else
            accv = _mm256_add_ps(accv, _mm256_mul_ps(av, bv));
#endif
        }
        {
            __m128 lo = _mm256_castps256_ps128(accv);
            __m128 hi = _mm256_extractf128_ps(accv, 1);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_add_ps(s, _mm_movehdup_ps(s));
            s = _mm_add_ss(s, _mm_movehl_ps(s, s));
            acc = _mm_cvtss_f32(s);
        }
    }
#elif defined(EDGEXPU_NEON)
    {
        float32x4_t accv = vdupq_n_f32(0.0f);
        for (; i + 4 <= n; i += 4) {
            accv = vfmaq_f32(accv, vld1q_f32(a + i), vld1q_f32(b + i));
        }
        {
            float32x2_t s = vadd_f32(vget_low_f32(accv), vget_high_f32(accv));
            s = vpadd_f32(s, s);
            acc = vget_lane_f32(s, 0);
        }
    }
#endif
    for (; i < n; i++) {
        acc += a[i] * b[i];
    }
    return acc;
}

void edgexpu_cpu_saxpy(float *y, const float *x, float a, int n) {
    int i = 0;
    if (y == NULL || x == NULL || n <= 0) {
        return;
    }
#if defined(EDGEXPU_AVX2)
    {
        __m256 av = _mm256_set1_ps(a);
        for (; i + 8 <= n; i += 8) {
            __m256 yv = _mm256_loadu_ps(y + i);
            __m256 xv = _mm256_loadu_ps(x + i);
#if defined(EDGEXPU_FMA)
            yv = _mm256_fmadd_ps(av, xv, yv);
#else
            yv = _mm256_add_ps(yv, _mm256_mul_ps(av, xv));
#endif
            _mm256_storeu_ps(y + i, yv);
        }
    }
#elif defined(EDGEXPU_NEON)
    {
        float32x4_t av = vdupq_n_f32(a);
        for (; i + 4 <= n; i += 4) {
            float32x4_t yv = vld1q_f32(y + i);
            yv = vfmaq_f32(yv, av, vld1q_f32(x + i));
            vst1q_f32(y + i, yv);
        }
    }
#endif
    for (; i < n; i++) {
        y[i] += a * x[i];
    }
}

void edgexpu_cpu_linear(
    float *out,
    const float *x,
    const float *weight,
    const float *bias,
    int n_out,
    int n_in
) {
    int o;
    int n_threads;
    if (out == NULL || x == NULL || weight == NULL || n_out <= 0 || n_in <= 0) {
        return;
    }
    tune_openmp_threads();
    n_threads = linear_thread_count(n_out);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(n_threads) if(n_threads > 1)
#endif
    for (o = 0; o < n_out; o++) {
        int i = 0;
        double acc = bias != NULL ? (double)bias[o] : 0.0;
        const float *row = weight + (size_t)o * (size_t)n_in;
#if defined(EDGEXPU_AVX2)
        {
            __m256 accv = _mm256_setzero_ps();
            for (; i + 8 <= n_in; i += 8) {
                __m256 xv = _mm256_loadu_ps(x + i);
                __m256 wv = _mm256_loadu_ps(row + i);
#if defined(EDGEXPU_FMA)
                accv = _mm256_fmadd_ps(xv, wv, accv);
#else
                accv = _mm256_add_ps(accv, _mm256_mul_ps(xv, wv));
#endif
            }
            {
                __m128 lo = _mm256_castps256_ps128(accv);
                __m128 hi = _mm256_extractf128_ps(accv, 1);
                __m128 s = _mm_add_ps(lo, hi);
                s = _mm_add_ps(s, _mm_movehdup_ps(s));
                s = _mm_add_ss(s, _mm_movehl_ps(s, s));
                acc += (double)_mm_cvtss_f32(s);
            }
        }
#elif defined(EDGEXPU_NEON)
        {
            float32x4_t accv = vdupq_n_f32(0.0f);
            for (; i + 4 <= n_in; i += 4) {
                accv = vfmaq_f32(accv, vld1q_f32(x + i), vld1q_f32(row + i));
            }
            {
                float32x2_t s = vadd_f32(vget_low_f32(accv), vget_high_f32(accv));
                s = vpadd_f32(s, s);
                acc += (double)vget_lane_f32(s, 0);
            }
        }
#endif
        for (; i < n_in; i++) {
            acc += (double)x[i] * (double)row[i];
        }
        out[o] = (float)acc;
    }
}

static uint8_t *g_act_scratch;
static size_t g_act_scratch_cap;

static uint8_t *act_scratch_ensure(size_t nbytes) {
    uint8_t *p;
    if (nbytes == 0) {
        return NULL;
    }
    if (g_act_scratch_cap >= nbytes) {
        return g_act_scratch;
    }
    p = (uint8_t *)realloc(g_act_scratch, nbytes);
    if (p == NULL) {
        return NULL;
    }
    g_act_scratch = p;
    g_act_scratch_cap = nbytes;
    return g_act_scratch;
}

void edgexpu_cpu_linear_quant_batch(
    float *out,
    const float *x,
    const uint8_t *weight,
    uint32_t type,
    const float *bias,
    int m,
    int n_out,
    int n_in
) {
    size_t row_bytes;
    int o;
    int n_threads;
    uint8_t *xq;
    size_t qbytes;
    int t;

    if (out == NULL || x == NULL || weight == NULL || m <= 0 || n_out <= 0 || n_in <= 0) {
        return;
    }
    tune_openmp_threads();
    if (m == 1 && type == EDGEXPU_GGUF_TYPE_F32) {
        edgexpu_cpu_linear(out, x, (const float *)weight, bias, n_out, n_in);
        return;
    }

    row_bytes = edgexpu_gguf_row_bytes(type, n_in);
    if (row_bytes == 0) {
        return;
    }
    n_threads = linear_thread_count(n_out);

    if (edgexpu_gguf_can_dot_q8(type) && n_in % 32 == 0) {
        qbytes = edgexpu_gguf_q8_0_nbytes(n_in);
        xq = act_scratch_ensure((size_t)m * qbytes);
        if (xq != NULL && qbytes > 0) {
            for (t = 0; t < m; t++) {
                if (!edgexpu_gguf_quantize_q8_0(x + (size_t)t * (size_t)n_in, n_in, xq + (size_t)t * qbytes)) {
                    xq = NULL;
                    break;
                }
            }
            if (xq != NULL) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(n_threads) if(n_threads > 1)
#endif
                for (o = 0; o < n_out; o++) {
                    const uint8_t *wrow = weight + (size_t)o * row_bytes;
                    float b = bias != NULL ? bias[o] : 0.0f;
                    int tok;
                    for (tok = 0; tok < m; tok++) {
                        out[(size_t)tok * (size_t)n_out + (size_t)o] =
                            b + edgexpu_gguf_dot_quant_q8(type, wrow, xq + (size_t)tok * qbytes, n_in);
                    }
                }
                return;
            }
        }
    }

    if (edgexpu_gguf_can_dot_q8k(type) && n_in % 256 == 0) {
        qbytes = edgexpu_gguf_q8_k_nbytes(n_in);
        xq = act_scratch_ensure((size_t)m * qbytes);
        if (xq != NULL && qbytes > 0) {
            for (t = 0; t < m; t++) {
                if (!edgexpu_gguf_quantize_q8_k(x + (size_t)t * (size_t)n_in, n_in, xq + (size_t)t * qbytes)) {
                    xq = NULL;
                    break;
                }
            }
            if (xq != NULL) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(n_threads) if(n_threads > 1)
#endif
                for (o = 0; o < n_out; o++) {
                    const uint8_t *wrow = weight + (size_t)o * row_bytes;
                    float b = bias != NULL ? bias[o] : 0.0f;
                    int tok;
                    for (tok = 0; tok < m; tok++) {
                        out[(size_t)tok * (size_t)n_out + (size_t)o] =
                            b + edgexpu_gguf_dot_quant_q8k(type, wrow, xq + (size_t)tok * qbytes, n_in);
                    }
                }
                return;
            }
        }
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(n_threads) if(n_threads > 1)
#endif
    for (o = 0; o < n_out; o++) {
        const uint8_t *wrow = weight + (size_t)o * row_bytes;
        float b = bias != NULL ? bias[o] : 0.0f;
        for (t = 0; t < m; t++) {
            out[(size_t)t * (size_t)n_out + (size_t)o] =
                b + edgexpu_gguf_dot_quant(type, wrow, x + (size_t)t * (size_t)n_in, n_in);
        }
    }
}

void edgexpu_cpu_linear_quant(
    float *out,
    const float *x,
    const uint8_t *weight,
    uint32_t type,
    const float *bias,
    int n_out,
    int n_in
) {
    edgexpu_cpu_linear_quant_batch(out, x, weight, type, bias, 1, n_out, n_in);
}

/* 与 ggml_rope_cache_init 相同：theta 从 pos 起，每对乘 theta_scale = freq_base^(-2/n_dims)。 */
static void rope_fill_cache(float *cache, int head_dim, int pos, float freq_base) {
    float theta_scale = powf(freq_base, -2.0f / (float)head_dim);
    float theta = (float)pos;
    int i;

    for (i = 0; i < head_dim; i += 2) {
        cache[i] = cosf(theta);
        cache[i + 1] = sinf(theta);
        theta *= theta_scale;
    }
}

/* 前半维与后半维配对，Qwen2 / GPT-NeoX。 */
void edgexpu_cpu_rope_neox(
    float *x,
    int n_heads,
    int head_dim,
    int pos,
    float freq_base
) {
    int h;
    int i;
    int half;
    float cache[512];

    if (x == NULL || n_heads <= 0 || head_dim < 2 || freq_base <= 0.0f) {
        return;
    }
    if (head_dim > 512) {
        return;
    }

    half = head_dim / 2;
    rope_fill_cache(cache, head_dim, pos, freq_base);
    for (h = 0; h < n_heads; h++) {
        float *head = x + h * head_dim;
        for (i = 0; i < half; i++) {
            float c = cache[i * 2];
            float s = cache[i * 2 + 1];
            float x0 = head[i];
            float x1 = head[i + half];
            head[i] = x0 * c - x1 * s;
            head[i + half] = x0 * s + x1 * c;
        }
    }
}

/* 相邻维配对，Llama / Mistral。 */
void edgexpu_cpu_rope_norm(
    float *x,
    int n_heads,
    int head_dim,
    int pos,
    float freq_base
) {
    int h;
    int i;
    float cache[512];

    if (x == NULL || n_heads <= 0 || head_dim < 2 || freq_base <= 0.0f) {
        return;
    }
    if (head_dim > 512) {
        return;
    }

    rope_fill_cache(cache, head_dim, pos, freq_base);
    for (h = 0; h < n_heads; h++) {
        float *head = x + h * head_dim;
        for (i = 0; i + 1 < head_dim; i += 2) {
            float c = cache[i];
            float s = cache[i + 1];
            float x0 = head[i];
            float x1 = head[i + 1];
            head[i] = x0 * c - x1 * s;
            head[i + 1] = x0 * s + x1 * c;
        }
    }
}

typedef struct edgexpu_sample_item {
    float p;
    uint32_t id;
} edgexpu_sample_item;

static int sample_item_desc(const void *a, const void *b) {
    const edgexpu_sample_item *ia = (const edgexpu_sample_item *)a;
    const edgexpu_sample_item *ib = (const edgexpu_sample_item *)b;
    if (ib->p > ia->p) {
        return 1;
    }
    if (ib->p < ia->p) {
        return -1;
    }
    if (ia->id < ib->id) {
        return -1;
    }
    if (ia->id > ib->id) {
        return 1;
    }
    return 0;
}

int edgexpu_cpu_sample_softmax(
    float *logits,
    int n,
    float top_p,
    uint64_t *rng,
    uint32_t *out
) {
    int i;
    int kept = n;
    double sum = 0.0;
    double cursor = 0.0;
    double pick;
    edgexpu_sample_item *items = NULL;

    if (logits == NULL || rng == NULL || out == NULL || n <= 0) {
        return 0;
    }

    edgexpu_cpu_softmax(logits, n);
    if (*rng == 0) {
        *rng = 0x00ED6E50ULL;
    }
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;

    if (top_p > 0.0f && top_p < 1.0f) {
        items = (edgexpu_sample_item *)malloc((size_t)n * sizeof(*items));
        if (items == NULL) {
            return 0;
        }
        for (i = 0; i < n; i++) {
            items[i].p = logits[i];
            items[i].id = (uint32_t)i;
        }
        qsort(items, (size_t)n, sizeof(*items), sample_item_desc);
        sum = 0.0;
        kept = n;
        for (i = 0; i < n; i++) {
            sum += (double)items[i].p;
            if (sum >= (double)top_p) {
                kept = i + 1;
                break;
            }
        }
        if (kept < 1) {
            kept = 1;
        }
        sum = 0.0;
        for (i = 0; i < kept; i++) {
            sum += (double)items[i].p;
        }
        if (sum <= 0.0) {
            *out = items[0].id;
            free(items);
            return 1;
        }
        pick = ((double)(*rng & 0xFFFFFFFu) / (double)0x10000000u) * sum;
        *out = items[kept - 1].id;
        for (i = 0; i < kept; i++) {
            cursor += (double)items[i].p;
            if (cursor >= pick) {
                *out = items[i].id;
                break;
            }
        }
        free(items);
        return 1;
    }

    for (i = 0; i < n; i++) {
        sum += (double)logits[i];
    }
    if (sum <= 0.0) {
        *out = 0;
        return 1;
    }
    pick = ((double)(*rng & 0xFFFFFFFu) / (double)0x10000000u) * sum;
    *out = (uint32_t)(n - 1);
    for (i = 0; i < n; i++) {
        cursor += (double)logits[i];
        if (cursor >= pick) {
            *out = (uint32_t)i;
            break;
        }
    }
    return 1;
}

const char *edgexpu_cpu_simd_name(void) {
#if defined(EDGEXPU_AVX2)
    return "avx2";
#elif defined(EDGEXPU_NEON)
    return "neon";
#else
    return "scalar";
#endif
}

int edgexpu_cpu_kernel_selftest(char *error, size_t error_size) {
    float x[2] = {3.0f, 4.0f};
    float w[2] = {1.0f, 1.0f};
    float y[2];
    float a[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b[6] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float c[4];
    float silu0;

    edgexpu_cpu_rmsnorm(y, x, w, 2, 1e-6f);
    if (y[0] <= 0.0f || y[1] <= 0.0f) {
        snprintf(error, error_size, "rmsnorm selftest failed");
        return 0;
    }

    y[0] = 0.0f;
    y[1] = 0.0f;
    edgexpu_cpu_softmax(y, 2);
    if (fabsf(y[0] - 0.5f) > 1e-5f || fabsf(y[1] - 0.5f) > 1e-5f) {
        snprintf(error, error_size, "softmax selftest failed");
        return 0;
    }

    silu0 = 0.0f;
    edgexpu_cpu_silu(&silu0, &silu0, 1);
    if (fabsf(silu0) > 1e-6f) {
        snprintf(error, error_size, "silu selftest failed");
        return 0;
    }

    edgexpu_cpu_matmul_f32(c, a, b, 2, 3, 2);
    if (fabsf(c[0] - 58.0f) > 1e-3f || fabsf(c[3] - 154.0f) > 1e-3f) {
        snprintf(error, error_size, "matmul selftest failed");
        return 0;
    }

    {
        float da[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float db[8] = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
        float y[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        float dot = edgexpu_cpu_dot(da, db, 8);
        if (fabsf(dot - 120.0f) > 1e-3f) {
            snprintf(error, error_size, "dot selftest failed");
            return 0;
        }
        edgexpu_cpu_saxpy(y, da, 2.0f, 8);
        if (fabsf(y[0] - 3.0f) > 1e-4f || fabsf(y[7] - 17.0f) > 1e-4f) {
            snprintf(error, error_size, "saxpy selftest failed");
            return 0;
        }
    }

    {
        float lx[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        float lw[16] = {
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };
        float ly[2];
        edgexpu_cpu_linear(ly, lx, lw, NULL, 2, 8);
        if (fabsf(ly[0] - 1.0f) > 1e-4f || fabsf(ly[1] - 2.0f) > 1e-4f) {
            snprintf(error, error_size, "linear selftest failed");
            return 0;
        }
        edgexpu_cpu_linear_quant(ly, lx, (const uint8_t *)lw, 0, NULL, 2, 8);
        if (fabsf(ly[0] - 1.0f) > 1e-4f || fabsf(ly[1] - 2.0f) > 1e-4f) {
            snprintf(error, error_size, "linear_quant f32 selftest failed");
            return 0;
        }
    }

    if (!edgexpu_gguf_quant_selftest(error, error_size)) {
        return 0;
    }

    {
        float neox[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        float norm[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        edgexpu_cpu_rope_neox(neox, 1, 4, 0, 10000.0f);
        edgexpu_cpu_rope_norm(norm, 1, 4, 0, 10000.0f);
        if (fabsf(neox[0] - 1.0f) > 1e-5f || fabsf(norm[0] - 1.0f) > 1e-5f) {
            snprintf(error, error_size, "rope pos=0 identity failed");
            return 0;
        }
        edgexpu_cpu_rope_norm(norm, 1, 4, 1, 10000.0f);
        if (fabsf(norm[0] - 1.0f) < 1e-6f && fabsf(norm[1] - 2.0f) < 1e-6f) {
            snprintf(error, error_size, "rope_norm pos=1 did not rotate");
            return 0;
        }
    }

    {
        float logits[4] = {0.0f, 20.0f, 0.0f, 0.0f};
        uint64_t rng = 1;
        uint32_t id = 99;
        if (!edgexpu_cpu_sample_softmax(logits, 4, 1.0f, &rng, &id) || id != 1) {
            snprintf(error, error_size, "softmax sample selftest failed id=%u", id);
            return 0;
        }
        logits[0] = 0.0f;
        logits[1] = 20.0f;
        logits[2] = 0.0f;
        logits[3] = 0.0f;
        id = 99;
        if (!edgexpu_cpu_sample_softmax(logits, 4, 0.9f, &rng, &id) || id != 1) {
            snprintf(error, error_size, "top_p sample selftest failed id=%u", id);
            return 0;
        }
    }

    return 1;
}
