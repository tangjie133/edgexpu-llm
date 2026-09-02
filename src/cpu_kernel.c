#include "edgexpu/cpu_kernel.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#if defined(EDGEXPU_AVX2)
#include <immintrin.h>
#endif

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
    float ss = 0.0f;
    float scale;

    if (out == NULL || x == NULL || weight == NULL || n <= 0) {
        return;
    }

    for (i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (i = 0; i < n; i++) {
        out[i] = x[i] * scale * weight[i];
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

void edgexpu_cpu_softmax(float *io, int n) {
    int i;
    float max_value;
    float sum = 0.0f;

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
        sum += io[i];
    }
    if (sum <= 0.0f) {
        return;
    }
    for (i = 0; i < n; i++) {
        io[i] /= sum;
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

void edgexpu_cpu_linear(
    float *out,
    const float *x,
    const float *weight,
    const float *bias,
    int n_out,
    int n_in
) {
    int o;
    int i;
    if (out == NULL || x == NULL || weight == NULL || n_out <= 0 || n_in <= 0) {
        return;
    }
    for (o = 0; o < n_out; o++) {
        float acc = bias != NULL ? bias[o] : 0.0f;
        const float *row = weight + (size_t)o * (size_t)n_in;
        i = 0;
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
                acc += _mm_cvtss_f32(s);
            }
        }
#endif
        for (; i < n_in; i++) {
            acc += x[i] * row[i];
        }
        out[o] = acc;
    }
}

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

    if (x == NULL || n_heads <= 0 || head_dim < 2 || freq_base <= 0.0f) {
        return;
    }

    half = head_dim / 2;
    for (h = 0; h < n_heads; h++) {
        float *head = x + h * head_dim;
        for (i = 0; i < half; i++) {
            float freq = 1.0f / powf(freq_base, (2.0f * (float)i) / (float)head_dim);
            float angle = (float)pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            float x0 = head[i];
            float x1 = head[i + half];
            head[i] = x0 * c - x1 * s;
            head[i + half] = x0 * s + x1 * c;
        }
    }
}

void edgexpu_cpu_rope_norm(
    float *x,
    int n_heads,
    int head_dim,
    int pos,
    float freq_base
) {
    int h;
    int i;

    if (x == NULL || n_heads <= 0 || head_dim < 2 || freq_base <= 0.0f) {
        return;
    }

    for (h = 0; h < n_heads; h++) {
        float *head = x + h * head_dim;
        for (i = 0; i + 1 < head_dim; i += 2) {
            float freq = 1.0f / powf(freq_base, (float)i / (float)head_dim);
            float angle = (float)pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            float x0 = head[i];
            float x1 = head[i + 1];
            head[i] = x0 * c - x1 * s;
            head[i + 1] = x0 * s + x1 * c;
        }
    }
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

    return 1;
}
