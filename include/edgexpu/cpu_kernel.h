#ifndef EDGEXPU_CPU_KERNEL_H
#define EDGEXPU_CPU_KERNEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void edgexpu_cpu_add(float *out, const float *a, const float *b, int n);
void edgexpu_cpu_mul(float *out, const float *a, const float *b, int n);
void edgexpu_cpu_rmsnorm(float *out, const float *x, const float *weight, int n, float eps);
void edgexpu_cpu_silu(float *out, const float *x, int n);
void edgexpu_cpu_softmax(float *io, int n);
void edgexpu_cpu_matmul_f32(
    float *out,
    const float *a,
    const float *b,
    int m,
    int k,
    int n
);

void edgexpu_cpu_linear(
    float *out,
    const float *x,
    const float *weight,
    const float *bias,
    int n_out,
    int n_in
);

void edgexpu_cpu_rope_neox(
    float *x,
    int n_heads,
    int head_dim,
    int pos,
    float freq_base
);

void edgexpu_cpu_rope_norm(
    float *x,
    int n_heads,
    int head_dim,
    int pos,
    float freq_base
);

int edgexpu_cpu_kernel_selftest(char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
