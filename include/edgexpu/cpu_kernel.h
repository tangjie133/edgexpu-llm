#ifndef EDGEXPU_CPU_KERNEL_H
#define EDGEXPU_CPU_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* f32 / 量化 CPU kernel。算子按 adapter 需要的子集启用，不绑定某一个模型。
 * linear 与量化点积在 x86 走 AVX2+FMA，在 ARM 走 NEON，入口相同。
 * 线程数默认按一次 GEMV 探测选择，不写死核数比例；OMP_NUM_THREADS 可覆盖。
 * bias 允许为 NULL。
 */

void edgexpu_cpu_add(float *out, const float *a, const float *b, int n);
void edgexpu_cpu_mul(float *out, const float *a, const float *b, int n);
void edgexpu_cpu_rmsnorm(float *out, const float *x, const float *weight, int n, float eps);
void edgexpu_cpu_silu(float *out, const float *x, int n);
void edgexpu_cpu_sigmoid(float *io, int n);
void edgexpu_cpu_softplus(float *io, int n);
void edgexpu_cpu_l2_normalize(float *x, int n, float eps);
void edgexpu_cpu_softmax(float *io, int n);

/* Gated DeltaNet 一步：S 为 [dk, dv] 行主序 S[i*dv+col]。g_log 是 log 衰减（再 exp）。 */
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
);

/* out[m, n] = a[m, k] * b[k, n]，行主序。 */
void edgexpu_cpu_matmul_f32(
    float *out,
    const float *a,
    const float *b,
    int m,
    int k,
    int n
);

/* y = x * W^T + b。weight 按 [n_out, n_in] 存放。 */
void edgexpu_cpu_linear(
    float *out,
    const float *x,
    const float *weight,
    const float *bias,
    int n_out,
    int n_in
);

/* 量化权重的 linear：W 保持压缩行，不先展开成 f32。n_out 较大时走 OpenMP。 */
void edgexpu_cpu_linear_quant(
    float *out,
    const float *x,
    const uint8_t *weight,
    uint32_t type,
    const float *bias,
    int n_out,
    int n_in
);

/* out[t, n_out] = x[t, n_in] * W^T + b。同一行权重扫过全部 token，供 prefill。 */
void edgexpu_cpu_linear_quant_batch(
    float *out,
    const float *x,
    const uint8_t *weight,
    uint32_t type,
    const float *bias,
    int m,
    int n_out,
    int n_in
);

/* 短向量点积 / y += a*x，attention 热路径。 */
float edgexpu_cpu_dot(const float *a, const float *b, int n);
void edgexpu_cpu_saxpy(float *y, const float *x, float a, int n);

/* logits 已除以 temperature。就地 softmax，top_p∈(0,1) 时做 nucleus，再按 rng 采样。
 * top_p<=0 或 >=1 表示不截断。 */
int edgexpu_cpu_sample_softmax(
    float *logits,
    int n,
    float top_p,
    uint64_t *rng,
    uint32_t *out
);

/* GPT-NeoX / Qwen2 RoPE：同一 head 内前半维与后半维配对旋转。 */
void edgexpu_cpu_rope_neox(
    float *x,
    int n_heads,
    int head_dim,
    int pos,
    float freq_base
);

/* Llama 风格 RoPE：相邻 (x0, x1)、(x2, x3) … 配对旋转。 */
void edgexpu_cpu_rope_norm(
    float *x,
    int n_heads,
    int head_dim,
    int pos,
    float freq_base
);

/* "avx2" / "neon" / "scalar"。给 capabilities 和 ARM 验证脚本用。 */
const char *edgexpu_cpu_simd_name(void);

int edgexpu_cpu_kernel_selftest(char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
