#ifndef EDGEXPU_GGUF_QUANT_H
#define EDGEXPU_GGUF_QUANT_H

#include <stddef.h>
#include <stdint.h>

#include "edgexpu/gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* GGUF 量化编解码。Kernel 层通用，不绑定某一个模型。
 * 类型编号与 GGUF spec 一致。
 */

#define EDGEXPU_GGUF_TYPE_F32 0
#define EDGEXPU_GGUF_TYPE_F16 1
#define EDGEXPU_GGUF_TYPE_Q4_0 2
#define EDGEXPU_GGUF_TYPE_Q5_0 6
#define EDGEXPU_GGUF_TYPE_Q8_0 8
#define EDGEXPU_GGUF_TYPE_Q4_K 12
#define EDGEXPU_GGUF_TYPE_Q5_K 13
#define EDGEXPU_GGUF_TYPE_Q6_K 14

size_t edgexpu_gguf_type_block_size(uint32_t type);  /* 每个量化块覆盖的元素数 */
size_t edgexpu_gguf_type_block_bytes(uint32_t type); /* 每个量化块的字节数 */
uint64_t edgexpu_gguf_tensor_elements(const edgexpu_gguf_tensor *tensor);
size_t edgexpu_gguf_tensor_nbytes(const edgexpu_gguf_tensor *tensor);

/* 把整段量化数据反量化到 f32。n_elements 必须对齐到 block size。 */
int edgexpu_gguf_dequantize(
    const uint8_t *src,
    uint32_t type,
    uint64_t n_elements,
    float *dst,
    char *error,
    size_t error_size
);

/* Q8_0 embedding 按行反量化，供 prefill/decode 取 token 向量。 */
int edgexpu_gguf_dequantize_q8_0_rows(
    const uint8_t *src,
    uint64_t row,
    int n_embd,
    float *dst
);

/* 按行反量化任意已支持类型。用于非 Q8_0 的 embedding / output.weight。 */
int edgexpu_gguf_dequantize_row(
    const uint8_t *src,
    const edgexpu_gguf_tensor *tensor,
    uint64_t row,
    int n_embd,
    float *dst,
    char *error,
    size_t error_size
);

/* Q8_0 行与隐状态点积，避免先反量化再乘。Decode logits 热路径。 */
float edgexpu_gguf_q8_0_dot_row(
    const uint8_t *src,
    uint64_t row,
    int n_embd,
    const float *x
);

/* 一行量化权重的字节数。n 为该行元素数。 */
size_t edgexpu_gguf_row_bytes(uint32_t type, int n);

/* 量化行与 f32 向量点积。Q4_0/Q4_K/Q5_0/Q5_K/Q6_K/Q8_0/F16/F32 不展开成 f32。AVX2 与 NEON 同一入口。 */
float edgexpu_gguf_dot_quant(
    uint32_t type,
    const uint8_t *row,
    const float *x,
    int n
);

/* 把 f32 激活量化成 Q8_0，供后续整数点积。n 必须整除 32。 */
size_t edgexpu_gguf_q8_0_nbytes(int n);
int edgexpu_gguf_quantize_q8_0(const float *x, int n, uint8_t *dst);
int edgexpu_gguf_can_dot_q8(uint32_t type);

/* Q8_K 激活：K-quant 权重用。n 必须整除 256。 */
size_t edgexpu_gguf_q8_k_nbytes(int n);
int edgexpu_gguf_quantize_q8_k(const float *x, int n, uint8_t *dst);
int edgexpu_gguf_can_dot_q8k(uint32_t type);

/* 量化权重行 × Q8_0 激活。与 llama.cpp 相同：激活只量化一次，再对所有输出行做整数点积。 */
float edgexpu_gguf_dot_quant_q8(
    uint32_t type,
    const uint8_t *wrow,
    const uint8_t *x_q8,
    int n
);

float edgexpu_gguf_dot_quant_q8k(
    uint32_t type,
    const uint8_t *wrow,
    const uint8_t *x_q8k,
    int n
);

int edgexpu_gguf_quant_selftest(char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
