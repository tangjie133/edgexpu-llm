#ifndef EDGEXPU_GGUF_QUANT_H
#define EDGEXPU_GGUF_QUANT_H

#include <stddef.h>
#include <stdint.h>

#include "edgexpu/gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

#define EDGEXPU_GGUF_TYPE_F32 0
#define EDGEXPU_GGUF_TYPE_Q5_0 6
#define EDGEXPU_GGUF_TYPE_Q8_0 8
#define EDGEXPU_GGUF_TYPE_Q4_K 12
#define EDGEXPU_GGUF_TYPE_Q6_K 14

size_t edgexpu_gguf_type_block_size(uint32_t type);
size_t edgexpu_gguf_type_block_bytes(uint32_t type);
uint64_t edgexpu_gguf_tensor_elements(const edgexpu_gguf_tensor *tensor);
size_t edgexpu_gguf_tensor_nbytes(const edgexpu_gguf_tensor *tensor);

int edgexpu_gguf_dequantize(
    const uint8_t *src,
    uint32_t type,
    uint64_t n_elements,
    float *dst,
    char *error,
    size_t error_size
);

int edgexpu_gguf_dequantize_q8_0_rows(
    const uint8_t *src,
    uint64_t row,
    int n_embd,
    float *dst
);

int edgexpu_gguf_dequantize_row(
    const uint8_t *src,
    const edgexpu_gguf_tensor *tensor,
    uint64_t row,
    int n_embd,
    float *dst,
    char *error,
    size_t error_size
);

float edgexpu_gguf_q8_0_dot_row(
    const uint8_t *src,
    uint64_t row,
    int n_embd,
    const float *x
);

#ifdef __cplusplus
}
#endif

#endif
