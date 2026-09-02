#ifndef EDGEXPU_ARCH_H
#define EDGEXPU_ARCH_H

#include "edgexpu/gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum edgexpu_rope_type {
    EDGEXPU_ROPE_NONE = 0,
    EDGEXPU_ROPE_NEOX = 1,
    EDGEXPU_ROPE_NORM = 2
} edgexpu_rope_type;

typedef enum edgexpu_ffn_type {
    EDGEXPU_FFN_SWIGLU = 0
} edgexpu_ffn_type;

typedef enum edgexpu_tokenizer_kind {
    EDGEXPU_TOKENIZER_GPT2_BPE = 0
} edgexpu_tokenizer_kind;

typedef struct edgexpu_arch_adapter {
    char name[EDGEXPU_TEXT_SMALL];
    int has_qkv_bias;
    edgexpu_rope_type rope;
    edgexpu_ffn_type ffn;
    edgexpu_tokenizer_kind tokenizer;
} edgexpu_arch_adapter;

int edgexpu_arch_from_gguf(
    const edgexpu_gguf_info *info,
    edgexpu_arch_adapter *adapter,
    char *error,
    size_t error_size
);

const char *edgexpu_rope_type_name(edgexpu_rope_type rope);
const char *edgexpu_ffn_type_name(edgexpu_ffn_type ffn);
const char *edgexpu_tokenizer_kind_name(edgexpu_tokenizer_kind kind);

#ifdef __cplusplus
}
#endif

#endif
