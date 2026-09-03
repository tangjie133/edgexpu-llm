#ifndef EDGEXPU_ARCH_H
#define EDGEXPU_ARCH_H

#include "edgexpu/gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 架构适配器：由 GGUF general.architecture 描述模型形状差异。
 * Runtime 核心不写死某一个模型。Qwen2 只是第一个实现；换 Llama 等应走这里，
 * 而不是在 native.c 里加 Qwen 专用分支。
 */

typedef enum edgexpu_rope_type {
    EDGEXPU_ROPE_NONE = 0,
    EDGEXPU_ROPE_NEOX = 1,  /* 前后半维配对，Qwen2 使用 */
    EDGEXPU_ROPE_NORM = 2   /* 相邻维配对，Llama / Mistral 使用 */
} edgexpu_rope_type;

typedef enum edgexpu_ffn_type {
    EDGEXPU_FFN_SWIGLU = 0 /* silu(gate) * up，再 down 投影 */
} edgexpu_ffn_type;

typedef enum edgexpu_tokenizer_kind {
    EDGEXPU_TOKENIZER_GPT2_BPE = 0 /* 当前只接 GPT-2 BPE；SentencePiece 尚未实现 */
} edgexpu_tokenizer_kind;

typedef struct edgexpu_arch_adapter {
    char name[EDGEXPU_TEXT_SMALL]; /* 与 GGUF architecture 字符串一致 */
    int has_qkv_bias;              /* 1：加载 attn_q/k/v.bias；Llama 为 0 */
    edgexpu_rope_type rope;
    edgexpu_ffn_type ffn;
    edgexpu_tokenizer_kind tokenizer;
} edgexpu_arch_adapter;

/* 按 GGUF metadata 填充 adapter。不支持的 architecture / tokenizer 返回 0。 */
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
