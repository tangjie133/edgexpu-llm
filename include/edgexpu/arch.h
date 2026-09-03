#ifndef EDGEXPU_ARCH_H
#define EDGEXPU_ARCH_H

#include "edgexpu/gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 架构插件：tensor 名模板 + RoPE/bias/FFN/tokenizer。
 * 新增 GGUF architecture 时加 src/arch/<name>.c 并在 register.c 登记，不要改 native.c 层循环。
 */

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

typedef enum edgexpu_layer_kind {
    EDGEXPU_LAYER_ATTN_SWIGLU = 0, /* dense MHA + SwiGLU，cpu.native 已实现 */
    EDGEXPU_LAYER_UNSUPPORTED = 1
} edgexpu_layer_kind;

/* GGUF 常见 llama.cpp 命名。模板里的 %d 是层号；无 %d 的是全局 tensor。 */
typedef struct edgexpu_arch_tensor_names {
    const char *token_embd;
    const char *output_weight;
    const char *output_norm;
    const char *output_bias;
    const char *attn_norm;
    const char *ffn_norm;
    const char *attn_q;
    const char *attn_k;
    const char *attn_v;
    const char *attn_output;
    const char *ffn_gate;
    const char *ffn_up;
    const char *ffn_down;
    const char *attn_q_bias;
    const char *attn_k_bias;
    const char *attn_v_bias;
    const char *attn_output_bias;
    const char *ffn_gate_bias;
    const char *ffn_up_bias;
    const char *ffn_down_bias;
} edgexpu_arch_tensor_names;

typedef struct edgexpu_arch_adapter edgexpu_arch_adapter;

typedef struct edgexpu_arch_plugin {
    const char *id;
    int native_forward; /* 1：cpu.native 可跑；0：只识别 / tokenize */
    int (*match)(const edgexpu_gguf_info *info);
    int (*configure)(
        const edgexpu_gguf_info *info,
        edgexpu_arch_adapter *adapter,
        char *error,
        size_t error_size
    );
    edgexpu_layer_kind (*layer_kind)(const edgexpu_gguf_info *info, int layer_index);
} edgexpu_arch_plugin;

struct edgexpu_arch_adapter {
    char name[EDGEXPU_TEXT_SMALL];
    int has_qkv_bias;
    int native_forward;
    edgexpu_rope_type rope;
    edgexpu_ffn_type ffn;
    edgexpu_tokenizer_kind tokenizer;
    const edgexpu_arch_tensor_names *tensors;
    const edgexpu_arch_plugin *plugin;
};

const edgexpu_arch_tensor_names *edgexpu_arch_tensors_llama_gguf(void);

int edgexpu_arch_format_name(char *out, size_t out_size, const char *tmpl, int layer);

void edgexpu_arch_register(const edgexpu_arch_plugin *plugin);
void edgexpu_arch_init(void);

int edgexpu_arch_plugin_count(void);
const edgexpu_arch_plugin *edgexpu_arch_plugin_at(int index);

int edgexpu_arch_select_gpt2_tokenizer(
    const edgexpu_gguf_info *info,
    edgexpu_arch_adapter *adapter,
    char *error,
    size_t error_size
);

int edgexpu_arch_from_gguf(
    const edgexpu_gguf_info *info,
    edgexpu_arch_adapter *adapter,
    char *error,
    size_t error_size
);

edgexpu_layer_kind edgexpu_arch_layer_kind(
    const edgexpu_arch_adapter *adapter,
    const edgexpu_gguf_info *info,
    int layer_index
);

const char *edgexpu_rope_type_name(edgexpu_rope_type rope);
const char *edgexpu_ffn_type_name(edgexpu_ffn_type ffn);
const char *edgexpu_tokenizer_kind_name(edgexpu_tokenizer_kind kind);

#ifdef __cplusplus
}
#endif

#endif
