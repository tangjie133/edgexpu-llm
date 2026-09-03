#include "edgexpu/arch.h"

#include <stdio.h>
#include <string.h>

static const edgexpu_arch_tensor_names k_llama_gguf = {
    "token_embd.weight",
    "output.weight",
    "output_norm.weight",
    "output.bias",
    "blk.%d.attn_norm.weight",
    "blk.%d.ffn_norm.weight",
    "blk.%d.attn_q.weight",
    "blk.%d.attn_k.weight",
    "blk.%d.attn_v.weight",
    "blk.%d.attn_output.weight",
    "blk.%d.ffn_gate.weight",
    "blk.%d.ffn_up.weight",
    "blk.%d.ffn_down.weight",
    "blk.%d.attn_q.bias",
    "blk.%d.attn_k.bias",
    "blk.%d.attn_v.bias",
    "blk.%d.attn_output.bias",
    "blk.%d.ffn_gate.bias",
    "blk.%d.ffn_up.bias",
    "blk.%d.ffn_down.bias"
};

const edgexpu_arch_tensor_names *edgexpu_arch_tensors_llama_gguf(void) {
    return &k_llama_gguf;
}

int edgexpu_arch_format_name(char *out, size_t out_size, const char *tmpl, int layer) {
    int written;
    if (out == NULL || out_size == 0 || tmpl == NULL) {
        return 0;
    }
    if (strstr(tmpl, "%d") != NULL) {
        written = snprintf(out, out_size, tmpl, layer);
    } else {
        written = snprintf(out, out_size, "%s", tmpl);
    }
    return written > 0 && (size_t)written < out_size;
}

int edgexpu_arch_select_gpt2_tokenizer(
    const edgexpu_gguf_info *info,
    edgexpu_arch_adapter *adapter,
    char *error,
    size_t error_size
) {
    if (info->tokenizer_model[0] == '\0' || strcmp(info->tokenizer_model, "gpt2") == 0) {
        adapter->tokenizer = EDGEXPU_TOKENIZER_GPT2_BPE;
        return 1;
    }
    snprintf(
        error,
        error_size,
        "不支持的 tokenizer.ggml.model: %s（当前 plugin 只接 GPT-2 BPE）",
        info->tokenizer_model
    );
    return 0;
}

edgexpu_layer_kind edgexpu_arch_layer_kind(
    const edgexpu_arch_adapter *adapter,
    const edgexpu_gguf_info *info,
    int layer_index
) {
    if (adapter != NULL && adapter->plugin != NULL && adapter->plugin->layer_kind != NULL) {
        return adapter->plugin->layer_kind(info, layer_index);
    }
    return EDGEXPU_LAYER_ATTN_SWIGLU;
}

int edgexpu_layer_kind_uses_kv(edgexpu_layer_kind kind) {
    return kind == EDGEXPU_LAYER_ATTN_SWIGLU || kind == EDGEXPU_LAYER_ATTN_QK_NORM;
}

const char *edgexpu_rope_type_name(edgexpu_rope_type rope) {
    switch (rope) {
        case EDGEXPU_ROPE_NEOX:
            return "neox";
        case EDGEXPU_ROPE_NORM:
            return "norm";
        default:
            return "none";
    }
}

const char *edgexpu_ffn_type_name(edgexpu_ffn_type ffn) {
    (void)ffn;
    return "swiglu";
}

const char *edgexpu_tokenizer_kind_name(edgexpu_tokenizer_kind kind) {
    (void)kind;
    return "gpt2_bpe";
}
