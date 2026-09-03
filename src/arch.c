#include "edgexpu/arch.h"

#include <stdio.h>
#include <string.h>

/* 按 GGUF general.architecture 选择 RoPE / bias / FFN / tokenizer。
 * 新增架构时在这里加 adapter，不要改 native.c 默认成某一个模型。
 */

static void set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
    }
}

static int arch_is(const char *value, const char *expected) {
    return value != NULL && expected != NULL && strcmp(value, expected) == 0;
}

static int select_tokenizer(
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
        "不支持的 tokenizer.ggml.model: %s（当前 adapter 只接 GPT-2 BPE）",
        info->tokenizer_model
    );
    return 0;
}

int edgexpu_arch_from_gguf(
    const edgexpu_gguf_info *info,
    edgexpu_arch_adapter *adapter,
    char *error,
    size_t error_size
) {
    const char *name;

    if (info == NULL || adapter == NULL) {
        set_error(error, error_size, "architecture adapter 参数为空");
        return 0;
    }

    memset(adapter, 0, sizeof(*adapter));
    name = info->architecture;

    if (arch_is(name, "qwen2") || arch_is(name, "qwen2vl")) {
        snprintf(adapter->name, sizeof(adapter->name), "%s", name);
        adapter->has_qkv_bias = 1;
        adapter->rope = EDGEXPU_ROPE_NEOX;
        adapter->ffn = EDGEXPU_FFN_SWIGLU;
        return select_tokenizer(info, adapter, error, error_size);
    }

    if (arch_is(name, "llama") || arch_is(name, "mistral")) {
        snprintf(adapter->name, sizeof(adapter->name), "%s", name);
        adapter->has_qkv_bias = 0;
        adapter->rope = EDGEXPU_ROPE_NORM;
        adapter->ffn = EDGEXPU_FFN_SWIGLU;
        return select_tokenizer(info, adapter, error, error_size);
    }

    snprintf(error, error_size, "不支持的 GGUF architecture: %s", name != NULL ? name : "");
    return 0;
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
