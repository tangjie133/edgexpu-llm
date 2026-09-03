#include "edgexpu/arch.h"

#include <stdio.h>
#include <string.h>

static int arch_is(const char *value, const char *expected) {
    return value != NULL && expected != NULL && strcmp(value, expected) == 0;
}

static int match_llama(const edgexpu_gguf_info *info) {
    return info != NULL &&
        (arch_is(info->architecture, "llama") || arch_is(info->architecture, "mistral"));
}

static int configure_llama(
    const edgexpu_gguf_info *info,
    edgexpu_arch_adapter *adapter,
    char *error,
    size_t error_size
) {
    snprintf(adapter->name, sizeof(adapter->name), "%s", info->architecture);
    adapter->has_qkv_bias = 0;
    adapter->rope = EDGEXPU_ROPE_NORM;
    adapter->ffn = EDGEXPU_FFN_SWIGLU;
    adapter->tensors = edgexpu_arch_tensors_llama_gguf();
    return edgexpu_arch_select_gpt2_tokenizer(info, adapter, error, error_size);
}

static const edgexpu_arch_plugin k_plugin = {
    "llama",
    1,
    match_llama,
    configure_llama,
    NULL
};

void edgexpu_arch_register_llama(void) {
    edgexpu_arch_register(&k_plugin);
}
