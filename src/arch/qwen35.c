#include "edgexpu/arch.h"

#include <stdio.h>
#include <string.h>

static int arch_is(const char *value, const char *expected) {
    return value != NULL && expected != NULL && strcmp(value, expected) == 0;
}

static int match_qwen35(const edgexpu_gguf_info *info) {
    return info != NULL &&
        (arch_is(info->architecture, "qwen35") || arch_is(info->architecture, "qwen3.5"));
}

static int configure_qwen35(
    const edgexpu_gguf_info *info,
    edgexpu_arch_adapter *adapter,
    char *error,
    size_t error_size
) {
    snprintf(adapter->name, sizeof(adapter->name), "%s", info->architecture);
    adapter->tensors = edgexpu_arch_tensors_llama_gguf();
    snprintf(
        error,
        error_size,
        "architecture=%s 是 Attention+SSM 混合结构，当前 cpu.native 尚未实现该 adapter",
        info->architecture
    );
    return 0;
}

static edgexpu_layer_kind layer_kind_qwen35(const edgexpu_gguf_info *info, int layer_index) {
    (void)info;
    (void)layer_index;
    return EDGEXPU_LAYER_UNSUPPORTED;
}

static const edgexpu_arch_plugin k_plugin = {
    "qwen35",
    0,
    match_qwen35,
    configure_qwen35,
    layer_kind_qwen35
};

void edgexpu_arch_register_qwen35(void) {
    edgexpu_arch_register(&k_plugin);
}
