/* 新 architecture 插件模板。复制为 src/arch/<id>.c，在 register.c 调用 edgexpu_arch_register_<id>()。
 * 不要编进默认库：文件名含 _template，CMake 会排除。
 *
 * 1. match()：识别 GGUF general.architecture
 * 2. configure()：填 RoPE / QKV bias / tensor 名（默认同 llama.cpp blk.%d.*）
 * 3. layer_kind()：每层 ATTN_SWIGLU / GATED_DELTA / ATTN_QK_NORM，或未实现时 UNSUPPORTED
 * 4. native_forward=1 才允许 dump-logits / generate 走 cpu.native
 * 5. 加 examples/models/<pack>/ 与 verify.lock；tensor 名在 configure() 里填，不要改 src/native.c 层循环
 */

#include "edgexpu/arch.h"

#include <stdio.h>
#include <string.h>

static int match_example(const edgexpu_gguf_info *info) {
    return info != NULL && strcmp(info->architecture, "example") == 0;
}

static int configure_example(
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
    "example",
    1,
    match_example,
    configure_example,
    NULL
};

void edgexpu_arch_register_example(void) {
    edgexpu_arch_register(&k_plugin);
}
