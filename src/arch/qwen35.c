#include "edgexpu/arch.h"

#include <stdio.h>
#include <string.h>

static int arch_is(const char *value, const char *expected) {
    return value != NULL && expected != NULL && strcmp(value, expected) == 0;
}

static const edgexpu_arch_tensor_names k_tensors = {
    "token_embd.weight",
    "output.weight",
    "output_norm.weight",
    "output.bias",
    "blk.%d.attn_norm.weight",
    "blk.%d.post_attention_norm.weight",
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
    "blk.%d.ffn_down.bias",
    "blk.%d.attn_q_norm.weight",
    "blk.%d.attn_k_norm.weight",
    "blk.%d.attn_qkv.weight",
    "blk.%d.attn_gate.weight",
    "blk.%d.ssm_conv1d.weight",
    "blk.%d.ssm_dt.bias",
    "blk.%d.ssm_a",
    "blk.%d.ssm_alpha.weight",
    "blk.%d.ssm_beta.weight",
    "blk.%d.ssm_norm.weight",
    "blk.%d.ssm_out.weight"
};

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
    adapter->has_qkv_bias = 0;
    adapter->rope = EDGEXPU_ROPE_NEOX;
    adapter->ffn = EDGEXPU_FFN_SWIGLU;
    adapter->tensors = &k_tensors;
    if (info->ssm_inner_size == 0 || info->ssm_state_size == 0 ||
        info->ssm_group_count == 0 || info->ssm_time_step_rank == 0) {
        snprintf(
            error,
            error_size,
            "architecture=%s 缺少 ssm 超参",
            info->architecture
        );
        return 0;
    }
    return edgexpu_arch_select_gpt2_tokenizer(info, adapter, error, error_size);
}

static int layer_has_tensor(const edgexpu_gguf_info *info, const char *tmpl, int layer_index) {
    char name[96];
    if (info == NULL || tmpl == NULL ||
        !edgexpu_arch_format_name(name, sizeof(name), tmpl, layer_index)) {
        return 0;
    }
    return edgexpu_gguf_find_tensor(info, name) != NULL;
}

static edgexpu_layer_kind layer_kind_qwen35(const edgexpu_gguf_info *info, int layer_index) {
    uint32_t interval;
    if (layer_has_tensor(info, k_tensors.ssm_a, layer_index)) {
        return EDGEXPU_LAYER_GATED_DELTA;
    }
    if (layer_has_tensor(info, k_tensors.attn_q, layer_index)) {
        return EDGEXPU_LAYER_ATTN_QK_NORM;
    }
    interval = info != NULL && info->full_attention_interval > 0 ? info->full_attention_interval : 4u;
    if (interval > 0 && ((uint32_t)layer_index + 1u) % interval == 0u) {
        return EDGEXPU_LAYER_ATTN_QK_NORM;
    }
    return EDGEXPU_LAYER_GATED_DELTA;
}

static const edgexpu_arch_plugin k_plugin = {
    "qwen35",
    1,
    match_qwen35,
    configure_qwen35,
    layer_kind_qwen35
};

void edgexpu_arch_register_qwen35(void) {
    edgexpu_arch_register(&k_plugin);
}
