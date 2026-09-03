#include "edgexpu/scheduler.h"

#include "edgexpu/arch.h"
#include "edgexpu/gguf_quant.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 按 job 类型填 backend/device/policy。native 就绪时 tokenize/prefill/decode 标 cpu.native。 */

static void set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
    }
}

static void copy_text(char *output, size_t output_size, const char *input) {
    if (output != NULL && output_size > 0) {
        snprintf(output, output_size, "%s", input != NULL ? input : "");
    }
}

static int manifest_supports_cpu_baseline(const edgexpu_model_manifest *manifest) {
    if (manifest == NULL) {
        return 0;
    }

    return strcmp(manifest->fallback_policy, "cpu.llama_cpp") == 0 ||
        strcmp(manifest->fallback_policy, "cpu.baseline") == 0 ||
        strcmp(manifest->fallback_policy, "cpu.native") == 0 ||
        strcmp(manifest->primary_artifact.backend, "cpu.llama_cpp") == 0 ||
        strcmp(manifest->primary_artifact.backend, "cpu.baseline") == 0 ||
        strcmp(manifest->primary_artifact.backend, "cpu.native") == 0;
}

static const char *cpu_baseline_reason(edgexpu_executor_job_type job_type) {
    switch (job_type) {
        case EDGEXPU_EXECUTOR_JOB_LOAD_MODEL:
            return "model artifact loading is coordinated by the CPU runtime path";
        case EDGEXPU_EXECUTOR_JOB_PREPARE_PROMPT:
            return "prompt preparation stays on CPU before native tokenizer integration";
        case EDGEXPU_EXECUTOR_JOB_TOKENIZE:
            return "tokenizer is not connected to the native backend yet; CPU placeholder is used";
        case EDGEXPU_EXECUTOR_JOB_PREFILL:
            return "prefill uses CPU baseline until native session or NPU/dNPU is available";
        case EDGEXPU_EXECUTOR_JOB_DECODE_STEP:
            return "decode uses CPU baseline until native session or NPU/dNPU is available";
        case EDGEXPU_EXECUTOR_JOB_UPDATE_KV_CACHE:
            return "KV cache bookkeeping belongs to the memory pipeline";
        case EDGEXPU_EXECUTOR_JOB_PREFETCH_WEIGHTS:
            return "weight prefetch belongs to the flash pipeline";
        case EDGEXPU_EXECUTOR_JOB_STREAM_TOKEN:
            return "stream_token jobs replay approximate tokens until native decode streaming is available";
        case EDGEXPU_EXECUTOR_JOB_COLLECT_TELEMETRY:
            return "telemetry is copied on CPU after backend-owned metrics are reported";
        default:
            return "unknown job type";
    }
}

/* 无 NPU runtime 时回落到 cpu.baseline。 */
static void select_execution_backend(
    const edgexpu_model_manifest *manifest,
    const edgexpu_device_profile *profile,
    char *backend,
    size_t backend_size,
    char *device,
    size_t device_size,
    char *fallback_reason,
    size_t fallback_reason_size
) {
    if (manifest == NULL) {
        copy_text(backend, backend_size, "unknown");
        copy_text(device, device_size, "unknown");
        return;
    }

    if (strcmp(manifest->primary_artifact.backend, "rockchip.rkllm") == 0) {
        if (profile != NULL && !profile->has_rockchip_runtime) {
            copy_text(backend, backend_size, "cpu.baseline");
            copy_text(device, device_size, "cpu");
            copy_text(fallback_reason, fallback_reason_size, "rockchip runtime unavailable in device profile");
            return;
        }
        copy_text(backend, backend_size, "rockchip.rkllm");
        copy_text(device, device_size, "npu.rockchip");
        return;
    }

    if (strcmp(manifest->primary_artifact.backend, "qualcomm.qnn") == 0) {
        if (profile != NULL && !profile->has_qualcomm_runtime) {
            copy_text(backend, backend_size, "cpu.baseline");
            copy_text(device, device_size, "cpu");
            copy_text(fallback_reason, fallback_reason_size, "qualcomm runtime unavailable in device profile");
            return;
        }
        copy_text(backend, backend_size, "qualcomm.qnn");
        copy_text(device, device_size, "npu.qualcomm");
        return;
    }

    if (strcmp(manifest->primary_artifact.backend, "dnpu") == 0) {
        copy_text(backend, backend_size, "dnpu");
        copy_text(device, device_size, "dnpu");
        return;
    }

    if (strcmp(manifest->primary_artifact.backend, "cpu.native") == 0) {
        copy_text(backend, backend_size, "cpu.native");
        copy_text(device, device_size, "cpu");
        return;
    }

    copy_text(backend, backend_size, "cpu.baseline");
    copy_text(device, device_size, "cpu");
}

static void select_policy_name(edgexpu_executor_job_type job_type, char *output, size_t output_size) {
    switch (job_type) {
        case EDGEXPU_EXECUTOR_JOB_LOAD_MODEL:
        case EDGEXPU_EXECUTOR_JOB_PREPARE_PROMPT:
        case EDGEXPU_EXECUTOR_JOB_TOKENIZE:
        case EDGEXPU_EXECUTOR_JOB_STREAM_TOKEN:
        case EDGEXPU_EXECUTOR_JOB_COLLECT_TELEMETRY:
            copy_text(output, output_size, "host_cpu");
            break;
        case EDGEXPU_EXECUTOR_JOB_PREFILL:
        case EDGEXPU_EXECUTOR_JOB_DECODE_STEP:
            copy_text(output, output_size, "stage_execution");
            break;
        case EDGEXPU_EXECUTOR_JOB_UPDATE_KV_CACHE:
            copy_text(output, output_size, "memory_pipeline");
            break;
        case EDGEXPU_EXECUTOR_JOB_PREFETCH_WEIGHTS:
            copy_text(output, output_size, "flash_pipeline");
            break;
        default:
            copy_text(output, output_size, "unknown");
            break;
    }
}

/* 用上一次 decode tok/s 标注下次 plan；native 用 telemetry_keep_native，避免看起来像 baseline。 */
static void apply_telemetry_hint(
    const edgexpu_backend_telemetry *last_telemetry,
    edgexpu_schedule_decision *decision
) {
    double decode_tokens_per_second;
    char annotated_reason[EDGEXPU_TEXT_MEDIUM];
    size_t used;

    if (last_telemetry == NULL || decision == NULL) {
        return;
    }

    if (last_telemetry->decode_seconds <= 0.0 ||
        last_telemetry->completion_tokens_approx <= 0 ||
        (strcmp(decision->backend, "cpu.baseline") != 0 &&
         strcmp(decision->backend, "cpu.native") != 0)) {
        return;
    }

    decode_tokens_per_second =
        (double)last_telemetry->completion_tokens_approx / last_telemetry->decode_seconds;

    copy_text(
        decision->policy,
        sizeof(decision->policy),
        strcmp(decision->backend, "cpu.native") == 0 ? "telemetry_keep_native" : "telemetry_keep_cpu"
    );
    copy_text(annotated_reason, sizeof(annotated_reason), decision->reason);
    used = strlen(annotated_reason);
    if (used + 1 < sizeof(annotated_reason)) {
        snprintf(
            annotated_reason + used,
            sizeof(annotated_reason) - used,
            "; previous %s decode throughput %.2f tok/s",
            last_telemetry->backend,
            decode_tokens_per_second
        );
    }
    copy_text(decision->reason, sizeof(decision->reason), annotated_reason);
    snprintf(
        decision->fallback_reason,
        sizeof(decision->fallback_reason),
        "previous %s decode throughput %.2f tok/s; keep %s until NPU/dNPU backend is available",
        last_telemetry->backend,
        decode_tokens_per_second,
        decision->backend
    );
}

const edgexpu_backend *edgexpu_scheduler_select_backend(
    const edgexpu_model_manifest *manifest,
    char *error,
    size_t error_size
) {
    if (manifest == NULL) {
        set_error(error, error_size, "调度失败：manifest 为空");
        return NULL;
    }

    if (!manifest_supports_cpu_baseline(manifest)) {
        set_error(error, error_size, "manifest 未声明当前支持的 backend");
        return NULL;
    }

    /* 产品执行只选 cpu.native。llama 不由 scheduler 作为某一类模型的正式 backend。 */
    return edgexpu_backend_cpu_native();
}

static size_t mul_size(size_t a, size_t b) {
    if (a != 0 && b > (SIZE_MAX / a)) {
        return SIZE_MAX;
    }
    return a * b;
}

static size_t add_size(size_t a, size_t b) {
    if (b > SIZE_MAX - a) {
        return SIZE_MAX;
    }
    return a + b;
}

static size_t weight_working_heuristic(uint64_t file_size, uint32_t block_count, uint32_t n_embd) {
    size_t max_layer;
    size_t chunk = EDGEXPU_BUDGET_OUTPUT_CHUNK_BYTES;
    size_t row = (size_t)n_embd * 16u + 256u;

    if (file_size == 0) {
        return add_size(chunk, row);
    }
    if (block_count > 0) {
        max_layer = (size_t)(file_size / (uint64_t)block_count);
        max_layer = mul_size(max_layer, 2u);
        if (max_layer > (size_t)file_size) {
            max_layer = (size_t)file_size;
        }
    } else {
        max_layer = (size_t)file_size;
    }
    if (chunk > (size_t)file_size) {
        chunk = (size_t)file_size;
    }
    return add_size(add_size(max_layer, chunk), row);
}

static size_t gguf_weight_working_set(const edgexpu_gguf_info *info) {
    size_t max_blk;
    size_t out_bytes = 0;
    size_t row_bytes = 0;
    const edgexpu_gguf_tensor *tensor;
    uint32_t i;

    if (info == NULL) {
        return 0;
    }
    max_blk = edgexpu_gguf_max_block_bytes(info);
    for (i = 0; i < info->n_tensors; i++) {
        tensor = &info->tensors[i];
        if (edgexpu_gguf_tensor_block_index(tensor->name) >= 0) {
            continue;
        }
        if (strcmp(tensor->name, "output.weight") == 0 ||
            strcmp(tensor->name, "token_embd.weight") == 0) {
            size_t nbytes = edgexpu_gguf_tensor_nbytes(tensor);
            size_t one_row = 0;
            if (tensor->n_dims >= 1) {
                one_row = edgexpu_gguf_row_bytes(tensor->type, (int)tensor->dims[0]);
            }
            if (strcmp(tensor->name, "output.weight") == 0) {
                if (nbytes > EDGEXPU_BUDGET_OUTPUT_CHUNK_BYTES) {
                    nbytes = EDGEXPU_BUDGET_OUTPUT_CHUNK_BYTES;
                }
                if (nbytes > out_bytes) {
                    out_bytes = nbytes;
                }
            }
            if (one_row > row_bytes) {
                row_bytes = one_row;
            }
        }
    }
    if (out_bytes == 0) {
        out_bytes = EDGEXPU_BUDGET_OUTPUT_CHUNK_BYTES;
        if ((size_t)info->file_size < out_bytes) {
            out_bytes = (size_t)info->file_size;
        }
    }
    if (max_blk == 0) {
        return weight_working_heuristic(info->file_size, info->block_count, info->embedding_length);
    }
    return add_size(add_size(max_blk, out_bytes), row_bytes);
}

static uint32_t gguf_kv_layer_count(const edgexpu_gguf_info *info) {
    edgexpu_arch_adapter adapter;
    char error[64];
    uint32_t n_kv = 0;
    uint32_t i;

    if (info == NULL || info->block_count == 0) {
        return 0;
    }
    memset(&adapter, 0, sizeof(adapter));
    if (!edgexpu_arch_from_gguf(info, &adapter, error, sizeof(error))) {
        return info->block_count;
    }
    for (i = 0; i < info->block_count; i++) {
        if (edgexpu_layer_kind_uses_kv(edgexpu_arch_layer_kind(&adapter, info, (int)i))) {
            n_kv++;
        }
    }
    return n_kv;
}

static void fill_native_plan(
    size_t weight_bytes,
    uint64_t file_size,
    uint32_t n_kv_layers,
    uint32_t n_kv_heads,
    int head_dim,
    uint32_t n_embd,
    uint32_t n_ff,
    uint32_t n_heads,
    int window,
    const edgexpu_device_profile *profile,
    edgexpu_resource_plan *plan
) {
    size_t seq;
    size_t per_pos;
    size_t usable_mb;

    if (plan == NULL) {
        return;
    }
    memset(plan, 0, sizeof(*plan));
    if (window < 1) {
        window = 1;
    }
    if (head_dim <= 0 && n_embd > 0 && n_heads > 0) {
        head_dim = (int)(n_embd / n_heads);
    }
    if (n_ff == 0 && n_embd > 0) {
        n_ff = n_embd * 4u;
    }
    plan->window = window;
    plan->file_bytes = (size_t)file_size;
    plan->weight_bytes = weight_bytes;
    plan->kv_bytes = mul_size(
        2u,
        mul_size(
            (size_t)n_kv_layers,
            mul_size((size_t)n_kv_heads, mul_size((size_t)head_dim, mul_size((size_t)window, sizeof(float))))
        )
    );
    seq = (size_t)window;
    if (seq > (size_t)EDGEXPU_BUDGET_SCRATCH_SEQ) {
        seq = (size_t)EDGEXPU_BUDGET_SCRATCH_SEQ;
    }
    per_pos = (size_t)n_embd * 6u + (size_t)n_ff * 2u + (size_t)n_heads * (size_t)head_dim +
        2u * (size_t)n_kv_heads * (size_t)head_dim;
    plan->scratch_bytes = mul_size(seq, mul_size(per_pos, sizeof(float)));
    if (plan->weight_bytes == SIZE_MAX || plan->kv_bytes == SIZE_MAX || plan->scratch_bytes == SIZE_MAX) {
        plan->total_bytes = SIZE_MAX;
    } else {
        plan->total_bytes = add_size(add_size(plan->weight_bytes, plan->kv_bytes), plan->scratch_bytes);
    }

    if (getenv("EDGEXPU_BUDGET_DISABLE") != NULL) {
        plan->limit_bytes = SIZE_MAX;
        plan->admitted = 1;
        snprintf(plan->reason, sizeof(plan->reason), "budget disabled by EDGEXPU_BUDGET_DISABLE");
        return;
    }

    usable_mb = 0;
    if (profile != NULL && profile->memory_total_mb > 0) {
        usable_mb = (size_t)profile->memory_total_mb * 85u / 100u;
    }
    if (usable_mb == 0) {
        plan->limit_bytes = 0;
        plan->admitted = 1;
        snprintf(plan->reason, sizeof(plan->reason), "device memory unknown; admit native load");
        return;
    }

    plan->limit_bytes = usable_mb * 1024u * 1024u;
    if (plan->total_bytes > plan->limit_bytes) {
        plan->admitted = 0;
        snprintf(
            plan->reason,
            sizeof(plan->reason),
            "native budget %zu MB (weights working-set %zu KV %zu scratch %zu window %d, file %zu MB not resident) exceeds %zu MB (85%% of %d MB RAM)",
            plan->total_bytes / (1024u * 1024u),
            plan->weight_bytes / (1024u * 1024u),
            plan->kv_bytes / (1024u * 1024u),
            plan->scratch_bytes / (1024u * 1024u),
            window,
            plan->file_bytes / (1024u * 1024u),
            plan->limit_bytes / (1024u * 1024u),
            profile->memory_total_mb
        );
        return;
    }
    plan->admitted = 1;
    snprintf(
        plan->reason,
        sizeof(plan->reason),
        "native budget %zu MB within %zu MB (weights working-set %zu KV %zu scratch %zu window %d, file %zu MB not resident)",
        plan->total_bytes / (1024u * 1024u),
        plan->limit_bytes / (1024u * 1024u),
        plan->weight_bytes / (1024u * 1024u),
        plan->kv_bytes / (1024u * 1024u),
        plan->scratch_bytes / (1024u * 1024u),
        window,
        plan->file_bytes / (1024u * 1024u)
    );
}

void edgexpu_scheduler_estimate_native(
    uint64_t file_size,
    uint32_t block_count,
    uint32_t n_kv_heads,
    int head_dim,
    uint32_t n_embd,
    uint32_t n_ff,
    uint32_t n_heads,
    int window,
    const edgexpu_device_profile *profile,
    edgexpu_resource_plan *plan
) {
    fill_native_plan(
        weight_working_heuristic(file_size, block_count, n_embd),
        file_size,
        block_count,
        n_kv_heads,
        head_dim,
        n_embd,
        n_ff,
        n_heads,
        window,
        profile,
        plan
    );
}

void edgexpu_scheduler_estimate_gguf(
    const edgexpu_gguf_info *info,
    const edgexpu_device_profile *profile,
    int window,
    edgexpu_resource_plan *plan
) {
    int head_dim = 0;
    uint32_t n_kv_layers;
    if (info == NULL) {
        if (plan != NULL) {
            memset(plan, 0, sizeof(*plan));
        }
        return;
    }
    head_dim = edgexpu_gguf_attn_head_dim(info);
    if (head_dim <= 0) {
        head_dim = edgexpu_gguf_head_dim(info);
    }
    n_kv_layers = gguf_kv_layer_count(info);
    fill_native_plan(
        gguf_weight_working_set(info),
        info->file_size,
        n_kv_layers,
        info->head_count_kv,
        head_dim,
        info->embedding_length,
        info->feed_forward_length,
        info->head_count,
        window,
        profile,
        plan
    );
}

int edgexpu_scheduler_admit(
    const edgexpu_resource_plan *plan,
    char *error,
    size_t error_size
) {
    if (plan == NULL) {
        set_error(error, error_size, "resource plan 为空");
        return 0;
    }
    if (!plan->admitted) {
        set_error(error, error_size, plan->reason[0] != '\0' ? plan->reason : "native load rejected by resource scheduler");
        return 0;
    }
    return 1;
}

int edgexpu_scheduler_plan_job(
    const edgexpu_model_manifest *manifest,
    edgexpu_executor_job_type job_type,
    edgexpu_schedule_decision *decision,
    char *error,
    size_t error_size
) {
    return edgexpu_scheduler_plan_job_with_profile(
        manifest,
        NULL,
        job_type,
        decision,
        error,
        error_size
    );
}

int edgexpu_scheduler_plan_job_with_profile(
    const edgexpu_model_manifest *manifest,
    const edgexpu_device_profile *profile,
    edgexpu_executor_job_type job_type,
    edgexpu_schedule_decision *decision,
    char *error,
    size_t error_size
) {
    return edgexpu_scheduler_plan_job_with_context(
        manifest,
        profile,
        NULL,
        NULL,
        job_type,
        decision,
        error,
        error_size
    );
}

int edgexpu_scheduler_plan_job_with_context(
    const edgexpu_model_manifest *manifest,
    const edgexpu_device_profile *profile,
    const edgexpu_backend_telemetry *last_telemetry,
    const edgexpu_schedule_native_ready *native,
    edgexpu_executor_job_type job_type,
    edgexpu_schedule_decision *decision,
    char *error,
    size_t error_size
) {
    if (manifest == NULL || decision == NULL) {
        set_error(error, error_size, "调度失败：job plan 参数为空");
        return 0;
    }

    memset(decision, 0, sizeof(*decision));
    decision->job_type = job_type;
    select_policy_name(job_type, decision->policy, sizeof(decision->policy));

    switch (job_type) {
        case EDGEXPU_EXECUTOR_JOB_LOAD_MODEL:
        case EDGEXPU_EXECUTOR_JOB_PREPARE_PROMPT:
        case EDGEXPU_EXECUTOR_JOB_TOKENIZE:
        case EDGEXPU_EXECUTOR_JOB_STREAM_TOKEN:
        case EDGEXPU_EXECUTOR_JOB_COLLECT_TELEMETRY:
            copy_text(decision->backend, sizeof(decision->backend), "cpu.runtime");
            copy_text(decision->device, sizeof(decision->device), "cpu");
            break;
        case EDGEXPU_EXECUTOR_JOB_PREFILL:
        case EDGEXPU_EXECUTOR_JOB_DECODE_STEP:
            select_execution_backend(
                manifest,
                profile,
                decision->backend,
                sizeof(decision->backend),
                decision->device,
                sizeof(decision->device),
                decision->fallback_reason,
                sizeof(decision->fallback_reason)
            );
            break;
        case EDGEXPU_EXECUTOR_JOB_UPDATE_KV_CACHE:
            copy_text(decision->backend, sizeof(decision->backend), "memory.manager");
            copy_text(decision->device, sizeof(decision->device), "memory");
            break;
        case EDGEXPU_EXECUTOR_JOB_PREFETCH_WEIGHTS:
            copy_text(decision->backend, sizeof(decision->backend), "flash.manager");
            copy_text(decision->device, sizeof(decision->device), "flash");
            break;
        default:
            set_error(error, error_size, "当前调度器不支持该 job type");
            return 0;
    }

    copy_text(decision->reason, sizeof(decision->reason), cpu_baseline_reason(job_type));
    if (native != NULL) {
        if (native->loader && job_type == EDGEXPU_EXECUTOR_JOB_LOAD_MODEL) {
            copy_text(
                decision->reason,
                sizeof(decision->reason),
                "native GGUF metadata, tokenizer and KV ownership; llama CLI is optional bootstrap"
            );
        }
        if (native->tokenizer && job_type == EDGEXPU_EXECUTOR_JOB_TOKENIZE) {
            copy_text(decision->backend, sizeof(decision->backend), "cpu.native");
            copy_text(decision->policy, sizeof(decision->policy), "native_tokenizer");
            copy_text(
                decision->reason,
                sizeof(decision->reason),
                "native GGUF GPT-2 BPE tokenizer owns this job"
            );
        }
        if (native->kv && job_type == EDGEXPU_EXECUTOR_JOB_PREFILL) {
            if (native->kernel) {
                copy_text(decision->backend, sizeof(decision->backend), "cpu.native");
                copy_text(decision->policy, sizeof(decision->policy), "native_prefill");
                copy_text(
                    decision->reason,
                    sizeof(decision->reason),
                    "native prefill runs all transformer layers and writes full KV cache"
                );
            } else {
                copy_text(
                    decision->reason,
                    sizeof(decision->reason),
                    "native KV cache is reserved; prefill compute still uses CPU baseline"
                );
            }
        }
        if (native->kv && job_type == EDGEXPU_EXECUTOR_JOB_UPDATE_KV_CACHE) {
            copy_text(decision->backend, sizeof(decision->backend), "memory.native");
            copy_text(decision->policy, sizeof(decision->policy), "native_kv");
            copy_text(
                decision->reason,
                sizeof(decision->reason),
                "native KV cache owns sequence slots within the limited window"
            );
        }
        if (native->kernel && job_type == EDGEXPU_EXECUTOR_JOB_DECODE_STEP) {
            copy_text(decision->backend, sizeof(decision->backend), "cpu.native");
            copy_text(decision->policy, sizeof(decision->policy), "native_decode");
            copy_text(
                decision->reason,
                sizeof(decision->reason),
                "native token-by-token decode writes the next KV slot"
            );
        }
        if (native->kernel && job_type == EDGEXPU_EXECUTOR_JOB_STREAM_TOKEN) {
            copy_text(
                decision->reason,
                sizeof(decision->reason),
                "stream_token emits native decoded pieces as they are generated"
            );
        }
    }
    if (job_type == EDGEXPU_EXECUTOR_JOB_PREFILL || job_type == EDGEXPU_EXECUTOR_JOB_DECODE_STEP) {
        apply_telemetry_hint(last_telemetry, decision);
    }
    return 1;
}
