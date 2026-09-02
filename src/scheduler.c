#include "edgexpu/scheduler.h"

#include <stdio.h>
#include <string.h>

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
        strcmp(manifest->primary_artifact.backend, "cpu.llama_cpp") == 0 ||
        strcmp(manifest->primary_artifact.backend, "cpu.baseline") == 0;
}

static const char *cpu_baseline_reason(edgexpu_executor_job_type job_type) {
    switch (job_type) {
        case EDGEXPU_EXECUTOR_JOB_LOAD_MODEL:
            return "model artifact loading is coordinated by the CPU runtime path";
        case EDGEXPU_EXECUTOR_JOB_PREPARE_PROMPT:
            return "prompt preparation stays on CPU before native tokenizer integration";
        case EDGEXPU_EXECUTOR_JOB_PREFILL:
            return "prefill uses CPU baseline until NPU/dNPU execution is available";
        case EDGEXPU_EXECUTOR_JOB_DECODE_STEP:
            return "decode uses CPU baseline until native async executor is available";
        case EDGEXPU_EXECUTOR_JOB_UPDATE_KV_CACHE:
            return "KV cache bookkeeping belongs to the memory pipeline";
        case EDGEXPU_EXECUTOR_JOB_PREFETCH_WEIGHTS:
            return "weight prefetch belongs to the flash pipeline";
        case EDGEXPU_EXECUTOR_JOB_STREAM_TOKEN:
            return "stream handoff is handled by the CPU server path";
        case EDGEXPU_EXECUTOR_JOB_COLLECT_TELEMETRY:
            return "telemetry collection stays on CPU until backend-owned metrics are available";
        default:
            return "unknown job type";
    }
}

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

    copy_text(backend, backend_size, "cpu.baseline");
    copy_text(device, device_size, "cpu");
}

static void select_policy_name(edgexpu_executor_job_type job_type, char *output, size_t output_size) {
    switch (job_type) {
        case EDGEXPU_EXECUTOR_JOB_LOAD_MODEL:
        case EDGEXPU_EXECUTOR_JOB_PREPARE_PROMPT:
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

const edgexpu_backend *edgexpu_scheduler_select_backend(
    const edgexpu_model_manifest *manifest,
    char *error,
    size_t error_size
) {
    const edgexpu_backend *cpu_backend;

    if (manifest == NULL) {
        set_error(error, error_size, "调度失败：manifest 为空");
        return NULL;
    }

    /* 初版只选择 CPU baseline。
     * 后续这里会扩展为按 prefill/decode/verification 分阶段选择 backend。
     */
    cpu_backend = edgexpu_backend_cpu_baseline();
    if (manifest_supports_cpu_baseline(manifest)) {
        return cpu_backend;
    }

    set_error(error, error_size, "manifest 未声明当前初版支持的 backend");
    return NULL;
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
    return 1;
}
