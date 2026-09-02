#include "edgexpu/scheduler.h"

#include <stdio.h>
#include <string.h>

static void set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
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
    if (strcmp(manifest->fallback_policy, "cpu.llama_cpp") == 0 ||
        strcmp(manifest->fallback_policy, "cpu.baseline") == 0 ||
        strcmp(manifest->primary_artifact.backend, "cpu.llama_cpp") == 0 ||
        strcmp(manifest->primary_artifact.backend, "cpu.baseline") == 0) {
        return cpu_backend;
    }

    set_error(error, error_size, "manifest 未声明当前初版支持的 backend");
    return NULL;
}
