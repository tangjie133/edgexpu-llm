#include "edgexpu/runtime.h"

#include "edgexpu/executor.h"
#include "edgexpu/scheduler.h"

#include <stdio.h>
#include <string.h>

static void set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
    }
}

typedef struct generation_job_context {
    edgexpu_runtime *runtime;
    const edgexpu_generation_request *request;
    edgexpu_generation_result *result;
} generation_job_context;

static int run_scheduled_job(
    edgexpu_runtime *runtime,
    edgexpu_executor_job_type type,
    edgexpu_executor_job_callback callback,
    void *user_data,
    char *error,
    size_t error_size
) {
    edgexpu_schedule_decision decision;
    uint64_t job_id = 0;
    char ignored[128] = {0};

    if (runtime == NULL || runtime->backend == NULL) {
        set_error(error, error_size, "runtime 尚未准备好 executor job");
        return 0;
    }

    if (!edgexpu_scheduler_plan_job_with_profile(
            &runtime->manifest,
            runtime->has_device_profile ? &runtime->device_profile : NULL,
            type,
            &decision,
            ignored,
            sizeof(ignored))) {
        snprintf(decision.backend, sizeof(decision.backend), "%s", runtime->backend->name);
        snprintf(decision.device, sizeof(decision.device), "unknown");
        snprintf(decision.policy, sizeof(decision.policy), "fallback");
        snprintf(decision.reason, sizeof(decision.reason), "scheduler did not provide a job-level decision");
    }

    if (!edgexpu_executor_submit_runnable(
            &runtime->executor,
            type,
            runtime->manifest.model_id,
            decision.backend,
            decision.device,
            decision.reason,
            decision.policy,
            decision.fallback_reason,
            callback,
            user_data,
            &job_id,
            error,
            error_size)) {
        return 0;
    }

    (void)job_id;
    return edgexpu_executor_run_next(&runtime->executor, error, error_size);
}

static int noop_job_callback(
    edgexpu_executor_job *job,
    void *user_data,
    char *error,
    size_t error_size
) {
    (void)job;
    (void)user_data;
    (void)error;
    (void)error_size;
    return 1;
}

static int load_model_job_callback(
    edgexpu_executor_job *job,
    void *user_data,
    char *error,
    size_t error_size
) {
    edgexpu_runtime *runtime = (edgexpu_runtime *)user_data;

    (void)job;
    if (runtime == NULL || runtime->backend == NULL) {
        set_error(error, error_size, "load_model job 缺少 runtime backend");
        return 0;
    }

    return runtime->backend->load(&runtime->manifest, error, error_size);
}

static int decode_job_callback(
    edgexpu_executor_job *job,
    void *user_data,
    char *error,
    size_t error_size
) {
    generation_job_context *context = (generation_job_context *)user_data;

    (void)job;
    if (context == NULL || context->runtime == NULL || context->runtime->backend == NULL) {
        set_error(error, error_size, "decode job 缺少 runtime backend");
        return 0;
    }

    return context->runtime->backend->generate(
        context->request,
        context->result,
        error,
        error_size
    );
}

void edgexpu_runtime_init(edgexpu_runtime *runtime) {
    if (runtime == NULL) {
        return;
    }
    memset(runtime, 0, sizeof(*runtime));
    edgexpu_executor_init(&runtime->executor);
}

int edgexpu_runtime_load_model(
    edgexpu_runtime *runtime,
    const char *manifest_path,
    char *error,
    size_t error_size
) {
    const edgexpu_backend *backend;

    if (runtime == NULL) {
        set_error(error, error_size, "runtime 为空");
        return 0;
    }

    if (!edgexpu_manifest_load(manifest_path, &runtime->manifest, error, error_size)) {
        return 0;
    }

    runtime->has_device_profile = edgexpu_profile_device(&runtime->device_profile);

    backend = edgexpu_scheduler_select_backend(&runtime->manifest, error, error_size);
    if (backend == NULL) {
        return 0;
    }

    runtime->backend = backend;
    if (!run_scheduled_job(
            runtime,
            EDGEXPU_EXECUTOR_JOB_LOAD_MODEL,
            load_model_job_callback,
            runtime,
            error,
            error_size)) {
        return 0;
    }

    runtime->loaded = 1;
    return 1;
}

int edgexpu_runtime_generate(
    edgexpu_runtime *runtime,
    const edgexpu_generation_request *request,
    edgexpu_generation_result *result,
    char *error,
    size_t error_size
) {
    generation_job_context generation_context;

    if (runtime == NULL || !runtime->loaded || runtime->backend == NULL) {
        set_error(error, error_size, "runtime 尚未加载模型");
        return 0;
    }

    generation_context.runtime = runtime;
    generation_context.request = request;
    generation_context.result = result;

    if (!run_scheduled_job(
            runtime,
            EDGEXPU_EXECUTOR_JOB_PREPARE_PROMPT,
            noop_job_callback,
            NULL,
            error,
            error_size)) {
        return 0;
    }

    if (!run_scheduled_job(
            runtime,
            EDGEXPU_EXECUTOR_JOB_PREFETCH_WEIGHTS,
            noop_job_callback,
            NULL,
            error,
            error_size)) {
        return 0;
    }

    if (!run_scheduled_job(
            runtime,
            EDGEXPU_EXECUTOR_JOB_PREFILL,
            noop_job_callback,
            NULL,
            error,
            error_size)) {
        return 0;
    }

    if (!run_scheduled_job(
            runtime,
            EDGEXPU_EXECUTOR_JOB_DECODE_STEP,
            decode_job_callback,
            &generation_context,
            error,
            error_size)) {
        return 0;
    }

    if (!run_scheduled_job(
            runtime,
            EDGEXPU_EXECUTOR_JOB_UPDATE_KV_CACHE,
            noop_job_callback,
            NULL,
            error,
            error_size)) {
        return 0;
    }

    if (!run_scheduled_job(
            runtime,
            EDGEXPU_EXECUTOR_JOB_STREAM_TOKEN,
            noop_job_callback,
            NULL,
            error,
            error_size)) {
        return 0;
    }

    if (!run_scheduled_job(
            runtime,
            EDGEXPU_EXECUTOR_JOB_COLLECT_TELEMETRY,
            noop_job_callback,
            NULL,
            error,
            error_size)) {
        return 0;
    }

    return 1;
}
