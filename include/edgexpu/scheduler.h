#ifndef EDGEXPU_SCHEDULER_H
#define EDGEXPU_SCHEDULER_H

#include <stddef.h>

#include "edgexpu/backend.h"
#include "edgexpu/executor.h"
#include "edgexpu/gguf.h"
#include "edgexpu/manifest.h"
#include "edgexpu/profiler.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct edgexpu_schedule_native_ready {
    int loader;
    int tokenizer;
    int kernel;
    int kv;
} edgexpu_schedule_native_ready;

typedef struct edgexpu_schedule_decision {
    edgexpu_executor_job_type job_type;
    char backend[EDGEXPU_TEXT_SMALL];
    char device[EDGEXPU_TEXT_SMALL];
    char policy[EDGEXPU_TEXT_SMALL];
    char reason[EDGEXPU_TEXT_MEDIUM];
    char fallback_reason[EDGEXPU_TEXT_MEDIUM];
} edgexpu_schedule_decision;

typedef struct edgexpu_resource_plan {
    size_t weight_bytes;
    size_t kv_bytes;
    size_t scratch_bytes;
    size_t total_bytes;
    size_t limit_bytes;
    int window;
    int admitted;
    char reason[EDGEXPU_TEXT_MEDIUM];
} edgexpu_resource_plan;

const edgexpu_backend *edgexpu_scheduler_select_backend(
    const edgexpu_model_manifest *manifest,
    char *error,
    size_t error_size
);

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
);

void edgexpu_scheduler_estimate_gguf(
    const edgexpu_gguf_info *info,
    const edgexpu_device_profile *profile,
    int window,
    edgexpu_resource_plan *plan
);

int edgexpu_scheduler_admit(
    const edgexpu_resource_plan *plan,
    char *error,
    size_t error_size
);

int edgexpu_scheduler_plan_job(
    const edgexpu_model_manifest *manifest,
    edgexpu_executor_job_type job_type,
    edgexpu_schedule_decision *decision,
    char *error,
    size_t error_size
);

int edgexpu_scheduler_plan_job_with_profile(
    const edgexpu_model_manifest *manifest,
    const edgexpu_device_profile *profile,
    edgexpu_executor_job_type job_type,
    edgexpu_schedule_decision *decision,
    char *error,
    size_t error_size
);

int edgexpu_scheduler_plan_job_with_context(
    const edgexpu_model_manifest *manifest,
    const edgexpu_device_profile *profile,
    const edgexpu_backend_telemetry *last_telemetry,
    const edgexpu_schedule_native_ready *native,
    edgexpu_executor_job_type job_type,
    edgexpu_schedule_decision *decision,
    char *error,
    size_t error_size
);

#ifdef __cplusplus
}
#endif

#endif
