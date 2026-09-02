#ifndef EDGEXPU_SCHEDULER_H
#define EDGEXPU_SCHEDULER_H

#include <stddef.h>

#include "edgexpu/backend.h"
#include "edgexpu/executor.h"
#include "edgexpu/manifest.h"
#include "edgexpu/profiler.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct edgexpu_schedule_decision {
    edgexpu_executor_job_type job_type;
    char backend[EDGEXPU_TEXT_SMALL];
    char device[EDGEXPU_TEXT_SMALL];
    char policy[EDGEXPU_TEXT_SMALL];
    char reason[EDGEXPU_TEXT_MEDIUM];
    char fallback_reason[EDGEXPU_TEXT_MEDIUM];
} edgexpu_schedule_decision;

/* 初版调度器仍只选择一个 backend；job 级 decision 是后续异构调度入口。 */
const edgexpu_backend *edgexpu_scheduler_select_backend(
    const edgexpu_model_manifest *manifest,
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

#ifdef __cplusplus
}
#endif

#endif
