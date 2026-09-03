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

/* 按 job 类型、设备能力和最近 telemetry 给出执行决策。
 * 初版仍主要落到 CPU；NPU/dNPU 路由是后续阶段。
 */

/* native 路径各子系统是否已就绪，供 plan 决定 tokenize/prefill/decode 走哪条 backend。 */
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
