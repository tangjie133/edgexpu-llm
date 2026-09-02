#ifndef EDGEXPU_EXECUTOR_H
#define EDGEXPU_EXECUTOR_H

#include <stddef.h>
#include <stdint.h>

#include "edgexpu/manifest.h"

#ifdef __cplusplus
extern "C" {
#endif

#define EDGEXPU_EXECUTOR_MAX_JOBS 64

typedef struct edgexpu_executor_job edgexpu_executor_job;

typedef int (*edgexpu_executor_job_callback)(
    edgexpu_executor_job *job,
    void *user_data,
    char *error,
    size_t error_size
);

typedef enum edgexpu_executor_job_type {
    EDGEXPU_EXECUTOR_JOB_LOAD_MODEL = 0,
    EDGEXPU_EXECUTOR_JOB_PREPARE_PROMPT,
    EDGEXPU_EXECUTOR_JOB_PREFILL,
    EDGEXPU_EXECUTOR_JOB_DECODE_STEP,
    EDGEXPU_EXECUTOR_JOB_UPDATE_KV_CACHE,
    EDGEXPU_EXECUTOR_JOB_PREFETCH_WEIGHTS,
    EDGEXPU_EXECUTOR_JOB_STREAM_TOKEN,
    EDGEXPU_EXECUTOR_JOB_COLLECT_TELEMETRY
} edgexpu_executor_job_type;

typedef enum edgexpu_executor_job_status {
    EDGEXPU_EXECUTOR_JOB_PENDING = 0,
    EDGEXPU_EXECUTOR_JOB_RUNNING,
    EDGEXPU_EXECUTOR_JOB_COMPLETED,
    EDGEXPU_EXECUTOR_JOB_FAILED
} edgexpu_executor_job_status;

struct edgexpu_executor_job {
    uint64_t id;
    edgexpu_executor_job_type type;
    edgexpu_executor_job_status status;
    char model_id[EDGEXPU_TEXT_SMALL];
    char backend[EDGEXPU_TEXT_SMALL];
    char device[EDGEXPU_TEXT_SMALL];
    char scheduler_policy[EDGEXPU_TEXT_SMALL];
    char scheduler_reason[EDGEXPU_TEXT_MEDIUM];
    char fallback_reason[EDGEXPU_TEXT_MEDIUM];
    double submitted_at_seconds;
    double started_at_seconds;
    double finished_at_seconds;
    edgexpu_executor_job_callback callback;
    void *user_data;
    char error[EDGEXPU_TEXT_MEDIUM];
};

typedef struct edgexpu_executor {
    uint64_t next_job_id;
    size_t job_count;
    edgexpu_executor_job jobs[EDGEXPU_EXECUTOR_MAX_JOBS];
} edgexpu_executor;

typedef struct edgexpu_executor_queue_summary {
    size_t total;
    size_t pending;
    size_t running;
    size_t completed;
    size_t failed;
} edgexpu_executor_queue_summary;

void edgexpu_executor_init(edgexpu_executor *executor);

int edgexpu_executor_submit(
    edgexpu_executor *executor,
    edgexpu_executor_job_type type,
    const char *model_id,
    const char *backend,
    const char *device,
    uint64_t *job_id,
    char *error,
    size_t error_size
);

int edgexpu_executor_submit_with_reason(
    edgexpu_executor *executor,
    edgexpu_executor_job_type type,
    const char *model_id,
    const char *backend,
    const char *device,
    const char *scheduler_reason,
    uint64_t *job_id,
    char *error,
    size_t error_size
);

int edgexpu_executor_submit_runnable(
    edgexpu_executor *executor,
    edgexpu_executor_job_type type,
    const char *model_id,
    const char *backend,
    const char *device,
    const char *scheduler_reason,
    const char *scheduler_policy,
    const char *fallback_reason,
    edgexpu_executor_job_callback callback,
    void *user_data,
    uint64_t *job_id,
    char *error,
    size_t error_size
);

int edgexpu_executor_mark_running(
    edgexpu_executor *executor,
    uint64_t job_id,
    char *error,
    size_t error_size
);

int edgexpu_executor_mark_completed(
    edgexpu_executor *executor,
    uint64_t job_id,
    char *error,
    size_t error_size
);

int edgexpu_executor_mark_failed(
    edgexpu_executor *executor,
    uint64_t job_id,
    const char *job_error,
    char *error,
    size_t error_size
);

int edgexpu_executor_run_job(
    edgexpu_executor *executor,
    uint64_t job_id,
    edgexpu_executor_job_callback callback,
    void *user_data,
    char *error,
    size_t error_size
);

int edgexpu_executor_run_next(
    edgexpu_executor *executor,
    char *error,
    size_t error_size
);

const edgexpu_executor_job *edgexpu_executor_find_job(
    const edgexpu_executor *executor,
    uint64_t job_id
);

size_t edgexpu_executor_job_count(const edgexpu_executor *executor);

const edgexpu_executor_job *edgexpu_executor_job_at(
    const edgexpu_executor *executor,
    size_t index
);

void edgexpu_executor_get_queue_summary(
    const edgexpu_executor *executor,
    edgexpu_executor_queue_summary *summary
);

size_t edgexpu_executor_count_by_status(
    const edgexpu_executor *executor,
    edgexpu_executor_job_status status
);

int edgexpu_executor_has_pending(const edgexpu_executor *executor);

const char *edgexpu_executor_job_type_name(edgexpu_executor_job_type type);
const char *edgexpu_executor_job_status_name(edgexpu_executor_job_status status);

#ifdef __cplusplus
}
#endif

#endif
