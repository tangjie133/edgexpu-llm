#include "edgexpu/executor.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

/* 固定容量环形语义：jobs[] 线性追加，drop_terminal 压缩掉已结束项。
 * 当前 run_next 在调用线程同步执行 callback。
 */

#if defined(_WIN32)
#include <windows.h>
#endif

static void set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
    }
}

static double now_seconds(void) {
#if defined(_WIN32)
    return (double)GetTickCount64() / 1000.0;
#elif defined(CLOCK_MONOTONIC)
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) == 0) {
        return (double)timestamp.tv_sec + (double)timestamp.tv_nsec / 1000000000.0;
    }
#endif
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void copy_text(char *output, size_t output_size, const char *input) {
    if (output == NULL || output_size == 0) {
        return;
    }

    snprintf(output, output_size, "%s", input != NULL ? input : "");
}

static edgexpu_executor_job *find_mutable_job(edgexpu_executor *executor, uint64_t job_id) {
    size_t index;

    if (executor == NULL) {
        return NULL;
    }

    for (index = 0; index < executor->job_count; index++) {
        if (executor->jobs[index].id == job_id) {
            return &executor->jobs[index];
        }
    }

    return NULL;
}

void edgexpu_executor_init(edgexpu_executor *executor) {
    if (executor == NULL) {
        return;
    }

    memset(executor, 0, sizeof(*executor));
}

int edgexpu_executor_submit(
    edgexpu_executor *executor,
    edgexpu_executor_job_type type,
    const char *model_id,
    const char *backend,
    const char *device,
    uint64_t *job_id,
    char *error,
    size_t error_size
) {
    return edgexpu_executor_submit_with_reason(
        executor,
        type,
        model_id,
        backend,
        device,
        "",
        job_id,
        error,
        error_size
    );
}

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
) {
    return edgexpu_executor_submit_runnable(
        executor,
        type,
        model_id,
        backend,
        device,
        scheduler_reason,
        "",
        "",
        NULL,
        NULL,
        job_id,
        error,
        error_size
    );
}

/* 写入队列槽并挂上 callback；真正执行发生在 run_next。 */
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
) {
    edgexpu_executor_job *job;

    if (executor == NULL) {
        set_error(error, error_size, "executor 为空");
        return 0;
    }

    if (executor->job_count >= EDGEXPU_EXECUTOR_MAX_JOBS) {
        set_error(error, error_size, "executor job 队列已满");
        return 0;
    }

    job = &executor->jobs[executor->job_count++];
    memset(job, 0, sizeof(*job));
    job->id = ++executor->next_job_id;
    job->type = type;
    job->status = EDGEXPU_EXECUTOR_JOB_PENDING;
    job->submitted_at_seconds = now_seconds();
    copy_text(job->model_id, sizeof(job->model_id), model_id);
    copy_text(job->backend, sizeof(job->backend), backend);
    copy_text(job->device, sizeof(job->device), device);
    copy_text(job->scheduler_policy, sizeof(job->scheduler_policy), scheduler_policy);
    copy_text(job->scheduler_reason, sizeof(job->scheduler_reason), scheduler_reason);
    copy_text(job->fallback_reason, sizeof(job->fallback_reason), fallback_reason);
    job->callback = callback;
    job->user_data = user_data;

    if (job_id != NULL) {
        *job_id = job->id;
    }

    return 1;
}

int edgexpu_executor_mark_running(
    edgexpu_executor *executor,
    uint64_t job_id,
    char *error,
    size_t error_size
) {
    edgexpu_executor_job *job = find_mutable_job(executor, job_id);

    if (job == NULL) {
        set_error(error, error_size, "executor job 不存在");
        return 0;
    }

    job->status = EDGEXPU_EXECUTOR_JOB_RUNNING;
    job->started_at_seconds = now_seconds();
    return 1;
}

int edgexpu_executor_mark_completed(
    edgexpu_executor *executor,
    uint64_t job_id,
    char *error,
    size_t error_size
) {
    edgexpu_executor_job *job = find_mutable_job(executor, job_id);

    if (job == NULL) {
        set_error(error, error_size, "executor job 不存在");
        return 0;
    }

    job->status = EDGEXPU_EXECUTOR_JOB_COMPLETED;
    job->finished_at_seconds = now_seconds();
    job->error[0] = '\0';
    return 1;
}

int edgexpu_executor_mark_failed(
    edgexpu_executor *executor,
    uint64_t job_id,
    const char *job_error,
    char *error,
    size_t error_size
) {
    edgexpu_executor_job *job = find_mutable_job(executor, job_id);

    if (job == NULL) {
        set_error(error, error_size, "executor job 不存在");
        return 0;
    }

    job->status = EDGEXPU_EXECUTOR_JOB_FAILED;
    job->finished_at_seconds = now_seconds();
    copy_text(job->error, sizeof(job->error), job_error);
    return 1;
}

int edgexpu_executor_run_job(
    edgexpu_executor *executor,
    uint64_t job_id,
    edgexpu_executor_job_callback callback,
    void *user_data,
    char *error,
    size_t error_size
) {
    edgexpu_executor_job *job = find_mutable_job(executor, job_id);
    char job_error[EDGEXPU_TEXT_MEDIUM] = {0};
    edgexpu_executor_job_callback job_callback;
    void *job_user_data;

    if (job == NULL) {
        set_error(error, error_size, "executor job 不存在");
        return 0;
    }

    if (job->status != EDGEXPU_EXECUTOR_JOB_PENDING) {
        set_error(error, error_size, "executor job 不是 pending 状态");
        return 0;
    }

    if (!edgexpu_executor_mark_running(executor, job_id, error, error_size)) {
        return 0;
    }

    job_callback = callback != NULL ? callback : job->callback;
    job_user_data = callback != NULL ? user_data : job->user_data;

    if (job_callback != NULL && !job_callback(job, job_user_data, job_error, sizeof(job_error))) {
        if (job_error[0] == '\0') {
            snprintf(job_error, sizeof(job_error), "executor job callback 执行失败");
        }
        edgexpu_executor_mark_failed(executor, job_id, job_error, error, error_size);
        set_error(error, error_size, job_error);
        return 0;
    }

    return edgexpu_executor_mark_completed(executor, job_id, error, error_size);
}

/* 找第一个 pending job，调用其 callback，再标 completed/failed。 */
int edgexpu_executor_run_next(
    edgexpu_executor *executor,
    char *error,
    size_t error_size
) {
    size_t index;

    if (executor == NULL) {
        set_error(error, error_size, "executor 为空");
        return 0;
    }

    for (index = 0; index < executor->job_count; index++) {
        if (executor->jobs[index].status == EDGEXPU_EXECUTOR_JOB_PENDING) {
            return edgexpu_executor_run_job(
                executor,
                executor->jobs[index].id,
                NULL,
                NULL,
                error,
                error_size
            );
        }
    }

    set_error(error, error_size, "executor 没有 pending job");
    return 0;
}

const edgexpu_executor_job *edgexpu_executor_find_job(
    const edgexpu_executor *executor,
    uint64_t job_id
) {
    size_t index;

    if (executor == NULL) {
        return NULL;
    }

    for (index = 0; index < executor->job_count; index++) {
        if (executor->jobs[index].id == job_id) {
            return &executor->jobs[index];
        }
    }

    return NULL;
}

size_t edgexpu_executor_job_count(const edgexpu_executor *executor) {
    return executor != NULL ? executor->job_count : 0;
}

const edgexpu_executor_job *edgexpu_executor_job_at(
    const edgexpu_executor *executor,
    size_t index
) {
    if (executor == NULL || index >= executor->job_count) {
        return NULL;
    }

    return &executor->jobs[index];
}

void edgexpu_executor_get_queue_summary(
    const edgexpu_executor *executor,
    edgexpu_executor_queue_summary *summary
) {
    size_t index;

    if (summary == NULL) {
        return;
    }

    memset(summary, 0, sizeof(*summary));
    if (executor == NULL) {
        return;
    }

    summary->total = executor->job_count;
    for (index = 0; index < executor->job_count; index++) {
        switch (executor->jobs[index].status) {
            case EDGEXPU_EXECUTOR_JOB_PENDING:
                summary->pending++;
                break;
            case EDGEXPU_EXECUTOR_JOB_RUNNING:
                summary->running++;
                break;
            case EDGEXPU_EXECUTOR_JOB_COMPLETED:
                summary->completed++;
                break;
            case EDGEXPU_EXECUTOR_JOB_FAILED:
                summary->failed++;
                break;
            default:
                break;
        }
    }
}

size_t edgexpu_executor_count_by_status(
    const edgexpu_executor *executor,
    edgexpu_executor_job_status status
) {
    size_t index;
    size_t count = 0;

    if (executor == NULL) {
        return 0;
    }

    for (index = 0; index < executor->job_count; index++) {
        if (executor->jobs[index].status == status) {
            count++;
        }
    }

    return count;
}

int edgexpu_executor_has_pending(const edgexpu_executor *executor) {
    return edgexpu_executor_count_by_status(executor, EDGEXPU_EXECUTOR_JOB_PENDING) > 0;
}

size_t edgexpu_executor_remaining_capacity(const edgexpu_executor *executor) {
    if (executor == NULL || executor->job_count >= EDGEXPU_EXECUTOR_MAX_JOBS) {
        return 0;
    }

    return EDGEXPU_EXECUTOR_MAX_JOBS - executor->job_count;
}

void edgexpu_executor_drop_terminal(edgexpu_executor *executor) {
    size_t read_index;
    size_t write_index = 0;

    if (executor == NULL) {
        return;
    }

    for (read_index = 0; read_index < executor->job_count; read_index++) {
        edgexpu_executor_job_status status = executor->jobs[read_index].status;
        if (status == EDGEXPU_EXECUTOR_JOB_PENDING || status == EDGEXPU_EXECUTOR_JOB_RUNNING) {
            if (write_index != read_index) {
                executor->jobs[write_index] = executor->jobs[read_index];
            }
            write_index++;
        }
    }

    if (write_index < executor->job_count) {
        memset(&executor->jobs[write_index], 0, sizeof(executor->jobs[0]) * (executor->job_count - write_index));
    }
    executor->job_count = write_index;
}

const char *edgexpu_executor_job_type_name(edgexpu_executor_job_type type) {
    switch (type) {
        case EDGEXPU_EXECUTOR_JOB_LOAD_MODEL:
            return "load_model";
        case EDGEXPU_EXECUTOR_JOB_PREPARE_PROMPT:
            return "prepare_prompt";
        case EDGEXPU_EXECUTOR_JOB_TOKENIZE:
            return "tokenize";
        case EDGEXPU_EXECUTOR_JOB_PREFILL:
            return "prefill";
        case EDGEXPU_EXECUTOR_JOB_DECODE_STEP:
            return "decode_step";
        case EDGEXPU_EXECUTOR_JOB_UPDATE_KV_CACHE:
            return "update_kv_cache";
        case EDGEXPU_EXECUTOR_JOB_PREFETCH_WEIGHTS:
            return "prefetch_weights";
        case EDGEXPU_EXECUTOR_JOB_STREAM_TOKEN:
            return "stream_token";
        case EDGEXPU_EXECUTOR_JOB_COLLECT_TELEMETRY:
            return "collect_telemetry";
        default:
            return "unknown";
    }
}

const char *edgexpu_executor_job_status_name(edgexpu_executor_job_status status) {
    switch (status) {
        case EDGEXPU_EXECUTOR_JOB_PENDING:
            return "pending";
        case EDGEXPU_EXECUTOR_JOB_RUNNING:
            return "running";
        case EDGEXPU_EXECUTOR_JOB_COMPLETED:
            return "completed";
        case EDGEXPU_EXECUTOR_JOB_FAILED:
            return "failed";
        default:
            return "unknown";
    }
}
