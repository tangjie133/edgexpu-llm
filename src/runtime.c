#include "edgexpu/runtime.h"

#include "edgexpu/chat.h"
#include "edgexpu/executor.h"
#include "edgexpu/scheduler.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#endif

/* 把一次 generate 拆成 executor job。tokenize 前套模型包 chat template，
 * native 与 llama bootstrap 共用格式化后的 prompt。
 */

#define EDGEXPU_GENERATE_JOB_RESERVE 64

static int cpu_path_is_llama(const edgexpu_generation_request *request) {
    return request != NULL && request->cpu_path == EDGEXPU_CPU_PATH_LLAMA_BOOTSTRAP;
}

static int native_decode_ready(const edgexpu_native_session *native) {
    return native != NULL &&
        native->last_hidden != NULL &&
        native->layers != NULL &&
        native->n_layers_cached > 0;
}

static void set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
    }
}

typedef struct generation_job_context {
    edgexpu_runtime *runtime;
    const edgexpu_generation_request *request;
    edgexpu_generation_request local_request;
    edgexpu_generation_result *result;
    char formatted_prompt[EDGEXPU_TEXT_PROMPT];
    char last_piece[EDGEXPU_TEXT_SMALL];
    int decode_stopped;
    int decode_index;
} generation_job_context;

static double runtime_now_seconds(void) {
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) == 0) {
        return (double)timestamp.tv_sec + (double)timestamp.tv_nsec / 1e9;
    }
    return 0.0;
}

typedef struct stream_token_job_context {
    edgexpu_runtime_stream_callback on_token;
    void *user_data;
    char token[EDGEXPU_TEXT_SMALL];
    int token_index;
    int token_count;
} stream_token_job_context;

static size_t utf8_sequence_length(unsigned char lead) {
    if (lead < 0x80) {
        return 1;
    }
    if ((lead & 0xE0) == 0xC0) {
        return 2;
    }
    if ((lead & 0xF0) == 0xE0) {
        return 3;
    }
    if ((lead & 0xF8) == 0xF0) {
        return 4;
    }
    return 1;
}

static int next_approx_token(const char **cursor, char *token, size_t token_size) {
    size_t used = 0;

    if (cursor == NULL || *cursor == NULL || token == NULL || token_size < 2) {
        return 0;
    }

    while (**cursor != '\0' && isspace((unsigned char)**cursor)) {
        (*cursor)++;
    }
    if (**cursor == '\0') {
        return 0;
    }

    if ((unsigned char)**cursor >= 0x80) {
        size_t length = utf8_sequence_length((unsigned char)**cursor);
        if (length >= token_size) {
            length = token_size - 1;
        }
        memcpy(token, *cursor, length);
        token[length] = '\0';
        *cursor += length;
        return 1;
    }

    while (**cursor != '\0' &&
           !isspace((unsigned char)**cursor) &&
           (unsigned char)**cursor < 0x80 &&
           used + 1 < token_size) {
        token[used++] = **cursor;
        (*cursor)++;
    }
    token[used] = '\0';
    return used > 0;
}

static int count_approx_tokens(const char *text) {
    const char *cursor = text != NULL ? text : "";
    char token[EDGEXPU_TEXT_SMALL];
    int count = 0;

    while (next_approx_token(&cursor, token, sizeof(token))) {
        count++;
    }
    return count;
}

static void copy_replay_token(char *output, size_t output_size, const char *token, int token_index) {
    if (output == NULL || output_size == 0) {
        return;
    }

    if (token_index > 0 && token != NULL && token[0] != '\0' && (unsigned char)token[0] < 0x80) {
        size_t copy_len;
        if (output_size < 2) {
            output[0] = '\0';
            return;
        }
        copy_len = strlen(token);
        if (copy_len > output_size - 2) {
            copy_len = output_size - 2;
        }
        output[0] = ' ';
        memcpy(output + 1, token, copy_len);
        output[1 + copy_len] = '\0';
        return;
    }

    if (token == NULL) {
        output[0] = '\0';
        return;
    }
    {
        size_t copy_len = strlen(token);
        if (copy_len > output_size - 1) {
            copy_len = output_size - 1;
        }
        memcpy(output, token, copy_len);
        output[copy_len] = '\0';
    }
}

static int run_scheduled_job_ex(
    edgexpu_runtime *runtime,
    edgexpu_executor_job_type type,
    const char *reason_detail,
    edgexpu_executor_job_callback callback,
    void *user_data,
    char *error,
    size_t error_size
) {
    edgexpu_schedule_decision decision;
    edgexpu_schedule_native_ready native_ready;
    uint64_t job_id = 0;
    char ignored[128] = {0};
    char reason[EDGEXPU_TEXT_MEDIUM];

    if (runtime == NULL || runtime->backend == NULL) {
        set_error(error, error_size, "runtime 尚未准备好 executor job");
        return 0;
    }

    memset(&native_ready, 0, sizeof(native_ready));
    if (runtime->native.loaded) {
        native_ready.loader = 1;
        native_ready.tokenizer = runtime->native.tokenizer.ready;
        native_ready.kernel = 1;
        native_ready.kv = runtime->native.kv.k != NULL;
    }

    if (!edgexpu_scheduler_plan_job_with_context(
            &runtime->manifest,
            runtime->has_device_profile ? &runtime->device_profile : NULL,
            runtime->has_last_telemetry ? &runtime->last_telemetry : NULL,
            &native_ready,
            type,
            &decision,
            ignored,
            sizeof(ignored))) {
        snprintf(decision.backend, sizeof(decision.backend), "%s", runtime->backend->name);
        snprintf(decision.device, sizeof(decision.device), "unknown");
        snprintf(decision.policy, sizeof(decision.policy), "fallback");
        snprintf(decision.reason, sizeof(decision.reason), "scheduler did not provide a job-level decision");
    }

    if (reason_detail != NULL && reason_detail[0] != '\0') {
        size_t used;

        snprintf(reason, sizeof(reason), "%s", decision.reason);
        used = strlen(reason);
        if (used + 1 < sizeof(reason)) {
            snprintf(reason + used, sizeof(reason) - used, "; %s", reason_detail);
        }
    } else {
        snprintf(reason, sizeof(reason), "%s", decision.reason);
    }

    if (!edgexpu_executor_submit_runnable(
            &runtime->executor,
            type,
            runtime->manifest.model_id,
            decision.backend,
            decision.device,
            reason,
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

static int run_scheduled_job(
    edgexpu_runtime *runtime,
    edgexpu_executor_job_type type,
    edgexpu_executor_job_callback callback,
    void *user_data,
    char *error,
    size_t error_size
) {
    return run_scheduled_job_ex(
        runtime,
        type,
        NULL,
        callback,
        user_data,
        error,
        error_size
    );
}

static int bytes_to_mb(size_t bytes) {
    return (int)(bytes / (1024u * 1024u));
}

static void set_finish_reason(edgexpu_generation_result *result, const char *reason) {
    if (result == NULL) {
        return;
    }
    snprintf(result->finish_reason, sizeof(result->finish_reason), "%s", reason != NULL ? reason : "stop");
}

static int prefetch_job_callback(
    edgexpu_executor_job *job,
    void *user_data,
    char *error,
    size_t error_size
) {
    generation_job_context *context = (generation_job_context *)user_data;
    edgexpu_native_session *native;

    (void)job;
    (void)error;
    (void)error_size;
    if (context == NULL || context->runtime == NULL) {
        return 1;
    }
    native = &context->runtime->native;
    if (native->file_map == NULL || native->file_map_size == 0) {
        return 1;
    }
#if defined(POSIX_MADV_WILLNEED)
    (void)posix_madvise((void *)native->file_map, native->file_map_size, POSIX_MADV_WILLNEED);
#elif defined(MADV_WILLNEED)
    (void)madvise((void *)native->file_map, native->file_map_size, MADV_WILLNEED);
#endif
    return 1;
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

static int artifact_is_gguf(const edgexpu_model_manifest *manifest) {
    if (manifest == NULL) {
        return 0;
    }
    return strcmp(manifest->primary_artifact.format, "gguf") == 0 ||
        strstr(manifest->primary_artifact.path, ".gguf") != NULL;
}

static int load_model_job_callback(
    edgexpu_executor_job *job,
    void *user_data,
    char *error,
    size_t error_size
) {
    edgexpu_runtime *runtime = (edgexpu_runtime *)user_data;
    char native_error[256] = {0};
    int native_ok = 0;

    (void)job;
    if (runtime == NULL || runtime->backend == NULL) {
        set_error(error, error_size, "load_model job 缺少 runtime backend");
        return 0;
    }

    /* 产品路径只走 cpu.native。llama 仅 compare 显式 bootstrap，load 失败不改走 llama。 */
    if (artifact_is_gguf(&runtime->manifest) && runtime->backend->load != NULL) {
        native_ok = runtime->backend->load(
            &runtime->native,
            &runtime->manifest,
            native_error,
            sizeof(native_error)
        );
        if (!native_ok) {
            edgexpu_native_free(&runtime->native);
            set_error(error, error_size, native_error[0] != '\0' ? native_error : "cpu.native 加载失败");
            return 0;
        }
        edgexpu_backend_cpu_baseline_bind(&runtime->manifest);
        return 1;
    }

    set_error(error, error_size, "模型加载失败：需要 GGUF 与 cpu.native");
    return 0;
}

/* prompt 已由 generate_stream 套过 chat template。CLI tokenize 不走这条路径。 */
static int tokenize_job_callback(
    edgexpu_executor_job *job,
    void *user_data,
    char *error,
    size_t error_size
) {
    generation_job_context *context = (generation_job_context *)user_data;
    const char *prompt;

    (void)job;
    if (context == NULL || context->runtime == NULL || !context->runtime->native.loaded) {
        return 1;
    }
    prompt = context->request != NULL ? context->request->prompt : "";
    if (context->runtime->backend != NULL && context->runtime->backend->tokenize != NULL) {
        return context->runtime->backend->tokenize(
            &context->runtime->native,
            prompt,
            error,
            error_size
        );
    }
    return edgexpu_native_tokenize(&context->runtime->native, prompt, error, error_size);
}

/* llama 路径只占 KV 槽位；native 路径跑全层 prefill。 */
static int prefill_job_callback(
    edgexpu_executor_job *job,
    void *user_data,
    char *error,
    size_t error_size
) {
    generation_job_context *context = (generation_job_context *)user_data;
    edgexpu_runtime *runtime;
    int tokens;
    int n_new;

    (void)job;
    if (context == NULL || context->runtime == NULL) {
        return 1;
    }
    runtime = context->runtime;
    if (!runtime->native.loaded || runtime->native.kv.k == NULL) {
        return 1;
    }
    tokens = runtime->native.token_count;
    if (tokens < 0) {
        tokens = 0;
    }
    n_new = context->request != NULL && context->request->max_tokens > 0
        ? context->request->max_tokens
        : 1;
    if (runtime->backend != NULL && runtime->backend->ensure_window != NULL) {
        if (!runtime->backend->ensure_window(&runtime->native, tokens, n_new, error, error_size)) {
            return 0;
        }
    } else if (!edgexpu_native_ensure_window(&runtime->native, tokens, n_new, error, error_size)) {
        return 0;
    }
    if (!cpu_path_is_llama(context->request) && runtime->native.output_norm != NULL) {
        if (runtime->backend != NULL && runtime->backend->prefill != NULL) {
            return runtime->backend->prefill(&runtime->native, error, error_size);
        }
        return edgexpu_native_forward_prefill(&runtime->native, error, error_size);
    }
    if (runtime->backend != NULL && runtime->backend->reserve_kv != NULL) {
        return runtime->backend->reserve_kv(&runtime->native, tokens, error, error_size);
    }
    return edgexpu_native_reserve_kv(&runtime->native, tokens, error, error_size);
}

static int update_kv_job_callback(
    edgexpu_executor_job *job,
    void *user_data,
    char *error,
    size_t error_size
) {
    generation_job_context *context = (generation_job_context *)user_data;
    edgexpu_kv_cache *cache;
    int extra;
    int room;

    (void)job;
    if (context == NULL || context->runtime == NULL || !context->runtime->native.loaded) {
        return 1;
    }
    cache = &context->runtime->native.kv;
    if (cache->k == NULL) {
        return 1;
    }
    extra = context->result != NULL ? context->result->completion_tokens_approx : 1;
    if (context->runtime->native.generated_tokens > 0 ||
        native_decode_ready(&context->runtime->native)) {
        if (cache->seq_len > cache->max_seq) {
            snprintf(
                error,
                error_size,
                "KV 超出窗口：seq_len=%d > max_seq=%d",
                cache->seq_len,
                cache->max_seq
            );
            return 0;
        }
        return 1;
    }
    if (extra < 1) {
        extra = 1;
    }
    room = cache->max_seq - cache->seq_len;
    if (extra > room) {
        snprintf(
            error,
            error_size,
            "KV 超出窗口：seq_len=%d + %d > max_seq=%d",
            cache->seq_len,
            extra,
            cache->max_seq
        );
        return 0;
    }
    return edgexpu_kv_cache_extend(cache, extra, error, error_size);
}

/* native 就绪则 generate_next；否则只记完成 token 数，文本由 llama 路径回填。 */
static int decode_job_callback(
    edgexpu_executor_job *job,
    void *user_data,
    char *error,
    size_t error_size
) {
    generation_job_context *context = (generation_job_context *)user_data;
    edgexpu_native_session *native;
    uint32_t token = 0;
    int stopped = 0;
    float temperature = 0.0f;
    float top_p = 1.0f;
    size_t used;

    (void)job;
    if (context == NULL || context->runtime == NULL) {
        set_error(error, error_size, "decode job 缺少 runtime backend");
        return 0;
    }

    native = &context->runtime->native;
    context->last_piece[0] = '\0';
    if (!cpu_path_is_llama(context->request) && native_decode_ready(native)) {
        if (context->request != NULL) {
            temperature = context->request->temperature;
            top_p = context->request->top_p;
        }
        if (context->result != NULL && context->result->backend[0] == '\0') {
            snprintf(context->result->backend, sizeof(context->result->backend), "cpu.native");
            context->result->prompt_tokens_approx = native->prompt_token_count;
        }
        if (context->runtime->backend != NULL && context->runtime->backend->decode_step != NULL) {
            if (!context->runtime->backend->decode_step(
                    native,
                    temperature,
                    top_p,
                    &token,
                    context->last_piece,
                    sizeof(context->last_piece),
                    &stopped,
                    error,
                    error_size)) {
                return 0;
            }
        } else if (!edgexpu_native_generate_next(
                native,
                temperature,
                top_p,
                &token,
                context->last_piece,
                sizeof(context->last_piece),
                &stopped,
                error,
                error_size)) {
            return 0;
        }
        if (stopped) {
            context->decode_stopped = 1;
            return 1;
        }
        if (context->result != NULL && context->last_piece[0] != '\0') {
            used = strlen(context->result->text);
            if (used + strlen(context->last_piece) + 1 < sizeof(context->result->text)) {
                memcpy(context->result->text + used, context->last_piece, strlen(context->last_piece) + 1);
            }
            context->result->completion_tokens_approx += 1;
        }
        return 1;
    }

    if (cpu_path_is_llama(context->request)) {
        const edgexpu_backend *llama = edgexpu_backend_cpu_baseline();
        if (llama == NULL || llama->generate == NULL) {
            set_error(error, error_size, "llama bootstrap generate 不可用");
            return 0;
        }
        return llama->generate(NULL, context->request, context->result, error, error_size);
    }

    set_error(error, error_size, "cpu.native decode 未就绪；产品路径不回退 llama");
    return 0;
}

static int collect_telemetry_job_callback(
    edgexpu_executor_job *job,
    void *user_data,
    char *error,
    size_t error_size
) {
    generation_job_context *context = (generation_job_context *)user_data;

    (void)job;
    (void)error;
    (void)error_size;
    if (context == NULL || context->runtime == NULL || context->result == NULL) {
        return 1;
    }

    context->runtime->last_telemetry = context->result->telemetry;
    context->runtime->has_last_telemetry = 1;
    return 1;
}

static int stream_token_job_callback(
    edgexpu_executor_job *job,
    void *user_data,
    char *error,
    size_t error_size
) {
    stream_token_job_context *context = (stream_token_job_context *)user_data;

    (void)job;
    (void)error;
    (void)error_size;
    if (context == NULL || context->on_token == NULL || context->token[0] == '\0') {
        return 1;
    }

    context->on_token(
        context->token,
        context->token_index,
        context->token_count,
        context->user_data
    );
    return 1;
}

static int emit_one_stream_token(
    edgexpu_runtime *runtime,
    const char *piece,
    int token_index,
    int token_count,
    edgexpu_runtime_stream_callback on_token,
    void *stream_user_data,
    char *error,
    size_t error_size
) {
    stream_token_job_context stream_context;
    char detail[EDGEXPU_TEXT_MEDIUM];

    memset(&stream_context, 0, sizeof(stream_context));
    stream_context.on_token = on_token;
    stream_context.user_data = stream_user_data;
    stream_context.token_index = token_index;
    stream_context.token_count = token_count > 0 ? token_count : 1;
    snprintf(stream_context.token, sizeof(stream_context.token), "%s", piece != NULL ? piece : "");
    snprintf(
        detail,
        sizeof(detail),
        "native token %d/%d: %s",
        token_index + 1,
        stream_context.token_count,
        stream_context.token
    );
    return run_scheduled_job_ex(
        runtime,
        EDGEXPU_EXECUTOR_JOB_STREAM_TOKEN,
        detail,
        stream_token_job_callback,
        &stream_context,
        error,
        error_size
    );
}

static int emit_stream_token_jobs(
    edgexpu_runtime *runtime,
    const char *text,
    edgexpu_runtime_stream_callback on_token,
    void *stream_user_data,
    char *error,
    size_t error_size
) {
    const char *cursor = text != NULL ? text : "";
    stream_token_job_context stream_context;
    char token[EDGEXPU_TEXT_SMALL];
    char detail[EDGEXPU_TEXT_MEDIUM];
    int token_count = count_approx_tokens(text);
    int token_index = 0;
    int recorded_jobs = 0;
    size_t remaining;
    int max_visible_jobs;

    memset(&stream_context, 0, sizeof(stream_context));
    stream_context.on_token = on_token;
    stream_context.user_data = stream_user_data;
    stream_context.token_count = token_count > 0 ? token_count : 1;

    remaining = edgexpu_executor_remaining_capacity(&runtime->executor);
    if (remaining <= 1) {
        set_error(error, error_size, "executor 没有足够容量执行 stream_token job");
        return 0;
    }
    max_visible_jobs = (int)(remaining - 1);
    if (token_count > 0 && max_visible_jobs > token_count) {
        max_visible_jobs = token_count;
    }
    if (max_visible_jobs < 1) {
        max_visible_jobs = 1;
    }

    if (token_count <= 0) {
        snprintf(detail, sizeof(detail), "token 0/0 empty replay");
        return run_scheduled_job_ex(
            runtime,
            EDGEXPU_EXECUTOR_JOB_STREAM_TOKEN,
            detail,
            stream_token_job_callback,
            &stream_context,
            error,
            error_size
        );
    }

    while (next_approx_token(&cursor, token, sizeof(token))) {
        int is_last_visible = recorded_jobs + 1 >= max_visible_jobs && token_index + 1 < token_count;

        copy_replay_token(stream_context.token, sizeof(stream_context.token), token, token_index);
        stream_context.token_index = token_index;
        if (is_last_visible) {
            while (next_approx_token(&cursor, token, sizeof(token))) {
                if (on_token != NULL && stream_context.token[0] != '\0') {
                    on_token(stream_context.token, token_index, token_count, stream_user_data);
                }
                token_index++;
                copy_replay_token(stream_context.token, sizeof(stream_context.token), token, token_index);
                stream_context.token_index = token_index;
            }
            snprintf(
                detail,
                sizeof(detail),
                "replay remaining tokens %d-%d/%d after batched decode",
                recorded_jobs + 1,
                token_count,
                token_count
            );
        } else {
            snprintf(
                detail,
                sizeof(detail),
                "replay token %d/%d after batched decode: %s",
                token_index + 1,
                token_count,
                stream_context.token
            );
        }

        if (!run_scheduled_job_ex(
                runtime,
                EDGEXPU_EXECUTOR_JOB_STREAM_TOKEN,
                detail,
                stream_token_job_callback,
                &stream_context,
                error,
                error_size)) {
            return 0;
        }

        recorded_jobs++;
        token_index++;
        if (is_last_visible) {
            break;
        }
    }

    return 1;
}

void edgexpu_runtime_init(edgexpu_runtime *runtime) {
    if (runtime == NULL) {
        return;
    }
    memset(runtime, 0, sizeof(*runtime));
    edgexpu_executor_init(&runtime->executor);
    edgexpu_native_init(&runtime->native);
}

void edgexpu_runtime_shutdown(edgexpu_runtime *runtime) {
    if (runtime == NULL) {
        return;
    }
    edgexpu_native_free(&runtime->native);
    memset(runtime, 0, sizeof(*runtime));
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

    if (edgexpu_path_is_gguf(manifest_path)) {
        if (!edgexpu_manifest_from_gguf(manifest_path, &runtime->manifest, error, error_size)) {
            return 0;
        }
    } else if (!edgexpu_manifest_load(manifest_path, &runtime->manifest, error, error_size)) {
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
    return edgexpu_runtime_generate_stream(
        runtime,
        request,
        result,
        NULL,
        NULL,
        error,
        error_size
    );
}

int edgexpu_runtime_generate_stream(
    edgexpu_runtime *runtime,
    const edgexpu_generation_request *request,
    edgexpu_generation_result *result,
    edgexpu_runtime_stream_callback on_token,
    void *stream_user_data,
    char *error,
    size_t error_size
) {
    generation_job_context generation_context;

    if (runtime == NULL || !runtime->loaded || runtime->backend == NULL) {
        set_error(error, error_size, "runtime 尚未加载模型");
        return 0;
    }

    if (edgexpu_executor_remaining_capacity(&runtime->executor) < EDGEXPU_GENERATE_JOB_RESERVE) {
        edgexpu_executor_drop_terminal(&runtime->executor);
    }

    generation_context.runtime = runtime;
    generation_context.request = request;
    generation_context.result = result;
    generation_context.last_piece[0] = '\0';
    generation_context.formatted_prompt[0] = '\0';
    generation_context.decode_stopped = 0;
    generation_context.decode_index = 0;
    /* 优先模型包简化模板；没有 {{prompt}} / {{#message}} 的 GGUF Jinja 不会被执行。 */
    if (request != NULL) {
        const char *template_text = runtime->manifest.chat_template;
        if (template_text[0] == '\0') {
            template_text = runtime->native.gguf.chat_template;
        }
        generation_context.local_request = *request;
        if (request->prompt_is_formatted) {
            snprintf(
                generation_context.formatted_prompt,
                sizeof(generation_context.formatted_prompt),
                "%s",
                request->prompt != NULL ? request->prompt : ""
            );
        } else if (!edgexpu_chat_apply(
                template_text,
                request->prompt != NULL ? request->prompt : "",
                generation_context.formatted_prompt,
                sizeof(generation_context.formatted_prompt))) {
            set_error(error, error_size, "chat template 输出过长");
            return 0;
        }
        generation_context.local_request.prompt = generation_context.formatted_prompt;
        generation_context.request = &generation_context.local_request;
    }
    if (result != NULL) {
        memset(result, 0, sizeof(*result));
    }

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
            EDGEXPU_EXECUTOR_JOB_TOKENIZE,
            tokenize_job_callback,
            &generation_context,
            error,
            error_size)) {
        return 0;
    }

    {
        char prefetch_detail[EDGEXPU_TEXT_MEDIUM];
        snprintf(
            prefetch_detail,
            sizeof(prefetch_detail),
            "madvise WILLNEED mmap_bytes=%zu (not flash paging)",
            runtime->native.file_map_size
        );
        if (!run_scheduled_job_ex(
                runtime,
                EDGEXPU_EXECUTOR_JOB_PREFETCH_WEIGHTS,
                prefetch_detail,
                prefetch_job_callback,
                &generation_context,
                error,
                error_size)) {
            return 0;
        }
    }

    if (!run_scheduled_job(
            runtime,
            EDGEXPU_EXECUTOR_JOB_PREFILL,
            prefill_job_callback,
            &generation_context,
            error,
            error_size)) {
        return 0;
    }

    if (request != NULL &&
        request->cpu_path == EDGEXPU_CPU_PATH_NATIVE &&
        !native_decode_ready(&runtime->native)) {
        set_error(error, error_size, "native CPU fallback 尚未就绪");
        return 0;
    }

    if (!cpu_path_is_llama(request) && native_decode_ready(&runtime->native)) {
        int max_new = request != NULL && request->max_tokens > 0 ? request->max_tokens : 1;
        int produced = 0;
        int i;
        double started = runtime_now_seconds();
        char detail[EDGEXPU_TEXT_MEDIUM];

        for (i = 0; i < max_new; i++) {
            if (edgexpu_executor_remaining_capacity(&runtime->executor) < 4) {
                edgexpu_executor_drop_terminal(&runtime->executor);
            }
            generation_context.decode_index = i;
            snprintf(detail, sizeof(detail), "native decode token %d/%d", i + 1, max_new);
            if (!run_scheduled_job_ex(
                    runtime,
                    EDGEXPU_EXECUTOR_JOB_DECODE_STEP,
                    detail,
                    decode_job_callback,
                    &generation_context,
                    error,
                    error_size)) {
                return 0;
            }
            if (generation_context.decode_stopped) {
                break;
            }
            if (!emit_one_stream_token(
                    runtime,
                    generation_context.last_piece,
                    produced,
                    max_new,
                    on_token,
                    stream_user_data,
                    error,
                    error_size)) {
                return 0;
            }
            produced++;
        }
        if (produced == 0 &&
            !emit_one_stream_token(
                runtime,
                "",
                0,
                1,
                on_token,
                stream_user_data,
                error,
                error_size)) {
            return 0;
        }
        if (generation_context.decode_stopped) {
            set_finish_reason(result, "stop");
        } else if (produced >= max_new) {
            set_finish_reason(result, "length");
        } else {
            set_finish_reason(result, "stop");
        }
        if (result != NULL) {
            result->elapsed_seconds = runtime_now_seconds() - started;
            if (result->backend[0] == '\0') {
                snprintf(result->backend, sizeof(result->backend), "cpu.native");
            }
            result->prompt_tokens_approx = runtime->native.prompt_token_count;
            snprintf(result->telemetry.backend, sizeof(result->telemetry.backend), "%s", result->backend);
            snprintf(result->telemetry.device, sizeof(result->telemetry.device), "cpu");
            snprintf(
                result->telemetry.fallback_reason,
                sizeof(result->telemetry.fallback_reason),
                "native token-by-token decode"
            );
            result->telemetry.prefill_seconds = runtime->native.prefill_seconds;
            result->telemetry.decode_seconds = result->elapsed_seconds;
            result->telemetry.total_seconds =
                result->telemetry.prefill_seconds + result->telemetry.decode_seconds;
            result->telemetry.prompt_tokens_approx = result->prompt_tokens_approx;
            result->telemetry.completion_tokens_approx = result->completion_tokens_approx;
            result->telemetry.memory_used_mb = bytes_to_mb(edgexpu_native_memory_bytes(&runtime->native));
        }
    } else if (!run_scheduled_job(
            runtime,
            EDGEXPU_EXECUTOR_JOB_DECODE_STEP,
            decode_job_callback,
            &generation_context,
            error,
            error_size)) {
        return 0;
    } else if (!emit_stream_token_jobs(
            runtime,
            result != NULL ? result->text : "",
            on_token,
            stream_user_data,
            error,
            error_size)) {
        return 0;
    }

    if (result != NULL && result->finish_reason[0] == '\0') {
        int max_new = request != NULL && request->max_tokens > 0 ? request->max_tokens : 1;
        if (result->completion_tokens_approx >= max_new) {
            set_finish_reason(result, "length");
        } else {
            set_finish_reason(result, "stop");
        }
    }

    {
        char kv_detail[EDGEXPU_TEXT_MEDIUM];
        snprintf(
            kv_detail,
            sizeof(kv_detail),
            "seq_len=%d max_seq=%d kv_bytes=%zu mmap_bytes=%zu",
            runtime->native.kv.seq_len,
            runtime->native.kv.max_seq,
            edgexpu_kv_cache_bytes(&runtime->native.kv),
            runtime->native.file_map_size
        );
        if (!run_scheduled_job_ex(
                runtime,
                EDGEXPU_EXECUTOR_JOB_UPDATE_KV_CACHE,
                kv_detail,
                update_kv_job_callback,
                &generation_context,
                error,
                error_size)) {
            return 0;
        }
    }

    if (!run_scheduled_job(
            runtime,
            EDGEXPU_EXECUTOR_JOB_COLLECT_TELEMETRY,
            collect_telemetry_job_callback,
            &generation_context,
            error,
            error_size)) {
        return 0;
    }

    return 1;
}
