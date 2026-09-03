#include "edgexpu/executor.h"
#include "edgexpu/manifest.h"
#include "edgexpu/native.h"
#include "edgexpu/profiler.h"
#include "edgexpu/runtime.h"
#include "edgexpu/scheduler.h"
#include "edgexpu/server.h"

#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* native CLI 是初版最重要的验证入口。
 * 它对应 PowerInfer 的本地二进制使用方式，而不是 Python web app。
 */

static void print_usage(void) {
    printf("EdgeXPU-LLM native runtime\n");
    printf("\n");
    printf("Usage:\n");
    printf("  edgexpu capabilities\n");
    printf("  edgexpu inspect-manifest <manifest.json>\n");
    printf("  edgexpu generate <manifest.json|model.gguf> <prompt> [max_tokens]\n");
    printf("  edgexpu benchmark <manifest.json|model.gguf> <prompt> [max_tokens]\n");
    printf("  edgexpu compare <manifest.json|model.gguf> <prompt>\n");
    printf("  edgexpu trace <manifest.json|model.gguf> <prompt>\n");
    printf("  edgexpu inspect-gguf <model.gguf>\n");
    printf("  edgexpu tokenize <manifest.json|model.gguf> <text>\n");
    printf("  edgexpu dump-logits <manifest.json|model.gguf> <prompt> [n]\n");
    printf("  edgexpu executor-selftest\n");
    printf("  edgexpu scheduler-selftest\n");
    printf("  edgexpu native-selftest [model.gguf]\n");
    printf("  edgexpu serve <manifest.json> [port]\n");
}

/* 把 compare/benchmark 的文本安全写进 JSON 字符串。 */
static void json_escape(const char *input, char *output, size_t output_size) {
    size_t used = 0;

    if (output_size == 0) {
        return;
    }
    output[0] = '\0';
    if (input == NULL) {
        return;
    }

    while (*input != '\0' && used + 2 < output_size) {
        unsigned char c = (unsigned char)*input++;
        if (c == '"' || c == '\\') {
            output[used++] = '\\';
            output[used++] = (char)c;
        } else if (c == '\n') {
            output[used++] = '\\';
            output[used++] = 'n';
        } else if (c == '\r') {
            output[used++] = '\\';
            output[used++] = 'r';
        } else if (c >= 32) {
            output[used++] = (char)c;
        }
    }
    output[used] = '\0';
}

static int command_capabilities(void) {
    edgexpu_device_profile profile;

    if (!edgexpu_profile_device(&profile)) {
        fprintf(stderr, "能力探测失败\n");
        return 1;
    }

    edgexpu_profile_print_json(&profile);
    return 0;
}

static int command_inspect_manifest(const char *path) {
    edgexpu_model_manifest manifest;
    char error[256] = {0};

    if (!edgexpu_manifest_load(path, &manifest, error, sizeof(error))) {
        fprintf(stderr, "manifest 读取失败：%s\n", error);
        return 1;
    }

    edgexpu_manifest_print(&manifest);
    return 0;
}

static void print_executor_trace_json(const edgexpu_executor *executor) {
    size_t index;
    size_t count = edgexpu_executor_job_count(executor);

    printf("  \"executor_trace\": [\n");
    for (index = 0; index < count; index++) {
        const edgexpu_executor_job *job = edgexpu_executor_job_at(executor, index);
        char escaped_error[EDGEXPU_TEXT_MEDIUM * 2];
        char escaped_fallback[EDGEXPU_TEXT_MEDIUM * 2];
        char escaped_reason[EDGEXPU_TEXT_MEDIUM * 2];
        double elapsed_seconds = 0.0;

        if (job == NULL) {
            continue;
        }

        if (job->finished_at_seconds > 0.0 && job->started_at_seconds > 0.0) {
            elapsed_seconds = job->finished_at_seconds - job->started_at_seconds;
        }
        json_escape(job->error, escaped_error, sizeof(escaped_error));
        json_escape(job->fallback_reason, escaped_fallback, sizeof(escaped_fallback));
        json_escape(job->scheduler_reason, escaped_reason, sizeof(escaped_reason));

        printf("    {");
        printf("\"id\": %" PRIu64 ", ", job->id);
        printf("\"type\": \"%s\", ", edgexpu_executor_job_type_name(job->type));
        printf("\"status\": \"%s\", ", edgexpu_executor_job_status_name(job->status));
        printf("\"backend\": \"%s\", ", job->backend);
        printf("\"device\": \"%s\", ", job->device);
        printf("\"scheduler_policy\": \"%s\", ", job->scheduler_policy);
        printf("\"scheduler_reason\": \"%s\", ", escaped_reason);
        printf("\"fallback_reason\": \"%s\", ", escaped_fallback);
        printf("\"elapsed_seconds\": %.6f, ", elapsed_seconds);
        printf("\"error\": \"%s\"", escaped_error);
        printf("}%s\n", index + 1 < count ? "," : "");
    }
    printf("  ]");
}

static void print_executor_queue_summary_json(const edgexpu_executor *executor) {
    edgexpu_executor_queue_summary summary;

    edgexpu_executor_get_queue_summary(executor, &summary);
    printf("  \"queue_summary\": {");
    printf("\"total\": %u, ", (unsigned)summary.total);
    printf("\"pending\": %u, ", (unsigned)summary.pending);
    printf("\"running\": %u, ", (unsigned)summary.running);
    printf("\"completed\": %u, ", (unsigned)summary.completed);
    printf("\"failed\": %u", (unsigned)summary.failed);
    printf("}");
}

static void print_backend_telemetry_json(const edgexpu_backend_telemetry *telemetry) {
    char escaped_fallback[EDGEXPU_TEXT_MEDIUM * 2];

    if (telemetry == NULL) {
        printf("  \"backend_telemetry\": null");
        return;
    }

    json_escape(telemetry->fallback_reason, escaped_fallback, sizeof(escaped_fallback));
    printf("  \"backend_telemetry\": {");
    printf("\"backend\": \"%s\", ", telemetry->backend);
    printf("\"device\": \"%s\", ", telemetry->device);
    printf("\"total_seconds\": %.6f, ", telemetry->total_seconds);
    printf("\"prefill_seconds\": %.6f, ", telemetry->prefill_seconds);
    printf("\"decode_seconds\": %.6f, ", telemetry->decode_seconds);
    printf("\"prompt_tokens_approx\": %d, ", telemetry->prompt_tokens_approx);
    printf("\"completion_tokens_approx\": %d, ", telemetry->completion_tokens_approx);
    printf("\"memory_used_mb\": %d, ", telemetry->memory_used_mb);
    printf("\"fallback_reason\": \"%s\"", escaped_fallback);
    printf("}");
}

static void print_backend_telemetry_table(const edgexpu_backend_telemetry *telemetry) {
    if (telemetry == NULL) {
        return;
    }

    printf("\nBackend Telemetry\n");
    printf("-----------------\n");
    printf("backend=%s device=%s total=%.6fs prefill=%.6fs decode=%.6fs memory=%dMB\n",
           telemetry->backend,
           telemetry->device,
           telemetry->total_seconds,
           telemetry->prefill_seconds,
           telemetry->decode_seconds,
           telemetry->memory_used_mb);
    printf("fallback_reason=%s\n", telemetry->fallback_reason);
}

static void print_executor_trace_table(const edgexpu_executor *executor) {
    size_t index;
    size_t count = edgexpu_executor_job_count(executor);
    edgexpu_executor_queue_summary summary;

    edgexpu_executor_get_queue_summary(executor, &summary);

    printf("\nQueue Summary\n");
    printf("-------------\n");
    printf("total=%u pending=%u running=%u completed=%u failed=%u\n",
           (unsigned)summary.total,
           (unsigned)summary.pending,
           (unsigned)summary.running,
           (unsigned)summary.completed,
           (unsigned)summary.failed);
    printf("\nExecutor Trace\n");
    printf("--------------\n");
    printf("%-4s %-18s %-10s %-14s %-14s %-16s %-10s %s\n",
           "ID",
           "TYPE",
           "STATUS",
           "DEVICE",
           "BACKEND",
           "POLICY",
           "ELAPSED",
           "SCHEDULER_REASON");

    for (index = 0; index < count; index++) {
        const edgexpu_executor_job *job = edgexpu_executor_job_at(executor, index);
        double elapsed_seconds = 0.0;

        if (job == NULL) {
            continue;
        }

        if (job->finished_at_seconds > 0.0 && job->started_at_seconds > 0.0) {
            elapsed_seconds = job->finished_at_seconds - job->started_at_seconds;
        }

        printf("%-4" PRIu64 " %-18s %-10s %-14s %-14s %-16s %8.6f  %s\n",
               job->id,
               edgexpu_executor_job_type_name(job->type),
               edgexpu_executor_job_status_name(job->status),
               job->device,
               job->backend,
               job->scheduler_policy,
               elapsed_seconds,
               job->scheduler_reason);
    }
}

/* 一次 generate 并打印 JSON telemetry / executor_trace。
 * prefill/decode tok/s 用分项时间，不用整段 elapsed。 */
static int command_benchmark(const char *manifest_path, const char *prompt, int max_tokens) {
    edgexpu_runtime runtime;
    edgexpu_generation_request request;
    edgexpu_generation_result result;
    char escaped_text[EDGEXPU_TEXT_PROMPT * 2];
    char error[256] = {0};
    double prefill_tps;
    double decode_tps;

    if (max_tokens < 1) {
        max_tokens = 8;
    }

    edgexpu_runtime_init(&runtime);
    if (!edgexpu_runtime_load_model(&runtime, manifest_path, error, sizeof(error))) {
        fprintf(stderr, "模型加载失败：%s\n", error);
        edgexpu_runtime_shutdown(&runtime);
        return 1;
    }

    memset(&request, 0, sizeof(request));
    request.prompt = prompt;
    request.max_tokens = max_tokens;
    request.temperature = 0.0f;
    request.top_p = 1.0f;
    request.cpu_path = EDGEXPU_CPU_PATH_AUTO;

    if (!edgexpu_runtime_generate(&runtime, &request, &result, error, sizeof(error))) {
        fprintf(stderr, "benchmark 执行失败：%s\n", error);
        edgexpu_runtime_shutdown(&runtime);
        return 1;
    }

    prefill_tps = result.telemetry.prefill_seconds > 0.0
        ? (double)result.prompt_tokens_approx / result.telemetry.prefill_seconds
        : 0.0;
    decode_tps = result.telemetry.decode_seconds > 0.0
        ? (double)result.completion_tokens_approx / result.telemetry.decode_seconds
        : 0.0;

    json_escape(result.text, escaped_text, sizeof(escaped_text));
    printf("{\n");
    printf("  \"backend\": \"%s\",\n", result.backend);
    printf("  \"finish_reason\": \"%s\",\n", result.finish_reason[0] != '\0' ? result.finish_reason : "stop");
    printf("  \"max_tokens\": %d,\n", max_tokens);
    printf("  \"elapsed_seconds\": %.6f,\n", result.elapsed_seconds);
    printf("  \"prompt_tokens_approx\": %d,\n", result.prompt_tokens_approx);
    printf("  \"completion_tokens_approx\": %d,\n", result.completion_tokens_approx);
    printf("  \"prefill_tokens_per_second_approx\": %.6f,\n", prefill_tps);
    printf("  \"decode_tokens_per_second_approx\": %.6f,\n", decode_tps);
    printf("  \"stage_trace\": [\n");
    printf("    {\"stage\": \"prefill\", \"tokens_approx\": %d},\n", result.prompt_tokens_approx);
    printf("    {\"stage\": \"decode\", \"tokens_approx\": %d}\n", result.completion_tokens_approx);
    printf("  ],\n");
    print_backend_telemetry_json(&result.telemetry);
    printf(",\n");
    print_executor_queue_summary_json(&runtime.executor);
    printf(",\n");
    print_executor_trace_json(&runtime.executor);
    printf(",\n");
    printf("  \"text\": \"%s\"\n", escaped_text);
    printf("}\n");
    edgexpu_runtime_shutdown(&runtime);
    return 0;
}

static int command_generate(const char *manifest_path, const char *prompt, int max_tokens) {
    edgexpu_runtime runtime;
    edgexpu_generation_request request;
    edgexpu_generation_result result;
    char error[256] = {0};
    size_t text_len;

    if (max_tokens < 1) {
        max_tokens = 32;
    }

    edgexpu_runtime_init(&runtime);
    if (!edgexpu_runtime_load_model(&runtime, manifest_path, error, sizeof(error))) {
        fprintf(stderr, "模型加载失败：%s\n", error);
        edgexpu_runtime_shutdown(&runtime);
        return 1;
    }

    memset(&request, 0, sizeof(request));
    request.prompt = prompt;
    request.max_tokens = max_tokens;
    request.temperature = 0.0f;
    request.top_p = 1.0f;
    request.cpu_path = EDGEXPU_CPU_PATH_AUTO;

    if (!edgexpu_runtime_generate(&runtime, &request, &result, error, sizeof(error))) {
        fprintf(stderr, "generate 失败：%s\n", error);
        edgexpu_runtime_shutdown(&runtime);
        return 1;
    }

    fputs(result.text, stdout);
    text_len = strlen(result.text);
    if (text_len == 0 || result.text[text_len - 1] != '\n') {
        fputc('\n', stdout);
    }
    fprintf(
        stderr,
        "finish_reason=%s backend=%s prompt_tokens=%d completion_tokens=%d\n",
        result.finish_reason[0] != '\0' ? result.finish_reason : "stop",
        result.backend,
        result.prompt_tokens_approx,
        result.completion_tokens_approx
    );
    edgexpu_runtime_shutdown(&runtime);
    return 0;
}

static void print_compare_leg(const char *name, const edgexpu_generation_result *result) {
    char escaped_text[EDGEXPU_TEXT_PROMPT * 2];
    double total;
    double decode_tps;

    json_escape(result->text, escaped_text, sizeof(escaped_text));
    total = result->telemetry.total_seconds > 0.0 ? result->telemetry.total_seconds : result->elapsed_seconds;
    decode_tps = result->telemetry.decode_seconds > 0.0
        ? (double)result->completion_tokens_approx / result->telemetry.decode_seconds
        : 0.0;
    printf("  \"%s\": {\n", name);
    printf("    \"backend\": \"%s\",\n", result->backend);
    printf("    \"elapsed_seconds\": %.6f,\n", result->elapsed_seconds);
    printf("    \"total_seconds\": %.6f,\n", total);
    printf("    \"prefill_seconds\": %.6f,\n", result->telemetry.prefill_seconds);
    printf("    \"decode_seconds\": %.6f,\n", result->telemetry.decode_seconds);
    printf("    \"prompt_tokens_approx\": %d,\n", result->prompt_tokens_approx);
    printf("    \"completion_tokens_approx\": %d,\n", result->completion_tokens_approx);
    printf("    \"decode_tokens_per_second_approx\": %.6f,\n", decode_tps);
    printf("    \"text\": \"%s\"\n", escaped_text);
    printf("  }");
}

/* 同模型对比 native CPU fallback 与 llama bootstrap。 */
static int command_compare(const char *manifest_path, const char *prompt) {
    edgexpu_runtime runtime;
    edgexpu_generation_request request;
    edgexpu_generation_result native_result;
    edgexpu_generation_result llama_result;
    char error[256] = {0};
    double native_total;
    double llama_total;

    edgexpu_runtime_init(&runtime);
    if (!edgexpu_runtime_load_model(&runtime, manifest_path, error, sizeof(error))) {
        fprintf(stderr, "模型加载失败：%s\n", error);
        edgexpu_runtime_shutdown(&runtime);
        return 1;
    }

    memset(&request, 0, sizeof(request));
    request.prompt = prompt;
    request.max_tokens = 8;
    request.temperature = 0.0f;
    request.cpu_path = EDGEXPU_CPU_PATH_NATIVE;
    if (!edgexpu_runtime_generate(&runtime, &request, &native_result, error, sizeof(error))) {
        fprintf(stderr, "native CPU fallback 对比失败：%s\n", error);
        edgexpu_runtime_shutdown(&runtime);
        return 1;
    }

    edgexpu_executor_drop_terminal(&runtime.executor);
    memset(&request, 0, sizeof(request));
    request.prompt = prompt;
    request.max_tokens = 8;
    request.temperature = 0.0f;
    request.cpu_path = EDGEXPU_CPU_PATH_LLAMA_BOOTSTRAP;
    if (!edgexpu_runtime_generate(&runtime, &request, &llama_result, error, sizeof(error))) {
        fprintf(stderr, "llama bootstrap 对比失败：%s\n", error);
        edgexpu_runtime_shutdown(&runtime);
        return 1;
    }

    native_total = native_result.telemetry.total_seconds > 0.0
        ? native_result.telemetry.total_seconds
        : native_result.elapsed_seconds;
    llama_total = llama_result.telemetry.total_seconds > 0.0
        ? llama_result.telemetry.total_seconds
        : llama_result.elapsed_seconds;

    printf("{\n");
    printf("  \"model_id\": \"%s\",\n", runtime.manifest.model_id);
    printf("  \"max_tokens\": 8,\n");
    printf("  \"temperature\": 0.0,\n");
    print_compare_leg("native", &native_result);
    printf(",\n");
    print_compare_leg("llama_bootstrap", &llama_result);
    printf(",\n");
    printf("  \"comparison\": {\n");
    printf("    \"same_model\": true,\n");
    printf("    \"native_is_cpu_fallback\": %s,\n",
           strcmp(native_result.backend, "cpu.native") == 0 ? "true" : "false");
    printf("    \"llama_is_bootstrap\": %s,\n",
           strcmp(llama_result.backend, "cpu.baseline") == 0 ? "true" : "false");
    printf("    \"native_total_seconds\": %.6f,\n", native_total);
    printf("    \"llama_total_seconds\": %.6f,\n", llama_total);
    printf("    \"native_faster\": %s,\n", native_total < llama_total ? "true" : "false");
    printf("    \"completion_tokens_native\": %d,\n", native_result.completion_tokens_approx);
    printf("    \"completion_tokens_llama\": %d,\n", llama_result.completion_tokens_approx);
    printf("    \"texts_required_to_match\": false,\n");
    printf("    \"notes\": \"Same GGUF greedy check is edgexpu dump-logits vs llama-cli --temp 0 --no-conversation. compare still shells out llama with chat wrapping.\"\n");
    printf("  }\n");
    printf("}\n");

    edgexpu_runtime_shutdown(&runtime);
    return 0;
}

/* 人类可读的 executor / scheduler / telemetry 表。 */
static int command_trace(const char *manifest_path, const char *prompt) {
    edgexpu_runtime runtime;
    edgexpu_generation_request request;
    edgexpu_generation_result result;
    edgexpu_schedule_decision next_decode;
    edgexpu_schedule_native_ready native_ready;
    char error[256] = {0};

    edgexpu_runtime_init(&runtime);
    if (!edgexpu_runtime_load_model(&runtime, manifest_path, error, sizeof(error))) {
        fprintf(stderr, "模型加载失败：%s\n", error);
        edgexpu_runtime_shutdown(&runtime);
        return 1;
    }

    memset(&request, 0, sizeof(request));
    request.prompt = prompt;
    request.max_tokens = 4;
    request.temperature = 0.0f;
    request.cpu_path = EDGEXPU_CPU_PATH_AUTO;

    if (!edgexpu_runtime_generate(&runtime, &request, &result, error, sizeof(error))) {
        fprintf(stderr, "trace 执行失败：%s\n", error);
        edgexpu_runtime_shutdown(&runtime);
        return 1;
    }

    printf("EdgeXPU trace\n");
    printf("Model: %s\n", runtime.manifest.model_id);
    printf("Backend: %s\n", result.backend);
    printf("Elapsed: %.6f seconds\n", result.elapsed_seconds);
    printf("Prompt tokens approx: %d\n", result.prompt_tokens_approx);
    printf("Completion tokens approx: %d\n", result.completion_tokens_approx);
    print_backend_telemetry_table(&result.telemetry);
    print_executor_trace_table(&runtime.executor);

    memset(&native_ready, 0, sizeof(native_ready));
    if (runtime.native.loaded) {
        native_ready.loader = 1;
        native_ready.tokenizer = runtime.native.tokenizer.ready;
        native_ready.kernel = 1;
        native_ready.kv = runtime.native.kv.k != NULL;
    }

    if (edgexpu_scheduler_plan_job_with_context(
            &runtime.manifest,
            runtime.has_device_profile ? &runtime.device_profile : NULL,
            runtime.has_last_telemetry ? &runtime.last_telemetry : NULL,
            &native_ready,
            EDGEXPU_EXECUTOR_JOB_DECODE_STEP,
            &next_decode,
            error,
            sizeof(error))) {
        printf("\nNext Decode Plan\n");
        printf("----------------\n");
        printf("backend=%s device=%s policy=%s\n",
               next_decode.backend,
               next_decode.device,
               next_decode.policy);
        printf("reason=%s\n", next_decode.reason);
        if (next_decode.fallback_reason[0] != '\0') {
            printf("fallback_reason=%s\n", next_decode.fallback_reason);
        }
    }

    printf("\nGenerated Text\n");
    printf("--------------\n");
    printf("%s\n", result.text);
    edgexpu_runtime_shutdown(&runtime);
    return 0;
}

static int selftest_noop_callback(
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

static int command_executor_selftest(void) {
    edgexpu_executor executor;
    edgexpu_executor_queue_summary summary;
    uint64_t job_id = 0;
    char error[256] = {0};

    edgexpu_executor_init(&executor);

    if (edgexpu_executor_run_next(&executor, error, sizeof(error))) {
        fprintf(stderr, "executor selftest failed: empty run_next unexpectedly succeeded\n");
        return 1;
    }

    if (!edgexpu_executor_submit_runnable(
            &executor,
            EDGEXPU_EXECUTOR_JOB_PREPARE_PROMPT,
            "selftest",
            "cpu.runtime",
            "cpu",
            "selftest scheduler reason",
            "host_cpu",
            "",
            selftest_noop_callback,
            NULL,
            &job_id,
            error,
            sizeof(error))) {
        fprintf(stderr, "executor selftest failed: submit failed: %s\n", error);
        return 1;
    }

    if (!edgexpu_executor_has_pending(&executor)) {
        fprintf(stderr, "executor selftest failed: pending job not visible\n");
        return 1;
    }

    if (!edgexpu_executor_run_next(&executor, error, sizeof(error))) {
        fprintf(stderr, "executor selftest failed: run_next failed: %s\n", error);
        return 1;
    }

    if (edgexpu_executor_run_job(&executor, job_id, NULL, NULL, error, sizeof(error))) {
        fprintf(stderr, "executor selftest failed: completed job unexpectedly ran twice\n");
        return 1;
    }

    edgexpu_executor_get_queue_summary(&executor, &summary);
    if (summary.total != 1 || summary.pending != 0 || summary.running != 0 ||
        summary.completed != 1 || summary.failed != 0) {
        fprintf(stderr, "executor selftest failed: queue summary mismatch\n");
        return 1;
    }

    edgexpu_executor_drop_terminal(&executor);
    if (edgexpu_executor_job_count(&executor) != 0 ||
        edgexpu_executor_remaining_capacity(&executor) != EDGEXPU_EXECUTOR_MAX_JOBS) {
        fprintf(stderr, "executor selftest failed: drop_terminal did not reclaim completed jobs\n");
        return 1;
    }

    printf("executor selftest passed\n");
    return 0;
}

static int command_scheduler_selftest(void) {
    edgexpu_model_manifest manifest;
    edgexpu_backend_telemetry telemetry;
    edgexpu_schedule_decision decision;
    char error[256] = {0};

    memset(&manifest, 0, sizeof(manifest));
    memset(&telemetry, 0, sizeof(telemetry));
    snprintf(manifest.primary_artifact.backend, sizeof(manifest.primary_artifact.backend), "cpu.baseline");
    snprintf(telemetry.backend, sizeof(telemetry.backend), "cpu.baseline");
    telemetry.decode_seconds = 2.0;
    telemetry.completion_tokens_approx = 4;

    if (!edgexpu_scheduler_plan_job_with_context(
            &manifest,
            NULL,
            NULL,
            NULL,
            EDGEXPU_EXECUTOR_JOB_DECODE_STEP,
            &decision,
            error,
            sizeof(error))) {
        fprintf(stderr, "scheduler selftest failed: decode plan without telemetry failed: %s\n", error);
        return 1;
    }

    if (strcmp(decision.backend, "cpu.baseline") != 0 ||
        strcmp(decision.policy, "stage_execution") != 0) {
        fprintf(stderr, "scheduler selftest failed: baseline decode policy mismatch\n");
        return 1;
    }

    if (!edgexpu_scheduler_plan_job_with_context(
            &manifest,
            NULL,
            &telemetry,
            NULL,
            EDGEXPU_EXECUTOR_JOB_DECODE_STEP,
            &decision,
            error,
            sizeof(error))) {
        fprintf(stderr, "scheduler selftest failed: decode plan failed: %s\n", error);
        return 1;
    }

    if (strcmp(decision.backend, "cpu.baseline") != 0 ||
        strcmp(decision.device, "cpu") != 0 ||
        strcmp(decision.policy, "telemetry_keep_cpu") != 0 ||
        strstr(decision.fallback_reason, "previous cpu.baseline decode throughput") == NULL) {
        fprintf(stderr, "scheduler selftest failed: telemetry hint missing from decode plan\n");
        return 1;
    }

    if (!edgexpu_scheduler_plan_job_with_context(
            &manifest,
            NULL,
            &telemetry,
            NULL,
            EDGEXPU_EXECUTOR_JOB_PREFETCH_WEIGHTS,
            &decision,
            error,
            sizeof(error))) {
        fprintf(stderr, "scheduler selftest failed: prefetch plan failed: %s\n", error);
        return 1;
    }

    if (strcmp(decision.backend, "flash.manager") != 0 ||
        strcmp(decision.device, "flash") != 0 ||
        strcmp(decision.policy, "flash_pipeline") != 0) {
        fprintf(stderr, "scheduler selftest failed: flash policy mismatch\n");
        return 1;
    }

    {
        edgexpu_schedule_native_ready native_ready;
        memset(&native_ready, 0, sizeof(native_ready));
        native_ready.tokenizer = 1;
        native_ready.kv = 1;
        if (!edgexpu_scheduler_plan_job_with_context(
                &manifest,
                NULL,
                NULL,
                &native_ready,
                EDGEXPU_EXECUTOR_JOB_TOKENIZE,
                &decision,
                error,
                sizeof(error))) {
            fprintf(stderr, "scheduler selftest failed: native tokenize plan failed: %s\n", error);
            return 1;
        }
        if (strcmp(decision.backend, "cpu.native") != 0 ||
            strcmp(decision.policy, "native_tokenizer") != 0) {
            fprintf(stderr, "scheduler selftest failed: native tokenizer policy mismatch\n");
            return 1;
        }
        native_ready.kernel = 1;
        if (!edgexpu_scheduler_plan_job_with_context(
                &manifest,
                NULL,
                NULL,
                &native_ready,
                EDGEXPU_EXECUTOR_JOB_PREFILL,
                &decision,
                error,
                sizeof(error))) {
            fprintf(stderr, "scheduler selftest failed: native prefill plan failed: %s\n", error);
            return 1;
        }
        if (strcmp(decision.backend, "cpu.native") != 0 ||
            strcmp(decision.policy, "native_prefill") != 0) {
            fprintf(stderr, "scheduler selftest failed: native prefill policy mismatch backend=%s policy=%s\n",
                    decision.backend,
                    decision.policy);
            return 1;
        }
        snprintf(telemetry.backend, sizeof(telemetry.backend), "cpu.native");
        if (!edgexpu_scheduler_plan_job_with_context(
                &manifest,
                NULL,
                &telemetry,
                &native_ready,
                EDGEXPU_EXECUTOR_JOB_DECODE_STEP,
                &decision,
                error,
                sizeof(error))) {
            fprintf(stderr, "scheduler selftest failed: native decode telemetry plan failed: %s\n", error);
            return 1;
        }
        if (strcmp(decision.backend, "cpu.native") != 0 ||
            strcmp(decision.policy, "telemetry_keep_native") != 0) {
            fprintf(stderr, "scheduler selftest failed: expected telemetry_keep_native got backend=%s policy=%s\n",
                    decision.backend,
                    decision.policy);
            return 1;
        }
    }

    printf("scheduler selftest passed\n");
    return 0;
}

static int command_inspect_gguf(const char *gguf_path) {
    edgexpu_gguf_info info;
    edgexpu_tokenizer tokenizer;
    edgexpu_arch_adapter adapter;
    char error[256] = {0};

    memset(&info, 0, sizeof(info));
    edgexpu_tokenizer_init(&tokenizer);
    if (!edgexpu_gguf_load(gguf_path, &info, &tokenizer, error, sizeof(error))) {
        fprintf(stderr, "GGUF 加载失败：%s\n", error);
        edgexpu_gguf_info_free(&info);
        edgexpu_tokenizer_free(&tokenizer);
        return 1;
    }

    printf("architecture=%s\n", info.architecture);
    printf("name=%s\n", info.name);
    printf("file_size=%llu\n", (unsigned long long)info.file_size);
    printf("block_count=%u\n", info.block_count);
    printf("context_length=%u\n", info.context_length);
    printf("embedding_length=%u\n", info.embedding_length);
    printf("feed_forward_length=%u\n", info.feed_forward_length);
    printf("head_count=%u\n", info.head_count);
    printf("head_count_kv=%u\n", info.head_count_kv);
    printf("head_dim=%d\n", edgexpu_gguf_head_dim(&info));
    printf("tensor_count=%llu\n", (unsigned long long)info.tensor_count);
    printf("tokenizer_model=%s\n", info.tokenizer_model);
    printf("tokenizer_pre=%s\n", info.tokenizer_pre);
    printf("vocab_size=%u\n", tokenizer.vocab_size);
    printf("n_merges=%u\n", tokenizer.n_merges);
    printf("eos=%u pad=%u\n", info.eos_token_id, info.pad_token_id);

    if (!edgexpu_arch_from_gguf(&info, &adapter, error, sizeof(error))) {
        printf("native_adapter=unsupported\n");
        printf("native_adapter_error=%s\n", error);
        edgexpu_gguf_info_free(&info);
        edgexpu_tokenizer_free(&tokenizer);
        return 0;
    }
    printf("adapter=%s qkv_bias=%d rope=%s ffn=%s tokenizer=%s\n",
           adapter.name,
           adapter.has_qkv_bias,
           edgexpu_rope_type_name(adapter.rope),
           edgexpu_ffn_type_name(adapter.ffn),
           edgexpu_tokenizer_kind_name(adapter.tokenizer));
    edgexpu_gguf_info_free(&info);
    edgexpu_tokenizer_free(&tokenizer);
    return 0;
}

/* 对原文 encode/decode，不套 chat template，便于核对 vocab。 */
static int command_tokenize(const char *path, const char *text) {
    edgexpu_model_manifest manifest;
    edgexpu_gguf_info info;
    edgexpu_tokenizer tokenizer;
    char error[256] = {0};
    char decoded[EDGEXPU_TEXT_LARGE];
    const char *gguf_path = path;
    const char *model_id = "";
    uint32_t ids[4096];
    int id_count = 0;
    int i;

    memset(&info, 0, sizeof(info));
    edgexpu_tokenizer_init(&tokenizer);
    if (!edgexpu_path_is_gguf(path)) {
        if (!edgexpu_manifest_load(path, &manifest, error, sizeof(error))) {
            fprintf(stderr, "manifest 读取失败：%s\n", error);
            return 1;
        }
        gguf_path = manifest.primary_artifact.path;
        model_id = manifest.model_id;
    } else {
        model_id = path;
    }

    if (!edgexpu_gguf_load(gguf_path, &info, &tokenizer, error, sizeof(error))) {
        fprintf(stderr, "tokenizer 加载失败：%s\n", error);
        edgexpu_gguf_info_free(&info);
        edgexpu_tokenizer_free(&tokenizer);
        return 1;
    }
    if (!edgexpu_tokenizer_encode(&tokenizer, text, ids, 4096, &id_count, error, sizeof(error))) {
        fprintf(stderr, "tokenize 失败：%s\n", error);
        edgexpu_gguf_info_free(&info);
        edgexpu_tokenizer_free(&tokenizer);
        return 1;
    }

    printf("model_id=%s\n", model_id);
    printf("vocab_size=%u\n", tokenizer.vocab_size);
    printf("token_count=%d\n", id_count);
    printf("token_ids=");
    for (i = 0; i < id_count; i++) {
        printf("%s%u", i == 0 ? "" : ",", ids[i]);
    }
    printf("\n");
    edgexpu_tokenizer_decode(&tokenizer, ids, id_count, decoded, sizeof(decoded));
    printf("decoded=%s\n", decoded);
    edgexpu_gguf_info_free(&info);
    edgexpu_tokenizer_free(&tokenizer);
    return 0;
}

static int path_is_json(const char *path) {
    size_t len;
    if (path == NULL) {
        return 0;
    }
    len = strlen(path);
    return len >= 5 && strcmp(path + len - 5, ".json") == 0;
}

/* 原文 dump：token ids、layer0 算子、last hidden、lm_head top-k、greedy 前 N 个 token。 */
static int command_dump_logits(const char *path, const char *prompt, int greedy_n) {
    edgexpu_model_manifest manifest;
    edgexpu_native_session session;
    char error[256] = {0};
    const char *gguf_path = path;

    if (path_is_json(path)) {
        if (!edgexpu_manifest_load(path, &manifest, error, sizeof(error))) {
            fprintf(stderr, "manifest 读取失败：%s\n", error);
            return 1;
        }
        gguf_path = manifest.primary_artifact.path;
    }

    edgexpu_native_init(&session);
    if (!edgexpu_native_load(&session, gguf_path, error, sizeof(error))) {
        fprintf(stderr, "dump-logits 加载失败：%s\n", error);
        edgexpu_native_free(&session);
        return 1;
    }
    if (!edgexpu_native_dump_logits(&session, prompt, greedy_n, 8, error, sizeof(error))) {
        fprintf(stderr, "dump-logits 失败：%s\n", error);
        edgexpu_native_free(&session);
        return 1;
    }
    edgexpu_native_free(&session);
    return 0;
}

static int command_serve(const char *manifest_path, const char *port_text) {
    int port = 8000;

    if (port_text != NULL) {
        port = atoi(port_text);
    }
    return edgexpu_server_run(manifest_path, port);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    if (strcmp(argv[1], "capabilities") == 0) {
        return command_capabilities();
    }

    if (strcmp(argv[1], "inspect-manifest") == 0) {
        if (argc < 3) {
            print_usage();
            return 1;
        }
        return command_inspect_manifest(argv[2]);
    }

    if (strcmp(argv[1], "inspect-gguf") == 0) {
        if (argc < 3) {
            print_usage();
            return 1;
        }
        return command_inspect_gguf(argv[2]);
    }

    if (strcmp(argv[1], "tokenize") == 0) {
        if (argc < 4) {
            print_usage();
            return 1;
        }
        return command_tokenize(argv[2], argv[3]);
    }

    if (strcmp(argv[1], "dump-logits") == 0) {
        int greedy_n = 8;
        if (argc < 4) {
            print_usage();
            return 1;
        }
        if (argc >= 5) {
            greedy_n = atoi(argv[4]);
        }
        return command_dump_logits(argv[2], argv[3], greedy_n);
    }

    if (strcmp(argv[1], "generate") == 0) {
        int max_tokens = 32;
        if (argc < 4) {
            print_usage();
            return 1;
        }
        if (argc >= 5) {
            max_tokens = atoi(argv[4]);
        }
        return command_generate(argv[2], argv[3], max_tokens);
    }

    if (strcmp(argv[1], "benchmark") == 0) {
        int max_tokens = 8;
        if (argc < 4) {
            print_usage();
            return 1;
        }
        if (argc >= 5) {
            max_tokens = atoi(argv[4]);
        }
        return command_benchmark(argv[2], argv[3], max_tokens);
    }

    if (strcmp(argv[1], "compare") == 0) {
        if (argc < 4) {
            print_usage();
            return 1;
        }
        return command_compare(argv[2], argv[3]);
    }

    if (strcmp(argv[1], "trace") == 0) {
        if (argc < 4) {
            print_usage();
            return 1;
        }
        return command_trace(argv[2], argv[3]);
    }

    if (strcmp(argv[1], "executor-selftest") == 0) {
        return command_executor_selftest();
    }

    if (strcmp(argv[1], "scheduler-selftest") == 0) {
        return command_scheduler_selftest();
    }

    if (strcmp(argv[1], "native-selftest") == 0) {
        return edgexpu_native_selftest(argc >= 3 ? argv[2] : NULL);
    }

    if (strcmp(argv[1], "serve") == 0) {
        if (argc < 3) {
            print_usage();
            return 1;
        }
        return command_serve(argv[2], argc >= 4 ? argv[3] : NULL);
    }

    print_usage();
    return 1;
}
