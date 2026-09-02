#include "edgexpu/manifest.h"
#include "edgexpu/profiler.h"
#include "edgexpu/runtime.h"
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
    printf("  edgexpu benchmark <manifest.json> <prompt>\n");
    printf("  edgexpu trace <manifest.json> <prompt>\n");
    printf("  edgexpu executor-selftest\n");
    printf("  edgexpu serve <manifest.json> [port]\n");
}

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

static int command_benchmark(const char *manifest_path, const char *prompt) {
    edgexpu_runtime runtime;
    edgexpu_generation_request request;
    edgexpu_generation_result result;
    char escaped_text[EDGEXPU_TEXT_LARGE * 2];
    char error[256] = {0};

    edgexpu_runtime_init(&runtime);
    if (!edgexpu_runtime_load_model(&runtime, manifest_path, error, sizeof(error))) {
        fprintf(stderr, "模型加载失败：%s\n", error);
        return 1;
    }

    request.prompt = prompt;
    request.max_tokens = 64;
    request.temperature = 0.0f;

    if (!edgexpu_runtime_generate(&runtime, &request, &result, error, sizeof(error))) {
        fprintf(stderr, "benchmark 执行失败：%s\n", error);
        return 1;
    }

    json_escape(result.text, escaped_text, sizeof(escaped_text));
    printf("{\n");
    printf("  \"backend\": \"%s\",\n", result.backend);
    printf("  \"elapsed_seconds\": %.6f,\n", result.elapsed_seconds);
    printf("  \"prompt_tokens_approx\": %d,\n", result.prompt_tokens_approx);
    printf("  \"completion_tokens_approx\": %d,\n", result.completion_tokens_approx);
    printf("  \"prefill_tokens_per_second_approx\": %.6f,\n",
           result.elapsed_seconds > 0.0 ? result.prompt_tokens_approx / result.elapsed_seconds : 0.0);
    printf("  \"decode_tokens_per_second_approx\": %.6f,\n",
           result.elapsed_seconds > 0.0 ? result.completion_tokens_approx / result.elapsed_seconds : 0.0);
    printf("  \"stage_trace\": [\n");
    printf("    {\"stage\": \"prefill\", \"tokens_approx\": %d},\n", result.prompt_tokens_approx);
    printf("    {\"stage\": \"decode\", \"tokens_approx\": %d}\n", result.completion_tokens_approx);
    printf("  ],\n");
    print_executor_queue_summary_json(&runtime.executor);
    printf(",\n");
    print_executor_trace_json(&runtime.executor);
    printf(",\n");
    printf("  \"text\": \"%s\"\n", escaped_text);
    printf("}\n");

    return 0;
}

static int command_trace(const char *manifest_path, const char *prompt) {
    edgexpu_runtime runtime;
    edgexpu_generation_request request;
    edgexpu_generation_result result;
    char error[256] = {0};

    edgexpu_runtime_init(&runtime);
    if (!edgexpu_runtime_load_model(&runtime, manifest_path, error, sizeof(error))) {
        fprintf(stderr, "模型加载失败：%s\n", error);
        return 1;
    }

    request.prompt = prompt;
    request.max_tokens = 32;
    request.temperature = 0.0f;

    if (!edgexpu_runtime_generate(&runtime, &request, &result, error, sizeof(error))) {
        fprintf(stderr, "trace 执行失败：%s\n", error);
        return 1;
    }

    printf("EdgeXPU trace\n");
    printf("Model: %s\n", runtime.manifest.model_id);
    printf("Backend: %s\n", result.backend);
    printf("Elapsed: %.6f seconds\n", result.elapsed_seconds);
    printf("Prompt tokens approx: %d\n", result.prompt_tokens_approx);
    printf("Completion tokens approx: %d\n", result.completion_tokens_approx);
    print_executor_trace_table(&runtime.executor);
    printf("\nGenerated Text\n");
    printf("--------------\n");
    printf("%s\n", result.text);

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

    printf("executor selftest passed\n");
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

    if (strcmp(argv[1], "benchmark") == 0) {
        if (argc < 4) {
            print_usage();
            return 1;
        }
        return command_benchmark(argv[2], argv[3]);
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
