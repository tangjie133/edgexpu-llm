#include "edgexpu/manifest.h"
#include "edgexpu/profiler.h"
#include "edgexpu/runtime.h"
#include "edgexpu/server.h"

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
    printf("  \"text\": \"%s\"\n", escaped_text);
    printf("}\n");

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
