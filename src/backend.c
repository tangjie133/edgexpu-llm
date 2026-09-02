#include "edgexpu/backend.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#define EDGEXPU_POPEN _popen
#define EDGEXPU_PCLOSE _pclose
#else
#define EDGEXPU_POPEN popen
#define EDGEXPU_PCLOSE pclose
#endif

/* 初版 backend 先保存当前加载的 manifest。
 * 后续应替换为真正的模型句柄、权重驻留状态、KV cache 状态和 backend telemetry。
 */
static edgexpu_model_manifest g_loaded_manifest;
static int g_loaded = 0;

static void set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
    }
}

static int command_exists(const char *command) {
    char check_command[256];
#if defined(_WIN32)
    char line[512];
    FILE *pipe;
#endif

    if (command == NULL || command[0] == '\0') {
        return 0;
    }

#if defined(_WIN32)
    snprintf(check_command, sizeof(check_command), "where %s 2>nul", command);
    pipe = EDGEXPU_POPEN(check_command, "r");
    if (pipe == NULL) {
        return 0;
    }
    while (fgets(line, sizeof(line), pipe) != NULL) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len >= 4 &&
            (_stricmp(line + len - 4, ".exe") == 0 ||
             _stricmp(line + len - 4, ".bat") == 0 ||
             _stricmp(line + len - 4, ".cmd") == 0)) {
            EDGEXPU_PCLOSE(pipe);
            return 1;
        }
    }
    EDGEXPU_PCLOSE(pipe);
    return 0;
#else
    snprintf(check_command, sizeof(check_command), "command -v %s >/dev/null 2>&1", command);
    return system(check_command) == 0;
#endif
}

static int file_exists(const char *path) {
    FILE *file;

    if (path == NULL || path[0] == '\0') {
        return 0;
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    fclose(file);
    return 1;
}

static int command_is_invocable(const char *command) {
    if (command == NULL || command[0] == '\0') {
        return 0;
    }

    if (strpbrk(command, "\\/") != NULL) {
        return file_exists(command);
    }

    return command_exists(command);
}

static const char *select_command(const edgexpu_model_manifest *manifest) {
    if (manifest != NULL && manifest->primary_artifact.command[0] != '\0') {
        return manifest->primary_artifact.command;
    }
    if (command_exists("powerinfer")) {
        return "powerinfer";
    }
    if (command_exists("llama-cli")) {
        return "llama-cli";
    }
    if (command_exists("main")) {
        return "main";
    }
    return NULL;
}

static void shell_quote(const char *input, char *output, size_t output_size) {
    size_t used = 0;

    if (output_size < 3) {
        if (output_size > 0) {
            output[0] = '\0';
        }
        return;
    }

    output[used++] = '"';
    while (input != NULL && *input != '\0' && used + 2 < output_size) {
        if (*input == '"') {
            output[used++] = '\\';
            output[used++] = '"';
        } else {
            output[used++] = *input;
        }
        input++;
    }
    if (used + 1 < output_size) {
        output[used++] = '"';
    }
    output[used] = '\0';
}

static void shell_command_name(const char *input, char *output, size_t output_size) {
    if (input == NULL || output_size == 0) {
        return;
    }

    if (strpbrk(input, " \t\\/") == NULL) {
        snprintf(output, output_size, "%s", input);
        return;
    }

    shell_quote(input, output, output_size);
}

static int count_tokens_approx(const char *text) {
    int count = 0;
    int in_token = 0;

    if (text == NULL) {
        return 0;
    }

    while (*text != '\0') {
        if (isspace((unsigned char)*text)) {
            in_token = 0;
        } else if (!in_token) {
            count++;
            in_token = 1;
        }
        text++;
    }

    return count;
}

static int cpu_baseline_available(void) {
    return command_exists("powerinfer") || command_exists("llama-cli") || command_exists("main");
}

static int cpu_baseline_load(
    const edgexpu_model_manifest *manifest,
    char *error,
    size_t error_size
) {
    const char *command;

    if (manifest == NULL) {
        set_error(error, error_size, "backend 加载失败：manifest 为空");
        return 0;
    }

    command = select_command(manifest);
    if (command == NULL || !command_is_invocable(command)) {
        set_error(error, error_size, "未找到 powerinfer、llama-cli 或 manifest 指定的本地二进制");
        return 0;
    }

    g_loaded_manifest = *manifest;
    g_loaded = 1;
    return 1;
}

static int cpu_baseline_generate(
    const edgexpu_generation_request *request,
    edgexpu_generation_result *result,
    char *error,
    size_t error_size
) {
    const char *command;
    char shell_command[2048];
    char quoted_command[EDGEXPU_TEXT_MEDIUM + 4];
    char quoted_model[EDGEXPU_TEXT_LARGE + 4];
    char quoted_prompt[EDGEXPU_TEXT_LARGE + 4];
    char chunk[256];
    FILE *pipe;
    clock_t started;
    size_t used = 0;
    int close_code;

    if (!g_loaded) {
        set_error(error, error_size, "backend 尚未加载模型");
        return 0;
    }
    if (request == NULL || request->prompt == NULL || result == NULL) {
        set_error(error, error_size, "生成请求参数为空");
        return 0;
    }

    command = select_command(&g_loaded_manifest);
    if (command == NULL || !command_is_invocable(command)) {
        set_error(error, error_size, "未找到可执行的 CPU baseline backend");
        return 0;
    }

    memset(result, 0, sizeof(*result));
    snprintf(result->backend, sizeof(result->backend), "cpu.baseline");
    result->prompt_tokens_approx = count_tokens_approx(request->prompt);

    /* PowerInfer 和 llama.cpp 的 CLI 形态接近：-m 指模型，-p 指 prompt，-n 指输出 token。 */
    shell_command_name(command, quoted_command, sizeof(quoted_command));
    shell_quote(g_loaded_manifest.primary_artifact.path, quoted_model, sizeof(quoted_model));
    shell_quote(request->prompt, quoted_prompt, sizeof(quoted_prompt));

    snprintf(
        shell_command,
        sizeof(shell_command),
        "%s -m %s -p %s -n %d --temp %.3f",
        quoted_command,
        quoted_model,
        quoted_prompt,
        request->max_tokens,
        request->temperature
    );

    started = clock();
    pipe = EDGEXPU_POPEN(shell_command, "r");
    if (pipe == NULL) {
        set_error(error, error_size, "无法启动 CPU baseline backend");
        return 0;
    }

    while (fgets(chunk, sizeof(chunk), pipe) != NULL) {
        size_t chunk_len = strlen(chunk);
        size_t remaining = sizeof(result->text) - used - 1;
        if (remaining == 0) {
            break;
        }
        if (chunk_len > remaining) {
            chunk_len = remaining;
        }
        memcpy(result->text + used, chunk, chunk_len);
        used += chunk_len;
        result->text[used] = '\0';
    }

    close_code = EDGEXPU_PCLOSE(pipe);
    result->elapsed_seconds = (double)(clock() - started) / CLOCKS_PER_SEC;
    result->completion_tokens_approx = count_tokens_approx(result->text);

    if (close_code != 0) {
        set_error(error, error_size, "CPU baseline backend 执行失败");
        return 0;
    }

    return 1;
}

const edgexpu_backend *edgexpu_backend_cpu_baseline(void) {
    static const edgexpu_backend backend = {
        "cpu.baseline",
        cpu_baseline_available,
        cpu_baseline_load,
        cpu_baseline_generate
    };
    return &backend;
}
