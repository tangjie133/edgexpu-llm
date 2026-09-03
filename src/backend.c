#include "edgexpu/backend.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* 临时 CPU bootstrap：shell-out 到 llama / llama-cli。不是产品推理路径。 */

#if defined(_WIN32)
#include <windows.h>
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

static void set_error_path(char *error, size_t error_size, const char *prefix, const char *path) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s: %s", prefix, path != NULL ? path : "");
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

static int command_exists(const char *command) {
    char check_command[EDGEXPU_TEXT_MEDIUM + 64];
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

static const char *command_basename(const char *command) {
    const char *slash;
    const char *backslash;
    const char *separator;

    if (command == NULL) {
        return "";
    }

    slash = strrchr(command, '/');
    backslash = strrchr(command, '\\');
    separator = slash;
    if (backslash != NULL && (separator == NULL || backslash > separator)) {
        separator = backslash;
    }

    return separator != NULL ? separator + 1 : command;
}

static int command_uses_llama_subcommand(const char *command) {
    return strcmp(command_basename(command), "llama") == 0;
}

static const char *select_command(const edgexpu_model_manifest *manifest) {
    if (manifest != NULL && manifest->primary_artifact.command[0] != '\0') {
        return manifest->primary_artifact.command;
    }
    if (command_exists("powerinfer")) {
        return "powerinfer";
    }
    if (command_exists("llama")) {
        return "llama";
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

static int shell_command_prefix(const char *command, char *output, size_t output_size) {
    char quoted_command[EDGEXPU_TEXT_MEDIUM + 4];
    int written;

    if (output == NULL || output_size == 0) {
        return 0;
    }

    shell_command_name(command, quoted_command, sizeof(quoted_command));
    if (command_uses_llama_subcommand(command)) {
        written = snprintf(output, output_size, "%s cli", quoted_command);
    } else {
        written = snprintf(output, output_size, "%s", quoted_command);
    }

    return written >= 0 && (size_t)written < output_size;
}

static int command_can_start(const char *command) {
    char command_prefix[EDGEXPU_TEXT_MEDIUM + 8];
    char test_command[EDGEXPU_TEXT_MEDIUM + 64];
    int written;

    if (!shell_command_prefix(command, command_prefix, sizeof(command_prefix))) {
        return 0;
    }
#if defined(_WIN32)
    written = snprintf(test_command, sizeof(test_command), "%s --help >nul 2>&1", command_prefix);
#else
    written = snprintf(test_command, sizeof(test_command), "%s --help >/dev/null 2>&1", command_prefix);
#endif
    if (written < 0 || (size_t)written >= sizeof(test_command)) {
        return 0;
    }

    return system(test_command) == 0;
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

static void trim_trailing_space(char *text) {
    size_t length;

    if (text == NULL) {
        return;
    }

    length = strlen(text);
    while (length > 0 && isspace((unsigned char)text[length - 1])) {
        text[--length] = '\0';
    }
}

static void strip_llama_cli_output(char *text, const char *prompt) {
    char prompt_marker[EDGEXPU_TEXT_LARGE + 4];
    char *start = text;
    char *end;
    int written;

    if (text == NULL || text[0] == '\0') {
        return;
    }

    if (prompt != NULL && prompt[0] != '\0') {
        written = snprintf(prompt_marker, sizeof(prompt_marker), "> %s", prompt);
        if (written > 0 && (size_t)written < sizeof(prompt_marker)) {
            start = strstr(text, prompt_marker);
            if (start != NULL) {
                start += (size_t)written;
            } else {
                start = text;
            }
        }
    }

    while (*start == '\r' || *start == '\n') {
        start++;
    }

    end = strstr(start, "\n\n[ Prompt:");
    if (end == NULL) {
        end = strstr(start, "\r\n\r\n[ Prompt:");
    }
    if (end == NULL) {
        end = strstr(start, "\n\nExiting");
    }
    if (end != NULL) {
        *end = '\0';
    }

    if (start != text) {
        memmove(text, start, strlen(start) + 1);
    }
    trim_trailing_space(text);
}

void edgexpu_backend_cpu_baseline_bind(const edgexpu_model_manifest *manifest) {
    if (manifest == NULL) {
        return;
    }
    g_loaded_manifest = *manifest;
    g_loaded = 0;
}

static int cpu_baseline_available(void) {
    return command_exists("powerinfer") || command_exists("llama") || command_exists("llama-cli") || command_exists("main");
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
    if (!command_can_start(command)) {
        set_error(error, error_size, "CPU baseline backend 无法启动，请检查动态库或安装路径");
        return 0;
    }
    if (!file_exists(manifest->primary_artifact.path)) {
        set_error_path(error, error_size, "模型 artifact 不存在", manifest->primary_artifact.path);
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
    char shell_command[EDGEXPU_TEXT_LARGE * 3];
    char command_prefix[EDGEXPU_TEXT_MEDIUM + 8];
    char quoted_model[EDGEXPU_TEXT_LARGE + 4];
    char quoted_prompt[EDGEXPU_TEXT_LARGE + 4];
    char chunk[256];
    FILE *pipe;
    double started;
    size_t used = 0;
    int command_length;
    int close_code;
    const char *extra_args;

    if (!g_loaded) {
        if (!cpu_baseline_load(&g_loaded_manifest, error, error_size)) {
            return 0;
        }
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
    if (!shell_command_prefix(command, command_prefix, sizeof(command_prefix))) {
        set_error(error, error_size, "CPU baseline 命令过长");
        return 0;
    }
    extra_args = command_uses_llama_subcommand(command) ? " --single-turn --no-display-prompt" : "";
    shell_quote(g_loaded_manifest.primary_artifact.path, quoted_model, sizeof(quoted_model));
    shell_quote(request->prompt, quoted_prompt, sizeof(quoted_prompt));

    if (request->top_p > 0.0f && request->top_p < 1.0f) {
        command_length = snprintf(
            shell_command,
            sizeof(shell_command),
            "%s -m %s -p %s -n %d --temp %.3f --top-p %.3f%s",
            command_prefix,
            quoted_model,
            quoted_prompt,
            request->max_tokens,
            request->temperature,
            request->top_p,
            extra_args
        );
    } else {
        command_length = snprintf(
            shell_command,
            sizeof(shell_command),
            "%s -m %s -p %s -n %d --temp %.3f%s",
            command_prefix,
            quoted_model,
            quoted_prompt,
            request->max_tokens,
            request->temperature,
            extra_args
        );
    }
    if (command_length < 0 || (size_t)command_length >= sizeof(shell_command)) {
        set_error(error, error_size, "CPU baseline 命令过长");
        return 0;
    }

    started = now_seconds();
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
    result->elapsed_seconds = now_seconds() - started;
    if (command_uses_llama_subcommand(command)) {
        strip_llama_cli_output(result->text, request->prompt);
    }
    result->completion_tokens_approx = count_tokens_approx(result->text);

    if (close_code != 0) {
        set_error(error, error_size, "CPU baseline backend 执行失败");
        return 0;
    }

    snprintf(result->telemetry.backend, sizeof(result->telemetry.backend), "%s", result->backend);
    snprintf(result->telemetry.device, sizeof(result->telemetry.device), "cpu");
    snprintf(
        result->telemetry.fallback_reason,
        sizeof(result->telemetry.fallback_reason),
        "temporary llama.cpp bootstrap backend; native tokenizer/prefill/decode are not connected yet"
    );
    result->telemetry.total_seconds = result->elapsed_seconds;
    result->telemetry.prefill_seconds = 0.0;
    result->telemetry.decode_seconds = result->elapsed_seconds;
    result->telemetry.prompt_tokens_approx = result->prompt_tokens_approx;
    result->telemetry.completion_tokens_approx = result->completion_tokens_approx;
    result->telemetry.memory_used_mb = 0;

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
