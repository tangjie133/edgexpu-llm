#include "edgexpu/manifest.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 这里是初版最小 JSON 读取器，只服务当前 manifest schema。
 * 后续进入正式 runtime 时，应替换为 cJSON、yyjson 或项目内统一 JSON 解析器。
 */

static void set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
    }
}

static char *read_file(const char *path, char *error, size_t error_size) {
    FILE *file = fopen(path, "rb");
    long size;
    char *buffer;

    if (file == NULL) {
        set_error(error, error_size, "无法打开 manifest 文件");
        return NULL;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        set_error(error, error_size, "无法读取 manifest 文件大小");
        return NULL;
    }

    size = ftell(file);
    if (size < 0) {
        fclose(file);
        set_error(error, error_size, "manifest 文件大小无效");
        return NULL;
    }
    rewind(file);

    buffer = (char *)malloc((size_t)size + 1);
    if (buffer == NULL) {
        fclose(file);
        set_error(error, error_size, "manifest 读取内存不足");
        return NULL;
    }

    if (fread(buffer, 1, (size_t)size, file) != (size_t)size) {
        free(buffer);
        fclose(file);
        set_error(error, error_size, "manifest 文件读取失败");
        return NULL;
    }

    buffer[size] = '\0';
    fclose(file);
    return buffer;
}

static int extract_string(
    const char *json,
    const char *key,
    char *output,
    size_t output_size
) {
    char pattern[128];
    const char *cursor;
    const char *start;
    const char *end;
    size_t length;

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    cursor = strstr(json, pattern);
    if (cursor == NULL) {
        return 0;
    }

    cursor = strchr(cursor + strlen(pattern), ':');
    if (cursor == NULL) {
        return 0;
    }

    start = strchr(cursor, '"');
    if (start == NULL) {
        return 0;
    }
    start++;

    end = strchr(start, '"');
    if (end == NULL) {
        return 0;
    }

    length = (size_t)(end - start);
    if (length >= output_size) {
        length = output_size - 1;
    }

    memcpy(output, start, length);
    output[length] = '\0';
    return 1;
}

static int extract_int(const char *json, const char *key, int *output) {
    char pattern[128];
    const char *cursor;

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    cursor = strstr(json, pattern);
    if (cursor == NULL) {
        return 0;
    }

    cursor = strchr(cursor + strlen(pattern), ':');
    if (cursor == NULL) {
        return 0;
    }

    *output = atoi(cursor + 1);
    return 1;
}

static int path_is_absolute(const char *path) {
    if (path == NULL || path[0] == '\0') {
        return 0;
    }

    if (path[0] == '/' || path[0] == '\\') {
        return 1;
    }

    return isalpha((unsigned char)path[0]) &&
        path[1] == ':' &&
        (path[2] == '/' || path[2] == '\\');
}

static int manifest_directory(const char *path, char *output, size_t output_size) {
    const char *slash;
    const char *backslash;
    const char *separator;
    size_t length;

    if (path == NULL || output == NULL || output_size == 0) {
        return 0;
    }

    slash = strrchr(path, '/');
    backslash = strrchr(path, '\\');
    separator = slash;
    if (backslash != NULL && (separator == NULL || backslash > separator)) {
        separator = backslash;
    }

    if (separator == NULL) {
        snprintf(output, output_size, ".");
        return 1;
    }

    length = (size_t)(separator - path);
    if (length == 0) {
        length = 1;
    }
    if (length >= output_size) {
        return 0;
    }

    memcpy(output, path, length);
    output[length] = '\0';
    return 1;
}

static int resolve_artifact_path(
    const char *manifest_path,
    char *artifact_path,
    size_t artifact_path_size,
    char *error,
    size_t error_size
) {
    char directory[EDGEXPU_TEXT_LARGE];
    char resolved[EDGEXPU_TEXT_LARGE];
    int written;

    if (artifact_path == NULL || artifact_path[0] == '\0' || path_is_absolute(artifact_path)) {
        return 1;
    }

    if (!manifest_directory(manifest_path, directory, sizeof(directory))) {
        set_error(error, error_size, "manifest 路径过长");
        return 0;
    }

    written = snprintf(resolved, sizeof(resolved), "%s/%s", directory, artifact_path);
    if (written < 0 || (size_t)written >= sizeof(resolved) || (size_t)written >= artifact_path_size) {
        set_error(error, error_size, "artifact 路径过长");
        return 0;
    }

    snprintf(artifact_path, artifact_path_size, "%s", resolved);
    return 1;
}

int edgexpu_manifest_load(
    const char *path,
    edgexpu_model_manifest *manifest,
    char *error,
    size_t error_size
) {
    char *json;

    if (path == NULL || manifest == NULL) {
        set_error(error, error_size, "manifest 参数为空");
        return 0;
    }

    memset(manifest, 0, sizeof(*manifest));
    json = read_file(path, error, error_size);
    if (json == NULL) {
        return 0;
    }

    /* 初版只解析主 artifact；这能先打通模型合约，后续再扩展多 artifact。 */
    if (!extract_string(json, "model_id", manifest->model_id, sizeof(manifest->model_id)) ||
        !extract_string(json, "family", manifest->family, sizeof(manifest->family)) ||
        !extract_string(json, "parameter_size", manifest->parameter_size, sizeof(manifest->parameter_size)) ||
        !extract_int(json, "context_length", &manifest->context_length) ||
        !extract_string(json, "fallback_policy", manifest->fallback_policy, sizeof(manifest->fallback_policy)) ||
        !extract_string(json, "backend", manifest->primary_artifact.backend, sizeof(manifest->primary_artifact.backend)) ||
        !extract_string(json, "path", manifest->primary_artifact.path, sizeof(manifest->primary_artifact.path)) ||
        !extract_string(json, "format", manifest->primary_artifact.format, sizeof(manifest->primary_artifact.format))) {
        free(json);
        set_error(error, error_size, "manifest 缺少必要字段");
        return 0;
    }

    extract_int(json, "memory_required_mb", &manifest->memory_required_mb);
    extract_int(json, "kv_cache_required_mb", &manifest->kv_cache_required_mb);
    extract_string(json, "quantization", manifest->primary_artifact.quantization, sizeof(manifest->primary_artifact.quantization));
    extract_string(json, "command", manifest->primary_artifact.command, sizeof(manifest->primary_artifact.command));

    if (!resolve_artifact_path(path, manifest->primary_artifact.path, sizeof(manifest->primary_artifact.path), error, error_size)) {
        free(json);
        return 0;
    }

    free(json);
    return 1;
}

void edgexpu_manifest_print(const edgexpu_model_manifest *manifest) {
    if (manifest == NULL) {
        return;
    }

    printf("model_id: %s\n", manifest->model_id);
    printf("family: %s\n", manifest->family);
    printf("parameter_size: %s\n", manifest->parameter_size);
    printf("context_length: %d\n", manifest->context_length);
    printf("fallback_policy: %s\n", manifest->fallback_policy);
    printf("artifact.backend: %s\n", manifest->primary_artifact.backend);
    printf("artifact.path: %s\n", manifest->primary_artifact.path);
    printf("artifact.format: %s\n", manifest->primary_artifact.format);
    printf("artifact.quantization: %s\n", manifest->primary_artifact.quantization);
    printf("artifact.command: %s\n", manifest->primary_artifact.command);
}
