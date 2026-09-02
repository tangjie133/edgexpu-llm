#ifndef EDGEXPU_MANIFEST_H
#define EDGEXPU_MANIFEST_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 初版先使用固定长度字段，避免在边缘设备上引入复杂内存所有权。 */
#define EDGEXPU_TEXT_SMALL 64
#define EDGEXPU_TEXT_MEDIUM 256
#define EDGEXPU_TEXT_LARGE 4096

typedef struct edgexpu_model_artifact {
    char backend[EDGEXPU_TEXT_SMALL];
    char path[EDGEXPU_TEXT_LARGE];
    char format[EDGEXPU_TEXT_SMALL];
    char quantization[EDGEXPU_TEXT_SMALL];
    char command[EDGEXPU_TEXT_MEDIUM];
} edgexpu_model_artifact;

typedef struct edgexpu_model_manifest {
    char model_id[EDGEXPU_TEXT_SMALL];
    char family[EDGEXPU_TEXT_SMALL];
    char parameter_size[EDGEXPU_TEXT_SMALL];
    int context_length;
    int memory_required_mb;
    int kv_cache_required_mb;
    char fallback_policy[EDGEXPU_TEXT_SMALL];
    edgexpu_model_artifact primary_artifact;
} edgexpu_model_manifest;

/* 从 JSON manifest 读取最小字段。后续应替换为严格 JSON parser。 */
int edgexpu_manifest_load(
    const char *path,
    edgexpu_model_manifest *manifest,
    char *error,
    size_t error_size
);

/* 打印 manifest 摘要，供 CLI inspect 和调试阶段使用。 */
void edgexpu_manifest_print(const edgexpu_model_manifest *manifest);

#ifdef __cplusplus
}
#endif

#endif
