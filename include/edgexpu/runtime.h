#ifndef EDGEXPU_RUNTIME_H
#define EDGEXPU_RUNTIME_H

#include <stddef.h>

#include "edgexpu/backend.h"
#include "edgexpu/executor.h"
#include "edgexpu/manifest.h"
#include "edgexpu/profiler.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct edgexpu_runtime {
    edgexpu_model_manifest manifest;
    edgexpu_device_profile device_profile;
    edgexpu_executor executor;
    const edgexpu_backend *backend;
    int has_device_profile;
    int loaded;
} edgexpu_runtime;

/* 初始化 runtime。结构体由调用者持有，便于嵌入式场景控制内存。 */
void edgexpu_runtime_init(edgexpu_runtime *runtime);

/* 加载 manifest 并选择 backend。 */
int edgexpu_runtime_load_model(
    edgexpu_runtime *runtime,
    const char *manifest_path,
    char *error,
    size_t error_size
);

/* 执行一次生成或 benchmark 请求。 */
int edgexpu_runtime_generate(
    edgexpu_runtime *runtime,
    const edgexpu_generation_request *request,
    edgexpu_generation_result *result,
    char *error,
    size_t error_size
);

#ifdef __cplusplus
}
#endif

#endif
