#ifndef EDGEXPU_RUNTIME_H
#define EDGEXPU_RUNTIME_H

#include <stddef.h>

#include "edgexpu/backend.h"
#include "edgexpu/executor.h"
#include "edgexpu/manifest.h"
#include "edgexpu/native.h"
#include "edgexpu/profiler.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Runtime 门面：持有 manifest、native session、executor 和最近一次 telemetry。
 * generate 按 scheduler 提交 job；tokenize 前套用模型包 chat template。
 */

typedef struct edgexpu_runtime {
    edgexpu_model_manifest manifest;
    edgexpu_device_profile device_profile;
    edgexpu_backend_telemetry last_telemetry;
    edgexpu_native_session native;
    edgexpu_executor executor;
    const edgexpu_backend *backend;
    int has_device_profile;
    int has_last_telemetry;
    int loaded;
} edgexpu_runtime;

/* 初始化 runtime。结构体由调用者持有，便于嵌入式场景控制内存。 */
void edgexpu_runtime_init(edgexpu_runtime *runtime);
void edgexpu_runtime_shutdown(edgexpu_runtime *runtime);

/* 加载 manifest 并选择 backend。GGUF 同时尝试 native_load。 */
int edgexpu_runtime_load_model(
    edgexpu_runtime *runtime,
    const char *manifest_path,
    char *error,
    size_t error_size
);

typedef void (*edgexpu_runtime_stream_callback)(
    const char *token,
    int token_index,
    int token_count,
    void *user_data
);

/* 执行一次生成或 benchmark 请求。 */
int edgexpu_runtime_generate(
    edgexpu_runtime *runtime,
    const edgexpu_generation_request *request,
    edgexpu_generation_result *result,
    char *error,
    size_t error_size
);

/* 生成过程中每个 native token 提交 stream_token job；无 native decode 时回放 batched 文本。 */
int edgexpu_runtime_generate_stream(
    edgexpu_runtime *runtime,
    const edgexpu_generation_request *request,
    edgexpu_generation_result *result,
    edgexpu_runtime_stream_callback on_token,
    void *stream_user_data,
    char *error,
    size_t error_size
);

#ifdef __cplusplus
}
#endif

#endif
