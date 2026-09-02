#ifndef EDGEXPU_BACKEND_H
#define EDGEXPU_BACKEND_H

#include <stddef.h>

#include "edgexpu/manifest.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum edgexpu_cpu_path {
    EDGEXPU_CPU_PATH_AUTO = 0,
    EDGEXPU_CPU_PATH_NATIVE = 1,
    EDGEXPU_CPU_PATH_LLAMA_BOOTSTRAP = 2
} edgexpu_cpu_path;

typedef struct edgexpu_generation_request {
    const char *prompt;
    int max_tokens;
    float temperature;
    edgexpu_cpu_path cpu_path;
} edgexpu_generation_request;

typedef struct edgexpu_backend_telemetry {
    char backend[EDGEXPU_TEXT_SMALL];
    char device[EDGEXPU_TEXT_SMALL];
    char fallback_reason[EDGEXPU_TEXT_MEDIUM];
    double total_seconds;
    double prefill_seconds;
    double decode_seconds;
    int prompt_tokens_approx;
    int completion_tokens_approx;
    int memory_used_mb;
} edgexpu_backend_telemetry;

typedef struct edgexpu_generation_result {
    char text[EDGEXPU_TEXT_LARGE];
    char backend[EDGEXPU_TEXT_SMALL];
    double elapsed_seconds;
    int prompt_tokens_approx;
    int completion_tokens_approx;
    edgexpu_backend_telemetry telemetry;
} edgexpu_generation_result;

typedef struct edgexpu_backend {
    const char *name;

    /* 检查 backend 是否可用，例如二进制是否存在、runtime 是否安装。 */
    int (*is_available)(void);

    /* 加载模型 artifact。初版只保存 manifest 信息，不做常驻模型池。 */
    int (*load)(
        const edgexpu_model_manifest *manifest,
        char *error,
        size_t error_size
    );

    /* 执行一次生成。
     * 当前同步接口是 MVP 门面；后续内部应拆成异步 load/prefill/decode/KV/stream jobs，
     * 由调度器在 CPU、NPU、dNPU 和 flash/memory 管线之间编排。
     */
    int (*generate)(
        const edgexpu_generation_request *request,
        edgexpu_generation_result *result,
        char *error,
        size_t error_size
    );
} edgexpu_backend;

/* 返回临时 CPU bootstrap backend。它对应 PowerInfer/llama.cpp 风格的本地二进制调用。 */
const edgexpu_backend *edgexpu_backend_cpu_baseline(void);

#ifdef __cplusplus
}
#endif

#endif
