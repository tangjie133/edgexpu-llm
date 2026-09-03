#ifndef EDGEXPU_BACKEND_H
#define EDGEXPU_BACKEND_H

#include <stddef.h>

#include "edgexpu/manifest.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Backend 契约：可用性检查、加载 artifact、执行一次生成。
 * cpu.native 由 runtime 内部 native session 实现；这里的 cpu.baseline 是 llama.cpp shell-out。
 */

typedef enum edgexpu_cpu_path {
    EDGEXPU_CPU_PATH_AUTO = 0,           /* native 就绪则走 native，否则 llama bootstrap */
    EDGEXPU_CPU_PATH_NATIVE = 1,         /* 强制 native CPU fallback */
    EDGEXPU_CPU_PATH_LLAMA_BOOTSTRAP = 2 /* 强制临时 llama CLI */
} edgexpu_cpu_path;

typedef struct edgexpu_generation_request {
    const char *prompt;
    int max_tokens;
    float temperature;
    float top_p; /* <=0 或 >=1 表示不截断；temperature≈0 时忽略，走 greedy */
    edgexpu_cpu_path cpu_path;
    int prompt_is_formatted; /* HTTP 已按 messages 套模板，runtime 不再套一层 */
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
    int memory_used_mb; /* mmap 权重大小 + KV，向上取整到 MB */
} edgexpu_backend_telemetry;

typedef struct edgexpu_generation_result {
    char text[EDGEXPU_TEXT_PROMPT];
    char backend[EDGEXPU_TEXT_SMALL];
    char finish_reason[EDGEXPU_TEXT_SMALL]; /* stop | length */
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

/* 只记住 manifest，不探测 PATH。native 成功时用；llama generate 再真正 load。 */
void edgexpu_backend_cpu_baseline_bind(const edgexpu_model_manifest *manifest);

#ifdef __cplusplus
}
#endif

#endif
