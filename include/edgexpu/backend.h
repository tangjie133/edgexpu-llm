#ifndef EDGEXPU_BACKEND_H
#define EDGEXPU_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#include "edgexpu/manifest.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Backend 插件契约。engine 对 cpu.native 是 edgexpu_native_session*，对 llama 可为空。
 * 分步接口（tokenize/prefill/decode_step）为 NULL 时，runtime 走一次性 generate()。
 */

typedef enum edgexpu_cpu_path {
    EDGEXPU_CPU_PATH_AUTO = 0,
    EDGEXPU_CPU_PATH_NATIVE = 1,
    EDGEXPU_CPU_PATH_LLAMA_BOOTSTRAP = 2
} edgexpu_cpu_path;

typedef struct edgexpu_generation_request {
    const char *prompt;
    int max_tokens;
    float temperature;
    float top_p;
    edgexpu_cpu_path cpu_path;
    int prompt_is_formatted;
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
    char text[EDGEXPU_TEXT_PROMPT];
    char backend[EDGEXPU_TEXT_SMALL];
    char finish_reason[EDGEXPU_TEXT_SMALL];
    double elapsed_seconds;
    int prompt_tokens_approx;
    int completion_tokens_approx;
    edgexpu_backend_telemetry telemetry;
} edgexpu_generation_result;

typedef struct edgexpu_backend {
    const char *name;
    int (*is_available)(void);
    int (*load)(
        void *engine,
        const edgexpu_model_manifest *manifest,
        char *error,
        size_t error_size
    );
    int (*generate)(
        void *engine,
        const edgexpu_generation_request *request,
        edgexpu_generation_result *result,
        char *error,
        size_t error_size
    );
    int (*tokenize)(void *engine, const char *text, char *error, size_t error_size);
    int (*ensure_window)(void *engine, int n_prompt, int n_new, char *error, size_t error_size);
    int (*prefill)(void *engine, char *error, size_t error_size);
    int (*reserve_kv)(void *engine, int tokens, char *error, size_t error_size);
    int (*decode_step)(
        void *engine,
        float temperature,
        float top_p,
        uint32_t *token_id,
        char *piece,
        size_t piece_size,
        int *stopped,
        char *error,
        size_t error_size
    );
} edgexpu_backend;

const edgexpu_backend *edgexpu_backend_cpu_baseline(void);
const edgexpu_backend *edgexpu_backend_cpu_native(void);

void edgexpu_backend_cpu_baseline_bind(const edgexpu_model_manifest *manifest);

#ifdef __cplusplus
}
#endif

#endif
