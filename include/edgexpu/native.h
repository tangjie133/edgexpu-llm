#ifndef EDGEXPU_NATIVE_H
#define EDGEXPU_NATIVE_H

#include "edgexpu/arch.h"
#include "edgexpu/gguf.h"
#include "edgexpu/kv_cache.h"
#include "edgexpu/tokenizer.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EDGEXPU_NATIVE_MAX_TOKENS 1024

typedef struct edgexpu_native_layer {
    float *attn_norm;
    float *ffn_norm;
    float *wq;
    float *bq;
    float *wk;
    float *bk;
    float *wv;
    float *bv;
    float *wo;
    float *wgate;
    float *wup;
    float *wdown;
    int n_embd;
    int n_ff;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int ready;
} edgexpu_native_layer;

typedef struct edgexpu_native_session {
    edgexpu_gguf_info gguf;
    edgexpu_arch_adapter arch;
    edgexpu_tokenizer tokenizer;
    edgexpu_kv_cache kv;
    edgexpu_native_layer layer0;
    edgexpu_native_layer *layers;
    float *output_norm;
    float *last_hidden;
    uint32_t token_ids[EDGEXPU_NATIVE_MAX_TOKENS];
    const uint8_t *file_map;
    size_t file_map_size;
    int file_fd;
    int token_count;
    int prompt_token_count;
    int generated_tokens;
    int n_layers_cached;
    int loaded;
    int prefill_layers;
    float last_hidden_rms;
    double prefill_seconds;
} edgexpu_native_session;

void edgexpu_native_init(edgexpu_native_session *session);
void edgexpu_native_free(edgexpu_native_session *session);

int edgexpu_native_load(
    edgexpu_native_session *session,
    const char *gguf_path,
    char *error,
    size_t error_size
);

int edgexpu_native_tokenize(
    edgexpu_native_session *session,
    const char *text,
    char *error,
    size_t error_size
);

int edgexpu_native_reserve_kv(
    edgexpu_native_session *session,
    int tokens,
    char *error,
    size_t error_size
);

int edgexpu_native_forward_prefill(
    edgexpu_native_session *session,
    char *error,
    size_t error_size
);

int edgexpu_native_forward_layer0(
    edgexpu_native_session *session,
    char *error,
    size_t error_size
);

int edgexpu_native_generate_next(
    edgexpu_native_session *session,
    float temperature,
    uint32_t *token_id,
    char *piece,
    size_t piece_size,
    int *stopped,
    char *error,
    size_t error_size
);

void edgexpu_native_print_info(const edgexpu_native_session *session);

int edgexpu_native_selftest(const char *gguf_path);

#ifdef __cplusplus
}
#endif

#endif
