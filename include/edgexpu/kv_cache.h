#ifndef EDGEXPU_KV_CACHE_H
#define EDGEXPU_KV_CACHE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 有限窗口 KV cache。按层、KV 头、位置存放 f32 K/V。
 * 默认 256 只是 load 时的占位；推理前应按 prompt + max_tokens 扩到所需长度，
 * 并受模型 context_length 上限约束。超出窗口必须报错，不能截断。
 */

#define EDGEXPU_KV_DEFAULT_MAX_SEQ 256

typedef struct edgexpu_kv_cache {
    int n_layers;
    int n_kv_heads;
    int head_dim;
    int max_seq;
    int seq_len; /* 当前已占用的序列长度（prefill 后等于 prompt token 数） */
    float *k;
    float *v;
} edgexpu_kv_cache;

void edgexpu_kv_cache_init(edgexpu_kv_cache *cache);

int edgexpu_kv_cache_allocate(
    edgexpu_kv_cache *cache,
    int n_layers,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    char *error,
    size_t error_size
);

void edgexpu_kv_cache_reset(edgexpu_kv_cache *cache);
void edgexpu_kv_cache_free(edgexpu_kv_cache *cache);

/* 增加 seq_len。超出 max_seq 失败，由上层停止生成。 */
int edgexpu_kv_cache_extend(edgexpu_kv_cache *cache, int tokens, char *error, size_t error_size);

float *edgexpu_kv_cache_k_at(edgexpu_kv_cache *cache, int layer, int pos);
float *edgexpu_kv_cache_v_at(edgexpu_kv_cache *cache, int layer, int pos);

size_t edgexpu_kv_cache_bytes(const edgexpu_kv_cache *cache);

int edgexpu_kv_cache_selftest(char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
