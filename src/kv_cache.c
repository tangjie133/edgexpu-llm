#include "edgexpu/kv_cache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 连续缓冲：[layer][pos][kv_head * head_dim]。pos 超出 max_seq 由 extend 拒绝。 */

static void set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
    }
}

void edgexpu_kv_cache_init(edgexpu_kv_cache *cache) {
    if (cache == NULL) {
        return;
    }
    memset(cache, 0, sizeof(*cache));
}

int edgexpu_kv_cache_allocate(
    edgexpu_kv_cache *cache,
    int n_layers,
    int n_kv_heads,
    int head_dim,
    int max_seq,
    char *error,
    size_t error_size
) {
    size_t elements;

    if (cache == NULL || n_layers <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_seq <= 0) {
        set_error(error, error_size, "native KV cache 参数无效");
        return 0;
    }

    edgexpu_kv_cache_free(cache);
    elements = (size_t)n_layers * (size_t)max_seq * (size_t)n_kv_heads * (size_t)head_dim;
    cache->k = (float *)calloc(elements, sizeof(float));
    cache->v = (float *)calloc(elements, sizeof(float));
    if (cache->k == NULL || cache->v == NULL) {
        edgexpu_kv_cache_free(cache);
        set_error(error, error_size, "native KV cache 分配失败");
        return 0;
    }

    cache->n_layers = n_layers;
    cache->n_kv_heads = n_kv_heads;
    cache->head_dim = head_dim;
    cache->max_seq = max_seq;
    cache->seq_len = 0;
    return 1;
}

void edgexpu_kv_cache_reset(edgexpu_kv_cache *cache) {
    if (cache == NULL) {
        return;
    }
    cache->seq_len = 0;
}

void edgexpu_kv_cache_free(edgexpu_kv_cache *cache) {
    if (cache == NULL) {
        return;
    }
    free(cache->k);
    free(cache->v);
    memset(cache, 0, sizeof(*cache));
}

int edgexpu_kv_cache_extend(edgexpu_kv_cache *cache, int tokens, char *error, size_t error_size) {
    if (cache == NULL || cache->k == NULL || cache->v == NULL) {
        set_error(error, error_size, "native KV cache 尚未分配");
        return 0;
    }
    if (tokens < 0) {
        set_error(error, error_size, "native KV cache extend 参数无效");
        return 0;
    }
    if (cache->seq_len + tokens > cache->max_seq) {
        if (error != NULL && error_size > 0) {
            snprintf(
                error,
                error_size,
                "native KV cache 超出窗口：seq_len=%d + %d > max_seq=%d",
                cache->seq_len,
                tokens,
                cache->max_seq
            );
        }
        return 0;
    }
    cache->seq_len += tokens;
    return 1;
}

float *edgexpu_kv_cache_k_at(edgexpu_kv_cache *cache, int layer, int pos) {
    size_t index;
    if (cache == NULL || cache->k == NULL || layer < 0 || layer >= cache->n_layers ||
        pos < 0 || pos >= cache->max_seq) {
        return NULL;
    }
    index = (((size_t)layer * (size_t)cache->max_seq) + (size_t)pos) *
        (size_t)cache->n_kv_heads * (size_t)cache->head_dim;
    return cache->k + index;
}

float *edgexpu_kv_cache_v_at(edgexpu_kv_cache *cache, int layer, int pos) {
    size_t index;
    if (cache == NULL || cache->v == NULL || layer < 0 || layer >= cache->n_layers ||
        pos < 0 || pos >= cache->max_seq) {
        return NULL;
    }
    index = (((size_t)layer * (size_t)cache->max_seq) + (size_t)pos) *
        (size_t)cache->n_kv_heads * (size_t)cache->head_dim;
    return cache->v + index;
}

size_t edgexpu_kv_cache_bytes(const edgexpu_kv_cache *cache) {
    size_t elements;
    if (cache == NULL || cache->k == NULL) {
        return 0;
    }
    elements = (size_t)cache->n_layers * (size_t)cache->max_seq * (size_t)cache->n_kv_heads * (size_t)cache->head_dim;
    return elements * sizeof(float) * 2u;
}

int edgexpu_kv_cache_selftest(char *error, size_t error_size) {
    edgexpu_kv_cache cache;

    edgexpu_kv_cache_init(&cache);
    if (!edgexpu_kv_cache_allocate(&cache, 2, 2, 4, 8, error, error_size)) {
        return 0;
    }
    if (!edgexpu_kv_cache_extend(&cache, 3, error, error_size) || cache.seq_len != 3) {
        edgexpu_kv_cache_free(&cache);
        set_error(error, error_size, "KV extend selftest failed");
        return 0;
    }
    if (edgexpu_kv_cache_bytes(&cache) == 0) {
        edgexpu_kv_cache_free(&cache);
        set_error(error, error_size, "KV bytes selftest failed");
        return 0;
    }
    edgexpu_kv_cache_reset(&cache);
    if (cache.seq_len != 0) {
        edgexpu_kv_cache_free(&cache);
        set_error(error, error_size, "KV reset selftest failed");
        return 0;
    }
    if (!edgexpu_kv_cache_extend(&cache, 8, error, error_size)) {
        edgexpu_kv_cache_free(&cache);
        return 0;
    }
    if (edgexpu_kv_cache_extend(&cache, 1, error, error_size)) {
        edgexpu_kv_cache_free(&cache);
        set_error(error, error_size, "KV overflow selftest should fail");
        return 0;
    }
    edgexpu_kv_cache_free(&cache);
    return 1;
}
