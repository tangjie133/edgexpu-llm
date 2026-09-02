#include "edgexpu/native.h"

#include "edgexpu/chat.h"
#include "edgexpu/cpu_kernel.h"
#include "edgexpu/gguf_quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

static void set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
    }
}

static void native_unmap(edgexpu_native_session *session) {
    if (session == NULL) {
        return;
    }
#if defined(_WIN32)
    if (session->file_map != NULL) {
        UnmapViewOfFile(session->file_map);
    }
#else
    if (session->file_map != NULL && session->file_map_size > 0) {
        munmap((void *)session->file_map, session->file_map_size);
    }
    if (session->file_fd >= 0) {
        close(session->file_fd);
    }
#endif
    session->file_map = NULL;
    session->file_map_size = 0;
    session->file_fd = -1;
}

static int native_map_file(edgexpu_native_session *session, const char *path, char *error, size_t error_size) {
#if defined(_WIN32)
    HANDLE file;
    HANDLE mapping;
    LARGE_INTEGER size;
    file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) {
        set_error(error, error_size, "无法打开 GGUF 文件进行 mmap");
        return 0;
    }
    if (!GetFileSizeEx(file, &size)) {
        CloseHandle(file);
        set_error(error, error_size, "无法读取 GGUF 文件大小");
        return 0;
    }
    mapping = CreateFileMappingA(file, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(file);
    if (mapping == NULL) {
        set_error(error, error_size, "CreateFileMapping 失败");
        return 0;
    }
    session->file_map = (const uint8_t *)MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mapping);
    if (session->file_map == NULL) {
        set_error(error, error_size, "MapViewOfFile 失败");
        return 0;
    }
    session->file_map_size = (size_t)size.QuadPart;
    session->file_fd = -1;
    return 1;
#else
    int fd = open(path, O_RDONLY);
    void *mapped;
    if (fd < 0) {
        set_error(error, error_size, "无法打开 GGUF 文件进行 mmap");
        return 0;
    }
    mapped = mmap(NULL, (size_t)session->gguf.file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        set_error(error, error_size, "mmap GGUF 失败");
        return 0;
    }
    session->file_fd = fd;
    session->file_map = (const uint8_t *)mapped;
    session->file_map_size = (size_t)session->gguf.file_size;
    return 1;
#endif
}

static const uint8_t *tensor_bytes(
    const edgexpu_native_session *session,
    const edgexpu_gguf_tensor *tensor
) {
    uint64_t offset;
    if (session == NULL || session->file_map == NULL || tensor == NULL) {
        return NULL;
    }
    offset = session->gguf.data_offset + tensor->offset;
    if (offset >= session->file_map_size) {
        return NULL;
    }
    return session->file_map + offset;
}

static float default_rope_freq(const edgexpu_native_session *session) {
    if (session != NULL && session->gguf.rope_freq_base > 0.0f) {
        return session->gguf.rope_freq_base;
    }
    if (session != NULL && session->arch.rope == EDGEXPU_ROPE_NORM) {
        return 10000.0f;
    }
    return 1000000.0f;
}

static void apply_rope(
    const edgexpu_native_session *session,
    float *x,
    int n_heads,
    int head_dim,
    int pos
) {
    float freq_base = default_rope_freq(session);
    if (session != NULL && session->arch.rope == EDGEXPU_ROPE_NORM) {
        edgexpu_cpu_rope_norm(x, n_heads, head_dim, pos, freq_base);
        return;
    }
    edgexpu_cpu_rope_neox(x, n_heads, head_dim, pos, freq_base);
}

static const edgexpu_gguf_tensor *token_embedding_tensor(const edgexpu_native_session *session) {
    return session == NULL ? NULL : edgexpu_gguf_find_tensor(&session->gguf, "token_embd.weight");
}

static const edgexpu_gguf_tensor *logit_weight_tensor(const edgexpu_native_session *session) {
    const edgexpu_gguf_tensor *output;
    if (session == NULL) {
        return NULL;
    }
    output = edgexpu_gguf_find_tensor(&session->gguf, "output.weight");
    if (output != NULL) {
        return output;
    }
    return token_embedding_tensor(session);
}

static int load_embedding_row(
    const edgexpu_native_session *session,
    uint32_t token,
    int n_embd,
    float *dst,
    char *error,
    size_t error_size
) {
    const edgexpu_gguf_tensor *embd = token_embedding_tensor(session);
    const uint8_t *src = tensor_bytes(session, embd);
    if (embd == NULL || src == NULL) {
        set_error(error, error_size, "无法读取 token_embd.weight");
        return 0;
    }
    return edgexpu_gguf_dequantize_row(src, embd, token, n_embd, dst, error, error_size);
}

static float logit_dot_row(
    const edgexpu_native_session *session,
    const edgexpu_gguf_tensor *tensor,
    uint32_t row,
    int n_embd,
    const float *hidden,
    float *scratch
) {
    const uint8_t *src = tensor_bytes(session, tensor);
    int i;
    float acc = 0.0f;
    if (src == NULL || tensor == NULL || hidden == NULL) {
        return 0.0f;
    }
    if (tensor->type == EDGEXPU_GGUF_TYPE_Q8_0) {
        return edgexpu_gguf_q8_0_dot_row(src, row, n_embd, hidden);
    }
    if (scratch == NULL || !edgexpu_gguf_dequantize_row(src, tensor, row, n_embd, scratch, NULL, 0)) {
        return 0.0f;
    }
    for (i = 0; i < n_embd; i++) {
        acc += scratch[i] * hidden[i];
    }
    return acc;
}

static void layer_free(edgexpu_native_layer *layer) {
    if (layer == NULL) {
        return;
    }
    free(layer->attn_norm);
    free(layer->ffn_norm);
    free(layer->wq);
    free(layer->bq);
    free(layer->wk);
    free(layer->bk);
    free(layer->wv);
    free(layer->bv);
    free(layer->wo);
    free(layer->wgate);
    free(layer->wup);
    free(layer->wdown);
    memset(layer, 0, sizeof(*layer));
}

static int load_named_weight(
    edgexpu_native_session *session,
    const char *name,
    float **dst,
    char *error,
    size_t error_size
) {
    const edgexpu_gguf_tensor *tensor = edgexpu_gguf_find_tensor(&session->gguf, name);
    uint64_t n;
    const uint8_t *src;

    if (tensor == NULL) {
        snprintf(error, error_size, "缺少 tensor: %s", name);
        return 0;
    }
    n = edgexpu_gguf_tensor_elements(tensor);
    src = tensor_bytes(session, tensor);
    if (src == NULL || n == 0) {
        snprintf(error, error_size, "无法映射 tensor: %s", name);
        return 0;
    }
    *dst = (float *)malloc((size_t)n * sizeof(float));
    if (*dst == NULL) {
        set_error(error, error_size, "layer 权重分配失败");
        return 0;
    }
    if (!edgexpu_gguf_dequantize(src, tensor->type, n, *dst, error, error_size)) {
        free(*dst);
        *dst = NULL;
        return 0;
    }
    return 1;
}

static double native_now_seconds(void) {
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) == 0) {
        return (double)timestamp.tv_sec + (double)timestamp.tv_nsec / 1e9;
    }
    return 0.0;
}

static int load_layer_into(
    edgexpu_native_session *session,
    int layer_index,
    edgexpu_native_layer *layer,
    char *error,
    size_t error_size
) {
    char name[96];

    layer_free(layer);
    layer->n_embd = (int)session->gguf.embedding_length;
    layer->n_ff = (int)session->gguf.feed_forward_length;
    layer->n_heads = (int)session->gguf.head_count;
    layer->n_kv_heads = (int)session->gguf.head_count_kv;
    layer->head_dim = edgexpu_gguf_head_dim(&session->gguf);
    if (layer->n_embd <= 0 || layer->n_ff <= 0 || layer->n_heads <= 0 ||
        layer->n_kv_heads <= 0 || layer->head_dim <= 0 || layer->n_heads % layer->n_kv_heads != 0) {
        set_error(error, error_size, "native layer 架构参数无效");
        return 0;
    }

#define LOAD_BLK(field, suffix) \
    do { \
        snprintf(name, sizeof(name), "blk.%d.%s", layer_index, suffix); \
        if (!load_named_weight(session, name, &layer->field, error, error_size)) { \
            layer_free(layer); \
            return 0; \
        } \
    } while (0)

    LOAD_BLK(attn_norm, "attn_norm.weight");
    LOAD_BLK(ffn_norm, "ffn_norm.weight");
    LOAD_BLK(wq, "attn_q.weight");
    LOAD_BLK(wk, "attn_k.weight");
    LOAD_BLK(wv, "attn_v.weight");
    LOAD_BLK(wo, "attn_output.weight");
    LOAD_BLK(wgate, "ffn_gate.weight");
    LOAD_BLK(wup, "ffn_up.weight");
    LOAD_BLK(wdown, "ffn_down.weight");
    if (session->arch.has_qkv_bias) {
        LOAD_BLK(bq, "attn_q.bias");
        LOAD_BLK(bk, "attn_k.bias");
        LOAD_BLK(bv, "attn_v.bias");
    } else {
        layer->bq = NULL;
        layer->bk = NULL;
        layer->bv = NULL;
    }
#undef LOAD_BLK

    layer->ready = 1;
    return 1;
}

static int load_layer(edgexpu_native_session *session, int layer_index, char *error, size_t error_size) {
    return load_layer_into(session, layer_index, &session->layer0, error, error_size);
}

static void free_cached_layers(edgexpu_native_session *session) {
    int i;
    if (session == NULL) {
        return;
    }
    if (session->layers != NULL) {
        memset(&session->layer0, 0, sizeof(session->layer0));
        for (i = 0; i < session->n_layers_cached; i++) {
            layer_free(&session->layers[i]);
        }
        free(session->layers);
        session->layers = NULL;
        session->n_layers_cached = 0;
        return;
    }
    layer_free(&session->layer0);
}

static int ensure_layers_cached(edgexpu_native_session *session, char *error, size_t error_size) {
    int n_layers;
    int i;
    int start = 0;

    n_layers = (int)session->gguf.block_count;
    if (n_layers <= 0) {
        set_error(error, error_size, "native 模型 block_count 无效");
        return 0;
    }
    if (session->layers != NULL && session->n_layers_cached == n_layers) {
        return 1;
    }

    if (session->layers != NULL) {
        free_cached_layers(session);
    }
    session->layers = (edgexpu_native_layer *)calloc((size_t)n_layers, sizeof(edgexpu_native_layer));
    if (session->layers == NULL) {
        set_error(error, error_size, "native 全层权重缓存分配失败");
        return 0;
    }

    if (session->layer0.ready) {
        session->layers[0] = session->layer0;
        memset(&session->layer0, 0, sizeof(session->layer0));
        session->n_layers_cached = 1;
        start = 1;
    }

    for (i = start; i < n_layers; i++) {
        if (!load_layer_into(session, i, &session->layers[i], error, error_size)) {
            session->n_layers_cached = i;
            return 0;
        }
        session->n_layers_cached = i + 1;
    }
    session->layer0 = session->layers[0];
    return 1;
}

static float vector_rms(const float *x, int n) {
    int i;
    double ss = 0.0;
    if (x == NULL || n <= 0) {
        return 0.0f;
    }
    for (i = 0; i < n; i++) {
        ss += (double)x[i] * (double)x[i];
    }
    return (float)sqrt(ss / (double)n);
}

void edgexpu_native_init(edgexpu_native_session *session) {
    if (session == NULL) {
        return;
    }
    memset(session, 0, sizeof(*session));
    session->file_fd = -1;
    edgexpu_gguf_info_init(&session->gguf);
    edgexpu_tokenizer_init(&session->tokenizer);
    edgexpu_kv_cache_init(&session->kv);
}

void edgexpu_native_free(edgexpu_native_session *session) {
    if (session == NULL) {
        return;
    }
    free_cached_layers(session);
    free(session->output_norm);
    session->output_norm = NULL;
    free(session->last_hidden);
    session->last_hidden = NULL;
    edgexpu_tokenizer_free(&session->tokenizer);
    edgexpu_kv_cache_free(&session->kv);
    native_unmap(session);
    memset(session, 0, sizeof(*session));
    session->file_fd = -1;
}

int edgexpu_native_load(
    edgexpu_native_session *session,
    const char *gguf_path,
    char *error,
    size_t error_size
) {
    int head_dim;
    int max_seq;

    if (session == NULL) {
        set_error(error, error_size, "native session 为空");
        return 0;
    }

    edgexpu_native_free(session);
    if (!edgexpu_gguf_load(gguf_path, &session->gguf, &session->tokenizer, error, error_size)) {
        edgexpu_native_free(session);
        return 0;
    }
    if (!edgexpu_arch_from_gguf(&session->gguf, &session->arch, error, error_size)) {
        edgexpu_native_free(session);
        return 0;
    }

    if (!native_map_file(session, gguf_path, error, error_size)) {
        edgexpu_native_free(session);
        return 0;
    }

    head_dim = edgexpu_gguf_head_dim(&session->gguf);
    max_seq = EDGEXPU_KV_DEFAULT_MAX_SEQ;
    if (session->gguf.context_length > 0 && (int)session->gguf.context_length < max_seq) {
        max_seq = (int)session->gguf.context_length;
    }
    if (session->gguf.block_count > 0 && session->gguf.head_count_kv > 0 && head_dim > 0) {
        if (!edgexpu_kv_cache_allocate(
                &session->kv,
                (int)session->gguf.block_count,
                (int)session->gguf.head_count_kv,
                head_dim,
                max_seq,
                error,
                error_size)) {
            edgexpu_native_free(session);
            return 0;
        }
    }

    if (!load_layer(session, 0, error, error_size)) {
        edgexpu_native_free(session);
        return 0;
    }
    if (!load_named_weight(session, "output_norm.weight", &session->output_norm, error, error_size)) {
        edgexpu_native_free(session);
        return 0;
    }

    session->loaded = 1;
    return 1;
}

int edgexpu_native_tokenize(
    edgexpu_native_session *session,
    const char *text,
    char *error,
    size_t error_size
) {
    if (session == NULL || !session->loaded) {
        set_error(error, error_size, "native session 尚未加载");
        return 0;
    }

    return edgexpu_tokenizer_encode(
        &session->tokenizer,
        text,
        session->token_ids,
        EDGEXPU_NATIVE_MAX_TOKENS,
        &session->token_count,
        error,
        error_size
    );
}

int edgexpu_native_reserve_kv(
    edgexpu_native_session *session,
    int tokens,
    char *error,
    size_t error_size
) {
    if (session == NULL || !session->loaded || session->kv.k == NULL) {
        set_error(error, error_size, "native KV cache 尚未就绪");
        return 0;
    }

    edgexpu_kv_cache_reset(&session->kv);
    return edgexpu_kv_cache_extend(&session->kv, tokens, error, error_size);
}

static int apply_transformer_layer(
    edgexpu_native_session *session,
    const edgexpu_native_layer *layer,
    int layer_index,
    int seq,
    float *hidden,
    float *normed,
    float *residual,
    float *q,
    float *k,
    float *v,
    float *attn,
    float *proj,
    float *gate,
    float *up,
    float *down,
    float *scores,
    char *error,
    size_t error_size
) {
    int n_embd = layer->n_embd;
    int n_ff = layer->n_ff;
    int n_heads = layer->n_heads;
    int n_kv = layer->n_kv_heads;
    int head_dim = layer->head_dim;
    int n_rep = n_heads / n_kv;
    int t;
    float eps = session->gguf.rms_eps > 0.0f ? session->gguf.rms_eps : 1e-6f;
    float attn_scale = 1.0f / sqrtf((float)head_dim);

    for (t = 0; t < seq; t++) {
        float *xt = hidden + (size_t)t * (size_t)n_embd;
        edgexpu_cpu_rmsnorm(normed, xt, layer->attn_norm, n_embd, eps);
        edgexpu_cpu_linear(q + (size_t)t * (size_t)n_embd, normed, layer->wq, layer->bq, n_embd, n_embd);
        edgexpu_cpu_linear(k + (size_t)t * (size_t)n_kv * (size_t)head_dim, normed, layer->wk, layer->bk, n_kv * head_dim, n_embd);
        edgexpu_cpu_linear(v + (size_t)t * (size_t)n_kv * (size_t)head_dim, normed, layer->wv, layer->bv, n_kv * head_dim, n_embd);
        apply_rope(session, q + (size_t)t * (size_t)n_embd, n_heads, head_dim, t);
        apply_rope(session, k + (size_t)t * (size_t)n_kv * (size_t)head_dim, n_kv, head_dim, t);
    }

    for (t = 0; t < seq; t++) {
        int h;
        for (h = 0; h < n_heads; h++) {
            int kv_h = h / n_rep;
            float *qh = q + ((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)head_dim;
            float *out_h = attn + ((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)head_dim;
            int s;
            int i;
            for (s = 0; s <= t; s++) {
                float *kh = k + ((size_t)s * (size_t)n_kv + (size_t)kv_h) * (size_t)head_dim;
                float dot = 0.0f;
                for (i = 0; i < head_dim; i++) {
                    dot += qh[i] * kh[i];
                }
                scores[s] = dot * attn_scale;
            }
            edgexpu_cpu_softmax(scores, t + 1);
            memset(out_h, 0, (size_t)head_dim * sizeof(float));
            for (s = 0; s <= t; s++) {
                float *vh = v + ((size_t)s * (size_t)n_kv + (size_t)kv_h) * (size_t)head_dim;
                for (i = 0; i < head_dim; i++) {
                    out_h[i] += scores[s] * vh[i];
                }
            }
        }
    }

    for (t = 0; t < seq; t++) {
        float *k_slot = edgexpu_kv_cache_k_at(&session->kv, layer_index, t);
        float *v_slot = edgexpu_kv_cache_v_at(&session->kv, layer_index, t);
        if (k_slot == NULL || v_slot == NULL) {
            set_error(error, error_size, "native prefill 无法写入 KV cache");
            return 0;
        }
        memcpy(k_slot, k + (size_t)t * (size_t)n_kv * (size_t)head_dim, (size_t)n_kv * (size_t)head_dim * sizeof(float));
        memcpy(v_slot, v + (size_t)t * (size_t)n_kv * (size_t)head_dim, (size_t)n_kv * (size_t)head_dim * sizeof(float));
    }

    for (t = 0; t < seq; t++) {
        float *xt = hidden + (size_t)t * (size_t)n_embd;
        memcpy(residual, xt, (size_t)n_embd * sizeof(float));
        edgexpu_cpu_linear(proj, attn + (size_t)t * (size_t)n_embd, layer->wo, NULL, n_embd, n_embd);
        edgexpu_cpu_add(xt, residual, proj, n_embd);
        memcpy(residual, xt, (size_t)n_embd * sizeof(float));
        edgexpu_cpu_rmsnorm(normed, xt, layer->ffn_norm, n_embd, eps);
        edgexpu_cpu_linear(gate, normed, layer->wgate, NULL, n_ff, n_embd);
        edgexpu_cpu_linear(up, normed, layer->wup, NULL, n_ff, n_embd);
        edgexpu_cpu_silu(gate, gate, n_ff);
        edgexpu_cpu_mul(gate, gate, up, n_ff);
        edgexpu_cpu_linear(down, gate, layer->wdown, NULL, n_embd, n_ff);
        edgexpu_cpu_add(xt, residual, down, n_embd);
    }

    return 1;
}

int edgexpu_native_forward_prefill(
    edgexpu_native_session *session,
    char *error,
    size_t error_size
) {
    const edgexpu_gguf_tensor *embd;
    int seq;
    int t;
    int layer_index;
    int n_layers;
    int n_embd;
    int n_ff;
    int n_heads;
    int n_kv;
    int head_dim;
    float eps;
    float *hidden = NULL;
    float *normed = NULL;
    float *residual = NULL;
    float *q = NULL;
    float *k = NULL;
    float *v = NULL;
    float *attn = NULL;
    float *proj = NULL;
    float *gate = NULL;
    float *up = NULL;
    float *down = NULL;
    float *scores = NULL;
    int ok = 0;
    double started;

    if (session == NULL || !session->loaded || session->output_norm == NULL) {
        set_error(error, error_size, "native prefill 尚未就绪");
        return 0;
    }
    if (session->token_count <= 0) {
        set_error(error, error_size, "native prefill 需要先 tokenize");
        return 0;
    }
    if (session->kv.k == NULL) {
        set_error(error, error_size, "native KV cache 尚未就绪");
        return 0;
    }

    n_layers = (int)session->gguf.block_count;
    if (n_layers <= 0) {
        set_error(error, error_size, "native 模型 block_count 无效");
        return 0;
    }
    if (!ensure_layers_cached(session, error, error_size)) {
        return 0;
    }

    n_embd = session->layers[0].n_embd;
    n_ff = session->layers[0].n_ff;
    n_heads = session->layers[0].n_heads;
    n_kv = session->layers[0].n_kv_heads;
    head_dim = session->layers[0].head_dim;
    seq = session->token_count;
    if (seq > session->kv.max_seq) {
        seq = session->kv.max_seq;
    }
    eps = session->gguf.rms_eps > 0.0f ? session->gguf.rms_eps : 1e-6f;

    embd = token_embedding_tensor(session);
    if (embd == NULL) {
        set_error(error, error_size, "缺少 token_embd.weight");
        return 0;
    }

    hidden = (float *)calloc((size_t)seq * (size_t)n_embd, sizeof(float));
    normed = (float *)malloc((size_t)n_embd * sizeof(float));
    residual = (float *)malloc((size_t)n_embd * sizeof(float));
    q = (float *)malloc((size_t)seq * (size_t)n_heads * (size_t)head_dim * sizeof(float));
    k = (float *)malloc((size_t)seq * (size_t)n_kv * (size_t)head_dim * sizeof(float));
    v = (float *)malloc((size_t)seq * (size_t)n_kv * (size_t)head_dim * sizeof(float));
    attn = (float *)calloc((size_t)seq * (size_t)n_embd, sizeof(float));
    proj = (float *)malloc((size_t)n_embd * sizeof(float));
    gate = (float *)malloc((size_t)n_ff * sizeof(float));
    up = (float *)malloc((size_t)n_ff * sizeof(float));
    down = (float *)malloc((size_t)n_embd * sizeof(float));
    scores = (float *)malloc((size_t)seq * sizeof(float));
    if (hidden == NULL || normed == NULL || residual == NULL || q == NULL || k == NULL ||
        v == NULL || attn == NULL || proj == NULL || gate == NULL || up == NULL ||
        down == NULL || scores == NULL) {
        set_error(error, error_size, "native prefill workspace 分配失败");
        goto cleanup;
    }

    started = native_now_seconds();
    for (t = 0; t < seq; t++) {
        uint32_t token = session->token_ids[t];
        if (!load_embedding_row(session, token, n_embd, hidden + (size_t)t * (size_t)n_embd, error, error_size)) {
            goto cleanup;
        }
    }

    edgexpu_kv_cache_reset(&session->kv);
    for (layer_index = 0; layer_index < n_layers; layer_index++) {
        if (!apply_transformer_layer(
                session,
                &session->layers[layer_index],
                layer_index,
                seq,
                hidden,
                normed,
                residual,
                q,
                k,
                v,
                attn,
                proj,
                gate,
                up,
                down,
                scores,
                error,
                error_size)) {
            goto cleanup;
        }
    }

    for (t = 0; t < seq; t++) {
        float *xt = hidden + (size_t)t * (size_t)n_embd;
        memcpy(residual, xt, (size_t)n_embd * sizeof(float));
        edgexpu_cpu_rmsnorm(xt, residual, session->output_norm, n_embd, eps);
    }

    if (!edgexpu_kv_cache_extend(&session->kv, seq, error, error_size)) {
        goto cleanup;
    }

    session->prefill_layers = n_layers;
    session->prompt_token_count = seq;
    session->generated_tokens = 0;
    session->token_count = seq;
    session->prefill_seconds = native_now_seconds() - started;
    free(session->last_hidden);
    session->last_hidden = (float *)malloc((size_t)n_embd * sizeof(float));
    if (session->last_hidden == NULL) {
        set_error(error, error_size, "native last_hidden 分配失败");
        goto cleanup;
    }
    memcpy(session->last_hidden, hidden + (size_t)(seq - 1) * (size_t)n_embd, (size_t)n_embd * sizeof(float));
    session->last_hidden_rms = vector_rms(session->last_hidden, n_embd);
    if (!isfinite(session->last_hidden_rms) || session->last_hidden_rms <= 0.0f) {
        set_error(error, error_size, "native prefill 输出不是有限正 RMS");
        goto cleanup;
    }
    ok = 1;

cleanup:
    free(hidden);
    free(normed);
    free(residual);
    free(q);
    free(k);
    free(v);
    free(attn);
    free(proj);
    free(gate);
    free(up);
    free(down);
    free(scores);
    return ok;
}

int edgexpu_native_forward_layer0(
    edgexpu_native_session *session,
    char *error,
    size_t error_size
) {
    return edgexpu_native_forward_prefill(session, error, error_size);
}

static int sample_next_token(
    edgexpu_native_session *session,
    float temperature,
    uint32_t *token_id,
    char *error,
    size_t error_size
) {
    const edgexpu_gguf_tensor *weights;
    int n_embd;
    int vocab;
    uint32_t i;
    uint32_t best = 0;
    float best_score;
    float *row = NULL;

    n_embd = session->layers[0].n_embd;
    vocab = (int)session->tokenizer.vocab_size;
    weights = logit_weight_tensor(session);
    if (weights == NULL || session->last_hidden == NULL || vocab <= 0) {
        set_error(error, error_size, "native decode 无法计算 logits");
        return 0;
    }
    if (weights->type != EDGEXPU_GGUF_TYPE_Q8_0) {
        row = (float *)malloc((size_t)n_embd * sizeof(float));
        if (row == NULL) {
            set_error(error, error_size, "native logits 行缓冲分配失败");
            return 0;
        }
    }

    if (temperature > 1e-4f) {
        float *logits = (float *)malloc((size_t)vocab * sizeof(float));
        float max_value;
        double sum = 0.0;
        double cursor = 0.0;
        double pick;
        static uint64_t rng = 0x00ED6E50ULL;

        if (logits == NULL) {
            free(row);
            set_error(error, error_size, "native logits 分配失败");
            return 0;
        }
        for (i = 0; i < (uint32_t)vocab; i++) {
            logits[i] = logit_dot_row(session, weights, i, n_embd, session->last_hidden, row) / temperature;
        }
        max_value = logits[0];
        for (i = 1; i < (uint32_t)vocab; i++) {
            if (logits[i] > max_value) {
                max_value = logits[i];
            }
        }
        for (i = 0; i < (uint32_t)vocab; i++) {
            logits[i] = expf(logits[i] - max_value);
            sum += (double)logits[i];
        }
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        pick = ((double)(rng & 0xFFFFFFFu) / (double)0x10000000u) * sum;
        best = (uint32_t)(vocab - 1);
        for (i = 0; i < (uint32_t)vocab; i++) {
            cursor += (double)logits[i];
            if (cursor >= pick) {
                best = i;
                break;
            }
        }
        free(logits);
        free(row);
        *token_id = best;
        return 1;
    }

    best_score = logit_dot_row(session, weights, 0, n_embd, session->last_hidden, row);
    for (i = 1; i < (uint32_t)vocab; i++) {
        float score = logit_dot_row(session, weights, i, n_embd, session->last_hidden, row);
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
    free(row);
    *token_id = best;
    return 1;
}

static int forward_decode_token(
    edgexpu_native_session *session,
    uint32_t token,
    char *error,
    size_t error_size
) {
    const edgexpu_native_layer *layer;
    int pos;
    int layer_index;
    int n_layers;
    int n_embd;
    int n_ff;
    int n_heads;
    int n_kv;
    int head_dim;
    int n_rep;
    int h;
    int s;
    int i;
    float eps;
    float attn_scale;
    float *hidden = NULL;
    float *normed = NULL;
    float *residual = NULL;
    float *q = NULL;
    float *k = NULL;
    float *v = NULL;
    float *attn = NULL;
    float *proj = NULL;
    float *gate = NULL;
    float *up = NULL;
    float *down = NULL;
    float *scores = NULL;
    int ok = 0;

    pos = session->kv.seq_len;
    n_layers = session->n_layers_cached;
    n_embd = session->layers[0].n_embd;
    n_ff = session->layers[0].n_ff;
    n_heads = session->layers[0].n_heads;
    n_kv = session->layers[0].n_kv_heads;
    head_dim = session->layers[0].head_dim;
    n_rep = n_heads / n_kv;
    eps = session->gguf.rms_eps > 0.0f ? session->gguf.rms_eps : 1e-6f;
    attn_scale = 1.0f / sqrtf((float)head_dim);

    if (pos < 0 || pos >= session->kv.max_seq) {
        set_error(error, error_size, "native decode 超出 KV 窗口");
        return 0;
    }

    hidden = (float *)malloc((size_t)n_embd * sizeof(float));
    normed = (float *)malloc((size_t)n_embd * sizeof(float));
    residual = (float *)malloc((size_t)n_embd * sizeof(float));
    q = (float *)malloc((size_t)n_heads * (size_t)head_dim * sizeof(float));
    k = (float *)malloc((size_t)n_kv * (size_t)head_dim * sizeof(float));
    v = (float *)malloc((size_t)n_kv * (size_t)head_dim * sizeof(float));
    attn = (float *)calloc((size_t)n_embd, sizeof(float));
    proj = (float *)malloc((size_t)n_embd * sizeof(float));
    gate = (float *)malloc((size_t)n_ff * sizeof(float));
    up = (float *)malloc((size_t)n_ff * sizeof(float));
    down = (float *)malloc((size_t)n_embd * sizeof(float));
    scores = (float *)malloc((size_t)(pos + 1) * sizeof(float));
    if (hidden == NULL || normed == NULL || residual == NULL || q == NULL || k == NULL ||
        v == NULL || attn == NULL || proj == NULL || gate == NULL || up == NULL ||
        down == NULL || scores == NULL) {
        set_error(error, error_size, "native decode workspace 分配失败");
        goto cleanup;
    }
    if (!load_embedding_row(session, token, n_embd, hidden, error, error_size)) {
        goto cleanup;
    }

    for (layer_index = 0; layer_index < n_layers; layer_index++) {
        float *k_slot;
        float *v_slot;

        layer = &session->layers[layer_index];
        edgexpu_cpu_rmsnorm(normed, hidden, layer->attn_norm, n_embd, eps);
        edgexpu_cpu_linear(q, normed, layer->wq, layer->bq, n_embd, n_embd);
        edgexpu_cpu_linear(k, normed, layer->wk, layer->bk, n_kv * head_dim, n_embd);
        edgexpu_cpu_linear(v, normed, layer->wv, layer->bv, n_kv * head_dim, n_embd);
        apply_rope(session, q, n_heads, head_dim, pos);
        apply_rope(session, k, n_kv, head_dim, pos);

        k_slot = edgexpu_kv_cache_k_at(&session->kv, layer_index, pos);
        v_slot = edgexpu_kv_cache_v_at(&session->kv, layer_index, pos);
        if (k_slot == NULL || v_slot == NULL) {
            set_error(error, error_size, "native decode 无法写入 KV cache");
            goto cleanup;
        }
        memcpy(k_slot, k, (size_t)n_kv * (size_t)head_dim * sizeof(float));
        memcpy(v_slot, v, (size_t)n_kv * (size_t)head_dim * sizeof(float));

        for (h = 0; h < n_heads; h++) {
            int kv_h = h / n_rep;
            float *qh = q + (size_t)h * (size_t)head_dim;
            float *out_h = attn + (size_t)h * (size_t)head_dim;
            memset(out_h, 0, (size_t)head_dim * sizeof(float));
            for (s = 0; s <= pos; s++) {
                float *kh = edgexpu_kv_cache_k_at(&session->kv, layer_index, s);
                float dot = 0.0f;
                for (i = 0; i < head_dim; i++) {
                    dot += qh[i] * kh[i];
                }
                scores[s] = dot * attn_scale;
            }
            edgexpu_cpu_softmax(scores, pos + 1);
            for (s = 0; s <= pos; s++) {
                float *vh = edgexpu_kv_cache_v_at(&session->kv, layer_index, s);
                for (i = 0; i < head_dim; i++) {
                    out_h[i] += scores[s] * vh[i];
                }
            }
        }

        memcpy(residual, hidden, (size_t)n_embd * sizeof(float));
        edgexpu_cpu_linear(proj, attn, layer->wo, NULL, n_embd, n_embd);
        edgexpu_cpu_add(hidden, residual, proj, n_embd);
        memcpy(residual, hidden, (size_t)n_embd * sizeof(float));
        edgexpu_cpu_rmsnorm(normed, hidden, layer->ffn_norm, n_embd, eps);
        edgexpu_cpu_linear(gate, normed, layer->wgate, NULL, n_ff, n_embd);
        edgexpu_cpu_linear(up, normed, layer->wup, NULL, n_ff, n_embd);
        edgexpu_cpu_silu(gate, gate, n_ff);
        edgexpu_cpu_mul(gate, gate, up, n_ff);
        edgexpu_cpu_linear(down, gate, layer->wdown, NULL, n_embd, n_ff);
        edgexpu_cpu_add(hidden, residual, down, n_embd);
    }

    memcpy(residual, hidden, (size_t)n_embd * sizeof(float));
    edgexpu_cpu_rmsnorm(hidden, residual, session->output_norm, n_embd, eps);
    memcpy(session->last_hidden, hidden, (size_t)n_embd * sizeof(float));
    session->last_hidden_rms = vector_rms(session->last_hidden, n_embd);
    if (!edgexpu_kv_cache_extend(&session->kv, 1, error, error_size)) {
        goto cleanup;
    }
    if (session->token_count < EDGEXPU_NATIVE_MAX_TOKENS) {
        session->token_ids[session->token_count++] = token;
    }
    session->generated_tokens += 1;
    ok = 1;

cleanup:
    free(hidden);
    free(normed);
    free(residual);
    free(q);
    free(k);
    free(v);
    free(attn);
    free(proj);
    free(gate);
    free(up);
    free(down);
    free(scores);
    return ok;
}

int edgexpu_native_generate_next(
    edgexpu_native_session *session,
    float temperature,
    uint32_t *token_id,
    char *piece,
    size_t piece_size,
    int *stopped,
    char *error,
    size_t error_size
) {
    uint32_t sampled = 0;
    uint32_t eos;

    if (piece != NULL && piece_size > 0) {
        piece[0] = '\0';
    }
    if (stopped != NULL) {
        *stopped = 0;
    }
    if (session == NULL || !session->loaded || session->last_hidden == NULL ||
        session->layers == NULL || session->n_layers_cached <= 0) {
        set_error(error, error_size, "native decode 尚未就绪");
        return 0;
    }
    if (session->kv.seq_len >= session->kv.max_seq ||
        session->token_count >= EDGEXPU_NATIVE_MAX_TOKENS) {
        if (stopped != NULL) {
            *stopped = 1;
        }
        if (token_id != NULL) {
            *token_id = session->tokenizer.eos_token_id;
        }
        return 1;
    }
    if (!sample_next_token(session, temperature, &sampled, error, error_size)) {
        return 0;
    }
    if (token_id != NULL) {
        *token_id = sampled;
    }

    eos = session->tokenizer.eos_token_id;
    if (sampled == eos || (session->gguf.eos_token_id != 0 && sampled == session->gguf.eos_token_id)) {
        if (stopped != NULL) {
            *stopped = 1;
        }
        return 1;
    }

    if (piece != NULL && piece_size > 0) {
        edgexpu_tokenizer_decode(&session->tokenizer, &sampled, 1, piece, piece_size);
    }
    return forward_decode_token(session, sampled, error, error_size);
}

void edgexpu_native_print_info(const edgexpu_native_session *session) {
    uint32_t i;
    if (session == NULL) {
        return;
    }

    printf("architecture=%s\n", session->gguf.architecture);
    printf("adapter=%s qkv_bias=%d rope=%s ffn=%s tokenizer=%s\n",
           session->arch.name,
           session->arch.has_qkv_bias,
           edgexpu_rope_type_name(session->arch.rope),
           edgexpu_ffn_type_name(session->arch.ffn),
           edgexpu_tokenizer_kind_name(session->arch.tokenizer));
    printf("name=%s\n", session->gguf.name);
    printf("file_size=%llu\n", (unsigned long long)session->gguf.file_size);
    printf("block_count=%u\n", session->gguf.block_count);
    printf("context_length=%u\n", session->gguf.context_length);
    printf("embedding_length=%u\n", session->gguf.embedding_length);
    printf("feed_forward_length=%u\n", session->gguf.feed_forward_length);
    printf("head_count=%u\n", session->gguf.head_count);
    printf("head_count_kv=%u\n", session->gguf.head_count_kv);
    printf("head_dim=%d\n", edgexpu_gguf_head_dim(&session->gguf));
    printf("tensor_count=%u\n", session->gguf.n_tensors);
    printf("data_offset=%llu\n", (unsigned long long)session->gguf.data_offset);
    printf("tokenizer_model=%s\n", session->gguf.tokenizer_model);
    printf("tokenizer_pre=%s\n", session->gguf.tokenizer_pre);
    printf("chat_template_bytes=%zu\n", strlen(session->gguf.chat_template));
    printf("vocab_size=%u\n", session->tokenizer.vocab_size);
    printf("n_merges=%u\n", session->tokenizer.n_merges);
    printf("bos=%u eos=%u pad=%u add_bos=%d\n",
           session->tokenizer.bos_token_id,
           session->tokenizer.eos_token_id,
           session->tokenizer.pad_token_id,
           session->tokenizer.add_bos_token);
    printf("kv_bytes=%u seq_len=%d max_seq=%d\n",
           (unsigned)edgexpu_kv_cache_bytes(&session->kv),
           session->kv.seq_len,
           session->kv.max_seq);
    printf("layer0_ready=%d prefill_layers=%d generated_tokens=%d last_hidden_rms=%.6f\n",
           session->layer0.ready,
           session->prefill_layers,
           session->generated_tokens,
           session->last_hidden_rms);
    printf("tensors:\n");
    for (i = 0; i < session->gguf.n_tensors && i < 8; i++) {
        printf("  %s type=%u dims=%u\n",
               session->gguf.tensors[i].name,
               session->gguf.tensors[i].type,
               session->gguf.tensors[i].n_dims);
    }
}

int edgexpu_native_selftest(const char *gguf_path) {
    char error[256] = {0};
    edgexpu_native_session session;
    char decoded[256];
    const float *k0;
    const float *klast;
    int last_layer;

    if (!edgexpu_cpu_kernel_selftest(error, sizeof(error))) {
        fprintf(stderr, "native kernel selftest failed: %s\n", error);
        return 1;
    }
    {
        char formatted[64];
        if (!edgexpu_chat_apply("USER:{{prompt}}", "hi", formatted, sizeof(formatted)) ||
            strcmp(formatted, "USER:hi") != 0) {
            fprintf(stderr, "native chat template selftest failed: %s\n", formatted);
            return 1;
        }
        if (!edgexpu_chat_apply("", "raw", formatted, sizeof(formatted)) ||
            strcmp(formatted, "raw") != 0) {
            fprintf(stderr, "native empty chat template selftest failed\n");
            return 1;
        }
    }
    if (!edgexpu_kv_cache_selftest(error, sizeof(error))) {
        fprintf(stderr, "native kv selftest failed: %s\n", error);
        return 1;
    }

    if (gguf_path == NULL || gguf_path[0] == '\0') {
        printf("native selftest passed\n");
        return 0;
    }

    edgexpu_native_init(&session);
    if (!edgexpu_native_load(&session, gguf_path, error, sizeof(error))) {
        fprintf(stderr, "native GGUF load failed: %s\n", error);
        return 1;
    }
    if (!session.layer0.ready) {
        fprintf(stderr, "native layer0 was not loaded\n");
        edgexpu_native_free(&session);
        return 1;
    }
    if (!edgexpu_native_tokenize(&session, "Hello EdgeXPU", error, sizeof(error)) ||
        session.token_count <= 0) {
        fprintf(stderr, "native tokenize failed: %s\n", error);
        edgexpu_native_free(&session);
        return 1;
    }
    if (!edgexpu_native_forward_prefill(&session, error, sizeof(error))) {
        fprintf(stderr, "native prefill failed: %s\n", error);
        edgexpu_native_free(&session);
        return 1;
    }
    last_layer = (int)session.gguf.block_count - 1;
    k0 = edgexpu_kv_cache_k_at(&session.kv, 0, 0);
    klast = last_layer >= 0 ? edgexpu_kv_cache_k_at(&session.kv, last_layer, 0) : NULL;
    if (k0 == NULL || klast == NULL ||
        vector_rms(k0, session.layer0.n_kv_heads * session.layer0.head_dim) <= 0.0f ||
        vector_rms(klast, session.layer0.n_kv_heads * session.layer0.head_dim) <= 0.0f) {
        fprintf(stderr, "native prefill KV was empty\n");
        edgexpu_native_free(&session);
        return 1;
    }
    if (session.prefill_layers != (int)session.gguf.block_count) {
        fprintf(stderr, "native prefill_layers=%d expected=%u\n",
                session.prefill_layers,
                session.gguf.block_count);
        edgexpu_native_free(&session);
        return 1;
    }
    edgexpu_tokenizer_decode(
        &session.tokenizer,
        session.token_ids,
        session.token_count,
        decoded,
        sizeof(decoded)
    );
    if (strstr(decoded, "Hello") == NULL) {
        fprintf(stderr, "native decode roundtrip failed: %s\n", decoded);
        edgexpu_native_free(&session);
        return 1;
    }
    {
        char generated[256] = {0};
        size_t used = 0;
        int step;

        for (step = 0; step < 4; step++) {
            uint32_t token = 0;
            char piece[64] = {0};
            int stopped = 0;
            if (!edgexpu_native_generate_next(
                    &session,
                    0.0f,
                    &token,
                    piece,
                    sizeof(piece),
                    &stopped,
                    error,
                    sizeof(error))) {
                fprintf(stderr, "native decode step failed: %s\n", error);
                edgexpu_native_free(&session);
                return 1;
            }
            if (stopped) {
                break;
            }
            if (piece[0] != '\0' && used + strlen(piece) + 1 < sizeof(generated)) {
                memcpy(generated + used, piece, strlen(piece) + 1);
                used += strlen(piece);
            }
        }
        if (session.last_hidden == NULL ||
            !isfinite(session.last_hidden_rms) ||
            session.last_hidden_rms <= 0.0f) {
            fprintf(stderr, "native decode hidden was empty\n");
            edgexpu_native_free(&session);
            return 1;
        }

        printf("native selftest passed\n");
        printf("vocab_size=%u token_count=%d kv_seq=%d prefill_layers=%d decode_tokens=%d layer0_rms=%.6f decoded=%s generated=%s\n",
               session.tokenizer.vocab_size,
               session.prompt_token_count,
               session.kv.seq_len,
               session.prefill_layers,
               session.generated_tokens,
               session.last_hidden_rms,
               decoded,
               generated);
    }
    edgexpu_native_free(&session);
    return 0;
}
