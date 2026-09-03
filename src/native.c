#include "edgexpu/native.h"

#include "edgexpu/chat.h"
#include "edgexpu/cpu_kernel.h"
#include "edgexpu/gguf_quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* Native CPU fallback 实现。forward 形状由 session->arch 决定，不写死 Qwen2。 */

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

static int native_context_length(const edgexpu_native_session *session);
static int ensure_token_ids(edgexpu_native_session *session, int cap, char *error, size_t error_size);

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

/* GGUF 未给 freq_base 时：NORM RoPE 用 10000，Neox/Qwen 用 1e6。 */
static float default_rope_freq(const edgexpu_native_session *session) {
    if (session != NULL && session->gguf.rope_freq_base > 0.0f) {
        return session->gguf.rope_freq_base;
    }
    if (session != NULL && session->arch.rope == EDGEXPU_ROPE_NORM) {
        return 10000.0f;
    }
    return 1000000.0f;
}

/* 按 adapter.rope 选择 Neox 或 NORM，不在这里写死 Qwen。 */
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

/* 有独立 output.weight 则用它算 logits，否则 tied embedding。 */
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
    const float *hidden
) {
    const uint8_t *src = tensor_bytes(session, tensor);
    size_t row_bytes;
    if (src == NULL || tensor == NULL || hidden == NULL) {
        return 0.0f;
    }
    row_bytes = edgexpu_gguf_row_bytes(tensor->type, n_embd);
    if (row_bytes == 0) {
        return 0.0f;
    }
    return edgexpu_gguf_dot_quant(tensor->type, src + (size_t)row * row_bytes, hidden, n_embd);
}

static float logit_dot_q8(
    const edgexpu_native_session *session,
    const edgexpu_gguf_tensor *tensor,
    uint32_t row,
    int n_embd,
    const uint8_t *x_q8
) {
    const uint8_t *src = tensor_bytes(session, tensor);
    size_t row_bytes;
    if (src == NULL || tensor == NULL || x_q8 == NULL) {
        return 0.0f;
    }
    row_bytes = edgexpu_gguf_row_bytes(tensor->type, n_embd);
    if (row_bytes == 0) {
        return 0.0f;
    }
    return edgexpu_gguf_dot_quant_q8(tensor->type, src + (size_t)row * row_bytes, x_q8, n_embd);
}

static uint8_t *quantize_hidden_q8(const float *hidden, int n_embd) {
    size_t nbytes = edgexpu_gguf_q8_0_nbytes(n_embd);
    uint8_t *q8;
    if (hidden == NULL || nbytes == 0) {
        return NULL;
    }
    q8 = (uint8_t *)malloc(nbytes);
    if (q8 == NULL) {
        return NULL;
    }
    if (!edgexpu_gguf_quantize_q8_0(hidden, n_embd, q8)) {
        free(q8);
        return NULL;
    }
    return q8;
}

static float logit_score(
    const edgexpu_native_session *session,
    const edgexpu_gguf_tensor *weights,
    uint32_t row,
    int n_embd,
    const uint8_t *x_q8
) {
    float score = x_q8 != NULL
        ? logit_dot_q8(session, weights, row, n_embd, x_q8)
        : logit_dot_row(session, weights, row, n_embd, session->last_hidden);
    if (session->output_bias != NULL) {
        score += session->output_bias[row];
    }
    return score;
}

static void layer_free(edgexpu_native_layer *layer) {
    if (layer == NULL) {
        return;
    }
    free(layer->attn_norm);
    free(layer->ffn_norm);
    free(layer->bq);
    free(layer->bk);
    free(layer->bv);
    free(layer->bo);
    free(layer->bgate);
    free(layer->bup);
    free(layer->bdown);
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

static int load_named_weight_optional(
    edgexpu_native_session *session,
    const char *name,
    float **dst,
    char *error,
    size_t error_size
) {
    if (session == NULL || name == NULL || dst == NULL) {
        set_error(error, error_size, "optional weight 参数为空");
        return 0;
    }
    if (edgexpu_gguf_find_tensor(&session->gguf, name) == NULL) {
        *dst = NULL;
        return 1;
    }
    return load_named_weight(session, name, dst, error, error_size);
}

static int load_qweight(
    edgexpu_native_session *session,
    const char *name,
    int n_out,
    int n_in,
    edgexpu_qweight *weight,
    char *error,
    size_t error_size
) {
    const edgexpu_gguf_tensor *tensor;
    const uint8_t *src;
    uint64_t n;

    if (weight == NULL || n_out <= 0 || n_in <= 0) {
        set_error(error, error_size, "量化权重参数无效");
        return 0;
    }
    memset(weight, 0, sizeof(*weight));
    tensor = edgexpu_gguf_find_tensor(&session->gguf, name);
    if (tensor == NULL) {
        snprintf(error, error_size, "缺少或形状不匹配的 tensor: %s", name);
        return 0;
    }
    src = tensor_bytes(session, tensor);
    n = edgexpu_gguf_tensor_elements(tensor);
    if (src == NULL || n != (uint64_t)n_out * (uint64_t)n_in) {
        snprintf(error, error_size, "缺少或形状不匹配的 tensor: %s", name);
        return 0;
    }
    if (edgexpu_gguf_row_bytes(tensor->type, n_in) == 0) {
        snprintf(error, error_size, "不支持的量化类型: %s", name);
        return 0;
    }
    weight->data = src;
    weight->type = tensor->type;
    weight->n_out = n_out;
    weight->n_in = n_in;
    return 1;
}

static void apply_qlinear(float *out, const float *x, const edgexpu_qweight *weight, const float *bias) {
    if (weight == NULL || weight->data == NULL) {
        return;
    }
    edgexpu_cpu_linear_quant(out, x, weight->data, weight->type, bias, weight->n_out, weight->n_in);
}

static void apply_qlinear_batch(
    float *out,
    const float *x,
    const edgexpu_qweight *weight,
    const float *bias,
    int m
) {
    if (weight == NULL || weight->data == NULL || m <= 0) {
        return;
    }
    edgexpu_cpu_linear_quant_batch(
        out,
        x,
        weight->data,
        weight->type,
        bias,
        m,
        weight->n_out,
        weight->n_in
    );
}

static double native_now_seconds(void) {
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) == 0) {
        return (double)timestamp.tv_sec + (double)timestamp.tv_nsec / 1e9;
    }
    return 0.0;
}

/* 按 adapter 加载一层权重。QKV bias 仅在 has_qkv_bias 时必填。 */
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

#define LOAD_Q(field, suffix, nout, nin) \
    do { \
        snprintf(name, sizeof(name), "blk.%d.%s", layer_index, suffix); \
        if (!load_qweight(session, name, (nout), (nin), &layer->field, error, error_size)) { \
            layer_free(layer); \
            return 0; \
        } \
    } while (0)

    LOAD_BLK(attn_norm, "attn_norm.weight");
    LOAD_BLK(ffn_norm, "ffn_norm.weight");
    LOAD_Q(wq, "attn_q.weight", layer->n_embd, layer->n_embd);
    LOAD_Q(wk, "attn_k.weight", layer->n_kv_heads * layer->head_dim, layer->n_embd);
    LOAD_Q(wv, "attn_v.weight", layer->n_kv_heads * layer->head_dim, layer->n_embd);
    LOAD_Q(wo, "attn_output.weight", layer->n_embd, layer->n_embd);
    LOAD_Q(wgate, "ffn_gate.weight", layer->n_ff, layer->n_embd);
    LOAD_Q(wup, "ffn_up.weight", layer->n_ff, layer->n_embd);
    LOAD_Q(wdown, "ffn_down.weight", layer->n_embd, layer->n_ff);
    if (session->arch.has_qkv_bias) {
        LOAD_BLK(bq, "attn_q.bias");
        LOAD_BLK(bk, "attn_k.bias");
        LOAD_BLK(bv, "attn_v.bias");
    } else {
        snprintf(name, sizeof(name), "blk.%d.attn_q.bias", layer_index);
        if (!load_named_weight_optional(session, name, &layer->bq, error, error_size)) {
            layer_free(layer);
            return 0;
        }
        snprintf(name, sizeof(name), "blk.%d.attn_k.bias", layer_index);
        if (!load_named_weight_optional(session, name, &layer->bk, error, error_size)) {
            layer_free(layer);
            return 0;
        }
        snprintf(name, sizeof(name), "blk.%d.attn_v.bias", layer_index);
        if (!load_named_weight_optional(session, name, &layer->bv, error, error_size)) {
            layer_free(layer);
            return 0;
        }
    }
    snprintf(name, sizeof(name), "blk.%d.attn_output.bias", layer_index);
    if (!load_named_weight_optional(session, name, &layer->bo, error, error_size)) {
        layer_free(layer);
        return 0;
    }
    snprintf(name, sizeof(name), "blk.%d.ffn_gate.bias", layer_index);
    if (!load_named_weight_optional(session, name, &layer->bgate, error, error_size)) {
        layer_free(layer);
        return 0;
    }
    snprintf(name, sizeof(name), "blk.%d.ffn_up.bias", layer_index);
    if (!load_named_weight_optional(session, name, &layer->bup, error, error_size)) {
        layer_free(layer);
        return 0;
    }
    snprintf(name, sizeof(name), "blk.%d.ffn_down.bias", layer_index);
    if (!load_named_weight_optional(session, name, &layer->bdown, error, error_size)) {
        layer_free(layer);
        return 0;
    }
#undef LOAD_BLK
#undef LOAD_Q

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

typedef struct native_decode_scratch {
    float *hidden;
    float *normed;
    float *residual;
    float *q;
    float *k;
    float *v;
    float *attn;
    float *proj;
    float *gate;
    float *up;
    float *down;
    float *scores;
    int n_embd;
    int n_ff;
    int n_heads;
    int n_kv;
    int head_dim;
    int cap_seq;
} native_decode_scratch;

static void free_decode_scratch(edgexpu_native_session *session) {
    native_decode_scratch *ws;
    if (session == NULL || session->scratch == NULL) {
        return;
    }
    ws = (native_decode_scratch *)session->scratch;
    free(ws->hidden);
    free(ws->normed);
    free(ws->residual);
    free(ws->q);
    free(ws->k);
    free(ws->v);
    free(ws->attn);
    free(ws->proj);
    free(ws->gate);
    free(ws->up);
    free(ws->down);
    free(ws->scores);
    free(ws);
    session->scratch = NULL;
}

static int ensure_decode_scratch(
    edgexpu_native_session *session,
    int cap_seq,
    char *error,
    size_t error_size
) {
    native_decode_scratch *ws;
    int n_embd;
    int n_ff;
    int n_heads;
    int n_kv;
    int head_dim;

    if (session == NULL || session->layers == NULL) {
        set_error(error, error_size, "native decode workspace 尚未就绪");
        return 0;
    }
    n_embd = session->layers[0].n_embd;
    n_ff = session->layers[0].n_ff;
    n_heads = session->layers[0].n_heads;
    n_kv = session->layers[0].n_kv_heads;
    head_dim = session->layers[0].head_dim;
    if (cap_seq < 1) {
        cap_seq = 1;
    }
    ws = (native_decode_scratch *)session->scratch;
    if (ws != NULL &&
        ws->n_embd == n_embd &&
        ws->n_ff == n_ff &&
        ws->n_heads == n_heads &&
        ws->n_kv == n_kv &&
        ws->head_dim == head_dim &&
        ws->cap_seq >= cap_seq) {
        return 1;
    }
    free_decode_scratch(session);
    ws = (native_decode_scratch *)calloc(1, sizeof(*ws));
    if (ws == NULL) {
        set_error(error, error_size, "native decode workspace 分配失败");
        return 0;
    }
    ws->n_embd = n_embd;
    ws->n_ff = n_ff;
    ws->n_heads = n_heads;
    ws->n_kv = n_kv;
    ws->head_dim = head_dim;
    ws->cap_seq = cap_seq;
    ws->hidden = (float *)malloc((size_t)n_embd * sizeof(float));
    ws->normed = (float *)malloc((size_t)n_embd * sizeof(float));
    ws->residual = (float *)malloc((size_t)n_embd * sizeof(float));
    ws->q = (float *)malloc((size_t)n_heads * (size_t)head_dim * sizeof(float));
    ws->k = (float *)malloc((size_t)n_kv * (size_t)head_dim * sizeof(float));
    ws->v = (float *)malloc((size_t)n_kv * (size_t)head_dim * sizeof(float));
    ws->attn = (float *)malloc((size_t)n_embd * sizeof(float));
    ws->proj = (float *)malloc((size_t)n_embd * sizeof(float));
    ws->gate = (float *)malloc((size_t)n_ff * sizeof(float));
    ws->up = (float *)malloc((size_t)n_ff * sizeof(float));
    ws->down = (float *)malloc((size_t)n_embd * sizeof(float));
    ws->scores = (float *)malloc((size_t)cap_seq * sizeof(float));
    if (ws->hidden == NULL || ws->normed == NULL || ws->residual == NULL ||
        ws->q == NULL || ws->k == NULL || ws->v == NULL || ws->attn == NULL ||
        ws->proj == NULL || ws->gate == NULL || ws->up == NULL ||
        ws->down == NULL || ws->scores == NULL) {
        session->scratch = ws;
        free_decode_scratch(session);
        set_error(error, error_size, "native decode workspace 分配失败");
        return 0;
    }
    session->scratch = ws;
    return 1;
}

void edgexpu_native_free(edgexpu_native_session *session) {
    if (session == NULL) {
        return;
    }
    free_decode_scratch(session);
    free_cached_layers(session);
    free(session->output_norm);
    session->output_norm = NULL;
    free(session->output_bias);
    session->output_bias = NULL;
    free(session->last_hidden);
    session->last_hidden = NULL;
    free(session->token_ids);
    session->token_ids = NULL;
    session->token_ids_cap = 0;
    edgexpu_tokenizer_free(&session->tokenizer);
    edgexpu_kv_cache_free(&session->kv);
    native_unmap(session);
    memset(session, 0, sizeof(*session));
    session->file_fd = -1;
}

size_t edgexpu_native_memory_bytes(const edgexpu_native_session *session) {
    size_t bytes;

    if (session == NULL || !session->loaded) {
        return 0;
    }
    bytes = session->file_map_size + edgexpu_kv_cache_bytes(&session->kv);
    if (session->last_hidden != NULL && session->gguf.embedding_length > 0) {
        bytes += (size_t)session->gguf.embedding_length * sizeof(float);
    }
    return bytes;
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
    if (!load_named_weight_optional(session, "output.bias", &session->output_bias, error, error_size)) {
        edgexpu_native_free(session);
        return 0;
    }

    session->loaded = 1;
    if (!ensure_token_ids(session, native_context_length(session), error, error_size)) {
        edgexpu_native_free(session);
        return 0;
    }
    return 1;
}

int edgexpu_native_tokenize(
    edgexpu_native_session *session,
    const char *text,
    char *error,
    size_t error_size
) {
    int context;
    int cap;
    if (session == NULL || !session->loaded) {
        set_error(error, error_size, "native session 尚未加载");
        return 0;
    }
    context = native_context_length(session);
    cap = session->token_ids_cap > 0 ? session->token_ids_cap : context;
    if (!ensure_token_ids(session, cap < context ? context : cap, error, error_size)) {
        return 0;
    }
    if (!edgexpu_tokenizer_encode(
            &session->tokenizer,
            text,
            session->token_ids,
            session->token_ids_cap,
            &session->token_count,
            error,
            error_size)) {
        if (error != NULL && error_size > 0 && strstr(error, "输出过长") != NULL) {
            snprintf(error, error_size, "prompt 超出模型 context_length=%d", context);
        }
        return 0;
    }
    if (session->token_count > context) {
        snprintf(error, error_size, "prompt 长度 %d 超出模型 context_length=%d", session->token_count, context);
        return 0;
    }
    return 1;
}

static int native_context_length(const edgexpu_native_session *session) {
    if (session != NULL && session->gguf.context_length > 0) {
        return (int)session->gguf.context_length;
    }
    return EDGEXPU_KV_DEFAULT_MAX_SEQ;
}

static int ensure_token_ids(edgexpu_native_session *session, int cap, char *error, size_t error_size) {
    uint32_t *ids;
    if (session == NULL || cap <= 0) {
        set_error(error, error_size, "token_ids 容量无效");
        return 0;
    }
    if (session->token_ids != NULL && session->token_ids_cap >= cap) {
        return 1;
    }
    ids = (uint32_t *)realloc(session->token_ids, (size_t)cap * sizeof(uint32_t));
    if (ids == NULL) {
        set_error(error, error_size, "token_ids 分配失败");
        return 0;
    }
    if (session->token_ids == NULL) {
        memset(ids, 0, (size_t)cap * sizeof(uint32_t));
    } else if (cap > session->token_ids_cap) {
        memset(ids + session->token_ids_cap, 0, (size_t)(cap - session->token_ids_cap) * sizeof(uint32_t));
    }
    session->token_ids = ids;
    session->token_ids_cap = cap;
    return 1;
}

int edgexpu_native_ensure_window(
    edgexpu_native_session *session,
    int n_prompt,
    int n_new,
    char *error,
    size_t error_size
) {
    int context;
    int needed;
    int n_layers;
    int n_kv;
    int head_dim;

    if (session == NULL || !session->loaded) {
        set_error(error, error_size, "native session 尚未加载");
        return 0;
    }
    if (n_prompt < 0 || n_new < 0) {
        set_error(error, error_size, "KV 窗口参数无效");
        return 0;
    }

    context = native_context_length(session);
    if (n_prompt > context) {
        snprintf(error, error_size, "prompt 长度 %d 超出模型 context_length=%d", n_prompt, context);
        return 0;
    }
    needed = n_prompt + n_new;
    if (needed > context) {
        snprintf(
            error,
            error_size,
            "prompt (%d) + max_tokens (%d) 超出模型 context_length=%d",
            n_prompt,
            n_new,
            context
        );
        return 0;
    }
    if (needed < 1) {
        needed = 1;
    }
    if (!ensure_token_ids(session, context, error, error_size)) {
        return 0;
    }

    n_layers = (int)session->gguf.block_count;
    n_kv = (int)session->gguf.head_count_kv;
    head_dim = edgexpu_gguf_head_dim(&session->gguf);
    if (n_layers <= 0 || n_kv <= 0 || head_dim <= 0) {
        set_error(error, error_size, "native KV 架构参数无效");
        return 0;
    }

    if (session->kv.k != NULL && session->kv.max_seq >= needed) {
        return 1;
    }
    if (session->kv.seq_len > 0 && session->kv.k != NULL && session->kv.max_seq < needed) {
        snprintf(
            error,
            error_size,
            "KV 窗口 %d 不足（需要 %d），已占用 cache 无法扩容",
            session->kv.max_seq,
            needed
        );
        return 0;
    }

    return edgexpu_kv_cache_allocate(&session->kv, n_layers, n_kv, head_dim, needed, error, error_size);
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

/* 一层：RMSNorm → QKV(+RoPE) → GQA attention → 残差 → SwiGLU FFN。
 * prefill 把同一线性层的全部 token 一次算完，让量化权重行留在 cache 里。 */
static int apply_transformer_layer(
    edgexpu_native_session *session,
    const edgexpu_native_layer *layer,
    int layer_index,
    int seq,
    float *hidden,
    float *normed,
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
        edgexpu_cpu_rmsnorm(
            normed + (size_t)t * (size_t)n_embd,
            hidden + (size_t)t * (size_t)n_embd,
            layer->attn_norm,
            n_embd,
            eps
        );
    }
    apply_qlinear_batch(q, normed, &layer->wq, layer->bq, seq);
    apply_qlinear_batch(k, normed, &layer->wk, layer->bk, seq);
    apply_qlinear_batch(v, normed, &layer->wv, layer->bv, seq);
    for (t = 0; t < seq; t++) {
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
            for (s = 0; s <= t; s++) {
                float *kh = k + ((size_t)s * (size_t)n_kv + (size_t)kv_h) * (size_t)head_dim;
                scores[s] = edgexpu_cpu_dot(qh, kh, head_dim) * attn_scale;
            }
            edgexpu_cpu_softmax(scores, t + 1);
            memset(out_h, 0, (size_t)head_dim * sizeof(float));
            for (s = 0; s <= t; s++) {
                float *vh = v + ((size_t)s * (size_t)n_kv + (size_t)kv_h) * (size_t)head_dim;
                edgexpu_cpu_saxpy(out_h, vh, scores[s], head_dim);
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

    apply_qlinear_batch(proj, attn, &layer->wo, layer->bo, seq);
    for (t = 0; t < seq; t++) {
        float *xt = hidden + (size_t)t * (size_t)n_embd;
        edgexpu_cpu_add(xt, xt, proj + (size_t)t * (size_t)n_embd, n_embd);
        edgexpu_cpu_rmsnorm(
            normed + (size_t)t * (size_t)n_embd,
            xt,
            layer->ffn_norm,
            n_embd,
            eps
        );
    }
    apply_qlinear_batch(gate, normed, &layer->wgate, layer->bgate, seq);
    apply_qlinear_batch(up, normed, &layer->wup, layer->bup, seq);
    for (t = 0; t < seq; t++) {
        float *g = gate + (size_t)t * (size_t)n_ff;
        float *u = up + (size_t)t * (size_t)n_ff;
        edgexpu_cpu_silu(g, g, n_ff);
        edgexpu_cpu_mul(g, g, u, n_ff);
    }
    apply_qlinear_batch(down, gate, &layer->wdown, layer->bdown, seq);
    for (t = 0; t < seq; t++) {
        float *xt = hidden + (size_t)t * (size_t)n_embd;
        edgexpu_cpu_add(xt, xt, down + (size_t)t * (size_t)n_embd, n_embd);
    }

    return 1;
}

/* 全层 prefill：按 token 取 embedding，逐层写入 KV，最后做 output_norm。 */
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
        snprintf(
            error,
            error_size,
            "prefill 序列 %d 超出 KV 窗口 %d；请按 prompt+max_tokens 预留",
            seq,
            session->kv.max_seq
        );
        return 0;
    }
    eps = session->gguf.rms_eps > 0.0f ? session->gguf.rms_eps : 1e-6f;

    embd = token_embedding_tensor(session);
    if (embd == NULL) {
        set_error(error, error_size, "缺少 token_embd.weight");
        return 0;
    }

    hidden = (float *)calloc((size_t)seq * (size_t)n_embd, sizeof(float));
    normed = (float *)malloc((size_t)seq * (size_t)n_embd * sizeof(float));
    residual = (float *)malloc((size_t)n_embd * sizeof(float));
    q = (float *)malloc((size_t)seq * (size_t)n_heads * (size_t)head_dim * sizeof(float));
    k = (float *)malloc((size_t)seq * (size_t)n_kv * (size_t)head_dim * sizeof(float));
    v = (float *)malloc((size_t)seq * (size_t)n_kv * (size_t)head_dim * sizeof(float));
    attn = (float *)calloc((size_t)seq * (size_t)n_embd, sizeof(float));
    proj = (float *)malloc((size_t)seq * (size_t)n_embd * sizeof(float));
    gate = (float *)malloc((size_t)seq * (size_t)n_ff * sizeof(float));
    up = (float *)malloc((size_t)seq * (size_t)n_ff * sizeof(float));
    down = (float *)malloc((size_t)seq * (size_t)n_embd * sizeof(float));
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

/* temperature≈0 为 greedy：只扫 argmax。否则 softmax，可选 top_p nucleus。 */
static int sample_next_token(
    edgexpu_native_session *session,
    float temperature,
    float top_p,
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
    uint8_t *x_q8 = NULL;
    static uint64_t rng = 0x00ED6E50ULL;

    n_embd = session->layers[0].n_embd;
    vocab = (int)session->tokenizer.vocab_size;
    weights = logit_weight_tensor(session);
    if (weights == NULL || session->last_hidden == NULL || vocab <= 0) {
        set_error(error, error_size, "native decode 无法计算 logits");
        return 0;
    }
    if (edgexpu_gguf_can_dot_q8(weights->type)) {
        x_q8 = quantize_hidden_q8(session->last_hidden, n_embd);
    }

    if (temperature > 1e-4f) {
        float *logits = (float *)malloc((size_t)vocab * sizeof(float));
        if (logits == NULL) {
            free(x_q8);
            set_error(error, error_size, "native logits 分配失败");
            return 0;
        }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(vocab >= 256)
#endif
        for (i = 0; i < (uint32_t)vocab; i++) {
            logits[i] = logit_score(session, weights, i, n_embd, x_q8) / temperature;
        }
        if (!edgexpu_cpu_sample_softmax(logits, vocab, top_p, &rng, &best)) {
            free(logits);
            free(x_q8);
            set_error(error, error_size, "native softmax 采样失败");
            return 0;
        }
        free(logits);
        free(x_q8);
        *token_id = best;
        return 1;
    }

    best_score = -INFINITY;
    best = 0;
#if defined(_OPENMP)
#pragma omp parallel if(vocab >= 256)
    {
        uint32_t local_best = 0;
        float local_score = -INFINITY;
        uint32_t i_local;
#pragma omp for nowait schedule(static)
        for (i_local = 0; i_local < (uint32_t)vocab; i_local++) {
            float score = logit_score(session, weights, i_local, n_embd, x_q8);
            if (score > local_score) {
                local_score = score;
                local_best = i_local;
            }
        }
#pragma omp critical
        {
            if (local_score > best_score || (local_score == best_score && local_best < best)) {
                best_score = local_score;
                best = local_best;
            }
        }
    }
#else
    for (i = 0; i < (uint32_t)vocab; i++) {
        float score = logit_score(session, weights, i, n_embd, x_q8);
        if (score > best_score) {
            best_score = score;
            best = i;
        }
    }
#endif
    free(x_q8);
    *token_id = best;
    return 1;
}

/* 把新 token 的 embedding 跑完全部层，写入当前位置的 KV。 */
static int forward_decode_token(
    edgexpu_native_session *session,
    uint32_t token,
    char *error,
    size_t error_size
) {
    native_decode_scratch *ws;
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
    float eps;
    float attn_scale;
    float *hidden;
    float *normed;
    float *residual;
    float *q;
    float *k;
    float *v;
    float *attn;
    float *proj;
    float *gate;
    float *up;
    float *down;
    float *scores;

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
    if (!ensure_decode_scratch(session, session->kv.max_seq, error, error_size)) {
        return 0;
    }
    ws = (native_decode_scratch *)session->scratch;
    hidden = ws->hidden;
    normed = ws->normed;
    residual = ws->residual;
    q = ws->q;
    k = ws->k;
    v = ws->v;
    attn = ws->attn;
    proj = ws->proj;
    gate = ws->gate;
    up = ws->up;
    down = ws->down;
    scores = ws->scores;

    if (!load_embedding_row(session, token, n_embd, hidden, error, error_size)) {
        return 0;
    }

    for (layer_index = 0; layer_index < n_layers; layer_index++) {
        float *k_slot;
        float *v_slot;

        layer = &session->layers[layer_index];
        edgexpu_cpu_rmsnorm(normed, hidden, layer->attn_norm, n_embd, eps);
        apply_qlinear(q, normed, &layer->wq, layer->bq);
        apply_qlinear(k, normed, &layer->wk, layer->bk);
        apply_qlinear(v, normed, &layer->wv, layer->bv);
        apply_rope(session, q, n_heads, head_dim, pos);
        apply_rope(session, k, n_kv, head_dim, pos);

        k_slot = edgexpu_kv_cache_k_at(&session->kv, layer_index, pos);
        v_slot = edgexpu_kv_cache_v_at(&session->kv, layer_index, pos);
        if (k_slot == NULL || v_slot == NULL) {
            set_error(error, error_size, "native decode 无法写入 KV cache");
            return 0;
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
                if (kh == NULL) {
                    set_error(error, error_size, "native decode 无法读取 K");
                    return 0;
                }
                kh += (size_t)kv_h * (size_t)head_dim;
                scores[s] = edgexpu_cpu_dot(qh, kh, head_dim) * attn_scale;
            }
            edgexpu_cpu_softmax(scores, pos + 1);
            for (s = 0; s <= pos; s++) {
                float *vh = edgexpu_kv_cache_v_at(&session->kv, layer_index, s);
                if (vh == NULL) {
                    set_error(error, error_size, "native decode 无法读取 V");
                    return 0;
                }
                vh += (size_t)kv_h * (size_t)head_dim;
                edgexpu_cpu_saxpy(out_h, vh, scores[s], head_dim);
            }
        }

        memcpy(residual, hidden, (size_t)n_embd * sizeof(float));
        apply_qlinear(proj, attn, &layer->wo, layer->bo);
        edgexpu_cpu_add(hidden, residual, proj, n_embd);
        memcpy(residual, hidden, (size_t)n_embd * sizeof(float));
        edgexpu_cpu_rmsnorm(normed, hidden, layer->ffn_norm, n_embd, eps);
        apply_qlinear(gate, normed, &layer->wgate, layer->bgate);
        apply_qlinear(up, normed, &layer->wup, layer->bup);
        edgexpu_cpu_silu(gate, gate, n_ff);
        edgexpu_cpu_mul(gate, gate, up, n_ff);
        apply_qlinear(down, gate, &layer->wdown, layer->bdown);
        edgexpu_cpu_add(hidden, residual, down, n_embd);
    }

    memcpy(residual, hidden, (size_t)n_embd * sizeof(float));
    edgexpu_cpu_rmsnorm(hidden, residual, session->output_norm, n_embd, eps);
    memcpy(session->last_hidden, hidden, (size_t)n_embd * sizeof(float));
    session->last_hidden_rms = vector_rms(session->last_hidden, n_embd);
    if (!edgexpu_kv_cache_extend(&session->kv, 1, error, error_size)) {
        return 0;
    }
    if (session->token_count >= session->token_ids_cap) {
        set_error(error, error_size, "native decode token_ids 超出容量");
        return 0;
    }
    session->token_ids[session->token_count++] = token;
    session->generated_tokens += 1;
    return 1;
}

int edgexpu_native_generate_next(
    edgexpu_native_session *session,
    float temperature,
    float top_p,
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
        session->token_count >= session->token_ids_cap) {
        snprintf(
            error,
            error_size,
            "decode 超出 KV 窗口（seq=%d max_seq=%d tokens=%d cap=%d）",
            session->kv.seq_len,
            session->kv.max_seq,
            session->token_count,
            session->token_ids_cap
        );
        return 0;
    }
    if (!sample_next_token(session, temperature, top_p, &sampled, error, error_size)) {
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

static void print_float_prefix(const char *key, const float *x, int n, int count) {
    int i;
    int m = n < count ? n : count;
    printf("%s=", key);
    for (i = 0; i < m; i++) {
        printf("%s%.6g", i == 0 ? "" : ",", x[i]);
    }
    printf("\n");
}

/* dump-logits 单行输出：换行等控制字符写成 \n，避免 piece/greedy_text 把日志拆碎。 */
static void print_escaped(const char *s) {
    if (s == NULL) {
        return;
    }
    while (*s != '\0') {
        unsigned char c = (unsigned char)*s++;
        if (c == '\\') {
            fputs("\\\\", stdout);
        } else if (c == '\n') {
            fputs("\\n", stdout);
        } else if (c == '\r') {
            fputs("\\r", stdout);
        } else if (c == '\t') {
            fputs("\\t", stdout);
        } else if (c < 32u) {
            printf("\\x%02x", c);
        } else {
            putchar((int)c);
        }
    }
}

int edgexpu_native_dump_logits(
    edgexpu_native_session *session,
    const char *prompt,
    int greedy_n,
    int top_k,
    char *error,
    size_t error_size
) {
    const edgexpu_gguf_tensor *weights;
    const edgexpu_native_layer *layer;
    float *emb = NULL;
    float *normed = NULL;
    float *q = NULL;
    float *k = NULL;
    uint32_t *top_ids = NULL;
    float *top_scores = NULL;
    uint8_t *x_q8 = NULL;
    int n_embd;
    int n_kv;
    int head_dim;
    int vocab;
    int pos;
    int i;
    int t;
    int ok = 0;

    if (session == NULL || !session->loaded) {
        set_error(error, error_size, "native dump 尚未加载模型");
        return 0;
    }
    if (greedy_n < 0) {
        greedy_n = 0;
    }
    if (greedy_n > 32) {
        greedy_n = 32;
    }
    if (top_k <= 0) {
        top_k = 8;
    }
    if (top_k > 32) {
        top_k = 32;
    }

    if (!edgexpu_native_tokenize(session, prompt != NULL ? prompt : "", error, error_size)) {
        return 0;
    }
    if (!edgexpu_native_ensure_window(session, session->token_count, greedy_n, error, error_size)) {
        return 0;
    }

    n_embd = session->layer0.n_embd;
    n_kv = session->layer0.n_kv_heads;
    head_dim = session->layer0.head_dim;
    layer = &session->layer0;
    pos = session->token_count > 0 ? session->token_count - 1 : 0;
    weights = logit_weight_tensor(session);

    printf("prompt=%s\n", prompt != NULL ? prompt : "");
    printf("architecture=%s adapter=%s tokenizer_pre=%s\n",
           session->gguf.architecture,
           session->arch.name,
           session->tokenizer.pre);
    printf("rope=%s rope_freq_base=%.1f rms_eps=%.8g qkv_bias=%d\n",
           edgexpu_rope_type_name(session->arch.rope),
           default_rope_freq(session),
           session->gguf.rms_eps > 0.0f ? session->gguf.rms_eps : 1e-6f,
           session->arch.has_qkv_bias);
    printf("tied_output=%d output_type=%u output_bias=%d\n",
           edgexpu_gguf_find_tensor(&session->gguf, "output.weight") == NULL ? 1 : 0,
           weights != NULL ? weights->type : 0u,
           session->output_bias != NULL ? 1 : 0);
    printf("token_count=%d\n", session->token_count);
    printf("token_ids=");
    for (i = 0; i < session->token_count; i++) {
        printf("%s%u", i == 0 ? "" : ",", session->token_ids[i]);
    }
    printf("\n");

    emb = (float *)malloc((size_t)n_embd * sizeof(float));
    normed = (float *)malloc((size_t)n_embd * sizeof(float));
    q = (float *)malloc((size_t)n_embd * sizeof(float));
    k = (float *)malloc((size_t)n_kv * (size_t)head_dim * sizeof(float));
    if (emb == NULL || normed == NULL || q == NULL || k == NULL || session->token_count <= 0) {
        set_error(error, error_size, "native dump layer0 缓冲分配失败");
        goto cleanup;
    }
    if (!load_embedding_row(session, session->token_ids[pos], n_embd, emb, error, error_size)) {
        goto cleanup;
    }
    printf("emb_last_rms=%.6f\n", vector_rms(emb, n_embd));
    print_float_prefix("emb_last", emb, n_embd, 8);
    edgexpu_cpu_rmsnorm(normed, emb, layer->attn_norm, n_embd,
                        session->gguf.rms_eps > 0.0f ? session->gguf.rms_eps : 1e-6f);
    printf("layer0_rms=%.6f\n", vector_rms(normed, n_embd));
    print_float_prefix("layer0_rms_vec", normed, n_embd, 8);
    apply_qlinear(q, normed, &layer->wq, layer->bq);
    apply_qlinear(k, normed, &layer->wk, layer->bk);
    printf("layer0_q_pre_rope_rms=%.6f\n", vector_rms(q, n_embd));
    printf("layer0_k_pre_rope_rms=%.6f\n", vector_rms(k, n_kv * head_dim));
    apply_rope(session, q, layer->n_heads, head_dim, pos);
    apply_rope(session, k, n_kv, head_dim, pos);
    printf("layer0_q_rms=%.6f\n", vector_rms(q, n_embd));
    printf("layer0_k_rms=%.6f\n", vector_rms(k, n_kv * head_dim));
    print_float_prefix("layer0_q", q, n_embd, 8);
    print_float_prefix("layer0_k", k, n_kv * head_dim, 8);

    if (!edgexpu_native_forward_prefill(session, error, error_size)) {
        goto cleanup;
    }
    printf("last_hidden_rms=%.6f\n", session->last_hidden_rms);
    print_float_prefix("last_hidden", session->last_hidden, n_embd, 8);

    vocab = (int)session->tokenizer.vocab_size;
    if (weights == NULL || vocab <= 0) {
        set_error(error, error_size, "native dump 无法计算 logits");
        goto cleanup;
    }
    if (edgexpu_gguf_can_dot_q8(weights->type)) {
        x_q8 = quantize_hidden_q8(session->last_hidden, n_embd);
    }
    top_ids = (uint32_t *)calloc((size_t)top_k, sizeof(uint32_t));
    top_scores = (float *)malloc((size_t)top_k * sizeof(float));
    if (top_ids == NULL || top_scores == NULL) {
        set_error(error, error_size, "native dump top-k 分配失败");
        goto cleanup;
    }
    for (i = 0; i < top_k; i++) {
        top_scores[i] = -1.0e30f;
    }
    for (i = 0; i < vocab; i++) {
        float score = logit_score(session, weights, (uint32_t)i, n_embd, x_q8);
        int slot;
        slot = top_k - 1;
        if (score <= top_scores[slot]) {
            continue;
        }
        while (slot > 0 && score > top_scores[slot - 1]) {
            top_scores[slot] = top_scores[slot - 1];
            top_ids[slot] = top_ids[slot - 1];
            slot--;
        }
        top_scores[slot] = score;
        top_ids[slot] = (uint32_t)i;
    }
    printf("logits_topk=%d\n", top_k);
    for (i = 0; i < top_k; i++) {
        char piece[64];
        edgexpu_tokenizer_decode(&session->tokenizer, &top_ids[i], 1, piece, sizeof(piece));
        printf("  rank=%d id=%u logit=%.6f piece=", i, top_ids[i], top_scores[i]);
        print_escaped(piece);
        printf("\n");
    }

    printf("greedy_n=%d\n", greedy_n);
    printf("greedy_ids=");
    for (t = 0; t < greedy_n; t++) {
        uint32_t token = 0;
        char piece[64];
        int stopped = 0;
        if (!edgexpu_native_generate_next(session, 0.0f, 1.0f, &token, piece, sizeof(piece), &stopped, error, error_size)) {
            goto cleanup;
        }
        printf("%s%u", t == 0 ? "" : ",", token);
        if (stopped) {
            greedy_n = t + 1;
            break;
        }
    }
    printf("\n");
    {
        char greedy_text[512];
        int gen = session->generated_tokens;
        int start = session->token_count - gen;
        if (start < 0) {
            start = 0;
        }
        edgexpu_tokenizer_decode(
            &session->tokenizer,
            session->token_ids + start,
            gen,
            greedy_text,
            sizeof(greedy_text)
        );
        printf("greedy_text=");
        print_escaped(greedy_text);
        printf("\n");
    }
    ok = 1;

cleanup:
    free(emb);
    free(normed);
    free(q);
    free(k);
    free(top_ids);
    free(top_scores);
    free(x_q8);
    return ok;
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
    printf("simd=%s\n", edgexpu_cpu_simd_name());
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
        {
            edgexpu_chat_message turns[3];
            char out[256];
            memset(turns, 0, sizeof(turns));
            turns[0].role = "system";
            turns[0].content = "sys";
            turns[1].role = "user";
            turns[1].content = "hi";
            turns[2].role = "assistant";
            turns[2].content = "hello";
            if (!edgexpu_chat_apply_conversation(
                    "{{#system}}S:{{content}}\n{{/system}}{{#message}}{{role}}:{{content}}\n{{/message}}A:",
                    turns,
                    3,
                    out,
                    sizeof(out)) ||
                strcmp(out, "S:sys\nuser:hi\nassistant:hello\nA:") != 0) {
                fprintf(stderr, "native chat conversation selftest failed: %s\n", out);
                return 1;
            }
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
    {
        char overflow[256] = {0};
        int ctx = (int)session.gguf.context_length;
        if (ctx > 0 &&
            edgexpu_native_ensure_window(
                &session,
                session.token_count,
                ctx,
                overflow,
                sizeof(overflow))) {
            fprintf(stderr, "native KV overflow unexpectedly succeeded\n");
            edgexpu_native_free(&session);
            return 1;
        }
        if (ctx > 0 && strstr(overflow, "context_length") == NULL) {
            fprintf(stderr, "native KV overflow missing context_length: %s\n", overflow);
            edgexpu_native_free(&session);
            return 1;
        }
        printf("kv_overflow=ok context_length=%u\n", session.gguf.context_length);
    }
    if (!edgexpu_native_ensure_window(&session, session.token_count, 4, error, sizeof(error))) {
        fprintf(stderr, "native KV window failed: %s\n", error);
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
                    1.0f,
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
