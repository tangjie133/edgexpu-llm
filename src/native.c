#include "edgexpu/native.h"

#include "edgexpu/chat.h"
#include "edgexpu/cpu_kernel.h"
#include "edgexpu/gguf_quant.h"
#include "edgexpu/profiler.h"
#include "edgexpu/scheduler.h"

#include <math.h>
#include <stdint.h>
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

static const edgexpu_arch_tensor_names *session_tensors(const edgexpu_native_session *session) {
    if (session != NULL && session->arch.tensors != NULL) {
        return session->arch.tensors;
    }
    return edgexpu_arch_tensors_llama_gguf();
}

static int session_attn_head_dim(const edgexpu_native_session *session) {
    int dim;
    int i;
    if (session == NULL) {
        return 0;
    }
    dim = edgexpu_gguf_attn_head_dim(&session->gguf);
    if (session->layer0.ready && session->layer0.head_dim > dim) {
        dim = session->layer0.head_dim;
    }
    if (session->layers != NULL) {
        for (i = 0; i < session->n_layers_cached; i++) {
            if (session->layers[i].head_dim > dim) {
                dim = session->layers[i].head_dim;
            }
        }
    }
    return dim;
}

static int session_attn_inner(const edgexpu_native_session *session) {
    int n_heads;
    int inner;
    int n_embd;
    if (session == NULL) {
        return 0;
    }
    n_heads = session->gguf.head_count > 0 ? (int)session->gguf.head_count : 1;
    inner = n_heads * session_attn_head_dim(session);
    n_embd = (int)session->gguf.embedding_length;
    return inner > n_embd ? inner : n_embd;
}

static int admit_session_window(
    const edgexpu_native_session *session,
    int window,
    char *error,
    size_t error_size
) {
    edgexpu_device_profile profile;
    edgexpu_resource_plan plan;

    memset(&profile, 0, sizeof(profile));
    (void)edgexpu_profile_device(&profile);
    edgexpu_scheduler_estimate_gguf(&session->gguf, &profile, window, &plan);
    return edgexpu_scheduler_admit(&plan, error, error_size);
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

static size_t native_page_size(void) {
#if defined(_WIN32)
    return 4096u;
#else
    long page = sysconf(_SC_PAGESIZE);
    return page > 0 ? (size_t)page : 4096u;
#endif
}

static void native_page_advise(const void *ptr, size_t nbytes, int willneed) {
#if !defined(_WIN32)
    size_t page;
    uintptr_t start;
    uintptr_t end;
    if (ptr == NULL || nbytes == 0) {
        return;
    }
    page = native_page_size();
    start = (uintptr_t)ptr & ~(uintptr_t)(page - 1u);
    end = ((uintptr_t)ptr + nbytes + page - 1u) & ~(uintptr_t)(page - 1u);
    if (end <= start) {
        return;
    }
    if (willneed) {
#if defined(POSIX_MADV_WILLNEED)
        (void)posix_madvise((void *)start, (size_t)(end - start), POSIX_MADV_WILLNEED);
#elif defined(MADV_WILLNEED)
        (void)madvise((void *)start, (size_t)(end - start), MADV_WILLNEED);
#endif
    } else {
#if defined(POSIX_MADV_DONTNEED)
        (void)posix_madvise((void *)start, (size_t)(end - start), POSIX_MADV_DONTNEED);
#elif defined(MADV_DONTNEED)
        (void)madvise((void *)start, (size_t)(end - start), MADV_DONTNEED);
#endif
    }
#else
    (void)ptr;
    (void)nbytes;
    (void)willneed;
#endif
}

static int native_layer_span(
    const edgexpu_native_session *session,
    int layer_index,
    const uint8_t **start,
    size_t *nbytes
) {
    uint32_t i;
    uint64_t min_off = UINT64_MAX;
    uint64_t max_off = 0;
    if (session == NULL || session->file_map == NULL || layer_index < 0) {
        return 0;
    }
    for (i = 0; i < session->gguf.n_tensors; i++) {
        const edgexpu_gguf_tensor *tensor = &session->gguf.tensors[i];
        uint64_t off;
        size_t n;
        if (edgexpu_gguf_tensor_block_index(tensor->name) != layer_index) {
            continue;
        }
        off = session->gguf.data_offset + tensor->offset;
        n = edgexpu_gguf_tensor_nbytes(tensor);
        if (n == 0 || off >= session->file_map_size) {
            continue;
        }
        if (off < min_off) {
            min_off = off;
        }
        if (off + n > max_off) {
            max_off = off + n;
        }
    }
    if (min_off == UINT64_MAX || max_off <= min_off) {
        return 0;
    }
    if (max_off > session->file_map_size) {
        max_off = session->file_map_size;
    }
    if (start != NULL) {
        *start = session->file_map + min_off;
    }
    if (nbytes != NULL) {
        *nbytes = (size_t)(max_off - min_off);
    }
    return 1;
}

static void native_dontneed_layer(const edgexpu_native_session *session, int layer_index) {
    const uint8_t *start = NULL;
    size_t nbytes = 0;
    if (native_layer_span(session, layer_index, &start, &nbytes)) {
        native_page_advise(start, nbytes, 0);
    }
}

static void native_dontneed_tensor(const edgexpu_native_session *session, const edgexpu_gguf_tensor *tensor) {
    const uint8_t *src = tensor_bytes(session, tensor);
    size_t n = edgexpu_gguf_tensor_nbytes(tensor);
    native_page_advise(src, n, 0);
}

void edgexpu_native_prefetch_hint(edgexpu_native_session *session) {
    int layer;
    const uint8_t *start = NULL;
    size_t nbytes = 0;
    if (session == NULL || session->file_map == NULL) {
        return;
    }
    layer = session->staged_layer >= 0 ? session->staged_layer + 1 : 0;
    if (layer >= (int)session->gguf.block_count) {
        layer = 0;
    }
    if (native_layer_span(session, layer, &start, &nbytes)) {
        native_page_advise(start, nbytes, 1);
    }
}

static int native_kv_index(const edgexpu_native_session *session, int layer_index) {
    if (session == NULL || layer_index < 0) {
        return -1;
    }
    if (session->kv_slot == NULL) {
        return layer_index;
    }
    if ((uint32_t)layer_index >= session->gguf.block_count) {
        return -1;
    }
    return session->kv_slot[layer_index];
}

static float *native_kv_k_at(edgexpu_native_session *session, int layer_index, int pos) {
    int slot = native_kv_index(session, layer_index);
    if (slot < 0) {
        return NULL;
    }
    return edgexpu_kv_cache_k_at(&session->kv, slot, pos);
}

static float *native_kv_v_at(edgexpu_native_session *session, int layer_index, int pos) {
    int slot = native_kv_index(session, layer_index);
    if (slot < 0) {
        return NULL;
    }
    return edgexpu_kv_cache_v_at(&session->kv, slot, pos);
}

static int ensure_weight_stage(edgexpu_native_session *session, size_t need, char *error, size_t error_size) {
    uint8_t *grown;
    if (session == NULL) {
        set_error(error, error_size, "weight staging session 为空");
        return 0;
    }
    if (need < (size_t)(1u << 20)) {
        need = (size_t)(1u << 20);
    }
    if (session->weight_stage != NULL && session->weight_stage_cap >= need) {
        return 1;
    }
    grown = (uint8_t *)realloc(session->weight_stage, need);
    if (grown == NULL) {
        set_error(error, error_size, "weight staging 分配失败");
        return 0;
    }
    session->weight_stage = grown;
    session->weight_stage_cap = need;
    return 1;
}

static const uint8_t *stage_copy_bytes(
    edgexpu_native_session *session,
    const uint8_t *src,
    size_t nbytes,
    char *error,
    size_t error_size
) {
    size_t aligned;
    uint8_t *dst;
    if (src == NULL || nbytes == 0) {
        set_error(error, error_size, "staging 拷贝参数无效");
        return NULL;
    }
    aligned = (nbytes + 63u) & ~(size_t)63u;
    if (session->weight_stage == NULL ||
        session->weight_stage_used + aligned > session->weight_stage_cap) {
        size_t need = session->weight_stage_used + aligned;
        if (session->weight_stage_used != 0) {
            set_error(error, error_size, "weight staging 在层中途不足");
            return NULL;
        }
        if (!ensure_weight_stage(session, need, error, error_size)) {
            return NULL;
        }
    }
    dst = session->weight_stage + session->weight_stage_used;
    memcpy(dst, src, nbytes);
    native_page_advise(src, nbytes, 0);
    session->weight_stage_used += aligned;
    return dst;
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
    const edgexpu_arch_tensor_names *names = session_tensors(session);
    return session == NULL ? NULL : edgexpu_gguf_find_tensor(&session->gguf, names->token_embd);
}

/* 有独立 output.weight 则用它算 logits，否则 tied embedding。 */
static const edgexpu_gguf_tensor *logit_weight_tensor(const edgexpu_native_session *session) {
    const edgexpu_gguf_tensor *output;
    const edgexpu_arch_tensor_names *names = session_tensors(session);
    if (session == NULL) {
        return NULL;
    }
    output = edgexpu_gguf_find_tensor(&session->gguf, names->output_weight);
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
    free(layer->attn_q_norm);
    free(layer->attn_k_norm);
    free(layer->ssm_conv);
    free(layer->ssm_dt);
    free(layer->ssm_a);
    free(layer->ssm_norm);
    free(layer->conv_state);
    free(layer->ssm_state);
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
    native_dontneed_tensor(session, tensor);
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
        snprintf(
            error,
            error_size,
            "缺少或形状不匹配的 tensor: %s (have %llu want %d*%d dims=%u [%llu,%llu] type=%u)",
            name,
            (unsigned long long)n,
            n_out,
            n_in,
            tensor != NULL ? tensor->n_dims : 0,
            tensor != NULL ? (unsigned long long)tensor->dims[0] : 0,
            tensor != NULL ? (unsigned long long)tensor->dims[1] : 0,
            tensor != NULL ? tensor->type : 0
        );
        return 0;
    }
    if (edgexpu_gguf_row_bytes(tensor->type, n_in) == 0) {
        snprintf(error, error_size, "不支持的量化类型: %s", name);
        return 0;
    }
    {
        size_t nbytes = edgexpu_gguf_tensor_nbytes(tensor);
        const uint8_t *staged;
        if (nbytes == 0) {
            snprintf(error, error_size, "无法计算 tensor 字节数: %s", name);
            return 0;
        }
        staged = stage_copy_bytes(session, src, nbytes, error, error_size);
        if (staged == NULL) {
            return 0;
        }
        weight->data = staged;
    }
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

/* 按 adapter tensor 名 + GGUF 超参加载一层。不写死某一个模型的 blk 名。 */
static int format_layer_tensor(
    char *name,
    size_t name_size,
    const char *tmpl,
    int layer_index,
    char *error,
    size_t error_size
) {
    if (tmpl == NULL || tmpl[0] == '\0') {
        set_error(error, error_size, "adapter 未配置该层 tensor 名");
        return 0;
    }
    if (!edgexpu_arch_format_name(name, name_size, tmpl, layer_index)) {
        set_error(error, error_size, "tensor 名过长");
        return 0;
    }
    return 1;
}

static int load_qweight_gguf_shape(
    edgexpu_native_session *session,
    const char *tmpl,
    int layer_index,
    edgexpu_qweight *weight,
    char *error,
    size_t error_size
) {
    char name[96];
    const edgexpu_gguf_tensor *tensor;
    if (!format_layer_tensor(name, sizeof(name), tmpl, layer_index, error, error_size)) {
        return 0;
    }
    tensor = edgexpu_gguf_find_tensor(&session->gguf, name);
    if (tensor == NULL || tensor->n_dims < 2) {
        snprintf(error, error_size, "缺少 tensor: %s", name);
        return 0;
    }
    return load_qweight(
        session,
        name,
        (int)tensor->dims[1],
        (int)tensor->dims[0],
        weight,
        error,
        error_size
    );
}

static int load_layer_gated_delta(
    edgexpu_native_session *session,
    int layer_index,
    edgexpu_native_layer *layer,
    char *error,
    size_t error_size
) {
    const edgexpu_arch_tensor_names *names = session_tensors(session);
    char name[96];
    const edgexpu_gguf_info *gg = &session->gguf;
    int n_embd = layer->n_embd;
    int state_elems;

    if (gg->ssm_state_size == 0 || gg->ssm_group_count == 0 ||
        gg->ssm_time_step_rank == 0 || gg->ssm_inner_size == 0 ||
        gg->ssm_conv_kernel == 0) {
        set_error(error, error_size, "GATED_DELTA 层缺少 ssm 超参");
        return 0;
    }
    layer->ssm_dk = (int)gg->ssm_state_size;
    layer->ssm_n_k = (int)gg->ssm_group_count;
    layer->ssm_n_v = (int)gg->ssm_time_step_rank;
    if (layer->ssm_n_v == 0 || (int)gg->ssm_inner_size % layer->ssm_n_v != 0) {
        set_error(error, error_size, "GATED_DELTA inner_size 不能整除 time_step_rank");
        return 0;
    }
    layer->ssm_dv = (int)gg->ssm_inner_size / layer->ssm_n_v;
    layer->ssm_conv_k = (int)gg->ssm_conv_kernel;
    layer->qkv_dim = layer->ssm_dk * layer->ssm_n_k * 2 + layer->ssm_dv * layer->ssm_n_v;
    if (layer->ssm_dv <= 0 || layer->qkv_dim <= 0) {
        set_error(error, error_size, "GATED_DELTA 超参无效");
        return 0;
    }

#define LOAD_BLK(field, tmpl) \
    do { \
        if (!format_layer_tensor(name, sizeof(name), (tmpl), layer_index, error, error_size)) { \
            return 0; \
        } \
        if (!load_named_weight(session, name, &layer->field, error, error_size)) { \
            return 0; \
        } \
    } while (0)
#define LOAD_Q(field, tmpl, nout, nin) \
    do { \
        if (!format_layer_tensor(name, sizeof(name), (tmpl), layer_index, error, error_size)) { \
            return 0; \
        } \
        if (!load_qweight(session, name, (nout), (nin), &layer->field, error, error_size)) { \
            return 0; \
        } \
    } while (0)

    LOAD_BLK(attn_norm, names->attn_norm);
    LOAD_BLK(ffn_norm, names->ffn_norm);
    LOAD_Q(wqkv, names->attn_qkv, layer->qkv_dim, n_embd);
    LOAD_Q(wattn_gate, names->attn_gate, layer->ssm_dv * layer->ssm_n_v, n_embd);
    LOAD_Q(wssm_alpha, names->ssm_alpha, layer->ssm_n_v, n_embd);
    LOAD_Q(wssm_beta, names->ssm_beta, layer->ssm_n_v, n_embd);
    LOAD_Q(wssm_out, names->ssm_out, n_embd, layer->ssm_dv * layer->ssm_n_v);
    LOAD_Q(wgate, names->ffn_gate, layer->n_ff, n_embd);
    LOAD_Q(wup, names->ffn_up, layer->n_ff, n_embd);
    LOAD_Q(wdown, names->ffn_down, n_embd, layer->n_ff);
    LOAD_BLK(ssm_conv, names->ssm_conv1d);
    LOAD_BLK(ssm_dt, names->ssm_dt);
    LOAD_BLK(ssm_a, names->ssm_a);
    LOAD_BLK(ssm_norm, names->ssm_norm);
#undef LOAD_BLK
#undef LOAD_Q

    layer->conv_state = (float *)calloc((size_t)(layer->ssm_conv_k > 1 ? layer->ssm_conv_k - 1 : 1) *
                                            (size_t)layer->qkv_dim,
                                        sizeof(float));
    state_elems = layer->ssm_n_v * layer->ssm_dk * layer->ssm_dv;
    layer->ssm_state = (float *)calloc((size_t)state_elems, sizeof(float));
    if (layer->conv_state == NULL || layer->ssm_state == NULL) {
        set_error(error, error_size, "GATED_DELTA 状态分配失败");
        return 0;
    }
    layer->ready = 1;
    return 1;
}

static int load_layer_attn_qk_norm(
    edgexpu_native_session *session,
    int layer_index,
    edgexpu_native_layer *layer,
    char *error,
    size_t error_size
) {
    const edgexpu_arch_tensor_names *names = session_tensors(session);
    char name[96];
    int n_embd = layer->n_embd;

#define LOAD_BLK(field, tmpl) \
    do { \
        if (!format_layer_tensor(name, sizeof(name), (tmpl), layer_index, error, error_size)) { \
            return 0; \
        } \
        if (!load_named_weight(session, name, &layer->field, error, error_size)) { \
            return 0; \
        } \
    } while (0)
#define LOAD_Q(field, tmpl, nout, nin) \
    do { \
        if (!format_layer_tensor(name, sizeof(name), (tmpl), layer_index, error, error_size)) { \
            return 0; \
        } \
        if (!load_qweight(session, name, (nout), (nin), &layer->field, error, error_size)) { \
            return 0; \
        } \
    } while (0)

    LOAD_BLK(attn_norm, names->attn_norm);
    LOAD_BLK(ffn_norm, names->ffn_norm);
    LOAD_BLK(attn_q_norm, names->attn_q_norm);
    LOAD_BLK(attn_k_norm, names->attn_k_norm);
    if (!load_qweight_gguf_shape(session, names->attn_q, layer_index, &layer->wq, error, error_size) ||
        !load_qweight_gguf_shape(session, names->attn_k, layer_index, &layer->wk, error, error_size) ||
        !load_qweight_gguf_shape(session, names->attn_v, layer_index, &layer->wv, error, error_size) ||
        !load_qweight_gguf_shape(session, names->attn_output, layer_index, &layer->wo, error, error_size)) {
        return 0;
    }
    if (layer->n_kv_heads > 0 && layer->wk.n_out % layer->n_kv_heads == 0) {
        layer->head_dim = layer->wk.n_out / layer->n_kv_heads;
    }
    LOAD_Q(wgate, names->ffn_gate, layer->n_ff, n_embd);
    LOAD_Q(wup, names->ffn_up, layer->n_ff, n_embd);
    LOAD_Q(wdown, names->ffn_down, n_embd, layer->n_ff);
#undef LOAD_BLK
#undef LOAD_Q
    layer->ready = 1;
    return 1;
}

static int load_layer_into(
    edgexpu_native_session *session,
    int layer_index,
    edgexpu_native_layer *layer,
    char *error,
    size_t error_size
) {
    const edgexpu_arch_tensor_names *names = session_tensors(session);
    char name[96];

    edgexpu_layer_kind kind;
    float *saved_conv = NULL;
    float *saved_ssm = NULL;
    int had_state = 0;

    kind = edgexpu_arch_layer_kind(&session->arch, &session->gguf, layer_index);
    if (kind != EDGEXPU_LAYER_ATTN_SWIGLU &&
        kind != EDGEXPU_LAYER_GATED_DELTA &&
        kind != EDGEXPU_LAYER_ATTN_QK_NORM) {
        snprintf(error, error_size, "layer %d 不是 cpu.native 已实现的 block", layer_index);
        return 0;
    }

    if (layer->conv_state != NULL || layer->ssm_state != NULL) {
        had_state = 1;
        saved_conv = layer->conv_state;
        saved_ssm = layer->ssm_state;
        layer->conv_state = NULL;
        layer->ssm_state = NULL;
    }
    layer_free(layer);
    session->weight_stage_used = 0;
    session->staged_layer = layer_index;
    layer->kind = kind;
    layer->n_embd = (int)session->gguf.embedding_length;
    layer->n_ff = (int)session->gguf.feed_forward_length;
    layer->n_heads = (int)session->gguf.head_count;
    layer->n_kv_heads = (int)session->gguf.head_count_kv;
    layer->head_dim = (kind == EDGEXPU_LAYER_ATTN_SWIGLU)
        ? edgexpu_gguf_head_dim(&session->gguf)
        : session_attn_head_dim(session);
    if (layer->n_embd <= 0 || layer->n_ff <= 0 || layer->n_heads <= 0 ||
        layer->n_kv_heads <= 0 || layer->head_dim <= 0 || layer->n_heads % layer->n_kv_heads != 0) {
        set_error(error, error_size, "native layer 架构参数无效");
        goto load_fail;
    }
    if (kind == EDGEXPU_LAYER_GATED_DELTA) {
        if (!load_layer_gated_delta(session, layer_index, layer, error, error_size)) {
            goto load_fail;
        }
        goto load_ok;
    }
    if (kind == EDGEXPU_LAYER_ATTN_QK_NORM) {
        if (!load_layer_attn_qk_norm(session, layer_index, layer, error, error_size)) {
            goto load_fail;
        }
        goto load_ok;
    }

#define LOAD_BLK(field, tmpl) \
    do { \
        if (!edgexpu_arch_format_name(name, sizeof(name), (tmpl), layer_index)) { \
            set_error(error, error_size, "tensor 名过长"); \
            goto load_fail; \
        } \
        if (!load_named_weight(session, name, &layer->field, error, error_size)) { \
            goto load_fail; \
        } \
    } while (0)

#define LOAD_Q(field, tmpl, nout, nin) \
    do { \
        if (!edgexpu_arch_format_name(name, sizeof(name), (tmpl), layer_index)) { \
            set_error(error, error_size, "tensor 名过长"); \
            goto load_fail; \
        } \
        if (!load_qweight(session, name, (nout), (nin), &layer->field, error, error_size)) { \
            goto load_fail; \
        } \
    } while (0)

    LOAD_BLK(attn_norm, names->attn_norm);
    LOAD_BLK(ffn_norm, names->ffn_norm);
    LOAD_Q(wq, names->attn_q, layer->n_embd, layer->n_embd);
    LOAD_Q(wk, names->attn_k, layer->n_kv_heads * layer->head_dim, layer->n_embd);
    LOAD_Q(wv, names->attn_v, layer->n_kv_heads * layer->head_dim, layer->n_embd);
    LOAD_Q(wo, names->attn_output, layer->n_embd, layer->n_embd);
    LOAD_Q(wgate, names->ffn_gate, layer->n_ff, layer->n_embd);
    LOAD_Q(wup, names->ffn_up, layer->n_ff, layer->n_embd);
    LOAD_Q(wdown, names->ffn_down, layer->n_embd, layer->n_ff);
    if (session->arch.has_qkv_bias) {
        LOAD_BLK(bq, names->attn_q_bias);
        LOAD_BLK(bk, names->attn_k_bias);
        LOAD_BLK(bv, names->attn_v_bias);
    } else {
        if (!edgexpu_arch_format_name(name, sizeof(name), names->attn_q_bias, layer_index) ||
            !load_named_weight_optional(session, name, &layer->bq, error, error_size)) {
            goto load_fail;
        }
        if (!edgexpu_arch_format_name(name, sizeof(name), names->attn_k_bias, layer_index) ||
            !load_named_weight_optional(session, name, &layer->bk, error, error_size)) {
            goto load_fail;
        }
        if (!edgexpu_arch_format_name(name, sizeof(name), names->attn_v_bias, layer_index) ||
            !load_named_weight_optional(session, name, &layer->bv, error, error_size)) {
            goto load_fail;
        }
    }
    if (!edgexpu_arch_format_name(name, sizeof(name), names->attn_output_bias, layer_index) ||
        !load_named_weight_optional(session, name, &layer->bo, error, error_size)) {
        goto load_fail;
    }
    if (!edgexpu_arch_format_name(name, sizeof(name), names->ffn_gate_bias, layer_index) ||
        !load_named_weight_optional(session, name, &layer->bgate, error, error_size)) {
        goto load_fail;
    }
    if (!edgexpu_arch_format_name(name, sizeof(name), names->ffn_up_bias, layer_index) ||
        !load_named_weight_optional(session, name, &layer->bup, error, error_size)) {
        goto load_fail;
    }
    if (!edgexpu_arch_format_name(name, sizeof(name), names->ffn_down_bias, layer_index) ||
        !load_named_weight_optional(session, name, &layer->bdown, error, error_size)) {
        goto load_fail;
    }
#undef LOAD_BLK
#undef LOAD_Q

    layer->ready = 1;
load_ok:
    if (had_state) {
        free(layer->conv_state);
        free(layer->ssm_state);
        layer->conv_state = saved_conv;
        layer->ssm_state = saved_ssm;
    }
    return 1;
load_fail:
    layer_free(layer);
    layer->conv_state = saved_conv;
    layer->ssm_state = saved_ssm;
    return 0;
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
        for (i = 0; i < session->n_layers_cached; i++) {
            layer_free(&session->layers[i]);
        }
        free(session->layers);
        session->layers = NULL;
        session->n_layers_cached = 0;
    }
}

static int ensure_layers_cached(edgexpu_native_session *session, char *error, size_t error_size) {
    int n_layers;

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
        set_error(error, error_size, "native 层槽分配失败");
        return 0;
    }
    session->n_layers_cached = n_layers;
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
    session->staged_layer = -1;
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
    n_embd = (int)session->gguf.embedding_length;
    n_ff = (int)session->gguf.feed_forward_length;
    n_heads = (int)session->gguf.head_count;
    n_kv = (int)session->gguf.head_count_kv;
    head_dim = session_attn_head_dim(session);
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
    ws->attn = (float *)malloc((size_t)session_attn_inner(session) * sizeof(float));
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
    layer_free(&session->layer0);
    free(session->weight_stage);
    session->weight_stage = NULL;
    session->weight_stage_cap = 0;
    session->weight_stage_used = 0;
    session->staged_layer = -1;
    free(session->kv_slot);
    session->kv_slot = NULL;
    session->n_kv_layers = 0;
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
    edgexpu_gguf_info_free(&session->gguf);
    memset(session, 0, sizeof(*session));
    session->file_fd = -1;
}

size_t edgexpu_native_memory_bytes(const edgexpu_native_session *session) {
    size_t bytes;

    if (session == NULL || !session->loaded) {
        return 0;
    }
    bytes = session->weight_stage_cap + edgexpu_kv_cache_bytes(&session->kv);
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

    head_dim = session_attn_head_dim(session);
    max_seq = EDGEXPU_KV_DEFAULT_MAX_SEQ;
    if (session->gguf.context_length > 0 && (int)session->gguf.context_length < max_seq) {
        max_seq = (int)session->gguf.context_length;
    }
    if (!admit_session_window(session, max_seq, error, error_size)) {
        edgexpu_native_free(session);
        return 0;
    }

    if (!native_map_file(session, gguf_path, error, error_size)) {
        edgexpu_native_free(session);
        return 0;
    }
    {
        size_t need = edgexpu_gguf_max_block_bytes(&session->gguf);
        if (!ensure_weight_stage(session, need, error, error_size)) {
            edgexpu_native_free(session);
            return 0;
        }
    }
    {
        int n_layers = (int)session->gguf.block_count;
        int i;
        session->kv_slot = (int *)malloc((size_t)n_layers * sizeof(int));
        if (n_layers > 0 && session->kv_slot == NULL) {
            set_error(error, error_size, "native KV 槽表分配失败");
            edgexpu_native_free(session);
            return 0;
        }
        session->n_kv_layers = 0;
        for (i = 0; i < n_layers; i++) {
            edgexpu_layer_kind kind = edgexpu_arch_layer_kind(&session->arch, &session->gguf, i);
            if (edgexpu_layer_kind_uses_kv(kind)) {
                session->kv_slot[i] = session->n_kv_layers;
                session->n_kv_layers++;
            } else {
                session->kv_slot[i] = -1;
            }
        }
    }

    if (session->n_kv_layers > 0 && session->gguf.head_count_kv > 0 && head_dim > 0) {
        if (!edgexpu_kv_cache_allocate(
                &session->kv,
                session->n_kv_layers,
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
    if (!load_named_weight(session, session_tensors(session)->output_norm, &session->output_norm, error, error_size)) {
        edgexpu_native_free(session);
        return 0;
    }
    if (!load_named_weight_optional(session, session_tensors(session)->output_bias, &session->output_bias, error, error_size)) {
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
    if (!admit_session_window(session, needed, error, error_size)) {
        return 0;
    }
    if (!ensure_token_ids(session, context, error, error_size)) {
        return 0;
    }

    n_layers = session->n_kv_layers > 0 ? session->n_kv_layers : (int)session->gguf.block_count;
    n_kv = (int)session->gguf.head_count_kv;
    head_dim = session_attn_head_dim(session);
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

static void apply_rope_n_rot(
    const edgexpu_native_session *session,
    float *x,
    int n_heads,
    int head_dim,
    int pos
) {
    int h;
    float freq = default_rope_freq(session);
    int n_rot = session->gguf.rope_dimension_count > 0
        ? (int)session->gguf.rope_dimension_count
        : head_dim;
    if (n_rot > head_dim || n_rot < 2) {
        n_rot = head_dim;
    }
    /* 文本位置 t=h=w=e 时，MRoPE 退化为对 n_rot 维做 NeoX RoPE，不能按 section 各自用一套频率。 */
    for (h = 0; h < n_heads; h++) {
        edgexpu_cpu_rope_neox(x + (size_t)h * (size_t)head_dim, 1, n_rot, pos, freq);
    }
}

static int apply_swiglu_ffn(
    const edgexpu_native_layer *layer,
    int seq,
    float *hidden,
    float *normed,
    float *gate,
    float *up,
    float *down,
    float eps
) {
    int n_embd = layer->n_embd;
    int n_ff = layer->n_ff;
    int t;
    float *ffn_in = (float *)malloc((size_t)seq * (size_t)n_embd * sizeof(float));
    if (eps <= 0.0f) {
        eps = 1e-6f;
    }
    if (ffn_in == NULL) {
        return 0;
    }
    memcpy(ffn_in, hidden, (size_t)seq * (size_t)n_embd * sizeof(float));
    for (t = 0; t < seq; t++) {
        edgexpu_cpu_rmsnorm(
            normed + (size_t)t * (size_t)n_embd,
            ffn_in + (size_t)t * (size_t)n_embd,
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
        edgexpu_cpu_add(
            hidden + (size_t)t * (size_t)n_embd,
            ffn_in + (size_t)t * (size_t)n_embd,
            down + (size_t)t * (size_t)n_embd,
            n_embd
        );
    }
    free(ffn_in);
    return 1;
}

static void gdn_conv1d_token(
    const edgexpu_native_layer *layer,
    const float *mixed,
    float *out
) {
    int k;
    int c;
    int K = layer->ssm_conv_k;
    int C = layer->qkv_dim;
    int hist = K > 1 ? K - 1 : 0;
    for (c = 0; c < C; c++) {
        float acc = 0.0f;
        for (k = 0; k < K; k++) {
            const float *src;
            int lag = K - 1 - k;
            if (lag == 0) {
                src = mixed;
            } else {
                src = layer->conv_state + (size_t)(hist - lag) * (size_t)C;
            }
            acc += layer->ssm_conv[(size_t)c * (size_t)K + (size_t)k] * src[c];
        }
        out[c] = acc;
    }
    if (hist > 0) {
        if (hist > 1) {
            memmove(layer->conv_state, layer->conv_state + C, (size_t)(hist - 1) * (size_t)C * sizeof(float));
        }
        memcpy(layer->conv_state + (size_t)(hist - 1) * (size_t)C, mixed, (size_t)C * sizeof(float));
    }
}

static int apply_gated_delta_layer(
    edgexpu_native_session *session,
    edgexpu_native_layer *layer,
    int seq,
    float *hidden,
    float *normed,
    float *gate,
    float *up,
    float *down,
    char *error,
    size_t error_size
) {
    int t;
    int h;
    int n_embd = layer->n_embd;
    int dk = layer->ssm_dk;
    int dv = layer->ssm_dv;
    int n_k = layer->ssm_n_k;
    int n_v = layer->ssm_n_v;
    int qkv_dim = layer->qkv_dim;
    int value_dim = dv * n_v;
    int k_dim = dk * n_k;
    float eps = session->gguf.rms_eps > 0.0f ? session->gguf.rms_eps : 1e-6f;
    float *mixed = NULL;
    float *conv_y = NULL;
    float *z = NULL;
    float *alpha = NULL;
    float *beta = NULL;
    float *core = NULL;
    float *proj = NULL;
    int ok = 0;

    (void)session;
    mixed = (float *)malloc((size_t)qkv_dim * sizeof(float));
    conv_y = (float *)malloc((size_t)qkv_dim * sizeof(float));
    z = (float *)malloc((size_t)seq * (size_t)value_dim * sizeof(float));
    alpha = (float *)malloc((size_t)n_v * sizeof(float));
    beta = (float *)malloc((size_t)n_v * sizeof(float));
    core = (float *)malloc((size_t)seq * (size_t)value_dim * sizeof(float));
    proj = (float *)malloc((size_t)seq * (size_t)n_embd * sizeof(float));
    if (mixed == NULL || conv_y == NULL || z == NULL || alpha == NULL || beta == NULL ||
        core == NULL || proj == NULL) {
        set_error(error, error_size, "GATED_DELTA workspace 分配失败");
        goto done;
    }

    for (t = 0; t < seq; t++) {
        float *xt = hidden + (size_t)t * (size_t)n_embd;
        float *nt = normed + (size_t)t * (size_t)n_embd;
        edgexpu_cpu_rmsnorm(nt, xt, layer->attn_norm, n_embd, eps);
        apply_qlinear(mixed, nt, &layer->wqkv, NULL);
        apply_qlinear(z + (size_t)t * (size_t)value_dim, nt, &layer->wattn_gate, NULL);
        apply_qlinear(alpha, nt, &layer->wssm_alpha, layer->ssm_dt);
        apply_qlinear(beta, nt, &layer->wssm_beta, NULL);
        edgexpu_cpu_sigmoid(beta, n_v);
        edgexpu_cpu_softplus(alpha, n_v);
        for (h = 0; h < n_v; h++) {
            alpha[h] *= layer->ssm_a[h];
        }
        gdn_conv1d_token(layer, mixed, conv_y);
        edgexpu_cpu_silu(conv_y, conv_y, qkv_dim);
        for (h = 0; h < n_v; h++) {
            int kh = n_k > 0 ? (h % n_k) : 0;
            float *q = conv_y + (size_t)kh * (size_t)dk;
            float *k = conv_y + (size_t)k_dim + (size_t)kh * (size_t)dk;
            float *v = conv_y + (size_t)k_dim * 2 + (size_t)h * (size_t)dv;
            float qn[256];
            float kn[256];
            if (dk > 256 || dv > 256) {
                set_error(error, error_size, "GATED_DELTA head dim 超出临时缓冲");
                goto done;
            }
            memcpy(qn, q, (size_t)dk * sizeof(float));
            memcpy(kn, k, (size_t)dk * sizeof(float));
            edgexpu_cpu_l2_normalize(qn, dk, eps);
            edgexpu_cpu_l2_normalize(kn, dk, eps);
            edgexpu_cpu_gated_delta_step(
                layer->ssm_state + (size_t)h * (size_t)dk * (size_t)dv,
                qn,
                kn,
                v,
                alpha[h],
                beta[h],
                core + (size_t)t * (size_t)value_dim + (size_t)h * (size_t)dv,
                dk,
                dv
            );
        }
        for (h = 0; h < n_v; h++) {
            float *yh = core + (size_t)t * (size_t)value_dim + (size_t)h * (size_t)dv;
            float *zh = z + (size_t)t * (size_t)value_dim + (size_t)h * (size_t)dv;
            float tmp[256];
            int i;
            edgexpu_cpu_rmsnorm(tmp, yh, layer->ssm_norm, dv, eps);
            edgexpu_cpu_silu(zh, zh, dv);
            for (i = 0; i < dv; i++) {
                yh[i] = tmp[i] * zh[i];
            }
        }
    }
    apply_qlinear_batch(proj, core, &layer->wssm_out, NULL, seq);
    for (t = 0; t < seq; t++) {
        edgexpu_cpu_add(
            hidden + (size_t)t * (size_t)n_embd,
            hidden + (size_t)t * (size_t)n_embd,
            proj + (size_t)t * (size_t)n_embd,
            n_embd
        );
    }
    if (!apply_swiglu_ffn(layer, seq, hidden, normed, gate, up, down, eps)) {
        set_error(error, error_size, "SwiGLU FFN workspace 分配失败");
        goto done;
    }
    ok = 1;
done:
    free(mixed);
    free(conv_y);
    free(z);
    free(alpha);
    free(beta);
    free(core);
    free(proj);
    return ok;
}

static int apply_attn_qk_norm_layer(
    edgexpu_native_session *session,
    const edgexpu_native_layer *layer,
    int layer_index,
    int seq,
    int pos_base,
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
    int n_heads = layer->n_heads;
    int n_kv = layer->n_kv_heads;
    int head_dim = layer->head_dim;
    int n_rep = n_heads / n_kv;
    int t;
    int h;
    float eps = session->gguf.rms_eps > 0.0f ? session->gguf.rms_eps : 1e-6f;
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    float *q_full = NULL;
    float *q_use = NULL;
    float *gate_h = NULL;
    int q_stride = layer->wq.n_out / n_heads;
    if (q_stride < head_dim) {
        q_stride = head_dim * 2;
    }

    q_full = (float *)malloc((size_t)seq * (size_t)n_heads * (size_t)q_stride * sizeof(float));
    q_use = (float *)malloc((size_t)seq * (size_t)n_heads * (size_t)head_dim * sizeof(float));
    gate_h = (float *)malloc((size_t)seq * (size_t)n_heads * (size_t)head_dim * sizeof(float));
    if (q_full == NULL || q_use == NULL || gate_h == NULL) {
        free(q_full);
        free(q_use);
        free(gate_h);
        set_error(error, error_size, "ATTN_QK_NORM workspace 分配失败");
        return 0;
    }

    for (t = 0; t < seq; t++) {
        edgexpu_cpu_rmsnorm(
            normed + (size_t)t * (size_t)n_embd,
            hidden + (size_t)t * (size_t)n_embd,
            layer->attn_norm,
            n_embd,
            eps
        );
    }
    apply_qlinear_batch(q_full, normed, &layer->wq, NULL, seq);
    apply_qlinear_batch(k, normed, &layer->wk, NULL, seq);
    apply_qlinear_batch(v, normed, &layer->wv, NULL, seq);
    for (t = 0; t < seq; t++) {
        for (h = 0; h < n_heads; h++) {
            float *src = q_full + ((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)q_stride;
            float *qd = q_use + ((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)head_dim;
            float *gd = gate_h + ((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)head_dim;
            memcpy(qd, src, (size_t)head_dim * sizeof(float));
            if (q_stride >= head_dim * 2) {
                memcpy(gd, src + head_dim, (size_t)head_dim * sizeof(float));
            } else {
                memset(gd, 0, (size_t)head_dim * sizeof(float));
            }
            edgexpu_cpu_rmsnorm(qd, qd, layer->attn_q_norm, head_dim, eps);
        }
        for (h = 0; h < n_kv; h++) {
            float *kd = k + ((size_t)t * (size_t)n_kv + (size_t)h) * (size_t)head_dim;
            edgexpu_cpu_rmsnorm(kd, kd, layer->attn_k_norm, head_dim, eps);
        }
        apply_rope_n_rot(
            session,
            q_use + (size_t)t * (size_t)n_heads * (size_t)head_dim,
            n_heads,
            head_dim,
            pos_base + t
        );
        apply_rope_n_rot(
            session,
            k + (size_t)t * (size_t)n_kv * (size_t)head_dim,
            n_kv,
            head_dim,
            pos_base + t
        );
    }

    for (t = 0; t < seq; t++) {
        for (h = 0; h < n_heads; h++) {
            int kv_h = h / n_rep;
            float *qh = q_use + ((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)head_dim;
            float *out_h = attn + ((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)head_dim;
            int s;
            for (s = 0; s <= pos_base + t; s++) {
                float *kh;
                if (s < pos_base) {
                    kh = native_kv_k_at(session, layer_index, s);
                    if (kh == NULL) {
                        free(q_full);
                        free(q_use);
                        free(gate_h);
                        set_error(error, error_size, "native decode 无法读取 K");
                        return 0;
                    }
                    kh += (size_t)kv_h * (size_t)head_dim;
                } else {
                    kh = k + ((size_t)(s - pos_base) * (size_t)n_kv + (size_t)kv_h) * (size_t)head_dim;
                }
                scores[s] = edgexpu_cpu_dot(qh, kh, head_dim) * attn_scale;
            }
            edgexpu_cpu_softmax(scores, pos_base + t + 1);
            memset(out_h, 0, (size_t)head_dim * sizeof(float));
            for (s = 0; s <= pos_base + t; s++) {
                float *vh;
                if (s < pos_base) {
                    vh = native_kv_v_at(session, layer_index, s);
                    if (vh == NULL) {
                        free(q_full);
                        free(q_use);
                        free(gate_h);
                        set_error(error, error_size, "native decode 无法读取 V");
                        return 0;
                    }
                    vh += (size_t)kv_h * (size_t)head_dim;
                } else {
                    vh = v + ((size_t)(s - pos_base) * (size_t)n_kv + (size_t)kv_h) * (size_t)head_dim;
                }
                edgexpu_cpu_saxpy(out_h, vh, scores[s], head_dim);
            }
            {
                float *gd = gate_h + ((size_t)t * (size_t)n_heads + (size_t)h) * (size_t)head_dim;
                int i;
                for (i = 0; i < head_dim; i++) {
                    gd[i] = 1.0f / (1.0f + expf(-gd[i]));
                    out_h[i] *= gd[i];
                }
            }
        }
    }

    for (t = 0; t < seq; t++) {
        float *k_slot = native_kv_k_at(session, layer_index, pos_base + t);
        float *v_slot = native_kv_v_at(session, layer_index, pos_base + t);
        if (k_slot == NULL || v_slot == NULL) {
            free(q_full);
            free(q_use);
            free(gate_h);
            set_error(error, error_size, "native prefill 无法写入 KV cache");
            return 0;
        }
        memcpy(k_slot, k + (size_t)t * (size_t)n_kv * (size_t)head_dim, (size_t)n_kv * (size_t)head_dim * sizeof(float));
        memcpy(v_slot, v + (size_t)t * (size_t)n_kv * (size_t)head_dim, (size_t)n_kv * (size_t)head_dim * sizeof(float));
    }

    apply_qlinear_batch(proj, attn, &layer->wo, NULL, seq);
    for (t = 0; t < seq; t++) {
        edgexpu_cpu_add(
            hidden + (size_t)t * (size_t)n_embd,
            hidden + (size_t)t * (size_t)n_embd,
            proj + (size_t)t * (size_t)n_embd,
            n_embd
        );
    }
    free(q_full);
    free(q_use);
    free(gate_h);
    (void)q;
    if (!apply_swiglu_ffn(layer, seq, hidden, normed, gate, up, down, eps)) {
        set_error(error, error_size, "SwiGLU FFN workspace 分配失败");
        return 0;
    }
    return 1;
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

    if (layer->kind == EDGEXPU_LAYER_GATED_DELTA) {
        return apply_gated_delta_layer(
            session,
            (edgexpu_native_layer *)layer,
            seq,
            hidden,
            normed,
            gate,
            up,
            down,
            error,
            error_size
        );
    }
    if (layer->kind == EDGEXPU_LAYER_ATTN_QK_NORM) {
        return apply_attn_qk_norm_layer(
            session,
            layer,
            layer_index,
            seq,
            0,
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
            error_size
        );
    }

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
        float *k_slot = native_kv_k_at(session, layer_index, t);
        float *v_slot = native_kv_v_at(session, layer_index, t);
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
    if (session->n_kv_layers > 0 && session->kv.k == NULL) {
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

    n_embd = (int)session->gguf.embedding_length;
    n_ff = (int)session->gguf.feed_forward_length;
    n_heads = (int)session->gguf.head_count;
    n_kv = (int)session->gguf.head_count_kv;
    head_dim = session_attn_head_dim(session);
    seq = session->token_count;
    if (session->kv.k != NULL && seq > session->kv.max_seq) {
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
    attn = (float *)calloc((size_t)seq * (size_t)session_attn_inner(session), sizeof(float));
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
    native_dontneed_tensor(session, embd);

    edgexpu_kv_cache_reset(&session->kv);
    for (layer_index = 0; layer_index < n_layers; layer_index++) {
        edgexpu_native_layer *ly = &session->layers[layer_index];
        if (!load_layer_into(session, layer_index, ly, error, error_size)) {
            goto cleanup;
        }
        if (ly->conv_state != NULL && ly->qkv_dim > 0 && ly->ssm_conv_k > 1) {
            memset(
                ly->conv_state,
                0,
                (size_t)(ly->ssm_conv_k - 1) * (size_t)ly->qkv_dim * sizeof(float)
            );
        }
        if (ly->ssm_state != NULL && ly->ssm_n_v > 0 && ly->ssm_dk > 0 && ly->ssm_dv > 0) {
            memset(
                ly->ssm_state,
                0,
                (size_t)ly->ssm_n_v * (size_t)ly->ssm_dk * (size_t)ly->ssm_dv * sizeof(float)
            );
        }
        if (!apply_transformer_layer(
                session,
                ly,
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
        native_dontneed_layer(session, layer_index);
    }

    for (t = 0; t < seq; t++) {
        float *xt = hidden + (size_t)t * (size_t)n_embd;
        memcpy(residual, xt, (size_t)n_embd * sizeof(float));
        edgexpu_cpu_rmsnorm(xt, residual, session->output_norm, n_embd, eps);
    }

    if (session->kv.k != NULL && !edgexpu_kv_cache_extend(&session->kv, seq, error, error_size)) {
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

static size_t logit_chunk_rows(const edgexpu_gguf_tensor *tensor, int n_embd) {
    size_t row_bytes;
    size_t n;
    if (tensor == NULL || n_embd <= 0) {
        return 1;
    }
    row_bytes = edgexpu_gguf_row_bytes(tensor->type, n_embd);
    if (row_bytes == 0) {
        return 1;
    }
    n = EDGEXPU_BUDGET_OUTPUT_CHUNK_BYTES / row_bytes;
    return n < 1u ? 1u : n;
}

static void dontneed_logit_rows(
    const edgexpu_native_session *session,
    const edgexpu_gguf_tensor *tensor,
    uint32_t row0,
    uint32_t nrows,
    int n_embd
) {
    const uint8_t *src = tensor_bytes(session, tensor);
    size_t row_bytes = edgexpu_gguf_row_bytes(tensor->type, n_embd);
    if (src == NULL || row_bytes == 0 || nrows == 0) {
        return;
    }
    native_page_advise(src + (size_t)row0 * row_bytes, (size_t)nrows * row_bytes, 0);
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
    size_t chunk;

    n_embd = (int)session->gguf.embedding_length;
    vocab = (int)session->tokenizer.vocab_size;
    weights = logit_weight_tensor(session);
    if (weights == NULL || session->last_hidden == NULL || vocab <= 0) {
        set_error(error, error_size, "native decode 无法计算 logits");
        return 0;
    }
    if (edgexpu_gguf_can_dot_q8(weights->type)) {
        x_q8 = quantize_hidden_q8(session->last_hidden, n_embd);
    }
    chunk = logit_chunk_rows(weights, n_embd);

    if (temperature > 1e-4f) {
        float *logits = (float *)malloc((size_t)vocab * sizeof(float));
        uint32_t begin;
        if (logits == NULL) {
            free(x_q8);
            set_error(error, error_size, "native logits 分配失败");
            return 0;
        }
        for (begin = 0; begin < (uint32_t)vocab; begin += (uint32_t)chunk) {
            uint32_t end = begin + (uint32_t)chunk;
            if (end > (uint32_t)vocab) {
                end = (uint32_t)vocab;
            }
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if((end - begin) >= 256)
#endif
            for (i = begin; i < end; i++) {
                logits[i] = logit_score(session, weights, i, n_embd, x_q8) / temperature;
            }
            dontneed_logit_rows(session, weights, begin, end - begin, n_embd);
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
    {
        uint32_t begin;
        for (begin = 0; begin < (uint32_t)vocab; begin += (uint32_t)chunk) {
            uint32_t end = begin + (uint32_t)chunk;
            if (end > (uint32_t)vocab) {
                end = (uint32_t)vocab;
            }
#if defined(_OPENMP)
#pragma omp parallel if((end - begin) >= 256)
            {
                uint32_t local_best = 0;
                float local_score = -INFINITY;
                uint32_t i_local;
#pragma omp for nowait schedule(static)
                for (i_local = begin; i_local < end; i_local++) {
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
            for (i = begin; i < end; i++) {
                float score = logit_score(session, weights, i, n_embd, x_q8);
                if (score > best_score) {
                    best_score = score;
                    best = i;
                }
            }
#endif
            dontneed_logit_rows(session, weights, begin, end - begin, n_embd);
        }
    }
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
    n_layers = (int)session->gguf.block_count;
    n_embd = (int)session->gguf.embedding_length;
    n_ff = (int)session->gguf.feed_forward_length;
    n_heads = (int)session->gguf.head_count;
    n_kv = (int)session->gguf.head_count_kv;
    head_dim = session_attn_head_dim(session);
    n_rep = n_heads / n_kv;
    eps = session->gguf.rms_eps > 0.0f ? session->gguf.rms_eps : 1e-6f;
    attn_scale = 1.0f / sqrtf((float)head_dim);

    if (session->kv.k != NULL && (pos < 0 || pos >= session->kv.max_seq)) {
        set_error(error, error_size, "native decode 超出 KV 窗口");
        return 0;
    }
    if (!ensure_layers_cached(session, error, error_size)) {
        return 0;
    }
    if (!ensure_decode_scratch(session, session->kv.max_seq > 0 ? session->kv.max_seq : 1, error, error_size)) {
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
    native_dontneed_tensor(session, token_embedding_tensor(session));

    for (layer_index = 0; layer_index < n_layers; layer_index++) {
        float *k_slot;
        float *v_slot;
        edgexpu_native_layer *ly = &session->layers[layer_index];

        if (!load_layer_into(session, layer_index, ly, error, error_size)) {
            return 0;
        }
        layer = ly;
        if (layer->kind == EDGEXPU_LAYER_GATED_DELTA) {
            if (!apply_gated_delta_layer(
                    session,
                    &session->layers[layer_index],
                    1,
                    hidden,
                    normed,
                    gate,
                    up,
                    down,
                    error,
                    error_size)) {
                return 0;
            }
            native_dontneed_layer(session, layer_index);
            continue;
        }
        if (layer->kind == EDGEXPU_LAYER_ATTN_QK_NORM) {
            if (!apply_attn_qk_norm_layer(
                    session,
                    layer,
                    layer_index,
                    1,
                    pos,
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
                return 0;
            }
            native_dontneed_layer(session, layer_index);
            continue;
        }
        edgexpu_cpu_rmsnorm(normed, hidden, layer->attn_norm, n_embd, eps);
        apply_qlinear(q, normed, &layer->wq, layer->bq);
        apply_qlinear(k, normed, &layer->wk, layer->bk);
        apply_qlinear(v, normed, &layer->wv, layer->bv);
        apply_rope(session, q, n_heads, head_dim, pos);
        apply_rope(session, k, n_kv, head_dim, pos);

        k_slot = native_kv_k_at(session, layer_index, pos);
        v_slot = native_kv_v_at(session, layer_index, pos);
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
                float *kh = native_kv_k_at(session, layer_index, s);
                if (kh == NULL) {
                    set_error(error, error_size, "native decode 无法读取 K");
                    return 0;
                }
                kh += (size_t)kv_h * (size_t)head_dim;
                scores[s] = edgexpu_cpu_dot(qh, kh, head_dim) * attn_scale;
            }
            edgexpu_cpu_softmax(scores, pos + 1);
            for (s = 0; s <= pos; s++) {
                float *vh = native_kv_v_at(session, layer_index, s);
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
        native_dontneed_layer(session, layer_index);
    }

    memcpy(residual, hidden, (size_t)n_embd * sizeof(float));
    edgexpu_cpu_rmsnorm(hidden, residual, session->output_norm, n_embd, eps);
    memcpy(session->last_hidden, hidden, (size_t)n_embd * sizeof(float));
    session->last_hidden_rms = vector_rms(session->last_hidden, n_embd);
    if (session->kv.k != NULL && !edgexpu_kv_cache_extend(&session->kv, 1, error, error_size)) {
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
    head_dim = session->layer0.head_dim > 0 ? session->layer0.head_dim : session_attn_head_dim(session);
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
    {
        int q_cap = n_embd;
        if (layer->qkv_dim > q_cap) {
            q_cap = layer->qkv_dim;
        }
        if (layer->wq.n_out > q_cap) {
            q_cap = layer->wq.n_out;
        }
        q = (float *)malloc((size_t)q_cap * sizeof(float));
    }
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
    if (layer->wq.data != NULL && layer->wk.data != NULL) {
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
    } else if (layer->wqkv.data != NULL) {
        apply_qlinear(q, normed, &layer->wqkv, NULL);
        printf("layer0_kind=gated_delta qkv_dim=%d qkv_rms=%.6f\n", layer->qkv_dim, vector_rms(q, layer->qkv_dim));
        print_float_prefix("layer0_qkv", q, layer->qkv_dim, 8);
    }

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
    {
        size_t chunk = logit_chunk_rows(weights, n_embd);
        uint32_t begin;
        for (begin = 0; begin < (uint32_t)vocab; begin += (uint32_t)chunk) {
            uint32_t end = begin + (uint32_t)chunk;
            uint32_t row;
            if (end > (uint32_t)vocab) {
                end = (uint32_t)vocab;
            }
            for (row = begin; row < end; row++) {
                float score = logit_score(session, weights, row, n_embd, x_q8);
                int slot = top_k - 1;
                if (score <= top_scores[slot]) {
                    continue;
                }
                while (slot > 0 && score > top_scores[slot - 1]) {
                    top_scores[slot] = top_scores[slot - 1];
                    top_ids[slot] = top_ids[slot - 1];
                    slot--;
                }
                top_scores[slot] = score;
                top_ids[slot] = row;
            }
            dontneed_logit_rows(session, weights, begin, end - begin, n_embd);
        }
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
    {
        edgexpu_device_profile profile;
        edgexpu_resource_plan plan;
        memset(&profile, 0, sizeof(profile));
        (void)edgexpu_profile_device(&profile);
        edgexpu_scheduler_estimate_gguf(&session.gguf, &profile, session.kv.max_seq > 0 ? session.kv.max_seq : EDGEXPU_KV_DEFAULT_MAX_SEQ, &plan);
        printf(
            "budget_admitted=%d total_mb=%zu limit_mb=%zu window=%d plugin=%s\n",
            plan.admitted,
            plan.total_bytes / (1024u * 1024u),
            plan.limit_bytes / (1024u * 1024u),
            plan.window,
            session.arch.plugin != NULL ? session.arch.plugin->id : session.arch.name
        );
        printf("budget_reason=%s\n", plan.reason);
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
    {
        int kv_first = -1;
        int kv_last = -1;
        int li;
        int kv_dim;
        for (li = 0; li < (int)session.gguf.block_count; li++) {
            if (session.layers[li].kind == EDGEXPU_LAYER_GATED_DELTA) {
                continue;
            }
            if (kv_first < 0) {
                kv_first = li;
            }
            kv_last = li;
        }
        if (kv_first < 0) {
            kv_first = 0;
            kv_last = last_layer >= 0 ? last_layer : 0;
        }
        k0 = native_kv_k_at(&session, kv_first, 0);
        klast = kv_last >= 0 ? native_kv_k_at(&session, kv_last, 0) : NULL;
        kv_dim = session.layers[kv_last >= 0 ? kv_last : 0].n_kv_heads *
            session.layers[kv_last >= 0 ? kv_last : 0].head_dim;
        if (k0 == NULL || klast == NULL ||
            vector_rms(k0, kv_dim) <= 0.0f ||
            vector_rms(klast, kv_dim) <= 0.0f) {
            fprintf(stderr, "native prefill KV was empty\n");
            edgexpu_native_free(&session);
            return 1;
        }
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
