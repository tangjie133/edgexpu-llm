#ifndef EDGEXPU_NATIVE_H
#define EDGEXPU_NATIVE_H

#include "edgexpu/arch.h"
#include "edgexpu/gguf.h"
#include "edgexpu/kv_cache.h"
#include "edgexpu/tokenizer.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Native CPU fallback：mmap GGUF、按 adapter 跑 transformer、逐步 decode。
 * 量化权重按层拷进 staging，不按整文件计入工作集。
 */

#define EDGEXPU_NATIVE_MAX_TOKENS 1024 /* 仅作无 context_length 时的 token_ids 下限 */

/* mmap 上的量化矩阵，不展开成 f32。 */
typedef struct edgexpu_qweight {
    const uint8_t *data;
    uint32_t type;
    int n_out;
    int n_in;
} edgexpu_qweight;

typedef struct edgexpu_native_layer {
    float *attn_norm;
    float *ffn_norm;
    edgexpu_qweight wq;
    float *bq;     /* 无 QKV bias 的架构上为 NULL */
    edgexpu_qweight wk;
    float *bk;
    edgexpu_qweight wv;
    float *bv;
    edgexpu_qweight wo;
    float *bo;     /* attn_output.bias，GGUF 没有则为 NULL */
    edgexpu_qweight wgate;
    float *bgate;
    edgexpu_qweight wup;
    float *bup;
    edgexpu_qweight wdown;
    float *bdown;
    int n_embd;
    int n_ff;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    edgexpu_layer_kind kind;
    float *attn_q_norm;
    float *attn_k_norm;
    edgexpu_qweight wqkv;
    edgexpu_qweight wattn_gate;
    edgexpu_qweight wssm_out;
    edgexpu_qweight wssm_alpha;
    edgexpu_qweight wssm_beta;
    float *ssm_conv;
    float *ssm_dt;
    float *ssm_a;
    float *ssm_norm;
    float *conv_state;
    float *ssm_state;
    int qkv_dim;
    int ssm_dk;
    int ssm_dv;
    int ssm_n_k;
    int ssm_n_v;
    int ssm_conv_k;
    int ready;
} edgexpu_native_layer;

typedef struct edgexpu_native_session {
    edgexpu_gguf_info gguf;
    edgexpu_arch_adapter arch;
    edgexpu_tokenizer tokenizer;
    edgexpu_kv_cache kv;
    edgexpu_native_layer layer0; /* 加载后的第 0 层；全层缓存后与 layers[0] 共用指针 */
    edgexpu_native_layer *layers; /* 首次 prefill 绑定全部层的量化权重指针，不展开成 f32 */
    float *output_norm;
    float *output_bias;          /* output.bias，没有则为 NULL */
    float *last_hidden;          /* 最近一次 prefill/decode 的最后位置隐状态，用于算 logits */
    uint32_t *token_ids;         /* 容量 token_ids_cap，随 context / KV 窗口分配 */
    int token_ids_cap;
    const uint8_t *file_map;     /* 整文件 mmap，tensor 按偏移读取；RSS 靠 staging + DONTNEED */
    size_t file_map_size;
    int file_fd;
    uint8_t *weight_stage;       /* 当前层量化权重工作集 */
    size_t weight_stage_cap;
    size_t weight_stage_used;
    int staged_layer;
    int *kv_slot;                /* 层号 → KV 槽；GDN 为 -1 */
    int n_kv_layers;
    int token_count;
    int prompt_token_count;
    int generated_tokens;
    int n_layers_cached;
    int loaded;
    int prefill_layers;
    float last_hidden_rms;
    double prefill_seconds;
    void *scratch; /* decode 工作区，跨 token 复用，避免每步 malloc */
} edgexpu_native_session;

void edgexpu_native_init(edgexpu_native_session *session);
void edgexpu_native_free(edgexpu_native_session *session);

/* mmap 工作集（staging + KV + scratch），不是整文件。 */
size_t edgexpu_native_memory_bytes(const edgexpu_native_session *session);

/* prefetch job：对下一层 blk.* 做 WILLNEED，禁止整文件。 */
void edgexpu_native_prefetch_hint(edgexpu_native_session *session);

/* 读 GGUF、选 adapter、mmap、分配 KV、加载第 0 层与 output_norm。 */
int edgexpu_native_load(
    edgexpu_native_session *session,
    const char *gguf_path,
    char *error,
    size_t error_size
);

/* 对原文做 BPE。chat template 由 runtime generate 在调用前套用，这里不写死标记。 */
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

/* 按 prompt + max_tokens 分配 KV 与 token_ids。超出模型 context_length 时报错。 */
int edgexpu_native_ensure_window(
    edgexpu_native_session *session,
    int n_prompt,
    int n_new,
    char *error,
    size_t error_size
);

/* 全层 prefill：embedding → 各层 attention+FFN → output_norm，写入全部层 KV。 */
int edgexpu_native_forward_prefill(
    edgexpu_native_session *session,
    char *error,
    size_t error_size
);

/* 兼容旧调用名，实际等于 full prefill。 */
int edgexpu_native_forward_layer0(
    edgexpu_native_session *session,
    char *error,
    size_t error_size
);

/* 采样一个 token 并前向写入下一 KV 槽。stopped=1 表示遇到 EOS。超出 KV 窗口返回失败。 */
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
);

void edgexpu_native_print_info(const edgexpu_native_session *session);

/* 原文 tokenize（不套 chat template），打印 token ids、layer0 RMS/QKV/RoPE、
 * last hidden、lm_head top-k logits，以及 greedy 前 greedy_n 个 token。
 * 用于和 llama.cpp / llama-cli --temp 0 做数值回归。
 */
int edgexpu_native_dump_logits(
    edgexpu_native_session *session,
    const char *prompt,
    int greedy_n,
    int top_k,
    char *error,
    size_t error_size
);

/* gguf_path 为空时只测 kernel/KV；否则加载模型做 tokenize+prefill+若干 decode。 */
int edgexpu_native_selftest(const char *gguf_path);

#ifdef __cplusplus
}
#endif

#endif
