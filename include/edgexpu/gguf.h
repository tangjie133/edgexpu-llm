#ifndef EDGEXPU_GGUF_H
#define EDGEXPU_GGUF_H

#include <stddef.h>
#include <stdint.h>

#include "edgexpu/manifest.h"
#include "edgexpu/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 有限 GGUF v3 loader：读 metadata、tensor 目录和 GPT-2 tokenizer。
 * 不解析全部 KV；权重本体由 native 路径 mmap 后再按 tensor 反量化。
 */

#define EDGEXPU_GGUF_MAX_TENSORS 512
#define EDGEXPU_GGUF_TENSOR_NAME 96

typedef struct edgexpu_gguf_tensor {
    char name[EDGEXPU_GGUF_TENSOR_NAME];
    uint32_t n_dims;
    uint32_t type;     /* GGUF 量化类型，见 gguf_quant.h */
    uint64_t dims[4];
    uint64_t offset;   /* 相对 data_offset 的字节偏移 */
} edgexpu_gguf_tensor;

typedef struct edgexpu_gguf_info {
    char path[EDGEXPU_TEXT_LARGE];
    char architecture[EDGEXPU_TEXT_SMALL]; /* general.architecture，供 adapter 选择 */
    char name[EDGEXPU_TEXT_SMALL];
    char tokenizer_model[EDGEXPU_TEXT_SMALL];
    char tokenizer_pre[EDGEXPU_TEXT_SMALL];
    char chat_template[EDGEXPU_TEXT_LARGE]; /* GGUF tokenizer.chat_template；过长会截断 */
    uint32_t version;
    uint32_t block_count;
    uint32_t context_length;
    uint32_t embedding_length;
    uint32_t feed_forward_length;
    uint32_t head_count;
    uint32_t head_count_kv;
    uint32_t file_type;
    uint32_t eos_token_id;
    uint32_t bos_token_id;
    uint32_t pad_token_id;
    int add_bos_token;
    float rms_eps;
    float rope_freq_base;
    uint64_t tensor_count;
    uint64_t kv_count;
    uint64_t data_offset;
    uint64_t file_size;
    uint32_t n_tensors;
    edgexpu_gguf_tensor tensors[EDGEXPU_GGUF_MAX_TENSORS];
} edgexpu_gguf_info;

void edgexpu_gguf_info_init(edgexpu_gguf_info *info);

/* 打开 GGUF，填充 info；tokenizer 非空时同时加载 tokens/merges。 */
int edgexpu_gguf_load(
    const char *path,
    edgexpu_gguf_info *info,
    edgexpu_tokenizer *tokenizer,
    char *error,
    size_t error_size
);

/* embedding_length / head_count。GQA 时 KV 头更少，head_dim 仍按 query 头计算。 */
int edgexpu_gguf_head_dim(const edgexpu_gguf_info *info);

const edgexpu_gguf_tensor *edgexpu_gguf_find_tensor(
    const edgexpu_gguf_info *info,
    const char *name
);

#ifdef __cplusplus
}
#endif

#endif
