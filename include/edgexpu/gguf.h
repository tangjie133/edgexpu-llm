#ifndef EDGEXPU_GGUF_H
#define EDGEXPU_GGUF_H

#include <stddef.h>
#include <stdint.h>

#include "edgexpu/manifest.h"
#include "edgexpu/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define EDGEXPU_GGUF_MAX_TENSORS 512
#define EDGEXPU_GGUF_TENSOR_NAME 96

typedef struct edgexpu_gguf_tensor {
    char name[EDGEXPU_GGUF_TENSOR_NAME];
    uint32_t n_dims;
    uint32_t type;
    uint64_t dims[4];
    uint64_t offset;
} edgexpu_gguf_tensor;

typedef struct edgexpu_gguf_info {
    char path[EDGEXPU_TEXT_LARGE];
    char architecture[EDGEXPU_TEXT_SMALL];
    char name[EDGEXPU_TEXT_SMALL];
    char tokenizer_model[EDGEXPU_TEXT_SMALL];
    char tokenizer_pre[EDGEXPU_TEXT_SMALL];
    char chat_template[EDGEXPU_TEXT_LARGE];
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

int edgexpu_gguf_load(
    const char *path,
    edgexpu_gguf_info *info,
    edgexpu_tokenizer *tokenizer,
    char *error,
    size_t error_size
);

int edgexpu_gguf_head_dim(const edgexpu_gguf_info *info);

const edgexpu_gguf_tensor *edgexpu_gguf_find_tensor(
    const edgexpu_gguf_info *info,
    const char *name
);

#ifdef __cplusplus
}
#endif

#endif
