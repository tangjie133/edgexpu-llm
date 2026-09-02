#ifndef EDGEXPU_TOKENIZER_H
#define EDGEXPU_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EDGEXPU_TOKENIZER_MAX_VOCAB 200000
#define EDGEXPU_TOKENIZER_MAX_MERGES 200000
#define EDGEXPU_TOKENIZER_HASH_CAP 262144
#define EDGEXPU_TOKENIZER_BLOB_CAP (32u * 1024u * 1024u)

typedef struct edgexpu_tokenizer {
    uint32_t vocab_size;
    uint32_t n_merges;
    uint32_t bos_token_id;
    uint32_t eos_token_id;
    uint32_t pad_token_id;
    int add_bos_token;
    int ready;
    char *blob;
    size_t blob_used;
    uint32_t *offsets;
    uint32_t *vocab_hash;
    char *merge_blob;
    size_t merge_blob_used;
    uint32_t *merge_line_offsets;
    uint32_t *merge_left;
    uint32_t *merge_right;
    uint32_t *merge_result;
    uint32_t *merge_rank;
    uint32_t *pair_hash_left;
    uint32_t *pair_hash_right;
    uint32_t *pair_hash_rank;
    uint32_t *pair_hash_result;
    uint8_t *pair_hash_used;
} edgexpu_tokenizer;

void edgexpu_tokenizer_init(edgexpu_tokenizer *tokenizer);
void edgexpu_tokenizer_free(edgexpu_tokenizer *tokenizer);

int edgexpu_tokenizer_prepare(edgexpu_tokenizer *tokenizer, char *error, size_t error_size);
int edgexpu_tokenizer_append_piece(edgexpu_tokenizer *tokenizer, const char *piece, size_t len);
int edgexpu_tokenizer_append_merge_line(edgexpu_tokenizer *tokenizer, const char *line, size_t len);

int edgexpu_tokenizer_build_index(edgexpu_tokenizer *tokenizer, char *error, size_t error_size);

const char *edgexpu_tokenizer_piece(const edgexpu_tokenizer *tokenizer, uint32_t id);

int edgexpu_tokenizer_encode(
    const edgexpu_tokenizer *tokenizer,
    const char *text,
    uint32_t *ids,
    int max_ids,
    int *id_count,
    char *error,
    size_t error_size
);

int edgexpu_tokenizer_decode(
    const edgexpu_tokenizer *tokenizer,
    const uint32_t *ids,
    int id_count,
    char *output,
    size_t output_size
);

#ifdef __cplusplus
}
#endif

#endif
