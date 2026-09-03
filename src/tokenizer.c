#include "edgexpu/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* GPT-2 BPE：bytes-to-unicode + merge 表。特殊 token 以 `<...>` 整段匹配 vocab。 */

#define EDGEXPU_BPE_MAX_PIECES 256
#define EDGEXPU_MERGE_BLOB_CAP (12u * 1024u * 1024u)

static uint16_t gpt2_byte_to_cp[256];
static int16_t gpt2_cp_to_byte[512];
static int gpt2_tables_ready;

static void set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
    }
}

static uint32_t hash_bytes(const char *s, size_t len) {
    uint32_t hash = 2166136261u;
    size_t i;
    for (i = 0; i < len; i++) {
        hash ^= (unsigned char)s[i];
        hash *= 16777619u;
    }
    return hash;
}

static uint32_t hash_pair(uint32_t left, uint32_t right) {
    uint32_t hash = 2166136261u;
    hash ^= left;
    hash *= 16777619u;
    hash ^= right;
    hash *= 16777619u;
    return hash;
}

static size_t utf8_len(unsigned char lead) {
    if (lead < 0x80) {
        return 1;
    }
    if ((lead & 0xE0) == 0xC0) {
        return 2;
    }
    if ((lead & 0xF0) == 0xE0) {
        return 3;
    }
    if ((lead & 0xF8) == 0xF0) {
        return 4;
    }
    return 1;
}

/* HuggingFace GPT-2 bytes_to_unicode：可打印字节恒等，其余映射到 U+0100 起。 */
static void gpt2_init_tables(void) {
    int b;
    int n;
    uint8_t used[256];

    if (gpt2_tables_ready) {
        return;
    }
    memset(used, 0, sizeof(used));
    memset(gpt2_cp_to_byte, 0xff, sizeof(gpt2_cp_to_byte));
    for (b = 33; b <= 126; b++) {
        gpt2_byte_to_cp[b] = (uint16_t)b;
        gpt2_cp_to_byte[b] = (int16_t)b;
        used[b] = 1;
    }
    for (b = 161; b <= 172; b++) {
        gpt2_byte_to_cp[b] = (uint16_t)b;
        gpt2_cp_to_byte[b] = (int16_t)b;
        used[b] = 1;
    }
    for (b = 174; b <= 255; b++) {
        gpt2_byte_to_cp[b] = (uint16_t)b;
        gpt2_cp_to_byte[b] = (int16_t)b;
        used[b] = 1;
    }
    n = 0;
    for (b = 0; b < 256; b++) {
        if (!used[b]) {
            uint16_t cp = (uint16_t)(256 + n);
            gpt2_byte_to_cp[b] = cp;
            gpt2_cp_to_byte[cp] = (int16_t)b;
            n++;
        }
    }
    gpt2_tables_ready = 1;
}

static size_t utf8_write_cp(uint32_t cp, char *out) {
    if (cp < 0x80u) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800u) {
        out[0] = (char)(0xC0u | (cp >> 6));
        out[1] = (char)(0x80u | (cp & 0x3Fu));
        return 2;
    }
    if (cp < 0x10000u) {
        out[0] = (char)(0xE0u | (cp >> 12));
        out[1] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
        out[2] = (char)(0x80u | (cp & 0x3Fu));
        return 3;
    }
    out[0] = (char)(0xF0u | (cp >> 18));
    out[1] = (char)(0x80u | ((cp >> 12) & 0x3Fu));
    out[2] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
    out[3] = (char)(0x80u | (cp & 0x3Fu));
    return 4;
}

static uint32_t utf8_read_cp(const char **cursor, const char *end, const char **start) {
    const unsigned char *p = (const unsigned char *)*cursor;
    size_t n;
    uint32_t cp;

    if (start != NULL) {
        *start = *cursor;
    }
    if (*cursor >= end) {
        return 0;
    }
    n = utf8_len(p[0]);
    if (*cursor + n > end) {
        n = 1;
    }
    if (n == 1) {
        cp = p[0];
    } else if (n == 2) {
        cp = ((uint32_t)(p[0] & 0x1Fu) << 6) | (uint32_t)(p[1] & 0x3Fu);
    } else if (n == 3) {
        cp = ((uint32_t)(p[0] & 0x0Fu) << 12) |
            ((uint32_t)(p[1] & 0x3Fu) << 6) |
            (uint32_t)(p[2] & 0x3Fu);
    } else {
        cp = ((uint32_t)(p[0] & 0x07u) << 18) |
            ((uint32_t)(p[1] & 0x3Fu) << 12) |
            ((uint32_t)(p[2] & 0x3Fu) << 6) |
            (uint32_t)(p[3] & 0x3Fu);
    }
    *cursor += n;
    return cp;
}

static int gpt2_map_bytes(const char *src, size_t len, char *dst, size_t dst_size, size_t *out_len) {
    size_t i;
    size_t used = 0;

    gpt2_init_tables();
    for (i = 0; i < len; i++) {
        char tmp[4];
        size_t n = utf8_write_cp(gpt2_byte_to_cp[(unsigned char)src[i]], tmp);
        if (used + n >= dst_size) {
            return 0;
        }
        memcpy(dst + used, tmp, n);
        used += n;
    }
    dst[used] = '\0';
    if (out_len != NULL) {
        *out_len = used;
    }
    return 1;
}

void edgexpu_tokenizer_init(edgexpu_tokenizer *tokenizer) {
    if (tokenizer == NULL) {
        return;
    }
    memset(tokenizer, 0, sizeof(*tokenizer));
}

void edgexpu_tokenizer_free(edgexpu_tokenizer *tokenizer) {
    if (tokenizer == NULL) {
        return;
    }
    free(tokenizer->blob);
    free(tokenizer->offsets);
    free(tokenizer->vocab_hash);
    free(tokenizer->merge_blob);
    free(tokenizer->merge_line_offsets);
    free(tokenizer->merge_left);
    free(tokenizer->merge_right);
    free(tokenizer->merge_result);
    free(tokenizer->merge_rank);
    free(tokenizer->pair_hash_left);
    free(tokenizer->pair_hash_right);
    free(tokenizer->pair_hash_rank);
    free(tokenizer->pair_hash_result);
    free(tokenizer->pair_hash_used);
    memset(tokenizer, 0, sizeof(*tokenizer));
}

int edgexpu_tokenizer_prepare(edgexpu_tokenizer *tokenizer, char *error, size_t error_size) {
    if (tokenizer == NULL) {
        set_error(error, error_size, "tokenizer 为空");
        return 0;
    }

    edgexpu_tokenizer_free(tokenizer);
    tokenizer->blob = (char *)malloc(EDGEXPU_TOKENIZER_BLOB_CAP);
    tokenizer->offsets = (uint32_t *)calloc(EDGEXPU_TOKENIZER_MAX_VOCAB, sizeof(uint32_t));
    tokenizer->merge_blob = (char *)malloc(EDGEXPU_MERGE_BLOB_CAP);
    tokenizer->merge_line_offsets = (uint32_t *)calloc(EDGEXPU_TOKENIZER_MAX_MERGES, sizeof(uint32_t));
    if (tokenizer->blob == NULL || tokenizer->offsets == NULL ||
        tokenizer->merge_blob == NULL || tokenizer->merge_line_offsets == NULL) {
        edgexpu_tokenizer_free(tokenizer);
        set_error(error, error_size, "tokenizer 缓冲区分配失败");
        return 0;
    }
    tokenizer->blob_used = 0;
    tokenizer->merge_blob_used = 0;
    tokenizer->vocab_size = 0;
    tokenizer->n_merges = 0;
    return 1;
}

int edgexpu_tokenizer_append_piece(edgexpu_tokenizer *tokenizer, const char *piece, size_t len) {
    if (tokenizer == NULL || tokenizer->blob == NULL || piece == NULL) {
        return 0;
    }
    if (tokenizer->vocab_size >= EDGEXPU_TOKENIZER_MAX_VOCAB) {
        return 0;
    }
    if (tokenizer->blob_used + len + 1 > EDGEXPU_TOKENIZER_BLOB_CAP) {
        return 0;
    }
    tokenizer->offsets[tokenizer->vocab_size] = (uint32_t)tokenizer->blob_used;
    memcpy(tokenizer->blob + tokenizer->blob_used, piece, len);
    tokenizer->blob[tokenizer->blob_used + len] = '\0';
    tokenizer->blob_used += len + 1;
    tokenizer->vocab_size++;
    return 1;
}

int edgexpu_tokenizer_append_merge_line(edgexpu_tokenizer *tokenizer, const char *line, size_t len) {
    if (tokenizer == NULL || tokenizer->merge_blob == NULL || line == NULL) {
        return 0;
    }
    if (tokenizer->n_merges >= EDGEXPU_TOKENIZER_MAX_MERGES) {
        return 0;
    }
    if (tokenizer->merge_blob_used + len + 1 > EDGEXPU_MERGE_BLOB_CAP) {
        return 0;
    }
    tokenizer->merge_line_offsets[tokenizer->n_merges] = (uint32_t)tokenizer->merge_blob_used;
    memcpy(tokenizer->merge_blob + tokenizer->merge_blob_used, line, len);
    tokenizer->merge_blob[tokenizer->merge_blob_used + len] = '\0';
    tokenizer->merge_blob_used += len + 1;
    tokenizer->n_merges++;
    return 1;
}

const char *edgexpu_tokenizer_piece(const edgexpu_tokenizer *tokenizer, uint32_t id) {
    if (tokenizer == NULL || tokenizer->blob == NULL || tokenizer->offsets == NULL || id >= tokenizer->vocab_size) {
        return "";
    }
    return tokenizer->blob + tokenizer->offsets[id];
}

static int vocab_lookup(
    const edgexpu_tokenizer *tokenizer,
    const char *piece,
    size_t len,
    uint32_t *id
) {
    uint32_t hash;
    uint32_t slot;
    uint32_t i;

    if (tokenizer == NULL || tokenizer->vocab_hash == NULL || piece == NULL) {
        return 0;
    }

    hash = hash_bytes(piece, len);
    for (i = 0; i < EDGEXPU_TOKENIZER_HASH_CAP; i++) {
        slot = (hash + i) & (EDGEXPU_TOKENIZER_HASH_CAP - 1);
        if (tokenizer->vocab_hash[slot] == 0xFFFFFFFFu) {
            return 0;
        }
        if (strncmp(edgexpu_tokenizer_piece(tokenizer, tokenizer->vocab_hash[slot]), piece, len) == 0 &&
            edgexpu_tokenizer_piece(tokenizer, tokenizer->vocab_hash[slot])[len] == '\0') {
            if (id != NULL) {
                *id = tokenizer->vocab_hash[slot];
            }
            return 1;
        }
    }
    return 0;
}

/* 从 `<` 扫到 `>`，整段在 vocab 中则当作特殊 token（不走 byte mapping）。 */
static int try_angle_token(
    const edgexpu_tokenizer *tokenizer,
    const char *cursor,
    uint32_t *id,
    size_t *len
) {
    const char *end;
    size_t token_len;

    if (cursor == NULL || cursor[0] != '<') {
        return 0;
    }
    end = strchr(cursor, '>');
    if (end == NULL) {
        return 0;
    }
    token_len = (size_t)(end - cursor + 1);
    if (token_len < 2 || token_len > 64) {
        return 0;
    }
    if (!vocab_lookup(tokenizer, cursor, token_len, id)) {
        return 0;
    }
    if (len != NULL) {
        *len = token_len;
    }
    return 1;
}

static int pair_lookup(
    const edgexpu_tokenizer *tokenizer,
    uint32_t left,
    uint32_t right,
    uint32_t *rank,
    uint32_t *result
) {
    uint32_t hash = hash_pair(left, right);
    uint32_t i;
    uint32_t slot;

    if (tokenizer->pair_hash_used == NULL) {
        return 0;
    }

    for (i = 0; i < EDGEXPU_TOKENIZER_HASH_CAP; i++) {
        slot = (hash + i) & (EDGEXPU_TOKENIZER_HASH_CAP - 1);
        if (!tokenizer->pair_hash_used[slot]) {
            return 0;
        }
        if (tokenizer->pair_hash_left[slot] == left && tokenizer->pair_hash_right[slot] == right) {
            if (rank != NULL) {
                *rank = tokenizer->pair_hash_rank[slot];
            }
            if (result != NULL) {
                *result = tokenizer->pair_hash_result[slot];
            }
            return 1;
        }
    }
    return 0;
}

int edgexpu_tokenizer_build_index(edgexpu_tokenizer *tokenizer, char *error, size_t error_size) {
    uint32_t i;
    uint32_t slot;
    uint32_t probe;

    if (tokenizer == NULL || tokenizer->blob == NULL) {
        set_error(error, error_size, "tokenizer 尚未加载 vocab");
        return 0;
    }

    tokenizer->vocab_hash = (uint32_t *)malloc(EDGEXPU_TOKENIZER_HASH_CAP * sizeof(uint32_t));
    tokenizer->merge_left = (uint32_t *)calloc(tokenizer->n_merges, sizeof(uint32_t));
    tokenizer->merge_right = (uint32_t *)calloc(tokenizer->n_merges, sizeof(uint32_t));
    tokenizer->merge_result = (uint32_t *)calloc(tokenizer->n_merges, sizeof(uint32_t));
    tokenizer->merge_rank = (uint32_t *)calloc(tokenizer->n_merges, sizeof(uint32_t));
    tokenizer->pair_hash_left = (uint32_t *)calloc(EDGEXPU_TOKENIZER_HASH_CAP, sizeof(uint32_t));
    tokenizer->pair_hash_right = (uint32_t *)calloc(EDGEXPU_TOKENIZER_HASH_CAP, sizeof(uint32_t));
    tokenizer->pair_hash_rank = (uint32_t *)calloc(EDGEXPU_TOKENIZER_HASH_CAP, sizeof(uint32_t));
    tokenizer->pair_hash_result = (uint32_t *)calloc(EDGEXPU_TOKENIZER_HASH_CAP, sizeof(uint32_t));
    tokenizer->pair_hash_used = (uint8_t *)calloc(EDGEXPU_TOKENIZER_HASH_CAP, sizeof(uint8_t));
    if (tokenizer->vocab_hash == NULL || tokenizer->pair_hash_used == NULL) {
        set_error(error, error_size, "tokenizer 索引分配失败");
        return 0;
    }

    memset(tokenizer->vocab_hash, 0xFF, EDGEXPU_TOKENIZER_HASH_CAP * sizeof(uint32_t));
    for (i = 0; i < tokenizer->vocab_size; i++) {
        const char *piece = edgexpu_tokenizer_piece(tokenizer, i);
        uint32_t hash = hash_bytes(piece, strlen(piece));
        for (probe = 0; probe < EDGEXPU_TOKENIZER_HASH_CAP; probe++) {
            slot = (hash + probe) & (EDGEXPU_TOKENIZER_HASH_CAP - 1);
            if (tokenizer->vocab_hash[slot] == 0xFFFFFFFFu) {
                tokenizer->vocab_hash[slot] = i;
                break;
            }
        }
    }

    for (i = 0; i < tokenizer->n_merges; i++) {
        const char *line = tokenizer->merge_blob + tokenizer->merge_line_offsets[i];
        const char *space = strchr(line, ' ');
        char left[256];
        char right[256];
        char joined[512];
        uint32_t left_id = 0;
        uint32_t right_id = 0;
        uint32_t result_id = 0;
        size_t left_len;
        size_t right_len;

        if (space == NULL) {
            continue;
        }
        left_len = (size_t)(space - line);
        right_len = strlen(space + 1);
        if (left_len >= sizeof(left) || right_len >= sizeof(right)) {
            continue;
        }
        memcpy(left, line, left_len);
        left[left_len] = '\0';
        memcpy(right, space + 1, right_len + 1);
        if (!vocab_lookup(tokenizer, left, left_len, &left_id) ||
            !vocab_lookup(tokenizer, right, right_len, &right_id)) {
            continue;
        }
        snprintf(joined, sizeof(joined), "%s%s", left, right);
        if (!vocab_lookup(tokenizer, joined, strlen(joined), &result_id)) {
            continue;
        }

        tokenizer->merge_left[i] = left_id;
        tokenizer->merge_right[i] = right_id;
        tokenizer->merge_result[i] = result_id;
        tokenizer->merge_rank[i] = i;

        {
            uint32_t hash = hash_pair(left_id, right_id);
            for (probe = 0; probe < EDGEXPU_TOKENIZER_HASH_CAP; probe++) {
                slot = (hash + probe) & (EDGEXPU_TOKENIZER_HASH_CAP - 1);
                if (!tokenizer->pair_hash_used[slot]) {
                    tokenizer->pair_hash_used[slot] = 1;
                    tokenizer->pair_hash_left[slot] = left_id;
                    tokenizer->pair_hash_right[slot] = right_id;
                    tokenizer->pair_hash_rank[slot] = i;
                    tokenizer->pair_hash_result[slot] = result_id;
                    break;
                }
            }
        }
    }

    tokenizer->ready = tokenizer->vocab_size > 0;
    return tokenizer->ready;
}

/* 对单个“词”做贪心 BPE merge。word 已是 GPT-2 unicode 映射后的字节串。 */
static int encode_word(
    const edgexpu_tokenizer *tokenizer,
    const char *word,
    size_t word_len,
    uint32_t *ids,
    int max_ids,
    int *written
) {
    uint32_t pieces[EDGEXPU_BPE_MAX_PIECES];
    int n = 0;
    size_t cursor = 0;
    int changed = 1;

    while (cursor < word_len && n < EDGEXPU_BPE_MAX_PIECES) {
        size_t char_len = utf8_len((unsigned char)word[cursor]);
        uint32_t id = 0;
        if (cursor + char_len > word_len) {
            char_len = 1;
        }
        if (!vocab_lookup(tokenizer, word + cursor, char_len, &id)) {
            if (!vocab_lookup(tokenizer, word + cursor, 1, &id)) {
                id = tokenizer->pad_token_id;
            }
            char_len = 1;
        }
        pieces[n++] = id;
        cursor += char_len;
    }

    while (changed && n > 1) {
        uint32_t best_rank = 0xFFFFFFFFu;
        int best_i = -1;
        uint32_t best_result = 0;
        int i;
        changed = 0;
        for (i = 0; i < n - 1; i++) {
            uint32_t rank = 0;
            uint32_t result = 0;
            if (pair_lookup(tokenizer, pieces[i], pieces[i + 1], &rank, &result) && rank < best_rank) {
                best_rank = rank;
                best_i = i;
                best_result = result;
            }
        }
        if (best_i < 0) {
            break;
        }
        pieces[best_i] = best_result;
        memmove(&pieces[best_i + 1], &pieces[best_i + 2], (size_t)(n - best_i - 2) * sizeof(uint32_t));
        n--;
        changed = 1;
    }

    if (*written + n > max_ids) {
        return 0;
    }
    memcpy(ids + *written, pieces, (size_t)n * sizeof(uint32_t));
    *written += n;
    return 1;
}

#define EDGEXPU_PRE_MAX_CPTS 8192

typedef enum {
    EDGEXPU_PRE_GPT2 = 0,
    EDGEXPU_PRE_QWEN2 = 1,
    EDGEXPU_PRE_LLAMA3 = 2
} edgexpu_pre_kind;

static edgexpu_pre_kind pre_kind_from_name(const char *pre) {
    if (pre == NULL || pre[0] == '\0' ||
        strcmp(pre, "default") == 0 ||
        strcmp(pre, "gpt2") == 0 ||
        strcmp(pre, "olmo") == 0 ||
        strcmp(pre, "jais") == 0) {
        return EDGEXPU_PRE_GPT2;
    }
    if (strcmp(pre, "llama3") == 0 ||
        strcmp(pre, "llama-v3") == 0 ||
        strcmp(pre, "llama-bpe") == 0) {
        return EDGEXPU_PRE_LLAMA3;
    }
    /* qwen2 / smollm / stablelm2：单个数字，可选前缀标点 + 字母。 */
    return EDGEXPU_PRE_QWEN2;
}

static uint32_t utf8_next_cp(const char *s, const char *end, size_t *nbytes) {
    unsigned char lead;
    size_t n;
    uint32_t cp;

    if (s >= end) {
        *nbytes = 0;
        return 0;
    }
    lead = (unsigned char)s[0];
    n = utf8_len(lead);
    if (s + n > end) {
        n = 1;
    }
    if (n == 1) {
        cp = lead;
    } else if (n == 2) {
        cp = ((uint32_t)(lead & 0x1Fu) << 6) | (uint32_t)((unsigned char)s[1] & 0x3Fu);
    } else if (n == 3) {
        cp = ((uint32_t)(lead & 0x0Fu) << 12) |
            ((uint32_t)((unsigned char)s[1] & 0x3Fu) << 6) |
            (uint32_t)((unsigned char)s[2] & 0x3Fu);
    } else {
        cp = ((uint32_t)(lead & 0x07u) << 18) |
            ((uint32_t)((unsigned char)s[1] & 0x3Fu) << 12) |
            ((uint32_t)((unsigned char)s[2] & 0x3Fu) << 6) |
            (uint32_t)((unsigned char)s[3] & 0x3Fu);
    }
    *nbytes = n;
    return cp;
}

static int cpt_is_whitespace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
        cp == 0x0Bu || cp == 0x0Cu || cp == 0xA0u;
}

static int cpt_is_number(uint32_t cp) {
    return cp >= '0' && cp <= '9';
}

static int cpt_is_letter(uint32_t cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) {
        return 1;
    }
    if (cp < 0x80u) {
        return 0;
    }
    if (cpt_is_number(cp) || cpt_is_whitespace(cp)) {
        return 0;
    }
    if (cp == 0xD7u || cp == 0xF7u) {
        return 0;
    }
    if (cp >= 0xC0u && cp < 0x100u) {
        return 1;
    }
    if ((cp >= 0x2000u && cp <= 0x206Fu) ||
        (cp >= 0x3000u && cp <= 0x303Fu) ||
        (cp >= 0xFF01u && cp <= 0xFF0Fu) ||
        (cp >= 0xFF1Au && cp <= 0xFF20u) ||
        (cp >= 0xFF3Bu && cp <= 0xFF40u) ||
        (cp >= 0xFF5Bu && cp <= 0xFF65u)) {
        return 0;
    }
    return 1;
}

static uint32_t ascii_tolower(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') {
        return cp - (uint32_t)('A' - 'a');
    }
    return cp;
}

static int decode_codepoints(
    const char *text,
    size_t text_len,
    uint32_t *cps,
    size_t *byte_off,
    int max_n,
    int *n_out
) {
    const char *end = text + text_len;
    const char *p = text;
    int n = 0;

    while (p < end && n < max_n) {
        size_t nbytes = 0;
        cps[n] = utf8_next_cp(p, end, &nbytes);
        if (nbytes == 0) {
            break;
        }
        byte_off[n] = (size_t)(p - text);
        p += nbytes;
        n++;
    }
    *n_out = n;
    return p >= end;
}

static uint32_t cpt_at(const uint32_t *cps, int n, int pos) {
    if (pos < 0 || pos >= n) {
        return 0xFFFFFFFFu;
    }
    return cps[pos];
}

static int flags_letter(const uint32_t *cps, int n, int pos) {
    return pos >= 0 && pos < n && cpt_is_letter(cps[pos]);
}

static int flags_number(const uint32_t *cps, int n, int pos) {
    return pos >= 0 && pos < n && cpt_is_number(cps[pos]);
}

static int flags_ws(const uint32_t *cps, int n, int pos) {
    return pos >= 0 && pos < n && cpt_is_whitespace(cps[pos]);
}

/* 对照 llama.cpp unicode_regex_split_custom_{gpt2,qwen2,llama3}。 */
static int pretok_split(
    edgexpu_pre_kind kind,
    const uint32_t *cps,
    int n,
    int *starts,
    int *ends,
    int max_words
) {
    int pos = 0;
    int n_words = 0;
    const uint32_t OOR = 0xFFFFFFFFu;

    while (pos < n && n_words < max_words) {
        uint32_t cpt = cps[pos];
        int start = pos;

        if (cpt == '\'' && pos + 1 < n) {
            uint32_t next = cps[pos + 1];
            uint32_t next_l = (kind == EDGEXPU_PRE_GPT2) ? next : ascii_tolower(next);
            if (next_l == 's' || next_l == 't' || next_l == 'm' || next_l == 'd') {
                starts[n_words] = start;
                ends[n_words] = pos + 2;
                n_words++;
                pos += 2;
                continue;
            }
            if (pos + 2 < n) {
                uint32_t n2 = cps[pos + 2];
                uint32_t n2_l = (kind == EDGEXPU_PRE_GPT2) ? n2 : ascii_tolower(n2);
                if ((next_l == 'r' && n2_l == 'e') ||
                    (next_l == 'v' && n2_l == 'e') ||
                    (next_l == 'l' && n2_l == 'l')) {
                    starts[n_words] = start;
                    ends[n_words] = pos + 3;
                    n_words++;
                    pos += 3;
                    continue;
                }
            }
        }

        if (kind == EDGEXPU_PRE_GPT2) {
            int letter_pos = (cpt == ' ' && pos + 1 < n) ? pos + 1 : pos;
            if (flags_letter(cps, n, letter_pos)) {
                pos = letter_pos;
                while (flags_letter(cps, n, pos)) {
                    pos++;
                }
                starts[n_words] = start;
                ends[n_words] = pos;
                n_words++;
                continue;
            }
            if (flags_number(cps, n, letter_pos)) {
                pos = letter_pos;
                while (flags_number(cps, n, pos)) {
                    pos++;
                }
                starts[n_words] = start;
                ends[n_words] = pos;
                n_words++;
                continue;
            }
            if (!flags_ws(cps, n, letter_pos) &&
                !flags_letter(cps, n, letter_pos) &&
                !flags_number(cps, n, letter_pos) &&
                letter_pos < n) {
                pos = letter_pos;
                while (pos < n &&
                       !flags_ws(cps, n, pos) &&
                       !flags_letter(cps, n, pos) &&
                       !flags_number(cps, n, pos)) {
                    pos++;
                }
                starts[n_words] = start;
                ends[n_words] = pos;
                n_words++;
                continue;
            }
        } else {
            if (!(cpt == '\r' || cpt == '\n' || cpt_is_number(cpt))) {
                if (cpt_is_letter(cpt) || flags_letter(cps, n, pos + 1)) {
                    pos++;
                    while (flags_letter(cps, n, pos)) {
                        pos++;
                    }
                    starts[n_words] = start;
                    ends[n_words] = pos;
                    n_words++;
                    continue;
                }
            }
            if (cpt_is_number(cpt)) {
                if (kind == EDGEXPU_PRE_LLAMA3) {
                    int ini = pos;
                    while (flags_number(cps, n, pos)) {
                        pos++;
                        if (pos - ini >= 3) {
                            starts[n_words] = ini;
                            ends[n_words] = pos;
                            n_words++;
                            if (n_words >= max_words) {
                                return n_words;
                            }
                            ini = pos;
                        }
                    }
                    if (pos > ini) {
                        starts[n_words] = ini;
                        ends[n_words] = pos;
                        n_words++;
                    }
                } else {
                    starts[n_words] = start;
                    ends[n_words] = pos + 1;
                    n_words++;
                    pos++;
                }
                continue;
            }
            {
                int punct_pos = (cpt == ' ' && pos + 1 < n) ? pos + 1 : pos;
                if (punct_pos < n &&
                    !flags_ws(cps, n, punct_pos) &&
                    !flags_letter(cps, n, punct_pos) &&
                    !flags_number(cps, n, punct_pos)) {
                    pos = punct_pos;
                    while (pos < n &&
                           !flags_ws(cps, n, pos) &&
                           !flags_letter(cps, n, pos) &&
                           !flags_number(cps, n, pos)) {
                        pos++;
                    }
                    while (pos < n && (cps[pos] == '\r' || cps[pos] == '\n')) {
                        pos++;
                    }
                    starts[n_words] = start;
                    ends[n_words] = pos;
                    n_words++;
                    continue;
                }
            }
        }

        {
            int num_ws = 0;
            int last_nl = 0;
            while (flags_ws(cps, n, pos + num_ws)) {
                uint32_t w = cps[pos + num_ws];
                if (w == '\r' || w == '\n') {
                    last_nl = pos + num_ws + 1;
                }
                num_ws++;
            }
            if (kind != EDGEXPU_PRE_GPT2 && last_nl > 0) {
                starts[n_words] = start;
                ends[n_words] = last_nl;
                n_words++;
                pos = last_nl;
                continue;
            }
            if (num_ws > 1 && cpt_at(cps, n, pos + num_ws) != OOR) {
                pos += num_ws - 1;
                starts[n_words] = start;
                ends[n_words] = pos;
                n_words++;
                continue;
            }
            if (num_ws > 0) {
                pos += num_ws;
                starts[n_words] = start;
                ends[n_words] = pos;
                n_words++;
                continue;
            }
        }

        starts[n_words] = start;
        ends[n_words] = pos + 1;
        n_words++;
        pos++;
    }

    return n_words;
}

static int encode_chunk(
    const edgexpu_tokenizer *tokenizer,
    const char *chunk,
    size_t chunk_len,
    uint32_t *ids,
    int max_ids,
    int *written
) {
    uint32_t cps[EDGEXPU_PRE_MAX_CPTS];
    size_t byte_off[EDGEXPU_PRE_MAX_CPTS];
    int starts[EDGEXPU_PRE_MAX_CPTS];
    int ends[EDGEXPU_PRE_MAX_CPTS];
    int n_cps = 0;
    int n_words;
    int w;
    edgexpu_pre_kind kind = pre_kind_from_name(tokenizer->pre);

    if (chunk_len == 0) {
        return 1;
    }
    if (!decode_codepoints(chunk, chunk_len, cps, byte_off, EDGEXPU_PRE_MAX_CPTS, &n_cps)) {
        return 0;
    }
    n_words = pretok_split(kind, cps, n_cps, starts, ends, EDGEXPU_PRE_MAX_CPTS);
    for (w = 0; w < n_words; w++) {
        size_t b0 = byte_off[starts[w]];
        size_t b1 = (ends[w] < n_cps) ? byte_off[ends[w]] : chunk_len;
        char mapped[1024];
        size_t mapped_len = 0;
        if (b1 <= b0) {
            continue;
        }
        if (!gpt2_map_bytes(chunk + b0, b1 - b0, mapped, sizeof(mapped), &mapped_len) ||
            !encode_word(tokenizer, mapped, mapped_len, ids, max_ids, written)) {
            return 0;
        }
    }
    return 1;
}

/* 先抠特殊 token，再按 tokenizer.ggml.pre 做 GPT-2 族切词 + byte mapping + BPE。 */
int edgexpu_tokenizer_encode(
    const edgexpu_tokenizer *tokenizer,
    const char *text,
    uint32_t *ids,
    int max_ids,
    int *id_count,
    char *error,
    size_t error_size
) {
    const char *cursor;
    int written = 0;

    if (tokenizer == NULL || !tokenizer->ready || ids == NULL || max_ids <= 0 || id_count == NULL) {
        set_error(error, error_size, "native tokenizer 尚未就绪");
        return 0;
    }

    if (tokenizer->add_bos_token) {
        if (written >= max_ids) {
            set_error(error, error_size, "native tokenizer 输出过长");
            return 0;
        }
        ids[written++] = tokenizer->bos_token_id;
    }

    cursor = text != NULL ? text : "";
    while (*cursor != '\0') {
        uint32_t special_id = 0;
        size_t special_len = 0;
        const char *chunk_end;

        if (try_angle_token(tokenizer, cursor, &special_id, &special_len)) {
            if (written >= max_ids) {
                set_error(error, error_size, "native tokenizer 输出过长");
                return 0;
            }
            ids[written++] = special_id;
            cursor += special_len;
            continue;
        }

        chunk_end = cursor + 1;
        while (*chunk_end != '\0') {
            uint32_t ignore_id = 0;
            size_t ignore_len = 0;
            if (try_angle_token(tokenizer, chunk_end, &ignore_id, &ignore_len)) {
                break;
            }
            chunk_end++;
        }
        if (!encode_chunk(
                tokenizer,
                cursor,
                (size_t)(chunk_end - cursor),
                ids,
                max_ids,
                &written)) {
            set_error(error, error_size, "native tokenizer 输出过长");
            return 0;
        }
        cursor = chunk_end;
    }

    *id_count = written;
    return 1;
}

/* 把 vocab piece 的 unicode 码点映回原始字节；未映射码点保持 UTF-8。 */
int edgexpu_tokenizer_decode(
    const edgexpu_tokenizer *tokenizer,
    const uint32_t *ids,
    int id_count,
    char *output,
    size_t output_size
) {
    int i;
    size_t used = 0;

    if (output == NULL || output_size == 0) {
        return 0;
    }
    output[0] = '\0';
    if (tokenizer == NULL || ids == NULL) {
        return 0;
    }

    gpt2_init_tables();
    for (i = 0; i < id_count; i++) {
        const char *piece = edgexpu_tokenizer_piece(tokenizer, ids[i]);
        const char *cursor = piece;
        const char *end = piece + strlen(piece);
        while (cursor < end && used + 1 < output_size) {
            const char *ch_start = NULL;
            uint32_t cp = utf8_read_cp(&cursor, end, &ch_start);
            if (cp < 512u && gpt2_cp_to_byte[cp] >= 0) {
                output[used++] = (char)gpt2_cp_to_byte[cp];
            } else {
                while (ch_start < cursor && used + 1 < output_size) {
                    output[used++] = *ch_start++;
                }
            }
        }
    }
    output[used] = '\0';
    return 1;
}
