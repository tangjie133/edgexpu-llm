#include "edgexpu/gguf.h"
#include "edgexpu/gguf_quant.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* GGUF v3：读魔数、KV metadata、tensor 目录；可选加载 GPT-2 vocab/merges。 */

enum {
    GGUF_U8 = 0,
    GGUF_I8 = 1,
    GGUF_U16 = 2,
    GGUF_I16 = 3,
    GGUF_U32 = 4,
    GGUF_I32 = 5,
    GGUF_F32 = 6,
    GGUF_BOOL = 7,
    GGUF_STRING = 8,
    GGUF_ARRAY = 9,
    GGUF_U64 = 10,
    GGUF_I64 = 11,
    GGUF_F64 = 12
};

static const size_t k_gguf_scalar_size[] = {
    1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8
};

static void set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
    }
}

static void copy_text(char *output, size_t output_size, const char *input) {
    if (output != NULL && output_size > 0) {
        snprintf(output, output_size, "%s", input != NULL ? input : "");
    }
}

static int read_exact(FILE *file, void *buffer, size_t size) {
    return file != NULL && buffer != NULL && fread(buffer, 1, size, file) == size;
}

static int read_u32(FILE *file, uint32_t *value) {
    return read_exact(file, value, sizeof(*value));
}

static int read_u64(FILE *file, uint64_t *value) {
    return read_exact(file, value, sizeof(*value));
}

static int skip_bytes(FILE *file, uint64_t count) {
    if (file == NULL) {
        return 0;
    }
    if (count == 0) {
        return 1;
    }
    return fseek(file, (long)count, SEEK_CUR) == 0;
}

static int read_string_into(FILE *file, char *output, size_t output_size) {
    uint64_t length = 0;
    if (!read_u64(file, &length)) {
        return 0;
    }
    if (output != NULL && output_size > 0) {
        size_t copy = (size_t)length;
        if (copy >= output_size) {
            copy = output_size - 1;
        }
        if (copy > 0 && !read_exact(file, output, copy)) {
            return 0;
        }
        output[copy] = '\0';
        if (length > copy && !skip_bytes(file, length - copy)) {
            return 0;
        }
        return 1;
    }
    return skip_bytes(file, length);
}

static int skip_value(FILE *file, uint32_t type);

static void apply_gguf_u32_key(edgexpu_gguf_info *info, const char *key, uint32_t value) {
    if (strstr(key, "block_count") != NULL) {
        info->block_count = value;
    } else if (strstr(key, "context_length") != NULL) {
        info->context_length = value;
    } else if (strstr(key, "embedding_length") != NULL) {
        info->embedding_length = value;
    } else if (strstr(key, "feed_forward_length") != NULL) {
        info->feed_forward_length = value;
    } else if (strstr(key, "attention.head_count_kv") != NULL) {
        info->head_count_kv = value;
    } else if (strstr(key, "attention.head_count") != NULL) {
        info->head_count = value;
    } else if (strstr(key, "attention.key_length") != NULL) {
        info->attention_key_length = value;
    } else if (strstr(key, "attention.value_length") != NULL) {
        info->attention_value_length = value;
    } else if (strstr(key, "full_attention_interval") != NULL) {
        info->full_attention_interval = value;
    } else if (strstr(key, "ssm.conv_kernel") != NULL) {
        info->ssm_conv_kernel = value;
    } else if (strstr(key, "ssm.inner_size") != NULL) {
        info->ssm_inner_size = value;
    } else if (strstr(key, "ssm.state_size") != NULL) {
        info->ssm_state_size = value;
    } else if (strstr(key, "ssm.time_step_rank") != NULL) {
        info->ssm_time_step_rank = value;
    } else if (strstr(key, "ssm.group_count") != NULL) {
        info->ssm_group_count = value;
    } else if (strstr(key, "rope.dimension_count") != NULL) {
        info->rope_dimension_count = value;
    } else if (strcmp(key, "general.file_type") == 0) {
        info->file_type = value;
    } else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) {
        info->eos_token_id = value;
    } else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0) {
        info->bos_token_id = value;
    } else if (strcmp(key, "tokenizer.ggml.padding_token_id") == 0) {
        info->pad_token_id = value;
    }
}

static int skip_array(FILE *file) {
    uint32_t element_type = 0;
    uint64_t count = 0;
    uint64_t index;

    if (!read_u32(file, &element_type) || !read_u64(file, &count)) {
        return 0;
    }
    if (element_type == GGUF_STRING) {
        for (index = 0; index < count; index++) {
            uint64_t length = 0;
            if (!read_u64(file, &length) || !skip_bytes(file, length)) {
                return 0;
            }
        }
        return 1;
    }
    if (element_type > GGUF_F64 || k_gguf_scalar_size[element_type] == 0) {
        return 0;
    }
    return skip_bytes(file, count * k_gguf_scalar_size[element_type]);
}

static int skip_value(FILE *file, uint32_t type) {
    if (type == GGUF_STRING) {
        uint64_t length = 0;
        return read_u64(file, &length) && skip_bytes(file, length);
    }
    if (type == GGUF_ARRAY) {
        return skip_array(file);
    }
    if (type > GGUF_F64 || k_gguf_scalar_size[type] == 0) {
        return 0;
    }
    return skip_bytes(file, k_gguf_scalar_size[type]);
}

static int load_string_array(
    FILE *file,
    int is_merge,
    edgexpu_tokenizer *tokenizer
) {
    uint32_t element_type = 0;
    uint64_t count = 0;
    uint64_t index;
    char *buffer;

    if (!read_u32(file, &element_type) || !read_u64(file, &count) || element_type != GGUF_STRING) {
        return 0;
    }
    if (count > 0xFFFFFFFFu) {
        return 0;
    }
    if (is_merge) {
        if (!edgexpu_tokenizer_reserve(tokenizer, tokenizer->vocab_size, (uint32_t)count, NULL, 0)) {
            return 0;
        }
    } else if (!edgexpu_tokenizer_reserve(tokenizer, (uint32_t)count, tokenizer->n_merges, NULL, 0)) {
        return 0;
    }

    buffer = (char *)malloc(65536);
    if (buffer == NULL) {
        return 0;
    }

    for (index = 0; index < count; index++) {
        uint64_t length = 0;
        if (!read_u64(file, &length)) {
            free(buffer);
            return 0;
        }
        if (length >= 65536) {
            if (!skip_bytes(file, length)) {
                free(buffer);
                return 0;
            }
            continue;
        }
        if (length > 0 && !read_exact(file, buffer, (size_t)length)) {
            free(buffer);
            return 0;
        }
        buffer[length] = '\0';
        if (is_merge) {
            if (!edgexpu_tokenizer_append_merge_line(tokenizer, buffer, (size_t)length)) {
                free(buffer);
                return 0;
            }
        } else if (!edgexpu_tokenizer_append_piece(tokenizer, buffer, (size_t)length)) {
            free(buffer);
            return 0;
        }
    }

    free(buffer);
    return 1;
}

void edgexpu_gguf_info_init(edgexpu_gguf_info *info) {
    if (info == NULL) {
        return;
    }
    free(info->tensors);
    memset(info, 0, sizeof(*info));
}

void edgexpu_gguf_info_free(edgexpu_gguf_info *info) {
    if (info == NULL) {
        return;
    }
    free(info->tensors);
    info->tensors = NULL;
    info->tensors_cap = 0;
    info->n_tensors = 0;
}

int edgexpu_gguf_head_dim(const edgexpu_gguf_info *info) {
    if (info == NULL || info->head_count == 0) {
        return 0;
    }
    return (int)(info->embedding_length / info->head_count);
}

int edgexpu_gguf_attn_head_dim(const edgexpu_gguf_info *info) {
    uint32_t index;
    if (info == NULL) {
        return 0;
    }
    if (info->attention_key_length > 0) {
        return (int)info->attention_key_length;
    }
    if (info->attention_value_length > 0) {
        return (int)info->attention_value_length;
    }
    for (index = 0; index < info->n_tensors; index++) {
        const edgexpu_gguf_tensor *tensor = &info->tensors[index];
        if (strstr(tensor->name, ".attn_k.weight") == NULL || tensor->n_dims < 2) {
            continue;
        }
        if (info->head_count_kv > 0 && tensor->dims[1] % info->head_count_kv == 0) {
            return (int)(tensor->dims[1] / info->head_count_kv);
        }
    }
    return edgexpu_gguf_head_dim(info);
}

const edgexpu_gguf_tensor *edgexpu_gguf_find_tensor(
    const edgexpu_gguf_info *info,
    const char *name
) {
    uint32_t index;
    if (info == NULL || name == NULL) {
        return NULL;
    }
    for (index = 0; index < info->n_tensors; index++) {
        if (strcmp(info->tensors[index].name, name) == 0) {
            return &info->tensors[index];
        }
    }
    return NULL;
}

int edgexpu_gguf_tensor_block_index(const char *name) {
    unsigned long index;
    char *end = NULL;
    if (name == NULL || strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    index = strtoul(name + 4, &end, 10);
    if (end == name + 4 || end == NULL || *end != '.') {
        return -1;
    }
    if (index > (unsigned long)INT_MAX) {
        return -1;
    }
    return (int)index;
}

size_t edgexpu_gguf_max_block_bytes(const edgexpu_gguf_info *info) {
    size_t *sums = NULL;
    size_t max_bytes = 0;
    uint32_t n_layers;
    uint32_t i;

    if (info == NULL || info->n_tensors == 0) {
        return 0;
    }
    n_layers = info->block_count;
    if (n_layers == 0) {
        for (i = 0; i < info->n_tensors; i++) {
            int block = edgexpu_gguf_tensor_block_index(info->tensors[i].name);
            if (block >= 0 && (uint32_t)(block + 1) > n_layers) {
                n_layers = (uint32_t)(block + 1);
            }
        }
    }
    if (n_layers == 0) {
        return 0;
    }
    sums = (size_t *)calloc((size_t)n_layers, sizeof(size_t));
    if (sums == NULL) {
        return 0;
    }
    for (i = 0; i < info->n_tensors; i++) {
        int block = edgexpu_gguf_tensor_block_index(info->tensors[i].name);
        size_t nbytes;
        if (block < 0 || (uint32_t)block >= n_layers) {
            continue;
        }
        nbytes = edgexpu_gguf_tensor_nbytes(&info->tensors[i]);
        if (nbytes == 0 || sums[block] > SIZE_MAX - nbytes) {
            continue;
        }
        sums[block] += nbytes;
    }
    for (i = 0; i < n_layers; i++) {
        if (sums[i] > max_bytes) {
            max_bytes = sums[i];
        }
    }
    free(sums);
    return max_bytes;
}

/* 顺序读 header → KV（含 architecture / tokenizer / chat_template）→ tensor 表 → 对齐 data_offset。 */
int edgexpu_gguf_load(
    const char *path,
    edgexpu_gguf_info *info,
    edgexpu_tokenizer *tokenizer,
    char *error,
    size_t error_size
) {
    FILE *file;
    char magic[4];
    uint64_t kv_index;
    uint64_t tensor_index;
    long tensor_end;
    char key[256];

    if (path == NULL || info == NULL) {
        set_error(error, error_size, "GGUF 加载参数为空");
        return 0;
    }

    edgexpu_gguf_info_init(info);
    copy_text(info->path, sizeof(info->path), path);
    file = fopen(path, "rb");
    if (file == NULL) {
        set_error(error, error_size, "无法打开 GGUF 文件");
        return 0;
    }

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        set_error(error, error_size, "无法读取 GGUF 文件大小");
        return 0;
    }
    info->file_size = (uint64_t)ftell(file);
    rewind(file);

    if (!read_exact(file, magic, 4) || memcmp(magic, "GGUF", 4) != 0) {
        fclose(file);
        set_error(error, error_size, "文件不是 GGUF 格式");
        return 0;
    }
    if (!read_u32(file, &info->version) ||
        !read_u64(file, &info->tensor_count) ||
        !read_u64(file, &info->kv_count)) {
        fclose(file);
        set_error(error, error_size, "GGUF 头读取失败");
        return 0;
    }

    if (tokenizer != NULL && !edgexpu_tokenizer_prepare(tokenizer, error, error_size)) {
        fclose(file);
        return 0;
    }

    for (kv_index = 0; kv_index < info->kv_count; kv_index++) {
        uint32_t type = 0;
        if (!read_string_into(file, key, sizeof(key)) || !read_u32(file, &type)) {
            fclose(file);
            set_error(error, error_size, "GGUF metadata 读取失败");
            return 0;
        }

        if (strcmp(key, "general.architecture") == 0 && type == GGUF_STRING) {
            if (!read_string_into(file, info->architecture, sizeof(info->architecture))) {
                fclose(file);
                return 0;
            }
        } else if (strcmp(key, "general.name") == 0 && type == GGUF_STRING) {
            if (!read_string_into(file, info->name, sizeof(info->name))) {
                fclose(file);
                return 0;
            }
        } else if (strcmp(key, "tokenizer.ggml.model") == 0 && type == GGUF_STRING) {
            if (!read_string_into(file, info->tokenizer_model, sizeof(info->tokenizer_model))) {
                fclose(file);
                return 0;
            }
        } else if (strcmp(key, "tokenizer.ggml.pre") == 0 && type == GGUF_STRING) {
            if (!read_string_into(file, info->tokenizer_pre, sizeof(info->tokenizer_pre))) {
                fclose(file);
                return 0;
            }
        } else if (strcmp(key, "tokenizer.chat_template") == 0 && type == GGUF_STRING) {
            if (!read_string_into(file, info->chat_template, sizeof(info->chat_template))) {
                fclose(file);
                return 0;
            }
        } else if (strcmp(key, "tokenizer.ggml.tokens") == 0 && type == GGUF_ARRAY && tokenizer != NULL) {
            if (!load_string_array(file, 0, tokenizer)) {
                fclose(file);
                set_error(error, error_size, "GGUF tokenizer tokens 读取失败");
                return 0;
            }
        } else if (strcmp(key, "tokenizer.ggml.merges") == 0 && type == GGUF_ARRAY && tokenizer != NULL) {
            if (!load_string_array(file, 1, tokenizer)) {
                fclose(file);
                set_error(error, error_size, "GGUF tokenizer merges 读取失败");
                return 0;
            }
        } else if (type == GGUF_U32 || type == GGUF_I32) {
            uint32_t value = 0;
            if (!read_u32(file, &value)) {
                fclose(file);
                return 0;
            }
            apply_gguf_u32_key(info, key, value);
        } else if (type == GGUF_U64 || type == GGUF_I64) {
            uint64_t value = 0;
            if (!read_u64(file, &value)) {
                fclose(file);
                return 0;
            }
            if (value <= UINT32_MAX) {
                apply_gguf_u32_key(info, key, (uint32_t)value);
            }
        } else if (type == GGUF_BOOL) {
            uint8_t value = 0;
            if (!read_exact(file, &value, 1)) {
                fclose(file);
                return 0;
            }
            if (strcmp(key, "tokenizer.ggml.add_bos_token") == 0) {
                info->add_bos_token = value != 0;
            }
        } else if (type == GGUF_F32) {
            float value = 0.0f;
            if (!read_exact(file, &value, sizeof(value))) {
                fclose(file);
                return 0;
            }
            if (strstr(key, "layer_norm_rms_epsilon") != NULL) {
                info->rms_eps = value;
            } else if (strstr(key, "rope.freq_base") != NULL) {
                info->rope_freq_base = value;
            }
        } else if (type == GGUF_ARRAY) {
            uint32_t element_type = 0;
            uint64_t count = 0;
            uint64_t index;
            if (!read_u32(file, &element_type) || !read_u64(file, &count)) {
                fclose(file);
                return 0;
            }
            if (strstr(key, "rope.dimension_sections") != NULL &&
                (element_type == GGUF_I32 || element_type == GGUF_U32 ||
                 element_type == GGUF_I64 || element_type == GGUF_U64)) {
                for (index = 0; index < count; index++) {
                    int32_t value = 0;
                    if (element_type == GGUF_I64 || element_type == GGUF_U64) {
                        int64_t wide = 0;
                        if (!read_exact(file, &wide, sizeof(wide))) {
                            fclose(file);
                            return 0;
                        }
                        value = (int32_t)wide;
                    } else if (!read_exact(file, &value, sizeof(value))) {
                        fclose(file);
                        return 0;
                    }
                    if (index < 4) {
                        info->rope_sections[index] = value;
                    }
                }
            } else if (element_type == GGUF_STRING) {
                for (index = 0; index < count; index++) {
                    uint64_t length = 0;
                    if (!read_u64(file, &length) || !skip_bytes(file, length)) {
                        fclose(file);
                        return 0;
                    }
                }
            } else if (element_type > GGUF_F64 || k_gguf_scalar_size[element_type] == 0) {
                fclose(file);
                set_error(error, error_size, "GGUF metadata 跳过失败");
                return 0;
            } else if (!skip_bytes(file, count * k_gguf_scalar_size[element_type])) {
                fclose(file);
                return 0;
            }
        } else if (!skip_value(file, type)) {
            fclose(file);
            set_error(error, error_size, "GGUF metadata 跳过失败");
            return 0;
        }
    }

    if (info->tensor_count > 0xFFFFFFFFu) {
        fclose(file);
        set_error(error, error_size, "GGUF tensor 数量过大");
        return 0;
    }
    if (info->tensor_count > 0) {
        info->tensors = (edgexpu_gguf_tensor *)calloc(
            (size_t)info->tensor_count,
            sizeof(*info->tensors)
        );
        if (info->tensors == NULL) {
            fclose(file);
            set_error(error, error_size, "GGUF tensor 表分配失败");
            return 0;
        }
        info->tensors_cap = (uint32_t)info->tensor_count;
    }

    for (tensor_index = 0; tensor_index < info->tensor_count; tensor_index++) {
        edgexpu_gguf_tensor *tensor = &info->tensors[info->n_tensors];
        uint32_t dim;
        memset(tensor, 0, sizeof(*tensor));
        if (!read_string_into(file, tensor->name, sizeof(tensor->name)) ||
            !read_u32(file, &tensor->n_dims)) {
            fclose(file);
            set_error(error, error_size, "GGUF tensor info 读取失败");
            return 0;
        }
        if (tensor->n_dims > 4) {
            fclose(file);
            set_error(error, error_size, "GGUF tensor 维度超出支持范围");
            return 0;
        }
        for (dim = 0; dim < tensor->n_dims; dim++) {
            if (!read_u64(file, &tensor->dims[dim])) {
                fclose(file);
                return 0;
            }
        }
        if (!read_u32(file, &tensor->type) || !read_u64(file, &tensor->offset)) {
            fclose(file);
            return 0;
        }
        info->n_tensors++;
    }

    tensor_end = ftell(file);
    if (tensor_end < 0) {
        fclose(file);
        return 0;
    }
    info->data_offset = ((uint64_t)tensor_end + 31ull) & ~31ull;
    fclose(file);

    if (tokenizer != NULL) {
        tokenizer->bos_token_id = info->bos_token_id;
        tokenizer->eos_token_id = info->eos_token_id;
        tokenizer->pad_token_id = info->pad_token_id;
        tokenizer->add_bos_token = info->add_bos_token;
        snprintf(tokenizer->pre, sizeof(tokenizer->pre), "%s", info->tokenizer_pre);
        if (!edgexpu_tokenizer_build_index(tokenizer, error, error_size)) {
            return 0;
        }
    }

    return 1;
}
