#include "edgexpu/backend.h"
#include "edgexpu/native.h"

#include <stdio.h>

static void set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
    }
}

static edgexpu_native_session *as_session(void *engine) {
    return (edgexpu_native_session *)engine;
}

static int native_available(void) {
    return 1;
}

static int native_load(
    void *engine,
    const edgexpu_model_manifest *manifest,
    char *error,
    size_t error_size
) {
    edgexpu_native_session *session = as_session(engine);
    if (session == NULL || manifest == NULL) {
        set_error(error, error_size, "cpu.native load 缺少 session 或 manifest");
        return 0;
    }
    return edgexpu_native_load(session, manifest->primary_artifact.path, error, error_size);
}

static int native_tokenize(void *engine, const char *text, char *error, size_t error_size) {
    edgexpu_native_session *session = as_session(engine);
    if (session == NULL) {
        set_error(error, error_size, "cpu.native tokenize 缺少 session");
        return 0;
    }
    return edgexpu_native_tokenize(session, text, error, error_size);
}

static int native_ensure_window(
    void *engine,
    int n_prompt,
    int n_new,
    char *error,
    size_t error_size
) {
    edgexpu_native_session *session = as_session(engine);
    if (session == NULL) {
        set_error(error, error_size, "cpu.native ensure_window 缺少 session");
        return 0;
    }
    return edgexpu_native_ensure_window(session, n_prompt, n_new, error, error_size);
}

static int native_prefill(void *engine, char *error, size_t error_size) {
    edgexpu_native_session *session = as_session(engine);
    if (session == NULL) {
        set_error(error, error_size, "cpu.native prefill 缺少 session");
        return 0;
    }
    return edgexpu_native_forward_prefill(session, error, error_size);
}

static int native_reserve_kv(void *engine, int tokens, char *error, size_t error_size) {
    edgexpu_native_session *session = as_session(engine);
    if (session == NULL) {
        set_error(error, error_size, "cpu.native reserve_kv 缺少 session");
        return 0;
    }
    return edgexpu_native_reserve_kv(session, tokens, error, error_size);
}

static int native_decode_step(
    void *engine,
    float temperature,
    float top_p,
    uint32_t *token_id,
    char *piece,
    size_t piece_size,
    int *stopped,
    char *error,
    size_t error_size
) {
    edgexpu_native_session *session = as_session(engine);
    if (session == NULL) {
        set_error(error, error_size, "cpu.native decode_step 缺少 session");
        return 0;
    }
    return edgexpu_native_generate_next(
        session,
        temperature,
        top_p,
        token_id,
        piece,
        piece_size,
        stopped,
        error,
        error_size
    );
}

const edgexpu_backend *edgexpu_backend_cpu_native(void) {
    static const edgexpu_backend backend = {
        .name = "cpu.native",
        .is_available = native_available,
        .load = native_load,
        .generate = NULL,
        .tokenize = native_tokenize,
        .ensure_window = native_ensure_window,
        .prefill = native_prefill,
        .reserve_kv = native_reserve_kv,
        .decode_step = native_decode_step
    };
    return &backend;
}
