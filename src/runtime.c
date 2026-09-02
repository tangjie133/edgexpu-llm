#include "edgexpu/runtime.h"

#include "edgexpu/scheduler.h"

#include <stdio.h>
#include <string.h>

static void set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
    }
}

void edgexpu_runtime_init(edgexpu_runtime *runtime) {
    if (runtime == NULL) {
        return;
    }
    memset(runtime, 0, sizeof(*runtime));
}

int edgexpu_runtime_load_model(
    edgexpu_runtime *runtime,
    const char *manifest_path,
    char *error,
    size_t error_size
) {
    const edgexpu_backend *backend;

    if (runtime == NULL) {
        set_error(error, error_size, "runtime 为空");
        return 0;
    }

    if (!edgexpu_manifest_load(manifest_path, &runtime->manifest, error, error_size)) {
        return 0;
    }

    backend = edgexpu_scheduler_select_backend(&runtime->manifest, error, error_size);
    if (backend == NULL) {
        return 0;
    }

    if (!backend->load(&runtime->manifest, error, error_size)) {
        return 0;
    }

    runtime->backend = backend;
    runtime->loaded = 1;
    return 1;
}

int edgexpu_runtime_generate(
    edgexpu_runtime *runtime,
    const edgexpu_generation_request *request,
    edgexpu_generation_result *result,
    char *error,
    size_t error_size
) {
    if (runtime == NULL || !runtime->loaded || runtime->backend == NULL) {
        set_error(error, error_size, "runtime 尚未加载模型");
        return 0;
    }

    /* runtime 不直接关心 backend 细节，这一点是后续接 RKLLM/QNN 的基础。 */
    return runtime->backend->generate(request, result, error, error_size);
}
