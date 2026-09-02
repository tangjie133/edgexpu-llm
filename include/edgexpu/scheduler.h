#ifndef EDGEXPU_SCHEDULER_H
#define EDGEXPU_SCHEDULER_H

#include <stddef.h>

#include "edgexpu/backend.h"
#include "edgexpu/manifest.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 初版调度器只做 backend 选择，后续再拆 prefill/decode/verification。 */
const edgexpu_backend *edgexpu_scheduler_select_backend(
    const edgexpu_model_manifest *manifest,
    char *error,
    size_t error_size
);

#ifdef __cplusplus
}
#endif

#endif
