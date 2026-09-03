#ifndef EDGEXPU_PROFILER_H
#define EDGEXPU_PROFILER_H

#include <stddef.h>

#include "edgexpu/manifest.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 设备能力快照。调度器据此决定能否走 NPU，以及 llama CLI 是否可用。 */

typedef struct edgexpu_device_profile {
    char os[EDGEXPU_TEXT_SMALL];
    char arch[EDGEXPU_TEXT_SMALL];
    int cpu_count;
    int memory_total_mb;
    int has_llama_cli;
    int has_rockchip_runtime;
    int has_qualcomm_runtime;
    char simd[EDGEXPU_TEXT_SMALL];
    int emulated; /* qemu-user 等，cpu_count/memory 不是板级结果 */
} edgexpu_device_profile;

/* 探测当前设备能力。初版只做轻量本地探测，不访问网络。 */
int edgexpu_profile_device(edgexpu_device_profile *profile);

/* 用 JSON 形式打印能力，方便脚本和上层工具解析。 */
void edgexpu_profile_print_json(const edgexpu_device_profile *profile);

#ifdef __cplusplus
}
#endif

#endif
