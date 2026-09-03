#include "edgexpu/arch.h"
#include "edgexpu/cpu_kernel.h"
#include "edgexpu/profiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#define EDGEXPU_POPEN _popen
#define EDGEXPU_PCLOSE _pclose
#elif defined(__linux__)
#include <unistd.h>
#endif

/* profiler 是调度器的输入来源。初版先回答“当前机器大概是什么、有什么 runtime”。 */

static int command_exists(const char *command) {
    char check_command[EDGEXPU_TEXT_MEDIUM + 64];
#if defined(_WIN32)
    char line[512];
    FILE *pipe;
#endif

#if defined(_WIN32)
    snprintf(check_command, sizeof(check_command), "where %s 2>nul", command);
    pipe = EDGEXPU_POPEN(check_command, "r");
    if (pipe == NULL) {
        return 0;
    }
    while (fgets(line, sizeof(line), pipe) != NULL) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        if (len >= 4 &&
            (_stricmp(line + len - 4, ".exe") == 0 ||
             _stricmp(line + len - 4, ".bat") == 0 ||
             _stricmp(line + len - 4, ".cmd") == 0)) {
            EDGEXPU_PCLOSE(pipe);
            return 1;
        }
    }
    EDGEXPU_PCLOSE(pipe);
    return 0;
#else
    snprintf(check_command, sizeof(check_command), "command -v %s >/dev/null 2>&1", command);
    return system(check_command) == 0;
#endif
}

static int detect_emulated(void) {
    if (getenv("EDGEXPU_EMULATED") != NULL || getenv("QEMU_LD_PREFIX") != NULL) {
        return 1;
    }
#if defined(__linux__) && defined(__aarch64__)
    {
        FILE *cpuinfo = fopen("/proc/cpuinfo", "r");
        char line[256];
        if (cpuinfo != NULL) {
            while (fgets(line, sizeof(line), cpuinfo) != NULL) {
                const char *cursor = strstr(line, "CPU implementer");
                unsigned int implementer = 0xFFu;
                if (cursor == NULL) {
                    continue;
                }
                cursor = strchr(cursor, ':');
                if (cursor == NULL) {
                    continue;
                }
                /* qemu-user 合成 0x00；树莓派等实机是 0x41 等非零厂商号。 */
                if (sscanf(cursor + 1, " 0x%x", &implementer) == 1 && implementer == 0u) {
                    fclose(cpuinfo);
                    return 1;
                }
                break;
            }
            fclose(cpuinfo);
        }
    }
#endif
    return 0;
}

int edgexpu_profile_device(edgexpu_device_profile *profile) {
    if (profile == NULL) {
        return 0;
    }

    memset(profile, 0, sizeof(*profile));

#if defined(_WIN32)
    snprintf(profile->os, sizeof(profile->os), "windows");
    snprintf(profile->arch, sizeof(profile->arch), "amd64");
    SYSTEM_INFO info;
    MEMORYSTATUSEX memory;
    GetSystemInfo(&info);
    profile->cpu_count = (int)info.dwNumberOfProcessors;
    memory.dwLength = sizeof(memory);
    if (GlobalMemoryStatusEx(&memory)) {
        profile->memory_total_mb = (int)(memory.ullTotalPhys / (1024 * 1024));
    }
#elif defined(__aarch64__)
    snprintf(profile->os, sizeof(profile->os), "linux");
    snprintf(profile->arch, sizeof(profile->arch), "aarch64");
    profile->cpu_count = (int)sysconf(_SC_NPROCESSORS_ONLN);
    profile->memory_total_mb = (int)((sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE)) / (1024 * 1024));
#elif defined(__linux__)
    snprintf(profile->os, sizeof(profile->os), "linux");
    snprintf(profile->arch, sizeof(profile->arch), "x86_64");
    profile->cpu_count = (int)sysconf(_SC_NPROCESSORS_ONLN);
    profile->memory_total_mb = (int)((sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE)) / (1024 * 1024));
#else
    snprintf(profile->os, sizeof(profile->os), "unknown");
    snprintf(profile->arch, sizeof(profile->arch), "unknown");
    profile->cpu_count = 0;
    profile->memory_total_mb = 0;
#endif

    /* PowerInfer 风格最终应探测 native runtime；这里先用命令存在性作为最小信号。 */
    profile->has_llama_cli = command_exists("llama-cli") || command_exists("powerinfer") || command_exists("main");
    profile->has_rockchip_runtime = command_exists("rkllm");
    profile->has_qualcomm_runtime = command_exists("qnn-net-run");
    snprintf(profile->simd, sizeof(profile->simd), "%s", edgexpu_cpu_simd_name());
    profile->emulated = detect_emulated();
    return 1;
}

void edgexpu_profile_print_json(const edgexpu_device_profile *profile) {
    if (profile == NULL) {
        return;
    }

    printf("{\n");
    printf("  \"os\": \"%s\",\n", profile->os);
    printf("  \"arch\": \"%s\",\n", profile->arch);
    printf("  \"cpu_count\": %d,\n", profile->cpu_count);
    printf("  \"memory_total_mb\": %d,\n", profile->memory_total_mb);
    printf("  \"simd\": \"%s\",\n", profile->simd);
    printf("  \"emulated\": %s,\n", profile->emulated ? "true" : "false");
    printf("  \"runtimes\": {\n");
    printf("    \"cpu_baseline\": %s,\n", profile->has_llama_cli ? "true" : "false");
    printf("    \"cpu_native\": true,\n");
    printf("    \"rockchip\": %s,\n", profile->has_rockchip_runtime ? "true" : "false");
    printf("    \"qualcomm\": %s\n", profile->has_qualcomm_runtime ? "true" : "false");
    printf("  },\n");
    printf("  \"arch_plugins\": [");
    {
        int i;
        int n = edgexpu_arch_plugin_count();
        for (i = 0; i < n; i++) {
            const edgexpu_arch_plugin *plugin = edgexpu_arch_plugin_at(i);
            printf("%s\"%s\"", i == 0 ? "" : ", ", plugin != NULL ? plugin->id : "");
        }
    }
    printf("]\n");
    printf("}\n");
}
