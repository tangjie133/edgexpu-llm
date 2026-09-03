#include "edgexpu/arch.h"

#include <stdio.h>
#include <string.h>

#define EDGEXPU_ARCH_MAX_PLUGINS 32

void edgexpu_arch_register_qwen2(void);
void edgexpu_arch_register_llama(void);
void edgexpu_arch_register_qwen35(void);

static const edgexpu_arch_plugin *g_plugins[EDGEXPU_ARCH_MAX_PLUGINS];
static int g_plugin_count;
static int g_inited;

void edgexpu_arch_register(const edgexpu_arch_plugin *plugin) {
    int i;
    if (plugin == NULL || plugin->id == NULL || plugin->match == NULL || plugin->configure == NULL) {
        return;
    }
    for (i = 0; i < g_plugin_count; i++) {
        if (g_plugins[i] == plugin || strcmp(g_plugins[i]->id, plugin->id) == 0) {
            return;
        }
    }
    if (g_plugin_count >= EDGEXPU_ARCH_MAX_PLUGINS) {
        return;
    }
    g_plugins[g_plugin_count++] = plugin;
}

void edgexpu_arch_init(void) {
    if (g_inited) {
        return;
    }
    g_inited = 1;
    edgexpu_arch_register_qwen2();
    edgexpu_arch_register_llama();
    edgexpu_arch_register_qwen35();
}

int edgexpu_arch_plugin_count(void) {
    edgexpu_arch_init();
    return g_plugin_count;
}

const edgexpu_arch_plugin *edgexpu_arch_plugin_at(int index) {
    edgexpu_arch_init();
    if (index < 0 || index >= g_plugin_count) {
        return NULL;
    }
    return g_plugins[index];
}

int edgexpu_arch_from_gguf(
    const edgexpu_gguf_info *info,
    edgexpu_arch_adapter *adapter,
    char *error,
    size_t error_size
) {
    int i;
    const char *name;

    edgexpu_arch_init();
    if (info == NULL || adapter == NULL) {
        if (error != NULL && error_size > 0) {
            snprintf(error, error_size, "architecture adapter 参数为空");
        }
        return 0;
    }

    memset(adapter, 0, sizeof(*adapter));
    name = info->architecture;

    for (i = 0; i < g_plugin_count; i++) {
        if (!g_plugins[i]->match(info)) {
            continue;
        }
        if (!g_plugins[i]->configure(info, adapter, error, error_size)) {
            adapter->plugin = g_plugins[i];
            return 0;
        }
        adapter->plugin = g_plugins[i];
        adapter->native_forward = g_plugins[i]->native_forward;
        if (adapter->tensors == NULL) {
            adapter->tensors = edgexpu_arch_tensors_llama_gguf();
        }
        return 1;
    }

    snprintf(error, error_size, "不支持的 GGUF architecture: %s", name != NULL ? name : "");
    return 0;
}
