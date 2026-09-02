#ifndef EDGEXPU_CHAT_H
#define EDGEXPU_CHAT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 用模型包模板格式化 prompt。只替换 {{prompt}} / {{system}}，不解释 Jinja。
 * template 为空时原样复制 prompt。
 */
int edgexpu_chat_apply(
    const char *template_text,
    const char *prompt,
    char *output,
    size_t output_size
);

#ifdef __cplusplus
}
#endif

#endif
