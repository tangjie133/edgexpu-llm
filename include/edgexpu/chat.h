#ifndef EDGEXPU_CHAT_H
#define EDGEXPU_CHAT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EDGEXPU_CHAT_MAX_MESSAGES 48

typedef struct edgexpu_chat_message {
    const char *role;
    const char *content;
} edgexpu_chat_message;

/* 用模型包模板格式化对话。不解释 Jinja。
 * 占位符：{{prompt}} {{system}} {{role}} {{content}}
 * 分段：{{#system}}...{{/system}}（system 为空则整段省略）
 *       {{#message}}...{{/message}}（对每条非 system 消息展开）
 * template 为空或既没有 {{prompt}} 也没有 {{#message}} 时，原样复制 prompt/最后一条 user。
 * Runtime 禁止写死 <|im_start|> 等某一模型的 chat 标记。
 */
int edgexpu_chat_apply(
    const char *template_text,
    const char *prompt,
    char *output,
    size_t output_size
);

int edgexpu_chat_apply_conversation(
    const char *template_text,
    const edgexpu_chat_message *messages,
    size_t n_messages,
    char *output,
    size_t output_size
);

#ifdef __cplusplus
}
#endif

#endif
