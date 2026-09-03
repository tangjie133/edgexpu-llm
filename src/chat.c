#include "edgexpu/chat.h"

#include <stdio.h>
#include <string.h>

/* 线性扫描替换占位符与 {{#system}} / {{#message}} 段。不跑 Jinja。 */

static int append_text(char *output, size_t output_size, size_t *used, const char *text, size_t len) {
    if (text == NULL || len == 0) {
        return 1;
    }
    if (*used + len >= output_size) {
        return 0;
    }
    memcpy(output + *used, text, len);
    *used += len;
    output[*used] = '\0';
    return 1;
}

static int append_cstr(char *output, size_t output_size, size_t *used, const char *text) {
    return append_text(output, output_size, used, text, text != NULL ? strlen(text) : 0);
}

static int join_system(
    const edgexpu_chat_message *messages,
    size_t n_messages,
    char *output,
    size_t output_size
) {
    size_t used = 0;
    size_t i;

    if (output == NULL || output_size == 0) {
        return 0;
    }
    output[0] = '\0';
    for (i = 0; i < n_messages; i++) {
        const char *role = messages[i].role != NULL ? messages[i].role : "";
        const char *content = messages[i].content != NULL ? messages[i].content : "";
        if (strcmp(role, "system") != 0 || content[0] == '\0') {
            continue;
        }
        if (used > 0 && !append_text(output, output_size, &used, "\n", 1)) {
            return 0;
        }
        if (!append_cstr(output, output_size, &used, content)) {
            return 0;
        }
    }
    return 1;
}

static const char *last_user_content(const edgexpu_chat_message *messages, size_t n_messages) {
    size_t i;
    const char *last = "";
    for (i = 0; i < n_messages; i++) {
        const char *role = messages[i].role != NULL ? messages[i].role : "";
        if (strcmp(role, "user") == 0 && messages[i].content != NULL) {
            last = messages[i].content;
        }
    }
    return last;
}

static int expand_placeholders(
    const char *tmpl,
    size_t tmpl_len,
    const char *prompt,
    const char *system,
    const char *role,
    const char *content,
    char *output,
    size_t output_size,
    size_t *used
) {
    size_t i = 0;

    if (prompt == NULL) {
        prompt = "";
    }
    if (system == NULL) {
        system = "";
    }
    if (role == NULL) {
        role = "";
    }
    if (content == NULL) {
        content = "";
    }

    while (i < tmpl_len) {
        const char *mark;
        const char *value;
        size_t mark_len;

        if (tmpl[i] != '{' || i + 1 >= tmpl_len || tmpl[i + 1] != '{') {
            if (!append_text(output, output_size, used, tmpl + i, 1)) {
                return 0;
            }
            i++;
            continue;
        }

        if (tmpl_len - i >= 10 && memcmp(tmpl + i, "{{prompt}}", 10) == 0) {
            mark = "{{prompt}}";
            value = prompt;
            mark_len = 10;
        } else if (tmpl_len - i >= 10 && memcmp(tmpl + i, "{{system}}", 10) == 0) {
            mark = "{{system}}";
            value = system;
            mark_len = 10;
        } else if (tmpl_len - i >= 11 && memcmp(tmpl + i, "{{content}}", 11) == 0) {
            mark = "{{content}}";
            value = content;
            mark_len = 11;
        } else if (tmpl_len - i >= 8 && memcmp(tmpl + i, "{{role}}", 8) == 0) {
            mark = "{{role}}";
            value = role;
            mark_len = 8;
        } else {
            if (!append_text(output, output_size, used, tmpl + i, 1)) {
                return 0;
            }
            i++;
            continue;
        }

        (void)mark;
        if (!append_cstr(output, output_size, used, value)) {
            return 0;
        }
        i += mark_len;
    }
    return 1;
}

static const char *find_close(const char *start, const char *close) {
    return strstr(start, close);
}

int edgexpu_chat_apply_conversation(
    const char *template_text,
    const edgexpu_chat_message *messages,
    size_t n_messages,
    char *output,
    size_t output_size
) {
    char system[4096];
    const char *prompt;
    const char *cursor;
    size_t used = 0;
    int has_message_section;
    size_t i;

    if (output == NULL || output_size == 0) {
        return 0;
    }
    output[0] = '\0';
    if (!join_system(messages, n_messages, system, sizeof(system))) {
        return 0;
    }
    prompt = last_user_content(messages, n_messages);
    if (template_text == NULL) {
        template_text = "";
    }
    has_message_section = strstr(template_text, "{{#message}}") != NULL;
    if (template_text[0] == '\0' ||
        (strstr(template_text, "{{prompt}}") == NULL &&
         !has_message_section &&
         strstr(template_text, "{{system}}") == NULL &&
         strstr(template_text, "{{#system}}") == NULL)) {
        return append_cstr(output, output_size, &used, prompt) ? 1 : 0;
    }

    cursor = template_text;
    while (*cursor != '\0') {
        if (strncmp(cursor, "{{#system}}", 11) == 0) {
            const char *inner = cursor + 11;
            const char *close = find_close(inner, "{{/system}}");
            if (close == NULL) {
                return 0;
            }
            if (system[0] != '\0' &&
                !expand_placeholders(
                    inner,
                    (size_t)(close - inner),
                    prompt,
                    system,
                    "system",
                    system,
                    output,
                    output_size,
                    &used)) {
                return 0;
            }
            cursor = close + strlen("{{/system}}");
            continue;
        }
        if (strncmp(cursor, "{{#message}}", 12) == 0) {
            const char *inner = cursor + 12;
            const char *close = find_close(inner, "{{/message}}");
            if (close == NULL) {
                return 0;
            }
            for (i = 0; i < n_messages; i++) {
                const char *role = messages[i].role != NULL ? messages[i].role : "user";
                const char *content = messages[i].content != NULL ? messages[i].content : "";
                if (strcmp(role, "system") == 0) {
                    continue;
                }
                if (!expand_placeholders(
                        inner,
                        (size_t)(close - inner),
                        prompt,
                        system,
                        role,
                        content,
                        output,
                        output_size,
                        &used)) {
                    return 0;
                }
            }
            cursor = close + strlen("{{/message}}");
            continue;
        }
        if (strncmp(cursor, "{{prompt}}", 10) == 0) {
            if (!has_message_section && !append_cstr(output, output_size, &used, prompt)) {
                return 0;
            }
            cursor += 10;
            continue;
        }
        if (strncmp(cursor, "{{system}}", 10) == 0) {
            if (!append_cstr(output, output_size, &used, system)) {
                return 0;
            }
            cursor += 10;
            continue;
        }
        if (!append_text(output, output_size, &used, cursor, 1)) {
            return 0;
        }
        cursor++;
    }
    return 1;
}

int edgexpu_chat_apply(
    const char *template_text,
    const char *prompt,
    char *output,
    size_t output_size
) {
    edgexpu_chat_message message;

    memset(&message, 0, sizeof(message));
    message.role = "user";
    message.content = prompt != NULL ? prompt : "";
    return edgexpu_chat_apply_conversation(template_text, &message, 1, output, output_size);
}
