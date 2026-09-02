#include "edgexpu/chat.h"

#include <stdio.h>
#include <string.h>

static int append_text(char *output, size_t output_size, size_t *used, const char *text, size_t len) {
    if (text == NULL) {
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

int edgexpu_chat_apply(
    const char *template_text,
    const char *prompt,
    char *output,
    size_t output_size
) {
    const char *cursor;
    size_t used = 0;

    if (output == NULL || output_size == 0) {
        return 0;
    }
    output[0] = '\0';
    if (prompt == NULL) {
        prompt = "";
    }
    if (template_text == NULL || template_text[0] == '\0' || strstr(template_text, "{{prompt}}") == NULL) {
        snprintf(output, output_size, "%s", prompt);
        return 1;
    }

    cursor = template_text;
    while (*cursor != '\0') {
        const char *prompt_mark = strstr(cursor, "{{prompt}}");
        const char *system_mark = strstr(cursor, "{{system}}");
        const char *next = NULL;
        const char *kind = NULL;

        if (prompt_mark != NULL && (system_mark == NULL || prompt_mark <= system_mark)) {
            next = prompt_mark;
            kind = "prompt";
        } else if (system_mark != NULL) {
            next = system_mark;
            kind = "system";
        }

        if (next == NULL) {
            return append_text(output, output_size, &used, cursor, strlen(cursor));
        }
        if (!append_text(output, output_size, &used, cursor, (size_t)(next - cursor))) {
            return 0;
        }
        if (strcmp(kind, "prompt") == 0) {
            if (!append_text(output, output_size, &used, prompt, strlen(prompt))) {
                return 0;
            }
            cursor = next + strlen("{{prompt}}");
        } else {
            cursor = next + strlen("{{system}}");
        }
    }
    return 1;
}
