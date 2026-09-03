#include "edgexpu/server.h"

#include "edgexpu/chat.h"
#include "edgexpu/runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* 最小 HTTP：解析 messages / model / max_tokens / stream，转给 runtime generate。 */

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET edgexpu_socket_t;
#define EDGEXPU_INVALID_SOCKET INVALID_SOCKET
#define edgexpu_close_socket closesocket
#else
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int edgexpu_socket_t;
#define EDGEXPU_INVALID_SOCKET (-1)
#define edgexpu_close_socket close
#endif

#define EDGEXPU_HTTP_BUFFER 32768

static int sockets_init(void) {
#if defined(_WIN32)
    WSADATA data;
    return WSAStartup(MAKEWORD(2, 2), &data) == 0;
#else
    return 1;
#endif
}

static void sockets_cleanup(void) {
#if defined(_WIN32)
    WSACleanup();
#endif
}

static void json_escape(const char *input, char *output, size_t output_size) {
    size_t used = 0;

    if (output_size == 0) {
        return;
    }
    output[0] = '\0';
    if (input == NULL) {
        return;
    }

    while (*input != '\0' && used + 2 < output_size) {
        unsigned char c = (unsigned char)*input++;
        if (c == '"' || c == '\\') {
            if (used + 2 >= output_size) {
                break;
            }
            output[used++] = '\\';
            output[used++] = (char)c;
        } else if (c == '\n') {
            if (used + 2 >= output_size) {
                break;
            }
            output[used++] = '\\';
            output[used++] = 'n';
        } else if (c == '\r') {
            if (used + 2 >= output_size) {
                break;
            }
            output[used++] = '\\';
            output[used++] = 'r';
        } else if (c >= 32) {
            output[used++] = (char)c;
        }
    }
    output[used] = '\0';
}

static int extract_json_string(const char *json, const char *key, char *output, size_t output_size) {
    char pattern[128];
    const char *cursor;
    char *writer;
    size_t remaining;

    if (json == NULL || key == NULL || output == NULL || output_size == 0) {
        return 0;
    }

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    cursor = strstr(json, pattern);
    if (cursor == NULL) {
        return 0;
    }
    cursor = strchr(cursor + strlen(pattern), ':');
    if (cursor == NULL) {
        return 0;
    }
    cursor = strchr(cursor, '"');
    if (cursor == NULL) {
        return 0;
    }
    cursor++;

    writer = output;
    remaining = output_size - 1;
    while (*cursor != '\0' && *cursor != '"' && remaining > 0) {
        if (*cursor == '\\' && cursor[1] != '\0') {
            cursor++;
            if (*cursor == 'n') {
                *writer++ = '\n';
            } else if (*cursor == 'r') {
                *writer++ = '\r';
            } else {
                *writer++ = *cursor;
            }
        } else {
            *writer++ = *cursor;
        }
        cursor++;
        remaining--;
    }
    *writer = '\0';
    return 1;
}

static int extract_json_int(const char *json, const char *key, int default_value) {
    char pattern[128];
    const char *cursor;

    if (json == NULL || key == NULL) {
        return default_value;
    }

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    cursor = strstr(json, pattern);
    if (cursor == NULL) {
        return default_value;
    }
    cursor = strchr(cursor + strlen(pattern), ':');
    if (cursor == NULL) {
        return default_value;
    }
    return atoi(cursor + 1);
}

static int extract_json_float_milli(const char *json, const char *key, int default_milli) {
    char pattern[128];
    const char *cursor;

    if (json == NULL || key == NULL) {
        return default_milli;
    }

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    cursor = strstr(json, pattern);
    if (cursor == NULL) {
        return default_milli;
    }
    cursor = strchr(cursor + strlen(pattern), ':');
    if (cursor == NULL) {
        return default_milli;
    }
    return (int)(atof(cursor + 1) * 1000.0f);
}

static int request_wants_stream(const char *body) {
    const char *cursor = strstr(body, "\"stream\"");
    if (cursor == NULL) {
        return 0;
    }
    cursor = strchr(cursor, ':');
    if (cursor == NULL) {
        return 0;
    }
    while (*cursor == ':' || *cursor == ' ' || *cursor == '\t') {
        cursor++;
    }
    return strncmp(cursor, "true", 4) == 0;
}

static void send_all(edgexpu_socket_t client, const char *data) {
    size_t length = strlen(data);
    size_t sent = 0;

    while (sent < length) {
        int chunk = send(client, data + sent, (int)(length - sent), 0);
        if (chunk <= 0) {
            break;
        }
        sent += (size_t)chunk;
    }
}

static void send_json_response(edgexpu_socket_t client, int status, const char *status_text, const char *body) {
    char header[512];
    snprintf(
        header,
        sizeof(header),
        "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %u\r\nConnection: close\r\n\r\n",
        status,
        status_text,
        (unsigned)strlen(body)
    );
    send_all(client, header);
    send_all(client, body);
}

static void send_error(edgexpu_socket_t client, int status, const char *status_text, const char *message) {
    char escaped[1024];
    char body[1400];

    json_escape(message, escaped, sizeof(escaped));
    snprintf(body, sizeof(body), "{\"error\":{\"message\":\"%s\",\"type\":\"edgexpu_error\"}}\n", escaped);
    send_json_response(client, status, status_text, body);
}

static void send_chat_response(
    edgexpu_socket_t client,
    const edgexpu_runtime *runtime,
    const edgexpu_generation_result *result
) {
    char escaped_text[EDGEXPU_TEXT_PROMPT * 2];
    char body[EDGEXPU_TEXT_PROMPT * 2 + 1024];
    long created = (long)time(NULL);
    const char *finish = result->finish_reason[0] != '\0' ? result->finish_reason : "stop";

    json_escape(result->text, escaped_text, sizeof(escaped_text));
    snprintf(
        body,
        sizeof(body),
        "{"
        "\"id\":\"chatcmpl-edgexpu\","
        "\"object\":\"chat.completion\","
        "\"created\":%ld,"
        "\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},\"finish_reason\":\"%s\"}],"
        "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}"
        "}\n",
        created,
        runtime->manifest.model_id,
        escaped_text,
        finish,
        result->prompt_tokens_approx,
        result->completion_tokens_approx,
        result->prompt_tokens_approx + result->completion_tokens_approx
    );
    send_json_response(client, 200, "OK", body);
}

static void send_sse_chunk(edgexpu_socket_t client, const char *payload) {
    send_all(client, "data: ");
    send_all(client, payload);
    send_all(client, "\n\n");
}

static void send_stream_role_chunk(edgexpu_socket_t client, const char *model_id, long created) {
    char event[1024];

    snprintf(
        event,
        sizeof(event),
        "{\"id\":\"chatcmpl-edgexpu\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
        "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}",
        created,
        model_id
    );
    send_sse_chunk(client, event);
}

static void send_stream_token_chunk(
    edgexpu_socket_t client,
    const char *model_id,
    long created,
    const char *token
) {
    char escaped_text[EDGEXPU_TEXT_SMALL * 2];
    char event[EDGEXPU_TEXT_SMALL * 2 + 512];

    json_escape(token, escaped_text, sizeof(escaped_text));
    snprintf(
        event,
        sizeof(event),
        "{\"id\":\"chatcmpl-edgexpu\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
        "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}",
        created,
        model_id,
        escaped_text
    );
    send_sse_chunk(client, event);
}

static void send_stream_finish_chunk(
    edgexpu_socket_t client,
    const char *model_id,
    long created,
    const char *finish_reason
) {
    char event[1024];
    const char *finish = finish_reason != NULL && finish_reason[0] != '\0' ? finish_reason : "stop";

    snprintf(
        event,
        sizeof(event),
        "{\"id\":\"chatcmpl-edgexpu\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
        "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]}",
        created,
        model_id,
        finish
    );
    send_sse_chunk(client, event);
}

typedef struct sse_stream_state {
    edgexpu_socket_t client;
    const char *model_id;
    long created;
} sse_stream_state;

static void sse_on_token(const char *token, int token_index, int token_count, void *user_data) {
    sse_stream_state *state = (sse_stream_state *)user_data;

    (void)token_index;
    (void)token_count;
    if (state == NULL || token == NULL || token[0] == '\0') {
        return;
    }

    send_stream_token_chunk(state->client, state->model_id, state->created, token);
}

static int extract_string_in_range(
    const char *begin,
    const char *end,
    const char *key,
    char *output,
    size_t output_size
) {
    char saved;
    int ok;
    char *mutable_end;

    if (begin == NULL || end == NULL || begin >= end) {
        return 0;
    }
    mutable_end = (char *)end;
    saved = *mutable_end;
    *mutable_end = '\0';
    ok = extract_json_string(begin, key, output, output_size);
    *mutable_end = saved;
    return ok;
}

static int parse_chat_messages(
    const char *body,
    edgexpu_chat_message *messages,
    char roles[][EDGEXPU_TEXT_SMALL],
    char *pool,
    size_t pool_size,
    size_t *n_messages
) {
    const char *cursor;
    const char *array;
    size_t pool_used = 0;
    size_t count = 0;

    *n_messages = 0;
    cursor = strstr(body, "\"messages\"");
    if (cursor == NULL) {
        return 0;
    }
    array = strchr(cursor + 10, '[');
    if (array == NULL) {
        return 0;
    }
    cursor = array + 1;
    while (*cursor != '\0' && *cursor != ']' && count < EDGEXPU_CHAT_MAX_MESSAGES) {
        const char *obj;
        const char *obj_end;
        int depth;
        char role[EDGEXPU_TEXT_SMALL];
        char content[EDGEXPU_TEXT_LARGE];
        size_t content_len;

        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\n' || *cursor == '\r' || *cursor == ',') {
            cursor++;
        }
        if (*cursor == ']' || *cursor == '\0') {
            break;
        }
        if (*cursor != '{') {
            cursor++;
            continue;
        }
        obj = cursor;
        depth = 0;
        obj_end = cursor;
        do {
            if (*obj_end == '{') {
                depth++;
            } else if (*obj_end == '}') {
                depth--;
            }
            obj_end++;
        } while (*obj_end != '\0' && depth > 0);
        if (depth != 0) {
            return 0;
        }
        role[0] = '\0';
        content[0] = '\0';
        (void)extract_string_in_range(obj, obj_end, "role", role, sizeof(role));
        if (!extract_string_in_range(obj, obj_end, "content", content, sizeof(content))) {
            cursor = obj_end;
            continue;
        }
        if (role[0] == '\0') {
            snprintf(role, sizeof(role), "user");
        }
        content_len = strlen(content);
        if (pool_used + content_len + 1 > pool_size) {
            return 0;
        }
        memcpy(pool + pool_used, content, content_len + 1);
        snprintf(roles[count], EDGEXPU_TEXT_SMALL, "%s", role);
        messages[count].role = roles[count];
        messages[count].content = pool + pool_used;
        pool_used += content_len + 1;
        count++;
        cursor = obj_end;
    }
    *n_messages = count;
    return count > 0;
}

/* 解析 messages（含 system / 多轮）并套模型包模板；校验 model。 */
static void handle_chat_completion(edgexpu_socket_t client, edgexpu_runtime *runtime, const char *body) {
    edgexpu_generation_request request;
    edgexpu_generation_result result;
    edgexpu_chat_message messages[EDGEXPU_CHAT_MAX_MESSAGES];
    char roles[EDGEXPU_CHAT_MAX_MESSAGES][EDGEXPU_TEXT_SMALL];
    char content_pool[EDGEXPU_TEXT_PROMPT];
    char formatted[EDGEXPU_TEXT_PROMPT];
    char model[EDGEXPU_TEXT_SMALL];
    char error[512] = {0};
    const char *template_text;
    size_t n_messages = 0;
    int temperature_milli;
    int top_p_milli;

    if (extract_json_string(body, "model", model, sizeof(model)) &&
        model[0] != '\0' &&
        strcmp(model, runtime->manifest.model_id) != 0) {
        send_error(client, 400, "Bad Request", "model does not match the loaded manifest");
        return;
    }

    if (!parse_chat_messages(body, messages, roles, content_pool, sizeof(content_pool), &n_messages)) {
        edgexpu_chat_message single;
        if (!extract_json_string(body, "prompt", content_pool, sizeof(content_pool)) &&
            !extract_json_string(body, "content", content_pool, sizeof(content_pool))) {
            send_error(client, 400, "Bad Request", "request body must include messages or content");
            return;
        }
        memset(&single, 0, sizeof(single));
        single.role = "user";
        single.content = content_pool;
        messages[0] = single;
        n_messages = 1;
    }

    template_text = runtime->manifest.chat_template;
    if (template_text[0] == '\0') {
        template_text = runtime->native.gguf.chat_template;
    }
    if (!edgexpu_chat_apply_conversation(
            template_text,
            messages,
            n_messages,
            formatted,
            sizeof(formatted))) {
        send_error(client, 400, "Bad Request", "chat template output is too long");
        return;
    }

    memset(&request, 0, sizeof(request));
    request.prompt = formatted;
    request.prompt_is_formatted = 1;
    request.max_tokens = extract_json_int(body, "max_tokens", 128);
    temperature_milli = extract_json_float_milli(body, "temperature", 0);
    request.temperature = (float)temperature_milli / 1000.0f;
    top_p_milli = extract_json_float_milli(body, "top_p", 1000);
    request.top_p = (float)top_p_milli / 1000.0f;

    if (request_wants_stream(body)) {
        sse_stream_state stream_state;
        char header[256];
        long created = (long)time(NULL);

        snprintf(
            header,
            sizeof(header),
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
        );
        send_all(client, header);

        stream_state.client = client;
        stream_state.model_id = runtime->manifest.model_id;
        stream_state.created = created;
        send_stream_role_chunk(client, runtime->manifest.model_id, created);

        if (!edgexpu_runtime_generate_stream(
                runtime,
                &request,
                &result,
                sse_on_token,
                &stream_state,
                error,
                sizeof(error))) {
            send_sse_chunk(client, "{\"error\":{\"message\":\"generation failed\",\"type\":\"edgexpu_error\"}}");
            send_sse_chunk(client, "[DONE]");
            return;
        }

        send_stream_finish_chunk(client, runtime->manifest.model_id, created, result.finish_reason);
        send_sse_chunk(client, "[DONE]");
        return;
    }

    if (!edgexpu_runtime_generate(runtime, &request, &result, error, sizeof(error))) {
        send_error(client, 500, "Internal Server Error", error);
        return;
    }

    send_chat_response(client, runtime, &result);
}

static int parse_http_content_length(const char *headers, size_t header_bytes, int *out_length) {
    const char *cursor;
    const char *end = headers + header_bytes;
    char saved;
    int value;

    *out_length = -1;
    saved = headers[header_bytes];
    ((char *)headers)[header_bytes] = '\0';
    cursor = strstr(headers, "Content-Length:");
    if (cursor == NULL) {
        cursor = strstr(headers, "content-length:");
    }
    ((char *)headers)[header_bytes] = saved;
    (void)end;
    if (cursor == NULL || cursor >= headers + header_bytes) {
        return 1;
    }
    cursor += 15;
    while (*cursor == ' ' || *cursor == '\t') {
        cursor++;
    }
    value = atoi(cursor);
    if (value < 0) {
        return 0;
    }
    *out_length = value;
    return 1;
}

static void handle_client(edgexpu_socket_t client, edgexpu_runtime *runtime) {
    char buffer[EDGEXPU_HTTP_BUFFER + 1];
    int received = 0;
    char *header_end;
    const char *body;
    size_t header_bytes;
    int content_length = -1;

    while (received < EDGEXPU_HTTP_BUFFER) {
        int chunk = recv(client, buffer + received, EDGEXPU_HTTP_BUFFER - received, 0);
        if (chunk <= 0) {
            break;
        }
        received += chunk;
        buffer[received] = '\0';
        header_end = strstr(buffer, "\r\n\r\n");
        if (header_end == NULL) {
            continue;
        }
        header_bytes = (size_t)(header_end - buffer);
        if (!parse_http_content_length(buffer, header_bytes, &content_length)) {
            send_error(client, 400, "Bad Request", "invalid Content-Length");
            return;
        }
        if (content_length < 0) {
            break;
        }
        if (header_bytes + 4 + (size_t)content_length > (size_t)EDGEXPU_HTTP_BUFFER) {
            send_error(client, 413, "Payload Too Large", "request body exceeds server buffer");
            return;
        }
        if (received >= (int)header_bytes + 4 + content_length) {
            break;
        }
    }

    if (received <= 0) {
        return;
    }
    buffer[received] = '\0';

    header_end = strstr(buffer, "\r\n\r\n");
    if (header_end == NULL) {
        send_error(client, 400, "Bad Request", "invalid HTTP request");
        return;
    }
    body = header_end + 4;

    if (strncmp(buffer, "GET /v1/models ", 15) == 0 || strncmp(buffer, "GET /v1/models\r", 15) == 0) {
        char response[512];
        snprintf(
            response,
            sizeof(response),
            "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\",\"owned_by\":\"local\"}]}\n",
            runtime->manifest.model_id
        );
        send_json_response(client, 200, "OK", response);
        return;
    }

    if (strncmp(buffer, "POST /v1/chat/completions ", 26) != 0 &&
        strncmp(buffer, "POST /v1/chat/completions\r", 26) != 0) {
        send_error(client, 404, "Not Found", "only /v1/chat/completions and /v1/models are supported");
        return;
    }

    handle_chat_completion(client, runtime, body);
}

int edgexpu_server_run(const char *manifest_path, int port) {
    edgexpu_runtime runtime;
    edgexpu_socket_t server_socket;
    struct sockaddr_in address;
    char error[512] = {0};
    int enabled = 1;

    if (port <= 0) {
        port = 8000;
    }

    edgexpu_runtime_init(&runtime);
    if (!edgexpu_runtime_load_model(&runtime, manifest_path, error, sizeof(error))) {
        fprintf(stderr, "server model load failed: %s\n", error);
        return 1;
    }

    if (!sockets_init()) {
        fprintf(stderr, "socket initialization failed\n");
        return 1;
    }

    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket == EDGEXPU_INVALID_SOCKET) {
        fprintf(stderr, "socket creation failed\n");
        sockets_cleanup();
        return 1;
    }

    setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, (const char *)&enabled, sizeof(enabled));

    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons((unsigned short)port);

    if (bind(server_socket, (struct sockaddr *)&address, sizeof(address)) != 0) {
        fprintf(stderr, "bind failed on 127.0.0.1:%d\n", port);
        edgexpu_close_socket(server_socket);
        sockets_cleanup();
        return 1;
    }

    if (listen(server_socket, 8) != 0) {
        fprintf(stderr, "listen failed\n");
        edgexpu_close_socket(server_socket);
        sockets_cleanup();
        return 1;
    }

    printf("EdgeXPU native API server listening on http://127.0.0.1:%d\n", port);
    printf("Loaded model: %s via %s\n", runtime.manifest.model_id, runtime.backend->name);

    for (;;) {
        edgexpu_socket_t client = accept(server_socket, NULL, NULL);
        if (client == EDGEXPU_INVALID_SOCKET) {
            continue;
        }
        handle_client(client, &runtime);
        edgexpu_close_socket(client);
    }

    edgexpu_close_socket(server_socket);
    sockets_cleanup();
    return 0;
}
