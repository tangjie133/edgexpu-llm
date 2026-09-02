#include "edgexpu/server.h"

#include "edgexpu/runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

#define EDGEXPU_HTTP_BUFFER 16384

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
    const char *match = NULL;
    char *writer;
    size_t remaining;

    if (json == NULL || key == NULL || output == NULL || output_size == 0) {
        return 0;
    }

    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    cursor = json;
    while ((cursor = strstr(cursor, pattern)) != NULL) {
        match = cursor;
        cursor += strlen(pattern);
    }
    if (match == NULL) {
        return 0;
    }
    cursor = strchr(match + strlen(pattern), ':');
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
    char escaped_text[EDGEXPU_TEXT_LARGE * 2];
    char body[EDGEXPU_TEXT_LARGE * 2 + 1024];
    long created = (long)time(NULL);

    json_escape(result->text, escaped_text, sizeof(escaped_text));
    snprintf(
        body,
        sizeof(body),
        "{"
        "\"id\":\"chatcmpl-edgexpu\","
        "\"object\":\"chat.completion\","
        "\"created\":%ld,"
        "\"model\":\"%s\","
        "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}"
        "}\n",
        created,
        runtime->manifest.model_id,
        escaped_text,
        result->prompt_tokens_approx,
        result->completion_tokens_approx,
        result->prompt_tokens_approx + result->completion_tokens_approx
    );
    send_json_response(client, 200, "OK", body);
}

static void send_stream_response(
    edgexpu_socket_t client,
    const edgexpu_runtime *runtime,
    const edgexpu_generation_result *result
) {
    char header[256];
    char escaped_text[EDGEXPU_TEXT_LARGE * 2];
    char event[EDGEXPU_TEXT_LARGE * 2 + 1024];
    long created = (long)time(NULL);

    (void)runtime;
    snprintf(
        header,
        sizeof(header),
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    );
    send_all(client, header);

    json_escape(result->text, escaped_text, sizeof(escaped_text));
    snprintf(
        event,
        sizeof(event),
        "data: {\"id\":\"chatcmpl-edgexpu\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
        "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
        created,
        runtime->manifest.model_id,
        escaped_text
    );
    send_all(client, event);
    send_all(client, "data: [DONE]\n\n");
}

static void handle_chat_completion(edgexpu_socket_t client, edgexpu_runtime *runtime, const char *body) {
    edgexpu_generation_request request;
    edgexpu_generation_result result;
    char prompt[EDGEXPU_TEXT_LARGE];
    char error[512] = {0};
    int temperature_milli;

    if (!extract_json_string(body, "content", prompt, sizeof(prompt))) {
        send_error(client, 400, "Bad Request", "request body must include at least one message content");
        return;
    }

    request.prompt = prompt;
    request.max_tokens = extract_json_int(body, "max_tokens", 128);
    temperature_milli = extract_json_float_milli(body, "temperature", 0);
    request.temperature = (float)temperature_milli / 1000.0f;

    if (!edgexpu_runtime_generate(runtime, &request, &result, error, sizeof(error))) {
        send_error(client, 500, "Internal Server Error", error);
        return;
    }

    if (request_wants_stream(body)) {
        send_stream_response(client, runtime, &result);
    } else {
        send_chat_response(client, runtime, &result);
    }
}

static void handle_client(edgexpu_socket_t client, edgexpu_runtime *runtime) {
    char buffer[EDGEXPU_HTTP_BUFFER + 1];
    int received;
    char *body;

    received = recv(client, buffer, EDGEXPU_HTTP_BUFFER, 0);
    if (received <= 0) {
        return;
    }
    buffer[received] = '\0';

    body = strstr(buffer, "\r\n\r\n");
    if (body == NULL) {
        send_error(client, 400, "Bad Request", "invalid HTTP request");
        return;
    }
    body += 4;

    if (strncmp(buffer, "GET /v1/models ", 15) == 0) {
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

    if (strncmp(buffer, "POST /v1/chat/completions ", 26) != 0) {
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
