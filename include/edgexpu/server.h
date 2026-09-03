#ifndef EDGEXPU_SERVER_H
#define EDGEXPU_SERVER_H

#ifdef __cplusplus
extern "C" {
#endif

/* 本地 OpenAI 兼容 HTTP 门面：/v1/models 与 /v1/chat/completions（含 SSE）。
 * 只做最小解析，不引入完整 HTTP 框架。
 */
int edgexpu_server_run(const char *manifest_path, int port);

#ifdef __cplusplus
}
#endif

#endif
