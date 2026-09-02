#ifndef EDGEXPU_SERVER_H
#define EDGEXPU_SERVER_H

#ifdef __cplusplus
extern "C" {
#endif

int edgexpu_server_run(const char *manifest_path, int port);

#ifdef __cplusplus
}
#endif

#endif
