cmake_minimum_required(VERSION 3.17)

project(edgexpu_llm C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(edgexpu_runtime
    src/backend.c
    src/executor.c
    src/manifest.c
    src/profiler.c
    src/runtime.c
    src/scheduler.c
    src/server.c
)

target_include_directories(edgexpu_runtime
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(UNIX AND NOT APPLE)
    target_compile_definitions(edgexpu_runtime PRIVATE _POSIX_C_SOURCE=200809L)
endif()

add_executable(edgexpu
    src/main.c
)

target_link_libraries(edgexpu
    PRIVATE
        edgexpu_runtime
)

if(WIN32)
    target_link_libraries(edgexpu_runtime PUBLIC ws2_32)
endif()
