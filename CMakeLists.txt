cmake_minimum_required(VERSION 3.17)

project(edgexpu_llm C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(CheckCCompilerFlag)

add_library(edgexpu_runtime
    src/arch.c
    src/backend.c
    src/chat.c
    src/cpu_kernel.c
    src/executor.c
    src/gguf.c
    src/gguf_quant.c
    src/kv_cache.c
    src/manifest.c
    src/native.c
    src/profiler.c
    src/runtime.c
    src/scheduler.c
    src/server.c
    src/tokenizer.c
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

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    check_c_compiler_flag(-mavx2 EDGEXPU_HAS_AVX2)
    check_c_compiler_flag(-mfma EDGEXPU_HAS_FMA)
    if(EDGEXPU_HAS_AVX2)
        target_compile_options(edgexpu_runtime PRIVATE -mavx2)
        target_compile_options(edgexpu PRIVATE -mavx2)
        target_compile_definitions(edgexpu_runtime PRIVATE EDGEXPU_AVX2=1)
    endif()
    if(EDGEXPU_HAS_FMA)
        target_compile_options(edgexpu_runtime PRIVATE -mfma)
        target_compile_options(edgexpu PRIVATE -mfma)
        target_compile_definitions(edgexpu_runtime PRIVATE EDGEXPU_FMA=1)
    endif()
endif()

if(UNIX)
    target_link_libraries(edgexpu_runtime PUBLIC m)
endif()

if(WIN32)
    target_link_libraries(edgexpu_runtime PUBLIC ws2_32)
endif()
