cmake_minimum_required(VERSION 3.17)

# EdgeXPU-LLM：C runtime + CLI。默认 Release，避免空 CMAKE_BUILD_TYPE 落到 -O0。
project(edgexpu_llm C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(CheckCCompilerFlag)

add_library(edgexpu_runtime
    src/arch/llama.c
    src/arch/qwen2.c
    src/arch/qwen35.c
    src/arch/register.c
    src/arch/tensors.c
    src/backend.c
    src/backend_native.c
    src/chat.c
    src/cpu_kernel.c
    src/executor.c
    src/gguf.c
    src/gguf_quant.c
    src/gguf_dot_q8.c
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

# Pi 4 = Cortex-A72 / armv8-a（无 DOTPROD）。Pi 5 = A76，可开 EDGEXPU_ARM_DOTPROD。
option(EDGEXPU_ARM_DOTPROD "AArch64 SDOT (Pi 5 / Cortex-A76+); leave OFF for Pi 4" OFF)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" EDGEXPU_CPU)
    if(EDGEXPU_CPU MATCHES "x86_64|amd64|x86|i[3-6]86")
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
        check_c_compiler_flag(-mf16c EDGEXPU_HAS_F16C)
        if(EDGEXPU_HAS_F16C)
            target_compile_options(edgexpu_runtime PRIVATE -mf16c)
            target_compile_options(edgexpu PRIVATE -mf16c)
            target_compile_definitions(edgexpu_runtime PRIVATE EDGEXPU_F16C=1)
        endif()
        check_c_compiler_flag(-march=native EDGEXPU_HAS_MARCH_NATIVE)
        if(EDGEXPU_HAS_MARCH_NATIVE)
            target_compile_options(edgexpu_runtime PRIVATE -march=native)
            target_compile_options(edgexpu PRIVATE -march=native)
        endif()
    elseif(EDGEXPU_CPU MATCHES "aarch64|arm64|armv8")
        target_compile_definitions(edgexpu_runtime PRIVATE EDGEXPU_NEON=1)
        target_compile_definitions(edgexpu PRIVATE EDGEXPU_NEON=1)
        if(EDGEXPU_ARM_DOTPROD)
            target_compile_options(edgexpu_runtime PRIVATE -march=armv8.2-a+dotprod)
            target_compile_options(edgexpu PRIVATE -march=armv8.2-a+dotprod)
        elseif(CMAKE_CROSSCOMPILING)
            # 交叉编译禁止 -march=native（那会跟编译机走）。Pi 4 基线 armv8-a。
            target_compile_options(edgexpu_runtime PRIVATE -march=armv8-a)
            target_compile_options(edgexpu PRIVATE -march=armv8-a)
        else()
            check_c_compiler_flag(-march=native EDGEXPU_HAS_MARCH_NATIVE_ARM)
            if(EDGEXPU_HAS_MARCH_NATIVE_ARM)
                target_compile_options(edgexpu_runtime PRIVATE -march=native)
                target_compile_options(edgexpu PRIVATE -march=native)
            else()
                target_compile_options(edgexpu_runtime PRIVATE -march=armv8-a)
                target_compile_options(edgexpu PRIVATE -march=armv8-a)
            endif()
        endif()
    endif()
endif()

find_package(OpenMP)
if(OpenMP_C_FOUND)
    target_link_libraries(edgexpu_runtime PUBLIC OpenMP::OpenMP_C)
    target_compile_definitions(edgexpu_runtime PRIVATE EDGEXPU_OPENMP=1)
endif()

if(UNIX)
    target_link_libraries(edgexpu_runtime PUBLIC m)
endif()

if(WIN32)
    target_link_libraries(edgexpu_runtime PUBLIC ws2_32)
endif()
