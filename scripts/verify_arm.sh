#!/usr/bin/env bash
# Phase 3.3: ARM / NEON correctness. No relative-to-llama tok/s gate.
#
# aarch64 host: native Release build, then per-pack greedy locks (verify.lock).
# x86 host: cross-compile Pi 4 baseline (armv8-a + NEON, no DOTPROD) and qemu-user unit tests.
# Full GGUF lock under qemu is slow; set EDGEXPU_ARM_FULL=1 to run dump-logits n=4.
#
# Pi 5 on-device: cmake -DEDGEXPU_ARM_DOTPROD=ON
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=verify.common.sh
source "${ROOT_DIR}/scripts/verify.common.sh"

HOST_ARCH="$(uname -m)"
BUILD_DIR="${EDGEXPU_ARM_BUILD_DIR:-${ROOT_DIR}/build-aarch64}"
TOOLCHAIN="${ROOT_DIR}/cmake/aarch64-linux-gnu.cmake"
SYSROOT="${EDGEXPU_AARCH64_SYSROOT:-/usr/aarch64-linux-gnu}"

die() {
    echo "arm verification failed: $*" >&2
    exit 1
}

run_bin() {
    local bin="$1"
    shift
    if [[ "${HOST_ARCH}" == "aarch64" || "${HOST_ARCH}" == "arm64" ]]; then
        "${bin}" "$@"
    else
        local qemu=""
        if command -v qemu-aarch64-static >/dev/null 2>&1; then
            qemu="qemu-aarch64-static"
        elif command -v qemu-aarch64 >/dev/null 2>&1; then
            qemu="qemu-aarch64"
        else
            die "need qemu-aarch64 or qemu-aarch64-static to run ARM binaries on ${HOST_ARCH}"
        fi
        if [[ ! -d "${SYSROOT}" ]]; then
            die "missing aarch64 sysroot ${SYSROOT} (install gcc-aarch64-linux-gnu)"
        fi
        EDGEXPU_EMULATED=1 "${qemu}" -L "${SYSROOT}" "${bin}" "$@"
    fi
}

echo "== Phase 3.3 ARM / NEON =="
echo "host=${HOST_ARCH}"

if [[ "${HOST_ARCH}" == "aarch64" || "${HOST_ARCH}" == "arm64" ]]; then
    BUILD_DIR="${EDGEXPU_ARM_BUILD_DIR:-${ROOT_DIR}/build}"
    echo "== native aarch64 build (${BUILD_DIR}) =="
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${BUILD_DIR}" --config Release
else
    command -v aarch64-linux-gnu-gcc >/dev/null 2>&1 || \
        die "install gcc-aarch64-linux-gnu (Pi 4 baseline cross compiler)"
    echo "== cross-compile armv8-a / NEON (${BUILD_DIR}) =="
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DEDGEXPU_ARM_DOTPROD=OFF
    cmake --build "${BUILD_DIR}" --config Release
fi

BIN="${BUILD_DIR}/edgexpu"
[[ -x "${BIN}" ]] || die "missing ${BIN}"
file "${BIN}" || true

echo "== unit (NEON + KV overflow contract) =="
CAPS="$(run_bin "${BIN}" capabilities)"
require_contains "${CAPS}" "\"arch\": \"aarch64\"" "capabilities arch"
require_contains "${CAPS}" "\"simd\": \"neon\"" "capabilities simd"
if [[ "${HOST_ARCH}" != "aarch64" && "${HOST_ARCH}" != "arm64" ]]; then
    require_contains "${CAPS}" "\"emulated\": true" "qemu emulated flag"
    echo "note: under qemu, cpu_count/memory_total_mb are host values, not a Raspberry Pi"
fi
require_contains "$(run_bin "${BIN}" executor-selftest)" "executor selftest passed" "executor"
require_contains "$(run_bin "${BIN}" scheduler-selftest)" "scheduler selftest passed" "scheduler"
UNIT="$(run_bin "${BIN}" native-selftest)"
require_contains "${UNIT}" "simd=neon" "kernel simd"
require_contains "${UNIT}" "native selftest passed" "kernel native-selftest"

if [[ "${EDGEXPU_ARM_FULL:-0}" == "1" || "${HOST_ARCH}" == "aarch64" || "${HOST_ARCH}" == "arm64" ]]; then
    LOCKED=0
    while IFS= read -r pack_dir; do
        if ! load_pack "${pack_dir}"; then
            continue
        fi
        if [[ ! -f "${pack_dir}/verify.lock" || "${NATIVE}" != "1" ]]; then
            continue
        fi
        if [[ ! -f "${GGUF}" ]]; then
            echo "skip ${PACK_NAME}: missing GGUF ${GGUF}"
            continue
        fi
        echo "== ${PACK_NAME} greedy lock (mmap GGUF) =="
        SELF="$(run_bin "${BIN}" native-selftest "${GGUF}")"
        require_contains "${SELF}" "simd=neon" "${PACK_NAME} simd"
        require_contains "${SELF}" "kv_overflow=ok" "${PACK_NAME} kv overflow"
        require_contains "${SELF}" "native selftest passed" "${PACK_NAME} native-selftest"
        DUMP="$(run_bin "${BIN}" dump-logits "${GGUF}" "${PROMPT}" "${GREEDY_N}")"
        check_dump_lock "${DUMP}" "${PROMPT_IDS}" "${GREEDY_IDS}" "${PACK_NAME} dump-logits"
        LOCKED=$((LOCKED + 1))
    done < <(each_pack_dir)
    if [[ "${LOCKED}" -lt 1 ]]; then
        echo "skip GGUF greedy lock: no native pack with GGUF + verify.lock"
    else
        echo "ARM greedy lock passed (${LOCKED} pack(s))"
    fi
else
    echo "skip GGUF lock on ${HOST_ARCH} (set EDGEXPU_ARM_FULL=1 to run dump-logits under qemu)"
fi

echo "ARM verification passed"
