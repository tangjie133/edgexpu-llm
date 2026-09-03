#!/usr/bin/env bash
# Phase 3.3: ARM / NEON correctness. No relative-to-llama tok/s gate.
#
# aarch64 host: native Release build, then greedy locks (same scripts/verify.locks).
# x86 host: cross-compile Pi 4 baseline (armv8-a + NEON, no DOTPROD) and qemu-user unit tests.
# Full GGUF lock under qemu is slow; set EDGEXPU_ARM_FULL=1 to run dump-logits n=4.
#
# Pi 5 on-device: cmake -DEDGEXPU_ARM_DOTPROD=ON
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=verify.locks
source "${ROOT_DIR}/scripts/verify.locks"

HOST_ARCH="$(uname -m)"
BUILD_DIR="${EDGEXPU_ARM_BUILD_DIR:-${ROOT_DIR}/build-aarch64}"
TOOLCHAIN="${ROOT_DIR}/cmake/aarch64-linux-gnu.cmake"
QWEN_GGUF="${ROOT_DIR}/examples/models/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf"
SYSROOT="${EDGEXPU_AARCH64_SYSROOT:-/usr/aarch64-linux-gnu}"

die() {
    echo "arm verification failed: $*" >&2
    exit 1
}

require_contains() {
    local text="$1"
    local expected="$2"
    local label="$3"
    case "${text}" in
        *"${expected}"*) ;;
        *)
            echo "${text}" >&2
            die "${label} missing ${expected}"
            ;;
    esac
}

require_line() {
    local text="$1"
    local expected="$2"
    local label="$3"
    if ! printf '%s\n' "${text}" | grep -qxF "${expected}"; then
        echo "${text}" >&2
        die "${label} missing line ${expected}"
    fi
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
    [[ -f "${QWEN_GGUF}" ]] || die "missing ${QWEN_GGUF}"
    echo "== qwen greedy lock (mmap GGUF) =="
    QWEN="$(run_bin "${BIN}" native-selftest "${QWEN_GGUF}")"
    require_contains "${QWEN}" "simd=neon" "qwen simd"
    require_contains "${QWEN}" "kv_overflow=ok" "qwen kv overflow"
    require_contains "${QWEN}" "native selftest passed" "qwen native-selftest"
    DUMP="$(run_bin "${BIN}" dump-logits "${QWEN_GGUF}" "${PROMPT}" "${GREEDY_N}")"
    require_line "${DUMP}" "token_ids=${QWEN_PROMPT_IDS}" "qwen prompt ids"
    require_line "${DUMP}" "greedy_ids=${QWEN_GREEDY_IDS}" "qwen greedy ids"
    echo "ARM greedy lock passed"
else
    echo "skip GGUF lock on ${HOST_ARCH} (set EDGEXPU_ARM_FULL=1 to run dump-logits under qemu)"
fi

echo "ARM verification passed"
