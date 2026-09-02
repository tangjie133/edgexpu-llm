#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BIN="${BUILD_DIR}/edgexpu"
MANIFEST="${ROOT_DIR}/examples/models/qwen2.5-0.5b/model.manifest.json"
CHAT_REQUEST="${ROOT_DIR}/examples/requests/chat_completion.json"
STREAM_REQUEST="${ROOT_DIR}/examples/requests/chat_completion_stream.json"
PORT="${EDGEXPU_VERIFY_PORT:-18080}"
SERVER_LOG="$(mktemp)"
SERVER_PID=""

cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
    rm -f "${SERVER_LOG}"
}
trap cleanup EXIT

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "missing required file: ${path}" >&2
        exit 1
    fi
}

require_contains() {
    local text="$1"
    local expected="$2"
    local label="$3"

    case "${text}" in
        *"${expected}"*) ;;
        *)
            echo "verification failed: ${label} did not contain ${expected}" >&2
            echo "${text}" >&2
            exit 1
            ;;
    esac
}

echo "[1/10] checking required files"
require_file "${MANIFEST}"
require_file "${CHAT_REQUEST}"
require_file "${STREAM_REQUEST}"

echo "[2/10] building"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --config Release

echo "[3/10] checking capabilities"
CAPABILITIES="$("${BIN}" capabilities)"
require_contains "${CAPABILITIES}" "\"runtimes\"" "capabilities"

echo "[4/10] checking manifest"
MANIFEST_OUTPUT="$("${BIN}" inspect-manifest "${MANIFEST}")"
require_contains "${MANIFEST_OUTPUT}" "model_id: qwen2.5-0.5b" "manifest"

echo "[5/10] running benchmark"
BENCHMARK_OUTPUT="$("${BIN}" benchmark "${MANIFEST}" "Explain EdgeXPU-LLM briefly.")"
require_contains "${BENCHMARK_OUTPUT}" "\"backend\": \"cpu.baseline\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"stage_trace\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"queue_summary\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"executor_trace\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"type\": \"prefill\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"type\": \"decode_step\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"scheduler_policy\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"scheduler_reason\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"text\"" "benchmark"

echo "[6/10] checking executor selftest"
SELFTEST_OUTPUT="$("${BIN}" executor-selftest)"
require_contains "${SELFTEST_OUTPUT}" "executor selftest passed" "executor selftest"

echo "[7/10] checking trace command"
TRACE_OUTPUT="$("${BIN}" trace "${MANIFEST}" "Explain EdgeXPU-LLM briefly.")"
require_contains "${TRACE_OUTPUT}" "Queue Summary" "trace command"
require_contains "${TRACE_OUTPUT}" "completed=" "trace command"
require_contains "${TRACE_OUTPUT}" "Executor Trace" "trace command"
require_contains "${TRACE_OUTPUT}" "decode_step" "trace command"
require_contains "${TRACE_OUTPUT}" "POLICY" "trace command"
require_contains "${TRACE_OUTPUT}" "SCHEDULER_REASON" "trace command"
require_contains "${TRACE_OUTPUT}" "flash.manager" "trace command"
require_contains "${TRACE_OUTPUT}" "memory.manager" "trace command"

echo "[8/10] starting local API server on port ${PORT}"
"${BIN}" serve "${MANIFEST}" "${PORT}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"

for _ in {1..50}; do
    if curl -fs "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        break
    fi
    sleep 0.2
done

MODELS_OUTPUT="$(curl -fs "http://127.0.0.1:${PORT}/v1/models")"
require_contains "${MODELS_OUTPUT}" "qwen2.5-0.5b" "models API"

echo "[9/10] checking chat completions"
CHAT_OUTPUT="$(curl -fs "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @"${CHAT_REQUEST}")"
require_contains "${CHAT_OUTPUT}" "\"object\":\"chat.completion\"" "chat API"
require_contains "${CHAT_OUTPUT}" "\"content\"" "chat API"

STREAM_OUTPUT="$(curl -fs "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @"${STREAM_REQUEST}")"
require_contains "${STREAM_OUTPUT}" "data:" "stream API"
require_contains "${STREAM_OUTPUT}" "[DONE]" "stream API"

echo "[10/10] checking expected error paths"
if "${BIN}" inspect-manifest "${ROOT_DIR}/examples/models/missing/model.manifest.json" >/dev/null 2>&1; then
    echo "verification failed: missing manifest unexpectedly succeeded" >&2
    exit 1
fi

NOT_FOUND_OUTPUT="$(curl -sS "http://127.0.0.1:${PORT}/v1/unknown" 2>/dev/null || true)"
require_contains "${NOT_FOUND_OUTPUT}" "edgexpu_error" "unknown route"

echo "MVP verification passed"
