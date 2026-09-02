#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BIN="${BUILD_DIR}/edgexpu"
MANIFEST="${ROOT_DIR}/examples/models/qwen2.5-0.5b/model.manifest.json"
GGUF="${ROOT_DIR}/examples/models/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf"
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

require_min_count() {
    local text="$1"
    local expected="$2"
    local minimum="$3"
    local label="$4"
    local count

    count="$(printf '%s' "${text}" | grep -o "${expected}" | wc -l | tr -d ' ')"
    if [[ "${count}" -lt "${minimum}" ]]; then
        echo "verification failed: ${label} expected at least ${minimum} ${expected}, got ${count}" >&2
        echo "${text}" >&2
        exit 1
    fi
}

echo "[1/14] checking required files"
require_file "${MANIFEST}"
require_file "${GGUF}"
require_file "${CHAT_REQUEST}"
require_file "${STREAM_REQUEST}"

echo "[2/14] building"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --config Release

echo "[3/14] checking capabilities"
CAPABILITIES="$("${BIN}" capabilities)"
require_contains "${CAPABILITIES}" "\"runtimes\"" "capabilities"

echo "[4/14] checking manifest"
MANIFEST_OUTPUT="$("${BIN}" inspect-manifest "${MANIFEST}")"
require_contains "${MANIFEST_OUTPUT}" "model_id: qwen2.5-0.5b" "manifest"
require_contains "${MANIFEST_OUTPUT}" "chat_template_bytes:" "manifest"

echo "[5/14] checking native loader/tokenizer/kernels"
NATIVE_SELFTEST_OUTPUT="$("${BIN}" native-selftest "${GGUF}")"
require_contains "${NATIVE_SELFTEST_OUTPUT}" "native selftest passed" "native selftest"
require_contains "${NATIVE_SELFTEST_OUTPUT}" "vocab_size=151936" "native selftest"
require_contains "${NATIVE_SELFTEST_OUTPUT}" "layer0_rms=" "native selftest"
require_contains "${NATIVE_SELFTEST_OUTPUT}" "prefill_layers=24" "native selftest"
require_contains "${NATIVE_SELFTEST_OUTPUT}" "decode_tokens=" "native selftest"
GGUF_OUTPUT="$("${BIN}" inspect-gguf "${GGUF}")"
require_contains "${GGUF_OUTPUT}" "architecture=qwen2" "inspect-gguf"
require_contains "${GGUF_OUTPUT}" "adapter=qwen2" "inspect-gguf"
require_contains "${GGUF_OUTPUT}" "qkv_bias=1" "inspect-gguf"
require_contains "${GGUF_OUTPUT}" "rope=neox" "inspect-gguf"
require_contains "${GGUF_OUTPUT}" "layer0_ready=1" "inspect-gguf"
require_contains "${GGUF_OUTPUT}" "tokenizer_model=gpt2" "inspect-gguf"
TOKENIZE_OUTPUT="$("${BIN}" tokenize "${MANIFEST}" "Hello EdgeXPU")"
require_contains "${TOKENIZE_OUTPUT}" "token_count=" "tokenize"
require_contains "${TOKENIZE_OUTPUT}" "decoded=Hello EdgeXPU" "tokenize"

echo "[6/14] running benchmark"
BENCHMARK_OUTPUT="$("${BIN}" benchmark "${MANIFEST}" "Explain EdgeXPU-LLM briefly.")"
require_contains "${BENCHMARK_OUTPUT}" "\"backend\": \"cpu.native\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"backend_telemetry\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"decode_seconds\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"stage_trace\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"queue_summary\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"executor_trace\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"type\": \"tokenize\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"type\": \"prefill\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"type\": \"stream_token\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "native token" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"scheduler_policy\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"scheduler_reason\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"backend\": \"cpu.native\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "native_tokenizer" "benchmark"

echo "[7/14] comparing native CPU fallback vs llama bootstrap"
COMPARE_OUTPUT="$("${BIN}" compare "${MANIFEST}" "Explain EdgeXPU-LLM briefly.")"
require_contains "${COMPARE_OUTPUT}" "\"native_is_cpu_fallback\": true" "compare"
require_contains "${COMPARE_OUTPUT}" "\"llama_is_bootstrap\": true" "compare"
require_contains "${COMPARE_OUTPUT}" "\"same_model\": true" "compare"
require_contains "${COMPARE_OUTPUT}" "\"backend\": \"cpu.native\"" "compare"
require_contains "${COMPARE_OUTPUT}" "\"backend\": \"cpu.baseline\"" "compare"
require_contains "${COMPARE_OUTPUT}" "\"prefill_seconds\"" "compare"
require_contains "${COMPARE_OUTPUT}" "\"decode_seconds\"" "compare"

echo "[8/14] checking executor selftest"
SELFTEST_OUTPUT="$("${BIN}" executor-selftest)"
require_contains "${SELFTEST_OUTPUT}" "executor selftest passed" "executor selftest"

echo "[9/14] checking scheduler selftest"
SCHEDULER_SELFTEST_OUTPUT="$("${BIN}" scheduler-selftest)"
require_contains "${SCHEDULER_SELFTEST_OUTPUT}" "scheduler selftest passed" "scheduler selftest"

echo "[10/14] checking trace command"
TRACE_OUTPUT="$("${BIN}" trace "${MANIFEST}" "Explain EdgeXPU-LLM briefly.")"
require_contains "${TRACE_OUTPUT}" "Backend Telemetry" "trace command"
require_contains "${TRACE_OUTPUT}" "fallback_reason=" "trace command"
require_contains "${TRACE_OUTPUT}" "Queue Summary" "trace command"
require_contains "${TRACE_OUTPUT}" "completed=" "trace command"
require_contains "${TRACE_OUTPUT}" "Executor Trace" "trace command"
require_contains "${TRACE_OUTPUT}" "tokenize" "trace command"
require_contains "${TRACE_OUTPUT}" "decode_step" "trace command"
require_contains "${TRACE_OUTPUT}" "POLICY" "trace command"
require_contains "${TRACE_OUTPUT}" "Next Decode Plan" "trace command"
require_contains "${TRACE_OUTPUT}" "telemetry_keep_cpu" "trace command"
require_contains "${TRACE_OUTPUT}" "tok/s" "trace command"
require_contains "${TRACE_OUTPUT}" "native token" "trace command"
require_contains "${TRACE_OUTPUT}" "cpu.native" "trace command"
require_contains "${TRACE_OUTPUT}" "memory.native" "trace command"
require_contains "${TRACE_OUTPUT}" "flash.manager" "trace command"

echo "[11/14] starting local API server on port ${PORT}"
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

echo "[12/14] checking chat completions"
CHAT_OUTPUT="$(curl -fs "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @"${CHAT_REQUEST}")"
require_contains "${CHAT_OUTPUT}" "\"object\":\"chat.completion\"" "chat API"
require_contains "${CHAT_OUTPUT}" "\"content\"" "chat API"

STREAM_OUTPUT="$(curl -fs "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @"${STREAM_REQUEST}")"
require_contains "${STREAM_OUTPUT}" "data:" "stream API"
require_contains "${STREAM_OUTPUT}" "\"role\":\"assistant\"" "stream API"
require_contains "${STREAM_OUTPUT}" "\"content\"" "stream API"
require_contains "${STREAM_OUTPUT}" "\"finish_reason\":\"stop\"" "stream API"
require_contains "${STREAM_OUTPUT}" "[DONE]" "stream API"
require_min_count "${STREAM_OUTPUT}" "data:" 4 "stream API"

echo "[13/14] checking expected error paths"
if "${BIN}" inspect-manifest "${ROOT_DIR}/examples/models/missing/model.manifest.json" >/dev/null 2>&1; then
    echo "verification failed: missing manifest unexpectedly succeeded" >&2
    exit 1
fi

NOT_FOUND_OUTPUT="$(curl -sS "http://127.0.0.1:${PORT}/v1/unknown" 2>/dev/null || true)"
require_contains "${NOT_FOUND_OUTPUT}" "edgexpu_error" "unknown route"

echo "[14/14] checking second model pack swap"
SECOND_MANIFEST="${ROOT_DIR}/examples/models/smollm2-135m/model.manifest.json"
SECOND_GGUF="${ROOT_DIR}/examples/models/smollm2-135m/smollm2-135m-instruct-q4_k_m.gguf"
require_file "${SECOND_MANIFEST}"
require_file "${SECOND_GGUF}"
SECOND_MANIFEST_OUTPUT="$("${BIN}" inspect-manifest "${SECOND_MANIFEST}")"
require_contains "${SECOND_MANIFEST_OUTPUT}" "model_id: smollm2-135m" "second manifest"
require_contains "${SECOND_MANIFEST_OUTPUT}" "chat_template_bytes:" "second manifest"
SECOND_GGUF_OUTPUT="$("${BIN}" inspect-gguf "${SECOND_GGUF}")"
require_contains "${SECOND_GGUF_OUTPUT}" "adapter=llama" "second inspect-gguf"
require_contains "${SECOND_GGUF_OUTPUT}" "qkv_bias=0" "second inspect-gguf"
require_contains "${SECOND_GGUF_OUTPUT}" "rope=norm" "second inspect-gguf"
SECOND_TOKENIZE_OUTPUT="$("${BIN}" tokenize "${SECOND_MANIFEST}" "Hello EdgeXPU")"
require_contains "${SECOND_TOKENIZE_OUTPUT}" "token_count=" "second tokenize"
require_contains "${SECOND_TOKENIZE_OUTPUT}" "decoded=Hello EdgeXPU" "second tokenize"
SECOND_SELFTEST_OUTPUT="$("${BIN}" native-selftest "${SECOND_GGUF}")"
require_contains "${SECOND_SELFTEST_OUTPUT}" "native selftest passed" "second native selftest"

echo "MVP verification passed"
