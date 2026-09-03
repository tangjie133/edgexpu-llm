#!/usr/bin/env bash
# One CI entry: native CPU fallback only (no llama.cpp shell-out).
# Numerical locks: scripts/verify.locks
# Optional llama.cpp: scripts/align_llama.sh  or  edgexpu compare
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=verify.locks
source "${ROOT_DIR}/scripts/verify.locks"

BUILD_DIR="${ROOT_DIR}/build"
BIN="${BUILD_DIR}/edgexpu"
QWEN_MANIFEST="${ROOT_DIR}/examples/models/qwen2.5-0.5b/model.manifest.json"
QWEN_GGUF="${ROOT_DIR}/examples/models/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf"
SMOL_MANIFEST="${ROOT_DIR}/examples/models/smollm2-135m/model.manifest.json"
SMOL_GGUF="${ROOT_DIR}/examples/models/smollm2-135m/SmolLM2-135M-Instruct-Q4_K_M.gguf"
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

stage() {
    echo "== $* =="
}

die() {
    echo "verification failed: $*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || die "missing file $1"
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

require_min_count() {
    local text="$1"
    local expected="$2"
    local minimum="$3"
    local label="$4"
    local count
    count="$(printf '%s' "${text}" | grep -o "${expected}" | wc -l | tr -d ' ')"
    if [[ "${count}" -lt "${minimum}" ]]; then
        echo "${text}" >&2
        die "${label} expected >=${minimum} ${expected}, got ${count}"
    fi
}

check_dump_lock() {
    local dump="$1"
    local prompt_ids="$2"
    local greedy_ids="$3"
    local label="$4"
    require_line "${dump}" "token_ids=${prompt_ids}" "${label} prompt ids"
    require_line "${dump}" "greedy_ids=${greedy_ids}" "${label} greedy ids"
}

stage "files"
require_file "${QWEN_MANIFEST}"
require_file "${QWEN_GGUF}"
require_file "${SMOL_MANIFEST}"
require_file "${SMOL_GGUF}"
require_file "${CHAT_REQUEST}"
require_file "${STREAM_REQUEST}"

stage "build"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --config Release

stage "unit"
require_contains "$("${BIN}" capabilities)" "\"runtimes\"" "capabilities"
require_contains "$("${BIN}" capabilities)" "\"simd\"" "capabilities simd"
require_contains "$("${BIN}" executor-selftest)" "executor selftest passed" "executor"
require_contains "$("${BIN}" scheduler-selftest)" "scheduler selftest passed" "scheduler"
require_contains "$("${BIN}" native-selftest)" "native selftest passed" "kernel native-selftest"

stage "qwen lock"
require_contains "$("${BIN}" inspect-manifest "${QWEN_MANIFEST}")" "model_id: qwen2.5-0.5b" "qwen manifest"
require_contains "$("${BIN}" inspect-manifest "${QWEN_MANIFEST}")" "name: Qwen2.5-Coder-0.5B-Instruct" "qwen pack identity"
require_contains "$("${BIN}" inspect-manifest "${QWEN_MANIFEST}")" "artifact.backend: cpu.native" "qwen native artifact"
require_contains "$("${BIN}" inspect-gguf "${QWEN_GGUF}")" "adapter=qwen2" "qwen inspect-gguf"
require_contains "$("${BIN}" native-selftest "${QWEN_GGUF}")" "native selftest passed" "qwen native-selftest"
TOKENIZE_OUTPUT="$("${BIN}" tokenize "${QWEN_MANIFEST}" "${PROMPT}")"
require_contains "${TOKENIZE_OUTPUT}" "decoded=${PROMPT}" "qwen tokenize"
require_line "${TOKENIZE_OUTPUT}" "token_ids=${QWEN_PROMPT_IDS}" "qwen tokenize ids"
DUMP_OUTPUT="$("${BIN}" dump-logits "${QWEN_GGUF}" "${PROMPT}" "${GREEDY_N}")"
check_dump_lock "${DUMP_OUTPUT}" "${QWEN_PROMPT_IDS}" "${QWEN_GREEDY_IDS}" "qwen dump-logits"

stage "smollm lock"
require_contains "$("${BIN}" inspect-gguf "${SMOL_GGUF}")" "adapter=llama" "smollm inspect-gguf"
require_contains "$("${BIN}" native-selftest "${SMOL_GGUF}")" "native selftest passed" "smollm native-selftest"
SMOL_TOKENIZE="$("${BIN}" tokenize "${SMOL_MANIFEST}" "${PROMPT}")"
require_contains "${SMOL_TOKENIZE}" "decoded=${PROMPT}" "smollm tokenize"
require_line "${SMOL_TOKENIZE}" "token_ids=${SMOL_PROMPT_IDS}" "smollm tokenize ids"
SMOL_DUMP="$("${BIN}" dump-logits "${SMOL_GGUF}" "${PROMPT}" "${GREEDY_N}")"
check_dump_lock "${SMOL_DUMP}" "${SMOL_PROMPT_IDS}" "${SMOL_GREEDY_IDS}" "smollm dump-logits"

stage "generate"
# PATH 上无 llama / llama-cli 时 generate 仍须走 native（树莓派 P0）。
NO_LLAMA_DIR="$(mktemp -d)"
PATH="${NO_LLAMA_DIR}" "${BIN}" generate "${QWEN_MANIFEST}" "What is 2+2? Answer with just the number." 8 >/dev/null
rmdir "${NO_LLAMA_DIR}"
BENCHMARK_OUTPUT="$("${BIN}" benchmark "${QWEN_MANIFEST}" "Explain EdgeXPU-LLM briefly.")"
require_contains "${BENCHMARK_OUTPUT}" "\"backend\": \"cpu.native\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"decode_seconds\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"executor_trace\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"finish_reason\"" "benchmark finish_reason"
TRACE_OUTPUT="$("${BIN}" trace "${QWEN_MANIFEST}" "Explain EdgeXPU-LLM briefly.")"
require_contains "${TRACE_OUTPUT}" "cpu.native" "trace"
require_contains "${TRACE_OUTPUT}" "native_prefill" "trace prefill"
require_contains "${TRACE_OUTPUT}" "telemetry_keep_native" "trace next decode policy"
require_contains "${TRACE_OUTPUT}" "decode_step" "trace"

stage "server"
"${BIN}" serve "${QWEN_MANIFEST}" "${PORT}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"
for _ in {1..50}; do
    if curl -fs "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        break
    fi
    sleep 0.2
done
require_contains "$(curl -fs "http://127.0.0.1:${PORT}/v1/models")" "qwen2.5-0.5b" "models API"
CHAT_OUTPUT="$(curl -fs "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" -d @"${CHAT_REQUEST}")"
require_contains "${CHAT_OUTPUT}" "\"object\":\"chat.completion\"" "chat API"
require_contains "${CHAT_OUTPUT}" "\"finish_reason\"" "chat finish_reason"
LENGTH_OUTPUT="$(curl -fs "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen2.5-0.5b","messages":[{"role":"user","content":"Hi"}],"max_tokens":1,"temperature":0}')"
require_contains "${LENGTH_OUTPUT}" "\"finish_reason\":\"length\"" "chat length finish_reason"
WRONG_MODEL="$(curl -sS "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"not-this-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":1}' || true)"
require_contains "${WRONG_MODEL}" "edgexpu_error" "model mismatch"
require_contains "${WRONG_MODEL}" "does not match" "model mismatch message"
STREAM_OUTPUT="$(curl -fs "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" -d @"${STREAM_REQUEST}")"
require_contains "${STREAM_OUTPUT}" "[DONE]" "stream API"
require_min_count "${STREAM_OUTPUT}" "data:" 4 "stream API"

stage "errors"
if "${BIN}" inspect-manifest "${ROOT_DIR}/examples/models/missing/model.manifest.json" >/dev/null 2>&1; then
    die "missing manifest unexpectedly succeeded"
fi
require_contains "$(curl -sS "http://127.0.0.1:${PORT}/v1/unknown" 2>/dev/null || true)" "edgexpu_error" "unknown route"

echo "MVP verification passed"
