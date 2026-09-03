#!/usr/bin/env bash
# One CI entry: native CPU fallback (no llama.cpp shell-out on the product path).
# Greedy locks live next to each model pack: examples/models/<pack>/verify.lock
# Shared prompt/n: scripts/verify.locks
# Optional llama.cpp: scripts/align_llama.sh (not part of this script).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=verify.common.sh
source "${ROOT_DIR}/scripts/verify.common.sh"

BUILD_DIR="${ROOT_DIR}/build"
BIN="${BUILD_DIR}/edgexpu"
CHAT_REQUEST="${ROOT_DIR}/examples/requests/chat_completion.json"
STREAM_REQUEST="${ROOT_DIR}/examples/requests/chat_completion_stream.json"
PORT="${EDGEXPU_VERIFY_PORT:-18080}"
SERVER_LOG="$(mktemp)"
SERVER_PID=""
NATIVE_MANIFEST=""
NATIVE_MODEL_ID=""
NATIVE_PACKS=0

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

run_pack_lock() {
    local pack_dir="$1"
    if [[ "$(basename "${pack_dir}")" == "_template" ]]; then
        echo "skip _template: contributor template"
        return 0
    fi
    if ! load_pack "${pack_dir}"; then
        return 0
    fi
    if [[ "${PACK_NAME}" == "_template" ]]; then
        echo "skip ${PACK_NAME}: contributor template"
        return 0
    fi
    if [[ ! -f "${pack_dir}/verify.lock" ]]; then
        echo "skip ${PACK_NAME}: no verify.lock"
        return 0
    fi
    if [[ ! -f "${GGUF}" ]]; then
        echo "skip ${PACK_NAME}: missing GGUF ${GGUF}"
        return 0
    fi

    echo "-- pack ${PACK_NAME} model_id=${MODEL_ID} native=${NATIVE} --"
    if [[ -n "${MANIFEST_CONTAINS}" ]]; then
        require_contains "$("${BIN}" inspect-manifest "${MANIFEST}")" "${MANIFEST_CONTAINS}" "${PACK_NAME} manifest"
    fi

    if [[ "${NATIVE}" == "1" ]]; then
        require_contains "$("${BIN}" inspect-manifest "${MANIFEST}")" "artifact.backend: cpu.native" "${PACK_NAME} native artifact"
        require_contains "$("${BIN}" inspect-gguf "${GGUF}")" "adapter=${ADAPTER}" "${PACK_NAME} inspect-gguf"
        require_contains "$("${BIN}" native-selftest "${GGUF}")" "native selftest passed" "${PACK_NAME} native-selftest"
        TOKENIZE_OUTPUT="$("${BIN}" tokenize "${MANIFEST}" "${PROMPT}")"
        require_contains "${TOKENIZE_OUTPUT}" "decoded=${PROMPT}" "${PACK_NAME} tokenize"
        require_line "${TOKENIZE_OUTPUT}" "token_ids=${PROMPT_IDS}" "${PACK_NAME} tokenize ids"
        DUMP_OUTPUT="$("${BIN}" dump-logits "${GGUF}" "${PROMPT}" "${GREEDY_N}")"
        check_dump_lock "${DUMP_OUTPUT}" "${PROMPT_IDS}" "${GREEDY_IDS}" "${PACK_NAME} dump-logits"
        NATIVE_PACKS=$((NATIVE_PACKS + 1))
        if [[ -z "${NATIVE_MANIFEST}" ]]; then
            NATIVE_MANIFEST="${MANIFEST}"
            NATIVE_MODEL_ID="${MODEL_ID}"
        fi
    else
        require_contains "$("${BIN}" inspect-manifest "${MANIFEST}")" "artifact.backend: cpu.native" "${PACK_NAME} native artifact"
        require_contains "$("${BIN}" inspect-gguf "${GGUF}")" "native_adapter=unsupported" "${PACK_NAME} inspect-gguf"
        TOKENIZE_OUTPUT="$("${BIN}" tokenize "${GGUF}" "${PROMPT}")"
        require_contains "${TOKENIZE_OUTPUT}" "decoded=${PROMPT}" "${PACK_NAME} tokenize"
        if [[ -n "${PROMPT_IDS}" ]]; then
            require_line "${TOKENIZE_OUTPUT}" "token_ids=${PROMPT_IDS}" "${PACK_NAME} tokenize ids"
        fi
        if GEN_ERR="$("${BIN}" generate "${MANIFEST}" "${PROMPT}" 1 2>&1)"; then
            echo "${GEN_ERR}" >&2
            die "${PACK_NAME} generate must fail until the native adapter exists"
        fi
        require_contains "${GEN_ERR}" "尚未实现该 adapter" "${PACK_NAME} generate adapter error"
    fi
}

stage "files"
require_file "${CHAT_REQUEST}"
require_file "${STREAM_REQUEST}"

stage "build"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --config Release

stage "unit"
require_contains "$("${BIN}" capabilities)" "\"runtimes\"" "capabilities"
require_contains "$("${BIN}" capabilities)" "\"simd\"" "capabilities simd"
require_contains "$("${BIN}" capabilities)" "\"arch_plugins\"" "capabilities arch plugins"
require_contains "$("${BIN}" capabilities)" "\"cpu_native\": true" "capabilities cpu.native"
require_contains "$("${BIN}" executor-selftest)" "executor selftest passed" "executor"
SCHEDULER_OUTPUT="$("${BIN}" scheduler-selftest)"
require_contains "${SCHEDULER_OUTPUT}" "scheduler selftest passed" "scheduler"
require_contains "${SCHEDULER_OUTPUT}" "budget_reject=ok" "scheduler budget"
require_contains "$("${BIN}" native-selftest)" "native selftest passed" "kernel native-selftest"

stage "packs"
while IFS= read -r pack_dir; do
    run_pack_lock "${pack_dir}"
done < <(each_pack_dir)

if [[ "${NATIVE_PACKS}" -lt 1 || -z "${NATIVE_MANIFEST}" ]]; then
    die "no native-capable pack with GGUF (copy SmolLM2-135M-Instruct-Q4_K_M.gguf into examples/models/smollm2-135m/; *.gguf is gitignored)"
fi
echo "product pack: ${NATIVE_MANIFEST} (${NATIVE_MODEL_ID})"

stage "generate"
NO_LLAMA_DIR="$(mktemp -d)"
PATH="${NO_LLAMA_DIR}" "${BIN}" generate "${NATIVE_MANIFEST}" "What is 2+2? Answer with just the number." 8 >/dev/null
rmdir "${NO_LLAMA_DIR}"
BENCHMARK_OUTPUT="$("${BIN}" benchmark "${NATIVE_MANIFEST}" "Explain EdgeXPU-LLM briefly.")"
require_contains "${BENCHMARK_OUTPUT}" "\"backend\": \"cpu.native\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"decode_seconds\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"executor_trace\"" "benchmark"
require_contains "${BENCHMARK_OUTPUT}" "\"finish_reason\"" "benchmark finish_reason"
TRACE_OUTPUT="$("${BIN}" trace "${NATIVE_MANIFEST}" "Explain EdgeXPU-LLM briefly.")"
require_contains "${TRACE_OUTPUT}" "cpu.native" "trace"
require_contains "${TRACE_OUTPUT}" "native_prefill" "trace prefill"
require_contains "${TRACE_OUTPUT}" "telemetry_keep_native" "trace next decode policy"
require_contains "${TRACE_OUTPUT}" "decode_step" "trace"

stage "server"
"${BIN}" serve "${NATIVE_MANIFEST}" "${PORT}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID="$!"
for _ in {1..50}; do
    if curl -fs "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        break
    fi
    sleep 0.2
done
require_contains "$(curl -fs "http://127.0.0.1:${PORT}/v1/models")" "${NATIVE_MODEL_ID}" "models API"
CHAT_OUTPUT="$(curl -fs "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data "{\"model\":\"${NATIVE_MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":8,\"temperature\":0}")"
require_contains "${CHAT_OUTPUT}" "\"object\":\"chat.completion\"" "chat API"
require_contains "${CHAT_OUTPUT}" "\"finish_reason\"" "chat finish_reason"
LENGTH_OUTPUT="$(curl -fs "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    --data "{\"model\":\"${NATIVE_MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":1,\"temperature\":0}")"
require_contains "${LENGTH_OUTPUT}" "\"finish_reason\":\"length\"" "chat length finish_reason"
WRONG_MODEL="$(curl -sS "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"not-this-model","messages":[{"role":"user","content":"Hi"}],"max_tokens":1}' || true)"
require_contains "${WRONG_MODEL}" "edgexpu_error" "model mismatch"
require_contains "${WRONG_MODEL}" "does not match" "model mismatch message"
STREAM_BODY="$(mktemp)"
sed "s/smollm2-135m/${NATIVE_MODEL_ID}/g" "${STREAM_REQUEST}" >"${STREAM_BODY}"
STREAM_OUTPUT="$(curl -fs "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" -d @"${STREAM_BODY}")"
rm -f "${STREAM_BODY}"
require_contains "${STREAM_OUTPUT}" "[DONE]" "stream API"
require_min_count "${STREAM_OUTPUT}" "data:" 4 "stream API"

stage "errors"
if "${BIN}" inspect-manifest "${ROOT_DIR}/examples/models/missing/model.manifest.json" >/dev/null 2>&1; then
    die "missing manifest unexpectedly succeeded"
fi
require_contains "$(curl -sS "http://127.0.0.1:${PORT}/v1/unknown" 2>/dev/null || true)" "edgexpu_error" "unknown route"

echo "MVP verification passed"
