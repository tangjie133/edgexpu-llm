#!/usr/bin/env bash
# Phase 3.1: fair CPU-only decode compare vs llama.cpp.
# Same GGUF, greedy, llama --n-gpu-layers 0, decode n>=32.
# GPU llama numbers are not a KPI. Does not run in verify_mvp.sh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT_DIR}/build/edgexpu"
MANIFEST="${1:-${ROOT_DIR}/examples/models/smollm2-135m/model.manifest.json}"
PROMPT="${2:-Explain EdgeXPU-LLM briefly.}"
N="${3:-32}"
LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-/home/tj/Desktop/llama.cpp/build/bin}"
export LD_LIBRARY_PATH="${LLAMA_BIN_DIR}:${LD_LIBRARY_PATH:-}"

if [[ ! -x "${BIN}" ]]; then
    echo "missing ${BIN}; build Release first" >&2
    exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
    echo "missing manifest ${MANIFEST}" >&2
    exit 1
fi
if [[ "${N}" -lt 32 ]]; then
    echo "decode n=${N} < 32; Phase 3.1 requires at least 32 tokens" >&2
    exit 1
fi

MANIFEST_DIR="$(cd "$(dirname "${MANIFEST}")" && pwd)"
if [[ -z "${GGUF:-}" ]]; then
    shopt -s nullglob
    gguf_files=("${MANIFEST_DIR}"/*.gguf)
    shopt -u nullglob
    GGUF="${gguf_files[0]:-}"
fi
if [[ -z "${GGUF}" || ! -f "${GGUF}" ]]; then
    echo "no GGUF next to ${MANIFEST}" >&2
    exit 1
fi

CLI="${LLAMA_BIN_DIR}/llama-cli"
if [[ ! -x "${CLI}" ]]; then
    CLI="$(command -v llama-cli || true)"
fi
if [[ -z "${CLI}" || ! -x "${CLI}" ]]; then
    echo "llama-cli not found; set LLAMA_BIN_DIR" >&2
    exit 1
fi

json_field() {
    local text="$1"
    local key="$2"
    python3 -c "
import json, sys
key = sys.argv[1]
j = json.load(sys.stdin)
t = j.get('backend_telemetry') or {}
if key in j:
    print(j[key])
elif key in t:
    print(t[key])
" "${key}" <<<"${text}"
}

echo "== Phase 3.1 fair CPU bench =="
echo "manifest=${MANIFEST}"
echo "gguf=${GGUF}"
echo "prompt=${PROMPT}"
echo "n=${N}"
echo "llama=${CLI}"
echo "rule: llama --n-gpu-layers 0 ; greedy ; decode>=32"
echo

echo "== native warmup (discard; OpenMP probe) =="
"${BIN}" benchmark "${MANIFEST}" "${PROMPT}" "${N}" >/dev/null
echo "warmup done"

echo
echo "== native timed =="
NATIVE_JSON="$("${BIN}" benchmark "${MANIFEST}" "${PROMPT}" "${N}")"
NATIVE_BACKEND="$(json_field "${NATIVE_JSON}" "backend")"
NATIVE_PROMPT_N="$(json_field "${NATIVE_JSON}" "prompt_tokens_approx")"
NATIVE_COMP_N="$(json_field "${NATIVE_JSON}" "completion_tokens_approx")"
NATIVE_PREFILL="$(json_field "${NATIVE_JSON}" "prefill_tokens_per_second_approx")"
NATIVE_DECODE="$(json_field "${NATIVE_JSON}" "decode_tokens_per_second_approx")"
echo "backend=${NATIVE_BACKEND}"
echo "prompt_tokens=${NATIVE_PROMPT_N} completion_tokens=${NATIVE_COMP_N}"
echo "native_prefill_tok_s=${NATIVE_PREFILL}"
echo "native_decode_tok_s=${NATIVE_DECODE}"

echo
echo "== llama-cli CPU-only (-ngl 0, CUDA_VISIBLE_DEVICES empty) =="
NPROC="$(nproc)"
set +e
LLAMA_LOG="$(
    CUDA_VISIBLE_DEVICES="" \
    "${CLI}" -m "${GGUF}" -p "${PROMPT}" -n "${N}" --temp 0 \
        --no-display-prompt -ngl 0 -t "${NPROC}" --single-turn 2>&1
)"
LLAMA_STATUS=$?
set -e
if [[ "${LLAMA_STATUS}" -ne 0 ]]; then
    printf '%s\n' "${LLAMA_LOG}" | tail -n 40 >&2
    echo "llama-cli failed" >&2
    exit 1
fi

printf '%s\n' "${LLAMA_LOG}" | grep -E 'n_gpu_layers|offloaded|CUDA|using device|n_threads' | head -n 20 || true
if printf '%s\n' "${LLAMA_LOG}" | grep -Eq 'offloaded [1-9][0-9]* /'; then
    echo "llama still offloaded layers to GPU; refuse as unfair" >&2
    exit 1
fi

# Newer llama-cli prints: [ Prompt: 229.5 t/s | Generation: 45.1 t/s ]
# Older ggml prints llama_perf_context_print eval time.
LLAMA_PREFILL="$(printf '%s\n' "${LLAMA_LOG}" | sed -n 's/.*Prompt:[[:space:]]*\([0-9.][0-9.]*\) t\/s.*/\1/p' | tail -n 1)"
LLAMA_DECODE="$(printf '%s\n' "${LLAMA_LOG}" | sed -n 's/.*Generation:[[:space:]]*\([0-9.][0-9.]*\) t\/s.*/\1/p' | tail -n 1)"
if [[ -z "${LLAMA_PREFILL}" ]]; then
    LLAMA_PREFILL="$(printf '%s\n' "${LLAMA_LOG}" | grep 'prompt eval time' | tail -n 1 \
        | sed -n 's/.* \([0-9.][0-9.]*\) tokens per second.*/\1/p')"
fi
if [[ -z "${LLAMA_DECODE}" ]]; then
    LLAMA_DECODE="$(printf '%s\n' "${LLAMA_LOG}" | grep 'eval time' | grep -v 'prompt eval' | tail -n 1 \
        | sed -n 's/.* \([0-9.][0-9.]*\) tokens per second.*/\1/p')"
fi

echo "llama_prefill_tok_s=${LLAMA_PREFILL:-?}"
echo "llama_decode_tok_s=${LLAMA_DECODE:-?}"
printf '%s\n' "${LLAMA_LOG}" | grep -E 'Prompt:|Generation:|prompt eval time|[[:space:]]eval time' || true

echo
echo "== ratio (native / llama CPU, decode) =="
python3 -c "
native_d = float('${NATIVE_DECODE}' or 0)
llama_d = float('${LLAMA_DECODE}' or 0)
native_p = float('${NATIVE_PREFILL}' or 0)
llama_p = float('${LLAMA_PREFILL}' or 0)
ratio = (native_d / llama_d) if llama_d > 0 else 0.0
print('native_prefill_tok_s=%.3f llama_prefill_tok_s=%.3f' % (native_p, llama_p))
print('native_decode_tok_s=%.3f llama_decode_tok_s=%.3f' % (native_d, llama_d))
print('decode_ratio=%.3f  (stop line 0.60, stretch 0.80)' % ratio)
print('status=%s' % ('MEETS_0.6' if ratio >= 0.60 else 'BELOW_0.6'))
"
