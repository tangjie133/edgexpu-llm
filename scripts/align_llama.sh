#!/usr/bin/env bash
# Optional: same GGUF, raw prompt, temperature=0 vs llama.cpp.
# Default n=4 matches scripts/verify.locks. Needs llama-cli / llama-tokenize.
# Do not use edgexpu compare text for this check (compare wraps chat).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=verify.common.sh
source "${ROOT_DIR}/scripts/verify.common.sh"

BIN="${ROOT_DIR}/build/edgexpu"
DEFAULT_GGUF=""
for candidate in \
    "${ROOT_DIR}/examples/models/smollm2-135m/SmolLM2-135M-Instruct-Q4_K_M.gguf"
do
    if [[ -f "${candidate}" ]]; then
        DEFAULT_GGUF="${candidate}"
        break
    fi
done
GGUF="${1:-${DEFAULT_GGUF}}"
PROMPT="${2:-${PROMPT}}"
N="${3:-${GREEDY_N}}"
LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-/home/tj/Desktop/llama.cpp/build/bin}"
export LD_LIBRARY_PATH="${LLAMA_BIN_DIR}:${LD_LIBRARY_PATH:-}"

if [[ -z "${GGUF}" || ! -f "${GGUF}" ]]; then
    echo "missing GGUF; pass a path or place a native pack under examples/models/" >&2
    exit 1
fi
if [[ ! -x "${BIN}" ]]; then
    echo "missing ${BIN}; build first" >&2
    exit 1
fi

echo "== edgexpu dump-logits =="
DUMP="$("${BIN}" dump-logits "${GGUF}" "${PROMPT}" "${N}")"
printf '%s\n' "${DUMP}"
EDGE_IDS="$(printf '%s\n' "${DUMP}" | sed -n 's/^token_ids=//p')"
EDGE_GREEDY="$(printf '%s\n' "${DUMP}" | sed -n 's/^greedy_ids=//p')"
EDGE_GREEDY_TEXT="$(printf '%s\n' "${DUMP}" | sed -n 's/^greedy_text=//p')"

if [[ "${N}" == "${GREEDY_N}" ]]; then
    if match_pack_by_gguf "${GGUF}" && [[ "${NATIVE}" == "1" && -n "${GREEDY_IDS}" ]]; then
        echo "lock pack=${PACK_NAME} token_ids=${PROMPT_IDS} greedy_ids=${GREEDY_IDS}"
        if [[ "${EDGE_IDS}" != "${PROMPT_IDS}" || "${EDGE_GREEDY}" != "${GREEDY_IDS}" ]]; then
            echo "lock MISMATCH edge_ids=${EDGE_IDS} edge_greedy=${EDGE_GREEDY}" >&2
            exit 1
        fi
        echo "lock: MATCH"
    else
        echo "lock: no native verify.lock for this GGUF (skip id check)"
    fi
fi

TOKENIZE="${LLAMA_BIN_DIR}/llama-tokenize"
if [[ -x "${TOKENIZE}" ]]; then
    echo
    echo "== llama-tokenize --no-bos =="
    LLAMA_IDS="$("${TOKENIZE}" -m "${GGUF}" --ids --no-bos -p "${PROMPT}" | tr -d '[] ' )"
    echo "llama_token_ids=${LLAMA_IDS}"
    if [[ "${EDGE_IDS}" == "${LLAMA_IDS}" ]]; then
        echo "token_ids: MATCH"
    else
        echo "token_ids: MISMATCH edge=${EDGE_IDS} llama=${LLAMA_IDS}" >&2
        exit 1
    fi
fi

CLI="${LLAMA_BIN_DIR}/llama-cli"
if [[ -x "${CLI}" ]]; then
    echo
    echo "== llama-cli --temp 0 --no-conversation -n ${N} =="
    LLAMA_OUT="$("${CLI}" -m "${GGUF}" -p "${PROMPT}" -n "${N}" --temp 0 --no-conversation \
        --no-display-prompt -ngl 0 --log-disable 2>/dev/null | tr -d '\r')"
    echo "llama_text=${LLAMA_OUT}"
    echo "edge_greedy_text=${EDGE_GREEDY_TEXT}"
    echo "edge_greedy_ids=${EDGE_GREEDY}"
fi
