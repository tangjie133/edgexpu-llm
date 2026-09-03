# Shared helpers for verify_mvp.sh / verify_arm.sh / align_llama.sh
# Per-pack numerical locks: examples/models/<pack>/verify.lock
# Caller must set ROOT_DIR. BIN is required for load_pack / pack_gguf_path.
# shellcheck source=verify.locks
source "${ROOT_DIR}/scripts/verify.locks"

MODELS_DIR="${ROOT_DIR}/examples/models"

die() {
    echo "verification failed: $*" >&2
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

pack_gguf_path() {
    local manifest="$1"
    local line
    line="$("${BIN}" inspect-manifest "${manifest}" | sed -n 's/^artifact.path: //p')"
    printf '%s' "${line}"
}

pack_model_id() {
    local manifest="$1"
    "${BIN}" inspect-manifest "${manifest}" | sed -n 's/^model_id: //p'
}

# Sets: PACK_DIR PACK_NAME MANIFEST GGUF MODEL_ID ADAPTER NATIVE PROMPT_IDS GREEDY_IDS MANIFEST_CONTAINS
load_pack() {
    local pack_dir="$1"
    PACK_DIR="${pack_dir}"
    PACK_NAME="$(basename "${pack_dir}")"
    MANIFEST="${pack_dir}/model.manifest.json"
    ADAPTER=""
    NATIVE=0
    PROMPT_IDS=""
    GREEDY_IDS=""
    MANIFEST_CONTAINS=""
    MODEL_ID=""
    GGUF=""

    if [[ ! -f "${MANIFEST}" ]]; then
        return 1
    fi
    if [[ -f "${pack_dir}/verify.lock" ]]; then
        # shellcheck source=/dev/null
        source "${pack_dir}/verify.lock"
    fi
    GGUF="$(pack_gguf_path "${MANIFEST}")"
    MODEL_ID="$(pack_model_id "${MANIFEST}")"
    return 0
}

each_pack_dir() {
    find "${MODELS_DIR}" -mindepth 1 -maxdepth 1 -type d | sort
}

# Resolve verify.lock for a GGUF by matching inspect-manifest artifact.path.
match_pack_by_gguf() {
    local want="$1"
    local want_abs pack_dir have_abs
    want_abs="$(cd "$(dirname "${want}")" && pwd)/$(basename "${want}")"
    while IFS= read -r pack_dir; do
        if ! load_pack "${pack_dir}"; then
            continue
        fi
        if [[ ! -f "${pack_dir}/verify.lock" || ! -f "${GGUF}" ]]; then
            continue
        fi
        have_abs="$(cd "$(dirname "${GGUF}")" && pwd)/$(basename "${GGUF}")"
        if [[ "${have_abs}" == "${want_abs}" ]]; then
            return 0
        fi
    done < <(each_pack_dir)
    return 1
}
