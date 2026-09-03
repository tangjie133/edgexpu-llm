# Shared helpers for verify_mvp.sh / verify_arm.sh / align_llama.sh
# Per-pack numerical locks: examples/models/<pack>/verify.lock
# Caller must set ROOT_DIR. Pack metadata is read from JSON (no exec of BIN),
# so cross-compiled aarch64 edgexpu is not required to resolve GGUF paths.
# shellcheck source=verify.locks
source "${ROOT_DIR}/scripts/verify.locks"

MODELS_DIR="${ROOT_DIR}/examples/models"

if ! command -v python3 >/dev/null 2>&1; then
    echo "verification failed: python3 is required to read model.manifest.json" >&2
    exit 1
fi

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
    local dir rel
    dir="$(cd "$(dirname "${manifest}")" && pwd)"
    rel="$(python3 -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    m = json.load(f)
arts = m.get("artifacts") or []
print(arts[0]["path"] if arts else "")
' "${manifest}")"
    if [[ -z "${rel}" ]]; then
        printf ''
        return 0
    fi
    if [[ "${rel}" = /* ]]; then
        printf '%s' "${rel}"
    else
        python3 -c 'import os,sys; print(os.path.normpath(os.path.join(sys.argv[1], sys.argv[2])))' "${dir}" "${rel}"
    fi
}

pack_model_id() {
    local manifest="$1"
    python3 -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    m = json.load(f)
print(m.get("model_id") or "")
' "${manifest}"
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
