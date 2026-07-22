#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

VARIANT="${1:?Usage: $0 B0|B1|C|M|M_PAIR_SHUFFLE [Hydra overrides...]}"
shift
USER_OVERRIDES=("$@")
VARIANT="$(printf '%s' "${VARIANT}" | tr '[:lower:]' '[:upper:]')"
case "${VARIANT}" in
  B0|B1|C|M|M_PAIR_SHUFFLE) ;;
  *)
    echo "Unsupported variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

STRICT_COMMON_INIT_COMPARISON="${FITWAM_STRICT_COMMON_INIT_COMPARISON:-0}"
case "${STRICT_COMMON_INIT_COMPARISON}" in
  0|1) ;;
  *)
    echo "FITWAM_STRICT_COMMON_INIT_COMPARISON must be 0 or 1." >&2
    exit 2
    ;;
esac
if [[ "${VARIANT}" == "M_PAIR_SHUFFLE" && \
      "${STRICT_COMMON_INIT_COMPARISON}" != "1" ]]; then
  echo \
    "M_PAIR_SHUFFLE requires FITWAM_STRICT_COMMON_INIT_COMPARISON=1." \
    >&2
  exit 2
fi
STRICT_COMMON_INIT_FOR_SELECTED=0
if [[ "${STRICT_COMMON_INIT_COMPARISON}" == "1" && \
      ( "${VARIANT}" == "C" || "${VARIANT}" == "M" || \
        "${VARIANT}" == "M_PAIR_SHUFFLE" ) ]]; then
  STRICT_COMMON_INIT_FOR_SELECTED=1
fi
INCLUDE_PAIR_SHUFFLE_CONTROL=0
if [[ "${VARIANT}" == "M_PAIR_SHUFFLE" ]]; then
  INCLUDE_PAIR_SHUFFLE_CONTROL=1
fi

PREFORMAL_MODE="${FITWAM_PREFORMAL_MODE:-formal}"
FIXED_TRAINING_OVERRIDES=()
OUTPUT_NAMESPACE="./runs/dexjoco_water_plant_offline_self_improving"
WANDB_GROUP="water_plant_offline_self_improving"
EXPECTED_RESUME_STEP=""
case "${PREFORMAL_MODE}" in
  formal)
    MAX_STEPS=6500
    EVAL_EVERY=500
    SAVE_WEIGHTS_EVERY=0
    SAVE_STATE_EVERY=1500
    FIXED_TRAINING_OVERRIDES=(
      "max_steps=${MAX_STEPS}"
      "eval_every=${EVAL_EVERY}"
      "save_weights_every=${SAVE_WEIGHTS_EVERY}"
      "save_weight_steps=[500,1000,3000,5000,6000,6500]"
      "save_state_every=${SAVE_STATE_EVERY}"
      "state_keep_last=1"
      "wandb.mode=online"
      "wandb.group=${WANDB_GROUP}"
    )
    ;;
  smoke20)
    MAX_STEPS=20
    EVAL_EVERY=10
    SAVE_WEIGHTS_EVERY=10
    SAVE_STATE_EVERY=20
    OUTPUT_NAMESPACE="${OUTPUT_NAMESPACE}/preformal_smoke/smoke20"
    WANDB_GROUP="water_plant_offline_self_improving_preformal_smoke"
    FIXED_TRAINING_OVERRIDES=(
      "max_steps=${MAX_STEPS}"
      "eval_every=${EVAL_EVERY}"
      "save_weights_every=${SAVE_WEIGHTS_EVERY}"
      "save_weight_steps=null"
      "save_state_every=${SAVE_STATE_EVERY}"
      "state_keep_last=1"
      "+lr_scheduler_total_steps=500"
      "wandb.mode=online"
      "wandb.group=${WANDB_GROUP}"
      "+experiment_provenance.run_mode=preformal_smoke"
    )
    ;;
  smoke500)
    MAX_STEPS=500
    EVAL_EVERY=100
    SAVE_WEIGHTS_EVERY=100
    SAVE_STATE_EVERY=500
    OUTPUT_NAMESPACE="${OUTPUT_NAMESPACE}/preformal_smoke/smoke500"
    WANDB_GROUP="water_plant_offline_self_improving_preformal_smoke"
    FIXED_TRAINING_OVERRIDES=(
      "max_steps=${MAX_STEPS}"
      "eval_every=${EVAL_EVERY}"
      "save_weights_every=${SAVE_WEIGHTS_EVERY}"
      "save_weight_steps=null"
      "save_state_every=${SAVE_STATE_EVERY}"
      "state_keep_last=1"
      "+lr_scheduler_total_steps=500"
      "wandb.mode=online"
      "wandb.group=${WANDB_GROUP}"
      "+experiment_provenance.run_mode=preformal_smoke"
    )
    ;;
  *)
    echo \
      "FITWAM_PREFORMAL_MODE must be formal, smoke20, or smoke500; got ${PREFORMAL_MODE}" \
      >&2
    exit 2
    ;;
esac

PAIR_QUALITY_GATE_STATUS="${PAIR_QUALITY_GATE_STATUS:?Source the frozen execution environment containing PAIR_QUALITY_GATE_STATUS}"
PAIR_QUALITY_GATE_MODE="${PAIR_QUALITY_GATE_MODE:?Source the frozen execution environment containing PAIR_QUALITY_GATE_MODE}"
case "${PAIR_QUALITY_GATE_STATUS}" in
  passed) ;;
  failed)
    if [[ "${PREFORMAL_MODE}" == "formal" || \
          "${PAIR_QUALITY_GATE_MODE}" != "preformal" ]]; then
      echo \
        "Formal training is blocked because event-pair quality status is failed." \
        >&2
      exit 2
    fi
    ;;
  *)
    echo \
      "PAIR_QUALITY_GATE_STATUS must be passed or failed; got ${PAIR_QUALITY_GATE_STATUS}" \
      >&2
    exit 2
    ;;
esac
if [[ "${PREFORMAL_MODE}" != "formal" ]]; then
  FIXED_TRAINING_OVERRIDES+=(
    "+experiment_provenance.pair_quality_gate_status=${PAIR_QUALITY_GATE_STATUS}"
  )
fi

validate_user_overrides() {
  local raw normalized key
  for raw in "${USER_OVERRIDES[@]+"${USER_OVERRIDES[@]}"}"; do
    if [[ "${raw}" == -* ]]; then
      echo "Hydra CLI flags are not allowed by the formal launcher: ${raw}" >&2
      exit 2
    fi
    normalized="${raw}"
    while [[ "${normalized}" == [+\~]* ]]; do
      normalized="${normalized:1}"
    done
    if [[ "${normalized}" != *=* ]]; then
      echo \
        "Formal protocol override rejected: expected key=value, got ${raw}" \
        >&2
      exit 2
    fi
    key="${normalized%%=*}"
    case "${key}" in
      num_workers)
        ;;
      *)
        echo \
          "Formal protocol override rejected: ${raw}. Only num_workers=... is allowed." \
          >&2
        exit 2
        ;;
    esac
  done
}
validate_user_overrides

if [[ "${PREFORMAL_MODE}" == "smoke500" && -z "${RESUME_STATE_DIR:-}" ]]; then
  echo \
    "smoke500 requires non-empty RESUME_STATE_DIR pointing to a complete step-20 state." \
    >&2
  exit 2
fi

if [[ "${FITWAM_VALIDATE_OVERRIDES_ONLY:-0}" == "1" ]]; then
  printf \
    'execution_mode=%s max_steps=%s eval_every=%s save_weights_every=%s save_state_every=%s output_namespace=%s wandb_group=%s\n' \
    "${PREFORMAL_MODE}" \
    "${MAX_STEPS}" \
    "${EVAL_EVERY}" \
    "${SAVE_WEIGHTS_EVERY}" \
    "${SAVE_STATE_EVERY}" \
    "${OUTPUT_NAMESPACE}" \
    "${WANDB_GROUP}"
  exit 0
fi

activate_fitwam_environment() {
  if [[ -n "${FITWAM_ENV_PREFIX:-}" ]]; then
    if [[ ! -x "${FITWAM_ENV_PREFIX}/bin/python" ]]; then
      echo \
        "FITWAM_ENV_PREFIX has no executable bin/python: ${FITWAM_ENV_PREFIX}" \
        >&2
      exit 2
    fi
    export PATH="${FITWAM_ENV_PREFIX}/bin:${PATH}"
    return
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo \
      "Neither FITWAM_ENV_PREFIX nor conda is available; cannot activate the training environment." \
      >&2
    exit 2
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${FITWAM_ENV:-fitwam}"
}
activate_fitwam_environment
if [[ -d "${HOME}/.local/bin" ]]; then
  export PATH="${HOME}/.local/bin:${PATH}"
fi
if ! command -v accelerate >/dev/null 2>&1; then
  echo \
    "accelerate is unavailable after environment activation; check FITWAM_ENV_PREFIX and PATH." \
    >&2
  exit 2
fi
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#GPU_ARRAY[@]}" -ne 4 ]]; then
  echo "Formal Offline Self-Improving runs require exactly four GPUs." >&2
  exit 2
fi

export INIT_WEIGHTS="${INIT_WEIGHTS:?Set INIT_WEIGHTS to the verified S0 checkpoint}"
if [[ ! -f "${INIT_WEIGHTS}" ]]; then
  echo "INIT_WEIGHTS must be a readable file: ${INIT_WEIGHTS}" >&2
  exit 2
fi
export SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?Set SOURCE_CHECKPOINT to the verified S0 .pt checkpoint}"
if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "SOURCE_CHECKPOINT must be a readable file: ${SOURCE_CHECKPOINT}" >&2
  exit 2
fi
SELECTED_INIT_WEIGHTS="${INIT_WEIGHTS}"
if [[ "${STRICT_COMMON_INIT_FOR_SELECTED}" == "1" ]]; then
  export COMMON_INIT_WEIGHTS="${COMMON_INIT_WEIGHTS:?Strict common-init comparison requires COMMON_INIT_WEIGHTS}"
  export COMMON_INIT_PROOF="${COMMON_INIT_PROOF:?Strict common-init comparison requires COMMON_INIT_PROOF}"
  export COMMON_INIT_CONFIG="${COMMON_INIT_CONFIG:?Strict common-init comparison requires COMMON_INIT_CONFIG}"
  export COMMON_INIT_SEED="${COMMON_INIT_SEED:?Strict common-init comparison requires COMMON_INIT_SEED}"
  export COMMON_INIT_WEIGHTS_SHA256="${COMMON_INIT_WEIGHTS_SHA256:?Strict common-init comparison requires COMMON_INIT_WEIGHTS_SHA256}"
  export COMMON_INIT_PROOF_SHA256="${COMMON_INIT_PROOF_SHA256:?Strict common-init comparison requires COMMON_INIT_PROOF_SHA256}"
  export COMMON_INIT_BASELINE_SHA256="${COMMON_INIT_BASELINE_SHA256:?Strict common-init comparison requires COMMON_INIT_BASELINE_SHA256}"
  export COMMON_INIT_CONFIG_SHA256="${COMMON_INIT_CONFIG_SHA256:?Strict common-init comparison requires COMMON_INIT_CONFIG_SHA256}"
  for common_artifact in \
    "${COMMON_INIT_WEIGHTS}" \
    "${COMMON_INIT_PROOF}" \
    "${COMMON_INIT_CONFIG}"; do
    if [[ ! -f "${common_artifact}" ]]; then
      echo "Missing strict common-init artifact: ${common_artifact}" >&2
      exit 2
    fi
  done
  if [[ ! "${COMMON_INIT_SEED}" =~ ^[0-9]+$ ]]; then
    echo "COMMON_INIT_SEED must be a non-negative integer." >&2
    exit 2
  fi
  SELECTED_INIT_WEIGHTS="${COMMON_INIT_WEIGHTS}"
fi
if [[ -n "${RESUME_STATE_DIR:-}" && ! -d "${RESUME_STATE_DIR}" ]]; then
  echo "RESUME_STATE_DIR must be a full state directory: ${RESUME_STATE_DIR}" >&2
  exit 2
fi
if [[ "${PREFORMAL_MODE}" == "smoke20" && -n "${RESUME_STATE_DIR:-}" ]]; then
  echo "smoke20 must start from INIT_WEIGHTS; RESUME_STATE_DIR is not allowed." >&2
  exit 2
fi
if [[ "${PREFORMAL_MODE}" == "smoke500" ]]; then
  EXPECTED_RESUME_STEP=20
fi
export FASTWAM_SOURCE_CONFIG="${FASTWAM_SOURCE_CONFIG:?Set FASTWAM_SOURCE_CONFIG to S0 resolved config.yaml}"
export SOURCE_BUNDLE_MANIFEST="${SOURCE_BUNDLE_MANIFEST:?Set SOURCE_BUNDLE_MANIFEST to the atomic S0 bundle_manifest.txt}"
export BASE_DATASET="${BASE_DATASET:?Set BASE_DATASET to the frozen expert-success LeRobot root}"
export ROLLOUT_RAW="${ROLLOUT_RAW:?Set ROLLOUT_RAW to the frozen S0 rollout LeRobot root}"
export NORM_STATS_SOURCE="${NORM_STATS_SOURCE:?Set NORM_STATS_SOURCE to meta or compute}"
export NORM_STATS_BUNDLE_SHA256="${NORM_STATS_BUNDLE_SHA256:?Set NORM_STATS_BUNDLE_SHA256 from the frozen execution environment}"
export TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:?Set TEXT_EMBEDDING_CACHE_DIR to the FastWAM WAN text cache}"
case "${NORM_STATS_SOURCE}" in
  meta)
    export NORM_STATS_META_DIR="${NORM_STATS_META_DIR:?meta normalization requires NORM_STATS_META_DIR}"
    unset PRETRAINED_NORM_STATS || true
    ;;
  compute)
    export PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS:?dataset normalization requires PRETRAINED_NORM_STATS}"
    unset NORM_STATS_META_DIR || true
    ;;
  *)
    echo "NORM_STATS_SOURCE must be meta or compute; got ${NORM_STATS_SOURCE}" >&2
    exit 2
    ;;
esac
export B0_MANIFEST_PATH="${B0_MANIFEST_PATH:?Set B0_MANIFEST_PATH}"
export B1_MANIFEST_PATH="${B1_MANIFEST_PATH:?Set B1_MANIFEST_PATH}"
export C_MANIFEST_PATH="${C_MANIFEST_PATH:-${B1_MANIFEST_PATH}}"
export M_MANIFEST_PATH="${M_MANIFEST_PATH:?Set M_MANIFEST_PATH}"
export EVE_VAL_MANIFEST_PATH="${EVE_VAL_MANIFEST_PATH:?Set EVE_VAL_MANIFEST_PATH to the frozen shared validation manifest}"
export PAIR_TARGETS_PATH="${PAIR_TARGETS_PATH:?Set PAIR_TARGETS_PATH}"
PAIR_TARGETS_ARTIFACT_PATH="${PAIR_TARGETS_PATH}"
PAIR_SHUFFLE_TARGETS_ARTIFACT_PATH=""
if [[ "${INCLUDE_PAIR_SHUFFLE_CONTROL}" == "1" ]]; then
  export M_PAIR_SHUFFLE_MANIFEST_PATH="${M_PAIR_SHUFFLE_MANIFEST_PATH:?M_PAIR_SHUFFLE requires M_PAIR_SHUFFLE_MANIFEST_PATH}"
  export PAIR_SHUFFLE_TARGETS_PATH="${PAIR_SHUFFLE_TARGETS_PATH:?M_PAIR_SHUFFLE requires PAIR_SHUFFLE_TARGETS_PATH}"
  PAIR_SHUFFLE_TARGETS_ARTIFACT_PATH="${PAIR_SHUFFLE_TARGETS_PATH}"
  export PAIR_SHUFFLE_PROOF_PATH="${PAIR_SHUFFLE_PROOF_PATH:?M_PAIR_SHUFFLE requires PAIR_SHUFFLE_PROOF_PATH}"
  export PAIR_SHUFFLE_SEED="${PAIR_SHUFFLE_SEED:?M_PAIR_SHUFFLE requires PAIR_SHUFFLE_SEED}"
  if [[ ! "${PAIR_SHUFFLE_SEED}" =~ ^[0-9]+$ ]]; then
    echo "PAIR_SHUFFLE_SEED must be a non-negative integer." >&2
    exit 2
  fi
fi
export TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:?Set TEACHER_CHECKPOINT to the frozen best Teacher checkpoint}"
PROTOCOL_BUNDLE_BASE_PATH="${PROTOCOL_BUNDLE_PATH:?Set PROTOCOL_BUNDLE_PATH for this frozen experiment matrix}"
export PROTOCOL_BUNDLE_PATH="${PROTOCOL_BUNDLE_BASE_PATH}.${VARIANT}"
if [[ "${STRICT_COMMON_INIT_FOR_SELECTED}" == "1" ]]; then
  export PROTOCOL_BUNDLE_PATH="${PROTOCOL_BUNDLE_PATH}.strict_common_init"
fi
export FITWAM_VARIANT="${VARIANT}"
if [[ "${PREFORMAL_MODE}" == "formal" ]]; then
  export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_${VARIANT}}"
  if [[ "${STRICT_COMMON_INIT_FOR_SELECTED}" == "1" ]]; then
    export WANDB_RUN_NAME="${WANDB_RUN_NAME:-water_plant_offline_strict_common_init_${VARIANT}_${RUN_ID}}"
  else
    export WANDB_RUN_NAME="${WANDB_RUN_NAME:-water_plant_offline_${VARIANT}_${RUN_ID}}"
  fi
else
  export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_${VARIANT}_${PREFORMAL_MODE}}"
  if [[ "${STRICT_COMMON_INIT_FOR_SELECTED}" == "1" ]]; then
    export WANDB_RUN_NAME="water_plant_offline_${PREFORMAL_MODE}_strict_common_init_${VARIANT}_${RUN_ID}"
  else
    export WANDB_RUN_NAME="water_plant_offline_${PREFORMAL_MODE}_${VARIANT}_${RUN_ID}"
  fi
  export PROTOCOL_BUNDLE_PATH="${PROTOCOL_BUNDLE_PATH}.${PREFORMAL_MODE}"
fi

REUSED_PREFLIGHT_REPORT="${FITWAM_REUSE_PREFLIGHT_REPORT:-}"
if [[ -n "${REUSED_PREFLIGHT_REPORT}" ]]; then
  if [[ "${STRICT_COMMON_INIT_FOR_SELECTED}" != "1" ]]; then
    echo "Pinned preflight reuse is limited to strict common-init comparisons." >&2
    exit 2
  fi
  python - \
    "${REUSED_PREFLIGHT_REPORT}" \
    "${FITWAM_REUSE_PREFLIGHT_REPORT_SHA256:-}" \
    "${VARIANT}" \
    "${PREFORMAL_MODE}" \
    "${PROTOCOL_BUNDLE_PATH}" <<'PY'
import hashlib
import json
import pathlib
import sys

report_path = pathlib.Path(sys.argv[1]).expanduser().resolve()
expected_sha256, variant, execution_mode = sys.argv[2:5]
protocol_bundle = str(pathlib.Path(sys.argv[5]).expanduser().resolve())

if not expected_sha256:
    raise SystemExit("FITWAM_REUSE_PREFLIGHT_REPORT_SHA256 is required")
payload = report_path.read_bytes()
actual_sha256 = hashlib.sha256(payload).hexdigest()
if actual_sha256 != expected_sha256:
    raise SystemExit(
        f"reused preflight SHA256 mismatch: {actual_sha256} != {expected_sha256}"
    )
report = json.loads(payload)
expected = {
    "status": "passed",
    "variant": variant,
    "execution_mode": execution_mode,
    "protocol_bundle": protocol_bundle,
}
for key, value in expected.items():
    observed = report.get(key)
    if observed != value:
        raise SystemExit(
            f"reused preflight {key} mismatch: {observed!r} != {value!r}"
        )
print(f"[offline] validated pinned preflight report {report_path}")
PY
fi

hash_file() {
  if [[ -n "${REUSED_PREFLIGHT_REPORT}" ]]; then
    case "$1" in
      "${INIT_WEIGHTS}"|"${SOURCE_CHECKPOINT}")
        printf '%s\n' "${COMMON_INIT_BASELINE_SHA256}"
        return
        ;;
      "${COMMON_INIT_WEIGHTS}")
        printf '%s\n' "${COMMON_INIT_WEIGHTS_SHA256}"
        return
        ;;
      "${COMMON_INIT_PROOF}")
        printf '%s\n' "${COMMON_INIT_PROOF_SHA256}"
        return
        ;;
      "${COMMON_INIT_CONFIG}")
        printf '%s\n' "${COMMON_INIT_CONFIG_SHA256}"
        return
        ;;
      "${TEACHER_CHECKPOINT}")
        printf '%s\n' "${PAIR_TARGETS_TEACHER_SHA256}"
        return
        ;;
    esac
  fi
  python - "$1" <<'PY'
import hashlib
from pathlib import Path
import sys

path = Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

S0_INIT_WEIGHTS_SHA256="$(hash_file "${INIT_WEIGHTS}")"
SOURCE_CHECKPOINT_SHA256="$(hash_file "${SOURCE_CHECKPOINT}")"
if [[ "${S0_INIT_WEIGHTS_SHA256}" != "${SOURCE_CHECKPOINT_SHA256}" ]]; then
  echo "INIT_WEIGHTS must be the same frozen S0 weights as SOURCE_CHECKPOINT." >&2
  exit 2
fi
if [[ "${STRICT_COMMON_INIT_FOR_SELECTED}" == "1" ]]; then
  if [[ "$(hash_file "${COMMON_INIT_WEIGHTS}")" != \
        "${COMMON_INIT_WEIGHTS_SHA256}" ]]; then
    echo "COMMON_INIT_WEIGHTS_SHA256 does not match COMMON_INIT_WEIGHTS." >&2
    exit 2
  fi
  if [[ "$(hash_file "${COMMON_INIT_PROOF}")" != \
        "${COMMON_INIT_PROOF_SHA256}" ]]; then
    echo "COMMON_INIT_PROOF_SHA256 does not match COMMON_INIT_PROOF." >&2
    exit 2
  fi
  if [[ "$(hash_file "${COMMON_INIT_CONFIG}")" != \
        "${COMMON_INIT_CONFIG_SHA256}" ]]; then
    echo "COMMON_INIT_CONFIG_SHA256 does not match COMMON_INIT_CONFIG." >&2
    exit 2
  fi
  if [[ "${SOURCE_CHECKPOINT_SHA256}" != \
        "${COMMON_INIT_BASELINE_SHA256}" ]]; then
    echo "COMMON_INIT_BASELINE_SHA256 does not match the frozen S0." >&2
    exit 2
  fi
fi
export FASTWAM_RESUME_SHA256="$(hash_file "${SELECTED_INIT_WEIGHTS}")"
export FASTWAM_SOURCE_CONFIG_SHA256="$(hash_file "${FASTWAM_SOURCE_CONFIG}")"
export FITWAM_CODE_SNAPSHOT_SHA256="$(
  python - <<'PY'
from scripts.everobot.preflight_offline_run import build_code_snapshot

print(build_code_snapshot()["snapshot_sha256"])
PY
)"
PAIR_TARGETS_FILE_SHA256="$(hash_file "${PAIR_TARGETS_ARTIFACT_PATH}")"
PAIR_SHUFFLE_TARGETS_FILE_SHA256=""
PAIR_SHUFFLE_PROOF_FILE_SHA256=""
if [[ "${INCLUDE_PAIR_SHUFFLE_CONTROL}" == "1" ]]; then
  PAIR_SHUFFLE_TARGETS_FILE_SHA256="$(hash_file "${PAIR_SHUFFLE_TARGETS_ARTIFACT_PATH}")"
  PAIR_SHUFFLE_PROOF_FILE_SHA256="$(hash_file "${PAIR_SHUFFLE_PROOF_PATH}")"
fi
TEACHER_FILE_SHA256="$(hash_file "${TEACHER_CHECKPOINT}")"
if [[ -n "${PAIR_TARGETS_TEACHER_SHA256:-}" && \
      "${PAIR_TARGETS_TEACHER_SHA256}" != "${TEACHER_FILE_SHA256}" ]]; then
  echo "PAIR_TARGETS_TEACHER_SHA256 disagrees with TEACHER_CHECKPOINT." >&2
  exit 2
fi
export PAIR_TARGETS_TEACHER_SHA256="${TEACHER_FILE_SHA256}"

PREFLIGHT_DIR="${PREFLIGHT_DIR:-${ROOT_DIR}/runs/preflight}"
MIN_DISK_FREE_GIB="${FITWAM_MIN_DISK_FREE_GIB:-500}"
mkdir -p "${PREFLIGHT_DIR}"
PREFLIGHT_REPORT="${PREFLIGHT_DIR}/${RUN_ID}.json"
RESOLVED_CONFIG_DIR="${PROTOCOL_BUNDLE_PATH}.resolved"
mkdir -p "${RESOLVED_CONFIG_DIR}"

manifest_for_variant() {
  case "$1" in
    B0) printf '%s\n' "${B0_MANIFEST_PATH}" ;;
    B1) printf '%s\n' "${B1_MANIFEST_PATH}" ;;
    C) printf '%s\n' "${C_MANIFEST_PATH}" ;;
    M) printf '%s\n' "${M_MANIFEST_PATH}" ;;
    M_PAIR_SHUFFLE) printf '%s\n' "${M_PAIR_SHUFFLE_MANIFEST_PATH}" ;;
  esac
}

pair_targets_for_variant() {
  case "$1" in
    M_PAIR_SHUFFLE) printf '%s\n' "${PAIR_SHUFFLE_TARGETS_ARTIFACT_PATH}" ;;
    *) printf '%s\n' "${PAIR_TARGETS_ARTIFACT_PATH}" ;;
  esac
}

init_weights_for_variant() {
  case "$1" in
    C|M|M_PAIR_SHUFFLE)
      if [[ "${STRICT_COMMON_INIT_FOR_SELECTED}" == "1" ]]; then
        printf '%s\n' "${COMMON_INIT_WEIGHTS}"
      else
        printf '%s\n' "${INIT_WEIGHTS}"
      fi
      ;;
    *) printf '%s\n' "${INIT_WEIGHTS}" ;;
  esac
}

TEXT_CACHE_MANIFESTS=(
  "${B0_MANIFEST_PATH}"
  "${B1_MANIFEST_PATH}"
  "${C_MANIFEST_PATH}"
  "${M_MANIFEST_PATH}"
  "${EVE_VAL_MANIFEST_PATH}"
)
if [[ "${INCLUDE_PAIR_SHUFFLE_CONTROL}" == "1" ]]; then
  TEXT_CACHE_MANIFESTS+=("${M_PAIR_SHUFFLE_MANIFEST_PATH}")
fi
export TEXT_EMBEDDING_CACHE_SHA256="$(
  python - \
    "${TEXT_EMBEDDING_CACHE_DIR}" \
    "${TEXT_CACHE_MANIFESTS[@]}" <<'PY'
from pathlib import Path
import sys

from scripts.everobot.preflight_offline_run import (
    build_text_cache_contract,
    read_json,
    referenced_task_texts,
)

cache_dir = Path(sys.argv[1])
manifest_paths = [Path(path) for path in sys.argv[2:]]
samples = [
    sample
    for path in manifest_paths
    for sample in read_json(path).get("samples", [])
]
tasks = referenced_task_texts(
    samples,
    strip_marker="Failed to finish the whole process.",
)
contract = build_text_cache_contract(
    cache_dir,
    tasks=tasks,
    context_len=128,
)
print(contract["bundle_sha256"])
PY
)"

variant_contract() {
  case "$1" in
    B0|B1) printf '%s\n' "false 0.0 0" ;;
    C) printf '%s\n' "true 0.0 0" ;;
    M|M_PAIR_SHUFFLE) printf '%s\n' "true 0.1 500" ;;
  esac
}

generate_resolved_config() (
  local protocol_variant="$1"
  local output="$2"
  local manifest_path pair_targets_path init_weights_path
  local steer_enabled pair_weight pair_warmup
  local -a variant_provenance_overrides=()
  manifest_path="$(manifest_for_variant "${protocol_variant}")"
  pair_targets_path="$(pair_targets_for_variant "${protocol_variant}")"
  init_weights_path="$(init_weights_for_variant "${protocol_variant}")"
  read -r steer_enabled pair_weight pair_warmup <<< "$(
    variant_contract "${protocol_variant}"
  )"

  export FITWAM_VARIANT="${protocol_variant}"
  export EVE_MANIFEST_PATH="${manifest_path}"
  export EVE_MANIFEST_SHA256="$(hash_file "${manifest_path}")"
  export EVE_VAL_MANIFEST_PATH
  export EVE_VAL_MANIFEST_SHA256="$(hash_file "${EVE_VAL_MANIFEST_PATH}")"
  export FASTWAM_RESUME="${init_weights_path}"
  export FASTWAM_RESUME_SHA256="$(hash_file "${init_weights_path}")"
  if [[ "${STRICT_COMMON_INIT_FOR_SELECTED}" == "1" && \
        ( "${protocol_variant}" == "C" || \
          "${protocol_variant}" == "M" || \
          "${protocol_variant}" == "M_PAIR_SHUFFLE" ) ]]; then
    COMMON_INIT_PAYLOAD_PROOF_SHA256="$(
      python - "${COMMON_INIT_PROOF}" <<'PY'
import json
from pathlib import Path
import sys

print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["proof_sha256"])
PY
    )"
    variant_provenance_overrides+=(
      "+experiment_provenance.comparison_mode=strict_common_init_pair_shuffle"
      "+experiment_provenance.common_init_checkpoint_sha256=${COMMON_INIT_WEIGHTS_SHA256}"
      "+experiment_provenance.common_init_proof_file_sha256=${COMMON_INIT_PROOF_SHA256}"
      "+experiment_provenance.common_init_payload_proof_sha256=${COMMON_INIT_PAYLOAD_PROOF_SHA256}"
      "+experiment_provenance.common_init_baseline_sha256=${COMMON_INIT_BASELINE_SHA256}"
      "+experiment_provenance.common_init_config_sha256=${COMMON_INIT_CONFIG_SHA256}"
      "+experiment_provenance.common_init_seed=${COMMON_INIT_SEED}"
    )
  fi
  if [[ "${protocol_variant}" == "M" || \
        "${protocol_variant}" == "M_PAIR_SHUFFLE" ]]; then
    export PAIR_TARGETS_PATH="${pair_targets_path}"
    if [[ "${protocol_variant}" == "M_PAIR_SHUFFLE" ]]; then
      export PAIR_TARGETS_SHA256="${PAIR_SHUFFLE_TARGETS_FILE_SHA256}"
      variant_provenance_overrides+=(
        "+experiment_provenance.pair_shuffle_proof_sha256=${PAIR_SHUFFLE_PROOF_FILE_SHA256}"
        "+experiment_provenance.pair_shuffle_seed=${PAIR_SHUFFLE_SEED}"
        "+experiment_provenance.pair_shuffle_source_pair_targets_sha256=${PAIR_TARGETS_FILE_SHA256}"
      )
    else
      export PAIR_TARGETS_SHA256="${PAIR_TARGETS_FILE_SHA256}"
    fi
    export PAIR_TARGETS_TEACHER_SHA256
  else
    unset PAIR_TARGETS_PATH || true
    export PAIR_TARGETS_SHA256=none
    unset PAIR_TARGETS_TEACHER_SHA256 || true
  fi

  local temporary="${output}.tmp"
  python scripts/train.py --cfg job --resolve \
    task=dexjoco/dexjoco_water_plant_offline_self_improving_2cam_proprio_1e-4 \
    "${USER_OVERRIDES[@]+"${USER_OVERRIDES[@]}"}" \
    "${FIXED_TRAINING_OVERRIDES[@]+"${FIXED_TRAINING_OVERRIDES[@]}"}" \
    "${variant_provenance_overrides[@]+"${variant_provenance_overrides[@]}"}" \
    "output_dir=__PROTOCOL_OUTPUT__" \
    "wandb.name=protocol_${protocol_variant}" \
    "model.offline_steer.enabled=${steer_enabled}" \
    "model.offline_steer.pair_loss_weight=${pair_weight}" \
    "model.offline_steer.pair_loss_warmup_steps=${pair_warmup}" \
    > "${temporary}"
  mv "${temporary}" "${output}"
)

RESOLVED_CONFIG_ARGS=()
PROTOCOL_MANIFEST_ARGS=()
PROTOCOL_VARIANTS=(B0 B1 C M)
if [[ "${INCLUDE_PAIR_SHUFFLE_CONTROL}" == "1" ]]; then
  PROTOCOL_VARIANTS+=(M_PAIR_SHUFFLE)
fi
for protocol_variant in "${PROTOCOL_VARIANTS[@]}"; do
  protocol_manifest="$(manifest_for_variant "${protocol_variant}")"
  resolved_config="${RESOLVED_CONFIG_DIR}/${protocol_variant}.yaml"
  generate_resolved_config "${protocol_variant}" "${resolved_config}"
  PROTOCOL_MANIFEST_ARGS+=(
    --protocol-manifest "${protocol_variant}=${protocol_manifest}"
  )
  RESOLVED_CONFIG_ARGS+=(
    --resolved-config "${protocol_variant}=${resolved_config}"
  )
done

EVE_MANIFEST_PATH="$(manifest_for_variant "${VARIANT}")"
export EVE_MANIFEST_PATH
export EVE_MANIFEST_SHA256="$(hash_file "${EVE_MANIFEST_PATH}")"
export EVE_VAL_MANIFEST_SHA256="$(hash_file "${EVE_VAL_MANIFEST_PATH}")"
read -r STEER_ENABLED PAIR_WEIGHT PAIR_WARMUP <<< "$(
  variant_contract "${VARIANT}"
)"
VARIANT_PROVENANCE_OVERRIDES=()
if [[ "${STRICT_COMMON_INIT_FOR_SELECTED}" == "1" ]]; then
  COMMON_INIT_PAYLOAD_PROOF_SHA256="$(
    python - "${COMMON_INIT_PROOF}" <<'PY'
import json
from pathlib import Path
import sys

print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["proof_sha256"])
PY
  )"
  VARIANT_PROVENANCE_OVERRIDES+=(
    "+experiment_provenance.comparison_mode=strict_common_init_pair_shuffle"
    "+experiment_provenance.common_init_checkpoint_sha256=${COMMON_INIT_WEIGHTS_SHA256}"
    "+experiment_provenance.common_init_proof_file_sha256=${COMMON_INIT_PROOF_SHA256}"
    "+experiment_provenance.common_init_payload_proof_sha256=${COMMON_INIT_PAYLOAD_PROOF_SHA256}"
    "+experiment_provenance.common_init_baseline_sha256=${COMMON_INIT_BASELINE_SHA256}"
    "+experiment_provenance.common_init_config_sha256=${COMMON_INIT_CONFIG_SHA256}"
    "+experiment_provenance.common_init_seed=${COMMON_INIT_SEED}"
  )
fi
if [[ "${VARIANT}" == "M" || "${VARIANT}" == "M_PAIR_SHUFFLE" ]]; then
  export PAIR_TARGETS_PATH="$(pair_targets_for_variant "${VARIANT}")"
  if [[ "${VARIANT}" == "M_PAIR_SHUFFLE" ]]; then
    export PAIR_TARGETS_SHA256="${PAIR_SHUFFLE_TARGETS_FILE_SHA256}"
    VARIANT_PROVENANCE_OVERRIDES+=(
      "+experiment_provenance.pair_shuffle_proof_sha256=${PAIR_SHUFFLE_PROOF_FILE_SHA256}"
      "+experiment_provenance.pair_shuffle_seed=${PAIR_SHUFFLE_SEED}"
      "+experiment_provenance.pair_shuffle_source_pair_targets_sha256=${PAIR_TARGETS_FILE_SHA256}"
    )
  else
    export PAIR_TARGETS_SHA256="${PAIR_TARGETS_FILE_SHA256}"
  fi
  export PAIR_TARGETS_TEACHER_SHA256
else
  unset PAIR_TARGETS_PATH || true
  export PAIR_TARGETS_SHA256=none
  unset PAIR_TARGETS_TEACHER_SHA256 || true
fi
export FASTWAM_RESUME="${RESUME_STATE_DIR:-${SELECTED_INIT_WEIGHTS}}"

RESUME_ARGS=()
if [[ -n "${RESUME_STATE_DIR:-}" ]]; then
  RESUME_ARGS=(--resume-state-dir "${RESUME_STATE_DIR}")
  if [[ -n "${EXPECTED_RESUME_STEP}" ]]; then
    RESUME_ARGS+=(--expected-resume-step "${EXPECTED_RESUME_STEP}")
  fi
fi
SYSTEM_CHECK_ARGS=()
if [[ "${FITWAM_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  SYSTEM_CHECK_ARGS=(--skip-system-checks)
fi

PAIR_SHUFFLE_PREFLIGHT_ARGS=()
if [[ "${INCLUDE_PAIR_SHUFFLE_CONTROL}" == "1" ]]; then
  PAIR_SHUFFLE_PREFLIGHT_ARGS=(
    --include-pair-shuffle-control
    --pair-shuffle-targets "${PAIR_SHUFFLE_TARGETS_ARTIFACT_PATH}"
    --pair-shuffle-proof "${PAIR_SHUFFLE_PROOF_PATH}"
    --pair-shuffle-seed "${PAIR_SHUFFLE_SEED}"
  )
fi

COMMON_INIT_PREFLIGHT_ARGS=()
if [[ "${STRICT_COMMON_INIT_FOR_SELECTED}" == "1" ]]; then
  COMMON_INIT_PREFLIGHT_ARGS=(
    --strict-common-init-comparison
    --common-init-weights "${COMMON_INIT_WEIGHTS}"
    --common-init-proof "${COMMON_INIT_PROOF}"
    --common-init-config "${COMMON_INIT_CONFIG}"
    --common-init-seed "${COMMON_INIT_SEED}"
    --expected-common-init-weights-sha256 "${COMMON_INIT_WEIGHTS_SHA256}"
    --expected-common-init-proof-sha256 "${COMMON_INIT_PROOF_SHA256}"
    --expected-common-init-baseline-sha256 "${COMMON_INIT_BASELINE_SHA256}"
    --expected-common-init-config-sha256 "${COMMON_INIT_CONFIG_SHA256}"
  )
fi

if [[ -n "${REUSED_PREFLIGHT_REPORT}" ]]; then
  python - "${REUSED_PREFLIGHT_REPORT}" "${EVE_MANIFEST_PATH}" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
manifest = str(pathlib.Path(sys.argv[2]).expanduser().resolve())
if report.get("manifest") != manifest:
    raise SystemExit(
        f"reused preflight manifest mismatch: {report.get('manifest')!r} != {manifest!r}"
    )
PY
  cp "${REUSED_PREFLIGHT_REPORT}" "${PREFLIGHT_REPORT}"
else
  python scripts/everobot/preflight_offline_run.py \
    --variant "${VARIANT}" \
    --manifest "${EVE_MANIFEST_PATH}" \
    --selection-manifest "${EVE_VAL_MANIFEST_PATH}" \
    "${PROTOCOL_MANIFEST_ARGS[@]}" \
    "${RESOLVED_CONFIG_ARGS[@]}" \
    --init-weights "${INIT_WEIGHTS}" \
    --source-checkpoint "${SOURCE_CHECKPOINT}" \
    "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}" \
    --source-config "${FASTWAM_SOURCE_CONFIG}" \
    --source-bundle-manifest "${SOURCE_BUNDLE_MANIFEST}" \
    --pair-targets "${PAIR_TARGETS_ARTIFACT_PATH}" \
    "${PAIR_SHUFFLE_PREFLIGHT_ARGS[@]+"${PAIR_SHUFFLE_PREFLIGHT_ARGS[@]}"}" \
    "${COMMON_INIT_PREFLIGHT_ARGS[@]+"${COMMON_INIT_PREFLIGHT_ARGS[@]}"}" \
    --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
    --expected-teacher-sha256 "${TEACHER_FILE_SHA256}" \
    --expected-normalization-bundle-sha256 "${NORM_STATS_BUNDLE_SHA256}" \
    --expected-text-cache-sha256 "${TEXT_EMBEDDING_CACHE_SHA256}" \
    --protocol-bundle "${PROTOCOL_BUNDLE_PATH}" \
    --execution-mode "${PREFORMAL_MODE}" \
    --gpus "${CUDA_VISIBLE_DEVICES}" \
    --disk-root "${FITWAM_DISK_ROOT:-/data_all}" \
    --min-disk-free-gib "${MIN_DISK_FREE_GIB}" \
    "${SYSTEM_CHECK_ARGS[@]+"${SYSTEM_CHECK_ARGS[@]}"}" \
    --output "${PREFLIGHT_REPORT}"
fi

echo "[offline] preflight=${PREFLIGHT_REPORT}"
echo "[offline] variant=${VARIANT} run_id=${RUN_ID} gpus=${CUDA_VISIBLE_DEVICES}"
echo "[offline] init_weights=${SELECTED_INIT_WEIGHTS} strict_common_init=${STRICT_COMMON_INIT_FOR_SELECTED}"
echo \
  "[offline] execution_mode=${PREFORMAL_MODE} max_steps=${MAX_STEPS} " \
  "eval_every=${EVAL_EVERY} save_weights_every=${SAVE_WEIGHTS_EVERY} " \
  "save_state_every=${SAVE_STATE_EVERY}"
echo "[offline] load_mode=$([[ -n "${RESUME_STATE_DIR:-}" ]] && echo resume_state || echo init_weights)"

if [[ "${FITWAM_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[offline] preflight-only complete; training was not started"
  exit 0
fi

bash scripts/train_zero1.sh 4 \
  "${USER_OVERRIDES[@]+"${USER_OVERRIDES[@]}"}" \
  "${FIXED_TRAINING_OVERRIDES[@]+"${FIXED_TRAINING_OVERRIDES[@]}"}" \
  "${VARIANT_PROVENANCE_OVERRIDES[@]+"${VARIANT_PROVENANCE_OVERRIDES[@]}"}" \
  task=dexjoco/dexjoco_water_plant_offline_self_improving_2cam_proprio_1e-4 \
  "output_dir=${OUTPUT_NAMESPACE}/${RUN_ID}" \
  "wandb.name=${WANDB_RUN_NAME}" \
  "model.offline_steer.enabled=${STEER_ENABLED}" \
  "model.offline_steer.pair_loss_weight=${PAIR_WEIGHT}" \
  "model.offline_steer.pair_loss_warmup_steps=${PAIR_WARMUP}" \
  "resume=${FASTWAM_RESUME}"
