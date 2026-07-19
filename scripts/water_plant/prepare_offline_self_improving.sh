#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

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
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"

BASE_DATASET="${BASE_DATASET:?Set BASE_DATASET to the standard expert-success LeRobot root}"
ROLLOUT_DATASET="${ROLLOUT_DATASET:?Set ROLLOUT_DATASET to the fixed S0 rollout LeRobot root}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?Set SOURCE_CHECKPOINT to the verified S0 .pt}"
SOURCE_CONFIG="${SOURCE_CONFIG:?Set SOURCE_CONFIG to the resolved S0 config.yaml}"
SOURCE_BUNDLE_MANIFEST="${SOURCE_BUNDLE_MANIFEST:?Set SOURCE_BUNDLE_MANIFEST to the atomic S0 bundle_manifest.txt}"
EVE_ROOT="${EVE_ROOT:?Set EVE_ROOT to a new EveRobot v0.2 sidecar directory}"
TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:?Set TEXT_EMBEDDING_CACHE_DIR to the FastWAM WAN text-cache directory}"
export BASE_DATASET
export ROLLOUT_RAW="${ROLLOUT_DATASET}"
export TEXT_EMBEDDING_CACHE_DIR

BASE_DATASET_ID="${BASE_DATASET_ID:-water_plant_expert_success}"
ROLLOUT_DATASET_ID="${ROLLOUT_DATASET_ID:-water_plant_s0_rollout}"
TASK_NAME="${TASK_NAME:-water_plant}"
SOURCE_POLICY="${SOURCE_POLICY:-fastwam_s0}"
SPLIT_SEED="${SPLIT_SEED:-20260717}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
TEACHER_DEVICE="${TEACHER_DEVICE:-cuda}"
PREFLIGHT_GPUS="${PREFLIGHT_GPUS:-0,1,2,3}"
TEACHER_OUTPUT="${TEACHER_OUTPUT:-${EVE_ROOT}/teacher/offline_steer_v1}"
SKIP_TEACHER="${SKIP_TEACHER:-0}"
RESUME_EXISTING_EVE="${RESUME_EXISTING_EVE:-0}"
PAIR_QUALITY_GATE_MODE="${PAIR_QUALITY_GATE_MODE:-formal}"
if [[ "${RESUME_EXISTING_EVE}" != "0" && "${RESUME_EXISTING_EVE}" != "1" ]]; then
  echo "RESUME_EXISTING_EVE must be 0 or 1." >&2
  exit 2
fi
case "${PAIR_QUALITY_GATE_MODE}" in
  formal|preformal) ;;
  *)
    echo \
      "PAIR_QUALITY_GATE_MODE must be formal or preformal; got ${PAIR_QUALITY_GATE_MODE}" \
      >&2
    exit 2
    ;;
esac
ROLLOUT_COLLECTION_ROOT="$(
  cd "$(dirname "${ROLLOUT_DATASET}")"
  pwd
)"
COLLECTION_PROTOCOL="${COLLECTION_PROTOCOL:-${ROLLOUT_COLLECTION_ROOT}/collection_protocol.json}"
OUTCOME_VALIDATION_REPORT="${OUTCOME_VALIDATION_REPORT:-${ROLLOUT_COLLECTION_ROOT}/outcome_validation.json}"
FORMAL_VALIDATION_REPORT="${FORMAL_VALIDATION_REPORT:-${ROLLOUT_COLLECTION_ROOT}/formal_protocol_validation.json}"
export FORMAL_VALIDATION_REPORT
WATER_PLANT_TEXT_CACHE_BASENAME="$(
  python - <<'PY'
import hashlib

task = "Grasp the watering can and apply water to the plant."
prompt = (
    "A video recorded from a robot's point of view executing the following "
    f"instruction: {task}"
)
digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
print(f"{digest}.t5_len128.wan22ti2v5b.pt")
PY
)"
TEXT_EMBEDDING_CACHE_FILE="${TEXT_EMBEDDING_CACHE_DIR}/${WATER_PLANT_TEXT_CACHE_BASENAME}"

NORMALIZATION_KIND="$(
  python - "${SOURCE_CONFIG}" <<'PY'
from pathlib import Path
import sys

import yaml

path = Path(sys.argv[1])
payload = yaml.safe_load(path.read_text(encoding="utf-8"))
try:
    source = payload["data"]["train"]["processor"].get(
        "norm_stats_source", "compute"
    )
except (KeyError, TypeError, AttributeError) as exc:
    raise SystemExit(f"{path}: missing resolved data.train.processor: {exc}")
print("meta" if str(source).strip().lower() == "meta" else "dataset")
PY
)"

case "${NORMALIZATION_KIND}" in
  meta)
    export NORM_STATS_SOURCE=meta
    export NORM_STATS_META_DIR="${NORM_STATS_META_DIR:-${BASE_DATASET}/meta}"
    unset PRETRAINED_NORM_STATS || true
    for required in \
      "${NORM_STATS_META_DIR}/stats.json" \
      "${NORM_STATS_META_DIR}/modality.json"; do
      if [[ ! -s "${required}" ]]; then
        echo "Missing or empty meta normalization artifact: ${required}" >&2
        exit 2
      fi
    done
    ;;
  dataset)
    export NORM_STATS_SOURCE=compute
    export PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS:?Source config uses dataset normalization; set PRETRAINED_NORM_STATS}"
    unset NORM_STATS_META_DIR || true
    if [[ ! -s "${PRETRAINED_NORM_STATS}" ]]; then
      echo \
        "Missing or empty dataset normalization artifact: ${PRETRAINED_NORM_STATS}" \
        >&2
      exit 2
    fi
    ;;
  *)
    echo "Unsupported normalization kind: ${NORMALIZATION_KIND}" >&2
    exit 2
    ;;
esac

if [[ ! -s "${TEXT_EMBEDDING_CACHE_FILE}" ]]; then
  echo \
    "Missing FastWAM WAN text cache: ${TEXT_EMBEDDING_CACHE_FILE}. " \
    "The rollout adapter's .npz cache is not a training cache." \
    >&2
  exit 2
fi

SPLIT_MAP="${EVE_ROOT}/splits/episode_splits.jsonl"
STATE_SCORES="${EVE_ROOT}/annotations/state_line_scores.parquet"
FEATURE_DIR="${EVE_ROOT}/features"
FEATURE_CALIBRATION="${FEATURE_DIR}/event_pair_calibration.json"
TRAIN_FEATURES="${FEATURE_DIR}/event_pair_train.parquet"
TRAIN_FEATURES_JSONL="${FEATURE_DIR}/event_pair_train.jsonl"
VAL_FEATURES="${FEATURE_DIR}/event_pair_val.parquet"
VAL_FEATURES_JSONL="${FEATURE_DIR}/event_pair_val.jsonl"
PAIR_LEDGER="${EVE_ROOT}/pairs/offline_steer_v1.jsonl"
PAIR_CALIBRATION="${EVE_ROOT}/pairs/offline_steer_v1_calibration.json"
PAIR_DIAGNOSTICS="${EVE_ROOT}/pairs/offline_steer_v1_diagnostics.json"
PAIR_QUALITY_REPORT="${EVE_ROOT}/quality/offline_event_pair_quality_v1.json"
STATE_LINE_AUDIT_DIR="${EVE_ROOT}/quality/state_line_audit_seed${SPLIT_SEED}"
STATE_LINE_AUDIT_INDEX="${STATE_LINE_AUDIT_DIR}/audit_index.json"
PAIR_TARGETS="${TEACHER_OUTPUT}/pair_targets.npz"
MANIFEST_B0_RAW="${EVE_ROOT}/manifests/offline_b0_success_budget_control_raw.json"
MANIFEST_B0="${EVE_ROOT}/manifests/offline_b0_success_budget_control.json"
MANIFEST_B0_MATCH_DIAGNOSTICS="${EVE_ROOT}/manifests/offline_b0_auxiliary_budget_match.json"
MANIFEST_B1="${EVE_ROOT}/manifests/offline_b1_failure_video_control.json"
MANIFEST_M="${EVE_ROOT}/manifests/offline_m_failure_steer.json"
MANIFEST_SELECTION="${EVE_ROOT}/manifests/offline_selection_primary_success.json"

hash_file() {
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

SOURCE_CHECKPOINT_SHA256="$(hash_file "${SOURCE_CHECKPOINT}")"
CODE_SNAPSHOT_SHA256="$(
  python - <<'PY'
from scripts.everobot.preflight_offline_run import build_code_snapshot

print(build_code_snapshot()["snapshot_sha256"])
PY
)"
SOURCE_GIT_COMMIT="$(
  git rev-parse HEAD 2>/dev/null ||
    printf '%s\n' "${SOURCE_GIT_COMMIT:-unknown}"
)"
CODE_COMMIT="${SOURCE_GIT_COMMIT}+snapshot.${CODE_SNAPSHOT_SHA256}"
export NORM_STATS_BUNDLE_SHA256="$(
  python - "${NORM_STATS_SOURCE}" \
    "${NORM_STATS_META_DIR:-}" \
    "${PRETRAINED_NORM_STATS:-}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


source, meta_dir, dataset_stats = sys.argv[1:]
if source == "meta":
    root = Path(meta_dir)
    artifacts = {
        "stats.json": sha256_file(root / "stats.json"),
        "modality.json": sha256_file(root / "modality.json"),
    }
else:
    artifacts = {
        "dataset_stats.json": sha256_file(Path(dataset_stats)),
    }
payload = {"kind": source, "artifacts": artifacts}
encoded = json.dumps(
    payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
).encode("utf-8")
print(hashlib.sha256(encoded).hexdigest())
PY
)"
SOURCE_BUNDLE_NORMALIZATION_SHA256="$(
  python - \
    "${SOURCE_BUNDLE_MANIFEST}" \
    "${SOURCE_CHECKPOINT}" \
    "${SOURCE_CONFIG}" <<'PY'
import hashlib
from pathlib import Path
import sys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest_path = Path(sys.argv[1]).expanduser().resolve()
checkpoint = Path(sys.argv[2]).expanduser().resolve()
config = Path(sys.argv[3]).expanduser().resolve()
if not manifest_path.is_file():
    raise SystemExit(f"Missing S0 bundle manifest: {manifest_path}")
root = manifest_path.parent
if checkpoint.parent != root or config.parent != root:
    raise SystemExit(
        "SOURCE_CHECKPOINT, SOURCE_CONFIG, and SOURCE_BUNDLE_MANIFEST "
        "must come from the same atomic S0 bundle directory"
    )

metadata = {}
listed = {}
in_hashes = False
for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line:
        continue
    if line == "sha256":
        in_hashes = True
        continue
    if not in_hashes:
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key] = value
        continue
    digest, relative = line.split(None, 1)
    relative = relative.strip()
    while relative.startswith("./"):
        relative = relative[2:]
    artifact = (root / relative).resolve()
    if root not in artifact.parents:
        raise SystemExit(f"Bundle hash path escapes root: {relative}")
    actual = sha256_file(artifact)
    if actual != digest:
        raise SystemExit(
            f"S0 bundle artifact hash mismatch: {relative}: {actual} != {digest}"
        )
    listed[relative] = digest

for name, path in (("step_006500.pt", checkpoint), ("config.yaml", config)):
    expected = listed.get(name)
    if expected is None or expected != sha256_file(path):
        raise SystemExit(f"S0 bundle does not bind {name}")
normalization_sha256 = metadata.get("normalization_bundle_sha256")
if not normalization_sha256:
    raise SystemExit("S0 bundle lacks normalization_bundle_sha256")
print(normalization_sha256)
PY
)"
if [[ "${SOURCE_BUNDLE_NORMALIZATION_SHA256}" != \
      "${NORM_STATS_BUNDLE_SHA256}" ]]; then
  echo \
    "Selected normalization does not match the atomic S0 source bundle." \
    >&2
  exit 2
fi

# Revalidate the complete frozen rollout immediately before any split or
# sidecar artifact is written. This removes operator memory from the protocol
# gate and rejects stale reports or datasets changed after collection.
python scripts/water_plant/validate_s0_formal_outputs.py \
  --protocol "${COLLECTION_PROTOCOL}" \
  --dataset "${ROLLOUT_DATASET}" \
  --outcome-validation "${OUTCOME_VALIDATION_REPORT}" \
  --report "${FORMAL_VALIDATION_REPORT}"

if [[ "${RESUME_EXISTING_EVE}" == "1" ]]; then
  for required in \
    "${SPLIT_MAP}" \
    "${EVE_ROOT}/schema_version.json" \
    "${EVE_ROOT}/round_meta.jsonl" \
    "${EVE_ROOT}/episode_meta.jsonl" \
    "${EVE_ROOT}/event_meta.jsonl" \
    "${STATE_SCORES}"; do
    if [[ ! -s "${required}" ]]; then
      echo "Cannot resume incomplete EveRobot preparation: ${required}" >&2
      exit 2
    fi
  done
else
  if [[ ! -f "${SPLIT_MAP}" ]]; then
    python scripts/everobot/build_episode_split.py \
      --dataset "${BASE_DATASET_ID}=${BASE_DATASET}" \
      --dataset "${ROLLOUT_DATASET_ID}=${ROLLOUT_DATASET}" \
      --force-success-dataset-id "${BASE_DATASET_ID}" \
      --require-explicit-outcome-dataset-id "${ROLLOUT_DATASET_ID}" \
      --val-fraction "${VAL_FRACTION}" \
      --seed "${SPLIT_SEED}" \
      --output "${SPLIT_MAP}"
  fi

  python scripts/everobot/build_eve_sidecar.py init-base \
    --dataset-root "${BASE_DATASET}" \
    --dataset-id "${BASE_DATASET_ID}" \
    --eve-root "${EVE_ROOT}" \
    --task-name "${TASK_NAME}" \
    --source-type expert_success \
    --source-policy expert \
    --collection-round -1 \
    --force-success \
    --split-map "${SPLIT_MAP}" \
    --config-path "${SOURCE_CONFIG}" \
    --code-commit "${CODE_COMMIT}"

  python scripts/everobot/build_eve_sidecar.py append-rollout \
    --base-eve-root "${EVE_ROOT}" \
    --rollout-root "${ROLLOUT_DATASET}" \
    --dataset-id "${ROLLOUT_DATASET_ID}" \
    --task-name "${TASK_NAME}" \
    --source-policy "${SOURCE_POLICY}" \
    --source-checkpoint "${SOURCE_CHECKPOINT}" \
    --source-checkpoint-sha256 "${SOURCE_CHECKPOINT_SHA256}" \
    --collection-round 0 \
    --failure-action-loss disabled \
    --require-explicit-outcomes \
    --split-map "${SPLIT_MAP}" \
    --config-path "${SOURCE_CONFIG}" \
    --code-commit "${CODE_COMMIT}"

  if [[ ! -f "${STATE_SCORES}" ]]; then
    python scripts/everobot/extract_state_line_events.py \
      --eve-root "${EVE_ROOT}" \
      --dataset-ids "${BASE_DATASET_ID}" "${ROLLOUT_DATASET_ID}" \
      --extract-splits train val \
      --calibration-split train \
      --scores-path "${STATE_SCORES}" \
      --median-window 5 \
      --ema-alpha 0.25 \
      --high-threshold 0.55 \
      --low-threshold 0.35 \
      --min-run 5 \
      --max-gap 8 \
      --pre-padding 12 \
      --post-padding 12 \
      --min-window 33 \
      --max-candidate 96 \
      --max-candidates-per-episode 10
  fi
fi

mkdir -p "${FEATURE_DIR}"
if [[ ! -f "${TRAIN_FEATURES}" ]]; then
  TRAIN_CALIBRATION_ARGS=(--fit-calibration "${FEATURE_CALIBRATION}")
  if [[ -f "${FEATURE_CALIBRATION}" ]]; then
    TRAIN_CALIBRATION_ARGS=(--calibration "${FEATURE_CALIBRATION}")
  fi
  python scripts/everobot/extract_event_pair_features.py \
    --eve-root "${EVE_ROOT}" \
    --split train \
    --output "${TRAIN_FEATURES}" \
    --jsonl-output "${TRAIN_FEATURES_JSONL}" \
    "${TRAIN_CALIBRATION_ARGS[@]}" \
    --event-types interaction_candidate \
    --pre-state-frames 4
fi
if [[ ! -f "${VAL_FEATURES}" ]]; then
  python scripts/everobot/extract_event_pair_features.py \
    --eve-root "${EVE_ROOT}" \
    --split val \
    --output "${VAL_FEATURES}" \
    --jsonl-output "${VAL_FEATURES_JSONL}" \
    --calibration "${FEATURE_CALIBRATION}" \
    --event-types interaction_candidate \
    --pre-state-frames 4
fi

if [[ ! -f "${PAIR_LEDGER}" ]]; then
  python scripts/everobot/build_event_pairs.py \
    --eve-root "${EVE_ROOT}" \
    --features "${TRAIN_FEATURES_JSONL}" "${VAL_FEATURES_JSONL}" \
    --output "${PAIR_LEDGER}" \
    --pairing-version offline_steer_v1 \
    --matching bounded \
    --max-success-uses 2 \
    --max-failure-uses 1 \
    --min-pair-weight 0.05 \
    --max-progress-delta "${MAX_PROGRESS_DELTA:-0.12}" \
    --max-pre-state-distance "${MAX_PRE_STATE_DISTANCE:-1.0}" \
    --tau-progress "${TAU_PROGRESS:-0.08}" \
    --fit-calibration "${PAIR_CALIBRATION}" \
    --diagnostics-output "${PAIR_DIAGNOSTICS}" \
    --event-types interaction_candidate \
    --splits train val
fi

set +e
python scripts/everobot/validate_offline_event_pair_quality.py \
  --episode-meta "${EVE_ROOT}/episode_meta.jsonl" \
  --event-meta "${EVE_ROOT}/event_meta.jsonl" \
  --pair-ledger "${PAIR_LEDGER}" \
  --pair-diagnostics "${PAIR_DIAGNOSTICS}" \
  --output "${PAIR_QUALITY_REPORT}"
PAIR_QUALITY_EXIT_CODE=$?
set -e
if [[ "${PAIR_QUALITY_EXIT_CODE}" -ne 0 ]]; then
  if [[ "${PAIR_QUALITY_EXIT_CODE}" -ne 1 || \
        "${PAIR_QUALITY_GATE_MODE}" != "preformal" ]]; then
    exit "${PAIR_QUALITY_EXIT_CODE}"
  fi
  echo \
    "[offline-prepare] WARNING: event-pair quality failed; continuing only " \
    "to build preformal smoke artifacts." \
    >&2
fi
PAIR_QUALITY_GATE_STATUS="$(
  python - "${PAIR_QUALITY_REPORT}" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = payload.get("status")
if status not in {"passed", "failed"}:
    raise SystemExit(f"Unsupported pair-quality status: {status!r}")
print(status)
PY
)"
export PAIR_QUALITY_GATE_MODE
export PAIR_QUALITY_GATE_STATUS

if [[ ! -f "${STATE_LINE_AUDIT_INDEX}" ]]; then
  python scripts/everobot/render_state_line_audit.py \
    --eve-root "${EVE_ROOT}" \
    --output-dir "${STATE_LINE_AUDIT_DIR}" \
    --num-episodes 20 \
    --seed "${SPLIT_SEED}"
fi

if [[ "${SKIP_TEACHER}" != "1" && ! -f "${PAIR_TARGETS}" ]]; then
  WANDB_ENTITY_ARGS=()
  TEACHER_RESUME_ARGS=()
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ENTITY_ARGS=(--wandb-entity "${WANDB_ENTITY}")
  fi
  if [[ -f "${TEACHER_OUTPUT}/last_teacher.pt" ]]; then
    TEACHER_RESUME_ARGS=(--resume)
  fi
  python scripts/everobot/train_offline_steer_teacher.py \
    --eve-root "${EVE_ROOT}" \
    --pair-ledger "${PAIR_LEDGER}" \
    --output-dir "${TEACHER_OUTPUT}" \
    "${TEACHER_RESUME_ARGS[@]}" \
    --epochs 100 \
    --batch-size 32 \
    --num-workers 4 \
    --learning-rate 3e-4 \
    --weight-decay 1e-4 \
    --grad-clip 1.0 \
    --temperature 0.07 \
    --hard-negative-bias 0.5 \
    --mask-probability 0.1 \
    --jitter-std 0.01 \
    --device "${TEACHER_DEVICE}" \
    --export-format both \
    --wandb-mode online \
    --wandb-project "${WANDB_TEACHER_PROJECT:-fitwam-offline-steer-teacher}" \
    --wandb-name "${WANDB_TEACHER_NAME:-water_plant_offline_steer_teacher_v1}" \
    "${WANDB_ENTITY_ARGS[@]}"
fi

if [[ ! -f "${MANIFEST_B0_RAW}" ]]; then
  python scripts/everobot/build_eve_sidecar.py build-manifest \
    --eve-root "${EVE_ROOT}" \
    --manifest-name offline_b0_success_budget_control_raw \
    --include-outcomes success \
    --success-dataset-ids "${BASE_DATASET_ID}" \
    --success-auxiliary-dataset-ids "${ROLLOUT_DATASET_ID}" \
    --success-sample-mode event_only \
    --event-types interaction_candidate \
    --splits train
fi

if [[ ! -f "${MANIFEST_B1}" ]]; then
  python scripts/everobot/build_eve_sidecar.py build-manifest \
    --eve-root "${EVE_ROOT}" \
    --manifest-name offline_b1_failure_video_control \
    --include-outcomes success failure \
    --success-dataset-ids "${BASE_DATASET_ID}" \
    --failure-dataset-ids "${ROLLOUT_DATASET_ID}" \
    --success-sample-mode episode_only \
    --failure-sample-mode event_only \
    --event-types interaction_candidate \
    --failure-action-loss disabled \
    --splits train
fi

if [[ ! -f "${MANIFEST_B0}" ]]; then
  python scripts/everobot/match_auxiliary_manifest_budget.py \
    --control-manifest "${MANIFEST_B0_RAW}" \
    --reference-manifest "${MANIFEST_B1}" \
    --eve-root "${EVE_ROOT}" \
    --output "${MANIFEST_B0}" \
    --diagnostics-output "${MANIFEST_B0_MATCH_DIAGNOSTICS}" \
    --seed "${SPLIT_SEED}"
fi

if [[ ! -f "${MANIFEST_SELECTION}" ]]; then
  python scripts/everobot/build_eve_sidecar.py build-manifest \
    --eve-root "${EVE_ROOT}" \
    --manifest-name offline_selection_primary_success \
    --include-outcomes success \
    --success-dataset-ids "${BASE_DATASET_ID}" \
    --success-sample-mode episode_only \
    --splits val
fi

if [[ "${SKIP_TEACHER}" != "1" && ! -f "${MANIFEST_M}" ]]; then
  python scripts/everobot/attach_event_pairs_to_manifest.py \
    --manifest "${MANIFEST_B1}" \
    --pair-ledger "${PAIR_LEDGER}" \
    --pair-targets "${PAIR_TARGETS}" \
    --output "${MANIFEST_M}" \
    --attach-side failure
fi

if [[ "${SKIP_TEACHER}" != "1" ]]; then
  PAIR_TARGETS_TEACHER_SHA256="$(
    python - "${PAIR_TARGETS}" <<'PY'
from pathlib import Path
import sys

from fastwam.datasets.eve.pair_targets import PairTargetStore

with PairTargetStore(Path(sys.argv[1])) as store:
    print(store.teacher_sha256)
PY
  )"
  export INIT_WEIGHTS="${SOURCE_CHECKPOINT}"
  export FASTWAM_SOURCE_CONFIG="${SOURCE_CONFIG}"
  export B0_MANIFEST_PATH="${MANIFEST_B0}"
  export B1_MANIFEST_PATH="${MANIFEST_B1}"
  export C_MANIFEST_PATH="${MANIFEST_B1}"
  export M_MANIFEST_PATH="${MANIFEST_M}"
  export EVE_VAL_MANIFEST_PATH="${MANIFEST_SELECTION}"
  export PAIR_TARGETS_PATH="${PAIR_TARGETS}"
  export PAIR_LEDGER_PATH="${PAIR_LEDGER}"
  export PAIR_CALIBRATION_PATH="${PAIR_CALIBRATION}"
  export PAIR_DIAGNOSTICS_PATH="${PAIR_DIAGNOSTICS}"
  export PAIR_QUALITY_REPORT_PATH="${PAIR_QUALITY_REPORT}"
  export STATE_LINE_AUDIT_INDEX_PATH="${STATE_LINE_AUDIT_INDEX}"
  export B0_AUXILIARY_MATCH_DIAGNOSTICS_PATH="${MANIFEST_B0_MATCH_DIAGNOSTICS}"
  export TEACHER_CHECKPOINT="${TEACHER_OUTPUT}/best_teacher.pt"
  export PAIR_TARGETS_TEACHER_SHA256
  export PROTOCOL_BUNDLE_PATH="${EVE_ROOT}/protocol/offline_v1.json"
  EXECUTION_ENV="${EVE_ROOT}/protocol/offline_v1.env"
  mkdir -p "$(dirname "${EXECUTION_ENV}")"
  EXECUTION_ENV_TMP="${EXECUTION_ENV}.tmp"
  {
    printf '# Generated by prepare_offline_self_improving.sh; do not edit.\n'
    for variable in \
      FITWAM_ENV_PREFIX \
      BASE_DATASET \
      ROLLOUT_RAW \
      INIT_WEIGHTS \
      FASTWAM_SOURCE_CONFIG \
      SOURCE_BUNDLE_MANIFEST \
      FORMAL_VALIDATION_REPORT \
      B0_MANIFEST_PATH \
      B1_MANIFEST_PATH \
      C_MANIFEST_PATH \
      M_MANIFEST_PATH \
      EVE_VAL_MANIFEST_PATH \
      PAIR_LEDGER_PATH \
      PAIR_CALIBRATION_PATH \
      PAIR_DIAGNOSTICS_PATH \
      PAIR_QUALITY_REPORT_PATH \
      PAIR_QUALITY_GATE_MODE \
      PAIR_QUALITY_GATE_STATUS \
      STATE_LINE_AUDIT_INDEX_PATH \
      B0_AUXILIARY_MATCH_DIAGNOSTICS_PATH \
      PAIR_TARGETS_PATH \
      TEACHER_CHECKPOINT \
      PAIR_TARGETS_TEACHER_SHA256 \
      PROTOCOL_BUNDLE_PATH \
      NORM_STATS_SOURCE \
      NORM_STATS_META_DIR \
      PRETRAINED_NORM_STATS \
      NORM_STATS_BUNDLE_SHA256 \
      TEXT_EMBEDDING_CACHE_DIR; do
      if [[ -n "${!variable:-}" ]]; then
        printf 'export %s=%q\n' "${variable}" "${!variable}"
      fi
    done
  } > "${EXECUTION_ENV_TMP}"
  chmod 0640 "${EXECUTION_ENV_TMP}"
  if [[ -e "${EXECUTION_ENV}" ]]; then
    if ! cmp -s "${EXECUTION_ENV_TMP}" "${EXECUTION_ENV}"; then
      echo \
        "Existing frozen execution environment differs: ${EXECUTION_ENV}. " \
        "Use a new EVE_ROOT for changed inputs." \
        >&2
      rm -f "${EXECUTION_ENV_TMP}"
      exit 2
    fi
    rm -f "${EXECUTION_ENV_TMP}"
  else
    mv "${EXECUTION_ENV_TMP}" "${EXECUTION_ENV}"
  fi
  PREFLIGHT_EXECUTION_MODE=formal
  if [[ "${PAIR_QUALITY_GATE_STATUS}" == "failed" ]]; then
    PREFLIGHT_EXECUTION_MODE=smoke20
  fi
  CUDA_VISIBLE_DEVICES="${PREFLIGHT_GPUS}" \
    FITWAM_PREFORMAL_MODE="${PREFLIGHT_EXECUTION_MODE}" \
    FITWAM_PREFLIGHT_ONLY=1 \
    bash scripts/water_plant/train_offline_self_improving.sh B0
fi

echo "[offline-prepare] eve_root=${EVE_ROOT}"
echo "[offline-prepare] B0=${MANIFEST_B0}"
echo "[offline-prepare] B1/C=${MANIFEST_B1}"
echo "[offline-prepare] selection=${MANIFEST_SELECTION}"
if [[ "${SKIP_TEACHER}" != "1" ]]; then
  echo "[offline-prepare] M=${MANIFEST_M}"
  echo "[offline-prepare] pair_targets=${PAIR_TARGETS}"
  echo "[offline-prepare] execution_env=${EXECUTION_ENV}"
fi
