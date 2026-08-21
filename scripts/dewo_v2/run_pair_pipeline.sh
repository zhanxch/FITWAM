#!/usr/bin/env bash
# End-to-end DEWO v2 recoverability-pair recipe for any registered task.
# Scan -> materialize pair LeRobot -> Eve/text/FAST (prepare).
# Train is optional via RUN_TRAIN=1.
#
#   TASK=water_plant SOURCE_ROOT=... GPUS=4,5,6,7 bash scripts/dewo_v2/run_pair_pipeline.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_require_gpus
dewo_v2_load_task "${TASK}"

OPEN_REPO="${OPEN_REPO:-${ROOT_DIR}/../FastWAM-infer-in-DexJoco}"
OPEN_REPO="$(cd "${OPEN_REPO}" && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT to the opensource collect root}"
ROLLOUT_RAW="${ROLLOUT_RAW:-${SOURCE_ROOT}/rollout_raw_200}"
SCAN_ROOT="${SCAN_ROOT:-${SOURCE_ROOT}/recoverability_pairs_v2}"
PAIR_DATASET="${PAIR_DATASET:-${SCAN_ROOT}/pair_lerobot}"
PIPELINE_STAMP="${PIPELINE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/data/${TASK}_dewo_v2_pair_${PIPELINE_STAMP}}"
RUN_TRAIN="${RUN_TRAIN:-0}"

mkdir -p "${SCAN_ROOT}/logs" "${EXP_ROOT}/logs"
MASTER_LOG="${EXP_ROOT}/logs/pipeline.log"
log() { echo "[dewo-v2-pair ${TASK} $(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

[[ -d "${ROLLOUT_RAW}/meta" ]] || { log "ERROR missing ${ROLLOUT_RAW}"; exit 2; }

if [[ "${SKIP_SCAN:-0}" == "1" ]]; then
  if [[ ! -f "${SCAN_ROOT}/summary.json" ]]; then
    log "ERROR: SKIP_SCAN=1 but ${SCAN_ROOT}/summary.json is missing"
    exit 2
  fi
  log "reuse existing scan ${SCAN_ROOT}"
elif [[ ! -f "${SCAN_ROOT}/summary.json" ]]; then
  log "scan recoverability pairs gpus=${GPUS} max_steps=${MAX_STEPS}"
  python scripts/fold_glasses/run_recoverability_pair_scan.py \
    --gpus "${GPUS}" \
    --dataset "${ROLLOUT_RAW}" \
    --output "${SCAN_ROOT}" \
    --checkpoint "${CKPT}" \
    --model-config "${OPEN_REPO}/configs/fastwam_dexjoco.yaml" \
    --dataset-stats "${STATS}" \
    --text-embedding "${TEXT_EMB}" \
    --task-name "${TASK}" \
    --max-steps "${MAX_STEPS}" \
    2>&1 | tee -a "${MASTER_LOG}"
else
  log "reuse existing scan ${SCAN_ROOT}"
fi

if [[ ! -f "${PAIR_DATASET}/pair_index.json" ]]; then
  log "materialize pair LeRobot dataset"
  python scripts/fold_glasses/materialize_recoverability_pairs_lerobot.py \
    --scan-root "${SCAN_ROOT}" \
    --source-dataset "${ROLLOUT_RAW}" \
    --output-dataset "${PAIR_DATASET}" \
    --success-prompt "${SUCCESS_PROMPT}" \
    --overwrite \
    2>&1 | tee -a "${MASTER_LOG}"
fi

log "prepare Eve + text/FAST caches kind=${PRIMARY_KIND:-expert}"
TASK="${TASK}" PAIR_DATASET="${PAIR_DATASET}" EXP_ROOT="${EXP_ROOT}" \
  TEXT_CACHE="${EXP_ROOT}/text_embeds_cache" \
  FAST_PRECOMPUTE_GPUS="${GPUS}" \
  GPUS="${GPUS}" \
  PRIMARY_KIND="${PRIMARY_KIND:-expert}" \
  PRIMARY_N="${PRIMARY_N:-15}" \
  PRIMARY_DATASET="${PRIMARY_DATASET:-${ROLLOUT_RAW}}" \
  ROLLOUT_RAW="${ROLLOUT_RAW}" \
  bash scripts/dewo_v2/prepare_pair_eve.sh \
  2>&1 | tee -a "${MASTER_LOG}"

export ENV_FILE="${EXP_ROOT}/eve_v02/protocol/offline_v1_b1_jump_fast.env"
log "prepare DONE env=${ENV_FILE}"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
  log "train INIT=${INIT:-scratch} VAE_CACHE=${USE_VAE_LATENT_CACHE:-0}"
  RUN_INLINE=1 GPUS="${GPUS}" \
    bash scripts/dewo_v2/train.sh \
    2>&1 | tee -a "${MASTER_LOG}"
fi
