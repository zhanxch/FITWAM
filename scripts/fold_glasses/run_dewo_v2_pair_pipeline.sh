#!/usr/bin/env bash
# End-to-end fold_glasses DEWO v2 recoverability-pair recipe.
# Scan -> materialize pair LeRobot -> Eve/text/FAST/VAE -> train.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

SOURCE_ROOT="${SOURCE_ROOT:-${ROOT_DIR}/data/fold_glasses_opensource_s0_collect_4x50_20260812_112113}"
ROLLOUT_RAW="${ROLLOUT_RAW:-${SOURCE_ROOT}/rollout_raw_200}"
SCAN_ROOT="${SCAN_ROOT:-${SOURCE_ROOT}/recoverability_pairs_v2}"
PAIR_DATASET="${PAIR_DATASET:-${SCAN_ROOT}/pair_lerobot}"
PIPELINE_STAMP="${PIPELINE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/data/fold_glasses_dewo_v2_pair_${PIPELINE_STAMP}}"
GPUS="${GPUS:-0,1,2,3}"

mkdir -p "${SCAN_ROOT}/logs" "${EXP_ROOT}/logs"
MASTER_LOG="${EXP_ROOT}/logs/pipeline.log"
log() { echo "[dewo-v2-pair $(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

if [[ "${SKIP_SCAN:-0}" == "1" ]]; then
  if [[ ! -f "${SCAN_ROOT}/summary.json" ]]; then
    log "ERROR: SKIP_SCAN=1 but ${SCAN_ROOT}/summary.json is missing"
    exit 2
  fi
  log "reuse existing scan ${SCAN_ROOT}"
elif [[ ! -f "${SCAN_ROOT}/summary.json" ]]; then
  log "scan recoverability pairs gpus=${GPUS}"
  python scripts/fold_glasses/run_recoverability_pair_scan.py \
    --gpus "${GPUS}" \
    --dataset "${ROLLOUT_RAW}" \
    --output "${SCAN_ROOT}" \
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
    --overwrite \
    2>&1 | tee -a "${MASTER_LOG}"
fi

log "prepare Eve + text/FAST caches"
PAIR_DATASET="${PAIR_DATASET}" EXP_ROOT="${EXP_ROOT}" \
  TEXT_CACHE="${EXP_ROOT}/text_embeds_cache" \
  FAST_PRECOMPUTE_GPUS="${GPUS}" \
  bash scripts/fold_glasses/prepare_dewo_v2_pair_eve.sh \
  2>&1 | tee -a "${MASTER_LOG}"

ENV_FILE="${EXP_ROOT}/eve_v02/protocol/offline_v1_b1_jump_fast.env"
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

log "VAE pre-encode then train (train-only VAE, eval_every=0)"
VAE_ENCODE_VAL=false DEWO_HYDRA_OVERRIDES="eval_every=0" \
FILL_VAE_LATENT_CACHE=0 SKIP_VAE_PREENCODE=0 RUN_INLINE=1 GPUS="${GPUS}" \
  bash scripts/fold_glasses/train_dewo_v2_jump_fast_lora.sh \
  2>&1 | tee -a "${MASTER_LOG}"
