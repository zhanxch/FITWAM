#!/usr/bin/env bash
# Wait for free GPUs, collect opensource S0 rollouts, then DEWO v2 pair prepare.
# Does not start train unless RUN_TRAIN=1.
#
#   TASK=water_plant GPUS=4,5,6,7 MAX_USED_MIB=1500 RUN_TRAIN=0 \
#     bash scripts/dewo_v2/wait_collect_then_prepare.sh
#
# CFG overrides (optional): CFG_PRIMARY / CFG_AUX_SUCCESS / CFG_AUX_FAIL /
# CFG_SUCCESS_SUFFIX / CFG_FAILURE_SUFFIX / CFG_DROPOUT / CFG_FAST_FAIL_CLOSED
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_activate_fastwam
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${ROOT_DIR}/checkpoints"

dewo_v2_require_task
dewo_v2_require_gpus
dewo_v2_load_task "${TASK}"

# 1500 blocks hammer VAE (~2.7GB) and other-user jobs; 3000 would leak onto VAE.
MAX_USED_MIB="${MAX_USED_MIB:-1500}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
COLLECT_OUT="${COLLECT_OUT:-${ROOT_DIR}/data/${TASK}_opensource_s0_collect_4x50_${STAMP}}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/data/${TASK}_dewo_v2_pair_${STAMP}}"
RUN_TRAIN="${RUN_TRAIN:-0}"
LOG_DIR="${COLLECT_OUT}/logs"
mkdir -p "${LOG_DIR}"
MASTER="${LOG_DIR}/orchestrator_${STAMP}.log"

log() { echo "[${TASK}-collect-prepare $(date -Is)] $*" | tee -a "${MASTER}"; }

log "task=${TASK} wait until each GPU used<=${MAX_USED_MIB}MiB (gpus=${GPUS})"
log "CFG_PRIMARY=${CFG_PRIMARY} AUX_SUCCESS=${CFG_AUX_SUCCESS} AUX_FAIL=${CFG_AUX_FAIL}"
if [[ "${WAIT_IDLE:-1}" == "1" ]]; then
  MAX_UTIL="${MAX_UTIL:-0}" dewo_v2_wait_gpus_idle "${MASTER}" "${TASK}-collect-prepare"
fi

log "LAUNCH collect -> ${COLLECT_OUT}"
OVERWRITE=1 TASK="${TASK}" GPUS="${GPUS}" OUTPUT_DIR="${COLLECT_OUT}" \
  bash scripts/dewo_v2/collect_opensource_4x50.sh \
  2>&1 | tee -a "${LOG_DIR}/collect_${STAMP}.log"

RAW="${COLLECT_OUT}/rollout_raw_200"
[[ -d "${RAW}/meta" ]] || { log "ERROR missing ${RAW}"; exit 2; }

log "LAUNCH pair prepare pipeline -> ${EXP_ROOT}"
TASK="${TASK}" SOURCE_ROOT="${COLLECT_OUT}" EXP_ROOT="${EXP_ROOT}" GPUS="${GPUS}" \
  PRIMARY_KIND="${PRIMARY_KIND:-expert}" \
  PRIMARY_N="${PRIMARY_N:-15}" \
  PRIMARY_DATASET="${PRIMARY_DATASET:-${RAW}}" \
  ROLLOUT_RAW="${RAW}" \
  RUN_TRAIN=0 \
  bash scripts/dewo_v2/run_pair_pipeline.sh \
  2>&1 | tee -a "${LOG_DIR}/prepare_pipeline_${STAMP}.log"

ENV_FILE="${EXP_ROOT}/eve_v02/protocol/offline_v1_b1_jump_fast.env"
log "DONE collect=${COLLECT_OUT} exp=${EXP_ROOT}"
log "env=${ENV_FILE}"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  log "LAUNCH train INIT=${INIT:-scratch} USE_VAE_LATENT_CACHE=${USE_VAE_LATENT_CACHE:-1}"
  ENV_FILE="${ENV_FILE}" \
  INIT="${INIT:-scratch}" \
  RUN_INLINE=1 \
    bash scripts/dewo_v2/train.sh \
    2>&1 | tee -a "${LOG_DIR}/train_${STAMP}.log"
fi
