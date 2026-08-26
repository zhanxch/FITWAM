#!/usr/bin/env bash
# Wait until ENV_FILE is ready and GPUs are idle, then launch TRAIN_LAUNCHER.
#   TASK=hammer_nail ENV_FILE=... GPUS=0,1,2,3 \
#     TRAIN_LAUNCHER=scripts/dewo_v2/train.sh \
#     bash scripts/dewo_v2/wait_then_train.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_require_gpus

ENV_FILE="${ENV_FILE:?Set ENV_FILE to offline_v1_b1_jump_fast.env}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-scripts/dewo_v2/train.sh}"
PREP_TMUX="${PREP_TMUX:-}"
MAX_USED_MIB="${MAX_USED_MIB:-3000}"
POLL_SEC="${POLL_SEC:-60}"
TMUX_SESSION="${TMUX_SESSION:-${TASK}_dewo_v2_train}"
# Preserve caller knobs; prepare env overwrites DEWO_TASK / DEWO_OUTPUT_DIR.
CALLER_INIT="${INIT:-}"
CALLER_DEWO_VERSION="${DEWO_VERSION:-}"
CALLER_OUTPUT_DIR="${DEWO_OUTPUT_DIR:-}"
CALLER_LR="${LR:-}"
CALLER_TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-}"
CALLER_USE_VAE="${USE_VAE_LATENT_CACHE:-}"
CALLER_SKIP_VAE="${SKIP_VAE_PREENCODE:-}"
CALLER_PRIMARY_PER_BATCH="${PRIMARY_PER_BATCH:-}"
CALLER_PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS:-}"
CALLER_CFG_PRIMARY="${CFG_PRIMARY:-}"
CALLER_CFG_AUX_SUCCESS="${CFG_AUX_SUCCESS:-}"
CALLER_CFG_AUX_FAIL="${CFG_AUX_FAIL:-}"
CALLER_HYDRA="${DEWO_HYDRA_OVERRIDES:-}"
CALLER_RUN_INLINE="${RUN_INLINE:-}"
CALLER_WAIT_IDLE="${WAIT_IDLE:-}"
LOG_DIR="${LOG_DIR:-$(dirname "${ENV_FILE}")/../logs}"
mkdir -p "${LOG_DIR}"
MASTER="${LOG_DIR}/wait_then_train_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[wait-train $(date -Is)] $*" | tee -a "${MASTER}"; }

log "wait for prepare env: ${ENV_FILE}"
while true; do
  if [[ -f "${ENV_FILE}" ]] && grep -q 'TEXT_EMBEDDING_CACHE_SHA256=' "${ENV_FILE}"; then
    log "prepare ready (env+text sha present)"
    break
  fi
  if [[ -n "${PREP_TMUX}" ]] && ! tmux has-session -t "${PREP_TMUX}" 2>/dev/null; then
    if [[ ! -f "${ENV_FILE}" ]]; then
      log "ERROR: prep tmux gone and env missing: ${ENV_FILE}"
      exit 2
    fi
    log "prep tmux gone; waiting for TEXT_EMBEDDING_CACHE_SHA256 in env"
  fi
  sleep "${POLL_SEC}"
done

if [[ "${WAIT_IDLE:-1}" == "1" ]]; then
  log "wait until GPUs ${GPUS} used<=${MAX_USED_MIB}MiB"
  MAX_UTIL="${MAX_UTIL:-0}" WAIT_POLL_SEC="${POLL_SEC}" dewo_v2_wait_gpus_idle "${MASTER}" "wait-train"
else
  log "WAIT_IDLE=0; skipping GPU idle wait"
fi

log "source ${ENV_FILE}"
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
[[ -z "${CALLER_INIT}" ]] || export INIT="${CALLER_INIT}"
[[ -z "${CALLER_DEWO_VERSION}" ]] || export DEWO_VERSION="${CALLER_DEWO_VERSION}"
[[ -z "${CALLER_OUTPUT_DIR}" ]] || export DEWO_OUTPUT_DIR="${CALLER_OUTPUT_DIR}"
[[ -z "${CALLER_LR}" ]] || export LR="${CALLER_LR}"
[[ -z "${CALLER_TRAIN_MAX_STEPS}" ]] || export TRAIN_MAX_STEPS="${CALLER_TRAIN_MAX_STEPS}"
[[ -z "${CALLER_USE_VAE}" ]] || export USE_VAE_LATENT_CACHE="${CALLER_USE_VAE}"
[[ -z "${CALLER_SKIP_VAE}" ]] || export SKIP_VAE_PREENCODE="${CALLER_SKIP_VAE}"
[[ -z "${CALLER_PRIMARY_PER_BATCH}" ]] || export PRIMARY_PER_BATCH="${CALLER_PRIMARY_PER_BATCH}"
[[ -z "${CALLER_PRETRAINED_NORM_STATS}" ]] || export PRETRAINED_NORM_STATS="${CALLER_PRETRAINED_NORM_STATS}"
[[ -z "${CALLER_CFG_PRIMARY}" ]] || export CFG_PRIMARY="${CALLER_CFG_PRIMARY}"
[[ -z "${CALLER_CFG_AUX_SUCCESS}" ]] || export CFG_AUX_SUCCESS="${CALLER_CFG_AUX_SUCCESS}"
[[ -z "${CALLER_CFG_AUX_FAIL}" ]] || export CFG_AUX_FAIL="${CALLER_CFG_AUX_FAIL}"
[[ -z "${CALLER_HYDRA}" ]] || export DEWO_HYDRA_OVERRIDES="${CALLER_HYDRA}"
[[ -z "${CALLER_RUN_INLINE}" ]] || export RUN_INLINE="${CALLER_RUN_INLINE}"
[[ -z "${CALLER_WAIT_IDLE}" ]] || export WAIT_IDLE="${CALLER_WAIT_IDLE}"
# ENV_FILE always exports v2 CFG triples. If the caller did not set them,
# drop those so train.sh can apply the selected v2/v5/v6 defaults.
if [[ -z "${CALLER_CFG_PRIMARY}" ]]; then
  unset CFG_PRIMARY CFG_PRIMARY_OUTCOME CFG_PRIMARY_FAST CFG_PRIMARY_BASE || true
fi
if [[ -z "${CALLER_CFG_AUX_SUCCESS}" ]]; then
  unset CFG_AUX_SUCCESS CFG_AUX_SUCCESS_OUTCOME CFG_AUX_SUCCESS_FAST CFG_AUX_SUCCESS_BASE || true
fi
if [[ -z "${CALLER_CFG_AUX_FAIL}" ]]; then
  unset CFG_AUX_FAIL CFG_AUX_FAIL_OUTCOME CFG_AUX_FAIL_FAST CFG_AUX_FAIL_BASE || true
fi

log "LAUNCH ${TRAIN_LAUNCHER} on GPUS=${GPUS}"
ENV_FILE="${ENV_FILE}" \
TASK="${TASK}" \
GPUS="${GPUS}" \
TMUX_SESSION="${TMUX_SESSION}" \
  bash "${TRAIN_LAUNCHER}" \
  2>&1 | tee -a "${MASTER}"

log "train launcher returned; attach: tmux attach -t ${TMUX_SESSION}"
