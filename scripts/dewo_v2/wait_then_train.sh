#!/usr/bin/env bash
# Wait until ENV_FILE is ready and GPUs are idle, then launch TRAIN_LAUNCHER.
#   TASK=hammer_nail ENV_FILE=... GPUS=0,1,2,3 \
#     TRAIN_LAUNCHER=scripts/dewo_v2/train_jump_fast_lora.sh \
#     bash scripts/dewo_v2/wait_then_train.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_require_gpus

ENV_FILE="${ENV_FILE:?Set ENV_FILE to offline_v1_b1_jump_fast.env}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-scripts/dewo_v2/train_jump_fast_lora.sh}"
PREP_TMUX="${PREP_TMUX:-}"
MAX_USED_MIB="${MAX_USED_MIB:-3000}"
POLL_SEC="${POLL_SEC:-60}"
TMUX_SESSION="${TMUX_SESSION:-${TASK}_dewo_v2_train}"
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

log "wait until GPUs ${GPUS} used<=${MAX_USED_MIB}MiB"
MAX_UTIL="${MAX_UTIL:-0}" WAIT_POLL_SEC="${POLL_SEC}" dewo_v2_wait_gpus_idle "${MASTER}" "wait-train"

log "source ${ENV_FILE}"
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

log "LAUNCH ${TRAIN_LAUNCHER} on GPUS=${GPUS}"
ENV_FILE="${ENV_FILE}" \
TASK="${TASK}" \
GPUS="${GPUS}" \
TMUX_SESSION="${TMUX_SESSION}" \
  bash "${TRAIN_LAUNCHER}" \
  2>&1 | tee -a "${MASTER}"

log "train launcher returned; attach: tmux attach -t ${TMUX_SESSION}"
