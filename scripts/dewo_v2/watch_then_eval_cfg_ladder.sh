#!/usr/bin/env bash
# After a train tmux session exits, run the CFG ckpt ladder.
#   TASK=fold_glasses WAIT_TMUX=... WAIT_RUN_DIR=... WAIT_FINAL_CKPT=... \
#     TEXT_EMBEDDING_CACHE_DIR=... GPUS=4,5,6,7 \
#     bash scripts/dewo_v2/watch_then_eval_cfg_ladder.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_require_gpus

WAIT_TMUX="${WAIT_TMUX:?Set WAIT_TMUX to the training tmux session name}"
WAIT_RUN_DIR="${WAIT_RUN_DIR:?Set WAIT_RUN_DIR to the training run directory}"
WAIT_FINAL_CKPT="${WAIT_FINAL_CKPT:?Set WAIT_FINAL_CKPT to the last expected weight}"
TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:?Set TEXT_EMBEDDING_CACHE_DIR}"
POLL_SEC="${POLL_SEC:-60}"
WAIT_IDLE="${WAIT_IDLE:-0}"
WATCHER_TMUX="${WATCHER_TMUX:-${TASK}_cfg_eval_watcher}"
EVAL_TMUX="${EVAL_TMUX:-${TASK}_cfg_ladder}"
LOG_DIR="${LOG_DIR:-${WAIT_RUN_DIR}/logs}"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/watch_then_eval_cfg_ladder.log"

log() { echo "[cfg-eval-watcher $(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

if tmux has-session -t "${WATCHER_TMUX}" 2>/dev/null; then
  echo "[cfg-eval-watcher] tmux '${WATCHER_TMUX}' already exists; attach or kill it first."
  exit 1
fi

WORKER="${LOG_DIR}/watch_then_eval_cfg_ladder_worker.sh"
cat > "${WORKER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT_DIR}"
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FITWAM_ENV:-fastwam}"

log() { echo "[cfg-eval-watcher \$(date -Is)] \$*" | tee -a "${MASTER_LOG}"; }

log "waiting for tmux=${WAIT_TMUX} to exit"
log "require final ckpt=${WAIT_FINAL_CKPT}"
log "then CFG ladder on GPUS=${GPUS} (WAIT_IDLE=${WAIT_IDLE})"

while tmux has-session -t "${WAIT_TMUX}" 2>/dev/null; do
  sleep "${POLL_SEC}"
done

log "tmux ${WAIT_TMUX} gone"
if [[ ! -f "${WAIT_FINAL_CKPT}" ]]; then
  log "ERROR: expected ${WAIT_FINAL_CKPT} missing — not launching eval ladder"
  exit 2
fi

sleep 15
if tmux has-session -t "${EVAL_TMUX}" 2>/dev/null; then
  log "ERROR eval tmux ${EVAL_TMUX} already exists"
  exit 1
fi

log "launching CFG ladder in tmux ${EVAL_TMUX}"
tmux new-session -d -s "${EVAL_TMUX}" "\\
  cd '${ROOT_DIR}' && \\
  TASK='${TASK}' \\
  RUN_DIR='${WAIT_RUN_DIR}' \\
  GPUS='${GPUS}' \\
  WAIT_IDLE='${WAIT_IDLE}' \\
  CFG_SCALE='${CFG_SCALE:-2.0}' \\
  TEXT_EMBEDDING_CACHE_DIR='${TEXT_EMBEDDING_CACHE_DIR}' \\
  bash '${ROOT_DIR}/scripts/dewo_v2/eval_cfg_ckpt_ladder.sh' \\
  2>&1 | tee -a '${MASTER_LOG}'"
log "eval ladder tmux started: ${EVAL_TMUX}"
log "attach: tmux attach -t ${EVAL_TMUX}"
EOF
chmod +x "${WORKER}"

tmux new-session -d -s "${WATCHER_TMUX}" "bash '${WORKER}'"
log "started watcher tmux=${WATCHER_TMUX}"
log "log=${MASTER_LOG}"
log "attach: tmux attach -t ${WATCHER_TMUX}"
