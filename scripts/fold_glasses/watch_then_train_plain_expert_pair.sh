#!/usr/bin/env bash
# Wait for a train tmux session, then launch plain offline ablation.
# Required: WAIT_TMUX, WAIT_RUN_DIR, WAIT_FINAL_CKPT, ENV_FILE, GPUS
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

WAIT_TMUX="${WAIT_TMUX:?Set WAIT_TMUX}"
WAIT_RUN_DIR="${WAIT_RUN_DIR:?Set WAIT_RUN_DIR}"
WAIT_FINAL_CKPT="${WAIT_FINAL_CKPT:?Set WAIT_FINAL_CKPT}"
ENV_FILE="${ENV_FILE:?Set ENV_FILE}"
if [[ -z "${GPUS:-}" ]]; then
  echo "[plain-watcher] ERROR: set GPUS" >&2
  exit 2
fi
POLL_SEC="${POLL_SEC:-60}"
WATCHER_TMUX="${WATCHER_TMUX:-fold_plain_offline_watcher}"
TRAIN_TMUX="${TRAIN_TMUX:-fold_plain_expert_pair_full_1e-4}"
LOG_DIR="${LOG_DIR:-${WAIT_RUN_DIR}/logs}"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/watch_then_train_plain_expert_pair.log"

log() { echo "[plain-watcher $(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

if tmux has-session -t "${WATCHER_TMUX}" 2>/dev/null; then
  echo "[plain-watcher] tmux '${WATCHER_TMUX}' already exists; attach or kill it first."
  exit 1
fi

WORKER="${LOG_DIR}/watch_then_train_plain_expert_pair_worker.sh"
cat > "${WORKER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT_DIR}"
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FITWAM_ENV:-fastwam}"

log() { echo "[plain-watcher \$(date -Is)] \$*" | tee -a "${MASTER_LOG}"; }

log "waiting for tmux=${WAIT_TMUX} to exit"
log "require final ckpt=${WAIT_FINAL_CKPT}"

while tmux has-session -t "${WAIT_TMUX}" 2>/dev/null; do
  sleep "${POLL_SEC}"
done

log "tmux ${WAIT_TMUX} gone"
if [[ ! -f "${WAIT_FINAL_CKPT}" ]]; then
  log "ERROR: expected ${WAIT_FINAL_CKPT} missing — not launching plain train"
  exit 2
fi

sleep 30
log "launching plain expert+pair offline ablation"
export GPUS="${GPUS}"
export TMUX_SESSION="${TRAIN_TMUX}"
export ENV_FILE="${ENV_FILE}"
bash "${ROOT_DIR}/scripts/fold_glasses/train_plain_expert_pair_full_1e-4.sh" \\
  2>&1 | tee -a "${MASTER_LOG}"
log "launch returned EXIT=\$?"
EOF
chmod +x "${WORKER}"

tmux new-session -d -s "${WATCHER_TMUX}" "bash '${WORKER}'"
log "started watcher tmux=${WATCHER_TMUX}"
log "attach: tmux attach -t ${WATCHER_TMUX}"
