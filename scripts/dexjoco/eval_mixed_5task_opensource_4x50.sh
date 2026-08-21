#!/usr/bin/env bash
# Official 4×50 eval of mixed_5task step_055000 with mixed (5-task pooled) z-score stats.
#   GPUS=4,5,6,7 bash scripts/dexjoco/eval_mixed_5task_opensource_4x50.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/dexjoco/lib.sh"
cd "${ROOT}"

dexjoco_require_gpus
dexjoco_opensource_setup
dexjoco_activate_fastwam
dexjoco_export_pythonpath
dexjoco_assert_pins

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/evaluate_results/dexjoco/opensource_mixed_5task_mixedstats_4x50_${STAMP}}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/artifacts/opensource_ckpt_links/mixed_5task}"
STATS="${STATS:-${ROOT}/artifacts/mixed_5task/dataset_stats.json}"
export FASTWAM_DEXJOCO_TASKS_YAML="${FASTWAM_DEXJOCO_TASKS_YAML:-${ROOT}/configs/eval/dexjoco/opensource_baseline_tasks.yaml}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/orchestrator.log"
log() { echo "[mixed-5task-mixedstats $(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

log "OUT_ROOT=${OUT_ROOT} GPUS=${GPUS} STATS=${STATS}"
[[ -f "${STATS}" ]] || { log "ERROR missing mixed stats ${STATS}"; exit 1; }
[[ -e "${CKPT_DIR}/step_055000.pt" ]] || { log "ERROR missing mixed ckpt"; exit 1; }

if [[ "${WAIT_IDLE:-0}" == "1" ]]; then
  dexjoco_wait_gpus_idle "${MASTER_LOG}"
fi

JOBS=(
  "mixed_fold_glasses|fold_glasses"
  "mixed_hammer_nail|hammer_nail"
  "mixed_pick_bucket|pick_bucket"
  "mixed_pinch_tongs|pinch_tongs"
  "mixed_water_plant|water_plant"
)

fail=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r jid task <<<"${job}"
  log "START ${jid}"
  if ! TASK="${task}" \
    CKPT_DIR="${CKPT_DIR}" \
    CHECKPOINT_STEPS=55000 \
    DATASET_STATS="${STATS}" \
    GPUS="${GPUS}" \
    WAIT_IDLE=0 \
    OUT_ROOT="${OUT_ROOT}/${jid}" \
    bash "${ROOT}/scripts/dexjoco/eval_opensource_4x50.sh"
  then
    log "FAIL ${jid}"
    fail=1
  fi
done

"${ENV_PREFIX}/bin/python" "${ROOT}/scripts/dexjoco/aggregate_opensource_baseline_results.py" \
  --out-root "${OUT_ROOT}" \
  --master-log "${MASTER_LOG}"

log "ALL complete fail=${fail} out=${OUT_ROOT}"
exit "${fail}"
