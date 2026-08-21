#!/usr/bin/env bash
# Official 4×50 eval for all checkpoints/dexjoco baselines on the OPEN stack.
#   GPUS=4,5,6,7 bash scripts/dexjoco/eval_all_opensource_baselines_4x50.sh
#
# Session knobs: GPUS (required), WAIT_IDLE (default 1), OUT_ROOT, STAMP
# Protocol is pinned in scripts/dexjoco/eval_opensource_4x50.sh
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
OUT_ROOT="${OUT_ROOT:-${ROOT}/evaluate_results/dexjoco/opensource_all_baselines_4x50_${STAMP}}"
LINK_ROOT="${LINK_ROOT:-${ROOT}/artifacts/opensource_ckpt_links}"
TASKS_YAML="${TASKS_YAML:-${ROOT}/configs/eval/dexjoco/opensource_baseline_tasks.yaml}"
export FASTWAM_DEXJOCO_TASKS_YAML="${TASKS_YAML}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/orchestrator.log"
log() { echo "[opensource-baselines $(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

log "OUT_ROOT=${OUT_ROOT} GPUS=${GPUS}"
[[ -f "${TASKS_YAML}" ]] || { log "ERROR missing ${TASKS_YAML}"; exit 1; }

# job_id|task|step|ckpt_dir_name
JOBS=(
  "single_fold_glasses|fold_glasses|10000|fold_glasses"
  "single_hammer_nail|hammer_nail|2500|hammer_nail"
  "single_pick_bucket|pick_bucket|10000|pick_bucket"
  "single_pinch_tongs|pinch_tongs|10000|pinch_tongs"
  "single_water_plant|water_plant|12500|water_plant"
  "mixed_fold_glasses|fold_glasses|55000|mixed_5task"
  "mixed_hammer_nail|hammer_nail|55000|mixed_5task"
  "mixed_pick_bucket|pick_bucket|55000|mixed_5task"
  "mixed_pinch_tongs|pinch_tongs|55000|mixed_5task"
  "mixed_water_plant|water_plant|55000|mixed_5task"
)

if [[ "${WAIT_IDLE:-1}" == "1" ]]; then
  MAX_USED_MIB="${MAX_USED_MIB:-8000}" MAX_UTIL="${MAX_UTIL:-15}" \
    dexjoco_wait_gpus_idle "${MASTER_LOG}"
fi

fail=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r jid task step ckpt_name <<<"${job}"
  task_out="${OUT_ROOT}/${jid}"
  log "START ${jid}"
  if ! TASK="${task}" \
    CKPT_DIR="${LINK_ROOT}/${ckpt_name}" \
    CHECKPOINT_STEPS="${step}" \
    GPUS="${GPUS}" \
    WAIT_IDLE=0 \
    OUT_ROOT="${task_out}" \
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
