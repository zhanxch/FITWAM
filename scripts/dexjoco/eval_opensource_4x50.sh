#!/usr/bin/env bash
# Official 4×50 DexJoCo eval on the open-source stack (224 / z-score).
#
# Required:
#   GPUS=4,5,6,7
#   TASK=fold_glasses          # or TASKS=fold_glasses,hammer_nail,...
#
# Optional session knobs (do not fork a new .sh for these):
#   WAIT_IDLE=1|0              # default 1
#   CKPT_DIR / CHECKPOINT_STEPS / OUT_ROOT / STAMP / BASE_PORT
#   DATASET_STATS / TEXT_EMBEDDING  (defaults: OPEN artifacts for TASK)
#
# Protocol is pinned: seeds 0..49 × 4, replan=24, horizon=32, nfe=10, max_steps=1000.
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

if [[ -n "${TASKS:-}" ]]; then
  IFS=',' read -r -a TASK_ARR <<< "${TASKS}"
  for i in "${!TASK_ARR[@]}"; do
    TASK_ARR[$i]="${TASK_ARR[$i]// /}"
  done
elif [[ -n "${TASK:-}" ]]; then
  TASK_ARR=("${TASK}")
else
  echo "[opensource-eval] ERROR: set TASK or TASKS" >&2
  exit 2
fi

WAIT_IDLE="${WAIT_IDLE:-1}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SEED_START="${SEED_START:-0}"
SEED_END="${SEED_END:-49}"
REPEATS="${REPEATS:-4}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
NFE="${NFE:-10}"
MAX_STEPS="${MAX_STEPS:-1000}"
MODEL_CONFIG="${MODEL_CONFIG:-${OPEN_REPO}/configs/fastwam_dexjoco.yaml}"
[[ -f "${MODEL_CONFIG}" ]] || { echo "[opensource-eval] ERROR missing ${MODEL_CONFIG}" >&2; exit 1; }

if [[ "${#TASK_ARR[@]}" -eq 1 ]]; then
  OUT_ROOT="${OUT_ROOT:-${ROOT}/evaluate_results/dexjoco/${TASK_ARR[0]}_opensource_4x50_${STAMP}}"
else
  OUT_ROOT="${OUT_ROOT:-${ROOT}/evaluate_results/dexjoco/opensource_multitask_4x50_${STAMP}}"
fi
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/orchestrator.log"
log() { echo "[opensource-eval $(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

log "OPEN_REPO=${OPEN_REPO}"
log "GPUS=${GPUS} WAIT_IDLE=${WAIT_IDLE} tasks=${TASK_ARR[*]}"
log "protocol: seeds ${SEED_START}..${SEED_END} × ${REPEATS}, replan=${REPLAN_STEPS}, max_steps=${MAX_STEPS}, nfe=${NFE}"
log "out=${OUT_ROOT}"

resolve_task_paths() {
  local task="$1"
  local ckpt_dir steps stats emb
  if [[ -n "${CKPT_DIR:-}" && "${#TASK_ARR[@]}" -eq 1 ]]; then
    ckpt_dir="${CKPT_DIR}"
  else
    ckpt_dir="${OPEN_REPO}/checkpoints/${task}"
  fi
  if [[ -n "${CHECKPOINT_STEPS:-}" && "${#TASK_ARR[@]}" -eq 1 ]]; then
    steps="${CHECKPOINT_STEPS}"
  else
    steps="$(python "${ROOT}/scripts/dewo_v2/tasks.py" export-env --task "${task}" \
      | sed -n 's/^export CKPT_STEP=//p')"
  fi
  stats="${DATASET_STATS:-${OPEN_REPO}/artifacts/${task}/dataset_stats.json}"
  if [[ -n "${TEXT_EMBEDDING:-}" && "${#TASK_ARR[@]}" -eq 1 ]]; then
    emb="${TEXT_EMBEDDING}"
  else
    emb="$(dexjoco_find_t5 "${task}")"
  fi
  ckpt="${ckpt_dir}/step_$(printf '%06d' "${steps}").pt"
  [[ -e "${ckpt}" ]] || { log "ERROR missing ${ckpt}"; return 2; }
  [[ -f "${stats}" ]] || { log "ERROR missing ${stats}"; return 2; }
  [[ -f "${emb}" ]] || { log "ERROR missing ${emb}"; return 2; }
  printf '%s|%s|%s|%s' "${ckpt_dir}" "${steps}" "${stats}" "${emb}"
}

declare -a TASK_SPECS=()
for task in "${TASK_ARR[@]}"; do
  spec="$(resolve_task_paths "${task}")"
  TASK_SPECS+=("${spec}")
  IFS='|' read -r ckpt_dir steps stats emb <<<"${spec}"
  log "ready ${task} step=${steps} ckpt_dir=${ckpt_dir} emb=$(basename "${emb}")"
done

if [[ "${WAIT_IDLE}" == "1" ]]; then
  dexjoco_wait_gpus_idle "${MASTER_LOG}"
fi

run_one() {
  local task="$1" ckpt_dir="$2" steps="$3" stats="$4" emb="$5" task_out="$6"
  mkdir -p "${task_out}"
  log "START ${task} step=${steps} -> ${task_out}"
  (
    cd "${OPEN_REPO}"
    "${ENV_PREFIX}/bin/python" "${ROOT}/scripts/dexjoco/run_opensource_eval_dexjoco.py" \
      --task-name "${task}" \
      --checkpoint-dir "${ckpt_dir}" \
      --checkpoint-steps "${steps}" \
      --model-config "${MODEL_CONFIG}" \
      --dataset-stats "${stats}" \
      --text-embedding "${emb}" \
      --gpus "${GPUS}" \
      --seed-start "${SEED_START}" \
      --seed-end "${SEED_END}" \
      --repeats "${REPEATS}" \
      --action-horizon "${ACTION_HORIZON}" \
      --replan-steps "${REPLAN_STEPS}" \
      --num-inference-steps "${NFE}" \
      --max-steps "${MAX_STEPS}" \
      --output-dir "${task_out}"
  ) >"${LOG_DIR}/eval_${task}.log" 2>&1
  log "DONE ${task}"
}

fail=0
pids=()
if [[ "${#TASK_ARR[@]}" -eq 1 ]]; then
  IFS='|' read -r ckpt_dir steps stats emb <<<"${TASK_SPECS[0]}"
  if ! run_one "${TASK_ARR[0]}" "${ckpt_dir}" "${steps}" "${stats}" "${emb}" "${OUT_ROOT}"; then
    fail=1
  fi
else
  for i in "${!TASK_ARR[@]}"; do
    task="${TASK_ARR[$i]}"
    IFS='|' read -r ckpt_dir steps stats emb <<<"${TASK_SPECS[$i]}"
    run_one "${task}" "${ckpt_dir}" "${steps}" "${stats}" "${emb}" "${OUT_ROOT}/${task}" &
    pids+=("$!")
  done
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      log "FAIL ${TASK_ARR[$i]}"
      fail=1
    fi
  done
fi

log "ALL complete fail=${fail} out=${OUT_ROOT}"
exit "${fail}"
