#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/data_all/share/FastWAM_zhaoyc_failure}
RESULT_ROOT=${RESULT_ROOT:-/data_all/share/dexjoco_fastwam_results}
PY=${PY:-/home/gzr1/miniconda3/envs/residual/bin/python}
TASK=${TASK:?Set TASK, e.g. click_mouse}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:?Set RESUME_CHECKPOINT to the success-only baseline checkpoint}
GPUS=${GPUS:-4,5,6,7}
STRUCTURE_VARIANT=${STRUCTURE_VARIANT:-failure_embedding}
CONCAT_VARIANT=${CONCAT_VARIANT:-text_failure}
FIRST_MAX_STEPS=${FIRST_MAX_STEPS:-6000}
CONCAT_LONG_STEPS=${CONCAT_LONG_STEPS:-12240}
SAVE_EVERY=${SAVE_EVERY:-500}
EVAL_EVERY=${EVAL_EVERY:-500}
EVAL_EPISODES=${EVAL_EPISODES:-50}
EVAL_SEED=${EVAL_SEED:-0}
REPLAN_STEPS=${REPLAN_STEPS:-24}
MAX_ENV_STEPS=${MAX_ENV_STEPS:-1500}
REQUIRE_FREE_MB=${REQUIRE_FREE_MB:-60000}
RUN_ID=${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}
WATCH_TASKS=${WATCH_TASKS:-"click_mouse pick_bucket pinch_tongs fold_glasses"}
RESTART_WATCHER_ON_PASS=${RESTART_WATCHER_ON_PASS:-1}

cd "$ROOT"
mkdir -p artifacts/logs artifacts/gates artifacts/evals
export PATH="$(dirname "$PY")":/home/zhaoyc/.local/bin:/home/gzr1/miniconda3/bin:$PATH
export PYTHONPATH="$ROOT/src:$ROOT/scripts:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/checkpoints"
export PYTHONUNBUFFERED=1

LOG="artifacts/logs/supervise_${TASK}_fallback_${RUN_ID}.log"
GATE_TSV="artifacts/gates/${TASK}_fallback_${RUN_ID}.tsv"
SELECTED_FILE="artifacts/gates/${TASK}_selected_${RUN_ID}.txt"
FAILED_FILE="artifacts/gates/${TASK}_failed_all_fallbacks_${RUN_ID}.txt"
STATS_PATH="${ROOT}/artifacts/dataset_stats/dexjoco_${TASK}_success_action_state.json"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

step_tag() {
  printf 'step_%06d' "$1"
}

variant_run_dir() {
  local variant=$1
  local run_id=$2
  printf '%s/dexjoco_%s_%s_2cam_proprio_1e-4/%s' "$RESULT_ROOT" "$TASK" "$variant" "$run_id"
}

summary_for() {
  local variant=$1
  local step=$2
  local run_id=$3
  printf '%s/artifacts/evals/%s_%s_step_%s_seed%s_%sep_%s/summary.json' \
    "$ROOT" "$TASK" "$variant" "$step" "$EVAL_SEED" "$EVAL_EPISODES" "$run_id"
}

train_variant() {
  local variant=$1
  local max_steps=$2
  local resume=$3
  local label=$4
  local run_id="${RUN_ID}_${label}_${variant}_${max_steps}"
  local run_dir
  run_dir="$(variant_run_dir "$variant" "$run_id")"
  log "train start variant=$variant max_steps=$max_steps resume=$resume run_id=$run_id"
  TASK="$TASK" \
    VARIANT="$variant" \
    RESUME_CHECKPOINT="$resume" \
    GPUS="$GPUS" \
    WAIT_FOR_GPUS=1 \
    REQUIRE_FREE_MB="$REQUIRE_FREE_MB" \
    MAX_STEPS="$max_steps" \
    SAVE_EVERY="$SAVE_EVERY" \
    EVAL_EVERY="$EVAL_EVERY" \
    EVAL_EPISODES="$EVAL_EPISODES" \
    EVAL_SEED="$EVAL_SEED" \
    RUN_ID="$run_id" \
    bash scripts/run_dexjoco_failure_task_once.sh 2>&1 | tee -a "$LOG"
  LAST_VARIANT="$variant"
  LAST_STEPS="$max_steps"
  LAST_RUN_ID="$run_id"
  LAST_RUN_DIR="$run_dir"
  LAST_SUMMARY="$(summary_for "$variant" "$max_steps" "$run_id")"
  if [[ ! -f "$LAST_SUMMARY" ]]; then
    log "missing final summary after train: $LAST_SUMMARY"
    exit 20
  fi
}

eval_checkpoint() {
  local variant=$1
  local run_id=$2
  local step=$3
  local label=$4
  local run_dir ckpt out_dir
  run_dir="$(variant_run_dir "$variant" "$run_id")"
  ckpt="${run_dir}/checkpoints/weights/$(step_tag "$step").pt"
  out_dir="${ROOT}/artifacts/evals/${TASK}_${variant}_step_${step}_seed${EVAL_SEED}_${EVAL_EPISODES}ep_${label}_${RUN_ID}"
  if [[ ! -f "$ckpt" ]]; then
    log "missing checkpoint for eval: $ckpt"
    exit 21
  fi
  log "rollout start variant=$variant step=$step ckpt=$ckpt out=$out_dir"
  stats_args=()
  if [[ -f "$STATS_PATH" ]]; then
    stats_args=(--dataset-stats-path "$STATS_PATH")
  fi
  "$PY" scripts/water_plant/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
    --gpus "$GPUS" \
    --run-dir "$run_dir" \
    --checkpoint "$ckpt" \
    "${stats_args[@]}" \
    --no-load-text-encoder \
    --server-conda-env residual \
    --client-conda-env residual \
    --task-config-dir third_party/dexjoco/configs/rand_obj \
    --tasks "$TASK" \
    --episodes "$EVAL_EPISODES" \
    --seed "$EVAL_SEED" \
    --replan-steps "$REPLAN_STEPS" \
    --control-mode blocking \
    --max-env-steps "$MAX_ENV_STEPS" \
    --output-dir "$out_dir" 2>&1 | tee -a "$LOG"
  LAST_VARIANT="$variant"
  LAST_STEPS="$step"
  LAST_RUN_ID="$run_id"
  LAST_RUN_DIR="$run_dir"
  LAST_SUMMARY="${out_dir}/summary.json"
  if [[ ! -f "$LAST_SUMMARY" ]]; then
    log "missing rollout summary: $LAST_SUMMARY"
    exit 22
  fi
}

record_gate() {
  local variant=$1
  local step=$2
  local run_id=$3
  local run_dir=$4
  local summary=$5
  local label=$6
  local result_json="artifacts/gates/${TASK}_${variant}_step_${step}_${label}_${RUN_ID}.json"
  local status=0
  log "gate check variant=$variant step=$step label=$label summary=$summary"
  if "$PY" scripts/check_dexjoco_gate.py "$summary" --task "$TASK" --json-out "$result_json" 2>&1 | tee -a "$LOG"; then
    status=0
  else
    status=$?
  fi
  if [[ ! -f "$GATE_TSV" ]]; then
    printf 'task\tvariant\tstep\tlabel\tpassed\tsummary\tresult_json\trun_dir\n' > "$GATE_TSV"
  fi
  if (( status == 0 )); then
    printf '%s\t%s\t%s\t%s\ttrue\t%s\t%s\t%s\n' "$TASK" "$variant" "$step" "$label" "$summary" "$result_json" "$run_dir" >> "$GATE_TSV"
    {
      echo "task=$TASK"
      echo "variant=$variant"
      echo "step=$step"
      echo "label=$label"
      echo "run_id=$run_id"
      echo "run_dir=$run_dir"
      echo "summary=$summary"
      echo "result_json=$result_json"
      echo "selected_at=$(date '+%F %T')"
    } > "$SELECTED_FILE"
    log "gate PASS variant=$variant step=$step"
    return 0
  fi
  printf '%s\t%s\t%s\t%s\tfalse\t%s\t%s\t%s\n' "$TASK" "$variant" "$step" "$label" "$summary" "$result_json" "$run_dir" >> "$GATE_TSV"
  log "gate FAIL variant=$variant step=$step"
  return 1
}

restart_watcher_if_needed() {
  if [[ "$RESTART_WATCHER_ON_PASS" != "1" ]]; then
    return 0
  fi
  if tmux has-session -t fastwam_next_single_arm_pipeline_watch 2>/dev/null; then
    log "next-task watcher already active"
    return 0
  fi
  log "starting next-task watcher after pass"
  tmux new-session -d -s fastwam_next_single_arm_pipeline_watch \
    "cd $ROOT && TASKS=\"$WATCH_TASKS\" GPUS=$GPUS MIN_AGE_SECONDS=900 POLL_SECONDS=180 bash scripts/watch_next_single_arm_baseline_then_pipeline.sh"
}

pass_and_exit() {
  restart_watcher_if_needed
  exit 0
}

resume_for_concat_long() {
  local run_dir=$1
  local state_dir="${run_dir}/checkpoints/state/$(step_tag "$FIRST_MAX_STEPS")"
  local weight_path="${run_dir}/checkpoints/weights/$(step_tag "$FIRST_MAX_STEPS").pt"
  if [[ -d "$state_dir" ]]; then
    printf '%s' "$state_dir"
    return 0
  fi
  log "state resume missing for concat long run; falling back to weight-only resume: $weight_path"
  printf '%s' "$weight_path"
}

log "fallback supervisor start task=$TASK resume_checkpoint=$RESUME_CHECKPOINT gpus=$GPUS"
log "gate sequence: ${STRUCTURE_VARIANT} 6000, 5500, 5000; ${CONCAT_VARIANT} 6000, 5500, 5000; ${CONCAT_VARIANT} 12240, 12000, 11000"

train_variant "$STRUCTURE_VARIANT" "$FIRST_MAX_STEPS" "$RESUME_CHECKPOINT" "structure"
structure_run_id="$LAST_RUN_ID"
structure_run_dir="$LAST_RUN_DIR"
if record_gate "$LAST_VARIANT" "$LAST_STEPS" "$LAST_RUN_ID" "$LAST_RUN_DIR" "$LAST_SUMMARY" "final"; then
  pass_and_exit
fi
for step in 5500 5000; do
  eval_checkpoint "$STRUCTURE_VARIANT" "$structure_run_id" "$step" "structure_fallback_${step}"
  if record_gate "$LAST_VARIANT" "$LAST_STEPS" "$LAST_RUN_ID" "$LAST_RUN_DIR" "$LAST_SUMMARY" "fallback_${step}"; then
    pass_and_exit
  fi
done

train_variant "$CONCAT_VARIANT" "$FIRST_MAX_STEPS" "$RESUME_CHECKPOINT" "concat"
concat_run_id="$LAST_RUN_ID"
concat_run_dir="$LAST_RUN_DIR"
if record_gate "$LAST_VARIANT" "$LAST_STEPS" "$LAST_RUN_ID" "$LAST_RUN_DIR" "$LAST_SUMMARY" "final"; then
  pass_and_exit
fi
for step in 5500 5000; do
  eval_checkpoint "$CONCAT_VARIANT" "$concat_run_id" "$step" "concat_fallback_${step}"
  if record_gate "$LAST_VARIANT" "$LAST_STEPS" "$LAST_RUN_ID" "$LAST_RUN_DIR" "$LAST_SUMMARY" "fallback_${step}"; then
    pass_and_exit
  fi
done

concat_resume="$(resume_for_concat_long "$concat_run_dir")"
train_variant "$CONCAT_VARIANT" "$CONCAT_LONG_STEPS" "$concat_resume" "concat_long"
concat_long_run_id="$LAST_RUN_ID"
if record_gate "$LAST_VARIANT" "$LAST_STEPS" "$LAST_RUN_ID" "$LAST_RUN_DIR" "$LAST_SUMMARY" "final"; then
  pass_and_exit
fi
for step in 12000 11000; do
  eval_checkpoint "$CONCAT_VARIANT" "$concat_long_run_id" "$step" "concat_long_fallback_${step}"
  if record_gate "$LAST_VARIANT" "$LAST_STEPS" "$LAST_RUN_ID" "$LAST_RUN_DIR" "$LAST_SUMMARY" "fallback_${step}"; then
    pass_and_exit
  fi
done

{
  echo "task=$TASK"
  echo "failed_at=$(date '+%F %T')"
  echo "gate_tsv=$GATE_TSV"
  echo "log=$LOG"
} > "$FAILED_FILE"
log "all fallback gates failed for task=$TASK; leaving watcher stopped for inspection"
exit 42
