#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/data_all/share/FastWAM_zhaoyc_failure}
RESULT_ROOT=${RESULT_ROOT:-/data_all/share/dexjoco_fastwam_results}
PY=${PY:-/home/gzr1/miniconda3/envs/residual/bin/python}
TASK=${TASK:?Set TASK, e.g. hammer_nail}
VARIANT=${VARIANT:?Set VARIANT, e.g. failure_embedding or text_failure}
RUN_ID=${RUN_ID:?Set RUN_ID}
STEPS=${STEPS:?Set STEPS, e.g. "12240 12000 11000"}
GPUS=${GPUS:-4,5,6,7}
WAIT_SESSION=${WAIT_SESSION:-}
EVAL_EPISODES=${EVAL_EPISODES:-50}
EVAL_SEED=${EVAL_SEED:-0}
REPLAN_STEPS=${REPLAN_STEPS:-24}
MAX_ENV_STEPS=${MAX_ENV_STEPS:-1500}
POLL_SECONDS=${POLL_SECONDS:-180}
EVAL_ALL=${EVAL_ALL:-0}
LONG_ON_FAIL=${LONG_ON_FAIL:-0}
LONG_MAX_STEPS=${LONG_MAX_STEPS:-12240}
LONG_STEPS=${LONG_STEPS:-"12240 12000 11000"}
LONG_RUN_ID=${LONG_RUN_ID:-${RUN_ID}_long_${LONG_MAX_STEPS}}
LONG_SESSION=${LONG_SESSION:-fastwam_${TASK}_${VARIANT}_long_${LONG_MAX_STEPS}}
RUN_ID_STEM=${RUN_ID_STEM:-$(date +%Y-%m-%d_%H-%M-%S)}
RESTART_WATCHER_ON_PASS=${RESTART_WATCHER_ON_PASS:-0}
WATCH_TASKS=${WATCH_TASKS:-"click_mouse pick_bucket pinch_tongs fold_glasses"}
NEXT_WATCH_SESSION=${NEXT_WATCH_SESSION:-fastwam_next_single_arm_pipeline_watch}

cd "$ROOT"
mkdir -p artifacts/logs artifacts/gates artifacts/evals
export PATH="$(dirname "$PY")":/home/zhaoyc/.local/bin:/home/gzr1/miniconda3/bin:$PATH
export PYTHONPATH="$ROOT/src:$ROOT/scripts:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/checkpoints"
export PYTHONUNBUFFERED=1

LOG="artifacts/logs/watch_eval_${TASK}_${VARIANT}_${RUN_ID_STEM}.log"
GATE_TSV="artifacts/gates/${TASK}_${VARIANT}_${RUN_ID_STEM}.tsv"
STATS_PATH="${ROOT}/artifacts/dataset_stats/dexjoco_${TASK}_success_action_state.json"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

run_dir_for() {
  local variant=$1
  local run_id=$2
  printf '%s/dexjoco_%s_%s_2cam_proprio_1e-4/%s' "$RESULT_ROOT" "$TASK" "$variant" "$run_id"
}

step_tag() {
  printf 'step_%06d' "$1"
}

summary_path_for() {
  local variant=$1
  local step=$2
  local run_id=$3
  printf '%s/artifacts/evals/%s_%s_step_%s_seed%s_%sep_%s/summary.json' \
    "$ROOT" "$TASK" "$variant" "$step" "$EVAL_SEED" "$EVAL_EPISODES" "$run_id"
}

wait_for_session() {
  local session=$1
  if [[ -z "$session" ]]; then
    return 0
  fi
  log "waiting for session to finish: $session"
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
  log "session finished: $session"
}

eval_step_if_needed() {
  local variant=$1
  local run_id=$2
  local step=$3
  local label=$4
  local run_dir ckpt out_dir summary
  run_dir="$(run_dir_for "$variant" "$run_id")"
  summary="$(summary_path_for "$variant" "$step" "$run_id")"
  if [[ -f "$summary" ]]; then
    log "summary already exists: $summary"
    LAST_SUMMARY="$summary"
    LAST_RUN_DIR="$run_dir"
    LAST_STEP="$step"
    return 0
  fi
  ckpt="${run_dir}/checkpoints/weights/$(step_tag "$step").pt"
  out_dir="${ROOT}/artifacts/evals/${TASK}_${variant}_step_${step}_seed${EVAL_SEED}_${EVAL_EPISODES}ep_${label}_${RUN_ID_STEM}"
  if [[ ! -f "$ckpt" ]]; then
    log "missing checkpoint: $ckpt"
    return 2
  fi
  stats_args=()
  if [[ -f "$STATS_PATH" ]]; then
    stats_args=(--dataset-stats-path "$STATS_PATH")
  fi
  log "rollout variant=$variant run_id=$run_id step=$step out=$out_dir"
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
  LAST_SUMMARY="${out_dir}/summary.json"
  LAST_RUN_DIR="$run_dir"
  LAST_STEP="$step"
  [[ -f "$LAST_SUMMARY" ]]
}

check_gate() {
  local variant=$1
  local run_id=$2
  local step=$3
  local summary=$4
  local result_json="artifacts/gates/${TASK}_${variant}_step_${step}_${RUN_ID_STEM}.json"
  local status=0
  if "$PY" scripts/check_dexjoco_gate.py "$summary" --task "$TASK" --json-out "$result_json" 2>&1 | tee -a "$LOG"; then
    status=0
  else
    status=$?
  fi
  if [[ ! -f "$GATE_TSV" ]]; then
    printf 'task\tvariant\trun_id\tstep\tpassed\tsummary\tresult_json\trun_dir\n' > "$GATE_TSV"
  fi
  if (( status == 0 )); then
    printf '%s\t%s\t%s\t%s\ttrue\t%s\t%s\t%s\n' "$TASK" "$variant" "$run_id" "$step" "$summary" "$result_json" "$LAST_RUN_DIR" >> "$GATE_TSV"
    log "gate PASS variant=$variant run_id=$run_id step=$step"
    return 0
  fi
  printf '%s\t%s\t%s\t%s\tfalse\t%s\t%s\t%s\n' "$TASK" "$variant" "$run_id" "$step" "$summary" "$result_json" "$LAST_RUN_DIR" >> "$GATE_TSV"
  log "gate FAIL variant=$variant run_id=$run_id step=$step"
  return 1
}

restart_watcher_if_needed() {
  if [[ "$RESTART_WATCHER_ON_PASS" != "1" ]]; then
    return 0
  fi
  if tmux has-session -t "$NEXT_WATCH_SESSION" 2>/dev/null; then
    log "next-task watcher already active: $NEXT_WATCH_SESSION"
    return 0
  fi
  log "starting next-task watcher after pass: tasks=$WATCH_TASKS session=$NEXT_WATCH_SESSION"
  tmux new-session -d -s "$NEXT_WATCH_SESSION" \
    "cd $ROOT && TASKS=\"$WATCH_TASKS\" GPUS=$GPUS MIN_AGE_SECONDS=900 POLL_SECONDS=180 bash scripts/watch_next_single_arm_baseline_then_pipeline.sh"
}

eval_steps_until_pass() {
  local variant=$1
  local run_id=$2
  local steps=$3
  local label=$4
  local step
  for step in $steps; do
    eval_step_if_needed "$variant" "$run_id" "$step" "${label}_${step}"
    check_gate "$variant" "$run_id" "$step" "$LAST_SUMMARY" && {
      if [[ "$EVAL_ALL" != "1" ]]; then
        return 0
      fi
    }
  done
  return 1
}

continue_long_then_eval() {
  local run_dir resume
  run_dir="$(run_dir_for "$VARIANT" "$RUN_ID")"
  resume="${run_dir}/checkpoints/state/$(step_tag "$LONG_MAX_STEPS")"
  if [[ ! -d "$resume" ]]; then
    resume="${run_dir}/checkpoints/state/$(step_tag 6000)"
  fi
  if [[ ! -d "$resume" ]]; then
    resume="${run_dir}/checkpoints/weights/$(step_tag 6000).pt"
  fi
  if [[ ! -e "$resume" ]]; then
    log "cannot continue long run; no resume state/weight found under $run_dir"
    return 2
  fi
  log "starting long continuation variant=$VARIANT max_steps=$LONG_MAX_STEPS resume=$resume session=$LONG_SESSION run_id=$LONG_RUN_ID"
  TASK="$TASK" \
    VARIANT="$VARIANT" \
    RESUME_CHECKPOINT="$resume" \
    GPUS="$GPUS" \
    WAIT_FOR_GPUS=1 \
    REQUIRE_FREE_MB=60000 \
    MAX_STEPS="$LONG_MAX_STEPS" \
    SAVE_EVERY=500 \
    EVAL_EVERY=500 \
    EVAL_EPISODES="$EVAL_EPISODES" \
    EVAL_SEED="$EVAL_SEED" \
    RUN_ID="$LONG_RUN_ID" \
    bash scripts/run_dexjoco_failure_task_once.sh 2>&1 | tee -a "$LOG"
  eval_steps_until_pass "$VARIANT" "$LONG_RUN_ID" "$LONG_STEPS" "long"
}

log "watch-eval start task=$TASK variant=$VARIANT run_id=$RUN_ID steps=$STEPS wait_session=$WAIT_SESSION gpus=$GPUS"
wait_for_session "$WAIT_SESSION"

if eval_steps_until_pass "$VARIANT" "$RUN_ID" "$STEPS" "primary"; then
  log "watch-eval complete with passing checkpoint"
  restart_watcher_if_needed
  exit 0
fi

if [[ "$LONG_ON_FAIL" == "1" ]]; then
  if continue_long_then_eval; then
    restart_watcher_if_needed
    exit 0
  fi
  exit $?
fi

log "watch-eval completed without a passing checkpoint"
exit 42
