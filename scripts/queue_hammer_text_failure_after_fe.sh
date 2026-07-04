#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/data_all/zhaoyc/Summer2/FastWAM_zhaoyc_failure_moved_from_share_20260703}
WAIT_SESSIONS=${WAIT_SESSIONS:-"fastwam_hammer_failure_embedding_long_12240 fastwam_hammer_fe_long_eval_watch"}
GPUS=${GPUS:-4,5,6,7}
POLL_SECONDS=${POLL_SECONDS:-300}
BASELINE_CKPT=${BASELINE_CKPT:-/data_all/zhaoyc/Summer2/dexjoco_fastwam_results_moved_from_share_20260703/hammer_nail_uncond_2cam_384_1e-4/2026-07-01_10-04-05/checkpoints/weights/step_006650.pt}
RUN_ID=${RUN_ID:-2026-07-02_after_fe_hammer_text_failure_6000_g4567}
LONG_RUN_ID=${LONG_RUN_ID:-2026-07-02_after_fe_hammer_text_failure_long_12240_g4567}
RUN_ID_STEM=${RUN_ID_STEM:-2026-07-02_after_fe_hammer_text_failure_eval}
RESTART_NEXT_ON_DONE=${RESTART_NEXT_ON_DONE:-1}
WATCH_TASKS=${WATCH_TASKS:-"click_mouse pick_bucket pinch_tongs fold_glasses"}
NEXT_WATCH_SESSION=${NEXT_WATCH_SESSION:-fastwam_next_single_arm_pipeline_watch}

cd "$ROOT"
mkdir -p artifacts/logs
LOG="artifacts/logs/queue_hammer_text_failure_after_fe_2026-07-02.log"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

restart_next_watcher_if_needed() {
  if [[ "$RESTART_NEXT_ON_DONE" != "1" ]]; then
    return 0
  fi
  if tmux has-session -t "$NEXT_WATCH_SESSION" 2>/dev/null; then
    log "next-task watcher already active: $NEXT_WATCH_SESSION"
    return 0
  fi
  log "starting next-task watcher after hammer concat queue: tasks=$WATCH_TASKS session=$NEXT_WATCH_SESSION"
  tmux new-session -d -s "$NEXT_WATCH_SESSION" \
    "cd $ROOT && TASKS=\"$WATCH_TASKS\" GPUS=$GPUS MIN_AGE_SECONDS=900 POLL_SECONDS=180 bash scripts/watch_next_single_arm_baseline_then_pipeline.sh"
}

session_active() {
  local session
  for session in $WAIT_SESSIONS; do
    if tmux has-session -t "$session" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

log "waiting for sessions to finish: $WAIT_SESSIONS"
while session_active; do
  sleep "$POLL_SECONDS"
done

log "starting hammer_nail text_failure 6000 on GPUs=$GPUS run_id=$RUN_ID"
TASK=hammer_nail \
  VARIANT=text_failure \
  RESUME_CHECKPOINT="$BASELINE_CKPT" \
  GPUS="$GPUS" \
  WAIT_FOR_GPUS=1 \
  REQUIRE_FREE_MB=60000 \
  MAX_STEPS=6000 \
  SAVE_EVERY=500 \
  EVAL_EVERY=500 \
  EVAL_EPISODES=50 \
  EVAL_SEED=0 \
  RUN_ID="$RUN_ID" \
  bash scripts/run_dexjoco_failure_task_once.sh 2>&1 | tee -a "$LOG"

log "starting hammer_nail text_failure fallback eval"
set +e
TASK=hammer_nail \
  VARIANT=text_failure \
  RUN_ID="$RUN_ID" \
  STEPS="6000 5500 5000" \
  GPUS="$GPUS" \
  LONG_ON_FAIL=1 \
  LONG_RUN_ID="$LONG_RUN_ID" \
  LONG_STEPS="12240 12000 11000" \
  RESTART_WATCHER_ON_PASS=1 \
  WATCH_TASKS="$WATCH_TASKS" \
  NEXT_WATCH_SESSION="$NEXT_WATCH_SESSION" \
  RUN_ID_STEM="$RUN_ID_STEM" \
  bash scripts/watch_dexjoco_run_then_eval_steps.sh 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
restart_next_watcher_if_needed
exit "$status"
