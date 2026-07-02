#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/data_all/share/FastWAM_zhaoyc_failure}
TASK=${TASK:?Set TASK, e.g. hammer_nail}
TARGET_TOTAL_FAILURES=${TARGET_TOTAL_FAILURES:-100}
CHECK_INTERVAL=${CHECK_INTERVAL:-120}
TRAIN_GPUS=${TRAIN_GPUS:-4,5,6,7}
TRAIN_VARIANT=${TRAIN_VARIANT:-failure_embedding}
TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS:-6000}
TRAIN_SAVE_EVERY=${TRAIN_SAVE_EVERY:-500}
TRAIN_EVAL_EVERY=${TRAIN_EVAL_EVERY:-500}
TRAIN_EVAL_EPISODES=${TRAIN_EVAL_EPISODES:-50}
TRAIN_REQUIRE_FREE_MB=${TRAIN_REQUIRE_FREE_MB:-60000}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:?Set RESUME_CHECKPOINT to the success checkpoint}
DEFAULT_DATASET="${ROOT}/artifacts/datasets/${TASK}_failure_fastwam_2cam_text"
SHARD_SUFFIXES=${SHARD_SUFFIXES:-s20000 s30000 s40000}
COLLECT_SESSIONS=${COLLECT_SESSIONS:-fastwam_${TASK}_failure_collect fastwam_${TASK}_failure_collect_s20000 fastwam_${TASK}_failure_collect_s30000 fastwam_${TASK}_failure_collect_s40000}
TRAIN_SESSION=${TRAIN_SESSION:-fastwam_${TASK}_${TRAIN_VARIANT}_train}
TRAIN_USE_FALLBACK_SUPERVISOR=${TRAIN_USE_FALLBACK_SUPERVISOR:-1}
WATCH_TASKS=${WATCH_TASKS:-"click_mouse pick_bucket pinch_tongs fold_glasses"}
RUN_ID=${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}
LOGDIR="${ROOT}/artifacts/logs"
LOG="${LOGDIR}/watch_${TASK}_failures_then_train_${RUN_ID}.log"

cd "$ROOT"
mkdir -p "$LOGDIR"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

dataset_dirs() {
  printf '%s\n' "$DEFAULT_DATASET"
  for suffix in $SHARD_SUFFIXES; do
    printf '%s\n' "${DEFAULT_DATASET}_${suffix}"
  done
}

failure_total() {
  /home/gzr1/miniconda3/envs/residual/bin/python - "$@" <<'PY'
import json
import sys
from pathlib import Path

total_failures = 0
total_episodes = 0
rows = []
for raw in sys.argv[1:]:
    root = Path(raw)
    summary_path = root / "collection_summary.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    failures = 0
    if summary_path.exists():
        try:
            failures = int(json.loads(summary_path.read_text()).get("failures", 0))
        except Exception:
            failures = 0
    episodes = 0
    if episodes_path.exists():
        with episodes_path.open("r", encoding="utf-8") as f:
            episodes = sum(1 for line in f if line.strip())
    total_failures += failures
    total_episodes += episodes
    rows.append({"path": str(root), "failures": failures, "episodes": episodes})
print(json.dumps({"failures": total_failures, "episodes": total_episodes, "rows": rows}))
PY
}

read_total_field() {
  /home/gzr1/miniconda3/envs/residual/bin/python -c "import json,sys; print(json.load(sys.stdin).get('$1', 0))"
}

log "watching task=$TASK target_failures=$TARGET_TOTAL_FAILURES dirs=$(dataset_dirs | tr '\n' ' ')"
while true; do
  stats=$(failure_total $(dataset_dirs))
  failures=$(printf '%s' "$stats" | read_total_field failures)
  episodes=$(printf '%s' "$stats" | read_total_field episodes)
  log "current failures=$failures episodes=$episodes stats=$stats"
  if (( failures >= TARGET_TOTAL_FAILURES && episodes >= TARGET_TOTAL_FAILURES )); then
    break
  fi
  sleep "$CHECK_INTERVAL"
done

log "target reached; stopping collectors: $COLLECT_SESSIONS"
for session in $COLLECT_SESSIONS; do
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux send-keys -t "$session" C-c || true
  fi
done
sleep 45
for session in $COLLECT_SESSIONS; do
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux kill-session -t "$session" || true
  fi
done

stamp=$(date +%Y-%m-%d_%H-%M-%S)
raw_default="${DEFAULT_DATASET}_s10000_raw_${stamp}"
inputs=()
if [[ -d "$DEFAULT_DATASET" && -f "$DEFAULT_DATASET/collection_summary.json" ]]; then
  log "moving default collector shard to $raw_default"
  mv "$DEFAULT_DATASET" "$raw_default"
  inputs+=("$raw_default")
elif [[ -d "$DEFAULT_DATASET" ]]; then
  log "default dataset exists without collection_summary; keeping it out of raw merge inputs: $DEFAULT_DATASET"
fi
for suffix in $SHARD_SUFFIXES; do
  path="${DEFAULT_DATASET}_${suffix}"
  if [[ -f "$path/meta/episodes.jsonl" ]]; then
    inputs+=("$path")
  fi
done

if (( ${#inputs[@]} == 0 )); then
  log "no merge inputs found"
  exit 20
fi

tmp_output="${DEFAULT_DATASET}_merged_tmp_${stamp}"
log "merging first $TARGET_TOTAL_FAILURES failures into $tmp_output from inputs=${inputs[*]}"
PYTHONPATH="$ROOT/src:$ROOT/scripts" /home/gzr1/miniconda3/envs/residual/bin/python \
  scripts/merge_lerobot_failure_shards.py \
  --inputs "${inputs[@]}" \
  --output "$tmp_output" \
  --overwrite \
  --max-episodes "$TARGET_TOTAL_FAILURES" | tee -a "$LOG"

merged_episodes=$(/home/gzr1/miniconda3/envs/residual/bin/python -c "import json,pathlib; print(json.loads(pathlib.Path('$tmp_output/merge_summary.json').read_text()).get('episodes', 0))")
if (( merged_episodes < TARGET_TOTAL_FAILURES )); then
  log "merged only $merged_episodes episodes; refusing to train"
  exit 21
fi

if [[ -e "$DEFAULT_DATASET" ]]; then
  backup="${DEFAULT_DATASET}_pretrain_backup_${stamp}"
  log "moving existing default dataset to $backup"
  mv "$DEFAULT_DATASET" "$backup"
fi
mv "$tmp_output" "$DEFAULT_DATASET"
log "final merged dataset ready: $DEFAULT_DATASET episodes=$merged_episodes"

if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
  log "train session already exists: $TRAIN_SESSION"
  exit 0
fi

train_log="${LOGDIR}/train_${TASK}_${TRAIN_VARIANT}_autostart_${stamp}.log"
if [[ "$TRAIN_USE_FALLBACK_SUPERVISOR" == "1" ]]; then
  train_cmd="cd $ROOT && export PATH=/home/zhaoyc/.local/bin:/home/gzr1/miniconda3/bin:\$PATH && export PYTHONPATH=$ROOT/src:$ROOT/scripts:\${PYTHONPATH:-} && TASK=$TASK RESUME_CHECKPOINT=$RESUME_CHECKPOINT GPUS=$TRAIN_GPUS REQUIRE_FREE_MB=$TRAIN_REQUIRE_FREE_MB FIRST_MAX_STEPS=$TRAIN_MAX_STEPS SAVE_EVERY=$TRAIN_SAVE_EVERY EVAL_EVERY=$TRAIN_EVAL_EVERY EVAL_EPISODES=$TRAIN_EVAL_EPISODES WATCH_TASKS=\"$WATCH_TASKS\" bash scripts/run_dexjoco_fallback_supervisor.sh 2>&1 | tee -a $train_log"
else
  train_cmd="cd $ROOT && export PATH=/home/zhaoyc/.local/bin:/home/gzr1/miniconda3/bin:\$PATH && export PYTHONPATH=$ROOT/src:$ROOT/scripts:\${PYTHONPATH:-} && TASK=$TASK VARIANT=$TRAIN_VARIANT RESUME_CHECKPOINT=$RESUME_CHECKPOINT GPUS=$TRAIN_GPUS WAIT_FOR_GPUS=1 REQUIRE_FREE_MB=$TRAIN_REQUIRE_FREE_MB MAX_STEPS=$TRAIN_MAX_STEPS SAVE_EVERY=$TRAIN_SAVE_EVERY EVAL_EVERY=$TRAIN_EVAL_EVERY EVAL_EPISODES=$TRAIN_EVAL_EPISODES bash scripts/run_dexjoco_failure_task_once.sh 2>&1 | tee -a $train_log"
fi
tmux new-session -d -s "$TRAIN_SESSION" "$train_cmd"
log "started train session=$TRAIN_SESSION log=$train_log"
