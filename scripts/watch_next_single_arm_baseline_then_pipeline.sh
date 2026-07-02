#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/data_all/share/FastWAM_zhaoyc_failure}
PY=${PY:-/home/gzr1/miniconda3/envs/residual/bin/python}
SEARCH_ROOT=${SEARCH_ROOT:-/data_all/share/dexjoco_fastwam_results}
TASKS=${TASKS:-"click_mouse pick_bucket pinch_tongs fold_glasses"}
GPUS=${GPUS:-4,5,6,7}
POLL_SECONDS=${POLL_SECONDS:-180}
MIN_BYTES=${MIN_BYTES:-1000000000}
MIN_AGE_SECONDS=${MIN_AGE_SECONDS:-900}
EXIT_AFTER_LAUNCH=${EXIT_AFTER_LAUNCH:-1}
DRY_RUN=${DRY_RUN:-0}
RUN_ID=${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}

cd "$ROOT"
mkdir -p artifacts/baseline_watch artifacts/logs
LOG="artifacts/logs/watch_next_single_arm_baseline_then_pipeline_${RUN_ID}.log"
STATE_DIR="artifacts/baseline_watch"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

pipeline_active() {
  tmux ls 2>/dev/null \
    | cut -d: -f1 \
    | grep -Eq '^fastwam_[a-z_]+_(pipeline_[0-9]{8}_[0-9]{6}|failure_collect(_s[0-9]+)?|failure_embedding_train|text_failure_train|success_train)$'
}

find_checkpoint() {
  "$PY" - "$SEARCH_ROOT" "$MIN_BYTES" "$MIN_AGE_SECONDS" "$1" <<'PY'
import re
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
min_bytes = int(sys.argv[2])
min_age = int(sys.argv[3])
task = sys.argv[4]
now = time.time()
step_re = re.compile(r"step_(\d+)\.pt$")
bad = re.compile(r"(failure|structured|text_failure)", re.IGNORECASE)
runs = {}
for path in root.rglob("step_*.pt"):
    text = str(path)
    if task not in text:
        continue
    if bad.search(text):
        continue
    if "/checkpoints/weights/" not in text:
        continue
    try:
        st = path.stat()
    except OSError:
        continue
    match = step_re.search(path.name)
    if not match:
        continue
    run_dir = path.parent.parent.parent
    if not (run_dir / "config.yaml").exists():
        continue
    row = (int(match.group(1)), st.st_mtime, str(path), str(run_dir), st.st_size)
    runs.setdefault(str(run_dir), []).append(row)
candidates = []
for run_dir, rows in runs.items():
    # Do not start from a partially uploaded or still-training run.
    newest_mtime = max(row[1] for row in rows)
    if now - newest_mtime < min_age:
        continue
    good_rows = [row for row in rows if row[4] >= min_bytes]
    if not good_rows:
        continue
    candidates.append(max(good_rows, key=lambda row: (row[0], row[1])))
if not candidates:
    raise SystemExit(1)
candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
step, mtime, ckpt, run_dir, _size = candidates[0]
print(f"{step}\t{mtime:.0f}\t{ckpt}\t{run_dir}")
PY
}

log "watch start tasks=$TASKS search_root=$SEARCH_ROOT min_age=${MIN_AGE_SECONDS}s gpus=$GPUS dry_run=$DRY_RUN"
while true; do
  if pipeline_active; then
    log "pipeline/collector/train already active; waiting"
    sleep "$POLL_SECONDS"
    continue
  fi

  launched=0
  for task in $TASKS; do
    done_file="${STATE_DIR}/${task}_pipeline_launched.txt"
    if [[ -f "$done_file" ]]; then
      continue
    fi

    log "scanning next task=$task"
    if candidate="$(find_checkpoint "$task")"; then
      step="$(printf '%s' "$candidate" | cut -f1)"
      checkpoint="$(printf '%s' "$candidate" | cut -f3)"
      run_dir="$(printf '%s' "$candidate" | cut -f4)"
      session="fastwam_${task}_pipeline_$(date +%Y%m%d_%H%M%S)"
      log "launching task=$task step=$step checkpoint=$checkpoint run_dir=$run_dir session=$session"
      if [[ "$DRY_RUN" == "1" ]]; then
        log "dry-run: would launch task=$task session=$session"
        exit 0
      fi
      tmux new-session -d -s "$session" \
        "cd $ROOT && TASK=$task RUN_DIR=$run_dir CHECKPOINT=$checkpoint GPUS=$GPUS bash scripts/run_dexjoco_failure_pipeline_from_baseline.sh"
      {
        echo "task=$task"
        echo "step=$step"
        echo "checkpoint=$checkpoint"
        echo "run_dir=$run_dir"
        echo "session=$session"
        echo "launched_at=$(date '+%F %T')"
      } > "$done_file"
      launched=1
      if [[ "$EXIT_AFTER_LAUNCH" == "1" ]]; then
        log "launched one pipeline; exiting watcher"
        exit 0
      fi
      break
    else
      log "no stable baseline checkpoint for task=$task; preserving task order"
      break
    fi
  done

  if (( launched == 0 )); then
    sleep "$POLL_SECONDS"
  fi
done
