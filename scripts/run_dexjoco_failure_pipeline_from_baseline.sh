#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/data_all/zhaoyc/Summer2/FastWAM_zhaoyc_failure_moved_from_share_20260703}
FAILURE_DATASET_ROOT=${FAILURE_DATASET_ROOT:-/data_all/share/dexjoco_failure_datasets}
PY=${PY:-/home/gzr1/miniconda3/envs/residual/bin/python}
TASK=${TASK:?Set TASK, e.g. click_mouse}
RUN_DIR=${RUN_DIR:?Set RUN_DIR to the success-only baseline run directory}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to the success-only baseline checkpoint}
GPUS=${GPUS:-4,5,6,7}
BASELINE_EPISODES=${BASELINE_EPISODES:-50}
BASELINE_SEED=${BASELINE_SEED:-0}
REPLAN_STEPS=${REPLAN_STEPS:-24}
MAX_ENV_STEPS=${MAX_ENV_STEPS:-600}
TARGET_FAILURES=${TARGET_FAILURES:-100}
COLLECT_TARGET_FAILURES=${COLLECT_TARGET_FAILURES:-100}
COLLECT_MAX_ATTEMPTS=${COLLECT_MAX_ATTEMPTS:-260}
COLLECT_REPLAN_STEPS=${COLLECT_REPLAN_STEPS:-24}
COLLECT_MAX_ENV_STEPS=${COLLECT_MAX_ENV_STEPS:-600}
COLLECT_SEEDS=${COLLECT_SEEDS:-"10000 20000 30000 40000"}
PORT_BASE=${PORT_BASE:-5600}
TRAIN_VARIANT=${TRAIN_VARIANT:-failure_embedding}
TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS:-6000}
TRAIN_SAVE_EVERY=${TRAIN_SAVE_EVERY:-500}
TRAIN_EVAL_EVERY=${TRAIN_EVAL_EVERY:-500}
TRAIN_EVAL_EPISODES=${TRAIN_EVAL_EPISODES:-50}
TRAIN_REQUIRE_FREE_MB=${TRAIN_REQUIRE_FREE_MB:-60000}
OVERWRITE_FAILURE_DATASETS=${OVERWRITE_FAILURE_DATASETS:-0}
RUN_ID=${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}

cd "$ROOT"
mkdir -p artifacts/logs artifacts/evals "$FAILURE_DATASET_ROOT"
export PATH="$(dirname "$PY")":/home/zhaoyc/.local/bin:/home/gzr1/miniconda3/bin:$PATH
export PYTHONPATH="$ROOT/src:$ROOT/scripts:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/checkpoints"
export PYTHONUNBUFFERED=1

LOG="artifacts/logs/pipeline_${TASK}_${RUN_ID}.log"
STATS_PATH="artifacts/dataset_stats/dexjoco_${TASK}_success_action_state.json"
CKPT_TAG="$(basename "$CHECKPOINT" .pt)"
BASELINE_OUT="artifacts/evals/${TASK}_baseline_${CKPT_TAG}_seed${BASELINE_SEED}_${BASELINE_EPISODES}ep_${RUN_ID}"
DEFAULT_DATASET="${FAILURE_DATASET_ROOT}/${TASK}_failure_fastwam_2cam_text"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

if [[ ! -d "$RUN_DIR" ]]; then
  log "missing RUN_DIR=$RUN_DIR"
  exit 10
fi
if [[ ! -s "$CHECKPOINT" ]]; then
  log "missing CHECKPOINT=$CHECKPOINT"
  exit 11
fi
if [[ ! -d "/data_all/share/datasets/dexjoco/dexjoco_lerobot_datasets/${TASK}" ]]; then
  log "missing success dataset for task=$TASK"
  exit 12
fi

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
read -r -a SEED_LIST <<< "$COLLECT_SEEDS"
if (( ${#GPU_LIST[@]} == 0 )); then
  log "empty GPUS=$GPUS"
  exit 13
fi

log "pipeline start task=$TASK run_dir=$RUN_DIR checkpoint=$CHECKPOINT gpus=$GPUS"
if [[ ! -f "$STATS_PATH" ]]; then
  log "computing success stats: $STATS_PATH"
  "$PY" scripts/compute_dexjoco_success_stats.py --tasks "$TASK" | tee -a "$LOG"
fi

log "baseline rollout output=$BASELINE_OUT"
"$PY" scripts/water_plant/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
  --gpus "$GPUS" \
  --run-dir "$RUN_DIR" \
  --checkpoint "$CHECKPOINT" \
  --dataset-stats-path "$ROOT/$STATS_PATH" \
  --no-load-text-encoder \
  --server-conda-env residual \
  --client-conda-env residual \
  --task-config-dir third_party/dexjoco/configs/rand_obj \
  --tasks "$TASK" \
  --episodes "$BASELINE_EPISODES" \
  --seed "$BASELINE_SEED" \
  --replan-steps "$REPLAN_STEPS" \
  --control-mode blocking \
  --max-env-steps "$MAX_ENV_STEPS" \
  --output-dir "$BASELINE_OUT" 2>&1 | tee -a "$LOG"

if [[ ! -f "$BASELINE_OUT/summary.json" ]]; then
  log "baseline summary missing: $BASELINE_OUT/summary.json"
  exit 20
fi

log "baseline summary"
"$PY" - "$BASELINE_OUT/summary.json" <<'PY' | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1]))
print(json.dumps({
    "episodes": d.get("total_episodes"),
    "successes": d.get("total_successes"),
    "success_rate": d.get("overall_success_rate"),
}, indent=2))
PY

sessions=()
for i in "${!GPU_LIST[@]}"; do
  if (( i >= ${#SEED_LIST[@]} )); then
    break
  fi
  gpu="${GPU_LIST[$i]}"
  seed="${SEED_LIST[$i]}"
  port=$((PORT_BASE + i))
  suffix=""
  if (( i > 0 )); then
    suffix="_s${seed}"
  fi
  output="${DEFAULT_DATASET}${suffix}"
  session="fastwam_${TASK}_failure_collect${suffix}"
  sessions+=("$session")
  if [[ -e "$output" && "$OVERWRITE_FAILURE_DATASETS" != "1" ]]; then
    log "refusing to overwrite existing failure dataset: $output"
    exit 30
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    log "collector session already exists: $session"
    continue
  fi
  collect_log="artifacts/logs/collect_${TASK}_${seed}_${RUN_ID}.log"
  log "starting collector session=$session gpu=$gpu seed=$seed output=$output"
  tmux new-session -d -s "$session" \
    "cd $ROOT && TASK=$TASK RUN_DIR=$RUN_DIR CHECKPOINT=$CHECKPOINT GPU=$gpu PORT=$port TARGET_FAILURES=$COLLECT_TARGET_FAILURES MAX_ATTEMPTS=$COLLECT_MAX_ATTEMPTS SEED=$seed REPLAN_STEPS=$COLLECT_REPLAN_STEPS MAX_ENV_STEPS=$COLLECT_MAX_ENV_STEPS OUTPUT_DATASET=$output bash scripts/run_dexjoco_collect_failures_once.sh 2>&1 | tee -a $collect_log"
done

if (( ${#sessions[@]} == 0 )); then
  log "no collector sessions configured"
  exit 31
fi

collect_sessions="${sessions[*]}"
log "watching collectors and autostarting train: sessions=$collect_sessions"
TARGET_TOTAL_FAILURES="$TARGET_FAILURES" \
CHECK_INTERVAL=120 \
TRAIN_GPUS="$GPUS" \
TRAIN_VARIANT="$TRAIN_VARIANT" \
TRAIN_MAX_STEPS="$TRAIN_MAX_STEPS" \
TRAIN_SAVE_EVERY="$TRAIN_SAVE_EVERY" \
TRAIN_EVAL_EVERY="$TRAIN_EVAL_EVERY" \
TRAIN_EVAL_EPISODES="$TRAIN_EVAL_EPISODES" \
TRAIN_REQUIRE_FREE_MB="$TRAIN_REQUIRE_FREE_MB" \
RESUME_CHECKPOINT="$CHECKPOINT" \
COLLECT_SESSIONS="$collect_sessions" \
bash scripts/watch_dexjoco_failures_then_train.sh 2>&1 | tee -a "$LOG"

log "pipeline handoff complete task=$TASK"
