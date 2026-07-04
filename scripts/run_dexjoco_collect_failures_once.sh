#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/data_all/zhaoyc/Summer2/FastWAM_zhaoyc_failure_moved_from_share_20260703}
FAILURE_DATASET_ROOT=${FAILURE_DATASET_ROOT:-/data_all/share/dexjoco_failure_datasets}
PY=${PY:-/home/gzr1/miniconda3/envs/residual/bin/python}
TASK=${TASK:?Set TASK, e.g. TASK=hammer_nail}
RUN_DIR=${RUN_DIR:?Set RUN_DIR to the base policy training run directory}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to a FastWAM .pt checkpoint}
GPU=${GPU:-0}
PORT=${PORT:-5560}
TARGET_FAILURES=${TARGET_FAILURES:-100}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-260}
SEED=${SEED:-10000}
REPLAN_STEPS=${REPLAN_STEPS:-24}
MAX_ENV_STEPS=${MAX_ENV_STEPS:-600}
OUTPUT_DATASET=${OUTPUT_DATASET:-${FAILURE_DATASET_ROOT}/${TASK}_failure_fastwam_2cam_text}
DATASET_STATS_PATH=${DATASET_STATS_PATH:-${ROOT}/artifacts/dataset_stats/dexjoco_${TASK}_success_action_state.json}
LOGDIR=${ROOT}/artifacts/logs
RUN_ID=${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}
SERVER_LOG=${LOGDIR}/collect_${TASK}_${RUN_ID}_server.log
COLLECT_LOG=${LOGDIR}/collect_${TASK}_${RUN_ID}.log

cd "$ROOT"
mkdir -p "$LOGDIR"
export PYTHONPATH="$ROOT/src:$ROOT/scripts:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="$ROOT/checkpoints"
export PYTHONUNBUFFERED=1

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(date '+%F %T')] starting policy server task=$TASK gpu=$GPU port=$PORT" | tee -a "$COLLECT_LOG"
server_args=(
  --device cuda:0
  --host 0.0.0.0
  --port "$PORT"
  --run-dir "$RUN_DIR"
  --checkpoint "$CHECKPOINT"
  --no-load-text-encoder
)
if [[ -f "$DATASET_STATS_PATH" ]]; then
  server_args+=(--dataset-stats-path "$DATASET_STATS_PATH")
fi
CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/water_plant/dexjoco_async/run_fastwam_server_async.py \
  "${server_args[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + 900))
until grep -q "Async server ready" "$SERVER_LOG" 2>/dev/null; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Policy server exited before ready. Log:" | tee -a "$COLLECT_LOG"
    tail -n 160 "$SERVER_LOG" | tee -a "$COLLECT_LOG"
    exit 20
  fi
  if (( SECONDS > deadline )); then
    echo "Timed out waiting for policy server. Log:" | tee -a "$COLLECT_LOG"
    tail -n 160 "$SERVER_LOG" | tee -a "$COLLECT_LOG"
    exit 21
  fi
  sleep 5
done

echo "[$(date '+%F %T')] collecting failures output=$OUTPUT_DATASET" | tee -a "$COLLECT_LOG"
"$PY" scripts/collect_dexjoco_water_plant_failures.py \
  --task-name "$TASK" \
  --run-dir "$RUN_DIR" \
  --policy-host 127.0.0.1 \
  --policy-port "$PORT" \
  --target-failures "$TARGET_FAILURES" \
  --max-attempts "$MAX_ATTEMPTS" \
  --seed "$SEED" \
  --replan-steps "$REPLAN_STEPS" \
  --max-env-steps "$MAX_ENV_STEPS" \
  --output-dataset "$OUTPUT_DATASET" \
  --overwrite >> "$COLLECT_LOG" 2>&1

echo "[$(date '+%F %T')] done collect task=$TASK output=$OUTPUT_DATASET" | tee -a "$COLLECT_LOG"
