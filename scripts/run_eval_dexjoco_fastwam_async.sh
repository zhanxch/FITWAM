#!/usr/bin/env bash
# Async / batched DexJoCo closed-loop eval for FastWAM (parallel episode workers).
#
# Prerequisite: start the ASYNC policy server (`fitwam` env), e.g.
#   CUDA_VISIBLE_DEVICES=7 python scripts/run_fastwam_server_async.py \
#     --run-dir runs/dexjoco_fold_glasses_dewo_v2/<timestamp> \
#     --checkpoint checkpoints/weights/step_002500.pt \
#     --dataset-stats-path /path/to/OPEN/artifacts/fold_glasses/dataset_stats.json \
#     --device cuda:0 --host 0.0.0.0 --port 5561
#
# Example: bimanual_assembly 50 episodes, 10 parallel workers + batch infer:
#   TASKS=bimanual_assembly EPISODES=50 BATCH_SIZE=10 bash scripts/run_eval_dexjoco_fastwam_async.sh
#
# Sync server on :5560 is NOT compatible with parallel workers (ZMQ REP).

set -euo pipefail

FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$FASTWAM_ROOT"

RUN_DIR="${RUN_DIR:?Set RUN_DIR to the FastWAM training run}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-5561}"
EPISODES="${EPISODES:-50}"
BATCH_SIZE="${BATCH_SIZE:-10}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-$BATCH_SIZE}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/dexjoco_fastwam_eval_async}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-1500}"
USE_BATCH_INFER="${USE_BATCH_INFER:-1}"
FITWAM_ENV="${FITWAM_ENV:-fitwam}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda not found"
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dexjoco

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}/scripts:${PYTHONPATH:-}"

python - <<'PY' >/dev/null 2>&1 || python -m pip install -q pyzmq msgpack
import msgpack
import zmq
PY

CACHE_DIR="${FASTWAM_ROOT}/data/text_embeds_cache/dexjoco_ego"
if ! ls "${CACHE_DIR}"/*.npz >/dev/null 2>&1; then
  echo "[run_eval_dexjoco_fastwam_async] exporting text caches to .npz (one-time)..."
  conda activate "${FITWAM_ENV}"
  python "${FASTWAM_ROOT}/scripts/export_text_embed_cache_npz.py" --cache-dir "${CACHE_DIR}"
  conda activate dexjoco
fi

ARGS=(
  --run-dir "$RUN_DIR"
  --policy-host "$POLICY_HOST"
  --policy-port "$POLICY_PORT"
  --episodes "$EPISODES"
  --batch-size "$BATCH_SIZE"
  --infer-batch-size "$INFER_BATCH_SIZE"
  --seed "$SEED"
  --output-dir "$OUTPUT_DIR"
  --max-env-steps "$MAX_ENV_STEPS"
)

if [[ -n "${TASKS:-}" ]]; then
  # shellcheck disable=SC2206
  TASK_ARR=($TASKS)
  ARGS+=(--tasks "${TASK_ARR[@]}")
fi

if [[ "${SAVE_VIDEO:-1}" == "0" ]]; then
  ARGS+=(--no-save-video)
fi

if [[ "${USE_BATCH_INFER}" == "0" ]]; then
  ARGS+=(--no-use-batch-infer)
fi

if [[ "${RANDOMIZE:-0}" == "1" ]]; then
  ARGS+=(--randomize)
fi

echo "[run_eval_dexjoco_fastwam_async] FASTWAM_ROOT=${FASTWAM_ROOT}"
echo "[run_eval_dexjoco_fastwam_async] policy=${POLICY_HOST}:${POLICY_PORT}"
echo "[run_eval_dexjoco_fastwam_async] episodes=${EPISODES} batch_size=${BATCH_SIZE} infer_batch_size=${INFER_BATCH_SIZE}"
echo "[run_eval_dexjoco_fastwam_async] conda env=dexjoco MUJOCO_GL=${MUJOCO_GL}"
echo

python "${FASTWAM_ROOT}/scripts/eval_dexjoco_fastwam_async.py" "${ARGS[@]}"
