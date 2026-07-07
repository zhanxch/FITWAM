#!/usr/bin/env bash
# Generic DexJoCo rollout collection -> merge/trim dataset -> text embeds -> train.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TASK_NAME="${TASK_NAME:-water_plant}"
RUN_DIR="${RUN_DIR:?RUN_DIR must point to the base FastWAM training run directory}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT must point to the policy weight checkpoint}"
GPUS="${GPUS:-0,1,2,3}"
TOTAL_EPISODES="${TOTAL_EPISODES:-200}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-600}"
TRIM_FAILURE_SECONDS="${TRIM_FAILURE_SECONDS:-8}"
RAW_DATASET="${RAW_DATASET:-${ROOT_DIR}/data/${TASK_NAME}_rollout_200_step6500_raw}"
TRIMMED_DATASET="${TRIMMED_DATASET:-${ROOT_DIR}/data/${TASK_NAME}_rollout_200_step6500_trim8s}"
TEXT_CACHE="${TEXT_CACHE:-${ROOT_DIR}/data/text_embeds_cache/dexjoco_${TASK_NAME}_rollout_text_failure}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/${TASK_NAME}_rollout_200_step6500_trim8s}"
TRAIN_TASK="${TRAIN_TASK:-dexjoco/dexjoco_${TASK_NAME}_rollout_text_failure_2cam_proprio_1e-4}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-scripts/${TASK_NAME}/train_2cam.sh}"

mkdir -p "${LOG_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam

collect_args=(
  --gpus "${GPUS}"
  --run-dir "${RUN_DIR}"
  --checkpoint "${CHECKPOINT}"
  --no-load-text-encoder
  --tasks "${TASK_NAME}"
  --episodes "${TOTAL_EPISODES}"
  --replan-steps "${REPLAN_STEPS}"
  --max-env-steps "${MAX_ENV_STEPS}"
  --output-dir "${LOG_DIR}/collect"
  --raw-output-dataset "${RAW_DATASET}"
  --trimmed-output-dataset "${TRIMMED_DATASET}"
  --trim-failure-seconds "${TRIM_FAILURE_SECONDS}"
)

if [[ -n "${TASK_CONFIG_DIR:-}" ]]; then
  collect_args+=(--task-config-dir "${TASK_CONFIG_DIR}")
fi
if [[ -n "${SOURCE_DATASET:-}" ]]; then
  collect_args+=(--source-dataset "${SOURCE_DATASET}")
fi
if [[ -n "${SUCCESS_PROMPT:-}" ]]; then
  collect_args+=(--success-prompt "${SUCCESS_PROMPT}")
fi

python scripts/dexjoco_async/run_multi_gpu_dexjoco_collect.py "${collect_args[@]}" \
  2>&1 | tee "${LOG_DIR}/collect.log"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export FASTWAM_ROLLOUT_DATASET="${TRIMMED_DATASET}"
export FASTWAM_ROLLOUT_TEXT_CACHE="${TEXT_CACHE}"

python scripts/precompute_text_embeds.py \
  task="${TRAIN_TASK}" \
  2>&1 | tee "${LOG_DIR}/precompute_text_embeds.log"

WANDB_MODE="${WANDB_MODE:-offline}" \
bash "${TRAIN_LAUNCHER}" \
  task="${TRAIN_TASK}" \
  wandb.mode="${WANDB_MODE}" \
  "$@" \
  2>&1 | tee "${LOG_DIR}/train.log"
