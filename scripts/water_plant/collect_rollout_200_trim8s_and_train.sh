#!/usr/bin/env bash
# water_plant defaults for the generic DexJoCo rollout -> trim -> train flow.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export TASK_NAME="${TASK_NAME:-water_plant}"
export RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/water_plant_uncond_2cam_384_1e-4/2026-06-29_16-38-39}"
export CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/checkpoints/weights/step_006500.pt}"
export RAW_DATASET="${RAW_DATASET:-${ROOT_DIR}/data/water_plant_rollout_200_step6500_raw}"
export TRIMMED_DATASET="${TRIMMED_DATASET:-${ROOT_DIR}/data/water_plant_rollout_200_step6500_trim8s}"
export TEXT_CACHE="${TEXT_CACHE:-${ROOT_DIR}/data/text_embeds_cache/dexjoco_water_plant_rollout_text_failure}"
export LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/water_plant_rollout_200_step6500_trim8s}"
export TRAIN_TASK="${TRAIN_TASK:-dexjoco/dexjoco_water_plant_rollout_text_failure_2cam_proprio_1e-4}"
export TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-scripts/water_plant/train_2cam.sh}"

bash scripts/collect_rollout_trim_and_train.sh "$@"
