#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"

export BASE_DATASET="${BASE_DATASET:-${ROOT_DIR}/data/water_plant_fastwam}"
export ROLLOUT_RAW="${ROLLOUT_RAW:-${ROOT_DIR}/data/water_plant_rollout_200_step6500_raw}"
export EVE_ROOT="${EVE_ROOT:-${BASE_DATASET}/eve}"
export EVE_MANIFEST_PATH="${EVE_MANIFEST_PATH:-${EVE_ROOT}/manifests/train_round1_success_plus_failure_events.json}"
export NORM_STATS_META_DIR="${NORM_STATS_META_DIR:-${BASE_DATASET}/meta}"
export TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:-${ROOT_DIR}/data/text_embeds_cache/dexjoco_water_plant_rollout_text_failure}"
export FASTWAM_RESUME="${FASTWAM_RESUME:?Set FASTWAM_RESUME to the shared success-only checkpoint}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)"

echo "[train_eve_round1] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} num_gpus=${NUM_GPUS}"
echo "[train_eve_round1] manifest=${EVE_MANIFEST_PATH}"

if [[ ! -f "${EVE_MANIFEST_PATH}" ]]; then
  echo "[train_eve_round1][ERROR] manifest not found: ${EVE_MANIFEST_PATH}" >&2
  exit 1
fi

bash scripts/train_zero1.sh "${NUM_GPUS}" \
  task=dexjoco/dexjoco_water_plant_eve_round1_failure_events_2cam_proprio_1e-4 \
  "$@"
