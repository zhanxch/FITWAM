#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)"

echo "[train_spray_water] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} num_gpus=${NUM_GPUS}"

bash scripts/train_zero1.sh "${NUM_GPUS}" \
  task=spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4 \
  "$@"
