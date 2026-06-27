#!/usr/bin/env bash
# C3 diagnostic: train spray_water with rot6d dims UN-normalized (H2 fix).
#
# Clean A/B against scripts/train_spray_water_rot6d.sh (baseline: all 58 dims
# min/max). The only difference is the data config, which sets `norm_skip_dims`
# to leave rot6d identity while keeping min/max for xyz/hand.
#
# Requires the normalizer + processor patches:
#   src/fastwam/datasets/lerobot/utils/normalizer.py      (skip_dims)
#   src/fastwam/datasets/lerobot/processors/fastwam_processor.py (norm_skip_dims)
#
# Usage:
#   bash scripts/train_spray_water_rot6d_skip_rot6d.sh [hydra_overrides...]
#   # quick smoke test:
#   bash scripts/train_spray_water_rot6d_skip_rot6d.sh max_steps=200 save_every=100 eval_every=100
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)"

echo "[train_spray_water_skip_rot6d] C3: rot6d dims UN-normalized (H2 fix)"
echo "[train_spray_water_skip_rot6d] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} num_gpus=${NUM_GPUS}"

bash scripts/train_zero1.sh "${NUM_GPUS}" \
  task=spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4_skip_rot6d \
  "$@"
