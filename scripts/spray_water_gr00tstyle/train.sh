#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FITWAM_ENV:-fitwam}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)"

echo "[spray_water_gr00tstyle] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} num_gpus=${NUM_GPUS}"

bash scripts/train_zero1.sh "${NUM_GPUS}" \
  task=spray_water_rot6d_gr00tstyle_uncond_3cam_384_1e-4 \
  "$@"
