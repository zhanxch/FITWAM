#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/data/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_DATASETS_CACHE}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
NUM_GPUS="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)"

echo "[train_2cam] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} num_gpus=${NUM_GPUS}"

bash scripts/train_zero1.sh "${NUM_GPUS}" \
  task=fold_glasses_uncond_2cam_384_1e-4 \
  "$@"
