#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DOWNLOAD_SESSION="${DOWNLOAD_SESSION:-fastwam_wan22_download}"
ACTION_DIT="${ACTION_DIT:-checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
WAN22_DIR="${WAN22_DIR:-checkpoints/Wan-AI/Wan2.2-TI2V-5B}"
PYTHON_BIN="${PYTHON_BIN:-/home/gzr1/miniconda3/envs/residual/bin/python}"
FREE_MB="${FREE_MB:-45000}"
MAX_UTIL="${MAX_UTIL:-20}"
POLL_SECONDS="${POLL_SECONDS:-180}"

echo "===== $(date) wait_preprocess_actiondit start ====="
echo "root=${ROOT_DIR}"
echo "download_session=${DOWNLOAD_SESSION}"
echo "action_dit=${ACTION_DIT}"
echo "wan22_dir=${WAN22_DIR}"
echo "free_mb=${FREE_MB} max_util=${MAX_UTIL} poll_seconds=${POLL_SECONDS}"

while tmux has-session -t "${DOWNLOAD_SESSION}" 2>/dev/null; do
  echo "$(date) waiting for ${DOWNLOAD_SESSION}"
  sleep "${POLL_SECONDS}"
done

if [[ -f "${ACTION_DIT}" ]]; then
  echo "$(date) ActionDiT already exists: ${ACTION_DIT}"
  ls -lh "${ACTION_DIT}"
  exit 0
fi

shopt -s nullglob
shards=("${WAN22_DIR}"/diffusion_pytorch_model-*.safetensors)
shopt -u nullglob
if (( ${#shards[@]} < 3 )); then
  echo "$(date) expected 3 Wan2.2 DiT shards, found ${#shards[@]}" >&2
  ls -lah "${WAN22_DIR}" || true
  exit 1
fi
ls -lh "${shards[@]}"

while true; do
  echo "$(date) gpu snapshot"
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits

  selected_gpu=""
  while IFS=, read -r idx free_mb util; do
    idx="${idx//[[:space:]]/}"
    free_mb="${free_mb//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    if [[ "${free_mb}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ ]]; then
      if (( free_mb >= FREE_MB && util <= MAX_UTIL )); then
        selected_gpu="${idx}"
        break
      fi
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)

  if [[ -n "${selected_gpu}" ]]; then
    echo "$(date) selected GPU ${selected_gpu}; preprocessing ${ACTION_DIT}"
    CUDA_VISIBLE_DEVICES="${selected_gpu}" "${PYTHON_BIN}" scripts/preprocess_action_dit_backbone.py \
      --model-config configs/model/fastwam.yaml \
      --output "${ACTION_DIT}" \
      --device cuda \
      --dtype bfloat16
    ls -lh "${ACTION_DIT}"
    echo "$(date) wait_preprocess_actiondit done"
    exit 0
  fi

  sleep "${POLL_SECONDS}"
done
