#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/spray_water_rot6d_rosbag_ts_filter"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

GPUS=(0 1 2 3)
MEM_THRESHOLD_MB=5000
POLL_SEC=60

echo "[wait_and_train] waiting for GPUs ${GPUS[*]} to have < ${MEM_THRESHOLD_MB} MiB used..." | tee -a "${LOG_FILE}"

while true; do
  all_free=true
  for gpu in "${GPUS[@]}"; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}")
    if (( used > MEM_THRESHOLD_MB )); then
      all_free=false
      echo "[wait_and_train] GPU ${gpu} still busy (${used} MiB used)" | tee -a "${LOG_FILE}"
      break
    fi
  done
  if ${all_free}; then
    echo "[wait_and_train] all target GPUs are free, starting training..." | tee -a "${LOG_FILE}"
    break
  fi
  sleep "${POLL_SEC}"
done

cd "${ROOT_DIR}"
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_spray_water_rot6d.sh 2>&1 | tee -a "${LOG_FILE}"
