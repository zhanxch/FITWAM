#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/water_plant_rollout_200_step6500_trim8s}"
mkdir -p "${LOG_DIR}"

nohup bash scripts/water_plant/collect_rollout_200_trim8s_and_train.sh \
  > "${LOG_DIR}/launcher.nohup.log" 2>&1 &
pid="$!"
printf "%s\n" "${pid}" > "${LOG_DIR}/launcher.pid"
echo "[start] pid=${pid}"
echo "[start] log=${LOG_DIR}/launcher.nohup.log"
