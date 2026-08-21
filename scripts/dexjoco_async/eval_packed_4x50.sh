#!/usr/bin/env bash
# Packed local-async 4×50 helper. DexJoCo / DEWO v2 official eval is
# scripts/dexjoco/eval_opensource_4x50.sh (224 / z-score).
#
#   GPUS=0,1,2,3 RUN_DIR=... CKPT=... TASK=water_plant \
#     bash scripts/dexjoco_async/eval_packed_4x50.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ -z "${GPUS:-}" ]]; then
  echo "[packed-4x50] ERROR: set GPUS" >&2
  exit 2
fi
RUN_DIR="${RUN_DIR:?Set RUN_DIR}"
CKPT="${CKPT:?Set CKPT}"
TASK="${TASK:?Set TASK}"
TASK_CONFIG_DIR="${TASK_CONFIG_DIR:?Set TASK_CONFIG_DIR}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/evaluate_results/dexjoco/${TASK}_packed_4x50_$(date +%Y%m%d_%H%M%S)}"
SERVERS_PER_GPU="${SERVERS_PER_GPU:-4}"
REPLAN_STEPS="${REPLAN_STEPS:-25}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-1500}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FITWAM_ENV:-fastwam}"
ENV_PREFIX="${FITWAM_ENV_PREFIX:-$(conda info --base)/envs/${FITWAM_ENV:-fastwam}}"

extra=()
if [[ -n "${NORM_STATS_META_DIR:-}" ]]; then
  extra+=(--norm-stats-meta-dir "${NORM_STATS_META_DIR}")
fi
if [[ -n "${DATASET_STATS_PATH:-}" ]]; then
  extra+=(--dataset-stats-path "${DATASET_STATS_PATH}")
fi
if [[ -n "${TEXT_EMBEDDING_CACHE_DIR:-}" ]]; then
  extra+=(--text-embedding-cache-dir "${TEXT_EMBEDDING_CACHE_DIR}")
fi
if [[ "${NO_LOAD_TEXT_ENCODER:-1}" == "1" ]]; then
  extra+=(--no-load-text-encoder)
fi

exec "${ENV_PREFIX}/bin/python" scripts/dexjoco_async/run_packed_4x50_eval.py \
  --gpus "${GPUS}" \
  --servers-per-gpu "${SERVERS_PER_GPU}" \
  --run-dir "${RUN_DIR}" \
  --checkpoint "${CKPT}" \
  --tasks "${TASK}" \
  --task-config-dir "${TASK_CONFIG_DIR}" \
  --replan-steps "${REPLAN_STEPS}" \
  --control-mode blocking \
  --max-env-steps "${MAX_ENV_STEPS}" \
  --no-randomize \
  --no-randomize-dynamics \
  "${extra[@]}" \
  --output-dir "${OUT_ROOT}" \
  "$@"
