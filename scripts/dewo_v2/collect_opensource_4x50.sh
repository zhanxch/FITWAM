#!/usr/bin/env bash
# Opensource-aligned 4×50 collect (seeds 10086..10135 × 4 = 200).
#   TASK=water_plant GPUS=4,5,6,7 bash scripts/dewo_v2/collect_opensource_4x50.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_require_gpus
dewo_v2_load_task "${TASK}"

OPEN_REPO="${OPEN_REPO:-${ROOT_DIR}/../FastWAM-infer-in-DexJoco}"
OPEN_REPO="$(cd "${OPEN_REPO}" && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/data/${TASK}_opensource_s0_collect_4x50_${STAMP}}"

if [[ ! -f "${STATS}" ]]; then
  echo "[dewo-v2-collect] ERROR missing dataset_stats ${STATS}" >&2
  echo "  Run: python scripts/dewo_v2/export_opensource_artifacts.py --task ${TASK}" >&2
  exit 2
fi
if [[ ! -f "${TEXT_EMB}" ]]; then
  echo "[dewo-v2-collect] ERROR missing T5 ${TEXT_EMB}" >&2
  echo "  Run: python scripts/dewo_v2/export_opensource_artifacts.py --task ${TASK}" >&2
  exit 2
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "[dewo-v2-collect] ERROR missing ckpt ${CKPT}" >&2
  exit 2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FITWAM_ENV:-fastwam}"
ENV_PREFIX="${FITWAM_ENV_PREFIX:-$(conda info --base)/envs/fastwam}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${ROOT_DIR}/checkpoints"

mkdir -p "${OUTPUT_DIR}/logs"
echo "[dewo-v2-collect] task=${TASK} ckpt=${CKPT}"
echo "[dewo-v2-collect] output=${OUTPUT_DIR} gpus=${GPUS}"

overwrite_flag=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  overwrite_flag=(--overwrite)
fi

exec "${ENV_PREFIX}/bin/python" scripts/dexjoco/collect_opensource_4x50.py \
  --gpus "${GPUS}" \
  --seed-start "${SEED_START}" \
  --seed-end "${SEED_END}" \
  --repeats "${REPEATS}" \
  --max-steps "${MAX_STEPS}" \
  --action-horizon "${ACTION_HORIZON}" \
  --replan-steps "${REPLAN_STEPS}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --checkpoint "${CKPT}" \
  --model-config "${OPEN_REPO}/configs/fastwam_dexjoco.yaml" \
  --dataset-stats "${STATS}" \
  --text-embedding "${TEXT_EMB}" \
  --source-dataset "${SOURCE_DATASET}" \
  --output-dir "${OUTPUT_DIR}" \
  --task-name "${TASK}" \
  --success-prompt "${SUCCESS_PROMPT}" \
  "${overwrite_flag[@]}"
