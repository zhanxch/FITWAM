#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

S0_RUN_DIR="${S0_RUN_DIR:?Set S0_RUN_DIR to the verified S0 run directory}"
S0_CHECKPOINT="${S0_CHECKPOINT:?Set S0_CHECKPOINT to step_006500.pt}"
SANITY_ROOT="${SANITY_ROOT:?Set SANITY_ROOT to a new private output directory}"

ENV_PREFIX="${FITWAM_ENV_PREFIX:-${CONDA_PREFIX:-}}"
DEXJOCO_ROOT="${DEXJOCO_ROOT:-${ROOT_DIR}/third_party/dexjoco}"
SOURCE_DATASET="${SOURCE_DATASET:-${ROOT_DIR}/data/dexjoco/dexjoco_lerobot_datasets/water_plant}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-}"
NORM_STATS_META_DIR="${NORM_STATS_META_DIR:-}"
TEXT_CACHE_DIR="${TEXT_CACHE_DIR:-}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
BASE_SEED="${BASE_SEED:-20260718}"
EXPECTED_S0_CHECKPOINT_NAME="${EXPECTED_S0_CHECKPOINT_NAME:-step_006500.pt}"

if [[ -z "${ENV_PREFIX}" ]]; then
  echo \
    "Set FITWAM_ENV_PREFIX to a conda environment containing FITWAM dependencies." \
    >&2
  exit 2
fi
if [[ "${GPU_IDS}" != "0,1,2,3" ]]; then
  echo "S0 sanity rollout requires CUDA_VISIBLE_DEVICES=0,1,2,3." >&2
  exit 2
fi

for required_file in \
  "${S0_RUN_DIR}/config.yaml" \
  "${S0_CHECKPOINT}" \
  "${SOURCE_DATASET}/meta/info.json" \
  "${SOURCE_DATASET}/meta/episodes.jsonl" \
  "${DEXJOCO_ROOT}/configs/rand_obj/water_plant.yaml" \
  "${DEXJOCO_ROOT}/dexjoco/dexjoco/tasks/mappings.py" \
  "${ENV_PREFIX}/bin/python"; do
  if [[ ! -r "${required_file}" || ! -s "${required_file}" ]]; then
    echo "Missing or unreadable S0 sanity input: ${required_file}" >&2
    exit 2
  fi
done

if [[ -e "${SANITY_ROOT}" ]]; then
  echo "Refusing to overwrite S0 sanity output: ${SANITY_ROOT}" >&2
  exit 2
fi
mkdir -p "${SANITY_ROOT}"

CONDA_BASE="$(cd "${ENV_PREFIX}/../.." && pwd)"
if [[ ! -x "${CONDA_BASE}/bin/conda" ]]; then
  echo "Could not find conda for ${ENV_PREFIX}: ${CONDA_BASE}/bin/conda" >&2
  exit 2
fi
export PATH="${CONDA_BASE}/bin:${HOME}/.local/bin:${PATH}"

NORM_STATS_SOURCE="$(
  "${ENV_PREFIX}/bin/python" -c \
    'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(str(c["data"]["train"]["processor"].get("norm_stats_source", "compute")).strip().lower() or "compute")' \
    "${S0_RUN_DIR}/config.yaml"
)"
normalization_protocol_args=()
if [[ "${NORM_STATS_SOURCE}" == "meta" ]]; then
  if [[ -n "${DATASET_STATS_PATH}" ]]; then
    echo "norm_stats_source=meta forbids DATASET_STATS_PATH." >&2
    exit 2
  fi
  if [[ -z "${NORM_STATS_META_DIR}" ]]; then
    echo "norm_stats_source=meta requires NORM_STATS_META_DIR." >&2
    exit 2
  fi
  normalization_protocol_args=(--norm-stats-meta-dir "${NORM_STATS_META_DIR}")
else
  if [[ -n "${NORM_STATS_META_DIR}" ]]; then
    echo "norm_stats_source=${NORM_STATS_SOURCE} forbids NORM_STATS_META_DIR." >&2
    exit 2
  fi
  if [[ -z "${DATASET_STATS_PATH}" ]]; then
    echo "norm_stats_source=${NORM_STATS_SOURCE} requires DATASET_STATS_PATH." >&2
    exit 2
  fi
  if [[ ! -r "${DATASET_STATS_PATH}" || ! -s "${DATASET_STATS_PATH}" ]]; then
    echo "Missing or unreadable dataset stats: ${DATASET_STATS_PATH}" >&2
    exit 2
  fi
  normalization_protocol_args=(--dataset-stats "${DATASET_STATS_PATH}")
fi

text_cache_protocol_args=()
text_cache_eval_args=()
if [[ -n "${TEXT_CACHE_DIR}" ]]; then
  if [[ ! -d "${TEXT_CACHE_DIR}" ]]; then
    echo "Missing text embedding cache directory: ${TEXT_CACHE_DIR}" >&2
    exit 2
  fi
  text_cache_protocol_args=(--text-cache-dir "${TEXT_CACHE_DIR}")
  text_cache_eval_args=(--text-embedding-cache-dir "${TEXT_CACHE_DIR}")
fi

PROTOCOL_PATH="${SANITY_ROOT}/sanity_protocol.json"
EVAL_ROOT="${SANITY_ROOT}/eval"

"${ENV_PREFIX}/bin/python" scripts/water_plant/validate_s0_rollout_inputs.py \
  --run-dir "${S0_RUN_DIR}" \
  --checkpoint "${S0_CHECKPOINT}" \
  "${normalization_protocol_args[@]}" \
  "${text_cache_protocol_args[@]}" \
  --source-dataset "${SOURCE_DATASET}" \
  --base-checkpoints-dir "${ROOT_DIR}/checkpoints" \
  --dexjoco-root "${DEXJOCO_ROOT}" \
  --protocol-out "${PROTOCOL_PATH}" \
  --expected-checkpoint-name "${EXPECTED_S0_CHECKPOINT_NAME}" \
  --collection-kind sanity \
  --episodes 4 \
  --base-seed "${BASE_SEED}" \
  --gpus "${GPU_IDS}" \
  --replan-steps 25 \
  --max-env-steps 1500 \
  --video-fps 30 \
  --outcome-task-mode clean

normalization_server_args=()
if [[ "${NORM_STATS_SOURCE}" == "meta" ]]; then
  EFFECTIVE_NORM_STATS_META_DIR="$(
    "${ENV_PREFIX}/bin/python" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["model"]["normalization"]["meta_dir"])' \
      "${PROTOCOL_PATH}"
  )"
  normalization_server_args=(--norm-stats-meta-dir "${EFFECTIVE_NORM_STATS_META_DIR}")
else
  normalization_server_args=(--dataset-stats-path "${DATASET_STATS_PATH}")
fi

"${ENV_PREFIX}/bin/python" scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
  --gpus "${GPU_IDS}" \
  --episodes 4 \
  --seed "${BASE_SEED}" \
  --server-conda-env "${ENV_PREFIX}" \
  --client-conda-env "${ENV_PREFIX}" \
  --run-dir "${S0_RUN_DIR}" \
  --checkpoint "${S0_CHECKPOINT}" \
  "${normalization_server_args[@]}" \
  "${text_cache_eval_args[@]}" \
  --no-load-text-encoder \
  --task-config-dir "${DEXJOCO_ROOT}/configs/rand_obj" \
  --tasks water_plant \
  --dexjoco-py-root "${DEXJOCO_ROOT}/dexjoco" \
  --replan-steps 25 \
  --control-mode blocking \
  --max-env-steps 1500 \
  --video-fps 30 \
  --no-randomize \
  --no-randomize-dynamics \
  --save-video \
  --save-actions \
  --no-action-clip \
  --output-dir "${EVAL_ROOT}"

SANITY_VALIDATION="${SANITY_ROOT}/sanity_validation.json"
"${ENV_PREFIX}/bin/python" scripts/water_plant/validate_s0_sanity_outputs.py \
  --summary "${EVAL_ROOT}/summary.json" \
  --protocol "${PROTOCOL_PATH}" \
  --report "${SANITY_VALIDATION}" \
  --expected-episodes 4

echo "[s0-sanity] protocol=${PROTOCOL_PATH}"
echo "[s0-sanity] summary=${EVAL_ROOT}/summary.json"
echo "[s0-sanity] validation=${SANITY_VALIDATION}"
