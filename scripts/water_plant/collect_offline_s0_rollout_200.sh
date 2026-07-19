#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

S0_RUN_DIR="${S0_RUN_DIR:?Set S0_RUN_DIR to a readable directory containing config.yaml}"
S0_CHECKPOINT="${S0_CHECKPOINT:?Set S0_CHECKPOINT to the frozen S0 .pt file}"
COLLECTION_ROOT="${COLLECTION_ROOT:?Set COLLECTION_ROOT to a new private output directory}"

ENV_PREFIX="${FITWAM_ENV_PREFIX:-${CONDA_PREFIX:-}}"
DEXJOCO_ROOT="${DEXJOCO_ROOT:-${ROOT_DIR}/third_party/dexjoco}"
SOURCE_DATASET="${SOURCE_DATASET:-${ROOT_DIR}/data/dexjoco/dexjoco_lerobot_datasets/water_plant}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-}"
NORM_STATS_META_DIR="${NORM_STATS_META_DIR:-}"
TEXT_CACHE_DIR="${TEXT_CACHE_DIR:-}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
BASE_SEED="${BASE_SEED:-20260718}"
EPISODES="${EPISODES:-200}"
REPLAN_STEPS="${REPLAN_STEPS:-25}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-1500}"
RESUME_COLLECTION="${RESUME_COLLECTION:-0}"
EXPECTED_S0_CHECKPOINT_NAME="${EXPECTED_S0_CHECKPOINT_NAME:-step_006500.pt}"

if [[ -z "${ENV_PREFIX}" ]]; then
  echo \
    "Set FITWAM_ENV_PREFIX to a conda environment containing FITWAM dependencies." \
    >&2
  exit 2
fi
if [[ "${GPU_IDS}" != "0,1,2,3" ]]; then
  echo "Formal S0 rollout requires CUDA_VISIBLE_DEVICES=0,1,2,3." >&2
  exit 2
fi
if [[ "${EPISODES}" != "200" || "${REPLAN_STEPS}" != "25" || \
      "${MAX_ENV_STEPS}" != "1500" ]]; then
  echo "Formal rollout protocol is fixed at 200 episodes, replan 25, max-env 1500." >&2
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
    echo "Missing or unreadable formal rollout input: ${required_file}" >&2
    exit 2
  fi
done

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

SHARD_ROOT="${COLLECTION_ROOT}/shards"
RAW_DATASET="${COLLECTION_ROOT}/water_plant_s0_rollout_200"
VALIDATION_REPORT="${COLLECTION_ROOT}/outcome_validation.json"
FORMAL_VALIDATION_REPORT="${COLLECTION_ROOT}/formal_protocol_validation.json"
PROTOCOL_PATH="${COLLECTION_ROOT}/collection_protocol.json"
if [[ "${RESUME_COLLECTION}" == "1" ]]; then
  if [[ ! -d "${SHARD_ROOT}" || ! -s "${PROTOCOL_PATH}" ]]; then
    echo "Resume requires existing shards and collection_protocol.json." >&2
    exit 2
  fi
  collection_mode=(--resume)
  protocol_mode=(--resume)
elif [[ "${RESUME_COLLECTION}" == "0" ]]; then
  if [[ -e "${COLLECTION_ROOT}" ]]; then
    echo \
      "Refusing to overwrite an existing formal collection: ${COLLECTION_ROOT}" \
      >&2
    exit 2
  fi
  collection_mode=(--no-overwrite)
  protocol_mode=()
else
  echo "RESUME_COLLECTION must be 0 or 1." >&2
  exit 2
fi
mkdir -p "${COLLECTION_ROOT}"

text_cache_args=()
text_cache_collect_args=()
if [[ -n "${TEXT_CACHE_DIR}" ]]; then
  if [[ ! -d "${TEXT_CACHE_DIR}" ]]; then
    echo "Missing text embedding cache directory: ${TEXT_CACHE_DIR}" >&2
    exit 2
  fi
  text_cache_args=(--text-cache-dir "${TEXT_CACHE_DIR}")
  text_cache_collect_args=(--text-embedding-cache-dir "${TEXT_CACHE_DIR}")
fi

"${ENV_PREFIX}/bin/python" scripts/water_plant/validate_s0_rollout_inputs.py \
  --run-dir "${S0_RUN_DIR}" \
  --checkpoint "${S0_CHECKPOINT}" \
  "${normalization_protocol_args[@]}" \
  "${text_cache_args[@]}" \
  --source-dataset "${SOURCE_DATASET}" \
  --base-checkpoints-dir "${ROOT_DIR}/checkpoints" \
  --dexjoco-root "${DEXJOCO_ROOT}" \
  --protocol-out "${PROTOCOL_PATH}" \
  --expected-checkpoint-name "${EXPECTED_S0_CHECKPOINT_NAME}" \
  --collection-kind formal \
  --episodes "${EPISODES}" \
  --base-seed "${BASE_SEED}" \
  --gpus "${GPU_IDS}" \
  --replan-steps "${REPLAN_STEPS}" \
  --max-env-steps "${MAX_ENV_STEPS}" \
  --video-fps 30 \
  --outcome-task-mode clean \
  "${protocol_mode[@]}"

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

"${ENV_PREFIX}/bin/python" scripts/dexjoco_async/run_multi_gpu_dexjoco_collect.py \
  --gpus "${GPU_IDS}" \
  --episodes "${EPISODES}" \
  --seed "${BASE_SEED}" \
  --server-conda-env "${ENV_PREFIX}" \
  --client-conda-env "${ENV_PREFIX}" \
  --run-dir "${S0_RUN_DIR}" \
  --checkpoint "${S0_CHECKPOINT}" \
  "${normalization_server_args[@]}" \
  "${text_cache_collect_args[@]}" \
  --no-load-text-encoder \
  --task-config-dir "${DEXJOCO_ROOT}/configs/rand_obj" \
  --tasks water_plant \
  --source-dataset "${SOURCE_DATASET}" \
  --dexjoco-py-root "${DEXJOCO_ROOT}/dexjoco" \
  --replan-steps "${REPLAN_STEPS}" \
  --max-env-steps "${MAX_ENV_STEPS}" \
  --video-fps 30 \
  --no-randomize \
  --no-randomize-dynamics \
  --no-action-clip \
  --outcome-task-mode clean \
  --output-dir "${SHARD_ROOT}" \
  --raw-output-dataset "${RAW_DATASET}" \
  "${collection_mode[@]}"

"${ENV_PREFIX}/bin/python" scripts/build_rollout_datasets.py validate-outcomes \
  --dataset "${RAW_DATASET}" \
  --expected-episodes "${EPISODES}" \
  --check-media \
  --report "${VALIDATION_REPORT}"

"${ENV_PREFIX}/bin/python" scripts/water_plant/validate_s0_formal_outputs.py \
  --protocol "${PROTOCOL_PATH}" \
  --dataset "${RAW_DATASET}" \
  --outcome-validation "${VALIDATION_REPORT}" \
  --report "${FORMAL_VALIDATION_REPORT}"

echo "[formal-rollout] protocol=${PROTOCOL_PATH}"
echo "[formal-rollout] dataset=${RAW_DATASET}"
echo "[formal-rollout] validation=${VALIDATION_REPORT}"
echo "[formal-rollout] protocol_validation=${FORMAL_VALIDATION_REPORT}"
