#!/usr/bin/env bash
# Expert-only Eve sidecar + text embeds + VAE pre-encode for hammer_nail DEWOv2.
# Safe to run while recoverability scan is still going: does NOT write the pair
# env file (wait_then_train keys off that). Pair events are encoded later.
#
# VAE cache key is sample_id + window_start, so expert windows hit later when
# the pair mix reuses the same expert sample_ids.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FITWAM_ENV:-fastwam}"
ENV_PREFIX="${FITWAM_ENV_PREFIX:-$(conda info --base)/envs/fastwam}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${ROOT_DIR}/checkpoints"

OPEN_REPO="${OPEN_REPO:-${ROOT_DIR}/../FastWAM-infer-in-DexJoco}"
OPEN_REPO="$(cd "${OPEN_REPO}" && pwd)"
BASE_DATASET="${BASE_DATASET:-${ROOT_DIR}/data/dexjoco/dexjoco_lerobot_datasets/hammer_nail}"
SUCCESS_PROMPT="${SUCCESS_PROMPT:-Use the hammer to drive the nail into the wooden board.}"
CKPT="${CKPT:-${ROOT_DIR}/checkpoints/dexjoco/hammer_nail_fastwam/weights/step_002500.pt}"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="${OPEN_REPO}/checkpoints/hammer_nail/step_002500.pt"
fi
SOURCE_CONFIG="${SOURCE_CONFIG:-${OPEN_REPO}/configs/fastwam_dexjoco.yaml}"
PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS:-${OPEN_REPO}/artifacts/hammer_nail/dataset_stats.json}"
CFG_TASK_CONFIG_DIR="${CFG_TASK_CONFIG_DIR:-${ROOT_DIR}/configs/eval/dexjoco/hammer_nail_dewo_v2_cfg}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/data/hammer_nail_dewo_v2_pair_${STAMP}}"
TEXT_CACHE="${TEXT_CACHE:-${EXP_ROOT}/text_embeds_cache}"
EVE_ROOT="${EVE_ROOT:-${EXP_ROOT}/eve_v02}"
LOG_DIR="${LOG_DIR:-${EXP_ROOT}/logs}"
VAE_LATENT_CACHE_DIR="${VAE_LATENT_CACHE_DIR:-${EXP_ROOT}/vae_latent_cache}"
if [[ -z "${GPUS:-}" ]]; then
  echo "[hammer-nail-expert-vae] ERROR: set GPUS" >&2
  exit 2
fi
VAE_ENCODE_VAL="${VAE_ENCODE_VAL:-false}"
DEWO_TASK="${DEWO_TASK:-dexjoco/dexjoco_hammer_nail_offline_b1_jump_fast_full_1e-4_s0}"
mkdir -p "${EXP_ROOT}" "${LOG_DIR}" "${TEXT_CACHE}" "${VAE_LATENT_CACHE_DIR}" "${EXP_ROOT}/protocol"

log() { echo "[hammer-nail-expert-vae $(date -Is)] $*" | tee -a "${LOG_DIR}/prepare_expert_vae.log"; }

if [[ "${BASE_DATASET}" == *hammer_nail_fastwam* ]]; then
  BASE_DATASET="${ROOT_DIR}/data/dexjoco/dexjoco_lerobot_datasets/hammer_nail"
  log "aligned BASE_DATASET to opensource expert ${BASE_DATASET}"
fi
for required in "${PRETRAINED_NORM_STATS}" "${CKPT}" "${BASE_DATASET}"; do
  if [[ ! -e "${required}" ]]; then
    log "ERROR: missing ${required}"
    exit 2
  fi
done

log "stack=opensource_224_zscore expert-only VAE"
log "CKPT=${CKPT}"
log "BASE_DATASET=${BASE_DATASET}"
log "EXP_ROOT=${EXP_ROOT}"
log "VAE_GPUS=${GPUS}"

code_commit="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
mkdir -p "${EVE_ROOT}/splits" "${EVE_ROOT}/manifests" "${EVE_ROOT}/protocol"

if [[ ! -f "${EVE_ROOT}/splits/episode_splits.jsonl" ]]; then
  log "building expert episode splits"
  "${ENV_PREFIX}/bin/python" scripts/everobot/build_episode_split.py \
    --dataset "hammer_nail_expert_success=${BASE_DATASET}" \
    --force-success-dataset-id hammer_nail_expert_success \
    --val-fraction 0.2 \
    --seed 20260812 \
    --output "${EVE_ROOT}/splits/episode_splits.jsonl" \
    --report "${EVE_ROOT}/splits/episode_splits.report.json"
else
  log "reuse expert splits ${EVE_ROOT}/splits/episode_splits.jsonl"
fi

if [[ ! -f "${EVE_ROOT}/episode_meta.jsonl" ]]; then
  log "init-base expert success only"
  "${ENV_PREFIX}/bin/python" scripts/everobot/build_eve_sidecar.py init-base \
    --dataset-root "${BASE_DATASET}" \
    --dataset-id hammer_nail_expert_success \
    --eve-root "${EVE_ROOT}" \
    --task-name hammer_nail \
    --source-type expert_success \
    --source-policy expert \
    --collection-round -1 \
    --force-success \
    --split-map "${EVE_ROOT}/splits/episode_splits.jsonl" \
    --config-path "${SOURCE_CONFIG}" \
    --code-commit "${code_commit}"
else
  log "reuse episode_meta ${EVE_ROOT}/episode_meta.jsonl"
fi

expert_manifest="${EVE_ROOT}/manifests/offline_expert_success.json"
log "build expert-only train manifest"
"${ENV_PREFIX}/bin/python" scripts/everobot/build_eve_sidecar.py build-manifest \
  --eve-root "${EVE_ROOT}" \
  --manifest-name offline_expert_success \
  --include-outcomes success \
  --success-dataset-ids hammer_nail_expert_success \
  --success-sample-mode episode_only \
  --splits train

val_manifest="${EVE_ROOT}/manifests/offline_selection_primary_success.json"
log "build stride-1 val selection manifest (expert success)"
"${ENV_PREFIX}/bin/python" scripts/everobot/build_eve_sidecar.py build-manifest \
  --eve-root "${EVE_ROOT}" \
  --manifest-name offline_selection_primary_success \
  --include-outcomes success \
  --success-dataset-ids hammer_nail_expert_success \
  --success-sample-mode episode_only \
  --splits val

# Hydra data yaml requires ROLLOUT_RAW; expert-only uses the same expert root.
# Do NOT write offline_v1_b1_jump_fast.env — that would start the train waiter.
export FITWAM_ENV_PREFIX="${ENV_PREFIX}"
export OPEN_REPO="${OPEN_REPO}"
export BASE_DATASET="${BASE_DATASET}"
export ROLLOUT_RAW="${BASE_DATASET}"
export INIT_WEIGHTS="${CKPT}"
export SOURCE_CHECKPOINT="${CKPT}"
export FASTWAM_RESUME="${CKPT}"
export FASTWAM_SOURCE_CONFIG="${SOURCE_CONFIG}"
export PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS}"
export NORM_STATS_SOURCE=compute
export TEXT_EMBEDDING_CACHE_DIR="${TEXT_CACHE}"
export CFG_TASK_CONFIG_DIR="${CFG_TASK_CONFIG_DIR}"
export B1_VIDEO_EXPERIMENT_ROOT="${EXP_ROOT}"
export EVE_MANIFEST_PATH="${expert_manifest}"
export EVE_VAL_MANIFEST_PATH="${val_manifest}"
export VAE_LATENT_CACHE_DIR="${VAE_LATENT_CACHE_DIR}"
export REQUIRE_VAE_LATENT_CACHE=0
export DEWO_TASK="${DEWO_TASK}"

cat > "${EVE_ROOT}/protocol/offline_expert_success.env" <<EOF
# Expert-only prepare/VAE (pair env is written later by prepare_dewo_v2_pair_eve.sh)
export FITWAM_ENV_PREFIX=${ENV_PREFIX}
export OPEN_REPO=${OPEN_REPO}
export BASE_DATASET=${BASE_DATASET}
export ROLLOUT_RAW=${BASE_DATASET}
export INIT_WEIGHTS=${CKPT}
export SOURCE_CHECKPOINT=${CKPT}
export FASTWAM_RESUME=${CKPT}
export FASTWAM_SOURCE_CONFIG=${SOURCE_CONFIG}
export PRETRAINED_NORM_STATS=${PRETRAINED_NORM_STATS}
export TEXT_EMBEDDING_CACHE_DIR=${TEXT_CACHE}
export EVE_MANIFEST_PATH=${expert_manifest}
export EVE_VAL_MANIFEST_PATH=${val_manifest}
export B1_VIDEO_EXPERIMENT_ROOT=${EXP_ROOT}
export VAE_LATENT_CACHE_DIR=${VAE_LATENT_CACHE_DIR}
export REQUIRE_VAE_LATENT_CACHE=0
export DEWO_TASK=${DEWO_TASK}
EOF
log "wrote ${EVE_ROOT}/protocol/offline_expert_success.env"

log "precomputing base + outcome text embeds"
"${ENV_PREFIX}/bin/python" scripts/precompute_text_embeds.py \
  "task=${DEWO_TASK}" \
  2>&1 | tee -a "${LOG_DIR}/precompute_text_embeds_expert.log"

IFS=',' read -r -a gpu_arr <<< "${GPUS}"
nproc="${#gpu_arr[@]}"
if [[ "${nproc}" -lt 1 ]]; then
  log "ERROR: empty GPUS=${GPUS}"
  exit 2
fi

log "VAE pre-encode expert windows on GPUs ${GPUS} encode_val=${VAE_ENCODE_VAL}"
pids=()
if [[ "${nproc}" -gt 1 ]]; then
  world="${nproc}"
  for gpu_i in "${!gpu_arr[@]}"; do
    gpu_id="${gpu_arr[$gpu_i]}"
    shard_rank="${gpu_i}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      export WORLD_SIZE=1
      unset RANK LOCAL_RANK MASTER_ADDR MASTER_PORT GROUP_RANK LOCAL_WORLD_SIZE
      "${ENV_PREFIX}/bin/python" scripts/precompute_vae_latents.py \
        "task=${DEWO_TASK}" \
        "+vae_latent_cache_dir=${VAE_LATENT_CACHE_DIR}" \
        "+vae_shard_rank=${shard_rank}" \
        "+vae_shard_world=${world}" \
        "+encode_val=${VAE_ENCODE_VAL}" \
        2>&1 | tee -a "${LOG_DIR}/precompute_vae_latents_expert.shard${shard_rank}.log"
    ) &
    pids+=("$!")
  done
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      fail=1
    fi
  done
  if [[ "${fail}" -ne 0 ]]; then
    log "ERROR: one or more expert VAE shard workers failed"
    exit 2
  fi
else
  export CUDA_VISIBLE_DEVICES="${gpu_arr[0]}"
  "${ENV_PREFIX}/bin/python" scripts/precompute_vae_latents.py \
    "task=${DEWO_TASK}" \
    "+vae_latent_cache_dir=${VAE_LATENT_CACHE_DIR}" \
    "+encode_val=${VAE_ENCODE_VAL}" \
    2>&1 | tee -a "${LOG_DIR}/precompute_vae_latents_expert.log"
fi

n_vae="$(find "${VAE_LATENT_CACHE_DIR}" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
log "DONE expert VAE files=${n_vae} cache=${VAE_LATENT_CACHE_DIR}"
log "pair mix will reuse this cache; remaining pair windows encode after scan/materialize"
