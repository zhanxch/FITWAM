#!/usr/bin/env bash
# Prepare Eve manifests + text/VAE caches for DEWO v9 recoverability pairs.
#   TASK=fold_glasses PAIR_DATASET=... EXP_ROOT=... GPUS=4,5,6,7 \
#     bash scripts/dewo_v2/prepare_pair_eve.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
CALLER_DEWO_TASK="${DEWO_TASK:-}"
dewo_v2_load_task "${TASK}"
export DEWO_TASK="${CALLER_DEWO_TASK:-dexjoco/dexjoco_dewo_v9_offline_b1_jump_fast_uncond}"
# In-process text-embed recipe for v9. Do not persist CFG into the protocol env.
export CFG_SUCCESS_SUFFIX=' Successful execution.'
export CFG_FAILURE_SUFFIX=' Failed execution.'
export CFG_PRIMARY_OUTCOME=0.9
export CFG_PRIMARY_FAST=0.0
export CFG_PRIMARY_BASE=0.1
export CFG_AUX_SUCCESS_OUTCOME=1.0
export CFG_AUX_SUCCESS_FAST=0.0
export CFG_AUX_SUCCESS_BASE=0.0
export CFG_AUX_FAIL_OUTCOME=1.0
export CFG_AUX_FAIL_FAST=0.0
export CFG_AUX_FAIL_BASE=0.0
unset CFG_PRIMARY CFG_AUX_SUCCESS CFG_AUX_FAIL || true

PRIMARY_KIND="${PRIMARY_KIND:-expert}"
PRIMARY_N="${PRIMARY_N:-15}"
PRIMARY_SEED="${PRIMARY_SEED:-20260820}"
if [[ "${PRIMARY_KIND}" == "success_rollouts" || "${PRIMARY_KIND}" == "all_success_seeds" ]]; then
  PRIMARY_DATASET="${PRIMARY_DATASET:-${ROLLOUT_RAW:-}}"
  if [[ -z "${PRIMARY_DATASET}" ]]; then
    echo "[dewo-v2-pair] ERROR: PRIMARY_KIND=success_rollouts needs PRIMARY_DATASET or ROLLOUT_RAW" >&2
    exit 2
  fi
  BASE_DATASET="${PRIMARY_DATASET}"
  export BASE_DATASET PRIMARY_DATASET
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FITWAM_ENV:-fastwam}"
ENV_PREFIX="${FITWAM_ENV_PREFIX:-$(conda info --base)/envs/fastwam}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"

OPEN_REPO="${OPEN_REPO:-${ROOT_DIR}/../FastWAM-infer-in-DexJoco}"
OPEN_REPO="$(cd "${OPEN_REPO}" && pwd)"
PAIR_DATASET="${PAIR_DATASET:?Set PAIR_DATASET to the materialized pair LeRobot dataset}"
dewo_v2_assert_path_for_task PAIR_DATASET "${PAIR_DATASET}"
SOURCE_CONFIG="${SOURCE_CONFIG:-${OPEN_REPO}/configs/fastwam_dexjoco.yaml}"
TEXT_CACHE="${TEXT_CACHE:-${ROOT_DIR}/data/text_embeds_cache/${TASK}_dewo_v9_pair}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/data/${TASK}_dewo_v9_pair_${STAMP}}"
dewo_v2_assert_path_for_task EXP_ROOT "${EXP_ROOT}"
EVE_ROOT="${EVE_ROOT:-${EXP_ROOT}/eve_v02}"
LOG_DIR="${LOG_DIR:-${EXP_ROOT}/logs}"
VAE_LATENT_CACHE_DIR="${VAE_LATENT_CACHE_DIR:-${EXP_ROOT}/vae_latent_cache}"
EVAL_CFG_DIR="${EXP_ROOT}/eval_task_cfg"
mkdir -p "${EXP_ROOT}" "${LOG_DIR}" "${TEXT_CACHE}" "${VAE_LATENT_CACHE_DIR}" "${EXP_ROOT}/protocol" "${EVAL_CFG_DIR}"

log() { echo "[dewo-v2-pair ${TASK} $(date -Is)] $*" | tee -a "${LOG_DIR}/prepare.log"; }

dewo_v2_align_opensource_stack
for required in "${PRETRAINED_NORM_STATS}" "${CKPT}" "${PAIR_DATASET}/pair_index.json"; do
  if [[ ! -e "${required}" ]]; then
    log "ERROR: missing ${required}"
    exit 2
  fi
done

python "${ROOT_DIR}/scripts/dewo_v2/tasks.py" dump-cfg-json --task "${TASK}" \
  > "${EXP_ROOT}/protocol/cfg_recipe.json"
python "${ROOT_DIR}/scripts/dewo_v2/tasks.py" write-eval-yaml --task "${TASK}" \
  --output "${EVAL_CFG_DIR}/${TASK}.yaml"
CFG_TASK_CONFIG_DIR="${EVAL_CFG_DIR}"

log "stack=opensource_224_zscore cfg=$(python -c 'import json;print(json.load(open("'"${EXP_ROOT}/protocol/cfg_recipe.json"'"))["cfg"]["recipe_name"])')"
log "CKPT=${CKPT}"
log "BASE_DATASET=${BASE_DATASET}"
log "PAIR_DATASET=${PAIR_DATASET}"
log "EXP_ROOT=${EXP_ROOT}"
log "CFG D+ = 0.9/0/0.1 Successful; D_fail = 1.0/0/0 Failed (train.sh owns mix)"

BUNDLE_MANIFEST="${EXP_ROOT}/protocol/opensource_bundle_manifest.txt"
cat > "${BUNDLE_MANIFEST}" <<EOF
checkpoint=${CKPT}
model_config=${SOURCE_CONFIG}
dataset_stats=${PRETRAINED_NORM_STATS}
stack=opensource_FastWAMDexJocoPolicy
image_size=224
norm=z-score
recipe=recoverability_pairs_${PRIMARY_KIND}
task=${TASK}
cfg_recipe=${EXP_ROOT}/protocol/cfg_recipe.json
EOF

code_commit="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
src_cfg_sha="$(python -c "import hashlib,sys; from pathlib import Path; print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())" "${SOURCE_CONFIG}")"
norm_sha="$(python -c "import hashlib,sys; from pathlib import Path; print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())" "${PRETRAINED_NORM_STATS}")"

mkdir -p "${EVE_ROOT}/splits" "${EVE_ROOT}/manifests" "${EVE_ROOT}/protocol"
PAIR_ID="${TASK}_pair_events"
PRIMARY_MANIFEST_FLAGS=()
if [[ "${PRIMARY_KIND}" == "success_rollouts" || "${PRIMARY_KIND}" == "all_success_seeds" ]]; then
  PRIMARY_ID="${TASK}_s0_success_rollouts"
  PRIMARY_MANIFEST_FLAGS=(--primary-source "${PRIMARY_KIND}")
else
  PRIMARY_ID="${TASK}_expert_success"
  PRIMARY_MANIFEST_FLAGS=(--primary-source expert)
fi

if [[ ! -f "${EVE_ROOT}/splits/episode_splits.jsonl" ]]; then
  if [[ "${PRIMARY_KIND}" == "all_success_seeds" ]]; then
    log "selecting one complete success per 4/4 all-success seed as D0 (v9)"
    "${ENV_PREFIX}/bin/python" scripts/dewo_v2/select_success_rollout_primary.py \
      --dataset "${BASE_DATASET}" \
      --dataset-id "${PRIMARY_ID}" \
      --mode one_per_all_success_seed \
      --seed "${PRIMARY_SEED}" \
      --output-json "${EVE_ROOT}/protocol/primary_success_episodes.json" \
      --output-splits "${EVE_ROOT}/splits/episode_splits.jsonl"
  elif [[ "${PRIMARY_KIND}" == "success_rollouts" ]]; then
    log "selecting ${PRIMARY_N} complete success rollouts as primary (no expert)"
    "${ENV_PREFIX}/bin/python" scripts/dewo_v2/select_success_rollout_primary.py \
      --dataset "${BASE_DATASET}" \
      --dataset-id "${PRIMARY_ID}" \
      --n "${PRIMARY_N}" \
      --seed "${PRIMARY_SEED}" \
      --output-json "${EVE_ROOT}/protocol/primary_success_episodes.json" \
      --output-splits "${EVE_ROOT}/splits/episode_splits.jsonl"
  else
    log "building expert episode splits"
    "${ENV_PREFIX}/bin/python" scripts/everobot/build_episode_split.py \
      --dataset "${PRIMARY_ID}=${BASE_DATASET}" \
      --force-success-dataset-id "${PRIMARY_ID}" \
      --val-fraction 0.2 \
      --seed 20260812 \
      --output "${EVE_ROOT}/splits/episode_splits.jsonl" \
      --report "${EVE_ROOT}/splits/episode_splits.report.json"
  fi
fi

if [[ ! -f "${EVE_ROOT}/episode_meta.jsonl" ]]; then
  if [[ "${PRIMARY_KIND}" == "success_rollouts" || "${PRIMARY_KIND}" == "all_success_seeds" ]]; then
    log "init-base S0 success rollouts (force-success; failures are split=test)"
    source_type="policy_rollout"
    source_policy="s0_success_rollout"
    collection_round=0
  else
    log "init-base expert success only"
    source_type="expert_success"
    source_policy="expert"
    collection_round=-1
  fi
  "${ENV_PREFIX}/bin/python" scripts/everobot/build_eve_sidecar.py init-base \
    --dataset-root "${BASE_DATASET}" \
    --dataset-id "${PRIMARY_ID}" \
    --eve-root "${EVE_ROOT}" \
    --task-name "${TASK}" \
    --source-type "${source_type}" \
    --source-policy "${source_policy}" \
    --collection-round "${collection_round}" \
    --force-success \
    --split-map "${EVE_ROOT}/splits/episode_splits.jsonl" \
    --config-path "${SOURCE_CONFIG}" \
    --code-commit "${code_commit}"
fi

primary_manifest="${EVE_ROOT}/manifests/offline_primary_success.json"
log "build primary train manifest id=${PRIMARY_ID} kind=${PRIMARY_KIND}"
"${ENV_PREFIX}/bin/python" scripts/everobot/build_eve_sidecar.py build-manifest \
  --eve-root "${EVE_ROOT}" \
  --manifest-name offline_primary_success \
  --include-outcomes success \
  --success-dataset-ids "${PRIMARY_ID}" \
  --success-sample-mode episode_only \
  --splits train

pair_manifest="${EVE_ROOT}/manifests/offline_b1_jump_fast_pair.json"
PAIR_HORIZON="${PAIR_HORIZON:-crop33}"
PAIR_MANIFEST_FLAGS=()
if [[ "${PAIR_HORIZON}" == "full" ]]; then
  PAIR_MANIFEST_FLAGS+=(--horizon full --skip-aux-success)
fi
log "rewrite-manifest: ${PRIMARY_KIND} primary + dual-role pair events horizon=${PAIR_HORIZON}"
"${ENV_PREFIX}/bin/python" scripts/dewo_v2/build_pair_manifest.py \
  --expert-manifest "${primary_manifest}" \
  --pair-dataset "${PAIR_DATASET}" \
  --pair-dataset-id "${PAIR_ID}" \
  --prompt "${SUCCESS_PROMPT}" \
  --recipe "${TASK}_dewo_v9_recoverability_pairs" \
  --output "${pair_manifest}" \
  "${PRIMARY_MANIFEST_FLAGS[@]}" \
  "${PAIR_MANIFEST_FLAGS[@]}"

val_manifest="${EVE_ROOT}/manifests/offline_selection_primary_success.json"
log "build stride-1 val selection manifest (${PRIMARY_KIND})"
"${ENV_PREFIX}/bin/python" scripts/everobot/build_eve_sidecar.py build-manifest \
  --eve-root "${EVE_ROOT}" \
  --manifest-name offline_selection_primary_success \
  --include-outcomes success \
  --success-dataset-ids "${PRIMARY_ID}" \
  --success-sample-mode episode_only \
  --splits val

env_file="${EVE_ROOT}/protocol/offline_v1_b1_jump_fast.env"
{
  cat <<EOF
# Generated by scripts/dewo_v2/prepare_pair_eve.sh (opensource 224 / z-score)
# Paths / VAE / text cache only. CFG mixing is DEWO v9 in scripts/dewo_v2/train.sh
# (Successful / Failed execution., D+ 0.9/0/0.1, D_fail 1.0/0/0, no FAST).
export FITWAM_ENV_PREFIX=${ENV_PREFIX}
export OPEN_REPO=${OPEN_REPO}
export TASK=${TASK}
export DEWO_TASK_NAME=${TASK}
export BASE_DATASET=${BASE_DATASET}
export PAIR_DATASET=${PAIR_DATASET}
export ROLLOUT_RAW=${ROLLOUT_RAW:-${PRIMARY_DATASET:-${PAIR_DATASET}}}
export PRIMARY_KIND=${PRIMARY_KIND}
export PRIMARY_N=${PRIMARY_N}
export CKPT=${CKPT}
export INIT_WEIGHTS=${CKPT}
export SOURCE_CHECKPOINT=${CKPT}
export STATS=${STATS}
export FASTWAM_SOURCE_CONFIG=${SOURCE_CONFIG}
export FASTWAM_SOURCE_CONFIG_SHA256=${src_cfg_sha}
export SOURCE_BUNDLE_MANIFEST=${BUNDLE_MANIFEST}
export B1_MANIFEST_PATH=${pair_manifest}
export EVE_MANIFEST_PATH=${pair_manifest}
export EVE_VAL_MANIFEST_PATH=${val_manifest}
export PROTOCOL_BUNDLE_PATH=${EVE_ROOT}/protocol/offline_v1_b1_jump_fast.json
export PRETRAINED_NORM_STATS=${PRETRAINED_NORM_STATS}
export NORM_STATS_SOURCE=compute
export NORM_STATS_META_DIR=
export NORM_STATS_BUNDLE_SHA256=${norm_sha}
export TEXT_EMBEDDING_CACHE_DIR=${TEXT_CACHE}
export CFG_TASK_CONFIG_DIR=${CFG_TASK_CONFIG_DIR}
export B1_VIDEO_EXPERIMENT_ROOT=${EXP_ROOT}
export USE_VAE_LATENT_CACHE=1
export VAE_LATENT_CACHE_DIR=${VAE_LATENT_CACHE_DIR}
export REQUIRE_VAE_LATENT_CACHE=1
export DEWO_TASK=${DEWO_TASK}
export DEWO_VARIANT=B1-jump-fast-v9-uncond-adapter
export DEWO_PROTOCOL=${DEWO_PROTOCOL}
export DEWO_OUTPUT_DIR=${DEWO_OUTPUT_DIR}
export FITWAM_WANDB_GROUP=${TASK}_dewo_v9_opensource
export SUCCESS_PROMPT=${SUCCESS_PROMPT@Q}
EOF
} > "${env_file}"
log "wrote ${env_file}"

cat > "${EVE_ROOT}/protocol/offline_v1_b1_jump_fast.json" <<EOF
{
  "protocol": "${TASK}_dewo_v9_recoverability_pairs",
  "variant": "B1-jump-fast-v9-uncond-adapter",
  "stack": "opensource_224_zscore",
  "manifest": "${pair_manifest}",
  "val_manifest": "${val_manifest}",
  "pair_dataset": "${PAIR_DATASET}",
  "pretrained_norm_stats": "${PRETRAINED_NORM_STATS}",
  "source_config": "${SOURCE_CONFIG}",
  "checkpoint": "${CKPT}",
  "include_s0_success_rollouts": $([[ "${PRIMARY_KIND}" == "success_rollouts" || "${PRIMARY_KIND}" == "all_success_seeds" ]] && echo true || echo false),
  "primary_kind": "${PRIMARY_KIND}",
  "cfg": {
    "success_suffix": " Successful execution.",
    "failure_suffix": " Failed execution.",
    "primary": "0.9,0.0,0.1",
    "aux_success": "1.0,0.0,0.0",
    "aux_fail": "1.0,0.0,0.0",
    "note": "Owned by scripts/dewo_v2/train.sh; not mixed from this file."
  }
}
EOF

log "precomputing base + outcome text embeds"
set -a
# shellcheck disable=SC1090
source "${env_file}"
set +a
"${ENV_PREFIX}/bin/python" scripts/precompute_text_embeds.py \
  "task=${DEWO_TASK}" \
  2>&1 | tee -a "${LOG_DIR}/precompute_text_embeds.log"

VAE_ENCODE_VAL="${VAE_ENCODE_VAL:-false}"
if [[ "${SKIP_VAE_PREENCODE:-0}" == "1" || "${USE_VAE_LATENT_CACHE:-1}" == "0" ]]; then
  log "SKIP VAE pre-encode (USE_VAE_LATENT_CACHE=${USE_VAE_LATENT_CACHE:-1} SKIP_VAE_PREENCODE=${SKIP_VAE_PREENCODE:-0})"
else
  GPUS="${GPUS:?Set GPUS for VAE pre-encode}"
  IFS=',' read -r -a gpu_arr <<< "${GPUS}"
  nproc="${#gpu_arr[@]}"
  if [[ "${nproc}" -lt 1 ]]; then
    log "ERROR: empty GPUS=${GPUS}"
    exit 2
  fi
  log "VAE pre-encode pair windows on GPUs ${GPUS} encode_val=${VAE_ENCODE_VAL}"
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
          2>&1 | tee -a "${LOG_DIR}/precompute_vae_latents.shard${shard_rank}.log"
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
      log "ERROR: one or more VAE shard workers failed"
      exit 2
    fi
  else
    export CUDA_VISIBLE_DEVICES="${gpu_arr[0]}"
    "${ENV_PREFIX}/bin/python" scripts/precompute_vae_latents.py \
      "task=${DEWO_TASK}" \
      "+vae_latent_cache_dir=${VAE_LATENT_CACHE_DIR}" \
      "+encode_val=${VAE_ENCODE_VAL}" \
      2>&1 | tee -a "${LOG_DIR}/precompute_vae_latents.log"
  fi
  n_vae="$(find "${VAE_LATENT_CACHE_DIR}" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
  log "VAE pre-encode done files=${n_vae} cache=${VAE_LATENT_CACHE_DIR}"
fi

# FAST is unused in v9 (all FAST weights 0).
if [[ "${USE_FAST_TEXT:-0}" == "1" ]]; then
  log "USE_FAST_TEXT=1; FAST is not part of the v9 recipe"
else
  log "skipping FAST CFG text embeds (v9 has no FAST channel)"
fi
log "exporting torch-free base + success contexts for DexJoCo CFG eval"
"${ENV_PREFIX}/bin/python" scripts/export_text_embed_cache_npz.py \
  --cache-dir "${TEXT_CACHE}" \
  2>&1 | tee -a "${LOG_DIR}/export_text_embed_cache_npz.log"

text_sha="$(python - <<PY
import hashlib
from pathlib import Path
root = Path("${TEXT_CACHE}")
h = hashlib.sha256()
for p in sorted(root.glob("*.pt")):
    h.update(p.name.encode())
    h.update(str(p.stat().st_size).encode())
print(h.hexdigest())
PY
)"
echo "export TEXT_EMBEDDING_CACHE_SHA256=${text_sha}" >> "${env_file}"

log "DONE. Next (pre-encoded VAE cache by default):"
log "  TASK=${TASK} INIT=s0 DEWO_VERSION=v9 GPUS=<ids> ENV_FILE=${env_file} bash scripts/dewo_v2/train.sh"
