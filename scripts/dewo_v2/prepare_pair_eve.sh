#!/usr/bin/env bash
# Prepare Eve manifests + text/FAST caches for DEWO v2 recoverability pairs.
#   TASK=water_plant PAIR_DATASET=... EXP_ROOT=... GPUS=4,5,6,7 \
#     bash scripts/dewo_v2/prepare_pair_eve.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_load_task "${TASK}"

PRIMARY_KIND="${PRIMARY_KIND:-expert}"
PRIMARY_N="${PRIMARY_N:-15}"
PRIMARY_SEED="${PRIMARY_SEED:-20260820}"
if [[ "${PRIMARY_KIND}" == "success_rollouts" ]]; then
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
SOURCE_CONFIG="${SOURCE_CONFIG:-${OPEN_REPO}/configs/fastwam_dexjoco.yaml}"
TEXT_CACHE="${TEXT_CACHE:-${ROOT_DIR}/data/text_embeds_cache/${TASK}_dewo_v2_pair}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/data/${TASK}_dewo_v2_pair_${STAMP}}"
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
log "CFG_PRIMARY=${CFG_PRIMARY} AUX_SUCCESS=${CFG_AUX_SUCCESS} AUX_FAIL=${CFG_AUX_FAIL}"
log "CFG_SUCCESS_SUFFIX=${CFG_SUCCESS_SUFFIX} FAILURE=${CFG_FAILURE_SUFFIX}"

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
if [[ "${PRIMARY_KIND}" == "success_rollouts" ]]; then
  PRIMARY_ID="${TASK}_s0_success_rollouts"
  PRIMARY_MANIFEST_FLAGS=(--primary-source success_rollouts)
else
  PRIMARY_ID="${TASK}_expert_success"
  PRIMARY_MANIFEST_FLAGS=(--primary-source expert)
fi

if [[ ! -f "${EVE_ROOT}/splits/episode_splits.jsonl" ]]; then
  if [[ "${PRIMARY_KIND}" == "success_rollouts" ]]; then
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
  if [[ "${PRIMARY_KIND}" == "success_rollouts" ]]; then
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
log "rewrite-manifest: ${PRIMARY_KIND} primary + dual-role pair events"
"${ENV_PREFIX}/bin/python" scripts/fold_glasses/build_dewo_v2_pair_manifest.py \
  --expert-manifest "${primary_manifest}" \
  --pair-dataset "${PAIR_DATASET}" \
  --pair-dataset-id "${PAIR_ID}" \
  --prompt "${SUCCESS_PROMPT}" \
  --recipe "${TASK}_dewo_v2_recoverability_pairs" \
  --output "${pair_manifest}" \
  "${PRIMARY_MANIFEST_FLAGS[@]}"

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
export FITWAM_ENV_PREFIX=${ENV_PREFIX}
export OPEN_REPO=${OPEN_REPO}
export TASK=${TASK}
export DEWO_TASK_NAME=${TASK}
export BASE_DATASET=${BASE_DATASET}
export PAIR_DATASET=${PAIR_DATASET}
export ROLLOUT_RAW=${ROLLOUT_RAW:-${PRIMARY_DATASET:-${PAIR_DATASET}}}
export PRIMARY_KIND=${PRIMARY_KIND}
export PRIMARY_N=${PRIMARY_N}
export INIT_WEIGHTS=${CKPT}
export SOURCE_CHECKPOINT=${CKPT}
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
# VAE latent cache is opt-in (USE_VAE_LATENT_CACHE=1). Default train uses online VAE.
export VAE_LATENT_CACHE_DIR=${VAE_LATENT_CACHE_DIR}
export REQUIRE_VAE_LATENT_CACHE=0
export DEWO_TASK=${DEWO_TASK}
export DEWO_VARIANT=B1-jump-fast-lora-pair
export DEWO_PROTOCOL=${DEWO_PROTOCOL}
export DEWO_OUTPUT_DIR=${DEWO_OUTPUT_DIR}
export FITWAM_WANDB_GROUP=${TASK}_dewo_v2_pair
export SUCCESS_PROMPT=${SUCCESS_PROMPT@Q}
EOF
  python "${ROOT_DIR}/scripts/dewo_v2/tasks.py" export-env --task "${TASK}"
} > "${env_file}"
# export-env reprints single-task defaults; keep mixed-S0 / rollout-primary overrides.
{
  echo "export CKPT=${CKPT}"
  echo "export INIT_WEIGHTS=${CKPT}"
  echo "export SOURCE_CHECKPOINT=${CKPT}"
  echo "export STATS=${STATS}"
  echo "export PRETRAINED_NORM_STATS=${PRETRAINED_NORM_STATS}"
  echo "export BASE_DATASET=${BASE_DATASET}"
  echo "export PRIMARY_KIND=${PRIMARY_KIND}"
  echo "export ROLLOUT_RAW=${ROLLOUT_RAW:-${PRIMARY_DATASET:-}}"
} >> "${env_file}"
log "wrote ${env_file}"

cat > "${EVE_ROOT}/protocol/offline_v1_b1_jump_fast.json" <<EOF
{
  "protocol": "${TASK}_dewo_v2_recoverability_pairs",
  "variant": "B1-jump-fast-lora-pair",
  "stack": "opensource_224_zscore",
  "manifest": "${pair_manifest}",
  "val_manifest": "${val_manifest}",
  "pair_dataset": "${PAIR_DATASET}",
  "pretrained_norm_stats": "${PRETRAINED_NORM_STATS}",
  "source_config": "${SOURCE_CONFIG}",
  "checkpoint": "${CKPT}",
  "include_s0_success_rollouts": $([[ "${PRIMARY_KIND}" == "success_rollouts" ]] && echo true || echo false),
  "primary_kind": "${PRIMARY_KIND}",
  "cfg_recipe": "${EXP_ROOT}/protocol/cfg_recipe.json",
  "cfg": {
    "success_suffix": $(python -c "import json,os; print(json.dumps(os.environ['CFG_SUCCESS_SUFFIX']))"),
    "failure_suffix": $(python -c "import json,os; v=os.environ.get('CFG_FAILURE_SUFFIX','null'); print('null' if v in {'null','none',''} else json.dumps(v))"),
    "primary": "${CFG_PRIMARY}",
    "aux_success": "${CFG_AUX_SUCCESS}",
    "aux_fail": "${CFG_AUX_FAIL}"
  }
}
EOF

log "precomputing base + outcome text embeds"
set -a
# shellcheck disable=SC1090
source "${env_file}"
set +a
export REQUIRE_VAE_LATENT_CACHE=0
"${ENV_PREFIX}/bin/python" scripts/precompute_text_embeds.py \
  "task=${DEWO_TASK}" \
  2>&1 | tee -a "${LOG_DIR}/precompute_text_embeds.log"

# FAST text-emb: only needed when some CFG channel has fast>0.
# Expert/primary never uses FAST (CFG_PRIMARY_FAST must be 0); default aux still does.
if [[ "${SKIP_FAST_TEXT_PRECOMPUTE:-0}" == "1" ]]; then
  log "SKIP_FAST_TEXT_PRECOMPUTE=1; skipping FAST CFG text embeds"
elif ! dewo_v2_cfg_uses_fast; then
  log "all CFG_*_FAST=0; skipping FAST CFG text embeds"
else
  FAST_PRECOMPUTE_GPUS="${FAST_PRECOMPUTE_GPUS:-${GPUS:?Set GPUS or FAST_PRECOMPUTE_GPUS for FAST text precompute}}"
  if pgrep -f 'precompute_vae_latents.py' >/dev/null 2>&1 || pgrep -f 'scan_failure_recoverability_frontier.py' >/dev/null 2>&1; then
    FAST_PRECOMPUTE_GPUS="${FAST_PRECOMPUTE_GPUS_IF_BUSY:-${FAST_PRECOMPUTE_GPUS}}"
    log "VAE/scan still running; FAST pinned to GPUs ${FAST_PRECOMPUTE_GPUS}"
  fi
  log "precomputing FAST CFG text embeds on GPUs ${FAST_PRECOMPUTE_GPUS} (skips primary/expert windows)"
  IFS=',' read -r -a fast_gpu_array <<< "${FAST_PRECOMPUTE_GPUS}"
  CUDA_VISIBLE_DEVICES="${FAST_PRECOMPUTE_GPUS}" \
    torchrun --standalone --nproc_per_node="${#fast_gpu_array[@]}" \
    scripts/precompute_fast_cfg_text_embeds.py \
    "task=${DEWO_TASK}" \
    "+fast_cfg_batch_size=${FAST_CFG_BATCH_SIZE:-64}" \
    2>&1 | tee -a "${LOG_DIR}/precompute_fast_cfg_text_embeds.log"
fi
# Keep REQUIRE=0: default train path is online VAE (no pre-encode).
export REQUIRE_VAE_LATENT_CACHE=0

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

log "DONE. Next (online VAE by default; no VAE pre-encode):"
log "  source ${env_file}"
log "  TASK=${TASK} GPUS=<ids> bash scripts/dewo_v2/train_jump_fast_lora.sh"
log "  # opt-in cache: USE_VAE_LATENT_CACHE=1 TASK=... bash scripts/dewo_v2/train_jump_fast_lora.sh"
