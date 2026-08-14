#!/usr/bin/env bash
# Prepare Eve manifests for fold_glasses DEWO v2 recoverability pairs.
# Primary: expert success + pair success events (action loss on).
# Aux: pair success (action off) + pair failure (action off).
# No original S0 success rollouts. Opensource 224 / z-score stack.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FITWAM_ENV:-fastwam}"
ENV_PREFIX="${FITWAM_ENV_PREFIX:-$(conda info --base)/envs/fastwam}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"

OPEN_REPO="${OPEN_REPO:-/data_all/xiangchengzhan/FastWAM-infer-in-DexJoco}"
BASE_DATASET="${BASE_DATASET:-${ROOT_DIR}/data/dexjoco/dexjoco_lerobot_datasets/fold_glasses}"
PAIR_DATASET="${PAIR_DATASET:?Set PAIR_DATASET to the materialized pair LeRobot dataset}"
CKPT="${CKPT:-${ROOT_DIR}/checkpoints/dexjoco/fold_glasses_fastwam/weights/step_010000.pt}"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="${OPEN_REPO}/checkpoints/fold_glasses/step_010000.pt"
fi
SOURCE_CONFIG="${SOURCE_CONFIG:-${OPEN_REPO}/configs/fastwam_dexjoco.yaml}"
PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS:-${OPEN_REPO}/artifacts/fold_glasses/dataset_stats.json}"
TEXT_CACHE="${TEXT_CACHE:-${ROOT_DIR}/data/text_embeds_cache/fold_glasses_dewo_v2_pair}"
CFG_TASK_CONFIG_DIR="${CFG_TASK_CONFIG_DIR:-${ROOT_DIR}/configs/eval/dexjoco/fold_glasses_dewo_v2_cfg}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/data/fold_glasses_dewo_v2_pair_${STAMP}}"
EVE_ROOT="${EVE_ROOT:-${EXP_ROOT}/eve_v02}"
LOG_DIR="${LOG_DIR:-${EXP_ROOT}/logs}"
VAE_LATENT_CACHE_DIR="${VAE_LATENT_CACHE_DIR:-${EXP_ROOT}/vae_latent_cache}"
mkdir -p "${EXP_ROOT}" "${LOG_DIR}" "${TEXT_CACHE}" "${VAE_LATENT_CACHE_DIR}" "${EXP_ROOT}/protocol"

log() { echo "[fold-glasses-dewo-v2-pair $(date -Is)] $*" | tee -a "${LOG_DIR}/prepare.log"; }

if [[ "${BASE_DATASET}" == *fold_glasses_fastwam* ]]; then
  log "ERROR: BASE_DATASET looks like local FitWAM expert (${BASE_DATASET})."
  exit 2
fi
for required in "${PRETRAINED_NORM_STATS}" "${CKPT}" "${PAIR_DATASET}/pair_index.json"; do
  if [[ ! -e "${required}" ]]; then
    log "ERROR: missing ${required}"
    exit 2
  fi
done

log "stack=opensource_224_zscore"
log "CKPT=${CKPT}"
log "BASE_DATASET=${BASE_DATASET}"
log "PAIR_DATASET=${PAIR_DATASET}"
log "EXP_ROOT=${EXP_ROOT}"

BUNDLE_MANIFEST="${EXP_ROOT}/protocol/opensource_bundle_manifest.txt"
cat > "${BUNDLE_MANIFEST}" <<EOF
checkpoint=${CKPT}
model_config=${SOURCE_CONFIG}
dataset_stats=${PRETRAINED_NORM_STATS}
stack=opensource_FastWAMDexJocoPolicy
image_size=224
norm=z-score
recipe=recoverability_pairs_no_s0_success_rollouts
EOF

code_commit="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
ckpt_sha="$(python -c "import hashlib,sys; from pathlib import Path; p=Path(sys.argv[1]); h=hashlib.sha256();
f=p.open('rb');
[h.update(c) for c in iter(lambda:f.read(1<<20), b'')];
print(h.hexdigest())" "${CKPT}")"
src_cfg_sha="$(python -c "import hashlib,sys; from pathlib import Path; print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())" "${SOURCE_CONFIG}")"
norm_sha="$(python -c "import hashlib,sys; from pathlib import Path; print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())" "${PRETRAINED_NORM_STATS}")"

mkdir -p "${EVE_ROOT}/splits" "${EVE_ROOT}/manifests" "${EVE_ROOT}/protocol"

if [[ ! -f "${EVE_ROOT}/splits/episode_splits.jsonl" ]]; then
  log "building expert episode splits"
  "${ENV_PREFIX}/bin/python" scripts/everobot/build_episode_split.py \
    --dataset "fold_glasses_expert_success=${BASE_DATASET}" \
    --force-success-dataset-id fold_glasses_expert_success \
    --val-fraction 0.2 \
    --seed 20260812 \
    --output "${EVE_ROOT}/splits/episode_splits.jsonl" \
    --report "${EVE_ROOT}/splits/episode_splits.report.json"
fi

if [[ ! -f "${EVE_ROOT}/episode_meta.jsonl" ]]; then
  log "init-base expert success only"
  "${ENV_PREFIX}/bin/python" scripts/everobot/build_eve_sidecar.py init-base \
    --dataset-root "${BASE_DATASET}" \
    --dataset-id fold_glasses_expert_success \
    --eve-root "${EVE_ROOT}" \
    --task-name fold_glasses \
    --source-type expert_success \
    --source-policy expert \
    --collection-round -1 \
    --force-success \
    --split-map "${EVE_ROOT}/splits/episode_splits.jsonl" \
    --config-path "${SOURCE_CONFIG}" \
    --code-commit "${code_commit}"
fi

expert_manifest="${EVE_ROOT}/manifests/offline_expert_success.json"
log "build expert-only train manifest"
"${ENV_PREFIX}/bin/python" scripts/everobot/build_eve_sidecar.py build-manifest \
  --eve-root "${EVE_ROOT}" \
  --manifest-name offline_expert_success \
  --include-outcomes success \
  --success-dataset-ids fold_glasses_expert_success \
  --success-sample-mode episode_only \
  --splits train

pair_manifest="${EVE_ROOT}/manifests/offline_b1_jump_fast_pair.json"
log "rewrite-manifest: expert primary + dual-role pair events"
"${ENV_PREFIX}/bin/python" scripts/fold_glasses/build_dewo_v2_pair_manifest.py \
  --expert-manifest "${expert_manifest}" \
  --pair-dataset "${PAIR_DATASET}" \
  --output "${pair_manifest}"

val_manifest="${EVE_ROOT}/manifests/offline_selection_primary_success.json"
log "build stride-1 val selection manifest (expert success)"
"${ENV_PREFIX}/bin/python" scripts/everobot/build_eve_sidecar.py build-manifest \
  --eve-root "${EVE_ROOT}" \
  --manifest-name offline_selection_primary_success \
  --include-outcomes success \
  --success-dataset-ids fold_glasses_expert_success \
  --success-sample-mode episode_only \
  --splits val

env_file="${EVE_ROOT}/protocol/offline_v1_b1_jump_fast.env"
cat > "${env_file}" <<EOF
# Generated by prepare_dewo_v2_pair_eve.sh (opensource 224 / z-score, recoverability pairs)
export FITWAM_ENV_PREFIX=${ENV_PREFIX}
export OPEN_REPO=${OPEN_REPO}
export BASE_DATASET=${BASE_DATASET}
export PAIR_DATASET=${PAIR_DATASET}
export ROLLOUT_RAW=${PAIR_DATASET}
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
export VAE_LATENT_CACHE_DIR=${VAE_LATENT_CACHE_DIR}
export REQUIRE_VAE_LATENT_CACHE=1
export DEWO_TASK=dexjoco/dexjoco_fold_glasses_offline_b1_jump_fast_lora_3e-5
export DEWO_VARIANT=B1-jump-fast-lora-pair
export FITWAM_WANDB_GROUP=fold_glasses_dewo_v2_pair
EOF
log "wrote ${env_file}"

cat > "${EVE_ROOT}/protocol/offline_v1_b1_jump_fast.json" <<EOF
{
  "protocol": "fold_glasses_dewo_v2_recoverability_pairs",
  "variant": "B1-jump-fast-lora-pair",
  "stack": "opensource_224_zscore",
  "manifest": "${pair_manifest}",
  "val_manifest": "${val_manifest}",
  "pair_dataset": "${PAIR_DATASET}",
  "pretrained_norm_stats": "${PRETRAINED_NORM_STATS}",
  "source_config": "${SOURCE_CONFIG}",
  "checkpoint": "${CKPT}",
  "include_s0_success_rollouts": false,
  "cfg": {
    "success": ["base", "success_suffix", "FAST_suffix"],
    "failure": ["base", "FAST_suffix"]
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
  "task=dexjoco/dexjoco_fold_glasses_offline_b1_jump_fast_lora_3e-5" \
  2>&1 | tee -a "${LOG_DIR}/precompute_text_embeds.log"

log "precomputing FAST CFG text embeds on GPUs ${FAST_PRECOMPUTE_GPUS:-0,1,2,3}"
IFS=',' read -r -a fast_gpu_array <<< "${FAST_PRECOMPUTE_GPUS:-0,1,2,3}"
CUDA_VISIBLE_DEVICES="${FAST_PRECOMPUTE_GPUS:-0,1,2,3}" \
  torchrun --standalone --nproc_per_node="${#fast_gpu_array[@]}" \
  scripts/precompute_fast_cfg_text_embeds.py \
  "task=dexjoco/dexjoco_fold_glasses_offline_b1_jump_fast_lora_3e-5" \
  "+fast_cfg_batch_size=${FAST_CFG_BATCH_SIZE:-64}" \
  2>&1 | tee -a "${LOG_DIR}/precompute_fast_cfg_text_embeds.log"
export REQUIRE_VAE_LATENT_CACHE=1

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

log "DONE. Next:"
log "  source ${env_file}"
log "  FILL_VAE_LATENT_CACHE=0 SKIP_VAE_PREENCODE=0 GPUS=0,1,2,3 bash scripts/fold_glasses/train_dewo_v2_jump_fast_lora.sh"
