#!/usr/bin/env bash
# Prepare Eve manifests for fold_glasses DEWO v2 on the **opensource** stack:
#   - primary = expert success + same-seed paired rollout success
#     (drop never-failed seeds; early fail → crop failure interval;
#      mid/late fail → keep second half only)
#   - aux     = detected width-jump failure event windows
#   - norm    = OPEN artifacts z-score (NOT local meta min/max)
#   - images  = 224×224
#
# Prerequisites:
#   ROLLOUT_RAW          opensource collect rollout_raw_200
#   WIDTH_JUMP_LEDGER    failure event JSONL with core_start/core_end frames
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FITWAM_ENV:-fastwam}"
ENV_PREFIX="${FITWAM_ENV_PREFIX:-$(conda info --base)/envs/fastwam}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"

OPEN_REPO="${OPEN_REPO:-${ROOT_DIR}/../FastWAM-infer-in-DexJoco}"
OPEN_REPO="$(cd "${OPEN_REPO}" && pwd)"
BASE_DATASET="${BASE_DATASET:-${ROOT_DIR}/data/dexjoco/dexjoco_lerobot_datasets/fold_glasses}"
ROLLOUT_RAW="${ROLLOUT_RAW:?Set ROLLOUT_RAW to the opensource collect rollout_raw_200}"
WIDTH_JUMP_LEDGER="${WIDTH_JUMP_LEDGER:?Set WIDTH_JUMP_LEDGER to failure_events.jsonl}"
FAILURE_EVENT_ANNOTATION_METHOD="${FAILURE_EVENT_ANNOTATION_METHOD:-width_jump_centered_window}"
FAILURE_EVENT_ANNOTATION_VERSION="${FAILURE_EVENT_ANNOTATION_VERSION:-dewo_v2_width_jump_opensource_v1}"
CKPT="${CKPT:-${ROOT_DIR}/checkpoints/dexjoco/fold_glasses_fastwam/weights/step_010000.pt}"
# Prefer OPEN release if local dexjoco mirror missing.
if [[ ! -f "${CKPT}" ]]; then
  CKPT="${OPEN_REPO}/checkpoints/fold_glasses/step_010000.pt"
fi
SOURCE_CONFIG="${SOURCE_CONFIG:-${OPEN_REPO}/configs/fastwam_dexjoco.yaml}"
PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS:-${OPEN_REPO}/artifacts/fold_glasses/dataset_stats.json}"
TEXT_CACHE="${TEXT_CACHE:-${ROOT_DIR}/data/text_embeds_cache/fold_glasses_dewo_v2_opensource}"
CFG_TASK_CONFIG_DIR="${CFG_TASK_CONFIG_DIR:-${ROOT_DIR}/configs/eval/dexjoco/fold_glasses_dewo_v2_cfg}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/data/fold_glasses_dewo_v2_opensource_${STAMP}}"
TRIM_META_ROOT="${TRIM_META_ROOT:-${EXP_ROOT}/width_jump_trim_meta}"
EVE_ROOT="${EVE_ROOT:-${EXP_ROOT}/eve_v02}"
LOG_DIR="${LOG_DIR:-${EXP_ROOT}/logs}"
VAE_LATENT_CACHE_DIR="${VAE_LATENT_CACHE_DIR:-${EXP_ROOT}/vae_latent_cache}"
mkdir -p "${EXP_ROOT}" "${LOG_DIR}" "${TEXT_CACHE}" "${VAE_LATENT_CACHE_DIR}" "${EXP_ROOT}/protocol"

log() { echo "[fold-glasses-dewo-v2-opensource-prep $(date -Is)] $*" | tee -a "${LOG_DIR}/prepare.log"; }

if [[ "${BASE_DATASET}" == *fold_glasses_fastwam* ]]; then
  BASE_DATASET="${ROOT_DIR}/data/dexjoco/dexjoco_lerobot_datasets/fold_glasses"
  log "aligned BASE_DATASET to opensource expert ${BASE_DATASET}"
fi
if [[ ! -f "${PRETRAINED_NORM_STATS}" ]]; then
  log "ERROR: missing OPEN stats: ${PRETRAINED_NORM_STATS}"
  exit 2
fi
if [[ ! -f "${CKPT}" ]]; then
  log "ERROR: missing opensource/dexjoco ckpt: ${CKPT}"
  exit 2
fi

log "stack=opensource_224_zscore"
log "CKPT=${CKPT}"
log "BASE_DATASET=${BASE_DATASET}"
log "PRETRAINED_NORM_STATS=${PRETRAINED_NORM_STATS}"
log "SOURCE_CONFIG=${SOURCE_CONFIG}"
log "ROLLOUT_RAW=${ROLLOUT_RAW}"
log "WIDTH_JUMP_LEDGER=${WIDTH_JUMP_LEDGER}"
log "EXP_ROOT=${EXP_ROOT}"

# Minimal provenance bundle for offline launcher (opensource has no s0_bundle).
BUNDLE_MANIFEST="${EXP_ROOT}/protocol/opensource_bundle_manifest.txt"
cat > "${BUNDLE_MANIFEST}" <<EOF
checkpoint=${CKPT}
model_config=${SOURCE_CONFIG}
dataset_stats=${PRETRAINED_NORM_STATS}
stack=opensource_FastWAMDexJocoPolicy
image_size=224
norm=z-score
EOF

log "converting width-jump ledger -> trim report"
"${ENV_PREFIX}/bin/python" scripts/everobot/width_jump_ledger_to_trim_report.py \
  --rollout-root "${ROLLOUT_RAW}" \
  --ledger "${WIDTH_JUMP_LEDGER}" \
  --output "${TRIM_META_ROOT}"

code_commit="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
ckpt_sha="$(python -c "import hashlib,sys; from pathlib import Path; p=Path(sys.argv[1]); h=hashlib.sha256();
f=p.open('rb');
[h.update(c) for c in iter(lambda:f.read(1<<20), b'')];
print(h.hexdigest())" "${CKPT}")"
src_cfg_sha="$(python -c "import hashlib,sys; from pathlib import Path; print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())" "${SOURCE_CONFIG}")"
norm_sha="$(python -c "import hashlib,sys; from pathlib import Path; print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())" "${PRETRAINED_NORM_STATS}")"

mkdir -p "${EVE_ROOT}/splits" "${EVE_ROOT}/manifests" "${EVE_ROOT}/protocol"

if [[ ! -f "${EVE_ROOT}/splits/episode_splits.jsonl" ]]; then
  log "building episode splits"
  "${ENV_PREFIX}/bin/python" scripts/everobot/build_episode_split.py \
    --dataset "fold_glasses_expert_success=${BASE_DATASET}" \
    --dataset "fold_glasses_s0_rollout=${ROLLOUT_RAW}" \
    --force-success-dataset-id fold_glasses_expert_success \
    --require-explicit-outcome-dataset-id fold_glasses_s0_rollout \
    --val-fraction 0.2 \
    --seed 20260812 \
    --output "${EVE_ROOT}/splits/episode_splits.jsonl" \
    --report "${EVE_ROOT}/splits/episode_splits.report.json"
fi

if [[ ! -f "${EVE_ROOT}/episode_meta.jsonl" ]]; then
  log "init-base + append-rollout (width-jump events)"
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

  "${ENV_PREFIX}/bin/python" scripts/everobot/build_eve_sidecar.py append-rollout \
    --base-eve-root "${EVE_ROOT}" \
    --rollout-root "${ROLLOUT_RAW}" \
    --trimmed-event-root "${TRIM_META_ROOT}" \
    --dataset-id fold_glasses_s0_rollout \
    --task-name fold_glasses \
    --source-policy fastwam_opensource_s0 \
    --source-checkpoint "${CKPT}" \
    --source-checkpoint-sha256 "${ckpt_sha}" \
    --collection-round 0 \
    --failure-action-loss disabled \
    --require-explicit-outcomes \
    --split-map "${EVE_ROOT}/splits/episode_splits.jsonl" \
    --config-path "${SOURCE_CONFIG}" \
    --code-commit "${code_commit}" \
    --annotation-source auto \
    --annotation-method "${FAILURE_EVENT_ANNOTATION_METHOD}" \
    --annotation-version "${FAILURE_EVENT_ANNOTATION_VERSION}"
fi

b1_manifest="${EVE_ROOT}/manifests/offline_b1_jump_fast.json"
log "build-manifest: stride-1 success + detected failure events with sliding windows; discard fallbacks"
"${ENV_PREFIX}/bin/python" scripts/everobot/build_eve_sidecar.py build-manifest \
  --eve-root "${EVE_ROOT}" \
  --manifest-name offline_b1_jump_fast \
  --include-outcomes success failure \
  --success-dataset-ids fold_glasses_expert_success fold_glasses_s0_rollout \
  --failure-dataset-ids fold_glasses_s0_rollout \
  --success-sample-mode episode_only \
  --failure-sample-mode event_only \
  --failure-window-selection sliding \
  --failure-source-window-rules trimmed_failure_window \
  --event-types failure_event \
  --failure-action-loss disabled \
  --splits train

"${ENV_PREFIX}/bin/python" - "${b1_manifest}" \
  "${EXPECTED_FAILURE_EVENT_UNITS:-}" "${EXPECTED_FAILURE_WINDOWS:-}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
failures = [
    row for row in manifest["samples"]
    if row.get("episode_outcome") == "failure"
]
fallbacks = [
    row for row in failures
    if row.get("source_window_rule") == "full_failure_episode"
]
windows = sum(
    max(0, int(row["end_frame"]) - int(row["start_frame"]) - 33 + 1)
    for row in failures
)
print(
    f"[manifest-check] failure_units={len(failures)} "
    f"failure_windows={windows} fallbacks={len(fallbacks)}"
)
if fallbacks:
    raise SystemExit("full_failure_episode fallback leaked into training manifest")
expected_units = sys.argv[2].strip()
expected_windows = sys.argv[3].strip()
if expected_units and len(failures) != int(expected_units):
    raise SystemExit(
        f"expected {expected_units} failure units, observed {len(failures)}"
    )
if expected_windows and windows != int(expected_windows):
    raise SystemExit(
        f"expected {expected_windows} failure windows, observed {windows}"
    )
PY

seedpair_manifest="${EVE_ROOT}/manifests/offline_b1_jump_fast_seedpair.json"
log "rewrite-manifest: same-seed success pairing (drop never-failed; early crop / late second-half)"
"${ENV_PREFIX}/bin/python" scripts/fold_glasses/rewrite_manifest_seedpair_success.py \
  --manifest "${b1_manifest}" \
  --outcomes "${ROLLOUT_RAW}/meta/episode_outcomes.jsonl" \
  --episode-meta "${EVE_ROOT}/episode_meta.jsonl" \
  --failure-ledger "${WIDTH_JUMP_LEDGER}" \
  --output "${seedpair_manifest}"
b1_manifest="${seedpair_manifest}"

val_manifest="${EVE_ROOT}/manifests/offline_selection_primary_success.json"
log "build stride-1 val selection manifest (expert success)"
"${ENV_PREFIX}/bin/python" scripts/everobot/build_eve_sidecar.py build-manifest \
  --eve-root "${EVE_ROOT}" \
  --manifest-name offline_selection_primary_success \
  --include-outcomes success \
  --success-dataset-ids fold_glasses_expert_success \
  --success-sample-mode episode_only \
  --splits val

# Preflight treats non-meta as "compute"; OPEN stats live in pretrained_norm_stats.
env_file="${EVE_ROOT}/protocol/offline_v1_b1_jump_fast.env"
cat > "${env_file}" <<EOF
# Generated by prepare_dewo_v2_eve.sh (opensource 224 / z-score)
export FITWAM_ENV_PREFIX=${ENV_PREFIX}
export OPEN_REPO=${OPEN_REPO}
export BASE_DATASET=${BASE_DATASET}
export ROLLOUT_RAW=${ROLLOUT_RAW}
export INIT_WEIGHTS=${CKPT}
export SOURCE_CHECKPOINT=${CKPT}
export FASTWAM_SOURCE_CONFIG=${SOURCE_CONFIG}
export FASTWAM_SOURCE_CONFIG_SHA256=${src_cfg_sha}
export SOURCE_BUNDLE_MANIFEST=${BUNDLE_MANIFEST}
export B1_MANIFEST_PATH=${b1_manifest}
export EVE_MANIFEST_PATH=${b1_manifest}
export EVE_VAL_MANIFEST_PATH=${val_manifest}
export PROTOCOL_BUNDLE_PATH=${EVE_ROOT}/protocol/offline_v1_b1_jump_fast.json
export PRETRAINED_NORM_STATS=${PRETRAINED_NORM_STATS}
export NORM_STATS_SOURCE=compute
export NORM_STATS_META_DIR=
export NORM_STATS_BUNDLE_SHA256=${norm_sha}
export TEXT_EMBEDDING_CACHE_DIR=${TEXT_CACHE}
export CFG_TASK_CONFIG_DIR=${CFG_TASK_CONFIG_DIR}
export B1_VIDEO_TRIM_META=${TRIM_META_ROOT}/collection_summary.json
export B1_VIDEO_EXPERIMENT_ROOT=${EXP_ROOT}
export VAE_LATENT_CACHE_DIR=${VAE_LATENT_CACHE_DIR}
export REQUIRE_VAE_LATENT_CACHE=1
export DEWO_TASK=dexjoco/dexjoco_fold_glasses_offline_b1_jump_fast_lora_3e-5
export DEWO_VARIANT=B1-jump-fast-lora
export FITWAM_WANDB_GROUP=fold_glasses_dewo_v2_opensource
EOF
log "wrote ${env_file}"

proto_json="${EVE_ROOT}/protocol/offline_v1_b1_jump_fast.json"
if [[ ! -f "${proto_json}" ]]; then
  cat > "${proto_json}" <<EOF
{
  "protocol": "fold_glasses_dewo_v2_jump_fast_lora_opensource",
  "variant": "B1-jump-fast-lora",
  "stack": "opensource_224_zscore",
  "manifest": "${b1_manifest}",
  "val_manifest": "${val_manifest}",
  "width_jump_ledger": "${WIDTH_JUMP_LEDGER}",
  "trim_meta": "${TRIM_META_ROOT}/collection_summary.json",
  "pretrained_norm_stats": "${PRETRAINED_NORM_STATS}",
  "source_config": "${SOURCE_CONFIG}",
  "checkpoint": "${CKPT}"
}
EOF
  log "wrote ${proto_json}"
fi

log "precomputing base + outcome text embeds"
set -a
# shellcheck disable=SC1090
source "${env_file}"
set +a
# Text/FAST caches are built before VAE pre-encode; do not require latents yet.
export REQUIRE_VAE_LATENT_CACHE=0
"${ENV_PREFIX}/bin/python" scripts/precompute_text_embeds.py \
  "task=dexjoco/dexjoco_fold_glasses_offline_b1_jump_fast_lora_3e-5" \
  2>&1 | tee -a "${LOG_DIR}/precompute_text_embeds.log"

FAST_PRECOMPUTE_GPUS="${FAST_PRECOMPUTE_GPUS:-${GPUS:?Set GPUS or FAST_PRECOMPUTE_GPUS}}"
log "precomputing FAST CFG text embeds on GPUs ${FAST_PRECOMPUTE_GPUS}"
IFS=',' read -r -a fast_gpu_array <<< "${FAST_PRECOMPUTE_GPUS}"
CUDA_VISIBLE_DEVICES="${FAST_PRECOMPUTE_GPUS}" \
  torchrun --standalone --nproc_per_node="${#fast_gpu_array[@]}" \
  scripts/precompute_fast_cfg_text_embeds.py \
  "task=dexjoco/dexjoco_fold_glasses_offline_b1_jump_fast_lora_3e-5" \
  "+fast_cfg_batch_size=${FAST_CFG_BATCH_SIZE:-64}" \
  2>&1 | tee -a "${LOG_DIR}/precompute_fast_cfg_text_embeds.log"
# Restore train-time contract written into the env file.
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

log "DONE. Next (tmux: VAE pre-encode then train on opensource stack):"
log "  source ${env_file}"
log "  TASK=fold_glasses GPUS=<ids> bash scripts/dewo_v2/train_jump_fast_lora.sh"
log "  tmux attach -t fold_dewo_v2_train"
