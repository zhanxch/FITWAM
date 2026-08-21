#!/usr/bin/env bash
# Plain offline ablation: expert + pair-success, opensource 224/z-score.
# Recipe wrapper. Pass ENV_FILE and GPUS; no dated defaults.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ENV_FILE="${ENV_FILE:?Set ENV_FILE to offline_v1_b1_jump_fast.env}"
# shellcheck disable=SC1090
set -a
source "${ENV_FILE}"
set +a

PLAIN_MANIFEST="${PLAIN_MANIFEST:-${B1_VIDEO_EXPERIMENT_ROOT:?Set B1_VIDEO_EXPERIMENT_ROOT or PLAIN_MANIFEST}/eve_v02/manifests/offline_plain_expert_pair.json}"
if [[ ! -f "${PLAIN_MANIFEST}" ]]; then
  echo "[plain-offline] building ${PLAIN_MANIFEST}"
  PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" python \
    "${ROOT_DIR}/scripts/fold_glasses/build_plain_expert_pair_manifest.py" \
    --dewo-pair-manifest "${EVE_MANIFEST_PATH}" \
    --output "${PLAIN_MANIFEST}"
fi
export EVE_MANIFEST_PATH="${PLAIN_MANIFEST}"

export TASK="${TASK:-fold_glasses}"
export TMUX_SESSION="${TMUX_SESSION:-fold_plain_expert_pair_full_1e-4}"
export DEWO_TASK="${DEWO_TASK:-dexjoco/dexjoco_fold_glasses_offline_plain_expert_pair_full_1e-4}"
export DEWO_VARIANT="${DEWO_VARIANT:-plain-expert-pair-full-1e-4-scratch}"
export DEWO_OUTPUT_DIR="${DEWO_OUTPUT_DIR:-./runs/dexjoco_fold_glasses_plain_offline}"
export FITWAM_WANDB_GROUP="${FITWAM_WANDB_GROUP:-fold_glasses_plain_offline_opensource}"
export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_plain-expert-pair-full-1e-4-scratch}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-fold_glasses_plain_${RUN_ID}}"
export SKIP_VAE_PREENCODE="${SKIP_VAE_PREENCODE:-1}"
export FILL_VAE_LATENT_CACHE="${FILL_VAE_LATENT_CACHE:-0}"
export REQUIRE_VAE_LATENT_CACHE="${REQUIRE_VAE_LATENT_CACHE:-1}"
export DEWO_HYDRA_OVERRIDES="${DEWO_HYDRA_OVERRIDES:-eval_every=0 resume=null}"

exec bash "${ROOT_DIR}/scripts/dewo_v2/train_jump_fast_lora.sh"
