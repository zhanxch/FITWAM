#!/usr/bin/env bash
# DEWO v2 full-parameter continue-train from official fold_glasses S0.
# Opensource stack: 224 / z-score / OPEN stats / step_010000.pt init.
# Optimizer matches official uncond recipe (lr=1e-4, cosine, no LoRA).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ENV_FILE="${ENV_FILE:-${ROOT_DIR}/data/fold_glasses_dewo_v2_pair_20260813/eve_v02/protocol/offline_v1_b1_jump_fast.env}"
# shellcheck disable=SC1090
set -a
source "${ENV_FILE}"
set +a

export GPUS="${GPUS:-0,1,2,3}"
export TMUX_SESSION="${TMUX_SESSION:-fold_dewo_v2_full_1e-4}"
export DEWO_TASK=dexjoco/dexjoco_fold_glasses_offline_b1_jump_fast_full_1e-4
export DEWO_VARIANT=B1-jump-fast-full-1e-4
export DEWO_OUTPUT_DIR="${DEWO_OUTPUT_DIR:-./runs/dexjoco_fold_glasses_dewo_v2}"
export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_B1-jump-fast-full-1e-4}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-fold_glasses_dewo_v2_${RUN_ID}}"
export SKIP_VAE_PREENCODE=1
export FILL_VAE_LATENT_CACHE=0
export REQUIRE_VAE_LATENT_CACHE=1
export DEWO_HYDRA_OVERRIDES="${DEWO_HYDRA_OVERRIDES:-eval_every=0}"

exec bash "${ROOT_DIR}/scripts/fold_glasses/train_dewo_v2_jump_fast_lora.sh"
