#!/usr/bin/env bash
# DEWO v2 full-parameter continue-train from S0 (opensource stack).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -n "${ENV_FILE:-}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${ENV_FILE}"
  set +a
fi

export TASK="${TASK:-hammer_nail}"
export INIT_WEIGHTS="${INIT_WEIGHTS:-${ROOT_DIR}/checkpoints/dexjoco/hammer_nail_fastwam/weights/step_002500.pt}"
export SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${INIT_WEIGHTS}}"
export TMUX_SESSION="${TMUX_SESSION:-hammer_dewo_v2_full_1e-4_s0}"
export DEWO_TASK="dexjoco/dexjoco_hammer_nail_offline_b1_jump_fast_full_1e-4_s0"
export DEWO_VARIANT="${DEWO_VARIANT:-B1-jump-fast-full-1e-4-s0}"
export DEWO_OUTPUT_DIR="${DEWO_OUTPUT_DIR:-./runs/dexjoco_hammer_nail_dewo_v2}"
export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_B1-jump-fast-full-1e-4-s0}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-hammer_nail_dewo_v2_${RUN_ID}}"
export SKIP_VAE_PREENCODE="${SKIP_VAE_PREENCODE:-1}"
export FILL_VAE_LATENT_CACHE="${FILL_VAE_LATENT_CACHE:-0}"
export REQUIRE_VAE_LATENT_CACHE="${REQUIRE_VAE_LATENT_CACHE:-1}"
export DEWO_HYDRA_OVERRIDES="${DEWO_HYDRA_OVERRIDES:-eval_every=0}"

exec bash "${ROOT_DIR}/scripts/dewo_v2/train_run.sh"
