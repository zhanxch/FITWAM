#!/usr/bin/env bash
# DEWO v2 trainer. Task/data come from ENV_FILE. Hyperparams are extras.
# fold_glasses wrappers are unchanged (they still call train_jump_fast_lora.sh).
#
#   TASK=fold_glasses INIT=scratch|s0|lora GPUS=0,1,2,3 \
#     ENV_FILE=data/<task>_dewo_v2_pair_*/eve_v02/protocol/offline_v1_b1_jump_fast.env \
#     bash scripts/dewo_v2/train.sh
#   LR=3e-5 TRAIN_MAX_STEPS=2000 USE_VAE_LATENT_CACHE=1
#   DEWO_HYDRA_OVERRIDES='save_every=100 save_weights_every=100 save_state_every=0'
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_require_gpus
ENV_FILE="${ENV_FILE:?Set ENV_FILE to offline_v1_b1_jump_fast.env from prepare}"
[[ -f "${ENV_FILE}" ]] || { echo "[dewo-v2-train] ERROR missing ${ENV_FILE}" >&2; exit 2; }

INIT="${INIT:-scratch}"
CALLER_DEWO_TASK="${DEWO_TASK:-}"
CALLER_DEWO_VARIANT="${DEWO_VARIANT:-}"
CALLER_DEWO_PROTOCOL="${DEWO_PROTOCOL:-}"
CALLER_OUTPUT_DIR="${DEWO_OUTPUT_DIR:-}"
CALLER_WANDB_GROUP="${FITWAM_WANDB_GROUP:-}"
CALLER_EVE_MANIFEST_PATH="${EVE_MANIFEST_PATH:-}"
CALLER_EVE_VAL_MANIFEST_PATH="${EVE_VAL_MANIFEST_PATH:-}"
CALLER_LR="${LR:-${LEARNING_RATE:-}}"
CALLER_TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-}"
CALLER_USE_VAE="${USE_VAE_LATENT_CACHE:-}"
CALLER_VAE_CACHE_DIR="${VAE_LATENT_CACHE_DIR:-}"
USER_HYDRA="${DEWO_HYDRA_OVERRIDES:-}"

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
[[ -z "${CALLER_EVE_MANIFEST_PATH}" ]] || export EVE_MANIFEST_PATH="${CALLER_EVE_MANIFEST_PATH}"
[[ -z "${CALLER_EVE_VAL_MANIFEST_PATH}" ]] || export EVE_VAL_MANIFEST_PATH="${CALLER_EVE_VAL_MANIFEST_PATH}"
[[ -z "${CALLER_USE_VAE}" ]] || export USE_VAE_LATENT_CACHE="${CALLER_USE_VAE}"
[[ -z "${CALLER_VAE_CACHE_DIR}" ]] || export VAE_LATENT_CACHE_DIR="${CALLER_VAE_CACHE_DIR}"
dewo_v2_align_opensource_stack

case "${INIT}" in
  s0)
    hydra="eval_every=0"
    export DEWO_TASK="${CALLER_DEWO_TASK:-dexjoco/dexjoco_dewo_v2_offline_b1_jump_fast_full_s0}"
    export DEWO_VARIANT="${CALLER_DEWO_VARIANT:-B1-jump-fast-full-s0}"
    export DEWO_PROTOCOL="${CALLER_DEWO_PROTOCOL:-${TASK}_dewo_v2_jump_fast_full_from_s0}"
    export DEWO_SOURCE_SHA="${DEWO_SOURCE_SHA:-${FASTWAM_RESUME_SHA256:-s0}}"
    export TMUX_SESSION="${TMUX_SESSION:-${TASK}_dewo_v2_s0}"
    ;;
  lora)
    hydra="eval_every=0"
    export DEWO_TASK="${CALLER_DEWO_TASK:-${DEWO_TASK:-dexjoco/dexjoco_dewo_v2_offline_b1_jump_fast_lora_3e-5}}"
    export DEWO_VARIANT="${CALLER_DEWO_VARIANT:-B1-jump-fast-lora}"
    export DEWO_PROTOCOL="${CALLER_DEWO_PROTOCOL:-${TASK}_dewo_v2_jump_fast_lora}"
    export TMUX_SESSION="${TMUX_SESSION:-${TASK}_dewo_v2_lora}"
    ;;
  scratch)
    hydra="eval_every=0 resume=null"
    export DEWO_TASK="${CALLER_DEWO_TASK:-dexjoco/dexjoco_dewo_v2_offline_b1_jump_fast_full_1e-4}"
    export DEWO_VARIANT="${CALLER_DEWO_VARIANT:-B1-jump-fast-full-1e-4-scratch}"
    export DEWO_PROTOCOL="${CALLER_DEWO_PROTOCOL:-${TASK}_dewo_v2_jump_fast_full_1e-4_from_scratch}"
    export DEWO_SOURCE_SHA="${DEWO_SOURCE_SHA:-from_scratch_wan_actiondit}"
    export TMUX_SESSION="${TMUX_SESSION:-${TASK}_dewo_v2_scratch}"
    ;;
  *)
    echo "[dewo-v2-train] ERROR: INIT must be scratch|s0|lora, got ${INIT}" >&2
    exit 2
    ;;
esac

[[ -z "${CALLER_LR}" ]] || hydra+=" learning_rate=${CALLER_LR}"
if [[ -n "${CALLER_TRAIN_MAX_STEPS}" ]]; then
  hydra+=" max_steps=${CALLER_TRAIN_MAX_STEPS}"
fi
[[ -z "${BATCH_SIZE:-}" ]] || hydra+=" batch_size=${BATCH_SIZE}"
[[ -z "${PRIMARY_PER_BATCH:-}" ]] || hydra+=" role_balanced_sampling.primary_per_batch=${PRIMARY_PER_BATCH}"
[[ -z "${USER_HYDRA}" ]] || hydra+=" ${USER_HYDRA}"

export DEWO_OUTPUT_DIR="${CALLER_OUTPUT_DIR:-./runs/dexjoco_${TASK}_dewo_v2}"
export FITWAM_WANDB_GROUP="${CALLER_WANDB_GROUP:-${TASK}_dewo_v2_opensource}"
export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_${DEWO_VARIANT}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${TASK}_dewo_v2_${RUN_ID}}"

dewo_v2_apply_vae_policy
if [[ "${USE_VAE_LATENT_CACHE:-0}" == "1" ]]; then
  hydra+=" model.load_vae=false model.fill_vae_latent_cache=false"
else
  hydra+=" model.load_vae=true model.fill_vae_latent_cache=false"
fi
export DEWO_HYDRA_OVERRIDES="${hydra}"

export CFG_SUCCESS_SUFFIX="${CFG_SUCCESS_SUFFIX:- Successful execution.}"
export CFG_FAILURE_SUFFIX="${CFG_FAILURE_SUFFIX:-null}"
export CFG_DROPOUT="${CFG_DROPOUT:-0.0}"
export CFG_PRIMARY_OUTCOME="${CFG_PRIMARY_OUTCOME:-0.5}"
export CFG_PRIMARY_FAST="${CFG_PRIMARY_FAST:-0.0}"
export CFG_PRIMARY_BASE="${CFG_PRIMARY_BASE:-0.5}"
export CFG_AUX_SUCCESS_OUTCOME="${CFG_AUX_SUCCESS_OUTCOME:-0.4}"
export CFG_AUX_SUCCESS_FAST="${CFG_AUX_SUCCESS_FAST:-0.2}"
export CFG_AUX_SUCCESS_BASE="${CFG_AUX_SUCCESS_BASE:-0.4}"
export CFG_AUX_FAIL_OUTCOME="${CFG_AUX_FAIL_OUTCOME:-0.0}"
export CFG_AUX_FAIL_FAST="${CFG_AUX_FAIL_FAST:-0.2}"
export CFG_AUX_FAIL_BASE="${CFG_AUX_FAIL_BASE:-0.4}"
export CFG_FAST_MODEL_ID="${CFG_FAST_MODEL_ID:-physical-intelligence/fast}"
export CFG_FAST_MAX_TOKENS="${CFG_FAST_MAX_TOKENS:-32}"
export CFG_FAST_FAIL_CLOSED="${CFG_FAST_FAIL_CLOSED:-true}"

echo "[dewo-v2-train] TASK=${TASK} INIT=${INIT} GPUS=${GPUS}"
echo "[dewo-v2-train] DEWO_TASK=${DEWO_TASK}"
echo "[dewo-v2-train] hydra=${DEWO_HYDRA_OVERRIDES}"
echo "[dewo-v2-train] vae_cache=${USE_VAE_LATENT_CACHE:-0}"
echo "[dewo-v2-train] tmux=${TMUX_SESSION}"
exec bash "${ROOT_DIR}/scripts/dewo_v2/train_jump_fast_lora.sh"
