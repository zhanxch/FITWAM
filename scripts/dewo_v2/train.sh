#!/usr/bin/env bash
# DEWO v9 trainer. Task/data come from ENV_FILE (paths, VAE, text cache).
# CFG mixing is this recipe, not the prepare protocol env.
#
#   TASK=fold_glasses INIT=s0 GPUS=0,1,2,3 \
#     ENV_FILE=data/<task>_dewo_v9_pair_*/eve_v02/protocol/offline_v1_b1_jump_fast.env \
#     bash scripts/dewo_v2/train.sh
#
# D+ action dropout 0.9/0/0.1. D_fail is always Failed (1.0/0/0). No FAST.
# Suffixes: ' Successful execution.' / ' Failed execution.'
set -euo pipefail

SCRIPT_DIR="$(realpath -e -- "$(dirname -- "${BASH_SOURCE[0]}")")"
ROOT_DIR="$(realpath -e -- "${SCRIPT_DIR}/../..")"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_require_gpus
ENV_FILE="${ENV_FILE:?Set ENV_FILE to offline_v1_b1_jump_fast.env from prepare}"
ENV_FILE_INPUT="${ENV_FILE}"
if ! ENV_FILE="$(realpath -e -- "${ENV_FILE_INPUT}" 2>/dev/null)"; then
  echo "[dewo-train] ERROR missing ${ENV_FILE_INPUT}" >&2
  exit 2
fi
export ENV_FILE

INIT="${INIT:-s0}"
DEWO_VERSION="${DEWO_VERSION:-v9}"
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
CALLER_PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS:-}"
USER_HYDRA="${DEWO_HYDRA_OVERRIDES:-}"

if [[ "${CALLER_DEWO_VARIANT}" == B1-jump-fast-pair ]]; then
  CALLER_DEWO_VARIANT=""
fi
if [[ "${CALLER_DEWO_VARIANT}" == *lora* || "${CALLER_DEWO_VARIANT}" == *LoRA* ]]; then
  echo "[dewo-train] ignoring LoRA DEWO_VARIANT=${CALLER_DEWO_VARIANT}" >&2
  CALLER_DEWO_VARIANT=""
fi
if [[ "${CALLER_DEWO_PROTOCOL}" == *lora* || "${CALLER_DEWO_PROTOCOL}" == *LoRA* ]]; then
  echo "[dewo-train] ignoring LoRA DEWO_PROTOCOL=${CALLER_DEWO_PROTOCOL}" >&2
  CALLER_DEWO_PROTOCOL=""
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
# Protocol env may still stamp v2 FAST triples / null failure. Drop them.
dewo_v2_clear_cfg_mix
[[ -z "${CALLER_EVE_MANIFEST_PATH}" ]] || export EVE_MANIFEST_PATH="${CALLER_EVE_MANIFEST_PATH}"
[[ -z "${CALLER_EVE_VAL_MANIFEST_PATH}" ]] || export EVE_VAL_MANIFEST_PATH="${CALLER_EVE_VAL_MANIFEST_PATH}"
[[ -z "${CALLER_USE_VAE}" ]] || export USE_VAE_LATENT_CACHE="${CALLER_USE_VAE}"
[[ -z "${CALLER_VAE_CACHE_DIR}" ]] || export VAE_LATENT_CACHE_DIR="${CALLER_VAE_CACHE_DIR}"
[[ -z "${CALLER_PRETRAINED_NORM_STATS}" ]] || export PRETRAINED_NORM_STATS="${CALLER_PRETRAINED_NORM_STATS}"
dewo_v2_align_opensource_stack

if [[ "${DEWO_VERSION}" != "v9" ]]; then
  echo "[dewo-train] ERROR: only DEWO_VERSION=v9 is supported, got ${DEWO_VERSION}" >&2
  exit 2
fi
export DEWO_VERSION=v9

case "${INIT}" in
  s0)
    hydra="eval_every=0"
    export DEWO_TASK="dexjoco/dexjoco_dewo_v9_offline_b1_jump_fast_uncond"
    export DEWO_VARIANT="${CALLER_DEWO_VARIANT:-B1-jump-fast-v9-uncond-adapter}"
    export DEWO_PROTOCOL="${CALLER_DEWO_PROTOCOL:-${TASK}_dewo_v9_uncond_adapter_isolated}"
    export TMUX_SESSION="${TMUX_SESSION:-${TASK}_dewo_v9_uncond}"
    export DEWO_SOURCE_SHA="${DEWO_SOURCE_SHA:-${FASTWAM_RESUME_SHA256:-s0}}"
    ;;
  scratch|lora)
    echo "[dewo-train] ERROR: DEWO v9 is INIT=s0 only (frozen mixed-S0 MoT)." >&2
    exit 2
    ;;
  *)
    echo "[dewo-train] ERROR: INIT must be s0, got ${INIT}" >&2
    exit 2
    ;;
esac
dewo_v2_assert_not_lora DEWO_TASK "${DEWO_TASK}"
dewo_v2_assert_not_lora DEWO_VARIANT "${DEWO_VARIANT}"
dewo_v2_assert_not_lora DEWO_PROTOCOL "${DEWO_PROTOCOL}"

[[ -z "${CALLER_LR}" ]] || hydra+=" learning_rate=${CALLER_LR}"
if [[ -n "${CALLER_TRAIN_MAX_STEPS}" ]]; then
  hydra+=" max_steps=${CALLER_TRAIN_MAX_STEPS}"
fi
[[ -z "${BATCH_SIZE:-}" ]] || hydra+=" batch_size=${BATCH_SIZE}"
[[ -z "${PRIMARY_PER_BATCH:-}" ]] || hydra+=" role_balanced_sampling.primary_per_batch=${PRIMARY_PER_BATCH}"
[[ -z "${ADAPTER_RANK:-}" ]] || hydra+=" model.uncond_adapter.rank=${ADAPTER_RANK}"
[[ -z "${ADAPTER_ALPHA:-}" ]] || hydra+=" model.uncond_adapter.alpha=${ADAPTER_ALPHA}"
[[ -z "${USER_HYDRA}" ]] || hydra+=" ${USER_HYDRA}"

export DEWO_OUTPUT_DIR="${CALLER_OUTPUT_DIR:-./runs/dexjoco_${TASK}_dewo_v9}"
export FITWAM_WANDB_GROUP="${CALLER_WANDB_GROUP:-${TASK}_dewo_v9_opensource}"
export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_${DEWO_VARIANT}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${TASK}_dewo_v9_${RUN_ID}}"
# Cluster proxy often makes wandb.init hang on rank 0 and desync NCCL.
export WANDB_MODE="${WANDB_MODE:-offline}"

dewo_v2_apply_vae_policy
if [[ "${USE_VAE_LATENT_CACHE:-1}" == "1" ]]; then
  hydra+=" model.load_vae=false model.fill_vae_latent_cache=false"
else
  hydra+=" model.load_vae=true model.fill_vae_latent_cache=false"
fi
export DEWO_HYDRA_OVERRIDES="${hydra}"
export VAE_ENCODE_VAL="${VAE_ENCODE_VAL:-false}"

# v9 recipe. Protocol env must not override these.
export CFG_SUCCESS_SUFFIX=' Successful execution.'
export CFG_FAILURE_SUFFIX=' Failed execution.'
export CFG_DROPOUT=0.0
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

echo "[dewo-train] TASK=${TASK} INIT=${INIT} DEWO_VERSION=${DEWO_VERSION} GPUS=${GPUS}"
echo "[dewo-train] DEWO_TASK=${DEWO_TASK}"
echo "[dewo-train] hydra=${DEWO_HYDRA_OVERRIDES}"
echo "[dewo-train] cfg primary=${CFG_PRIMARY_OUTCOME}/${CFG_PRIMARY_FAST}/${CFG_PRIMARY_BASE} aux_s=${CFG_AUX_SUCCESS_OUTCOME}/${CFG_AUX_SUCCESS_FAST}/${CFG_AUX_SUCCESS_BASE} aux_f=${CFG_AUX_FAIL_OUTCOME}/${CFG_AUX_FAIL_FAST}/${CFG_AUX_FAIL_BASE}"
echo "[dewo-train] cfg suffixes success=${CFG_SUCCESS_SUFFIX} failure=${CFG_FAILURE_SUFFIX}"
echo "[dewo-train] vae_cache=${USE_VAE_LATENT_CACHE:-1} encode_val=${VAE_ENCODE_VAL}"
echo "[dewo-train] tmux=${TMUX_SESSION}"
exec bash "${ROOT_DIR}/scripts/dewo_v2/train_run.sh"
