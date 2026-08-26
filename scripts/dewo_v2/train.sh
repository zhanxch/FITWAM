#!/usr/bin/env bash
# DEWO trainer. Task/data come from ENV_FILE. Hyperparams are extras.
#   INIT=scratch|s0
#   DEWO_VERSION=v2|v5|v6|v7|v8|v9
#     v5: freeze the whole base; train a CFG-condition adapter on success text
#     v6: v5 frozen-adapter shell + D0/D+/D_fail shuffle pool (Successful / Failed)
#     v7: v6 pool; CFG residual is ε_+ − ε_- (failure action BC defines ε_-)
#     v8: v6 pool + mix; frozen-VAE value head drop-edge-gates sparse CFG
#     v9: v8 mix; VideoDiT V vs progress G_t; drop-only gate; full pair rollouts
#
#   TASK=fold_glasses INIT=scratch|s0 GPUS=0,1,2,3 \
#     ENV_FILE=data/<task>_dewo_v2_pair_*/eve_v02/protocol/offline_v1_b1_jump_fast.env \
#     bash scripts/dewo_v2/train.sh
#   DEWO_VERSION=v5 INIT=s0 TRAIN_MAX_STEPS=1500 \
#     bash scripts/dewo_v2/train.sh
set -euo pipefail

# Resolve the repository from this file, not from the caller's current
# directory. This keeps relative and absolute launch reports identical.
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

INIT="${INIT:-scratch}"
DEWO_VERSION="${DEWO_VERSION:-v2}"
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
CALLER_CFG_PRIMARY="${CFG_PRIMARY:-}"
CALLER_CFG_AUX_SUCCESS="${CFG_AUX_SUCCESS:-}"
CALLER_CFG_AUX_FAIL="${CFG_AUX_FAIL:-}"
CALLER_CFG_SUCCESS_SUFFIX="${CFG_SUCCESS_SUFFIX:-}"
CALLER_CFG_FAILURE_SUFFIX="${CFG_FAILURE_SUFFIX:-}"
CALLER_CFG_PRIMARY_OUTCOME="${CFG_PRIMARY_OUTCOME:-}"
CALLER_CFG_PRIMARY_FAST="${CFG_PRIMARY_FAST:-}"
CALLER_CFG_PRIMARY_BASE="${CFG_PRIMARY_BASE:-}"
CALLER_CFG_AUX_SUCCESS_OUTCOME="${CFG_AUX_SUCCESS_OUTCOME:-}"
CALLER_CFG_AUX_SUCCESS_FAST="${CFG_AUX_SUCCESS_FAST:-}"
CALLER_CFG_AUX_SUCCESS_BASE="${CFG_AUX_SUCCESS_BASE:-}"
CALLER_CFG_AUX_FAIL_OUTCOME="${CFG_AUX_FAIL_OUTCOME:-}"
CALLER_CFG_AUX_FAIL_FAST="${CFG_AUX_FAIL_FAST:-}"
CALLER_CFG_AUX_FAIL_BASE="${CFG_AUX_FAIL_BASE:-}"
USER_HYDRA="${DEWO_HYDRA_OVERRIDES:-}"

# Prepare env variant is not a training recipe name.
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
[[ -z "${CALLER_EVE_MANIFEST_PATH}" ]] || export EVE_MANIFEST_PATH="${CALLER_EVE_MANIFEST_PATH}"
[[ -z "${CALLER_EVE_VAL_MANIFEST_PATH}" ]] || export EVE_VAL_MANIFEST_PATH="${CALLER_EVE_VAL_MANIFEST_PATH}"
[[ -z "${CALLER_USE_VAE}" ]] || export USE_VAE_LATENT_CACHE="${CALLER_USE_VAE}"
[[ -z "${CALLER_VAE_CACHE_DIR}" ]] || export VAE_LATENT_CACHE_DIR="${CALLER_VAE_CACHE_DIR}"
[[ -z "${CALLER_PRETRAINED_NORM_STATS}" ]] || export PRETRAINED_NORM_STATS="${CALLER_PRETRAINED_NORM_STATS}"
dewo_v2_align_opensource_stack

if [[ "${DEWO_VERSION}" != "v2" && "${DEWO_VERSION}" != "v5" && "${DEWO_VERSION}" != "v6" && "${DEWO_VERSION}" != "v7" && "${DEWO_VERSION}" != "v8" && "${DEWO_VERSION}" != "v9" ]]; then
  echo "[dewo-train] ERROR: DEWO_VERSION must be v2|v5|v6|v7|v8|v9, got ${DEWO_VERSION}" >&2
  exit 2
fi
export DEWO_VERSION

case "${INIT}" in
  s0)
    hydra="eval_every=0"
    if [[ "${DEWO_VERSION}" == "v9" ]]; then
      export DEWO_TASK="dexjoco/dexjoco_dewo_v9_offline_b1_jump_fast_uncond"
      export DEWO_VARIANT="${CALLER_DEWO_VARIANT:-B1-jump-fast-v9-uncond-adapter}"
      export DEWO_PROTOCOL="${CALLER_DEWO_PROTOCOL:-${TASK}_dewo_v9_uncond_adapter_isolated}"
      export TMUX_SESSION="${TMUX_SESSION:-${TASK}_dewo_v9_uncond}"
    elif [[ "${DEWO_VERSION}" == "v8" ]]; then
      export DEWO_TASK="dexjoco/dexjoco_dewo_v8_offline_b1_jump_fast_uncond"
      export DEWO_VARIANT="${CALLER_DEWO_VARIANT:-B1-jump-fast-v8-uncond-adapter}"
      export DEWO_PROTOCOL="${CALLER_DEWO_PROTOCOL:-${TASK}_dewo_v8_uncond_adapter_isolated}"
      export TMUX_SESSION="${TMUX_SESSION:-${TASK}_dewo_v8_uncond}"
    elif [[ "${DEWO_VERSION}" == "v7" ]]; then
      export DEWO_TASK="dexjoco/dexjoco_dewo_v7_offline_b1_jump_fast_uncond"
      export DEWO_VARIANT="${CALLER_DEWO_VARIANT:-B1-jump-fast-v7-uncond-adapter}"
      export DEWO_PROTOCOL="${CALLER_DEWO_PROTOCOL:-${TASK}_dewo_v7_uncond_adapter_isolated}"
      export TMUX_SESSION="${TMUX_SESSION:-${TASK}_dewo_v7_uncond}"
    elif [[ "${DEWO_VERSION}" == "v6" ]]; then
      export DEWO_TASK="dexjoco/dexjoco_dewo_v6_offline_b1_jump_fast_uncond"
      export DEWO_VARIANT="${CALLER_DEWO_VARIANT:-B1-jump-fast-v6-uncond-adapter}"
      export DEWO_PROTOCOL="${CALLER_DEWO_PROTOCOL:-${TASK}_dewo_v6_uncond_adapter_isolated}"
      export TMUX_SESSION="${TMUX_SESSION:-${TASK}_dewo_v6_uncond}"
    elif [[ "${DEWO_VERSION}" == "v5" ]]; then
      export DEWO_TASK="dexjoco/dexjoco_dewo_v5_offline_b1_jump_fast_uncond"
      export DEWO_VARIANT="${CALLER_DEWO_VARIANT:-B1-jump-fast-v5-uncond-adapter}"
      export DEWO_PROTOCOL="${CALLER_DEWO_PROTOCOL:-${TASK}_dewo_v5_uncond_adapter_isolated}"
      export TMUX_SESSION="${TMUX_SESSION:-${TASK}_dewo_v5_uncond}"
    else
      export DEWO_TASK="dexjoco/dexjoco_dewo_v2_offline_b1_jump_fast_full_s0"
      export DEWO_VARIANT="${CALLER_DEWO_VARIANT:-B1-jump-fast-full-s0}"
      export DEWO_PROTOCOL="${CALLER_DEWO_PROTOCOL:-${TASK}_dewo_v2_jump_fast_full_from_s0}"
      export TMUX_SESSION="${TMUX_SESSION:-${TASK}_dewo_v2_s0}"
    fi
    export DEWO_SOURCE_SHA="${DEWO_SOURCE_SHA:-${FASTWAM_RESUME_SHA256:-s0}}"
    ;;
  scratch)
    if [[ "${DEWO_VERSION}" == "v5" || "${DEWO_VERSION}" == "v6" || "${DEWO_VERSION}" == "v7" || "${DEWO_VERSION}" == "v8" || "${DEWO_VERSION}" == "v9" ]]; then
      echo "[dewo-train] ERROR: DEWO_VERSION=${DEWO_VERSION} is S0-continue only. Use INIT=s0." >&2
      exit 2
    fi
    hydra="eval_every=0 resume=null"
    export DEWO_TASK="dexjoco/dexjoco_dewo_v2_offline_b1_jump_fast_full_1e-4"
    export DEWO_VARIANT="${CALLER_DEWO_VARIANT:-B1-jump-fast-full-1e-4-scratch}"
    export DEWO_PROTOCOL="${CALLER_DEWO_PROTOCOL:-${TASK}_dewo_v2_jump_fast_full_1e-4_from_scratch}"
    export DEWO_SOURCE_SHA="${DEWO_SOURCE_SHA:-from_scratch_wan_actiondit}"
    export TMUX_SESSION="${TMUX_SESSION:-${TASK}_dewo_v2_scratch}"
    ;;
  lora)
    echo "[dewo-train] ERROR: INIT=lora is removed. Use INIT=scratch or INIT=s0." >&2
    exit 2
    ;;
  *)
    echo "[dewo-train] ERROR: INIT must be scratch|s0, got ${INIT}" >&2
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
if [[ "${DEWO_VERSION}" == "v5" || "${DEWO_VERSION}" == "v6" || "${DEWO_VERSION}" == "v7" || "${DEWO_VERSION}" == "v8" || "${DEWO_VERSION}" == "v9" ]]; then
  [[ -z "${ADAPTER_RANK:-}" ]] || hydra+=" model.uncond_adapter.rank=${ADAPTER_RANK}"
  [[ -z "${ADAPTER_ALPHA:-}" ]] || hydra+=" model.uncond_adapter.alpha=${ADAPTER_ALPHA}"
fi
[[ -z "${USER_HYDRA}" ]] || hydra+=" ${USER_HYDRA}"

if [[ "${DEWO_VERSION}" == "v9" ]]; then
  export DEWO_OUTPUT_DIR="${CALLER_OUTPUT_DIR:-./runs/dexjoco_${TASK}_dewo_v9}"
  export FITWAM_WANDB_GROUP="${CALLER_WANDB_GROUP:-${TASK}_dewo_v9_opensource}"
elif [[ "${DEWO_VERSION}" == "v8" ]]; then
  export DEWO_OUTPUT_DIR="${CALLER_OUTPUT_DIR:-./runs/dexjoco_${TASK}_dewo_v8}"
  export FITWAM_WANDB_GROUP="${CALLER_WANDB_GROUP:-${TASK}_dewo_v8_opensource}"
elif [[ "${DEWO_VERSION}" == "v7" ]]; then
  export DEWO_OUTPUT_DIR="${CALLER_OUTPUT_DIR:-./runs/dexjoco_${TASK}_dewo_v7}"
  export FITWAM_WANDB_GROUP="${CALLER_WANDB_GROUP:-${TASK}_dewo_v7_opensource}"
elif [[ "${DEWO_VERSION}" == "v6" ]]; then
  export DEWO_OUTPUT_DIR="${CALLER_OUTPUT_DIR:-./runs/dexjoco_${TASK}_dewo_v6}"
  export FITWAM_WANDB_GROUP="${CALLER_WANDB_GROUP:-${TASK}_dewo_v6_opensource}"
elif [[ "${DEWO_VERSION}" == "v5" ]]; then
  export DEWO_OUTPUT_DIR="${CALLER_OUTPUT_DIR:-./runs/dexjoco_${TASK}_dewo_v5}"
  export FITWAM_WANDB_GROUP="${CALLER_WANDB_GROUP:-${TASK}_dewo_v5_opensource}"
else
  export DEWO_OUTPUT_DIR="${CALLER_OUTPUT_DIR:-./runs/dexjoco_${TASK}_dewo_v2}"
  export FITWAM_WANDB_GROUP="${CALLER_WANDB_GROUP:-${TASK}_dewo_v2_opensource}"
fi
export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_${DEWO_VARIANT}}"
if [[ -z "${WANDB_RUN_NAME:-}" ]]; then
  if [[ "${DEWO_VERSION}" == "v9" ]]; then
    export WANDB_RUN_NAME="${TASK}_dewo_v9_${RUN_ID}"
  elif [[ "${DEWO_VERSION}" == "v8" ]]; then
    export WANDB_RUN_NAME="${TASK}_dewo_v8_${RUN_ID}"
  elif [[ "${DEWO_VERSION}" == "v7" ]]; then
    export WANDB_RUN_NAME="${TASK}_dewo_v7_${RUN_ID}"
  elif [[ "${DEWO_VERSION}" == "v6" ]]; then
    export WANDB_RUN_NAME="${TASK}_dewo_v6_${RUN_ID}"
  elif [[ "${DEWO_VERSION}" == "v5" ]]; then
    export WANDB_RUN_NAME="${TASK}_dewo_v5_${RUN_ID}"
  else
    export WANDB_RUN_NAME="${TASK}_dewo_v2_${RUN_ID}"
  fi
fi

dewo_v2_apply_vae_policy
if [[ "${USE_VAE_LATENT_CACHE:-1}" == "1" ]]; then
  hydra+=" model.load_vae=false model.fill_vae_latent_cache=false"
else
  hydra+=" model.load_vae=true model.fill_vae_latent_cache=false"
fi
export DEWO_HYDRA_OVERRIDES="${hydra}"
export VAE_ENCODE_VAL="${VAE_ENCODE_VAL:-false}"

if [[ "${DEWO_VERSION}" == "v6" || "${DEWO_VERSION}" == "v7" || "${DEWO_VERSION}" == "v8" || "${DEWO_VERSION}" == "v9" ]]; then
  # ENV pair files stamp v2 Successful / null. Keep Successful; force Failed
  # unless the caller explicitly overrode the suffixes. Do not use Recovered.
  export CFG_SUCCESS_SUFFIX="${CALLER_CFG_SUCCESS_SUFFIX:- Successful execution.}"
  export CFG_FAILURE_SUFFIX="${CALLER_CFG_FAILURE_SUFFIX:- Failed execution.}"
  if [[ -z "${CALLER_CFG_PRIMARY}" ]]; then
    unset CFG_PRIMARY CFG_PRIMARY_OUTCOME CFG_PRIMARY_FAST CFG_PRIMARY_BASE || true
  fi
  if [[ -z "${CALLER_CFG_AUX_SUCCESS}" ]]; then
    unset CFG_AUX_SUCCESS CFG_AUX_SUCCESS_OUTCOME CFG_AUX_SUCCESS_FAST CFG_AUX_SUCCESS_BASE || true
  fi
  if [[ -z "${CALLER_CFG_AUX_FAIL}" ]]; then
    unset CFG_AUX_FAIL CFG_AUX_FAIL_OUTCOME CFG_AUX_FAIL_FAST CFG_AUX_FAIL_BASE || true
  fi
else
  export CFG_SUCCESS_SUFFIX="${CFG_SUCCESS_SUFFIX:- Successful execution.}"
  export CFG_FAILURE_SUFFIX="${CFG_FAILURE_SUFFIX:-null}"
fi
export CFG_DROPOUT="${CFG_DROPOUT:-0.0}"
if [[ "${DEWO_VERSION}" == "v6" || "${DEWO_VERSION}" == "v7" || "${DEWO_VERSION}" == "v8" || "${DEWO_VERSION}" == "v9" ]]; then
  # D+ action dropout matches v5 (0.9/0/0.1). D0 uses the base schedule in
  # the dataset, so this triple never suffixes success episodes. D_fail is
  # 1.0/0/0 Failed with no base dropout. No FAST. ENV v2 triples must not leak.
  if [[ -n "${CALLER_CFG_PRIMARY}" ]]; then
    IFS=',' read -r _o _f _b <<< "${CALLER_CFG_PRIMARY}"
    export CFG_PRIMARY_OUTCOME="${_o}"
    export CFG_PRIMARY_FAST="${_f}"
    export CFG_PRIMARY_BASE="${_b}"
  else
    export CFG_PRIMARY_OUTCOME=0.9
    export CFG_PRIMARY_FAST=0.0
    export CFG_PRIMARY_BASE=0.1
  fi
  if [[ -n "${CALLER_CFG_AUX_SUCCESS}" ]]; then
    IFS=',' read -r _o _f _b <<< "${CALLER_CFG_AUX_SUCCESS}"
    export CFG_AUX_SUCCESS_OUTCOME="${_o}"
    export CFG_AUX_SUCCESS_FAST="${_f}"
    export CFG_AUX_SUCCESS_BASE="${_b}"
  else
    export CFG_AUX_SUCCESS_OUTCOME=1.0
    export CFG_AUX_SUCCESS_FAST=0.0
    export CFG_AUX_SUCCESS_BASE=0.0
  fi
  if [[ -n "${CALLER_CFG_AUX_FAIL}" ]]; then
    IFS=',' read -r _o _f _b <<< "${CALLER_CFG_AUX_FAIL}"
    export CFG_AUX_FAIL_OUTCOME="${_o}"
    export CFG_AUX_FAIL_FAST="${_f}"
    export CFG_AUX_FAIL_BASE="${_b}"
  else
    export CFG_AUX_FAIL_OUTCOME=1.0
    export CFG_AUX_FAIL_FAST=0.0
    export CFG_AUX_FAIL_BASE=0.0
  fi
elif [[ "${DEWO_VERSION}" == "v5" ]]; then
  # Action-text dropout only (p=0.1 base). Video text stays on the S0
  # base prompt. Failure / aux_success are not sampled. FAST unused.
  if [[ -n "${CALLER_CFG_PRIMARY}" ]]; then
    IFS=',' read -r _o _f _b <<< "${CALLER_CFG_PRIMARY}"
    export CFG_PRIMARY_OUTCOME="${_o}"
    export CFG_PRIMARY_FAST="${_f}"
    export CFG_PRIMARY_BASE="${_b}"
  else
    export CFG_PRIMARY_OUTCOME=0.9
    export CFG_PRIMARY_FAST=0.0
    export CFG_PRIMARY_BASE=0.1
  fi
  if [[ -n "${CALLER_CFG_AUX_SUCCESS}" ]]; then
    IFS=',' read -r _o _f _b <<< "${CALLER_CFG_AUX_SUCCESS}"
    export CFG_AUX_SUCCESS_OUTCOME="${_o}"
    export CFG_AUX_SUCCESS_FAST="${_f}"
    export CFG_AUX_SUCCESS_BASE="${_b}"
  else
    export CFG_AUX_SUCCESS_OUTCOME=1.0
    export CFG_AUX_SUCCESS_FAST=0.0
    export CFG_AUX_SUCCESS_BASE=0.0
  fi
  if [[ -n "${CALLER_CFG_AUX_FAIL}" ]]; then
    IFS=',' read -r _o _f _b <<< "${CALLER_CFG_AUX_FAIL}"
    export CFG_AUX_FAIL_OUTCOME="${_o}"
    export CFG_AUX_FAIL_FAST="${_f}"
    export CFG_AUX_FAIL_BASE="${_b}"
  else
    export CFG_AUX_FAIL_OUTCOME=0.0
    export CFG_AUX_FAIL_FAST=0.0
    export CFG_AUX_FAIL_BASE=1.0
  fi
else
  export CFG_PRIMARY_OUTCOME="${CFG_PRIMARY_OUTCOME:-0.5}"
  export CFG_PRIMARY_FAST="${CFG_PRIMARY_FAST:-0.0}"
  export CFG_PRIMARY_BASE="${CFG_PRIMARY_BASE:-0.5}"
  export CFG_AUX_SUCCESS_OUTCOME="${CFG_AUX_SUCCESS_OUTCOME:-0.4}"
  export CFG_AUX_SUCCESS_FAST="${CFG_AUX_SUCCESS_FAST:-0.2}"
  export CFG_AUX_SUCCESS_BASE="${CFG_AUX_SUCCESS_BASE:-0.4}"
  export CFG_AUX_FAIL_OUTCOME="${CFG_AUX_FAIL_OUTCOME:-0.0}"
  export CFG_AUX_FAIL_FAST="${CFG_AUX_FAIL_FAST:-0.2}"
  export CFG_AUX_FAIL_BASE="${CFG_AUX_FAIL_BASE:-0.4}"
fi
export CFG_FAST_MODEL_ID="${CFG_FAST_MODEL_ID:-physical-intelligence/fast}"
export CFG_FAST_MAX_TOKENS="${CFG_FAST_MAX_TOKENS:-32}"
export CFG_FAST_FAIL_CLOSED="${CFG_FAST_FAIL_CLOSED:-true}"

echo "[dewo-train] TASK=${TASK} INIT=${INIT} DEWO_VERSION=${DEWO_VERSION} GPUS=${GPUS}"
echo "[dewo-train] DEWO_TASK=${DEWO_TASK}"
echo "[dewo-train] hydra=${DEWO_HYDRA_OVERRIDES}"
echo "[dewo-train] cfg primary=${CFG_PRIMARY_OUTCOME}/${CFG_PRIMARY_FAST}/${CFG_PRIMARY_BASE} aux_s=${CFG_AUX_SUCCESS_OUTCOME}/${CFG_AUX_SUCCESS_FAST}/${CFG_AUX_SUCCESS_BASE} aux_f=${CFG_AUX_FAIL_OUTCOME}/${CFG_AUX_FAIL_FAST}/${CFG_AUX_FAIL_BASE}"
echo "[dewo-train] cfg suffixes success=${CFG_SUCCESS_SUFFIX} failure=${CFG_FAILURE_SUFFIX}"
echo "[dewo-train] vae_cache=${USE_VAE_LATENT_CACHE:-1} encode_val=${VAE_ENCODE_VAL}"
echo "[dewo-train] tmux=${TMUX_SESSION}"
exec bash "${ROOT_DIR}/scripts/dewo_v2/train_run.sh"
