#!/usr/bin/env bash
# Internal DEWO v2 tmux worker. Prefer:
#   TASK=... INIT=scratch|s0 GPUS=... ENV_FILE=... bash scripts/dewo_v2/train.sh
#
# Full DiT only. There is no LoRA recipe.
#
# CFG env vars from prepare are forwarded into Hydra. Extra hydra knobs:
#   DEWO_HYDRA_OVERRIDES='eval_every=0'
set -euo pipefail

# Resolve the repository from this file, never from the caller's CWD. The
# physical path keeps logs and child processes consistent through symlinks.
SCRIPT_DIR="$(realpath -e -- "$(dirname -- "${BASH_SOURCE[0]}")")"
ROOT_DIR="$(realpath -e -- "${SCRIPT_DIR}/../..")"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

: "${EVE_MANIFEST_PATH:?source offline_v1_b1_jump_fast.env from prepare_pair_eve.sh first}"
: "${EVE_VAL_MANIFEST_PATH:?missing EVE_VAL_MANIFEST_PATH}"
: "${INIT_WEIGHTS:?missing INIT_WEIGHTS}"
: "${TEXT_EMBEDDING_CACHE_DIR:?missing TEXT_EMBEDDING_CACHE_DIR}"
: "${PRETRAINED_NORM_STATS:?missing PRETRAINED_NORM_STATS (OPEN artifacts dataset_stats.json)}"

dewo_v2_require_task
dewo_v2_require_gpus
dewo_v2_align_opensource_stack
dewo_v2_assert_not_lora DEWO_TASK "${DEWO_TASK:-}"
dewo_v2_assert_not_lora DEWO_VARIANT "${DEWO_VARIANT:-}"
TMUX_SESSION="${TMUX_SESSION:-${TASK}_dewo_v2_train}"
EXP_ROOT="${B1_VIDEO_EXPERIMENT_ROOT:-${ROOT_DIR}/data/${TASK}_dewo_v2_opensource}"
LOG_DIR="${LOG_DIR:-${EXP_ROOT}/logs}"
# Default: pre-encode VAE cache. Opt-out: USE_VAE_LATENT_CACHE=0.
dewo_v2_apply_vae_policy
if [[ "${USE_VAE_LATENT_CACHE:-1}" == "1" ]]; then
  VAE_LATENT_CACHE_DIR="${VAE_LATENT_CACHE_DIR:-${EXP_ROOT}/vae_latent_cache}"
  mkdir -p "${VAE_LATENT_CACHE_DIR}"
  export VAE_LATENT_CACHE_DIR
else
  unset VAE_LATENT_CACHE_DIR || true
fi
RUN_INLINE="${RUN_INLINE:-0}"
# Val loop is off (eval_every=0); do not spend GPU time encoding val latents.
VAE_ENCODE_VAL="${VAE_ENCODE_VAL:-false}"
DEWO_HYDRA_OVERRIDES="${DEWO_HYDRA_OVERRIDES:-}"
if [[ "${USE_VAE_LATENT_CACHE:-1}" != "1" ]]; then
  DEWO_HYDRA_OVERRIDES="${DEWO_HYDRA_OVERRIDES} model.load_vae=true model.fill_vae_latent_cache=false"
fi
mkdir -p "${LOG_DIR}"

ENV_FILE="${PROTOCOL_BUNDLE_PATH%/*}/offline_v1_b1_jump_fast.env"
if [[ -f "${ENV_FILE}" && "${USE_VAE_LATENT_CACHE:-1}" == "1" ]]; then
  if ! grep -q 'VAE_LATENT_CACHE_DIR=' "${ENV_FILE}"; then
    {
      echo "export VAE_LATENT_CACHE_DIR=${VAE_LATENT_CACHE_DIR}"
      echo "export REQUIRE_VAE_LATENT_CACHE=${REQUIRE_VAE_LATENT_CACHE}"
      echo "export FILL_VAE_LATENT_CACHE=${FILL_VAE_LATENT_CACHE}"
    } >> "${ENV_FILE}"
  fi
fi

# Do not re-export empty VAE_LATENT_CACHE_DIR (would override oc.env default null).
export REQUIRE_VAE_LATENT_CACHE
export FILL_VAE_LATENT_CACHE
export SKIP_VAE_PREENCODE
export CUDA_VISIBLE_DEVICES="${GPUS}"
export DEWO_TASK="${DEWO_TASK:-dexjoco/dexjoco_dewo_v2_offline_b1_jump_fast_full_1e-4}"
export DEWO_VARIANT="${DEWO_VARIANT:-B1-jump-fast-full-1e-4-scratch}"
dewo_v2_assert_not_lora DEWO_TASK "${DEWO_TASK}"
dewo_v2_assert_not_lora DEWO_VARIANT "${DEWO_VARIANT}"
export DEWO_OUTPUT_DIR="${DEWO_OUTPUT_DIR:-./runs/dexjoco_${TASK}_dewo_v2}"
export FITWAM_WANDB_GROUP="${FITWAM_WANDB_GROUP:-${TASK}_dewo_v2_opensource}"
export WANDB_MODE="${WANDB_MODE:-online}"
export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_${DEWO_VARIANT}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${TASK}_dewo_v2_${RUN_ID}}"
export DEWO_HYDRA_OVERRIDES

WORKER="${LOG_DIR}/tmux_train_${RUN_ID}.sh"
MASTER_LOG="${LOG_DIR}/tmux_train_${RUN_ID}.log"

cat > "${WORKER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT_DIR}"
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FITWAM_ENV:-fastwam}"
ENV_PREFIX="${FITWAM_ENV_PREFIX:-$(conda info --base)/envs/fastwam}"
export PATH="\${ENV_PREFIX}/bin:\${PATH}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:\${PYTHONPATH:-}"
export FITWAM_ENV_PREFIX="\${ENV_PREFIX}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export EVE_MANIFEST_PATH="${EVE_MANIFEST_PATH}"
export EVE_VAL_MANIFEST_PATH="${EVE_VAL_MANIFEST_PATH}"
export INIT_WEIGHTS="${INIT_WEIGHTS}"
export SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${INIT_WEIGHTS}}"
export FASTWAM_SOURCE_CONFIG="${FASTWAM_SOURCE_CONFIG}"
export SOURCE_BUNDLE_MANIFEST="${SOURCE_BUNDLE_MANIFEST}"
export PROTOCOL_BUNDLE_PATH="${PROTOCOL_BUNDLE_PATH}"
export BASE_DATASET="${BASE_DATASET}"
export ROLLOUT_RAW="${ROLLOUT_RAW}"
export PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS}"
export NORM_STATS_SOURCE="${NORM_STATS_SOURCE:-compute}"
export NORM_STATS_META_DIR="${NORM_STATS_META_DIR:-}"
export NORM_STATS_BUNDLE_SHA256="${NORM_STATS_BUNDLE_SHA256}"
export TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR}"
export TEXT_EMBEDDING_CACHE_SHA256="${TEXT_EMBEDDING_CACHE_SHA256:-}"
if [[ "${USE_VAE_LATENT_CACHE:-1}" == "1" ]]; then
  # :- avoids set -u when online-VAE mode unsets the cache dir before heredoc expand.
  export VAE_LATENT_CACHE_DIR="${VAE_LATENT_CACHE_DIR:-}"
else
  unset VAE_LATENT_CACHE_DIR || true
fi
export REQUIRE_VAE_LATENT_CACHE="${REQUIRE_VAE_LATENT_CACHE}"
export FILL_VAE_LATENT_CACHE="${FILL_VAE_LATENT_CACHE}"
export SKIP_VAE_PREENCODE="${SKIP_VAE_PREENCODE}"
export USE_VAE_LATENT_CACHE="${USE_VAE_LATENT_CACHE:-1}"
export OPEN_REPO="${OPEN_REPO:-${ROOT_DIR}/../FastWAM-infer-in-DexJoco}"
export DEWO_TASK="${DEWO_TASK}"
export DEWO_VARIANT="${DEWO_VARIANT}"
export DEWO_OUTPUT_DIR="${DEWO_OUTPUT_DIR}"
export DEWO_PROTOCOL="${DEWO_PROTOCOL:-}"
export FITWAM_WANDB_GROUP="${FITWAM_WANDB_GROUP}"
export WANDB_MODE="${WANDB_MODE}"
export RUN_ID="${RUN_ID}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME}"
export DEWO_HYDRA_OVERRIDES="${DEWO_HYDRA_OVERRIDES}"
export DEWO_VERSION="${DEWO_VERSION:-v2}"
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

log() { echo "[dewo-v2-tmux \$(date -Is)] \$*" | tee -a "${MASTER_LOG}"; }

log "gpus=${GPUS} task=${DEWO_TASK}"
log "manifest=${EVE_MANIFEST_PATH}"
log "init=${INIT_WEIGHTS}"
log "vae_cache=\${VAE_LATENT_CACHE_DIR:-<none>}"
log "use_vae_latent_cache=\${USE_VAE_LATENT_CACHE} fill=\${FILL_VAE_LATENT_CACHE} require=\${REQUIRE_VAE_LATENT_CACHE} skip_preencode=\${SKIP_VAE_PREENCODE}"
log "cfg primary=${CFG_PRIMARY_OUTCOME}/${CFG_PRIMARY_FAST}/${CFG_PRIMARY_BASE} aux_s=${CFG_AUX_SUCCESS_OUTCOME}/${CFG_AUX_SUCCESS_FAST}/${CFG_AUX_SUCCESS_BASE} aux_f=${CFG_AUX_FAIL_OUTCOME}/${CFG_AUX_FAIL_FAST}/${CFG_AUX_FAIL_BASE}"
log "cfg suffixes success=${CFG_SUCCESS_SUFFIX} failure=${CFG_FAILURE_SUFFIX} dewo_version=${DEWO_VERSION}"
log "vae_encode_val=${VAE_ENCODE_VAL} hydra_overrides=\${DEWO_HYDRA_OVERRIDES:-<none>}"

if [[ "\${SKIP_VAE_PREENCODE}" != "1" ]]; then
  if [[ -z "\${VAE_LATENT_CACHE_DIR:-}" ]]; then
    log "ERROR: VAE pre-encode requested but VAE_LATENT_CACHE_DIR is empty"
    exit 2
  fi
  log "VAE pre-encode start"
  IFS=',' read -r -a gpu_arr <<< "${GPUS}"
  nproc="\${#gpu_arr[@]}"
  workers_per_gpu="\${VAE_WORKERS_PER_GPU:-1}"
  mkdir -p "\${VAE_LATENT_CACHE_DIR}"
  if [[ "\${nproc}" -gt 1 || "\${workers_per_gpu}" -gt 1 ]]; then
    world=\$((nproc * workers_per_gpu))
    log "VAE workers: gpus=\${nproc} workers_per_gpu=\${workers_per_gpu} world=\${world}"
    pids=()
    for gpu_i in "\${!gpu_arr[@]}"; do
      gpu_id="\${gpu_arr[\$gpu_i]}"
      for replica in \$(seq 0 \$((workers_per_gpu - 1))); do
        shard_rank=\$((gpu_i * workers_per_gpu + replica))
        (
          export CUDA_VISIBLE_DEVICES="\${gpu_id}"
          export WORLD_SIZE=1
          unset RANK LOCAL_RANK MASTER_ADDR MASTER_PORT GROUP_RANK LOCAL_WORLD_SIZE
          python "${ROOT_DIR}/scripts/precompute_vae_latents.py" \\
            "task=\${DEWO_TASK}" \\
            "+vae_latent_cache_dir=\${VAE_LATENT_CACHE_DIR}" \\
            "+vae_shard_rank=\${shard_rank}" \\
            "+vae_shard_world=\${world}" \\
            "+encode_val=${VAE_ENCODE_VAL}" \\
            2>&1 | tee -a "${LOG_DIR}/precompute_vae_latents.shard\${shard_rank}.log"
        ) &
        pids+=("\$!")
      done
    done
    fail=0
    for pid in "\${pids[@]}"; do
      if ! wait "\$pid"; then
        fail=1
      fi
    done
    if [[ "\$fail" -ne 0 ]]; then
      log "ERROR: one or more VAE shard workers failed"
      exit 2
    fi
  else
    python "${ROOT_DIR}/scripts/precompute_vae_latents.py" \\
      "task=\${DEWO_TASK}" \\
      "+vae_latent_cache_dir=\${VAE_LATENT_CACHE_DIR}" \\
      "+encode_val=${VAE_ENCODE_VAL}" \\
      2>&1 | tee -a "${LOG_DIR}/precompute_vae_latents.log"
  fi
  log "VAE pre-encode done"
else
  log "SKIP_VAE_PREENCODE=1; online VAE (fill=\${FILL_VAE_LATENT_CACHE} cache=\${VAE_LATENT_CACHE_DIR:-<none>})"
fi

if [[ "\${REQUIRE_VAE_LATENT_CACHE}" == "1" ]]; then
  n_vae="\$(find "\${VAE_LATENT_CACHE_DIR}" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
  log "vae_latent_files=\${n_vae}"
  if [[ "\${n_vae}" -lt 1 && "\${FILL_VAE_LATENT_CACHE}" != "1" ]]; then
    log "ERROR: no VAE latent cache files under \${VAE_LATENT_CACHE_DIR} and fill mode is off"
    exit 2
  fi
else
  log "REQUIRE_VAE_LATENT_CACHE=0; skipping cache file preflight"
fi

if [[ "${DEWO_VERSION:-v2}" == "v6" || "${DEWO_VERSION:-v2}" == "v7" || "${DEWO_VERSION:-v2}" == "v8" || "${DEWO_VERSION:-v2}" == "v9" ]]; then
  log "${DEWO_VERSION}: precompute Successful/Failed text embeds (overwrite=false)"
  python "${ROOT_DIR}/scripts/precompute_text_embeds.py" \\
    "task=${DEWO_TASK}" \\
    "+overwrite=false" \\
    2>&1 | tee -a "${LOG_DIR}/precompute_text_embeds_${DEWO_VERSION}.log"
  python "${ROOT_DIR}/scripts/export_text_embed_cache_npz.py" \\
    --cache-dir "\${TEXT_EMBEDDING_CACHE_DIR}" \\
    2>&1 | tee -a "${LOG_DIR}/export_text_embed_cache_npz_${DEWO_VERSION}.log"
fi

log "root=${ROOT_DIR}"
log "launching DEWO train"
bash "${ROOT_DIR}/scripts/dewo/train.sh" \\
  \${DEWO_HYDRA_OVERRIDES} \\
  2>&1 | tee -a "${LOG_DIR}/train_b1_jump_fast.log"
log "train finished EXIT=\$?"
EOF
chmod +x "${WORKER}"

if [[ "${RUN_INLINE}" == "1" ]]; then
  echo "[dewo-v2-train] running train inline (use_cache=${USE_VAE_LATENT_CACHE:-1} fill=${FILL_VAE_LATENT_CACHE} skip_vae_preencode=${SKIP_VAE_PREENCODE})"
  exec bash "${WORKER}"
fi

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "[dewo-v2-train] tmux session '${TMUX_SESSION}' already exists."
  echo "  attach: tmux attach -t ${TMUX_SESSION}"
  echo "  or set TMUX_SESSION=other_name"
  exit 1
fi

tmux new-session -d -s "${TMUX_SESSION}" "bash '${WORKER}'"
echo "[dewo-v2-train] started tmux session: ${TMUX_SESSION}"
echo "[dewo-v2-train] log: ${MASTER_LOG}"
echo "[dewo-v2-train] attach: tmux attach -t ${TMUX_SESSION}"
echo "[dewo-v2-train] VAE: use_cache=${USE_VAE_LATENT_CACHE:-1} cache=${VAE_LATENT_CACHE_DIR:-<none>} skip_preencode=${SKIP_VAE_PREENCODE}"
