#!/usr/bin/env bash
# Train fold_glasses DEWO v2 inside tmux (opensource stack).
#
# Default: train-time VAE fill (miss→encode→cache). Optional offline pre-encode
# remains available via FILL_VAE_LATENT_CACHE=0 SKIP_VAE_PREENCODE=0.
#
# Prerequisites: source prepare_dewo_v2_eve.sh env first, e.g.
#   source data/fold_glasses_dewo_v2_*/eve_v02/protocol/offline_v1_b1_jump_fast.env
#   GPUS=0,1,2,3 bash scripts/fold_glasses/train_dewo_v2_jump_fast_lora.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

: "${EVE_MANIFEST_PATH:?source offline_v1_b1_jump_fast.env from prepare_dewo_v2_eve.sh first}"
: "${EVE_VAL_MANIFEST_PATH:?missing EVE_VAL_MANIFEST_PATH}"
: "${INIT_WEIGHTS:?missing INIT_WEIGHTS}"
: "${TEXT_EMBEDDING_CACHE_DIR:?missing TEXT_EMBEDDING_CACHE_DIR}"
: "${PRETRAINED_NORM_STATS:?missing PRETRAINED_NORM_STATS (OPEN artifacts dataset_stats.json)}"

# Refuse local 384/min-max continue-train for dexjoco release ckpts.
if [[ "${BASE_DATASET:-}" == *fold_glasses_fastwam* ]]; then
  echo "[dewo-v2-train] ERROR: BASE_DATASET is local FitWAM (${BASE_DATASET}). Opensource stack required."
  exit 2
fi
if [[ "${PRETRAINED_NORM_STATS}" == */fold_glasses_fastwam/meta/* ]]; then
  echo "[dewo-v2-train] ERROR: PRETRAINED_NORM_STATS looks like local meta min/max."
  exit 2
fi

GPUS="${GPUS:-4,5,6,7}"
TMUX_SESSION="${TMUX_SESSION:-fold_dewo_v2_train}"
EXP_ROOT="${B1_VIDEO_EXPERIMENT_ROOT:-${ROOT_DIR}/data/fold_glasses_dewo_v2_opensource}"
LOG_DIR="${LOG_DIR:-${EXP_ROOT}/logs}"
VAE_LATENT_CACHE_DIR="${VAE_LATENT_CACHE_DIR:-${EXP_ROOT}/vae_latent_cache}"
FILL_VAE_LATENT_CACHE="${FILL_VAE_LATENT_CACHE:-1}"
# Fill mode skips offline pre-encode by default; set SKIP_VAE_PREENCODE=0 to force it.
if [[ "${FILL_VAE_LATENT_CACHE}" == "1" ]]; then
  SKIP_VAE_PREENCODE="${SKIP_VAE_PREENCODE:-1}"
  REQUIRE_VAE_LATENT_CACHE="${REQUIRE_VAE_LATENT_CACHE:-0}"
else
  SKIP_VAE_PREENCODE="${SKIP_VAE_PREENCODE:-0}"
  REQUIRE_VAE_LATENT_CACHE="${REQUIRE_VAE_LATENT_CACHE:-1}"
fi
RUN_INLINE="${RUN_INLINE:-0}"
VAE_ENCODE_VAL="${VAE_ENCODE_VAL:-true}"
DEWO_HYDRA_OVERRIDES="${DEWO_HYDRA_OVERRIDES:-}"
mkdir -p "${LOG_DIR}" "${VAE_LATENT_CACHE_DIR}"

# Persist VAE paths into the sourced env file when possible.
ENV_FILE="${PROTOCOL_BUNDLE_PATH%/*}/offline_v1_b1_jump_fast.env"
if [[ -f "${ENV_FILE}" ]]; then
  if ! grep -q 'VAE_LATENT_CACHE_DIR=' "${ENV_FILE}"; then
    {
      echo "export VAE_LATENT_CACHE_DIR=${VAE_LATENT_CACHE_DIR}"
      echo "export REQUIRE_VAE_LATENT_CACHE=${REQUIRE_VAE_LATENT_CACHE}"
      echo "export FILL_VAE_LATENT_CACHE=${FILL_VAE_LATENT_CACHE}"
    } >> "${ENV_FILE}"
  fi
fi

export VAE_LATENT_CACHE_DIR
export REQUIRE_VAE_LATENT_CACHE
export FILL_VAE_LATENT_CACHE
export CUDA_VISIBLE_DEVICES="${GPUS}"
export DEWO_TASK="${DEWO_TASK:-dexjoco/dexjoco_fold_glasses_offline_b1_jump_fast_lora_3e-5}"
export DEWO_VARIANT="${DEWO_VARIANT:-B1-jump-fast-lora}"
export DEWO_OUTPUT_DIR="${DEWO_OUTPUT_DIR:-./runs/dexjoco_fold_glasses_dewo_v2}"
export FITWAM_WANDB_GROUP="${FITWAM_WANDB_GROUP:-fold_glasses_dewo_v2_opensource}"
export WANDB_MODE="${WANDB_MODE:-online}"
export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_B1-jump-fast-lora}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-fold_glasses_dewo_v2_${RUN_ID}}"

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

# Re-export training env inside tmux.
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
export VAE_LATENT_CACHE_DIR="${VAE_LATENT_CACHE_DIR}"
export REQUIRE_VAE_LATENT_CACHE="${REQUIRE_VAE_LATENT_CACHE}"
export FILL_VAE_LATENT_CACHE="${FILL_VAE_LATENT_CACHE}"
export OPEN_REPO="${OPEN_REPO:-/data_all/xiangchengzhan/FastWAM-infer-in-DexJoco}"
export DEWO_TASK="${DEWO_TASK}"
export DEWO_VARIANT="${DEWO_VARIANT}"
export DEWO_OUTPUT_DIR="${DEWO_OUTPUT_DIR}"
export FITWAM_WANDB_GROUP="${FITWAM_WANDB_GROUP}"
export WANDB_MODE="${WANDB_MODE}"
export RUN_ID="${RUN_ID}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME}"

log() { echo "[dewo-v2-tmux \$(date -Is)] \$*" | tee -a "${MASTER_LOG}"; }

log "gpus=${GPUS} task=${DEWO_TASK}"
log "manifest=${EVE_MANIFEST_PATH}"
log "init=${INIT_WEIGHTS}"
log "vae_cache=${VAE_LATENT_CACHE_DIR}"
log "fill_vae_latent_cache=${FILL_VAE_LATENT_CACHE} require_vae=${REQUIRE_VAE_LATENT_CACHE}"
log "vae_encode_val=${VAE_ENCODE_VAL} hydra_overrides=${DEWO_HYDRA_OVERRIDES:-<none>}"

if [[ "${SKIP_VAE_PREENCODE}" != "1" ]]; then
  log "VAE pre-encode start"
  IFS=',' read -r -a gpu_arr <<< "${GPUS}"
  nproc="\${#gpu_arr[@]}"
  workers_per_gpu="\${VAE_WORKERS_PER_GPU:-1}"
  mkdir -p "${VAE_LATENT_CACHE_DIR}"
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
          python scripts/precompute_vae_latents.py \\
            "task=${DEWO_TASK}" \\
            "+vae_latent_cache_dir=${VAE_LATENT_CACHE_DIR}" \\
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
    python scripts/precompute_vae_latents.py \\
      "task=${DEWO_TASK}" \\
      "+vae_latent_cache_dir=${VAE_LATENT_CACHE_DIR}" \\
      "+encode_val=${VAE_ENCODE_VAL}" \\
      2>&1 | tee -a "${LOG_DIR}/precompute_vae_latents.log"
  fi
  log "VAE pre-encode done"
else
  log "SKIP_VAE_PREENCODE=1; train-time fill=\${FILL_VAE_LATENT_CACHE} cache=${VAE_LATENT_CACHE_DIR}"
fi

n_vae="\$(find "${VAE_LATENT_CACHE_DIR}" -name '*.pt' 2>/dev/null | wc -l | tr -d ' ')"
log "vae_latent_files=\${n_vae}"
if [[ "\${n_vae}" -lt 1 && "${FILL_VAE_LATENT_CACHE}" != "1" ]]; then
  log "ERROR: no VAE latent cache files under ${VAE_LATENT_CACHE_DIR} and fill mode is off"
  exit 2
fi

log "launching DEWO v2 train"
bash scripts/dewo/train.sh \\
  ${DEWO_HYDRA_OVERRIDES} \\
  2>&1 | tee -a "${LOG_DIR}/train_b1_jump_fast_lora.log"
log "train finished EXIT=\$?"
EOF
chmod +x "${WORKER}"

if [[ "${RUN_INLINE}" == "1" ]]; then
  echo "[dewo-v2-train] running train inline (fill=${FILL_VAE_LATENT_CACHE} skip_vae_preencode=${SKIP_VAE_PREENCODE})"
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
echo "[dewo-v2-train] VAE cache: ${VAE_LATENT_CACHE_DIR}"
echo "[dewo-v2-train] fill_vae=${FILL_VAE_LATENT_CACHE} skip_preencode=${SKIP_VAE_PREENCODE}"
