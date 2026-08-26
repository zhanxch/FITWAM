# Shared bash helpers for DexJoCo DEWO v2. Source from other scripts:
#   source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"
#   dewo_v2_require_gpus
#   dewo_v2_load_task "${TASK}"
#
# Session knobs (never bake into a new .sh):
#   GPUS, WAIT_IDLE, RUN_DIR, CKPT, OUT_ROOT, BASE_PORT
# Task identity lives in scripts/dewo_v2/tasks.py.
# load_task pins the opensource 224 / z-score stack (OPEN yaml + artifacts stats).
#
# VAE policy (default = pre-encode + read cache at train):
#   USE_VAE_LATENT_CACHE=0  opt out to online VAE encode per step
#   Default: SKIP_VAE_PREENCODE=0, REQUIRE=1, FILL=0
# Val encode is off unless VAE_ENCODE_VAL=true (train eval_every=0).
#
# Training recipes: INIT=scratch|s0. DEWO_VERSION=v2 (default), v5, v6, v7, v8, or v9.
# v5 freezes the whole base and trains a CFG-condition adapter on ordinary
# success text; CFG mixes adapter-off + base text (S0) with adapter-on + success.
# v6 keeps the v5 frozen-adapter shell. One shuffle pool: D0 success episodes
# (base alignment) + D+ success-event windows (Successful, action BC) +
# D_fail failure events (Failed, video BC). Residual is still ε_+ − ε_0.
# v7 same pool as v6, but D_fail action BC defines ε_- and the mix is
# ε_0 + w(ε_+ − ε_-). No Recovered suffix, no 12:4.
# v8 same pool and mix as v6, plus a frozen-VAE value head that drop-edge
# gates sparse CFG. D0 is one episode per 4/4 all-success seed.
# v9 same pool/mix as v8, but V is pooled VideoDiT tokens vs progress
# return G_t=γ^{T-t} (fail=0). Gate is drop-only (no v_high floor).
# Pair data is full-horizon success stitch + fail cliff, not 33-frame crops.
#
# CFG knobs (hammer_nail defaults unless overridden):
#   CFG_PRIMARY=0.5,0.0,0.5          # outcome,fast,base — primary.fast MUST be 0
#   CFG_AUX_SUCCESS=0.4,0.2,0.4      # FAST only on aux channels by default
#   CFG_AUX_FAIL=0.0,0.2,0.4
#   CFG_SUCCESS_SUFFIX=' Successful execution.'
#   CFG_FAILURE_SUFFIX=null
#   CFG_DROPOUT=0.0
#   CFG_FAST_FAIL_CLOSED=1
#
# Text embeds:
#   - base + outcome: always needed (expert/primary uses outcome+base)
#   - FAST: only when any CFG_*_FAST > 0 (expert/primary never uses FAST)

dewo_v2_require_gpus() {
  if [[ -z "${GPUS:-}" ]]; then
    echo "[dewo-v2] ERROR: set GPUS to a comma-separated list of physical GPU ids." >&2
    echo "  Example: GPUS=4,5,6,7 TASK=fold_glasses bash scripts/dewo_v2/<launcher>.sh" >&2
    return 2
  fi
}

dewo_v2_require_task() {
  TASK="${TASK:-${DEWO_TASK_NAME:-}}"
  if [[ -z "${TASK}" ]]; then
    echo "[dewo-v2] ERROR: set TASK (fold_glasses|hammer_nail|water_plant|pick_bucket|pinch_tongs)." >&2
    return 2
  fi
}

# LoRA Hydra tasks / INIT=lora are removed. Full DiT only (scratch | s0).
dewo_v2_assert_not_lora() {
  local label="${1:-value}"
  local value="${2:-}"
  if [[ "${value}" == *lora* || "${value}" == *LoRA* ]]; then
    echo "[dewo-v2] ERROR: ${label}=${value} is a LoRA path." >&2
    echo "  LoRA recipes are removed. Use INIT=scratch or INIT=s0 (full DiT)." >&2
    echo "  Hydra: dexjoco/dexjoco_dewo_v2_offline_b1_jump_fast_full_1e-4 (scratch)" >&2
    echo "         dexjoco/dexjoco_dewo_v2_offline_b1_jump_fast_full_s0 (from S0)" >&2
    return 2
  fi
}

dewo_v2_load_task() {
  local task="${1:?task name required, e.g. water_plant}"
  local root="${ROOT_DIR:?ROOT_DIR must be set before sourcing lib.sh}"
  # Session overrides (mixed S0 ckpt/stats, rollout primary, ...) must survive
  # export-env, which always prints the single-task registry defaults.
  local _saved_ckpt="${CKPT:-}"
  local _saved_stats="${STATS:-}"
  local _saved_norm="${PRETRAINED_NORM_STATS:-}"
  local _saved_text="${TEXT_EMB:-}"
  local _saved_init="${INIT_WEIGHTS:-}"
  local _saved_src_ckpt="${SOURCE_CHECKPOINT:-}"
  local _saved_base="${BASE_DATASET:-}"
  local _saved_repeats="${REPEATS:-}"
  eval "$(python "${root}/scripts/dewo_v2/tasks.py" export-env --task "${task}")"
  [[ -z "${_saved_ckpt}" ]] || export CKPT="${_saved_ckpt}"
  [[ -z "${_saved_stats}" ]] || export STATS="${_saved_stats}"
  [[ -z "${_saved_norm}" ]] || export PRETRAINED_NORM_STATS="${_saved_norm}"
  [[ -z "${_saved_text}" ]] || export TEXT_EMB="${_saved_text}"
  [[ -z "${_saved_init}" ]] || export INIT_WEIGHTS="${_saved_init}"
  [[ -z "${_saved_src_ckpt}" ]] || export SOURCE_CHECKPOINT="${_saved_src_ckpt}"
  [[ -z "${_saved_base}" ]] || export BASE_DATASET="${_saved_base}"
  [[ -z "${_saved_repeats}" ]] || export REPEATS="${_saved_repeats}"
  if [[ -n "${_saved_stats}" && -z "${_saved_norm}" ]]; then
    export PRETRAINED_NORM_STATS="${_saved_stats}"
  fi
  if [[ -n "${_saved_ckpt}" && -z "${_saved_init}" ]]; then
    export INIT_WEIGHTS="${_saved_ckpt}"
  fi
  if [[ -n "${_saved_ckpt}" && -z "${_saved_src_ckpt}" ]]; then
    export SOURCE_CHECKPOINT="${_saved_ckpt}"
  fi
  dewo_v2_align_opensource_stack
}

dewo_v2_activate_fastwam() {
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${FITWAM_ENV:-fastwam}"
  ENV_PREFIX="${FITWAM_ENV_PREFIX:-$(conda info --base)/envs/${FITWAM_ENV:-fastwam}}"
  export PATH="${ENV_PREFIX}/bin:${PATH}"
}

dewo_v2_gpus_idle() {
  local max_used="${MAX_USED_MIB:-1500}"
  local max_util="${MAX_UTIL:-10}"
  python - "${GPUS}" "${max_used}" "${max_util}" <<'PY'
import subprocess, sys
gpus = [int(x) for x in sys.argv[1].split(",") if x.strip()]
max_used = int(sys.argv[2])
max_util = int(sys.argv[3])
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
    text=True,
)
rows = {}
for line in out.strip().splitlines():
    idx, mem, util = [p.strip() for p in line.split(",")]
    rows[int(idx)] = (int(float(mem)), int(float(util)))
ok = True
for g in gpus:
    mem, util = rows.get(g, (10**9, 100))
    print(f"gpu{g}: mem={mem}MiB util={util}%")
    if mem > max_used or (max_util > 0 and util > max_util):
        ok = False
raise SystemExit(0 if ok else 1)
PY
}

dewo_v2_wait_gpus_idle() {
  local log_file="${1:-/dev/null}"
  local label="${2:-dewo-v2}"
  local sleep_s="${WAIT_POLL_SEC:-60}"
  local streak_need="${WAIT_IDLE_STREAK:-2}"
  local idle_streak=0
  echo "[${label} $(date -Is)] waiting for GPUs ${GPUS} idle (used<=${MAX_USED_MIB:-1500}MiB util<=${MAX_UTIL:-10}%)" | tee -a "${log_file}"
  while true; do
    if dewo_v2_gpus_idle >>"${log_file}" 2>&1; then
      idle_streak=$((idle_streak + 1))
      echo "[${label} $(date -Is)] idle streak ${idle_streak}/${streak_need}" | tee -a "${log_file}"
      [[ "${idle_streak}" -ge "${streak_need}" ]] && break
    else
      idle_streak=0
    fi
    sleep "${sleep_s}"
  done
}

# Pin DexJoCo / DEWO v2 to the opensource 224 / z-score stack.
# Mixed S0 z-score stats (e.g. artifacts/mixed_5task) are kept; local
# data/<task>_fastwam and */meta/* min-max are rewritten.
dewo_v2_align_opensource_stack() {
  local task="${DEWO_TASK_NAME:-${TASK:?}}"
  local root="${ROOT_DIR:?ROOT_DIR must be set before sourcing lib.sh}"
  local open_repo="${OPEN_REPO:-${root}/../FastWAM-infer-in-DexJoco}"
  local expert="${root}/data/dexjoco/dexjoco_lerobot_datasets/${task}"
  local open_cfg="${open_repo}/configs/fastwam_dexjoco.yaml"
  local open_stats="${open_repo}/artifacts/${task}/dataset_stats.json"

  SOURCE_CONFIG="${SOURCE_CONFIG:-${open_cfg}}"
  FASTWAM_SOURCE_CONFIG="${FASTWAM_SOURCE_CONFIG:-${SOURCE_CONFIG}}"

  if [[ "${PRIMARY_KIND:-}" != "success_rollouts" && "${PRIMARY_KIND:-}" != "all_success_seeds" ]]; then
    if [[ -z "${BASE_DATASET:-}" || "${BASE_DATASET}" == *"${task}_fastwam"* ]]; then
      BASE_DATASET="${expert}"
    fi
    if [[ -z "${SOURCE_DATASET:-}" || "${SOURCE_DATASET}" == *"${task}_fastwam"* ]]; then
      SOURCE_DATASET="${expert}"
    fi
  fi

  local stats="${PRETRAINED_NORM_STATS:-${STATS:-}}"
  if [[ -z "${stats}" || "${stats}" == */meta/* || "${stats}" == *"${task}_fastwam"* ]]; then
    stats="${open_stats}"
  fi
  STATS="${stats}"
  PRETRAINED_NORM_STATS="${stats}"
  export SOURCE_CONFIG FASTWAM_SOURCE_CONFIG BASE_DATASET SOURCE_DATASET STATS PRETRAINED_NORM_STATS
}

# Apply default VAE policy after sourcing prepare env.
# Default: pre-encode at prepare/train and read cache. Opt out: USE_VAE_LATENT_CACHE=0.
dewo_v2_apply_vae_policy() {
  if [[ "${USE_VAE_LATENT_CACHE:-1}" == "0" ]]; then
    export USE_VAE_LATENT_CACHE=0
    export SKIP_VAE_PREENCODE=1
    export FILL_VAE_LATENT_CACHE=0
    export REQUIRE_VAE_LATENT_CACHE=0
    unset VAE_LATENT_CACHE_DIR || true
    echo "[dewo-v2] VAE policy: online encode (USE_VAE_LATENT_CACHE=0 opt-out)"
    return 0
  fi
  export USE_VAE_LATENT_CACHE=1
  export SKIP_VAE_PREENCODE="${SKIP_VAE_PREENCODE:-0}"
  export FILL_VAE_LATENT_CACHE="${FILL_VAE_LATENT_CACHE:-0}"
  export REQUIRE_VAE_LATENT_CACHE=1
  echo "[dewo-v2] VAE policy: pre-encode cache (set USE_VAE_LATENT_CACHE=0 for online VAE)"
}

# Return 0 if any CFG channel has FAST weight > 0 (needs FAST text-emb precompute).
dewo_v2_cfg_uses_fast() {
  python - <<'PY'
import os, sys

def f(name: str) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return 0.0
    return float(raw)

weights = [
    f("CFG_PRIMARY_FAST"),
    f("CFG_AUX_SUCCESS_FAST"),
    f("CFG_AUX_FAIL_FAST"),
]
# Also accept comma triples if component envs were not exported.
for key in ("CFG_PRIMARY", "CFG_AUX_SUCCESS", "CFG_AUX_FAIL"):
    raw = os.environ.get(key, "")
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if len(parts) == 3:
        weights.append(float(parts[1]))
sys.exit(0 if any(w > 0.0 for w in weights) else 1)
PY
}
