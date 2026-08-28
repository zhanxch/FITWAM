# Shared bash helpers for DexJoCo DEWO v9. Source from other scripts:
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
# Only DEWO_VERSION=v9 / INIT=s0 is supported. Frozen mixed-S0 MoT, text-side
# K/V adapter + VideoDiT value head. Mix ε_cfg = ε_0 + g w (ε_+ − ε_0).
# D0 = one episode per 4/4 all-success seed. D+ = full-horizon success stitch.
# D_fail = fail cliff [t, M+24). G_t=γ^{T-t} (fail=0). No FAST.
# CFG: D+ 0.9/0/0.1, D_fail 1.0/0/0, suffixes Successful / Failed execution.
#
# Protocol .env files are paths/VAE/text-cache only. train.sh owns CFG mixing.
#
# Text embeds: base + Successful/Failed outcome. FAST is unused.

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

# Prevent a leaked EXP_ROOT / PAIR_OUT from another TASK overwriting data.
dewo_v2_assert_path_for_task() {
  local label="${1:?}"
  local path="${2:-}"
  local task="${TASK:?}"
  if [[ -z "${path}" ]]; then
    return 0
  fi
  case "${path}" in
    *"${task}"*) return 0 ;;
    *)
      echo "[dewo-v2] ERROR: ${label}=${path} does not contain TASK=${task}." >&2
      echo "  Unset leaked EXP_ROOT / PAIR_OUT / PAIR_DATASET / COLLECT_ROOT from another task." >&2
      return 2
      ;;
  esac
}

# Protocol env and tasks.py export-env used to dump v2 FAST triples. Drop them
# so v9 train.sh can install Successful/Failed + 0.9/0/0.1 without leakage.
dewo_v2_clear_cfg_mix() {
  unset CFG_PRIMARY CFG_AUX_SUCCESS CFG_AUX_FAIL \
    CFG_PRIMARY_OUTCOME CFG_PRIMARY_FAST CFG_PRIMARY_BASE \
    CFG_AUX_SUCCESS_OUTCOME CFG_AUX_SUCCESS_FAST CFG_AUX_SUCCESS_BASE \
    CFG_AUX_FAIL_OUTCOME CFG_AUX_FAIL_FAST CFG_AUX_FAIL_BASE \
    CFG_SUCCESS_SUFFIX CFG_FAILURE_SUFFIX CFG_DROPOUT \
    CFG_RECIPE_NAME CFG_FAST_MODEL_ID CFG_FAST_MAX_TOKENS CFG_FAST_FAIL_CLOSED \
    || true
}

# LoRA Hydra tasks / INIT=lora are removed. Full DiT only (scratch | s0).
dewo_v2_assert_not_lora() {
  local label="${1:-value}"
  local value="${2:-}"
  if [[ "${value}" == *lora* || "${value}" == *LoRA* ]]; then
    echo "[dewo-v2] ERROR: ${label}=${value} is a LoRA path." >&2
    echo "  LoRA recipes are removed. DEWO v9 is INIT=s0 only." >&2
    echo "  Hydra: dexjoco/dexjoco_dewo_v9_offline_b1_jump_fast_uncond (INIT=s0)" >&2
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
