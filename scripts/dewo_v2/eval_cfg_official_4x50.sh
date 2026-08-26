#!/usr/bin/env bash
# Official 4×50 DEWO v2/v5/v6/v7 CFG eval. Opensource 224 / z-score.
#   TASK=water_plant RUN_DIR=... CKPT=... GPUS=4,5,6,7 CFG_SCALE=1.2 \
#     bash scripts/dewo_v2/eval_cfg_official_4x50.sh
#
# CFG terminology:
#   text_cfg_scale=1  → 本体 bypass (adapter off + cfg_base_prompt). Not mix w=1.
#   mix w=0           → ε_base, same policy as the bypass after both branches run.
#   v5/v6 mix w=1     → ε_posi (adapter on + success). Not 本体.
#   v7 mix w=1        → ε_base + (ε_posi − ε_fail). Not ε_posi. Never execute ε_fail.
#   CFG_SCALE=2       → v5/v6: ε_base + 2(ε_posi-ε_base); v7: ε_base + 2(ε_posi-ε_fail).
# Adaptive (optional): ADAPTIVE_CFG_TAU=0.05 with CFG_SCALE!=1
#   NFE0 exec RMS E>tau → mix w=CFG_SCALE; else mix w=0 (本体).
#   v7 energy is RMS(ε_posi − ε_fail), not RMS(ε_posi − ε_base).
#   ADAPTIVE_CFG_TAU=auto reads RUN_DIR/adaptive_cfg_tau.json; skips adaptive
#   if the file is missing or separable=false. Do not treat 0.05 as a v6/v7 prior.
# Residual trust region (optional): CFG_EPSILON_L=0.03 bounds the action
# residual before CFG_SCALE; CFG_RESIDUAL_CLIP_MODE is rms or elementwise.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_require_gpus
# Preserve caller overrides: load_task exports release CKPT/STATS and would clobber them.
CALLER_CKPT="${CKPT:-}"
CALLER_STATS="${PRETRAINED_NORM_STATS:-}"
CALLER_CFG_TASK_DIR="${CFG_TASK_DIR:-}"
CALLER_UNCOND_ADAPTER="${UNCOND_ADAPTER:-}"
CALLER_BACKBONE_CKPT="${BACKBONE_CKPT:-}"
CALLER_REPEATS="${REPEATS:-}"
CALLER_CFG_EPSILON_L="${CFG_EPSILON_L:-}"
CALLER_CFG_CLIP_MODE="${CFG_RESIDUAL_CLIP_MODE:-}"
dewo_v2_load_task "${TASK}"

RUN_DIR="${RUN_DIR:?Set RUN_DIR to the DEWO v2 training run directory}"
CKPT="${CALLER_CKPT:-${CKPT:-${RUN_DIR}/checkpoints/weights/step_002500.pt}}"
UNCOND_ADAPTER="${CALLER_UNCOND_ADAPTER:-}"
BACKBONE_CKPT="${CALLER_BACKBONE_CKPT:-}"
PRETRAINED_NORM_STATS="${CALLER_STATS:-${PRETRAINED_NORM_STATS:-${STATS}}}"
dewo_v2_align_opensource_stack
TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:?Set TEXT_EMBEDDING_CACHE_DIR}"
if [[ -z "${CALLER_CFG_TASK_DIR}" ]] && grep -q 'dewo_v9_uncond_adapter' "${RUN_DIR}/config.yaml" 2>/dev/null; then
  CFG_TASK_DIR="${ROOT_DIR}/configs/eval/dexjoco/${TASK}_dewo_v9_cfg"
elif [[ -z "${CALLER_CFG_TASK_DIR}" ]] && grep -q 'dewo_v8_uncond_adapter' "${RUN_DIR}/config.yaml" 2>/dev/null; then
  CFG_TASK_DIR="${ROOT_DIR}/configs/eval/dexjoco/${TASK}_dewo_v8_cfg"
elif [[ -z "${CALLER_CFG_TASK_DIR}" ]] && grep -q 'dewo_v7_uncond_adapter' "${RUN_DIR}/config.yaml" 2>/dev/null; then
  CFG_TASK_DIR="${ROOT_DIR}/configs/eval/dexjoco/${TASK}_dewo_v7_cfg"
elif [[ -z "${CALLER_CFG_TASK_DIR}" ]] && grep -q 'dewo_v6_uncond_adapter' "${RUN_DIR}/config.yaml" 2>/dev/null; then
  CFG_TASK_DIR="${ROOT_DIR}/configs/eval/dexjoco/${TASK}_dewo_v6_cfg"
else
  CFG_TASK_DIR="${CALLER_CFG_TASK_DIR:-${CFG_TASK_DIR:-${ROOT_DIR}/configs/eval/dexjoco/${TASK}_dewo_v2_cfg}}"
fi
DEXJOCO_PY_ROOT="${DEXJOCO_PY_ROOT:-${ROOT_DIR}/third_party/dexjoco/dexjoco}"

CFG_SCALE="${CFG_SCALE:-2.0}"
ADAPTIVE_CFG_TAU="${ADAPTIVE_CFG_TAU:-}"
if [[ "${ADAPTIVE_CFG_TAU}" == "auto" ]]; then
  _tau_json="${RUN_DIR}/adaptive_cfg_tau.json"
  if [[ ! -f "${_tau_json}" ]]; then
    echo "[dewo-v2-cfg-4x50] ADAPTIVE_CFG_TAU=auto but missing ${_tau_json}; skipping adaptive"
    ADAPTIVE_CFG_TAU=""
  else
    _resolved="$(python3 -c '
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
tau = payload.get("tau")
if (not payload.get("separable")) or tau is None:
    sys.exit(0)
print(tau)
' "${_tau_json}")"
    if [[ -z "${_resolved}" ]]; then
      echo "[dewo-v2-cfg-4x50] ADAPTIVE_CFG_TAU=auto not separable; skipping adaptive"
      ADAPTIVE_CFG_TAU=""
    else
      ADAPTIVE_CFG_TAU="${_resolved}"
      echo "[dewo-v2-cfg-4x50] ADAPTIVE_CFG_TAU=auto -> ${ADAPTIVE_CFG_TAU}"
    fi
  fi
fi
CFG_EPSILON_L="${CALLER_CFG_EPSILON_L}"
CFG_RESIDUAL_CLIP_MODE="${CALLER_CFG_CLIP_MODE:-rms}"
EPISODES="${EPISODES:-50}"
# 4 = official 4×50 (4 independent repeats). 1 = screening 1×50 split across GPUS.
REPEATS="${CALLER_REPEATS:-${REPEATS:-4}}"
ENV_SEED="${ENV_SEED:-0}"
INFERENCE_SEED="${INFERENCE_SEED:-20260812}"
# Screening 1×50 uses this as the OPEN-stack repeat index (noise seed).
# Official 4×50 ignores it and uses 0,1,2,3. Bump this to avoid reusing the
# same diffusion noise as a prior 0–49 screen (client seed, not --inference-seed).
SCREEN_EVAL_REPEAT="${SCREEN_EVAL_REPEAT:-0}"
NOISE_SEED_BASE="${NOISE_SEED_BASE:-}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-${MAX_STEPS}}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
BASE_PORT="${BASE_PORT:-6100}"
WAIT_IDLE="${WAIT_IDLE:-1}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
CKPT_TAG="${CKPT_TAG:-$(basename "${CKPT}" .pt)}"
if [[ -n "${ADAPTIVE_CFG_TAU}" && -n "${CFG_EPSILON_L}" ]]; then
  _CFG_OUT_TAG="cfg${CFG_SCALE}_adapt_tau${ADAPTIVE_CFG_TAU}_eps${CFG_EPSILON_L}_${CFG_RESIDUAL_CLIP_MODE}"
elif [[ -n "${ADAPTIVE_CFG_TAU}" ]]; then
  _CFG_OUT_TAG="cfg${CFG_SCALE}_adapt_tau${ADAPTIVE_CFG_TAU}"
elif [[ -n "${CFG_EPSILON_L}" ]]; then
  _CFG_OUT_TAG="cfg${CFG_SCALE}_eps${CFG_EPSILON_L}_${CFG_RESIDUAL_CLIP_MODE}"
else
  _CFG_OUT_TAG="cfg${CFG_SCALE}"
fi
if [[ "${CFG_GATE_MODE:-}" == "value_growth" || "${CFG_GATE_MODE:-}" == "growth" ]]; then
  _CFG_OUT_TAG="value_growth_tau${CFG_GROWTH_TAU:-0.05}_start${CFG_GROWTH_START_REPLAN:-2}_${_CFG_OUT_TAG}"
elif [[ "${CFG_GATE_MODE:-}" == "value" ]]; then
  _CFG_OUT_TAG="value_gate_${_CFG_OUT_TAG}"
fi
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/evaluate_results/dexjoco/${TASK}_dewo_v2_pair_${CKPT_TAG}_${_CFG_OUT_TAG}_4x50_${STAMP}}"

dewo_v2_activate_fastwam
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "${OUT_ROOT}/logs"
LOG="${OUT_ROOT}/logs/orchestrator.log"
log() { echo "[dewo-v2-cfg-4x50 ${TASK} $(date -Is)] $*" | tee -a "${LOG}"; }

if grep -q 'dewo_v7_uncond_adapter' "${RUN_DIR}/config.yaml" 2>/dev/null \
  && [[ ! -f "${CFG_TASK_DIR}/${TASK}.yaml" ]]; then
  mkdir -p "${CFG_TASK_DIR}"
  CFG_SUCCESS_SUFFIX="${CFG_SUCCESS_SUFFIX:- Successful execution.}" \
  CFG_FAILURE_SUFFIX="${CFG_FAILURE_SUFFIX:- Failed execution.}" \
    python "${ROOT_DIR}/scripts/dewo_v2/tasks.py" write-eval-yaml \
      --task "${TASK}" --output "${CFG_TASK_DIR}/${TASK}.yaml"
  log "wrote ${CFG_TASK_DIR}/${TASK}.yaml"
fi

for required in \
  "${RUN_DIR}/config.yaml" \
  "${CKPT}" \
  "${PRETRAINED_NORM_STATS}" \
  "${TEXT_EMBEDDING_CACHE_DIR}" \
  "${CFG_TASK_DIR}/${TASK}.yaml"
do
  [[ -e "${required}" ]] || { log "ERROR missing ${required}"; exit 2; }
done

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
if [[ "${REPEATS}" != "1" && "${REPEATS}" != "4" ]]; then
  log "ERROR: REPEATS must be 1 (screening 1×50) or 4 (official 4×50), got ${REPEATS}"
  exit 2
fi
if [[ "${REPEATS}" == "4" && "${#GPU_ARR[@]}" -ne 4 ]]; then
  log "ERROR: official 4×50 concurrent layout expects 4 GPUs, got ${GPUS}"
  exit 2
fi

if [[ "${WAIT_IDLE}" == "1" ]]; then
  dewo_v2_wait_gpus_idle "${LOG}" "dewo-v2-cfg-4x50"
fi

if [[ "${REPEATS}" == "1" ]]; then
  PROTOCOL_LABEL="screening_1x50_seeds_0_49_NON_STANDARD"
else
  PROTOCOL_LABEL="official_4x50_seeds_0_49"
fi
if grep -q 'dewo_v7_uncond_adapter' "${RUN_DIR}/config.yaml" 2>/dev/null; then
  log "CFG terminology: text_cfg_scale=1 is 本体 remap; v7 mix w=1 is ε_base+(ε_posi-ε_fail), not ε_posi."
else
  log "CFG terminology: text_cfg_scale=1 is 本体 remap (adapter off + cfg_base_prompt); mix w=1 would be ε_posi, not 本体."
fi
if [[ -n "${ADAPTIVE_CFG_TAU}" ]]; then
  log "adaptive CFG tau=${ADAPTIVE_CFG_TAU}: NFE0 exec RMS E>tau mix w=${CFG_SCALE}; else mix w=0 (本体)"
else
  log "CFG success-vs-base scale=${CFG_SCALE}"
fi
if [[ -n "${CFG_EPSILON_L}" ]]; then
  log "CFG residual epsilon_l=${CFG_EPSILON_L} clip_mode=${CFG_RESIDUAL_CLIP_MODE} (applied before scale)"
fi
log "ckpt=${CKPT} gpus=${GPUS} repeats=${REPEATS} protocol=${PROTOCOL_LABEL} replan=${REPLAN_STEPS}, max_steps=${MAX_ENV_STEPS}"
if [[ -n "${BACKBONE_CKPT}" ]]; then
  log "backbone=${BACKBONE_CKPT}"
fi
if [[ -n "${UNCOND_ADAPTER}" ]]; then
  log "uncond_adapter=${UNCOND_ADAPTER}"
fi
if [[ "${REPEATS}" == "1" ]]; then
  log "screening eval_repeat=${SCREEN_EVAL_REPEAT} (diffusion noise; env seeds still 0–49)"
fi
if [[ -n "${NOISE_SEED_BASE}" ]]; then
  if [[ "${REPEATS}" == "1" ]]; then
    log "independent noise_seed_base=${NOISE_SEED_BASE} (screening; env seeds 0–49)"
  else
    log "independent noise_seed_base=${NOISE_SEED_BASE}..$((NOISE_SEED_BASE + 3)) (4×50 repeats; env seeds 0–49)"
  fi
fi

EXTRA_EVAL_ARGS=()
if [[ -n "${BACKBONE_CKPT}" ]]; then
  EXTRA_EVAL_ARGS+=(--backbone-checkpoint "${BACKBONE_CKPT}")
fi
if [[ -n "${UNCOND_ADAPTER}" && "${UNCOND_ADAPTER}" != "${CKPT}" ]]; then
  EXTRA_EVAL_ARGS+=(--uncond-adapter "${UNCOND_ADAPTER}")
fi
if [[ -n "${ADAPTIVE_CFG_TAU}" ]]; then
  EXTRA_EVAL_ARGS+=(--adaptive-cfg-tau "${ADAPTIVE_CFG_TAU}")
fi
if [[ -n "${CFG_EPSILON_L}" ]]; then
  EXTRA_EVAL_ARGS+=(--cfg-epsilon-l "${CFG_EPSILON_L}")
  EXTRA_EVAL_ARGS+=(--cfg-residual-clip-mode "${CFG_RESIDUAL_CLIP_MODE}")
fi
if [[ -n "${CFG_GATE_MODE:-}" ]]; then
  EXTRA_EVAL_ARGS+=(--cfg-gate-mode "${CFG_GATE_MODE}")
fi
if [[ -n "${CFG_V_HIGH:-}" ]]; then
  EXTRA_EVAL_ARGS+=(--cfg-v-high "${CFG_V_HIGH}")
fi
if [[ -n "${CFG_DROP_DELTA:-}" ]]; then
  EXTRA_EVAL_ARGS+=(--cfg-drop-delta "${CFG_DROP_DELTA}")
fi
if [[ -n "${CFG_GROWTH_TAU:-}" ]]; then
  EXTRA_EVAL_ARGS+=(--cfg-growth-tau "${CFG_GROWTH_TAU}")
fi
if [[ -n "${CFG_GROWTH_START_REPLAN:-}" ]]; then
  EXTRA_EVAL_ARGS+=(--cfg-growth-start-replan "${CFG_GROWTH_START_REPLAN}")
fi
if [[ -n "${CFG_INTERVENE_SCHEDULE:-}" ]]; then
  EXTRA_EVAL_ARGS+=(--cfg-intervene-schedule "${CFG_INTERVENE_SCHEDULE}")
fi
if grep -q 'dewo_v9_uncond_adapter' "${RUN_DIR}/config.yaml" 2>/dev/null; then
  if [[ "${CFG_GATE_MODE:-}" == "value_growth" || "${CFG_GATE_MODE:-}" == "growth" ]]; then
    METHOD="${METHOD:-dewo_v9_uncond_adapter_value_growth}"
  elif [[ "${CFG_GATE_MODE:-}" == "value" ]]; then
    METHOD="${METHOD:-dewo_v9_uncond_adapter_value_gate}"
  elif [[ -n "${ADAPTIVE_CFG_TAU}" ]]; then
    METHOD="${METHOD:-dewo_v9_uncond_adapter_adaptive_cfg}"
  else
    METHOD="${METHOD:-dewo_v9_uncond_adapter_cfg}"
  fi
elif grep -q 'dewo_v8_uncond_adapter' "${RUN_DIR}/config.yaml" 2>/dev/null; then
  if [[ "${CFG_GATE_MODE:-}" == "value" ]]; then
    METHOD="${METHOD:-dewo_v8_uncond_adapter_value_gate}"
  elif [[ -n "${ADAPTIVE_CFG_TAU}" ]]; then
    METHOD="${METHOD:-dewo_v8_uncond_adapter_adaptive_cfg}"
  else
    METHOD="${METHOD:-dewo_v8_uncond_adapter_cfg}"
  fi
elif grep -q 'dewo_v7_uncond_adapter' "${RUN_DIR}/config.yaml" 2>/dev/null; then
  if [[ -n "${ADAPTIVE_CFG_TAU}" ]]; then
    METHOD="${METHOD:-dewo_v7_uncond_adapter_adaptive_cfg}"
  else
    METHOD="${METHOD:-dewo_v7_uncond_adapter_cfg}"
  fi
elif grep -q 'dewo_v6_uncond_adapter' "${RUN_DIR}/config.yaml" 2>/dev/null; then
  if [[ -n "${ADAPTIVE_CFG_TAU}" ]]; then
    METHOD="${METHOD:-dewo_v6_uncond_adapter_adaptive_cfg}"
  else
    METHOD="${METHOD:-dewo_v6_uncond_adapter_cfg}"
  fi
elif grep -q 'dewo_v5_uncond_adapter' "${RUN_DIR}/config.yaml" 2>/dev/null; then
  if [[ -n "${ADAPTIVE_CFG_TAU}" ]]; then
    METHOD="${METHOD:-dewo_v5_uncond_adapter_adaptive_cfg}"
  else
    METHOD="${METHOD:-dewo_v5_uncond_adapter_cfg}"
  fi
else
  METHOD="${METHOD:-dewo_v2_success_vs_base_cfg}"
fi
if [[ -n "${ADAPTIVE_CFG_TAU}" ]]; then
  if ! "${ENV_PREFIX}/bin/python" - "${CFG_SCALE}" <<'PY'
import sys
sys.exit(0 if abs(float(sys.argv[1]) - 1.0) >= 1e-12 else 1)
PY
  then
    log "ERROR: adaptive CFG needs CFG_SCALE!=1 (guided mix); scale=1 is 本体 bypass"
    exit 2
  fi
fi

eval_one() {
  local run_i="$1"
  local gpu_list="$2"
  local port="$3"
  local eval_repeat="$4"
  local infer_seed="$5"
  local out_dir="$6"
  local noise_seed_args=()
  if [[ -n "${NOISE_SEED_BASE}" ]]; then
    noise_seed_args=(--noise-seed-base "$((NOISE_SEED_BASE + eval_repeat))")
  fi
  mkdir -p "${out_dir}"
  if [[ -n "${NOISE_SEED_BASE}" ]]; then
    log "launch run${run_i} gpus=${gpu_list} env_seed=${ENV_SEED} infer_seed=${infer_seed} eval_repeat=${eval_repeat} noise_seed_base=$((NOISE_SEED_BASE + eval_repeat))"
  else
    log "launch run${run_i} gpus=${gpu_list} env_seed=${ENV_SEED} infer_seed=${infer_seed} eval_repeat=${eval_repeat}"
  fi
  "${ENV_PREFIX}/bin/python" scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
    --gpus "${gpu_list}" \
    --base-port "${port}" \
    --episodes "${EPISODES}" \
    --seed "${ENV_SEED}" \
    --eval-repeat "${eval_repeat}" \
    "${noise_seed_args[@]}" \
    --inference-seed "${infer_seed}" \
    --text-cfg-scale "${CFG_SCALE}" \
    --action-horizon "${ACTION_HORIZON}" \
    --server-conda-env "${ENV_PREFIX}" \
    --client-conda-env dexjoco \
    --run-dir "${RUN_DIR}" \
    --checkpoint "${CKPT}" \
    --dataset-stats-path "${PRETRAINED_NORM_STATS}" \
    --text-embedding-cache-dir "${TEXT_EMBEDDING_CACHE_DIR}" \
    --no-load-text-encoder \
    --task-config-dir "${CFG_TASK_DIR}" \
    --tasks "${TASK}" \
    --dexjoco-py-root "${DEXJOCO_PY_ROOT}" \
    --replan-steps "${REPLAN_STEPS}" \
    --control-mode blocking \
    --max-env-steps "${MAX_ENV_STEPS}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --video-fps 30 \
    --no-randomize \
    --no-randomize-dynamics \
    --save-actions \
    --save-video \
    --no-action-clip \
    --output-dir "${out_dir}" \
    "${EXTRA_EVAL_ARGS[@]}" \
    > "${OUT_ROOT}/logs/run${run_i}.log" 2>&1
}

N_RUNS="${REPEATS}"
if [[ "${REPEATS}" == "1" ]]; then
  eval_one 1 "${GPUS}" "${BASE_PORT}" "${SCREEN_EVAL_REPEAT}" "${INFERENCE_SEED}" "${OUT_ROOT}/run1"
else
  pids=()
  for i in 1 2 3 4; do
    gpu="${GPU_ARR[$((i - 1))]}"
    (
      eval_one "${i}" "${gpu}" "$((BASE_PORT + i * 20))" "$((i - 1))" "$((INFERENCE_SEED + i - 1))" "${OUT_ROOT}/run${i}"
    ) &
    pids+=("$!")
  done
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      fail=1
    fi
  done
  if [[ "${fail}" -ne 0 ]]; then
    log "ERROR: one or more CFG eval runs failed"
    exit 2
  fi
fi

"${ENV_PREFIX}/bin/python" - "${OUT_ROOT}" "${CKPT_TAG}" "${CFG_SCALE}" "${MAX_ENV_STEPS}" "${TASK}" "${METHOD}" "${N_RUNS}" "${PROTOCOL_LABEL}" "${ADAPTIVE_CFG_TAU}" "${SCREEN_EVAL_REPEAT}" "${CFG_EPSILON_L}" "${CFG_RESIDUAL_CLIP_MODE}" <<'PY' | tee -a "${LOG}"
import json, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
ckpt_tag = sys.argv[2]
cfg_scale = float(sys.argv[3])
max_steps = int(sys.argv[4])
task = sys.argv[5]
method = sys.argv[6]
n_runs = int(sys.argv[7])
protocol = sys.argv[8]
adaptive_raw = sys.argv[9] if len(sys.argv) > 9 else ""
adaptive_tau = float(adaptive_raw) if adaptive_raw else None
screen_repeat_raw = sys.argv[10] if len(sys.argv) > 10 else ""
epsilon_raw = sys.argv[11] if len(sys.argv) > 11 else ""
clip_mode = sys.argv[12] if len(sys.argv) > 12 else "rms"
epsilon_l = float(epsilon_raw) if epsilon_raw else None
rates, rows, pooled_s, pooled_n = [], [], 0, 0
guided_values, gate_values, value_means, fire_fracs = [], [], [], []
for i in range(1, n_runs + 1):
    d = json.loads((root / f"run{i}" / "summary.json").read_text())
    rate = float(d["overall_success_rate"])
    s, n = int(d["total_successes"]), int(d["total_episodes"])
    rates.append(rate)
    pooled_s += s
    pooled_n += n
    for task_row in d.get("tasks", []):
        for episode_row in task_row.get("episode_results", []):
            metrics = episode_row.get("metrics", {})
            guided = metrics.get("cfg_guided_chunk_fraction")
            gate = metrics.get("cfg_gate_exec_rms_mean")
            value_mean = metrics.get("cfg_value_mean")
            fire = metrics.get("cfg_gate_g_fire_fraction")
            if guided is not None:
                guided_values.append(float(guided))
            if gate is not None:
                gate_values.append(float(gate))
            if value_mean is not None:
                value_means.append(float(value_mean))
            if fire is not None:
                fire_fracs.append(float(fire))
    rows.append({"run": i, "successes": s, "episodes": n, "rate": rate})
var = statistics.pvariance(rates) if len(rates) > 1 else 0.0
std = statistics.pstdev(rates) if len(rates) > 1 else 0.0
agg = {
    "task": task,
    "method": method,
    "ckpt_tag": ckpt_tag,
    "text_cfg_scale": cfg_scale,
    "adaptive_cfg_tau": adaptive_tau,
    "cfg_epsilon_l": epsilon_l,
    "cfg_residual_clip_mode": clip_mode if epsilon_l is not None else None,
    "cfg_mix_high": cfg_scale if adaptive_tau is not None else None,
    "cfg_mix_low": 0.0 if adaptive_tau is not None else None,
    "protocol": protocol,
    "max_env_steps": max_steps,
    "runs": rows,
    "mean_success_rate": statistics.fmean(rates),
    "var_success_rate": var,
    "std_success_rate": std,
    "pooled_successes": pooled_s,
    "pooled_episodes": pooled_n,
    "pooled_success_rate": pooled_s / pooled_n if pooled_n else None,
    "cfg_guided_chunk_fraction_mean": (
        statistics.fmean(guided_values) if guided_values else None
    ),
    "cfg_gate_exec_rms_mean": (
        statistics.fmean(gate_values) if gate_values else None
    ),
    "cfg_gate_observations": len(gate_values),
    "cfg_value_mean": (
        statistics.fmean(value_means) if value_means else None
    ),
    "cfg_gate_g_fire_fraction_mean": (
        statistics.fmean(fire_fracs) if fire_fracs else None
    ),
    "cfg_value_episodes": len(value_means),
}
if n_runs == 1 and screen_repeat_raw != "":
    agg["screen_eval_repeat"] = int(screen_repeat_raw)
(root / "aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
print(json.dumps(agg, indent=2))
print(
    f"mean={agg['mean_success_rate']:.4f} var={agg['var_success_rate']:.6f} "
    f"pooled={pooled_s}/{pooled_n}"
)
PY

log "DONE ${OUT_ROOT}/aggregate.json"
