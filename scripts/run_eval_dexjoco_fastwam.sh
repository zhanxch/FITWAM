#!/usr/bin/env bash
# Closed-loop DexJoCo success-rate eval for FastWAM (all 11 tasks).
#
# Prerequisite: start the policy server in another terminal (fastwam env), e.g.
#   CUDA_VISIBLE_DEVICES=7 python scripts/run_fastwam_server.py \
#     --run-dir runs/dexjoco_ego_uncond_1cam_384_1e-4/2026-06-05_17-18-31 \
#     --checkpoint checkpoints/weights/step_002500.pt \
#     --device cuda:0 --host 0.0.0.0 --port 5560
#
# Usage:
#   bash scripts/run_eval_dexjoco_fastwam.sh
#   EPISODES=10 bash scripts/run_eval_dexjoco_fastwam.sh
#   TASKS="click_mouse hammer_nail" bash scripts/run_eval_dexjoco_fastwam.sh
#   ACTION_CLIP=1 REPLAN_STEPS=8 EPISODES=5 TASKS=bimanual_microwave_cook \
#     RUN_DIR=runs/dexjoco_microwave_cook_uncond_3cam_384_1e-4_egodex_pretrain/2026-06-15_15-32-35 \
#     OUTPUT_DIR=logs/dexjoco_fastwam_eval/egodex_pretrain_step3000_clipped \
#     bash scripts/run_eval_dexjoco_fastwam.sh

set -euo pipefail

FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$FASTWAM_ROOT"

RUN_DIR="${RUN_DIR:-runs/dexjoco_microwave_cook_uncond_3cam_384_1e-4/2026-06-09_16-54-35}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-5560}"
EPISODES="${EPISODES:-50}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/dexjoco_fastwam_eval}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-1500}"
REPLAN_STEPS="${REPLAN_STEPS:-}"
ACTION_CLIP="${ACTION_CLIP:-0}"
CLIP_MAX_XYZ_STEP="${CLIP_MAX_XYZ_STEP:-0.05}"
CLIP_MAX_DZ_DOWN="${CLIP_MAX_DZ_DOWN:-0.03}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda not found"
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dexjoco

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${FASTWAM_ROOT}/src:${FASTWAM_ROOT}/scripts:${PYTHONPATH:-}"

python - <<'PY' >/dev/null 2>&1 || python -m pip install -q pyzmq msgpack
import msgpack
import zmq
PY

CACHE_DIR="${FASTWAM_ROOT}/data/text_embeds_cache/dexjoco_microwave_cook"
if ! ls "${CACHE_DIR}"/*.npz >/dev/null 2>&1; then
  echo "[run_eval_dexjoco_fastwam] exporting text caches to .npz (one-time)..."
  conda activate fastwam
  python "${FASTWAM_ROOT}/scripts/export_text_embed_cache_npz.py" --cache-dir "${CACHE_DIR}"
  conda activate dexjoco
fi

ARGS=(
  --run-dir "$RUN_DIR"
  --policy-host "$POLICY_HOST"
  --policy-port "$POLICY_PORT"
  --episodes "$EPISODES"
  --seed "$SEED"
  --output-dir "$OUTPUT_DIR"
  --max-env-steps "$MAX_ENV_STEPS"
)

if [[ -n "${TASKS:-}" ]]; then
  # shellcheck disable=SC2206
  TASK_ARR=($TASKS)
  ARGS+=(--tasks "${TASK_ARR[@]}")
fi

if [[ "${SAVE_VIDEO:-1}" == "0" ]]; then
  ARGS+=(--no-save-video)
fi

if [[ "${RANDOMIZE:-0}" == "1" ]]; then
  ARGS+=(--randomize)
fi

if [[ -n "${REPLAN_STEPS}" ]]; then
  ARGS+=(--replan-steps "${REPLAN_STEPS}")
fi

if [[ "${ACTION_CLIP}" == "1" ]]; then
  ARGS+=(--action-clip)
  ARGS+=(--clip-max-xyz-step "${CLIP_MAX_XYZ_STEP}")
  ARGS+=(--clip-max-dz-down "${CLIP_MAX_DZ_DOWN}")
fi

if [[ "${SAVE_ACTIONS:-1}" == "0" ]]; then
  ARGS+=(--no-save-actions)
fi

echo "[run_eval_dexjoco_fastwam] FASTWAM_ROOT=${FASTWAM_ROOT}"
echo "[run_eval_dexjoco_fastwam] RUN_DIR=${RUN_DIR}"
echo "[run_eval_dexjoco_fastwam] policy=${POLICY_HOST}:${POLICY_PORT} episodes=${EPISODES} replan_steps=${REPLAN_STEPS:-auto(0.8*action_horizon)}"
echo "[run_eval_dexjoco_fastwam] action_clip=${ACTION_CLIP} clip_max_xyz_step=${CLIP_MAX_XYZ_STEP} clip_max_dz_down=${CLIP_MAX_DZ_DOWN}"
echo "[run_eval_dexjoco_fastwam] conda env=dexjoco MUJOCO_GL=${MUJOCO_GL}"
echo

python "${FASTWAM_ROOT}/scripts/eval_dexjoco_fastwam.py" "${ARGS[@]}"
