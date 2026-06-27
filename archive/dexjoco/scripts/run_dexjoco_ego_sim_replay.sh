#!/usr/bin/env bash
# Open-loop sim replay for the full dexjoco_ego dataset.
#
# Same environment setup as dexjoco-openpi-eval in third_party/dexjoco/README.md,
# but actions come from dataset parquet instead of an OpenPI policy server.
#
# Usage (from FastWAM repo root):
#   bash scripts/run_dexjoco_ego_sim_replay.sh
#
# Smoke test (first 2 episodes per task):
#   DEXJOCO_REPLAY_MAX_EPISODES=2 bash scripts/run_dexjoco_ego_sim_replay.sh
#
# Resume a long run:
#   DEXJOCO_REPLAY_SKIP_EXISTING=1 bash scripts/run_dexjoco_ego_sim_replay.sh

set -euo pipefail

FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${DEXJOCO_REPLAY_DATASET_DIR:-${FASTWAM_ROOT}/data/dexjoco_ego}"
OUTPUT_DIR="${DEXJOCO_REPLAY_OUTPUT_DIR:-${FASTWAM_ROOT}/data/dexjoco_ego/sim_replay}"
SEED="${DEXJOCO_REPLAY_SEED:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda not found"
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dexjoco

export MUJOCO_GL=egl
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

REPLAY_ARGS=(
  --dataset-dir "${DATASET_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${SEED}"
  --save-all
  --group-by-task
)

if [[ -n "${DEXJOCO_REPLAY_MAX_EPISODES:-}" ]]; then
  REPLAY_ARGS+=(--max-episodes "${DEXJOCO_REPLAY_MAX_EPISODES}")
fi

if [[ "${DEXJOCO_REPLAY_SKIP_EXISTING:-0}" == "1" ]]; then
  REPLAY_ARGS+=(--skip-existing)
fi

if [[ "${DEXJOCO_REPLAY_RANDOMIZE:-0}" == "1" ]]; then
  REPLAY_ARGS+=(--randomize)
else
  REPLAY_ARGS+=(--no-randomize)
fi

if [[ "${DEXJOCO_REPLAY_RESTORE_STATE:-0}" == "1" ]]; then
  REPLAY_ARGS+=(--restore-state)
fi

echo "FastWAM root:     ${FASTWAM_ROOT}"
echo "Dataset:          ${DATASET_DIR}"
echo "Output:           ${OUTPUT_DIR}"
echo "Conda env:        dexjoco"
echo "MUJOCO_GL:         ${MUJOCO_GL}"
echo

python "${FASTWAM_ROOT}/scripts/replay_dexjoco_ego.py" "${REPLAY_ARGS[@]}"
