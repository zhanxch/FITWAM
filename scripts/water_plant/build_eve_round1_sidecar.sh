#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${ALLOW_LEGACY_EVE_ROUND1:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
This legacy wrapper does not create the frozen episode-level train/val split
required by the formal Offline Self-Improving protocol.

Use scripts/water_plant/prepare_offline_self_improving.sh instead.
Set ALLOW_LEGACY_EVE_ROUND1=1 only to reproduce an older exploratory artifact.
EOF
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"

BASE_DATASET="${BASE_DATASET:-${ROOT_DIR}/data/water_plant_fastwam}"
ROLLOUT_RAW="${ROLLOUT_RAW:-${ROOT_DIR}/data/water_plant_rollout_200_step6500_raw}"
ROLLOUT_TRIM="${ROLLOUT_TRIM:-${ROOT_DIR}/data/water_plant_rollout_200_step6500_trim8s}"
EVE_ROOT="${EVE_ROOT:-${BASE_DATASET}/eve}"
SOURCE_POLICY="${SOURCE_POLICY:-fastwam_step6500}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${ROOT_DIR}/runs/water_plant_uncond_2cam_384_1e-4/2026-06-29_16-38-39/checkpoints/weights/step_006500.pt}"

python scripts/everobot/build_eve_sidecar.py init-base \
  --dataset-root "${BASE_DATASET}" \
  --dataset-id water_plant_fastwam \
  --eve-root "${EVE_ROOT}" \
  --task-name water_plant \
  --source-type expert_success \
  --source-policy human_or_expert \
  --collection-round -1

python scripts/everobot/build_eve_sidecar.py append-rollout \
  --base-eve-root "${EVE_ROOT}" \
  --rollout-root "${ROLLOUT_RAW}" \
  --trimmed-event-root "${ROLLOUT_TRIM}" \
  --dataset-id water_plant_rollout_200_step6500_raw \
  --task-name water_plant \
  --source-policy "${SOURCE_POLICY}" \
  --source-checkpoint "${SOURCE_CHECKPOINT}" \
  --collection-round 0 \
  --failure-action-loss disabled

python scripts/everobot/build_eve_sidecar.py build-manifest \
  --eve-root "${EVE_ROOT}" \
  --manifest-name train_round1_success_plus_failure_events \
  --include-outcomes success failure \
  --success-dataset-ids water_plant_fastwam \
  --failure-dataset-ids water_plant_rollout_200_step6500_raw \
  --failure-sample-mode event_only

echo "[build_eve_round1] wrote ${EVE_ROOT}"
