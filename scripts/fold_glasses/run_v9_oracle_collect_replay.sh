#!/usr/bin/env bash
# Thin wrapper: TASK=fold_glasses v9 base CFG collect event replay (oracle-once).
# Prefer: TASK=fold_glasses RUN_DIR=... GPUS=... bash scripts/dewo_v2/eval_v9_collect_event_replay.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASK="${TASK:-fold_glasses}"
export GPUS="${GPUS:-4,5,6,7}"
export CONDITIONS="${CONDITIONS:-v9_base,v9_oracle_once}"
export RUN_DIR="${RUN_DIR:-${ROOT_DIR}/runs/dexjoco_fold_glasses_dewo_v9/2026-08-27_11-21-16_B1-jump-fast-v9-uncond-adapter}"
exec bash "${ROOT_DIR}/scripts/dewo_v2/eval_v9_collect_event_replay.sh"
