#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TASK="${TASK:-water_plant}"
exec bash "${ROOT_DIR}/scripts/dewo_v2/collect_opensource_4x50.sh"
