#!/usr/bin/env bash
# Phase 1 (NON_STANDARD): scan NOISE_SEED_BASE candidates with baseline only
# (CFG_SCALE=1.0, mixed S0 本体 bypass). Pick the base whose 1×50 rate is
# closest to BASELINE_TARGET (~85%), ideally within BASELINE_MIN..BASELINE_MAX.
#
# Phase 2 (ckpt × cfg) runs separately after freezing the chosen base:
#   NOISE_SEED_BASE=... bash scripts/dewo_v2/search_noise_ckpt_cfg_grid.sh
#
# Example:
#   TASK=water_plant GPUS=4,5,6,7 RUN_DIR=... \
#     NOISE_SEED_START=20260824180000 NOISE_SEED_COUNT=12 \
#     bash scripts/dewo_v2/search_noise_seed_screen.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_require_gpus
CALLER_CKPT="${CKPT:-}"
dewo_v2_load_task "${TASK}"

RUN_DIR="${RUN_DIR:?Set RUN_DIR to the DEWO v6 training run directory}"
# Baseline uses adapter ckpt with w=1 bypass (same stack as v6 eval).
CKPT="${CALLER_CKPT:-${RUN_DIR}/checkpoints/weights/step_001500.pt}"
BACKBONE_CKPT="${BACKBONE_CKPT:-${ROOT_DIR}/artifacts/opensource_ckpt_links/mixed_5task/step_055000.pt}"
PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS:-${ROOT_DIR}/artifacts/mixed_5task/dataset_stats.json}"
TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:-${ROOT_DIR}/data/water_plant_mixed_s0_dewo_v2_pair_20260820_182236/text_embeds_cache}"
CFG_TASK_DIR="${CFG_TASK_DIR:-${ROOT_DIR}/configs/eval/dexjoco/${TASK}_dewo_v6_cfg}"

BASELINE_TARGET="${BASELINE_TARGET:-0.85}"
BASELINE_MIN="${BASELINE_MIN:-0.82}"
BASELINE_MAX="${BASELINE_MAX:-0.86}"
REPEATS="${REPEATS:-1}"
ENV_SEED="${ENV_SEED:-0}"
WAIT_IDLE="${WAIT_IDLE:-0}"
BASE_PORT="${BASE_PORT:-8200}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SEARCH_ROOT="${SEARCH_ROOT:-${ROOT_DIR}/evaluate_results/dexjoco/${TASK}_noise_seed_baseline_${STAMP}}"

mkdir -p "${SEARCH_ROOT}/logs"
MASTER_LOG="${SEARCH_ROOT}/logs/search.log"
CANDIDATES_JSONL="${SEARCH_ROOT}/candidates.jsonl"
touch "${CANDIDATES_JSONL}"

log() { echo "[noise-seed-baseline $(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

if [[ -z "${NOISE_SEED_BASES:-}" ]]; then
  NOISE_SEED_START="${NOISE_SEED_START:-$(date +%Y%m%d%H%M%S)}"
  NOISE_SEED_COUNT="${NOISE_SEED_COUNT:-12}"
  NOISE_SEED_STEP_MIN="${NOISE_SEED_STEP_MIN:-3}"
  log "generating ${NOISE_SEED_COUNT} bases from start=${NOISE_SEED_START} step_min=${NOISE_SEED_STEP_MIN}"
  NOISE_SEED_BASES="$(
    python3 - "${NOISE_SEED_START}" "${NOISE_SEED_COUNT}" "${NOISE_SEED_STEP_MIN}" <<'PY'
import datetime as dt
import sys

start = int(sys.argv[1])
count = int(sys.argv[2])
step_min = int(sys.argv[3])
base_dt = dt.datetime.strptime(str(start), "%Y%m%d%H%M%S")
print(" ".join(
    (base_dt + dt.timedelta(minutes=step_min * i)).strftime("%Y%m%d%H%M%S")
    for i in range(count)
))
PY
  )"
fi

log "task=${TASK} phase=baseline_only"
log "ckpt=${CKPT} backbone=${BACKBONE_CKPT}"
log "pick baseline closest to ${BASELINE_TARGET} (band ${BASELINE_MIN}..${BASELINE_MAX})"
log "noise_bases=${NOISE_SEED_BASES}"
log "search_root=${SEARCH_ROOT}"

read_rate() {
  local agg="$1"
  python3 - "${agg}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print("nan")
    raise SystemExit(0)
d = json.loads(p.read_text())
print(d.get("mean_success_rate", d.get("pooled_success_rate", "nan")))
PY
}

write_summary() {
  python3 - "${CANDIDATES_JSONL}" "${SEARCH_ROOT}" "${BASELINE_TARGET}" "${BASELINE_MIN}" "${BASELINE_MAX}" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
root = Path(sys.argv[2])
target = float(sys.argv[3])
lo = float(sys.argv[4])
hi = float(sys.argv[5])
rows = []
for line in out.read_text().splitlines():
    if line.strip():
        rows.append(json.loads(line))
valid = [r for r in rows if r.get("baseline_rate") is not None]
in_band = [r for r in valid if lo <= r["baseline_rate"] <= hi]

def score(r):
    rate = r["baseline_rate"]
    dist = abs(rate - target)
    in_band_bonus = 0.0 if lo <= rate <= hi else 1.0
    return (in_band_bonus, dist, -rate)

best = min(valid, key=score) if valid else None
summary = {
    "phase": "baseline_only",
    "search_root": str(root),
    "baseline_target": target,
    "baseline_band": [lo, hi],
    "n_candidates": len(rows),
    "n_valid": len(valid),
    "n_in_band": len(in_band),
    "best_noise_seed_base": best["noise_seed_base"] if best else None,
    "best_baseline_rate": best["baseline_rate"] if best else None,
    "best_in_band": (
        lo <= best["baseline_rate"] <= hi if best else False
    ),
    "frozen_noise_pack_draft": (
        {
            "env_seeds": [0, 49],
            "noise_seed_base": best["noise_seed_base"],
            "noise_seed_bases_4x50": [
                best["noise_seed_base"] + i for i in range(4)
            ],
            "baseline_rate_1x50": best["baseline_rate"],
            "note": "Phase 2: run search_noise_ckpt_cfg_grid.sh on this base",
        }
        if best
        else None
    ),
    "in_band": in_band,
    "all": rows,
}
(root / "search_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
if best:
    (root / "selected_noise_seed.json").write_text(
        json.dumps(summary["frozen_noise_pack_draft"], indent=2) + "\n"
    )
print(json.dumps({
    "best_noise_seed_base": summary["best_noise_seed_base"],
    "best_baseline_rate": summary["best_baseline_rate"],
    "n_in_band": summary["n_in_band"],
}, indent=2))
PY
}

port_offset=0
for base in ${NOISE_SEED_BASES}; do
  base_dir="${SEARCH_ROOT}/base_${base}"
  mkdir -p "${base_dir}/logs"
  log "=== baseline only NOISE_SEED_BASE=${base} ==="

  baseline_out="${base_dir}/baseline_cfg1"
  if [[ ! -f "${baseline_out}/aggregate.json" ]]; then
    TASK="${TASK}" GPUS="${GPUS}" WAIT_IDLE="${WAIT_IDLE}" REPEATS="${REPEATS}" \
      ENV_SEED="${ENV_SEED}" NOISE_SEED_BASE="${base}" \
      RUN_DIR="${RUN_DIR}" CKPT="${CKPT}" BACKBONE_CKPT="${BACKBONE_CKPT}" \
      PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS}" \
      TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR}" \
      CFG_TASK_DIR="${CFG_TASK_DIR}" CFG_SCALE=1.0 ADAPTIVE_CFG_TAU="" \
      BASE_PORT="$((BASE_PORT + port_offset))" \
      OUT_ROOT="${baseline_out}" \
      METHOD="mixed_s0_baseline_w1" \
      bash "${ROOT_DIR}/scripts/dewo_v2/eval_cfg_official_4x50.sh" \
      >> "${base_dir}/logs/baseline.log" 2>&1 || {
        log "WARN baseline failed base=${base}"
      }
  else
    log "skip baseline (exists) base=${base}"
  fi
  port_offset=$((port_offset + 50))

  baseline_rate="$(read_rate "${baseline_out}/aggregate.json")"
  log "baseline_rate=${baseline_rate} base=${base}"

  python3 - "${CANDIDATES_JSONL}" "${base}" "${baseline_rate}" "${baseline_out}" "${BASELINE_MIN}" "${BASELINE_MAX}" <<'PY'
import json, sys
from pathlib import Path

out, base_s, rate_s, bout, lo, hi = sys.argv[1:7]
rate = None if rate_s == "nan" else float(rate_s)
row = {
    "noise_seed_base": int(base_s),
    "baseline_rate": rate,
    "baseline_out": bout,
    "in_band": rate is not None and float(lo) <= rate <= float(hi),
}
Path(out).open("a", encoding="utf-8").write(json.dumps(row) + "\n")
print(json.dumps(row, indent=2))
PY
done

write_summary | tee -a "${MASTER_LOG}"
log "DONE ${SEARCH_ROOT}/search_summary.json selected=${SEARCH_ROOT}/selected_noise_seed.json"
