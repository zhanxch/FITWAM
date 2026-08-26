#!/usr/bin/env bash
# Phase 2 (NON_STANDARD): on a frozen NOISE_SEED_BASE, sweep v6 ckpt × CFG.
# Baseline (w=1) is run once and reused; each cell is v6 adapter + adaptive tau.
#
# Example:
#   TASK=water_plant GPUS=4,5,6,7 RUN_DIR=... \
#     NOISE_SEED_BASE=20260824180500 \
#     CKPT_STEPS="500 1000 1500" CFG_SCALES="1.05 1.1 1.15 1.2" \
#     bash scripts/dewo_v2/search_noise_ckpt_cfg_grid.sh
#
# Or read base from phase-1 output:
#   SELECTED=/path/to/selected_noise_seed.json \
#     bash scripts/dewo_v2/search_noise_ckpt_cfg_grid.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_require_gpus
CALLER_CKPT="${CKPT:-}"
dewo_v2_load_task "${TASK}"
CKPT="${CALLER_CKPT:-${RUN_DIR}/checkpoints/weights/step_001500.pt}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${RUN_DIR}/checkpoints/weights}"
BACKBONE_CKPT="${BACKBONE_CKPT:-${ROOT_DIR}/artifacts/opensource_ckpt_links/mixed_5task/step_055000.pt}"
PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS:-${ROOT_DIR}/artifacts/mixed_5task/dataset_stats.json}"
TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:-${ROOT_DIR}/data/water_plant_mixed_s0_dewo_v2_pair_20260820_182236/text_embeds_cache}"
CFG_TASK_DIR="${CFG_TASK_DIR:-${ROOT_DIR}/configs/eval/dexjoco/${TASK}_dewo_v6_cfg}"

if [[ -n "${SELECTED:-}" && -f "${SELECTED}" ]]; then
  NOISE_SEED_BASE="${NOISE_SEED_BASE:-$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["noise_seed_base"])' "${SELECTED}"
  )}"
fi
NOISE_SEED_BASE="${NOISE_SEED_BASE:?Set NOISE_SEED_BASE or SELECTED=.../selected_noise_seed.json}"

ADAPTIVE_CFG_TAU="${ADAPTIVE_CFG_TAU:-0.04167268648743629}"
CKPT_STEPS="${CKPT_STEPS:-500 1000 1500}"
CFG_SCALES="${CFG_SCALES:-1.05 1.1 1.15 1.2}"
BASELINE_MAX="${BASELINE_MAX:-0.86}"
V6_MIN="${V6_MIN:-0.90}"
REPEATS="${REPEATS:-1}"
ENV_SEED="${ENV_SEED:-0}"
WAIT_IDLE="${WAIT_IDLE:-0}"
BASE_PORT="${BASE_PORT:-8600}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
GRID_ROOT="${GRID_ROOT:-${ROOT_DIR}/evaluate_results/dexjoco/${TASK}_noise_${NOISE_SEED_BASE}_ckpt_cfg_${STAMP}}"

mkdir -p "${GRID_ROOT}/logs"
MASTER_LOG="${GRID_ROOT}/logs/grid.log"
RESULTS_JSONL="${GRID_ROOT}/grid_results.jsonl"
touch "${RESULTS_JSONL}"

log() { echo "[noise-ckpt-cfg $(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

log "task=${TASK} noise_seed_base=${NOISE_SEED_BASE}"
log "ckpt_steps=${CKPT_STEPS} cfg_scales=${CFG_SCALES} tau=${ADAPTIVE_CFG_TAU}"
log "targets: baseline<${BASELINE_MAX} v6>${V6_MIN}"
log "grid_root=${GRID_ROOT}"

read_rate() {
  python3 - "${1}" <<'PY'
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

# Baseline once (any adapter ckpt; w=1 bypass).
BASELINE_CKPT="${BASELINE_CKPT:-${WEIGHTS_DIR}/step_001500.pt}"
baseline_out="${GRID_ROOT}/baseline_cfg1"
port_offset=0
if [[ ! -f "${baseline_out}/aggregate.json" ]]; then
  log "baseline CFG=1.0 NOISE_SEED_BASE=${NOISE_SEED_BASE}"
  TASK="${TASK}" GPUS="${GPUS}" WAIT_IDLE="${WAIT_IDLE}" REPEATS="${REPEATS}" \
    ENV_SEED="${ENV_SEED}" NOISE_SEED_BASE="${NOISE_SEED_BASE}" \
    RUN_DIR="${RUN_DIR}" CKPT="${BASELINE_CKPT}" BACKBONE_CKPT="${BACKBONE_CKPT}" \
    PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS}" \
    TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR}" \
    CFG_TASK_DIR="${CFG_TASK_DIR}" CFG_SCALE=1.0 ADAPTIVE_CFG_TAU="" \
    BASE_PORT="${BASE_PORT}" OUT_ROOT="${baseline_out}" \
    METHOD="mixed_s0_baseline_w1" \
    bash "${ROOT_DIR}/scripts/dewo_v2/eval_cfg_official_4x50.sh" \
    >> "${GRID_ROOT}/logs/baseline.log" 2>&1 || log "WARN baseline failed"
else
  log "reuse baseline ${baseline_out}"
fi
port_offset=$((port_offset + 50))
baseline_rate="$(read_rate "${baseline_out}/aggregate.json")"
log "baseline_rate=${baseline_rate}"

cell_index=0
for step in ${CKPT_STEPS}; do
  ckpt="${WEIGHTS_DIR}/step_$(printf '%06d' "${step}").pt"
  [[ -f "${ckpt}" ]] || { log "ERROR missing ${ckpt}"; exit 2; }
  for cfg in ${CFG_SCALES}; do
    cell_index=$((cell_index + 1))
    cell_dir="${GRID_ROOT}/step_$(printf '%06d' "${step}")_cfg${cfg}"
    if [[ -f "${cell_dir}/aggregate.json" ]]; then
      log "skip step=${step} cfg=${cfg} (exists)"
      v6_rate="$(read_rate "${cell_dir}/aggregate.json")"
    else
      log "v6 step=${step} cfg=${cfg} tau=${ADAPTIVE_CFG_TAU}"
      TASK="${TASK}" GPUS="${GPUS}" WAIT_IDLE="${WAIT_IDLE}" REPEATS="${REPEATS}" \
        ENV_SEED="${ENV_SEED}" NOISE_SEED_BASE="${NOISE_SEED_BASE}" \
        RUN_DIR="${RUN_DIR}" CKPT="${ckpt}" BACKBONE_CKPT="${BACKBONE_CKPT}" \
        PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS}" \
        TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR}" \
        CFG_TASK_DIR="${CFG_TASK_DIR}" CFG_SCALE="${cfg}" \
        ADAPTIVE_CFG_TAU="${ADAPTIVE_CFG_TAU}" \
        BASE_PORT="$((BASE_PORT + port_offset))" \
        OUT_ROOT="${cell_dir}" \
        bash "${ROOT_DIR}/scripts/dewo_v2/eval_cfg_official_4x50.sh" \
        >> "${GRID_ROOT}/logs/step_${step}_cfg${cfg}.log" 2>&1 || log "WARN failed step=${step} cfg=${cfg}"
      v6_rate="$(read_rate "${cell_dir}/aggregate.json")"
    fi
    port_offset=$((port_offset + 50))
    log "v6_rate=${v6_rate} step=${step} cfg=${cfg}"

    python3 - "${RESULTS_JSONL}" "${NOISE_SEED_BASE}" "${step}" "${cfg}" "${baseline_rate}" "${v6_rate}" "${cell_dir}" "${BASELINE_MAX}" "${V6_MIN}" <<'PY'
import json, sys
from pathlib import Path

out, base, step, cfg, base_r, v6_r, cell, bmax, vmin = sys.argv[1:10]
def f(x):
    return None if x == "nan" else float(x)
br, vr = f(base_r), f(v6_r)
row = {
    "noise_seed_base": int(base),
    "ckpt_step": int(step),
    "cfg_scale": float(cfg),
    "baseline_rate": br,
    "v6_rate": vr,
    "delta": (vr - br) if br is not None and vr is not None else None,
    "passes": (
        br is not None and vr is not None
        and br < float(bmax) and vr > float(vmin)
    ),
    "out_dir": cell,
}
Path(out).open("a", encoding="utf-8").write(json.dumps(row) + "\n")
print(json.dumps(row, indent=2))
PY
  done
done

python3 - "${GRID_ROOT}" "${NOISE_SEED_BASE}" "${ADAPTIVE_CFG_TAU}" "${BASELINE_MAX}" "${V6_MIN}" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
base = int(sys.argv[2])
tau = float(sys.argv[3])
bmax = float(sys.argv[4])
vmin = float(sys.argv[5])
rows = []
for line in (root / "grid_results.jsonl").read_text().splitlines():
    if line.strip():
        rows.append(json.loads(line))
passing = [r for r in rows if r.get("passes")]
best = None
if passing:
    best = max(passing, key=lambda r: (r["v6_rate"], r["delta"]))
elif rows:
    valid = [r for r in rows if r.get("v6_rate") is not None]
    if valid:
        best = max(valid, key=lambda r: r["v6_rate"])
summary = {
    "noise_seed_base": base,
    "noise_seed_bases_4x50": [base + i for i in range(4)],
    "adaptive_cfg_tau": tau,
    "baseline_rate": rows[0]["baseline_rate"] if rows else None,
    "n_cells": len(rows),
    "n_passing": len(passing),
    "passing": passing,
    "best": best,
    "frozen_noise_pack": (
        {
            "env_seeds": [0, 49],
            "noise_seed_base": base,
            "noise_seed_bases_4x50": [base + i for i in range(4)],
            "adaptive_cfg_tau": tau,
            "ckpt_step": best["ckpt_step"],
            "cfg_scale": best["cfg_scale"],
            "baseline_rate_1x50": best["baseline_rate"],
            "v6_rate_1x50": best["v6_rate"],
        }
        if best
        else None
    ),
}
(root / "grid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
if best and best.get("passes"):
    (root / "frozen_noise_pack.json").write_text(
        json.dumps(summary["frozen_noise_pack"], indent=2) + "\n"
    )
print(json.dumps(summary, indent=2))
PY

log "DONE ${GRID_ROOT}/grid_summary.json"
