#!/usr/bin/env bash
# NON_STANDARD screening grid: ckpt × CFG_SCALE × adaptive tau × epsilon_l.
# Env knobs only — do not bake GPUS / dates / TASK into a copy of this file.
#
#   TASK=water_plant GPUS=0,1,2,3 RUN_DIR=... CKPT_STEPS="1500 1000 500" \
#     CFG_SCALES="1.2 1.3 1.5 2.0 2.5" ADAPTIVE_CFG_TAUS="none 0.04 0.05 0.06" \
#     CFG_EPSILON_LS="none 0.03 0.05" \
#     REPEATS=1 SCREEN_EVAL_REPEATS="1 0" GRID_ROOT=... \
#     bash scripts/dewo_v2/eval_cfg_hp_grid.sh
#
# ADAPTIVE_CFG_TAUS token "none" = constant mix (no adaptive).
# SCREEN_EVAL_REPEATS is the DexJoCo OPEN-stack noise index (client seed),
# not the unused server --inference-seed. Loop order: repeat, ckpt, scale, tau, epsilon_l.
# GRID_SHARD / GRID_SHARDS split the cartesian product across parallel workers.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/scripts/dewo_v2/lib.sh"

dewo_v2_require_task
dewo_v2_require_gpus
CALLER_REPEATS="${REPEATS:-}"
CALLER_CKPT="${CKPT:-}"
dewo_v2_load_task "${TASK}"
[[ -z "${CALLER_REPEATS}" ]] || export REPEATS="${CALLER_REPEATS}"
CKPT="${CALLER_CKPT:-${CKPT:-}}"

RUN_DIR="${RUN_DIR:?Set RUN_DIR}"
WEIGHTS_DIR="${WEIGHTS_DIR:-${RUN_DIR}/checkpoints/weights}"
CKPT_STEPS="${CKPT_STEPS:?Set CKPT_STEPS e.g. \"1500 1000 500\"}"
CFG_SCALES="${CFG_SCALES:-2.0}"
ADAPTIVE_CFG_TAUS="${ADAPTIVE_CFG_TAUS:-none}"
CFG_EPSILON_LS="${CFG_EPSILON_LS:-none}"
CFG_RESIDUAL_CLIP_MODE="${CFG_RESIDUAL_CLIP_MODE:-rms}"
REPEATS="${REPEATS:-1}"
SCREEN_EVAL_REPEATS="${SCREEN_EVAL_REPEATS:-${SCREEN_EVAL_REPEAT:-1}}"
WAIT_IDLE="${WAIT_IDLE:-0}"
BASE_PORT="${BASE_PORT:-7700}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
GRID_ROOT="${GRID_ROOT:-${ROOT_DIR}/evaluate_results/dexjoco/${TASK}_cfg_hp_grid_${STAMP}}"
GRID_SHARDS="${GRID_SHARDS:-1}"
GRID_SHARD="${GRID_SHARD:-0}"
TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR:?Set TEXT_EMBEDDING_CACHE_DIR}"
PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS:-${STATS}}"

mkdir -p "${GRID_ROOT}/logs"
MASTER_LOG="${GRID_ROOT}/logs/grid_shard${GRID_SHARD}.log"
log() { echo "[cfg-hp-grid shard${GRID_SHARD} $(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

log "task=${TASK} run_dir=${RUN_DIR}"
log "steps=${CKPT_STEPS} scales=${CFG_SCALES} taus=${ADAPTIVE_CFG_TAUS} epsilon_ls=${CFG_EPSILON_LS} clip_mode=${CFG_RESIDUAL_CLIP_MODE}"
log "gpus=${GPUS} repeats=${REPEATS} screen_eval_repeats=${SCREEN_EVAL_REPEATS} shard=${GRID_SHARD}/${GRID_SHARDS}"
log "grid_root=${GRID_ROOT}"

cell_index=-1
port_offset=0

write_summary() {
  python - "${GRID_ROOT}" "${TASK}" <<'PY'
import json, os, re, sys
from pathlib import Path
root = Path(sys.argv[1])
task = sys.argv[2]
rows = []
for agg in sorted(root.glob("step_*/aggregate.json")):
    d = json.loads(agg.read_text())
    name = agg.parent.name
    m = re.search(r"_rep(\d+)$", name)
    eval_repeat = d.get("screen_eval_repeat")
    if eval_repeat is None and m:
        eval_repeat = int(m.group(1))
    rows.append({
        "ckpt_tag": d.get("ckpt_tag"),
        "text_cfg_scale": d.get("text_cfg_scale"),
        "adaptive_cfg_tau": d.get("adaptive_cfg_tau"),
        "cfg_epsilon_l": d.get("cfg_epsilon_l"),
        "cfg_residual_clip_mode": d.get("cfg_residual_clip_mode"),
        "screen_eval_repeat": eval_repeat,
        "cfg_mix_high": d.get("cfg_mix_high"),
        "cfg_mix_low": d.get("cfg_mix_low"),
        "protocol": d.get("protocol"),
        "pooled_successes": d.get("pooled_successes"),
        "pooled_episodes": d.get("pooled_episodes"),
        "pooled_success_rate": d.get("pooled_success_rate"),
        "cfg_guided_chunk_fraction_mean": d.get("cfg_guided_chunk_fraction_mean"),
        "cfg_gate_exec_rms_mean": d.get("cfg_gate_exec_rms_mean"),
        "cfg_gate_observations": d.get("cfg_gate_observations"),
        "path": str(agg.parent),
    })
rows.sort(
    key=lambda r: (
        -(r["pooled_success_rate"] or 0.0),
        r["ckpt_tag"] or "",
        r["text_cfg_scale"] or 0,
        r["cfg_epsilon_l"] if r["cfg_epsilon_l"] is not None else -1,
        r["screen_eval_repeat"] if r["screen_eval_repeat"] is not None else -1,
    )
)
out = {
    "task": task,
    "method": "dewo_v5_cfg_hp_grid_NON_STANDARD",
    "protocol": "screening_1x50_seeds_0_49_NON_STANDARD",
    "note": (
        "Winner on this grid is a screen, not an official 4x50. "
        "screen_eval_repeat is the OPEN-stack diffusion noise index. "
        "Do not retune tau on the same 0-49 and report it as official."
    ),
    "n_cells": len(rows),
    "cells": rows,
}
tmp = root / f"grid_summary.json.tmp.{os.getpid()}"
tmp.write_text(json.dumps(out, indent=2) + "\n")
tmp.replace(root / "grid_summary.json")
print(json.dumps(out, indent=2))
for r in rows:
    tau = "none" if r["adaptive_cfg_tau"] is None else r["adaptive_cfg_tau"]
    epsilon = "none" if r["cfg_epsilon_l"] is None else r["cfg_epsilon_l"]
    print(
        f"{r['ckpt_tag']} scale={r['text_cfg_scale']} tau={tau} epsilon_l={epsilon} "
        f"rep={r['screen_eval_repeat']}: "
        f"pooled={r['pooled_successes']}/{r['pooled_episodes']} "
        f"rate={r['pooled_success_rate']} guided={r['cfg_guided_chunk_fraction_mean']}"
    )
PY
}

for eval_repeat in ${SCREEN_EVAL_REPEATS}; do
  for step in ${CKPT_STEPS}; do
    tag="$(printf 'step_%06d' "${step}")"
    ckpt="${WEIGHTS_DIR}/${tag}.pt"
    [[ -f "${ckpt}" ]] || { log "ERROR missing ${ckpt}"; exit 2; }
    for scale in ${CFG_SCALES}; do
      for tau in ${ADAPTIVE_CFG_TAUS}; do
        for epsilon_l in ${CFG_EPSILON_LS}; do
          cell_index=$((cell_index + 1))
          if (( GRID_SHARDS > 1 && cell_index % GRID_SHARDS != GRID_SHARD )); then
            continue
          fi
          if [[ "${tau}" == "none" ]]; then
            cell="${tag}_cfg${scale}_rep${eval_repeat}"
            tau_env=""
          else
            cell="${tag}_cfg${scale}_adapt_tau${tau}_rep${eval_repeat}"
            tau_env="${tau}"
          fi
          if [[ "${epsilon_l}" != "none" ]]; then
            cell="${cell%_rep${eval_repeat}}_eps${epsilon_l}_${CFG_RESIDUAL_CLIP_MODE}_rep${eval_repeat}"
            epsilon_env="${epsilon_l}"
          else
            epsilon_env=""
          fi
        out="${GRID_ROOT}/${cell}"
        legacy=""
        if [[ "${eval_repeat}" == "1" && "${epsilon_l}" == "none" ]]; then
          if [[ "${tau}" == "none" ]]; then
            legacy="${GRID_ROOT}/${tag}_cfg${scale}"
          else
            legacy="${GRID_ROOT}/${tag}_cfg${scale}_adapt_tau${tau}"
          fi
        fi
        if [[ -f "${out}/aggregate.json" ]]; then
          log "SKIP existing ${cell}"
          write_summary > /dev/null || true
          continue
        fi
        if [[ -n "${legacy}" && -f "${legacy}/aggregate.json" ]]; then
          log "SKIP legacy ${legacy##*/} → ${cell}"
          write_summary > /dev/null || true
          continue
        fi
        log "===== ${cell} ====="
        TASK="${TASK}" \
        RUN_DIR="${RUN_DIR}" \
        CKPT="${ckpt}" \
        CKPT_TAG="${tag}" \
        GPUS="${GPUS}" \
        WAIT_IDLE="${WAIT_IDLE}" \
        CFG_SCALE="${scale}" \
        ADAPTIVE_CFG_TAU="${tau_env}" \
        CFG_EPSILON_L="${epsilon_env}" \
        CFG_RESIDUAL_CLIP_MODE="${CFG_RESIDUAL_CLIP_MODE}" \
        REPEATS="${REPEATS}" \
        SCREEN_EVAL_REPEAT="${eval_repeat}" \
        BASE_PORT="$((BASE_PORT + port_offset))" \
        OUT_ROOT="${out}" \
        TEXT_EMBEDDING_CACHE_DIR="${TEXT_EMBEDDING_CACHE_DIR}" \
        PRETRAINED_NORM_STATS="${PRETRAINED_NORM_STATS}" \
        BACKBONE_CKPT="${BACKBONE_CKPT:-}" \
        UNCOND_ADAPTER="${UNCOND_ADAPTER:-}" \
        CFG_TASK_DIR="${CFG_TASK_DIR:-}" \
        bash "${ROOT_DIR}/scripts/dewo_v2/eval_cfg_official_4x50.sh" \
          2>&1 | tee -a "${MASTER_LOG}"
        port_offset=$((port_offset + 40))
        log "===== done ${cell} ====="
        write_summary | tee -a "${MASTER_LOG}"
        done
      done
    done
  done
done

log "DONE shard${GRID_SHARD} grid_summary=${GRID_ROOT}/grid_summary.json"
write_summary | tee -a "${MASTER_LOG}"
