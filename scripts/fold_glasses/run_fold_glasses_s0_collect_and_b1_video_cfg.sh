#!/usr/bin/env bash
# fold_glasses S0 → 4×50 rollout collect (seed 10086..10135, max_env_steps=1200)
# → success-length(+3s) failure trim → B1-video-cfg train on GPUs 4,5,6,7.
#
# Waits for:
#   1) step_010000.pt download complete (expected size)
#   2) GPUs 4-7 idle enough to host policy servers + collectors
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${FITWAM_ENV:-fastwam}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
ENV_PREFIX="${FITWAM_ENV_PREFIX:-$(conda info --base)/envs/fastwam}"
export FITWAM_ENV_PREFIX="${ENV_PREFIX}"
export PATH="${ENV_PREFIX}/bin:${PATH}"

EXPECTED_CKPT_BYTES="${EXPECTED_CKPT_BYTES:-12041919929}"
CKPT="${CKPT:-${ROOT_DIR}/checkpoints/dexjoco/fold_glasses_fastwam/weights/step_010000.pt}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/checkpoints/dexjoco/fold_glasses_fastwam/s0_bundle}"
BASE_DATASET="${BASE_DATASET:-${ROOT_DIR}/data/fold_glasses_fastwam}"
NORM_STATS_META_DIR="${NORM_STATS_META_DIR:-${BASE_DATASET}/meta}"
TEXT_CACHE="${TEXT_CACHE:-${ROOT_DIR}/data/text_embeds_cache/fold_glasses}"
TASK_CONFIG_DIR="${TASK_CONFIG_DIR:-${ROOT_DIR}/third_party/dexjoco/configs/rand_obj}"
SOURCE_DATASET="${SOURCE_DATASET:-${ROOT_DIR}/data/dexjoco/dexjoco_lerobot_datasets/fold_glasses}"
SUCCESS_PROMPT="${SUCCESS_PROMPT:-Fold the glasses and place them into the case.}"

GPUS="${GPUS:-4,5,6,7}"
BASE_SEED="${BASE_SEED:-10086}"
EPISODES_PER_RUN="${EPISODES_PER_RUN:-50}"
NUM_REPEATS="${NUM_REPEATS:-4}"
REPLAN_STEPS="${REPLAN_STEPS:-25}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-1200}"
VIDEO_FPS="${VIDEO_FPS:-30}"
EXTEND_SUCCESS_SECONDS="${EXTEND_SUCCESS_SECONDS:-3}"
IDLE_MEM_MIB="${IDLE_MEM_MIB:-12000}"
IDLE_UTIL_MAX="${IDLE_UTIL_MAX:-15}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-${ROOT_DIR}/data/fold_glasses_s0_b1_video_cfg_${STAMP}}"
COLLECTION_ROOT="${COLLECTION_ROOT:-${EXP_ROOT}/collection}"
ROLLOUT_RAW="${ROLLOUT_RAW:-${EXP_ROOT}/rollout_raw_200}"
TRIM_META_ROOT="${TRIM_META_ROOT:-${EXP_ROOT}/success_len_trim_meta}"
EVE_ROOT="${EVE_ROOT:-${EXP_ROOT}/eve_v02}"
LOG_DIR="${LOG_DIR:-${EXP_ROOT}/logs}"
mkdir -p "${COLLECTION_ROOT}" "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/orchestrator.log"

log() { echo "[fold-glasses-b1-video-cfg $(date -Is)] $*" | tee -a "${MASTER_LOG}"; }

wait_for_ckpt() {
  log "waiting for checkpoint ${CKPT} (expected ${EXPECTED_CKPT_BYTES} bytes)"
  while true; do
    if [[ -f "${CKPT}" ]]; then
      local actual
      actual="$(stat -c%s "${CKPT}" 2>/dev/null || echo 0)"
      # aria2 control file means still downloading
      if [[ ! -f "${CKPT}.aria2" && "${actual}" == "${EXPECTED_CKPT_BYTES}" ]]; then
        log "DOWNLOAD_OK bytes=${actual}"
        return 0
      fi
      log "download progress bytes=${actual}/$(printf '%s' "${EXPECTED_CKPT_BYTES}") aria2=$([ -f "${CKPT}.aria2" ] && echo yes || echo no)"
    else
      log "checkpoint missing"
    fi
    sleep 30
  done
}

gpus_idle() {
  python - "${GPUS}" "${IDLE_MEM_MIB}" "${IDLE_UTIL_MAX}" <<'PY'
import subprocess, sys
gpus = [int(x) for x in sys.argv[1].split(",") if x.strip()]
mem_lim = int(sys.argv[2])
util_lim = int(sys.argv[3])
out = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
rows = {}
for line in out.strip().splitlines():
    idx, mem, util = [p.strip() for p in line.split(",")]
    rows[int(idx)] = (int(float(mem)), int(float(util)))
ok = True
for g in gpus:
    mem, util = rows[g]
    # Allow residual fragmentation; treat as busy if high util OR heavy memory.
    if util > util_lim or mem > mem_lim:
        ok = False
    print(f"gpu{g}: mem={mem}MiB util={util}%")
raise SystemExit(0 if ok else 1)
PY
}

wait_for_gpus() {
  log "waiting for GPUs ${GPUS} idle (mem<=${IDLE_MEM_MIB}MiB util<=${IDLE_UTIL_MAX}%)"
  while true; do
    if gpus_idle >>"${MASTER_LOG}" 2>&1; then
      log "GPUs ${GPUS} idle"
      return 0
    fi
    sleep 60
  done
}

ensure_text_embed_npz() {
  # DexJoCo client (dexjoco env) cannot import torch; collectors require .npz caches.
  log "ensuring torch-free text embed npz under ${TEXT_CACHE}"
  mkdir -p "${TEXT_CACHE}"
  "${ENV_PREFIX}/bin/python" scripts/export_text_embed_cache_npz.py \
    --cache-dir "${TEXT_CACHE}"
  local npz_count
  npz_count="$(find "${TEXT_CACHE}" -maxdepth 1 -name '*.npz' | wc -l | tr -d ' ')"
  if [[ "${npz_count}" -lt 1 ]]; then
    log "ERROR: no .npz text embeds in ${TEXT_CACHE} after export"
    exit 2
  fi
  log "text embed npz count=${npz_count}"
}

finalize_s0_bundle() {
  local bundle="${RUN_DIR}"
  mkdir -p "${bundle}/norm_stats_meta" "${bundle}/text_cache"
  ln -sfn "${CKPT}" "${bundle}/step_010000.pt"
  rm -f "${bundle}/step_007500.pt"
  cp -f "${NORM_STATS_META_DIR}/stats.json" "${bundle}/norm_stats_meta/"
  cp -f "${NORM_STATS_META_DIR}/modality.json" "${bundle}/norm_stats_meta/" 2>/dev/null || true
  if [[ ! -f "${bundle}/config.yaml" ]]; then
    log "missing ${bundle}/config.yaml"
    exit 2
  fi
  python - "${bundle}" "${CKPT}" <<'PY'
from pathlib import Path
import hashlib, sys
bundle = Path(sys.argv[1])
ckpt = Path(sys.argv[2])

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

lines = [
    f"normalization_kind=meta",
    f"sha256",
    f"{sha(ckpt)}  step_010000.pt",
    f"{sha(bundle / 'config.yaml')}  config.yaml",
    f"{sha(bundle / 'norm_stats_meta' / 'stats.json')}  norm_stats_meta/stats.json",
]
(bundle / 'bundle_manifest.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('wrote', bundle / 'bundle_manifest.txt')
print('ckpt_sha', lines[2].split()[0])
PY
}

collect_one_run() {
  local run_i="$1"
  local out_dir="${COLLECTION_ROOT}/run${run_i}"
  local raw_ds="${out_dir}/raw"
  mkdir -p "${out_dir}"
  log "=== collect run ${run_i}/${NUM_REPEATS} seed=${BASE_SEED}..$((BASE_SEED + EPISODES_PER_RUN - 1)) ==="
  "${ENV_PREFIX}/bin/python" scripts/dexjoco_async/run_multi_gpu_dexjoco_collect.py \
    --gpus "${GPUS}" \
    --base-port "$((5600 + run_i * 20))" \
    --episodes "${EPISODES_PER_RUN}" \
    --seed "${BASE_SEED}" \
    --server-conda-env "${ENV_PREFIX}" \
    --client-conda-env dexjoco \
    --run-dir "${RUN_DIR}" \
    --checkpoint "${CKPT}" \
    --no-load-text-encoder \
    --norm-stats-meta-dir "${NORM_STATS_META_DIR}" \
    --text-embedding-cache-dir "${TEXT_CACHE}" \
    --task-config-dir "${TASK_CONFIG_DIR}" \
    --tasks fold_glasses \
    --source-dataset "${SOURCE_DATASET}" \
    --success-prompt "${SUCCESS_PROMPT}" \
    --replan-steps "${REPLAN_STEPS}" \
    --max-env-steps "${MAX_ENV_STEPS}" \
    --video-fps "${VIDEO_FPS}" \
    --no-randomize \
    --no-randomize-dynamics \
    --no-action-clip \
    --outcome-task-mode clean \
    --output-dir "${out_dir}" \
    --raw-output-dataset "${raw_ds}" \
    --overwrite \
    2>&1 | tee -a "${LOG_DIR}/collect_run${run_i}.log"
  log "collect run${run_i} done -> ${raw_ds}"
}

merge_repeats() {
  log "merging ${NUM_REPEATS} raw datasets -> ${ROLLOUT_RAW}"
  local shard_args=()
  local i
  for i in $(seq 1 "${NUM_REPEATS}"); do
    shard_args+=("${COLLECTION_ROOT}/run${i}/raw")
  done
  "${ENV_PREFIX}/bin/python" scripts/build_rollout_datasets.py merge-shards \
    --shard-datasets "${shard_args[@]}" \
    --output-dataset "${ROLLOUT_RAW}" \
    --overwrite
  "${ENV_PREFIX}/bin/python" scripts/build_rollout_datasets.py validate-outcomes \
    --dataset "${ROLLOUT_RAW}" \
    --expected-episodes "$((EPISODES_PER_RUN * NUM_REPEATS))" \
    --report "${EXP_ROOT}/rollout_outcome_validation.json"
}

write_success_len_trim_report() {
  log "building success-length(+${EXTEND_SUCCESS_SECONDS}s) failure trim report"
  python - \
    "${ROLLOUT_RAW}" \
    "${TRIM_META_ROOT}" \
    "${EXTEND_SUCCESS_SECONDS}" <<'PY'
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

rollout_root = Path(sys.argv[1]).resolve()
trim_root = Path(sys.argv[2]).resolve()
extend_seconds = float(sys.argv[3])

info = json.loads((rollout_root / "meta" / "info.json").read_text(encoding="utf-8"))
fps = int(info["fps"])
extend_steps = int(round(extend_seconds * fps))

episodes = [
    json.loads(line)
    for line in (rollout_root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
outcomes = {
    int(row["episode_index"]): row
    for row in (
        json.loads(line)
        for line in (rollout_root / "meta" / "episode_outcomes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
}

success_lengths_by_seed: dict[int, list[int]] = defaultdict(list)
all_success_lengths: list[int] = []
for ep in episodes:
    ep_idx = int(ep["episode_index"])
    length = int(ep["length"])
    row = outcomes[ep_idx]
    if row["outcome"] == "success":
        seed = int(row["seed"])
        success_lengths_by_seed[seed].append(length)
        all_success_lengths.append(length)

if not all_success_lengths:
    raise SystemExit("No success trajectories found; cannot set failure trim length.")

mean_success = statistics.mean(all_success_lengths)
trim_report = []
same_seed_hits = 0
mean_fallback = 0
for ep in sorted(episodes, key=lambda row: int(row["episode_index"])):
    ep_idx = int(ep["episode_index"])
    length = int(ep["length"])
    row = outcomes[ep_idx]
    seed = int(row["seed"])
    is_failure = row["outcome"] == "failure"
    ref_source = "success"
    ref_length = length
    if is_failure:
        same = success_lengths_by_seed.get(seed) or []
        if same:
            ref_length = float(statistics.mean(same))
            ref_source = "same_seed_success_mean"
            same_seed_hits += 1
        else:
            ref_length = float(mean_success)
            ref_source = "global_success_mean"
            mean_fallback += 1
        cutoff = int(math.floor(ref_length + extend_steps))
        keep = max(1, min(length, cutoff))
        should_trim = keep < length
    else:
        keep = length
        should_trim = False
        ref_source = "success_keep_full"
        ref_length = float(length)
    trim_report.append(
        {
            "episode_index": ep_idx,
            "seed": seed,
            "failure": is_failure,
            "trimmed": bool(should_trim),
            "original_length": length,
            "trimmed_length": keep,
            "trimmed_tail_steps": length - keep,
            "trim_start_frame": 0,
            "trim_end_frame": keep,
            "reference_source": ref_source,
            "reference_success_length": ref_length,
            "extend_seconds": extend_seconds,
            "extend_steps": extend_steps,
        }
    )

summary = {
    "status": "complete",
    "mode": "failure_cutoff_success_length_plus_extend",
    "source_dataset": str(rollout_root),
    "extend_success_seconds": extend_seconds,
    "fps": fps,
    "extend_steps": extend_steps,
    "mean_success_length": mean_success,
    "num_successes": len(all_success_lengths),
    "episodes": len(trim_report),
    "failures": sum(1 for row in trim_report if row["failure"]),
    "successes": sum(1 for row in trim_report if not row["failure"]),
    "trimmed_failures": sum(1 for row in trim_report if row["trimmed"]),
    "untrimmed_failures": sum(
        1 for row in trim_report if row["failure"] and not row["trimmed"]
    ),
    "same_seed_success_refs": same_seed_hits,
    "global_mean_success_refs": mean_fallback,
    "note": (
        "Metadata-only trim for B1-video-cfg. Each failure keeps "
        "[0, min(T, success_ref+extend)]. Same-seed success length preferred; "
        "else global mean success length."
    ),
    "trim_report": trim_report,
}
trim_root.mkdir(parents=True, exist_ok=True)
(trim_root / "collection_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(
    f"[trim] failures={summary['failures']} trimmed={summary['trimmed_failures']} "
    f"same_seed={same_seed_hits} mean_fallback={mean_fallback} "
    f"mean_success={mean_success:.1f} extend_steps={extend_steps}"
)
PY
}

prepare_eve_and_manifests() {
  log "preparing Eve sidecar + B1-video-cfg manifests"
  local code_commit
  code_commit="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
  local ckpt_sha
  ckpt_sha="$(python -c "import hashlib,sys; from pathlib import Path; p=Path(sys.argv[1]); h=hashlib.sha256();
f=p.open('rb');
import itertools
[h.update(c) for c in iter(lambda:f.read(1<<20), b'')];
print(h.hexdigest())" "${CKPT}")"

  mkdir -p "${EVE_ROOT}/splits" "${EVE_ROOT}/manifests" "${EVE_ROOT}/protocol"

  # Episode split: expert all-success + rollout outcomes, hold out ~20% expert for val.
  if [[ ! -f "${EVE_ROOT}/splits/episode_splits.jsonl" ]]; then
    "${ENV_PREFIX}/bin/python" scripts/everobot/build_episode_split.py \
      --dataset "fold_glasses_expert_success=${BASE_DATASET}" \
      --dataset "fold_glasses_s0_rollout=${ROLLOUT_RAW}" \
      --force-success-dataset-id fold_glasses_expert_success \
      --require-explicit-outcome-dataset-id fold_glasses_s0_rollout \
      --val-fraction 0.2 \
      --seed 20260807 \
      --output "${EVE_ROOT}/splits/episode_splits.jsonl" \
      --report "${EVE_ROOT}/splits/episode_splits.report.json"
  fi

  if [[ ! -f "${EVE_ROOT}/episode_meta.jsonl" ]]; then
    "${ENV_PREFIX}/bin/python" scripts/everobot/build_eve_sidecar.py init-base \
      --dataset-root "${BASE_DATASET}" \
      --dataset-id fold_glasses_expert_success \
      --eve-root "${EVE_ROOT}" \
      --task-name fold_glasses \
      --source-type expert_success \
      --source-policy expert \
      --collection-round -1 \
      --force-success \
      --split-map "${EVE_ROOT}/splits/episode_splits.jsonl" \
      --config-path "${RUN_DIR}/config.yaml" \
      --code-commit "${code_commit}"

    "${ENV_PREFIX}/bin/python" scripts/everobot/build_eve_sidecar.py append-rollout \
      --base-eve-root "${EVE_ROOT}" \
      --rollout-root "${ROLLOUT_RAW}" \
      --trimmed-event-root "${TRIM_META_ROOT}" \
      --dataset-id fold_glasses_s0_rollout \
      --task-name fold_glasses \
      --source-policy fastwam_s0 \
      --source-checkpoint "${CKPT}" \
      --source-checkpoint-sha256 "${ckpt_sha}" \
      --collection-round 0 \
      --failure-action-loss disabled \
      --require-explicit-outcomes \
      --split-map "${EVE_ROOT}/splits/episode_splits.jsonl" \
      --config-path "${RUN_DIR}/config.yaml" \
      --code-commit "${code_commit}" \
      --annotation-source auto \
      --annotation-method failure_cutoff_success_length_plus_extend \
      --annotation-version b1_video_cfg_success_len_plus3s_v1
  fi

  local b1_manifest="${EVE_ROOT}/manifests/offline_b1_video_cfg.json"
  if [[ ! -f "${b1_manifest}" ]]; then
    "${ENV_PREFIX}/bin/python" scripts/everobot/build_eve_sidecar.py build-manifest \
      --eve-root "${EVE_ROOT}" \
      --manifest-name offline_b1_video_cfg \
      --include-outcomes success failure \
      --success-dataset-ids fold_glasses_expert_success \
      --failure-dataset-ids fold_glasses_s0_rollout \
      --success-sample-mode episode_only \
      --failure-sample-mode event_only \
      --event-types failure_event \
      --failure-action-loss disabled \
      --splits train

    python - "${b1_manifest}" <<'PY'
import json
from pathlib import Path
import sys
from fastwam.everobot_schema import with_manifest_hash

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
changed = 0
for sample in payload.get("samples", []):
    if sample.get("batch_role") != "auxiliary":
        continue
    if sample.get("event_type") != "failure_event":
        continue
    if "window_selection" in sample:
        del sample["window_selection"]
        changed += 1
payload.setdefault("selection", {})
payload["selection"]["b1_video_window_policy"] = "sliding_over_trimmed_failure_event"
payload["selection"]["b1_video_note"] = (
    "Removed core_start_anchor from failure_event auxiliaries so Eve expands "
    "all legal 33-frame windows inside the success-length(+3s) trimmed interval."
)
payload.pop("manifest_hash", None)
fixed = with_manifest_hash(payload)
path.write_text(json.dumps(fixed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[b1-video-cfg] stripped window_selection on {changed} failure_event auxiliaries")
print(f"[b1-video-cfg] manifest_hash={fixed['manifest_hash']}")
PY
  fi

  local val_manifest="${EVE_ROOT}/manifests/offline_selection_primary_success.json"
  if [[ ! -f "${val_manifest}" ]]; then
    "${ENV_PREFIX}/bin/python" scripts/everobot/build_eve_sidecar.py build-manifest \
      --eve-root "${EVE_ROOT}" \
      --manifest-name offline_selection_primary_success \
      --include-outcomes success \
      --success-dataset-ids fold_glasses_expert_success \
      --success-sample-mode episode_only \
      --splits val
  fi

  # B0 budget control: success-only primary + seeded subsample of expert success
  # episodes as auxiliaries matching B1 failure_event count (protocol needs B0).
  local b0_manifest="${EVE_ROOT}/manifests/offline_b0_success_budget_control.json"
  if [[ ! -f "${b0_manifest}" ]]; then
    "${ENV_PREFIX}/bin/python" scripts/everobot/build_eve_sidecar.py build-manifest \
      --eve-root "${EVE_ROOT}" \
      --manifest-name offline_b0_success_budget_control_raw \
      --include-outcomes success \
      --success-dataset-ids fold_glasses_expert_success \
      --success-sample-mode episode_only \
      --splits train

    python - \
      "${EVE_ROOT}/manifests/offline_b0_success_budget_control_raw.json" \
      "${b1_manifest}" \
      "${b0_manifest}" \
      20260807 <<'PY'
import json, random, sys
from pathlib import Path
from fastwam.everobot_schema import with_manifest_hash

raw_path, b1_path, out_path, seed_s = sys.argv[1:]
seed = int(seed_s)
raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
b1 = json.loads(Path(b1_path).read_text(encoding="utf-8"))
b1_primary = [s for s in b1["samples"] if s.get("batch_role") == "primary"]
b1_aux = [s for s in b1["samples"] if s.get("batch_role") == "auxiliary"]
# Use expert success episodes as B0 auxiliaries (mark as success_event-like roles).
pool = [s for s in raw["samples"] if s.get("batch_role") == "primary"]
rng = random.Random(seed)
ordered = sorted(pool, key=lambda s: str(s["sample_id"]))
idxs = list(range(len(ordered)))
rng.shuffle(idxs)
chosen_raw = [ordered[i] for i in sorted(idxs[: len(b1_aux)])]
chosen = []
for s in chosen_raw:
    t = dict(s)
    t["batch_role"] = "auxiliary"
    t["sample_role"] = "success_event"
    t["action_loss"] = "disabled"
    t["sample_id"] = f"{t['sample_id']}__b0_aux"
    chosen.append(t)
samples = sorted(b1_primary + chosen, key=lambda s: str(s["sample_id"]))
payload = {
    "format": raw.get("format", b1.get("format")),
    "schema_version": raw.get("schema_version", b1.get("schema_version")),
    "manifest_name": "offline_b0_success_budget_control",
    "eve_root": b1.get("eve_root"),
    "frame_interval": raw.get("frame_interval", b1.get("frame_interval")),
    "selection": {
        "matched_to": "offline_b1_video_cfg",
        "match_mode": "seeded_subsample_expert_success_for_b1_video_count",
        "seed": seed,
        "reference_auxiliary_count": len(b1_aux),
    },
    # Keep the same dataset_dirs contract as B1 (expert + rollout).
    "dataset_roots": b1.get("dataset_roots", raw.get("dataset_roots")),
    "source_round_ids": raw.get("source_round_ids"),
    "source_hashes": raw.get("source_hashes"),
    "num_samples": len(samples),
    "samples": samples,
}
payload = with_manifest_hash(payload)
Path(out_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[b0] matched auxiliaries={len(chosen)} to b1_aux={len(b1_aux)} total={len(samples)}")
PY
  fi
}

write_protocol_env() {
  local norm_sha
  norm_sha="$(python -c "import hashlib,sys; from pathlib import Path; p=Path(sys.argv[1]); print(hashlib.sha256(p.read_bytes()).hexdigest())" "${NORM_STATS_META_DIR}/stats.json")"
  local env_file="${EVE_ROOT}/protocol/offline_v1_b1_video_cfg.env"
  cat > "${env_file}" <<EOF
# Generated by run_fold_glasses_s0_collect_and_b1_video_cfg.sh
export FITWAM_ENV_PREFIX=${ENV_PREFIX}
export BASE_DATASET=${BASE_DATASET}
export ROLLOUT_RAW=${ROLLOUT_RAW}
export INIT_WEIGHTS=${CKPT}
export SOURCE_CHECKPOINT=${CKPT}
export FASTWAM_SOURCE_CONFIG=${RUN_DIR}/config.yaml
export SOURCE_BUNDLE_MANIFEST=${RUN_DIR}/bundle_manifest.txt
export B1_MANIFEST_PATH=${EVE_ROOT}/manifests/offline_b1_video_cfg.json
export EVE_MANIFEST_PATH=${EVE_ROOT}/manifests/offline_b1_video_cfg.json
export EVE_VAL_MANIFEST_PATH=${EVE_ROOT}/manifests/offline_selection_primary_success.json
export PROTOCOL_BUNDLE_PATH=${EVE_ROOT}/protocol/offline_v1.json
export NORM_STATS_SOURCE=meta
export NORM_STATS_META_DIR=${NORM_STATS_META_DIR}
export NORM_STATS_BUNDLE_SHA256=${norm_sha}
export TEXT_EMBEDDING_CACHE_DIR=${TEXT_CACHE}
export B1_VIDEO_TRIM_META=${TRIM_META_ROOT}/collection_summary.json
export B1_VIDEO_EXPERIMENT_ROOT=${EXP_ROOT}
EOF
  log "wrote ${env_file}"
}

precompute_cfg_text_embeds() {
  log "precomputing text embeds (base + outcome suffixes)"
  set -a
  # shellcheck disable=SC1090
  source "${EVE_ROOT}/protocol/offline_v1_b1_video_cfg.env"
  set +a
  export EVE_MANIFEST_PATH="${B1_MANIFEST_PATH}"
  "${ENV_PREFIX}/bin/python" scripts/precompute_text_embeds.py \
    "task=dexjoco/dexjoco_fold_glasses_offline_b1_video_cfg_2cam_proprio_1e-4" \
    2>&1 | tee -a "${LOG_DIR}/precompute_text_embeds.log"
}

train_b1_video_cfg() {
  log "launching B1-video-cfg training on GPUs ${GPUS}"
  set -a
  # shellcheck disable=SC1090
  source "${EVE_ROOT}/protocol/offline_v1_b1_video_cfg.env"
  set +a
  export CUDA_VISIBLE_DEVICES="${GPUS}"
  export DEWO_TASK=dexjoco/dexjoco_fold_glasses_offline_b1_video_cfg_2cam_proprio_1e-4
  export DEWO_VARIANT=B1-video-cfg
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_B1-video-cfg}"
  export WANDB_RUN_NAME="${WANDB_RUN_NAME:-fold_glasses_offline_B1-video-cfg_${RUN_ID}}"
  export PROTOCOL_BUNDLE_PATH="${EVE_ROOT}/protocol/offline_v1_b1_video_cfg.json"

  # Reuse water_plant formal launcher (B0/B1 protocol freeze + train), with fold_glasses task override.
  bash scripts/dewo/train.sh \
    2>&1 | tee -a "${LOG_DIR}/train_b1_video_cfg.log"
  log "train finished EXIT=$?"
}

# ---------------- main ----------------
log "EXP_ROOT=${EXP_ROOT}"
log "CKPT=${CKPT} RUN_DIR=${RUN_DIR}"
log "protocol=seed ${BASE_SEED} x ${EPISODES_PER_RUN} x ${NUM_REPEATS} repeats; max_env_steps=${MAX_ENV_STEPS}; extend=${EXTEND_SUCCESS_SECONDS}s"

wait_for_ckpt
finalize_s0_bundle
ensure_text_embed_npz
wait_for_gpus

for run_i in $(seq 1 "${NUM_REPEATS}"); do
  collect_one_run "${run_i}"
done

merge_repeats
write_success_len_trim_report
prepare_eve_and_manifests
write_protocol_env
precompute_cfg_text_embeds

# Ensure GPUs still free after collect teardown before train
wait_for_gpus
train_b1_video_cfg

log "ALL_DONE exp=${EXP_ROOT}"
