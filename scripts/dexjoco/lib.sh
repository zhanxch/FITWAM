# Shared helpers for open-source DexJoCo 4×50 eval (FastWAM-infer-in-DexJoco stack).
# Source after setting ROOT:
#   ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
#   source "${ROOT}/scripts/dexjoco/lib.sh"

EXPECTED_FASTWAM_PIN="45d8e1458921d83f8ad6cf9ce993d371208dabd0"
EXPECTED_DEXJOCO_PIN="8d23b0fab23b17a58c4b55f3942e17013aaf8267"

dexjoco_opensource_setup() {
  local root="${ROOT:?Set ROOT to the FastWAM repo root}"
  OPEN_REPO="${OPEN_REPO:-${root}/../FastWAM-infer-in-DexJoco}"
  OPEN_REPO="$(cd "${OPEN_REPO}" && pwd)"
  FASTWAM_PIN="${FASTWAM_PIN:-${root}/third_party/FastWAM_pin_45d8e14}"
  DEXJOCO_ROOT="${DEXJOCO_ROOT:-${root}/third_party/dexjoco}"
  export OPEN_REPO FASTWAM_PIN DEXJOCO_ROOT
}

dexjoco_require_gpus() {
  if [[ -z "${GPUS:-}" ]]; then
    echo "[opensource-eval] ERROR: set GPUS (comma-separated physical ids), e.g. GPUS=4,5,6,7" >&2
    return 2
  fi
}

dexjoco_activate_fastwam() {
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${FITWAM_ENV:-fastwam}"
  ENV_PREFIX="${FITWAM_ENV_PREFIX:-$(conda info --base)/envs/${FITWAM_ENV:-fastwam}}"
  export PATH="${ENV_PREFIX}/bin:${PATH}"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export TOKENIZERS_PARALLELISM=false
  export DIFFSYNTH_SKIP_DOWNLOAD=true
  export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT}/checkpoints}"
}

dexjoco_export_pythonpath() {
  export PYTHONPATH="${OPEN_REPO}/src:${FASTWAM_PIN}/src:${DEXJOCO_ROOT}/dexjoco:${PYTHONPATH:-}"
}

dexjoco_assert_pins() {
  local pin_head dex_head
  pin_head="$(git -C "${FASTWAM_PIN}" rev-parse HEAD)"
  dex_head="$(git -C "${DEXJOCO_ROOT}" rev-parse HEAD)"
  echo "[opensource-eval] FastWAM_PIN=${FASTWAM_PIN} HEAD=${pin_head}"
  echo "[opensource-eval] DexJoco HEAD=${dex_head}"
  [[ "${pin_head}" == "${EXPECTED_FASTWAM_PIN}" ]] || {
    echo "[opensource-eval] ERROR: FastWAM pin mismatch (expected ${EXPECTED_FASTWAM_PIN})" >&2
    return 1
  }
  [[ "${dex_head}" == "${EXPECTED_DEXJOCO_PIN}" ]] || {
    echo "[opensource-eval] ERROR: DexJoco pin mismatch (expected ${EXPECTED_DEXJOCO_PIN})" >&2
    return 1
  }
}

dexjoco_find_t5() {
  local task="$1"
  local emb
  emb="$(ls "${OPEN_REPO}/artifacts/${task}/"*.t5_len128*.pt 2>/dev/null | head -1 || true)"
  if [[ -z "${emb}" || ! -f "${emb}" ]]; then
    echo "[opensource-eval] ERROR: missing T5 embedding under ${OPEN_REPO}/artifacts/${task}/" >&2
    return 2
  fi
  printf '%s' "${emb}"
}

dexjoco_gpus_idle() {
  local max_used="${MAX_USED_MIB:-1500}"
  local max_util="${MAX_UTIL:-10}"
  python - "${GPUS}" "${max_used}" "${max_util}" <<'PY'
import subprocess, sys
gpus = [int(x) for x in sys.argv[1].split(",") if x.strip()]
max_used = int(sys.argv[2])
max_util = int(sys.argv[3])
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
    text=True,
)
rows = {}
for line in out.strip().splitlines():
    idx, mem, util = [p.strip() for p in line.split(",")]
    rows[int(idx)] = (int(float(mem)), int(float(util)))
ok = True
for g in gpus:
    mem, util = rows.get(g, (10**9, 100))
    print(f"gpu{g}: mem={mem}MiB util={util}%")
    if mem > max_used or (max_util > 0 and util > max_util):
        ok = False
raise SystemExit(0 if ok else 1)
PY
}

dexjoco_wait_gpus_idle() {
  local log_file="${1:-/dev/null}"
  local sleep_s="${WAIT_POLL_SEC:-60}"
  local streak_need="${WAIT_IDLE_STREAK:-2}"
  local idle_streak=0
  echo "[opensource-eval $(date -Is)] waiting for GPUs ${GPUS} idle" | tee -a "${log_file}"
  while true; do
    if dexjoco_gpus_idle >>"${log_file}" 2>&1; then
      idle_streak=$((idle_streak + 1))
      echo "[opensource-eval $(date -Is)] idle streak ${idle_streak}/${streak_need}" | tee -a "${log_file}"
      [[ "${idle_streak}" -ge "${streak_need}" ]] && break
    else
      idle_streak=0
    fi
    sleep "${sleep_s}"
  done
}
