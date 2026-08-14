#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DEWO_TASK="${DEWO_TASK:-}"
: "${DEWO_TASK:?Set DEWO_TASK to a Hydra task config, for example dexjoco/dexjoco_fold_glasses_offline_b1_jump_fast_lora_3e-5}"
: "${EVE_MANIFEST_PATH:?Set EVE_MANIFEST_PATH to the DEWO training manifest}"
: "${EVE_VAL_MANIFEST_PATH:?Set EVE_VAL_MANIFEST_PATH to the validation manifest}"
: "${INIT_WEIGHTS:?Set INIT_WEIGHTS to the initialization checkpoint}"
: "${TEXT_EMBEDDING_CACHE_DIR:?Set TEXT_EMBEDDING_CACHE_DIR}"

for required_file in "${EVE_MANIFEST_PATH}" "${EVE_VAL_MANIFEST_PATH}" "${INIT_WEIGHTS}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required DEWO input: ${required_file}" >&2
    exit 2
  fi
done
if [[ ! -d "${TEXT_EMBEDDING_CACHE_DIR}" ]]; then
  echo "Missing DEWO text cache directory: ${TEXT_EMBEDDING_CACHE_DIR}" >&2
  exit 2
fi

if [[ -n "${FITWAM_ENV_PREFIX:-}" ]]; then
  export PATH="${FITWAM_ENV_PREFIX}/bin:${PATH}"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${FITWAM_ENV:-fastwam}"
fi
if ! command -v accelerate >/dev/null 2>&1; then
  echo "accelerate is unavailable; activate the FastWAM training environment." >&2
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a DEWO_GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${#DEWO_GPU_ARRAY[@]}"
if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain at least one GPU." >&2
  exit 2
fi

hash_file() {
  python - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser().resolve()
digest = hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

hash_text_cache() {
  python - "$1" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
digest = hashlib.sha256()
files = sorted(path for path in root.rglob("*") if path.is_file())
if not files:
    raise SystemExit(f"Text cache is empty: {root}")
for path in files:
    digest.update(path.relative_to(root).as_posix().encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(path.stat().st_size).encode("ascii"))
print(digest.hexdigest())
PY
}

hash_code_snapshot() {
  python - "${ROOT_DIR}" <<'PY'
import hashlib
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
result = subprocess.run(
    [
        "git", "ls-files", "--cached", "--others", "--exclude-standard", "--",
        "src", "scripts", "configs", "tests", "DEWO.md", "README.md",
        "docs/RELATED_WORK.md",
    ],
    cwd=root,
    check=True,
    capture_output=True,
    text=True,
)
digest = hashlib.sha256()
for rel in sorted(line for line in result.stdout.splitlines() if line):
    path = root / rel
    if not path.is_file():
        continue
    digest.update(rel.encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
print(digest.hexdigest())
PY
}

export FASTWAM_RESUME="${RESUME_STATE_DIR:-${INIT_WEIGHTS}}"
export FASTWAM_RESUME_SHA256="$(hash_file "${INIT_WEIGHTS}")"
export EVE_MANIFEST_SHA256="$(hash_file "${EVE_MANIFEST_PATH}")"
export EVE_VAL_MANIFEST_SHA256="$(hash_file "${EVE_VAL_MANIFEST_PATH}")"
export TEXT_EMBEDDING_CACHE_SHA256="${TEXT_EMBEDDING_CACHE_SHA256:-$(hash_text_cache "${TEXT_EMBEDDING_CACHE_DIR}")}"
export FASTWAM_SOURCE_CONFIG_SHA256="${FASTWAM_SOURCE_CONFIG_SHA256:-unknown}"
if [[ -n "${FASTWAM_SOURCE_CONFIG:-}" && -f "${FASTWAM_SOURCE_CONFIG}" ]]; then
  export FASTWAM_SOURCE_CONFIG_SHA256="$(hash_file "${FASTWAM_SOURCE_CONFIG}")"
fi
export FITWAM_CODE_SNAPSHOT_SHA256="${FITWAM_CODE_SNAPSHOT_SHA256:-$(hash_code_snapshot)}"
export FITWAM_VARIANT="${DEWO_VARIANT:-${FITWAM_PROVENANCE_VARIANT:-dewo}}"
export NORM_STATS_SOURCE="${NORM_STATS_SOURCE:-configured}"
export NORM_STATS_BUNDLE_SHA256="${NORM_STATS_BUNDLE_SHA256:-unknown}"
export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)_${FITWAM_VARIANT}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-dewo_${RUN_ID}}"

DEWO_OUTPUT_DIR="${DEWO_OUTPUT_DIR:-${FITWAM_OUTPUT_NAMESPACE:-./runs/dewo}}"
echo "[dewo] task=${DEWO_TASK} gpus=${CUDA_VISIBLE_DEVICES} run_id=${RUN_ID}"
echo "[dewo] train_manifest=${EVE_MANIFEST_PATH}"
echo "[dewo] init=${FASTWAM_RESUME}"

bash scripts/train_zero1.sh "${NPROC_PER_NODE}" \
  "task=${DEWO_TASK}" \
  "output_dir=${DEWO_OUTPUT_DIR}/${RUN_ID}" \
  "wandb.name=${WANDB_RUN_NAME}" \
  "resume=${FASTWAM_RESUME}" \
  "$@"
