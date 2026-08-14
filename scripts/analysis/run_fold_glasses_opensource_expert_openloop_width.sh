#!/usr/bin/env bash
# fold_glasses expert open-loop action width — OPENSOURCE-aligned inference.
#
# Must match scripts/fold_glasses/eval_opensource_4x50_gpus4567.sh:
#   FastWAM pin 45d8e14
#   FastWAMDexJocoPolicy + configs/fastwam_dexjoco.yaml (224, z-score)
#   artifacts/fold_glasses/{dataset_stats.json, text embedding}
#   action_horizon=32, num_inference_steps=10
#
# Pack GPUs 4-7: default 5 workers/GPU (~13.5GB each → ~67GB/GPU on A100-80G).
set -euo pipefail

ROOT=/data_all/xiangchengzhan/FastWAM
OPEN="${FASTWAM_OPEN_REPO:-/data_all/xiangchengzhan/FastWAM-infer-in-DexJoco}"
FASTWAM_PIN="${FASTWAM_PIN:-${ROOT}/third_party/FastWAM_pin_45d8e14}"
cd "${ROOT}"

source /home/xiangchengzhan/anaconda3/etc/profile.d/conda.sh
conda activate fastwam

GPUS="${GPUS:-4,5,6,7}"
# ~13.5GB/worker after load+infer; 5× ≈ 67GB leaves headroom on 80GB.
WORKERS_PER_GPU="${WORKERS_PER_GPU:-5}"
OUT="${OUT:-${ROOT}/results/fold_glasses_opensource_expert_openloop_width_K20_stride5_20260810}"
CKPT="${CKPT:-${OPEN}/checkpoints/fold_glasses/step_010000.pt}"
MODEL_CFG="${MODEL_CFG:-${OPEN}/configs/fastwam_dexjoco.yaml}"
STATS="${STATS:-${OPEN}/artifacts/fold_glasses/dataset_stats.json}"
TEXT="${TEXT:-${OPEN}/artifacts/fold_glasses/0c3367ce1d74848cc46b93c6d2eee5e2097dca410a2c95f3da48bd8c8673fa20.t5_len128.wan22ti2v5b.pt}"
mkdir -p "${OUT}/logs"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT}/checkpoints}"
export FASTWAM_OPEN_REPO="${OPEN}"
export FASTWAM_PIN="${FASTWAM_PIN}"
# Prefer OPEN + pin over workspace src (same as eval_opensource).
export PYTHONPATH="${OPEN}/src:${FASTWAM_PIN}/src:${ROOT}/third_party/dexjoco/dexjoco:${PYTHONPATH:-}"

pin_head="$(git -C "${FASTWAM_PIN}" rev-parse HEAD)"
echo "[launch $(date -Is)] FastWAM_PIN HEAD=${pin_head}"
if [[ "${pin_head}" != 45d8e1458921d83f8ad6cf9ce993d371208dabd0 ]]; then
  echo "ERROR: FastWAM pin mismatch (need 45d8e14)"
  exit 1
fi
for p in "${CKPT}" "${MODEL_CFG}" "${STATS}" "${TEXT}"; do
  [[ -e "$p" ]] || { echo "ERROR missing $p"; exit 1; }
done

echo "[launch $(date -Is)] GPUS=${GPUS} WPG=${WORKERS_PER_GPU} OUT=${OUT}"
echo "[launch] inference=opensource FastWAMDexJocoPolicy image=224 z-score"
exec python scripts/analysis/run_fold_glasses_opensource_expert_openloop_width.py \
  --mode orchestrate \
  --gpus "${GPUS}" \
  --workers-per-gpu "${WORKERS_PER_GPU}" \
  --checkpoint "${CKPT}" \
  --model-config "${MODEL_CFG}" \
  --dataset-stats "${STATS}" \
  --text-embedding "${TEXT}" \
  --expert-dataset "${ROOT}/data/fold_glasses_fastwam" \
  --num-samples 20 \
  --stride 5 \
  --num-inference-steps 10 \
  --action-horizon 32 \
  --replan-steps 24 \
  --sample-seed0 20260808 \
  --output "${OUT}" \
  "$@"
