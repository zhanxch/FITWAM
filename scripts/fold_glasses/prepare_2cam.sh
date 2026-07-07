#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/data/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data_all/xiangchengzhan/.cache/huggingface/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_DATASETS_CACHE}"

SOURCE_ROOT="${SOURCE_ROOT:-data/dexjoco/dexjoco_lerobot_datasets/fold_glasses}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/fold_glasses_fastwam}"
TASK=fold_glasses_uncond_2cam_384_1e-4
ACTION_DIT=checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt

echo "[prepare_2cam] source=${SOURCE_ROOT} output=${OUTPUT_ROOT}"
python scripts/water_plant/fix_lerobot_parquet_metadata.py \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${OUTPUT_ROOT}"

echo "[prepare_2cam] task=${TASK}"

if [[ ! -f "${ACTION_DIT}" ]]; then
  echo "[prepare_2cam] generating ActionDiT backbone -> ${ACTION_DIT}"
  python scripts/preprocess_action_dit_backbone.py \
    --model-config configs/model/fastwam.yaml \
    --output "${ACTION_DIT}" \
    --device cuda \
    --dtype bfloat16
else
  echo "[prepare_2cam] ActionDiT backbone already exists: ${ACTION_DIT}"
fi

echo "[prepare_2cam] precomputing T5 text embeddings"
python scripts/precompute_text_embeds.py "task=${TASK}"

echo "[prepare_2cam] validating dataset sample load"
python - <<'PY'
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()
config_dir = str(Path("configs").resolve())

with initialize_config_dir(version_base="1.3", config_dir=config_dir):
    cfg = compose(config_name="train", overrides=["task=fold_glasses_uncond_2cam_384_1e-4"])

train_ds = instantiate(cfg.data.train)
sample = train_ds[0]
assert "video" in sample
assert "action" in sample
assert "proprio" in sample
assert sample["proprio"].shape[-1] == 23
assert sample["video"].shape[-1] == 768
print(
    f"[prepare_2cam] ok: video={tuple(sample['video'].shape)} "
    f"action={tuple(sample['action'].shape)} proprio={tuple(sample['proprio'].shape)}"
)
PY

echo "[prepare_2cam] done. Run: CUDA_VISIBLE_DEVICES=0,1,2,3,4 bash scripts/fold_glasses/train_2cam.sh"
