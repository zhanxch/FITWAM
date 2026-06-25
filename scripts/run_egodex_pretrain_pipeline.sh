#!/usr/bin/env bash
# Resize egodex videos to 384x384, reuse text-embed cache, then launch pretraining.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE_DATA="${SOURCE_DATA:-data/egodex_part2_basic_pnp_fastwam_video_pretrain}"
OUTPUT_DATA="${OUTPUT_DATA:-data/egodex_part2_basic_pnp_fastwam_video_pretrain_384}"
SOURCE_CACHE="${SOURCE_CACHE:-data/text_embeds_cache/egodex_part2_basic_pnp_fastwam_video_pretrain}"
OUTPUT_CACHE="${OUTPUT_CACHE:-data/text_embeds_cache/egodex_part2_basic_pnp_fastwam_video_pretrain_384}"
TASK="${TASK:-egodex_part2_basic_pnp_fastwam_video_pretrain_uncond_1cam_384_1e-4}"
RESIZE_WORKERS="${RESIZE_WORKERS:-16}"
NGPU="${NGPU:-4}"
OVERWRITE="${OVERWRITE:-0}"

RESIZE_ARGS=(--source-root "$SOURCE_DATA" --output-root "$OUTPUT_DATA" --size 384 --workers "$RESIZE_WORKERS")
if [[ "$OVERWRITE" == "1" ]]; then
  RESIZE_ARGS+=(--overwrite)
fi

echo "[1/3] Resize videos -> ${OUTPUT_DATA}"
if [[ -d "$OUTPUT_DATA/meta" && "$OVERWRITE" != "1" ]]; then
  echo "Output dataset already exists, skipping resize. Set OVERWRITE=1 to rebuild."
else
  python scripts/resize_lerobot_dataset_videos.py "${RESIZE_ARGS[@]}"
fi

echo "[2/3] Reuse text embedding cache -> ${OUTPUT_CACHE}"
if [[ ! -d "$OUTPUT_CACHE" ]]; then
  if [[ -d "$SOURCE_CACHE" ]]; then
    mkdir -p "$(dirname "$OUTPUT_CACHE")"
    cp -al "$SOURCE_CACHE" "$OUTPUT_CACHE" 2>/dev/null || cp -r "$SOURCE_CACHE" "$OUTPUT_CACHE"
  else
    echo "Text cache not found at ${SOURCE_CACHE}; precomputing..."
    if [[ "$NGPU" -gt 1 ]]; then
      torchrun --standalone --nproc_per_node="$NGPU" \
        scripts/precompute_text_embeds.py "task=${TASK}"
    else
      python scripts/precompute_text_embeds.py "task=${TASK}"
    fi
  fi
else
  echo "Text cache already exists, skipping."
fi

echo "[3/3] Launch training on ${NGPU} GPU(s)"
bash scripts/train_zero1.sh "$NGPU" "task=${TASK}"
