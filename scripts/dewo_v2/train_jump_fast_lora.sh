#!/usr/bin/env bash
# Removed. LoRA is not a DEWO v2 recipe.
echo "[dewo-v2] ERROR: scripts/dewo_v2/train_jump_fast_lora.sh is removed." >&2
echo "  There is no LoRA training path. Use full DiT:" >&2
echo "    TASK=... INIT=scratch|s0 ENV_FILE=... GPUS=... bash scripts/dewo_v2/train.sh" >&2
exit 2
