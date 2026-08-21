TASK=hammer_nail GPUS=0,1,2,3 \
  ENV_FILE=data/hammer_nail_dewo_v2_pair_20260817_193358/eve_v02/protocol/offline_v1_b1_jump_fast.env \
  bash scripts/dewo_v2/train.sh
# 可选：LR=1e-4 MAX_STEPS=15000 BATCH_SIZE=16