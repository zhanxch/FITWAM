# DexJoCo Async Inference

Historical DexJoCo-only async/LPF evaluation utilities for FastWAM.

These scripts evaluate a trained FastWAM policy in DexJoCo with two control modes:

- `blocking`: wait for a policy chunk before executing the next segment.
- `overlap`: submit the next policy request while executing the current chunk.

Main files:

| File | Role |
|------|------|
| `run_fastwam_server_async.py` | Starts an async ZMQ FastWAM policy server. |
| `fastwam_policy_server_async.py` | ROUTER/DEALER ZMQ server/client implementation. |
| `eval_dexjoco_fastwam_control.py` | DexJoCo closed-loop evaluator with overlap control and optional LPF. |
| `run_dexjoco_async_lpf_eval_clients.sh` | Multi-condition evaluation launcher. |
| `summarize_dexjoco_async_ablation.py` | Aggregates summary and video manifests. |
| `run_multi_gpu_dexjoco_eval.py` | One-command multi-GPU parallel eval orchestrator (N servers + N sharded clients + merged report). |
| `multi_gpu_eval_utils.py` | Reusable utils: episode sharding, free-port allocation, server ping, conda-env subprocess launch. |
| `eval_summary_aggregator.py` | Reusable merger for per-shard `summary.json` files (library + CLI). |

The committed result subset is in [`results/dexjoco_async_microwave`](../../results/dexjoco_async_microwave).

## Multi-GPU parallel evaluation

`run_multi_gpu_dexjoco_eval.py` runs one policy server per GPU and one eval
client per shard in parallel, splits the total episode budget across them, then
merges the per-shard summaries into a single report. It mirrors the manual
two-terminal workflow (server in the `fastwam` env, client in the `dexjoco`
env) but automates conda activation, port allocation, readiness checks, and
cleanup.

Architecture (decoupled for reuse):

- `multi_gpu_eval_utils.py` — sharding, ports, ping, conda subprocess launch.
  No `torch`/`mujoco` imports, usable from either conda env.
- `eval_summary_aggregator.py` — merges shard `summary.json` files, recomputes
  per-task metric means from raw per-episode metrics, re-indexes episodes
  globally, and emits the same `summary.json` / `summary.csv` /
  `video_manifest.csv` triple a single eval produces. Usable as a library or
  CLI, with no `torch`/`mujoco` dependency.
- `run_multi_gpu_dexjoco_eval.py` — glue only: spawns servers, waits for pings,
  spawns sharded clients, injects shard provenance into each shard summary,
  calls the aggregator, and tears servers down on exit / Ctrl-C. The underlying
  server/eval scripts are not modified.

Example — 4 GPUs, 100 episodes of `water_plant` (25 per shard, seeds 0–99 split
contiguously across shards):

```bash
cd /data_all/xiangchengzhan/FastWAM
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam   # orchestrator only needs zmq + msgpack + conda on PATH

python scripts/water_plant/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
  --gpus 4,5,6,7 \
  --run-dir runs/water_plant_uncond_2cam_384_1e-4/2026-06-29_16-38-39 \
  --checkpoint runs/water_plant_uncond_2cam_384_1e-4/2026-06-29_16-38-39/checkpoints/weights/step_006500.pt \
  --no-load-text-encoder \
  --task-config-dir third_party/dexjoco/configs/rand_obj \
  --tasks water_plant \
  --episodes 100 --seed 0 \
  --replan-steps 24 --control-mode blocking \
  --max-env-steps 1500 \
  --output-dir evaluate_results/dexjoco/water_plant/step_006500
```

Output layout (combined report at the top level, per-shard artifacts below):

```
output-dir/
  summary.json            # combined: overall rate, per-task means, shards[] provenance
  summary.csv             # combined flat table
  video_manifest.csv      # all episodes across shards (with shard column)
  shard_0/ summary.json, server.log, client.log, water_plant/episode_*.mp4
  shard_1/ ...
  shard_2/ ...
  shard_3/ ...
```

Useful options:

- `--ports 5570,5571,5572,5573` — pin ports instead of auto-allocating free ones.
- `--no-launch-servers` — reuse servers you already started yourself (the
  orchestrator only pings, runs sharded clients, and aggregates).
- `--server-conda-env` / `--client-conda-env` — override the conda envs
  (defaults `fastwam` / `dexjoco`).
- `--episodes 103` — remainder episodes go to the first shards (26/26/26/25).
- All eval pass-through flags (`--control-mode`, `--low-pass-alpha`,
  `--action-clip`, `--save-video/--no-save-video`, ...) are forwarded to every
  shard client.

Aggregator CLI (reusable on its own, e.g. when shards were run manually):

```bash
python scripts/water_plant/dexjoco_async/eval_summary_aggregator.py \
  shard_0/summary.json shard_1/summary.json shard_2/summary.json shard_3/summary.json \
  --output-dir evaluate_results/dexjoco/water_plant/step_006500 \
  --label blocking_stride24_gpus4
```
