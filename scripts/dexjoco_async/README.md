# DexJoCo Closed-Loop Eval / Collect

FastWAM DexJoCo 仿真闭环的**唯一入口**。eval 与 collect 共用 async ZMQ 协议（`PolicyServerAsync` + `PolicyClientAsync`）。

DexJoCo 不随本仓库分发。按 [官方仓库](https://github.com/brave-eai/dexjoco) 安装，并将 checkout 放在 `third_party/dexjoco` 或通过命令行传入对应的 task config 路径。

## 文件

| File | Role |
|------|------|
| `run_multi_gpu_dexjoco_eval.py` | 多卡 eval orchestrator（N servers + N sharded clients + 合并报告） |
| `run_multi_gpu_dexjoco_collect.py` | 多卡 rollout 采集 orchestrator（写 LeRobot dataset） |
| `eval_dexjoco_fastwam_control.py` | 闭环 eval client（blocking / overlap 控制、LPF、action metrics） |
| `dexjoco_fastwam_adapter.py` | Sim ↔ policy observation 适配 |
| `multi_gpu_eval_utils.py` | 分片、端口分配、server ping、conda 子进程启动 |
| `eval_summary_aggregator.py` | 合并各 shard 的 `summary.json` |
| `../run_fastwam_server_async.py` | Async policy server launcher |
| `../collect_dexjoco_rollouts.py` | 通用 rollout collect client |
| `../build_rollout_datasets.py` | 通用 rollout dataset merge / trim |

## 多卡 eval 示例

```bash
cd /path/to/FITWAM
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fastwam

python scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
  --gpus 0,1,2,3 \
  --run-dir runs/water_plant_uncond_2cam_384_1e-4/2026-06-29_16-38-39 \
  --checkpoint runs/.../checkpoints/weights/step_006500.pt \
  --no-load-text-encoder \
  --task-config-dir third_party/dexjoco/configs/rand_obj \
  --tasks water_plant \
  --episodes 100 --seed 0 \
  --replan-steps 24 --control-mode blocking \
  --max-env-steps 1500 \
  --output-dir evaluate_results/dexjoco/water_plant/step_006500
```

## 输出目录

```

## 多卡 rollout collect 示例

```bash
python scripts/dexjoco_async/run_multi_gpu_dexjoco_collect.py \
  --gpus 0,1,2,3 \
  --run-dir runs/hammer_nail_uncond_2cam_384_1e-4/2026-07-01_10-04-05 \
  --checkpoint runs/hammer_nail_uncond_2cam_384_1e-4/2026-07-01_10-04-05/checkpoints/weights/step_006500.pt \
  --no-load-text-encoder \
  --tasks hammer_nail \
  --episodes 200 \
  --replan-steps 24 \
  --max-env-steps 600 \
  --output-dir logs/hammer_nail_rollout_200_step6500_trim8s/collect \
  --raw-output-dataset data/hammer_nail_rollout_200_step6500_raw \
  --trimmed-output-dataset data/hammer_nail_rollout_200_step6500_trim8s \
  --trim-failure-seconds 8
```
output-dir/
  summary.json
  summary.csv
  video_manifest.csv
  shard_0/ summary.json, server.log, client.log, water_plant/episode_*.mp4
  shard_1/ ...
```

## 常用选项

- `--control-mode overlap` — 边执行边提交下一段 policy 请求（默认 README 用 `blocking`）
- `--no-launch-servers` + `--ports` — 复用已启动的 server
- `--server-conda-env fastwam` / `--client-conda-env dexjoco` — 覆盖 conda 环境
