# Scripts

**Research direction:** [`README.md`](../README.md) · **Upstream setup:** [`docs/FASTWAM_UPSTREAM.md`](../docs/FASTWAM_UPSTREAM.md)

## 目录结构

```text
scripts/
├── train.py, train_zero1.sh, train_zero2.sh   # 通用 Hydra 训练入口
├── precompute_text_embeds.py                   # T5 embedding 预计算
├── export_text_embed_cache_npz.py              # .pt text cache → .npz（DexJoCo client 用）
├── run_fastwam_server_async.py                 # DexJoCo 闭环 policy server（async ZMQ）
├── policy_client_async.py, fastwam_policy_server_async.py, policy_msgpack.py
├── dexjoco_async/                              # DexJoCo 多卡闭环 eval / collect orchestrator
├── collect_dexjoco_rollouts.py                # 通用 DexJoCo rollout collector
├── build_rollout_datasets.py                  # 通用 rollout shard merge / failure trim
├── collect_rollout_trim_and_train.sh          # 通用 collect + trim + train wrapper
├── water_plant/                                # water_plant 数据准备与薄 wrapper
├── hammer_nail/                                # hammer_nail 数据准备与薄 wrapper
├── openloop/                                   # 开环 eval 引擎（多数据集共用）
├── spray_water_gr00tstyle/                     # 真机 spray_water GR00T-style
├── diagnose/                                   # 本地诊断（gitignore，不上传）
├── accelerate_configs/  ds_configs/
└── archive/  (repo 外)                         # 历史实验
```

## 按数据集

| 数据集 | 目录 | 说明 |
|--------|------|------|
| **water_plant** | [`water_plant/`](water_plant/) | 双视角 + proprio 23d；LeRobot 窗口 / EveRobot event sidecar；DexJoCo 闭环 |
| **spray_water_gr00tstyle** | [`spray_water_gr00tstyle/`](spray_water_gr00tstyle/) | 真机 3cam rot6d；训练、开环 eval、Wuji deploy |

## 通用训练流程

```bash
python scripts/precompute_text_embeds.py task=<task_name>
bash scripts/train_zero1.sh 4 task=<task_name>
```

## DexJoCo 闭环测试（唯一协议）

所有 DexJoCo 仿真闭环 **eval** 与 **rollout collect** 共用一套 async ZMQ 协议：

| 组件 | 文件 |
|------|------|
| Policy server | `run_fastwam_server_async.py` → `PolicyServerAsync` (ROUTER/DEALER) |
| Eval client | `dexjoco_async/eval_dexjoco_fastwam_control.py` → `policy_client_async.PolicyClientAsync` |
| Collect client | `collect_dexjoco_rollouts.py` → `policy_client_async.PolicyClientAsync` |
| Sim adapter | `dexjoco_async/dexjoco_fastwam_adapter.py` |

### 多卡闭环 eval（与根 README 一致）

```bash
python scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
  --gpus 0,1,2,3 \
  --run-dir runs/<task_name>/<run_timestamp> \
  --checkpoint runs/<task_name>/<run_timestamp>/checkpoints/weights/step_006500.pt \
  --no-load-text-encoder \
  --task-config-dir third_party/dexjoco/configs/rand_obj \
  --tasks water_plant \
  --episodes 100 --seed 0 \
  --replan-steps 24 --control-mode blocking \
  --max-env-steps 1500 \
  --output-dir evaluate_results/dexjoco/<task_name>/step_006500
```

### 多卡 rollout 采集

```bash
python scripts/dexjoco_async/run_multi_gpu_dexjoco_collect.py \
  --gpus 0,1,2,3 \
  --run-dir runs/<task_name>/<run_timestamp> \
  --checkpoint runs/<task_name>/<run_timestamp>/checkpoints/weights/step_006500.pt \
  --no-load-text-encoder \
  --tasks water_plant \
  --episodes 200 \
  --replan-steps 24 \
  --output-dir logs/dexjoco_collect/<task_name>/step_006500 \
  --raw-output-dataset data/dexjoco/rollouts/<task_name>_raw \
  --trimmed-output-dataset data/dexjoco/rollouts/<task_name>_trimmed \
  --trim-failure-seconds 8
```

water_plant 一键 collect + trim + train：`bash scripts/water_plant/collect_rollout_200_trim8s_and_train.sh`
hammer_nail 一键 collect + trim + train：`bash scripts/hammer_nail/collect_rollout_200_trim8s_and_train.sh`

详见 [`dexjoco_async/README.md`](dexjoco_async/README.md)。

## Video LoRA

```bash
bash scripts/water_plant/train_2cam.sh task=water_plant_uncond_2cam_384_1e-4_lora
```

详见根目录 [`README.md`](../README.md#video-lora独立可选路径)。

## 其他

| 位置 | 说明 |
|------|------|
| `scripts/diagnose/` | 本地 sim-vs-real 诊断，gitignore |
| `scripts/run_fastwam_server.py` | 真机 / 调试用 sync server（非 DexJoCo 闭环主线） |
| `openloop/run_robotwin_openloop*.py` | 已弃用 shim，请用 `run_openloop.py` |
