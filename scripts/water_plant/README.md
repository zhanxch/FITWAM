# water_plant（DexJoCo 仿真）

DexJoCo `water_plant` 任务：**双视角（front + wrist）+ proprio（23 维）**。

## 数据

| 路径 | 说明 |
|------|------|
| `data/water_plant` | 原始 LeRobot 数据 |
| `data/water_plant_fastwam` | 经 `fix_lerobot_parquet_metadata.py` 处理后的训练副本 |

## 训练

**LeRobot 固定窗口（baseline / baseline_lora）：**

```bash
bash scripts/water_plant/prepare_2cam.sh
bash scripts/water_plant/train_2cam.sh
bash scripts/water_plant/train_2cam.sh task=water_plant_uncond_2cam_384_1e-4_lora
```

**EveRobot sidecar v0.2（failure self-evolution）：**

EveRobot 是一个 **LeRobot-compatible sidecar**，用于失败轨迹自进化训练。
它不改动 LeRobot 原始 `data/`、`videos/`、`meta/`，只在
`data/water_plant_fastwam/eve/` 下维护 episode provenance、failure event
window 和训练 manifest。训练时由 manifest 决定使用哪些数据子集，以及 failure
episode 只采样哪些 event window。

```bash
bash scripts/water_plant/build_eve_round1_sidecar.sh
bash scripts/water_plant/train_eve_round1.sh
```

已有 v0.1 sidecar 时应指定新的目录，例如
`EVE_ROOT=data/water_plant_fastwam/eve_v02 bash scripts/water_plant/build_eve_round1_sidecar.sh`；loader 仍可读取旧 manifest。

默认构造：

| 来源 | 用途 |
|------|------|
| `data/water_plant_fastwam` | round0 success training data |
| `data/water_plant_rollout_200_step6500_raw` | round0 policy rollout provenance |
| `data/water_plant_rollout_200_step6500_trim8s` | failure event window 标注来源 |

生成的 sidecar：

| 文件 | 说明 |
|------|------|
| `eve/schema_version.json` | EveRobot 格式版本和兼容说明 |
| `eve/round_meta.jsonl` | 不可变 round provenance |
| `eve/episode_meta.jsonl` | episode 级 metadata：来源、round、policy、success/failure、seed、length |
| `eve/event_meta.jsonl` | event 级 metadata：failure window、failure type、标注来源和 action loss 策略 |
| `eve/manifests/train_round1_success_plus_failure_events.json` | round1 训练子集：100 条 success episode + 49 个 failure event |
| `eve/reports/*.json` | 构造统计报告 |

当前 round1 sidecar 的语义：

| 项 | 数量 / 策略 |
|----|-------------|
| base success episode | 100 |
| rollout success episode | 151，仅记录 provenance，默认不进入 round1 manifest |
| rollout failure episode | 49 |
| failure event | 49 |
| 600 帧 timeout failure | 使用 `trim8s` 标注出的 `[0, 360)` window |
| 短 failure | 使用完整 failure episode window |
| failure action loss | `disabled`，即 failure event 用于 video/proprio/context 学习，不模仿失败动作 |
| steer token | 模型参数，不写入 EveRobot metadata |

训练适配由 `fastwam.datasets.eve.manifest_dataset.EveManifestRobotVideoDataset`
完成。它复用现有 FastWAM 的视频解码、processor、text embedding cache 和 loss
接口，只改变采样范围：

- success episode：按普通 LeRobot 固定窗口采样。
- failure event：只在 `event_meta.start_frame <= window < end_frame` 内采样。
- `action_loss=disabled`：返回 `action_loss_weight=0.0`。
- prompt 会去掉 `"Failed to finish the whole process."` 后缀，避免用文本短语隐式控制 loss。

如果后续有 round2 rollout，不需要替换旧数据；继续 `append-rollout` 新 dataset，
再生成新的 `eve/manifests/train_round2_*.json` 即可。

**Legacy EveRobot 整 episode（DiffSynth 风格）：**

```bash
python scripts/water_plant/convert_lerobot_to_everobot.py \
  --dataset-dir data/water_plant_fastwam --video-keys front wrist
bash scripts/water_plant/train_everobot.sh task=everobot_water_plant_full_lora
```

Task 配置：`configs/task/water_plant_uncond_2cam_384_1e-4*.yaml`；数据配置 `configs/data/water_plant_2cam.yaml`（`proprio_output_dim: 23`）。

## 当前结果

`rand-obj` 设置下，已有 water_plant 闭环成功率：

| Method | Success Rate | Successes / Trials | Notes |
|--------|-------------:|-------------------:|-------|
| DP-T | 84.0±3.5 | -- | external baseline |
| DP-C | 63.3±3.1 | -- | external baseline |
| ACT | 47.3±4.6 | -- | external baseline |
| π₀.₅ | 88.7±3.1 | -- | external baseline |
| GR00T N1.5 | 72.7±1.2 | -- | external baseline |
| FastWAM | 75.5% | 151 / 200 | success-only FastWAM rollout |
| FastWAM + failure scratch | 82.0% | 82 / 100 | failure data from scratch |
| FastWAM + LoRA continuation | 81.5% | 163 / 200 | continued training with rollout data |

这里的 FastWAM `75.5% (151/200)` 也是 EveRobot round1 sidecar 中
`water_plant_rollout_200_step6500_raw` 的 rollout provenance：151 条 success、
49 条 failure。默认 round1 manifest 只把 49 条 failure 转成 failure event 加入训练，
151 条 rollout success 只记录 provenance，不进入训练子集。

## DexJoCo 闭环评估

```bash
python scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
  --gpus 0,1,2,3 \
  --run-dir runs/water_plant_uncond_2cam_384_1e-4_lora/<run_id> \
  --checkpoint runs/.../step_XXXXX.pt \
  --no-load-text-encoder \
  --tasks water_plant --episodes 100 \
  --output-dir evaluate_results/dexjoco/baseline_lora/step_XXXXX
```

详见 [`../dexjoco_async/README.md`](../dexjoco_async/README.md)。

## 本目录文件

| 文件 | 用途 |
|------|------|
| `prepare_2cam.sh` | 数据准备 + text embed + 样本校验 |
| `train_2cam.sh` | LeRobot 窗口训练 launcher |
| `build_eve_round1_sidecar.sh` | 构造 EveRobot v0.2 sidecar 和 round1 training manifest |
| `train_eve_round1.sh` | 使用 EveRobot manifest 训练 round1 failure self-evolution 模型 |
| `collect_rollout_200_trim8s_and_train.sh` | water_plant 默认参数 wrapper，实际调用 `../collect_rollout_trim_and_train.sh` |
| `train_everobot.py` / `train_everobot.sh` | Legacy EveRobot 整 episode 训练 |
| `convert_lerobot_to_everobot.py` | Legacy LeRobot → EveRobot manifest |
| `fix_lerobot_parquet_metadata.py` | parquet metadata 修复 |
| `collect_dexjoco_water_plant_failures.py` / `build_rollout_datasets.py` | 兼容旧路径的 shim，实际调用 `../collect_dexjoco_rollouts.py` / `../build_rollout_datasets.py` |
| `../dexjoco_async/` | 通用 DexJoCo 闭环 eval / collect orchestrator |
