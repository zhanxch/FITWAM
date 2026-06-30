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

**EveRobot 整 episode（DiffSynth 风格）：**

```bash
python scripts/water_plant/convert_lerobot_to_everobot.py \
  --dataset-dir data/water_plant_fastwam --video-keys front wrist
bash scripts/water_plant/train_everobot.sh task=everobot_water_plant_full_lora
```

Task 配置：`configs/task/water_plant_uncond_2cam_384_1e-4*.yaml`；数据配置 `configs/data/water_plant_2cam.yaml`（`proprio_output_dim: 23`）。

## DexJoCo 闭环评估

```bash
python scripts/water_plant/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
  --gpus 0,1,2,3 \
  --run-dir runs/water_plant_uncond_2cam_384_1e-4_lora/<run_id> \
  --checkpoint runs/.../step_XXXXX.pt \
  --no-load-text-encoder \
  --tasks water_plant --episodes 100 \
  --output-dir evaluate_results/dexjoco/baseline_lora/step_XXXXX
```

详见 [`dexjoco_async/README.md`](dexjoco_async/README.md)。

## 本目录文件

| 文件 | 用途 |
|------|------|
| `prepare_2cam.sh` | 数据准备 + text embed + 样本校验 |
| `train_2cam.sh` | LeRobot 窗口训练 launcher |
| `train_everobot.py` / `train_everobot.sh` | EveRobot 整 episode 训练 |
| `convert_lerobot_to_everobot.py` | LeRobot → EveRobot manifest |
| `fix_lerobot_parquet_metadata.py` | parquet metadata 修复 |
| `export_text_embed_cache_npz.py` | `.pt` text cache → `.npz`（DexJoCo 无 torch 客户端用） |
| `dexjoco_async/` | 闭环 eval server/client + 多卡 orchestrator |

## 历史 / 临时脚本（不在本目录）

- `run_dexjoco_async_lpf_eval_clients.sh`、`summarize_dexjoco_async_ablation.py` — 早期 microwave async/LPF 消融，保留作参考，非当前主线。
