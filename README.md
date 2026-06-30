# Interaction-centric WAM

> *WAM learns from state transitions, while physical understanding emerges from interaction transitions.*  
> 世界行动模型从状态转移中学习；物理理解来自交互转移。

[![上游 FastWAM](https://img.shields.io/badge/上游-FastWAM-111111.svg)](./docs/FASTWAM_UPSTREAM.md)

**文档：** [`docs/FASTWAM_UPSTREAM.md`](./docs/FASTWAM_UPSTREAM.md) · [`scripts/README.md`](./scripts/README.md)

---

## 核心贡献

1. **Interaction-centric Event 数据构造** — 以交互边界划分 event，替代 LeRobot 格式里的固定长度 clip 方案。
2. **触觉（Tactile）** ：

```text
tactile planning → future tactile prediction → tactile-refined action
```

3. **Failure 闭环** — 将部署阶段的失败回灌训练，迭代模型：

```text
Success → Deploy → Failure → Retrain
```

---

## 实验设计

当前实验先验证 Failure 数据是否能作为 Interaction-centric WAM 的第一条闭环信号。DexJoCo `water_plant` 使用双视角视频与 proprioception；真机 `spray_water` 保持现有 FastWAM deploy 链路。当前执行顺序以 **B → C → A/reference** 为主：A 组已有学长外部 success-only 参考结果，先把 B/C 的 failure-data 对照跑完整，再决定是否补同 pipeline 的 A。

训练预算以 B 的完整运行作为主要对齐目标。`6500`、`6000/6500` 等中间 checkpoint 保留为诊断点，但主 rollout 不再停在 `C@6500`：C 需要继续训到与 B late/final 接近的 `12240` steps 后再做闭环评估。结果分析同时报告中间 checkpoint 的 validation 轨迹和最终 rollout，不把短预算 checkpoint 当作最终结论。

| 组别 | 目标 | 训练数据 | 文本 / metadata | Loss 设计 | 状态 |
|------|------|----------|-----------------|-----------|------|
| B. Text failure | 验证将 failure 作为语言上下文加入视频预训练是否稳定 | Success + Failure | failure 样本在 task text 后追加 `Failed to finish the whole process.` | Success: video + action；Failure: video only，action loss weight = 0 | 已完成训练；25-episode rollout 已有，100-episode 待补 |
| A. Vanilla success | 构造同配置 success-only 对照 | Success only | 原始 task text | video + action | 外部 25-episode 参考约 70-80%；正式同 pipeline A 仍待导入或补跑 |
| C. Structured failure | 验证结构化 outcome 信号是否优于文本拼接 | Success + Failure | 独立 outcome / failure flag | Success: video + action；Failure: video only，action loss weight = 0 | 训练中；目标对齐 B late/final，训至约 12240 steps |

核心控制变量：

- 三组尽量保持相同模型、数据划分、评估脚本、双视角输入与 proprioception 设置；A 外部参考不能直接作为严格同 pipeline 结论。
- `6000/6500` checkpoint 作为诊断点保留；B/C 的主 rollout 以 late/final 预算为准，避免把 B@6500 的低成功率误读成完整训练结论。
- Failure 样本不参与 action loss；action loss 的分母只统计启用 action 监督的 success 样本。
- Failure 样本仍参与视频生成目标，用来测试失败轨迹中的视觉交互动态是否能改善或至少不破坏后续动作策略。
- 当前 rollout 顺序：C 训满 `12240` 后先跑 25 episodes 并更新中文 HTML；随后把 B 从既有 25 episodes 追加 75 到 100；最后把 C 从 25 追加 75 到 100。每完成一段都同步中文讲义式 HTML。
- Checkpoint 清理保留 best-val weight 和最近若干 weight；state checkpoint 只保留最近一个，避免远端存储被频繁保存占满。

阶段路线：

| 阶段 | 内容 | 目的 |
|------|------|------|
| 1 | Failure 闭环：采集 failure，训练 B/A/C 消融，逐组 eval | 验证 failure 数据是否值得进入主线 |
| 2 | Event / subtask metadata：按交互边界切段并加入轻量监督 | 从 episode 级 failure 走向 interaction 级分析 |
| 3 | Adaptive context：按当前交互状态决定是否使用 metadata、planning 或直接 action | 让额外模块在需要时介入，而不是固定增加推理负担 |
| 4 | Tactile：加入接触期触觉观测、预测与动作修正 | 把 interaction signal 从视觉扩展到真实接触 |

---

## 当前进展

三大贡献在方法上通用，但**真机与仿真需分别实现**（数据格式、采集与 deploy 链路不同）。

### 真机（`spray_water`）

| 方向 | 状态 |
|------|------|
| FastWAM 基线（训练、deploy、开环） | 进行中 |
| Failure 闭环 | 设计中 |
| 触觉 | 未开始 |
| Event 数据构造 | 未开始 |

### 仿真（DexJoCo）

| 方向 | 状态 |
|------|------|
| FastWAM 基线（全参 MoT） | `water_plant_uncond_2cam_384_1e-4` 已训至 step_6500，闭环 eval 完成；A 组外部参考约 70-80% |
| **baseline_lora**（Video LoRA） | **验证中**：LeRobot 固定窗口 + video LoRA，对照全参基线 |
| Failure 闭环 | B 已有 25-episode 结果；C 训至 ~12240 后按 C25→B100→C100 rollout |
| 触觉 | 未开始 |
| Event 数据构造 | 未开始 |

### 结果记录

| 组别 | 闭环成功率 | 主要失败模式 | checkpoint 依据 | 报告 |
|------|------------|--------------|-----------------|------|
| B. Text failure | 25-episode 已有初步结果，100-episode 待补 | 待分析 | `6500` / best-val / late-final checkpoints | 中文 HTML 待同步 |
| A. Vanilla success | 外部参考约 70-80% / 25 episodes | 待导入 | 学长 run，非严格同 pipeline | 待导入或补跑 |
| C. Structured failure | 待 12240 后评估 | 待评估 | 保留 `6000/6500` 中间 checkpoint，主结果用 late/final | 待更新 |

---

## Video LoRA（独立可选路径）

LoRA 是**与默认全参训练完全解耦**的可选模块，不影响现有 `fastwam` 基线：

| 维度 | 默认全参 (`model=fastwam`) | Video LoRA (`model=fastwam_video_lora`) |
|------|---------------------------|----------------------------------------|
| Video DiT | 全参微调 | **仅 LoRA adapter**（rank 32，self-attn + FFN） |
| ActionDiT / proprio | 全参微调 | 全参微调（不变） |
| 默认配置 | `configs/model/fastwam.yaml` | `configs/model/fastwam_video_lora.yaml` |
| Checkpoint | 标准 MoT 权重 | `checkpoint_format: video_lora_v1`（LoRA + action + proprio） |
| 实现 | — | `src/fastwam/models/wan22/video_lora.py` |

切换方式：在 task 配置里 `override /model: fastwam_video_lora`，或命令行传 `model=fastwam_video_lora`。训练入口（`train.py` / `train_everobot.py`）、数据 pipeline、DexJoCo deploy 均不变；推理时 run 的 `config.yaml` 含 `model.video_lora.enabled=true` 即自动加载 LoRA checkpoint。

### 使用

**LeRobot 固定窗口（当前 baseline_lora 对照实验）：**

```bash
# 数据准备（与全参基线相同）
bash scripts/prepare_water_plant_2cam.sh
python scripts/precompute_text_embeds.py task=water_plant_uncond_2cam_384_1e-4

# 训练：Video LoRA + ActionDiT 全参
bash scripts/train_water_plant_2cam.sh task=water_plant_uncond_2cam_384_1e-4_lora
```

**EveRobot 整 episode（DiffSynth 风格，可变 T）：**

```bash
python scripts/convert_lerobot_to_everobot.py \
  --dataset-dir data/water_plant_fastwam --video-keys front wrist

bash scripts/train_everobot.sh task=everobot_water_plant_full_lora
```

（EveRobot + LoRA 的 task / data 配置说明见 [`scripts/README.md`](./scripts/README.md)。）

**DexJoCo 闭环评估（与全参相同脚本，传入 LoRA run 目录即可）：**

```bash
python scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
  --gpus 0,1,2,3 \
  --run-dir runs/water_plant_uncond_2cam_384_1e-4_lora/<run_id> \
  --checkpoint runs/water_plant_uncond_2cam_384_1e-4_lora/<run_id>/checkpoints/weights/step_XXXXX.pt \
  --no-load-text-encoder \
  --task-config-dir third_party/dexjoco/configs/rand_obj \
  --tasks water_plant --episodes 100 --seed 25 \
  --replan-steps 24 --control-mode blocking --max-env-steps 1500 \
  --output-dir evaluate_results/dexjoco/baseline_lora/step_XXXXX
```

依赖 `peft`（已写入 `pyproject.toml`，`pip install -e .` 即可）。更多细节见 [`scripts/README.md`](./scripts/README.md#video-lora--actiondit-full-fine-tune-optional)。

### baseline_lora 验证进展

对照组为同数据、同双视角、同 proprio 的全参 run `water_plant_uncond_2cam_384_1e-4`（step_6500）。

| 实验 | Task / 采样 | Run | 状态 |
|------|-------------|-----|------|
| 全参基线 | LeRobot 固定窗口 | `runs/water_plant_uncond_2cam_384_1e-4/` | eval 完成（`evaluate_results/dexjoco/water_plant/step_006500`） |
| **baseline_lora** | LeRobot 固定窗口 + video LoRA | `runs/water_plant_uncond_2cam_384_1e-4_lora/` | **训练中**，训完即跑同配置闭环 eval |
| everobot_full_lora | EveRobot 整 episode + video LoRA | `runs/everobot/everobot_water_plant_full_lora/` | step_475 已 eval（`evaluate_results/dexjoco/everobot_full_lora/step_000475`），待与 baseline_lora 横向对比 |

目标：确认 video LoRA 能否在显著减少 video 侧可训练参数的同时，达到或接近全参基线的闭环成功率，作为后续 failure 闭环（B/A/C）的默认训练方式。

---

## 代码地图

```text
configs/          Hydra 配置（model/fastwam.yaml 与 model/fastwam_video_lora.yaml 并行）
src/fastwam/      模型（video_lora.py 为独立 LoRA 模块）
scripts/train.py  训练（LeRobot 固定窗口）
scripts/train_everobot.py  训练（EveRobot 整 episode）
scripts/wuji/     真机 deploy
scripts/openloop/ 开环评估
scripts/dexjoco_async/ DexJoCo 闭环评估
data/  runs/  evaluate_results/  logs/
```

---

## 引用

```bibtex
@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026}
}
```

多卡dexjoco并行测试
```python
python scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
  --gpus 0,1,2,3 \
  --run-dir runs/water_plant_uncond_2cam_384_1e-4/2026-06-29_16-38-39 \
  --checkpoint runs/water_plant_uncond_2cam_384_1e-4/2026-06-29_16-38-39/checkpoints/weights/step_006500.pt \
  --no-load-text-encoder \
  --task-config-dir third_party/dexjoco/configs/rand_obj \
  --tasks water_plant \
  --episodes 100 --seed 25 \
  --replan-steps 24 --control-mode blocking \
  --max-env-steps 1500 \
  --output-dir evaluate_results/dexjoco/water_plant/step_006500
```