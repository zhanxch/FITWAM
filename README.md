# Interaction-centric WAM

> *WAM learns from state transitions, while physical understanding emerges from interaction transitions.*  
> 世界行动模型从状态转移中学习；物理理解来自交互转移。

[![上游 FastWAM](https://img.shields.io/badge/上游-FastWAM-111111.svg)](./docs/FASTWAM_UPSTREAM.md)

**文档：** [`docs/FASTWAM_UPSTREAM.md`](./docs/FASTWAM_UPSTREAM.md) · [`scripts/README.md`](./scripts/README.md) · [`scripts/water_plant/README.md`](./scripts/water_plant/README.md)

---

## 核心贡献

研究路线按 5 个 Milestone 展开（详见 [实验设计](#实验设计)）：

1. **Failure video 训练** — 论证 failure 轨迹纳入 video 预训练能提升闭环表现（M1）
2. **EveRobot 数据格式** — LeRobot-compatible sidecar，记录 outcome、failure event、训练 manifest 与自进化 provenance（M2）
3. **面向 failure 的架构** — 更好消费 failure 信号的模型设计（M3）
4. **失败轨迹自进化** — Train → Test → append rollout → manifest subset retrain，后续接 steer token / RL token（M4）
5. **触觉（Tactile）** — 拓展 M2/M3 至接触期（M5，真机为主）：

```text
tactile planning → future tactile prediction → tactile-refined action
```

---

## 实验设计

研究按 **5 个 Milestone** 递进：先证明 failure 数据对 video 训练有价值，再改进数据构造与模型架构，最后建立可迭代闭环并在真机引入触觉。仿真主战场为 DexJoCo `water_plant`（双视角 + proprio）；真机主战场为 `spray_water`（现有 FastWAM deploy 链路）。

```text
M1  failure video 有用？  →  M2  怎么构造数据？  →  M3  什么架构更好？
                                      ↓                        ↓
M5  触觉拓展 M2/M3（真机）  ←  M4  迭代闭环 + RL（仿真/真机，架构最后验证）
```

### Milestone 1：Failure video 训练是否有效

**目标：** 鉴于现在的提升较为明显，所以想充分论证failure video 效果，后续更好开展。

**要论证：** 将 failure 轨迹纳入 **video 预训练**（failure 只训 video、不训 action）能否提升闭环表现，优于纯 success 训练。

| 组别 | 训练数据 | 文本 / metadata | Loss | 对应假设 |
|------|----------|---------------|------|----------|
| **Vanilla success（A）** | Success only | 原始 task text | video + action | 基线 |
| **Text failure（B）** | Success + Failure | failure 样本 text 后追加 `Failed to finish the whole process.` | Success: video + action；Failure: **video only**（action loss weight = 0） | failure 视觉动态有助于后续 action |

**控制变量：** 同模型、同数据划分、同 eval 脚本、同双视角与 proprio；failure 不参与 action loss，分母只统计 success 样本。

**当前状态（DexJoCo）：** B 已完成 100-episode 主评估：`38 -> 81 -> 82`（step 6500 / 11000 / 12240）；A 为外部参考约 `70-80%`，暂不混入本次 B/C 主证据。

---

### Milestone 2：Interaction-centric 数据构造与配套训练

**目标：** 提出 EveRobot，一种基于 LeRobot 的 failure/self-evolution sidecar 数据格式，使失败 rollout 能被系统地记录、筛选、切窗并纳入下一轮训练。

**要论证：** 相比 M1 的 episode 级 text 拼接，EveRobot 用结构化 metadata 明确表达 rollout 来源、success/failure outcome、failure event window 和训练子集，从而更有效地把 failure 轨迹转成可训练样本。

| 方向 | 内容 | 相对 M1 的增量 |
|------|------|----------------|
| **Episode provenance** | `source_policy`、`collection_round`、`seed`、`episode_outcome` | 支持 Train → Test → Retrain 的可追踪数据闭环 |
| **Failure event window** | 在完整 rollout 上用 `event_meta` 标注可训练窗口 | 不物理截断原始数据，训练时只采样 failure event 部分 |
| **Training manifest** | 每轮训练显式选择 success episode / failure event 子集 | 支持 append-only 数据扩充和 round-specific retrain |
| **Loss policy** | failure event 可设置 `action_loss=disabled` | failure 用于 video/proprio/context 学习，不直接模仿失败动作 |

**当前实现：** EveRobot v0.1 已在 `water_plant` 上落地为 sidecar 格式，保持 LeRobot 原始 `data/`、`videos/`、`meta/` 不变，只在 `data/water_plant_fastwam/eve/` 下生成：

| 文件 | 用途 |
|------|------|
| `schema_version.json` | EveRobot 格式版本和 LeRobot 兼容说明 |
| `episode_meta.jsonl` | episode 级 provenance、round、policy、success/failure、seed、length |
| `event_meta.jsonl` | failure event window、failure type、action loss 策略、预留 steer token |
| `manifests/train_round1_success_plus_failure_events.json` | round1 训练子集：100 条 success episode + 49 个 failure event |

训练适配由 `fastwam.datasets.eve.manifest_dataset.EveManifestRobotVideoDataset` 完成：success episode 按普通 LeRobot 固定窗口采样；failure event 只在 `start_frame <= window < end_frame` 内采样；`action_loss=disabled` 时返回 `action_loss_weight=0.0`。

---

### Milestone 3：面向 failure 学习的模型架构

**目标：** 架构上，为failure data的学习专门设计点模块，能涨点就行，主要是为了叙事上完整。

**要论证：** 在 M2 的最优数据构造之上，新架构比标准 FastWAM MoT **更能利用 failure 信号**（而非仅堆数据）。

候选方向（与 M2 解耦、逐步验证）：

- **Adaptive context：** 按当前交互状态决定是否启用 metadata / planning / 直接 action
- **Failure-aware video–action 耦合：** 接触期、失败边界处的差异化监督或推理路径
- 其他待 M2 结果收敛后选定的架构变体

**原则：** M3 只在 M2 确定「怎么喂 failure」之后启动；每次只改架构，数据与 eval 协议不变。

---

### Milestone 4：Train → Test → Retrain 迭代闭环

**目标：** 迭代闭环，参考Pi 0.6 的RECAP。他们的是RL，需要执行过程中的人工纠正，失败数据用于更新advantage计算。先做仿真，再做真机。

**要论证：** 部署失败样本回灌 + 多轮迭代，能否持续提升性能上限；并消融迭代轮数、回灌比例、checkpoint 选择策略。

```text
Success 数据预训练 → Deploy / 闭环测试 → 采集 failure
        ↑                                      ↓
        └──────────── Retrain（video ± action）┘
                      （重复 K 轮，测上限）
```

**RL 扩展（探索方向）：** 真机 RL 上限较高，拟在闭环稳定后引入 **RL token / 在线微调** 等机制，把 failure 轨迹转为可优化信号；架构改动放在 **M4 后期**，需 M1–M3 在仿真与真机均验证后再做。

| 验证场 | 优先级 | 说明 |
|--------|--------|------|
| DexJoCo | 先 | 低成本、可复现，主做迭代消融 |
| 真机 `spray_water` | 后 | 与 deploy 链路对齐；RL 与触觉更依赖真机 |

**当前状态：** round0 FastWAM 已完成 200 次 rollout，生成 151 条 success + 49 条 failure；EveRobot round1 manifest 默认只把 49 条 failure event 回灌训练，151 条 rollout success 仅记录 provenance。

---

### Milestone 5：触觉（Tactile）

**目标：** 加入触觉，计划参考周哥他们即将推出的基模，感觉不需要搞太复杂，就在他们基础上能基于触觉去更好区分成功失败轨迹就行。这个后面再议，而且估计只做真机。

**要论证：** 在接触密集阶段，触觉观测与预测能否拓展 M2 的 event 构造与 M3 的架构，进一步提升 failure 边界处的表现。

```text
tactile planning → future tactile prediction → tactile-refined action
```

| 范围 | 计划 |
|------|------|
| **真机** | 主战场；`TouchAnything` 等已有数据，与 M2 event / M3 架构联合设计 |
| **仿真** | 触觉夹爪环境可选，非必须；优先保证真机链路 |

**依赖：** M2（接触期 event 切段）与 M3（触觉分支架构）就绪后再系统推进；可与 M4 迭代闭环在真机侧汇合。

---

### 里程碑依赖与当前焦点

| Milestone | 依赖 | 状态 |
|-----------|------|------|
| **M1** | FastWAM 基线 / LoRA 基线 | **进行中**（B vs A） |
| **M2** | M1 阳性结果 | EveRobot v0.1 sidecar 已实现；round1 failure-event manifest 已生成 |
| **M3** | M2 最优构造 | 未开始 |
| **M4** | M1–M3 + deploy 链路 | round0 rollout → round1 manifest 闭环已打通 |
| **M5** | M2 + M3（真机） | 未开始 |

**工程并行项（非 milestone 结论）：** `baseline_lora` 验证 video LoRA 能否作为更轻量的默认训练方式，服务于 M1 及之后各阶段的训练效率，不改变上述论证顺序。

---

## DexJoCo 基准对比

各方法在 DexJoCo 任务上的闭环成功率（%，Mean ± Std）。**加粗**表示该表内该行最优值；`/B` 表示 blocking 控制模式。

### rand-obj

| Task | DP-T | DP-C | ACT | π₀.₅ | GR00T N1.5 |
|------|------|------|-----|------|------------|
| Hammer Nail | 81.3±3.1 | 58.7±4.2 | 50.0±7.2 | **84.7±5.0** | 67.3±4.2 |
| Click Mouse | 62.0±2.0 | 74.0±5.3 | 61.3±3.1 | 64.7±8.1 | **85.3±3.1** |
| Pick Bucket | 83.3±3.1 | 70.0±2.0 | 64.0±4.0 | **84.0±7.2** | 72.0±6.0 |
| Pinch Tongs | 22.7±5.8 | **57.3±6.4** | 31.3±3.1 | 24.0±6.9 | 12.7±2.3 |
| Fold Glasses | 53.3±3.1 | 54.0±15.9 | 47.3±11.0 | **72.0±3.5** | 27.3±2.3 |
| Water Plant | 84.0±3.5 | 63.3±3.1 | 47.3±4.6 | **88.7±3.1** | 72.7±1.2 |
| **Avg. (单臂)** | 64.4 | 62.9 | 50.2 | **69.7** | 56.2 |

### FastWAM / EveRobot water_plant

`rand-obj` `water_plant` 上的 FastWAM 当前闭环结果：

| Method | Success Rate | Successes / Trials | Notes |
|--------|-------------:|-------------------:|-------|
| FastWAM | 75.5% | 151 / 200 | success-only FastWAM rollout |
| FastWAM + failure scratch | 82.0% | 82 / 100 | failure data from scratch |
| FastWAM + LoRA continuation | 81.5% | 163 / 200 | continued training with rollout data |

其中 `FastWAM 75.5% (151/200)` 同时作为 EveRobot round1 的 rollout provenance：
`water_plant_rollout_200_step6500_raw` 中 151 条 success、49 条 failure。当前 round1
manifest 只将 49 条 failure 转成 failure event 进入训练，rollout success 默认不进入训练子集。


### multi-task

多任务联合训练（%，Mean ± Std）。**加粗**表示该表内该行最优值。

| Task | DP-T | π₀.₅ |
|------|------|------|
| Hammer Nail | 58.7±5.0 | **86.7±3.1** |
| Click Mouse | 38.7±3.1 | **80.7±3.1** |
| Pick Bucket | 55.3±7.6 | **83.3±8.1** |
| Pinch Tongs | 6.0±5.3 | **45.3±6.1** |
| Fold Glasses | 11.3±5.0 | **42.0±6.0** |
| Water Plant | 60.0±6.9 | **84.0±4.0** |
| **Avg. (单臂)** | 38.3 | **70.3** |

---

## 当前进展

三大贡献在方法上通用，但**真机与仿真需分别实现**（数据格式、采集与 deploy 链路不同）。下表按 **Milestone** 对齐当前状态。

### 真机（`spray_water`）

| Milestone | 状态 |
|-----------|------|
| M1 | FastWAM 基线训练 / deploy / 开环进行中；failure 对照未开始 |
| M2–M3 | 未开始 |
| M4 | 设计中（deploy 链路已有 ZMQ server/client） |
| M5 | 未开始（`TouchAnything` 数据本地已有） |

### 仿真（DexJoCo `water_plant`）

| Milestone | 状态 |
|-----------|------|
| **M1** | **进行中**：B（Text failure）100-ep 主结果完成；A（Vanilla）外部参考 ~70–80%，同 pipeline 主结果待补 |
| **M2** | EveRobot v0.1 已完成：sidecar schema、episode/event metadata、round1 manifest、manifest-driven dataset adapter |
| M3 | 未开始 |
| M4 | round0 FastWAM 200 次 rollout 已回灌为 round1 failure-event manifest；多轮自进化待继续 |
| M5 | 未开始（仿真触觉夹爪可选，非优先） |

**工程并行：** 全参基线 step_6500 eval 完成；`baseline_lora` 训练中，用于确认 LoRA 可否作为后续 milestone 的默认训练后端。

### M1 结果记录（DexJoCo）

| 组别 | 闭环成功率 | 主要失败模式 | checkpoint 依据 | 报告 |
|------|------------|--------------|-----------------|------|
| B. Text failure | 38/100 → 81/100 → 82/100 | 6500 明显未训够；后期稳定 | step 006500 / 011000 / 012240 | [`results/dexjoco_water_plant_failure_ablation`](./results/dexjoco_water_plant_failure_ablation/) |
| A. Vanilla success | 外部参考 ~70–80% | — | 学长 run（非同 pipeline） | 待补跑同 pipeline |
| C. Structured failure | 74/100 → 59/100 → 4/100 | late/final checkpoint 退化 | step 006500 / 011500 / 012240 | [`results/dexjoco_water_plant_failure_ablation`](./results/dexjoco_water_plant_failure_ablation/) |

DexJoCo async rollout 代码位于 [`scripts/dexjoco_async`](./scripts/dexjoco_async/)；这里只作为评估工具链使用，不是项目的主要方法贡献。

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
bash scripts/water_plant/prepare_2cam.sh
python scripts/precompute_text_embeds.py task=water_plant_uncond_2cam_384_1e-4

# 训练：Video LoRA + ActionDiT 全参
bash scripts/water_plant/train_2cam.sh task=water_plant_uncond_2cam_384_1e-4_lora
```

**Legacy EveRobot 整 episode（DiffSynth 风格，可变 T）：**

```bash
python scripts/water_plant/convert_lerobot_to_everobot.py \
  --dataset-dir data/water_plant_fastwam --video-keys front wrist

bash scripts/water_plant/train_everobot.sh task=everobot_water_plant_full_lora
```

（这是早期整 episode 训练路径；failure self-evolution 使用上文的 EveRobot v0.1 sidecar。）

**DexJoCo 闭环评估（与全参相同脚本，传入 LoRA run 目录即可）：**

```bash
python scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
  --gpus 0,1,2,3 \
  --run-dir runs/water_plant_uncond_2cam_384_1e-4/2026-06-29_16-38-39/ \
  --checkpoint runs/water_plant_uncond_2cam_384_1e-4/2026-06-29_16-38-39/checkpoints/weights/step_006500.pt \
  --no-load-text-encoder \
  --task-config-dir third_party/dexjoco/configs/rand_obj \
  --tasks water_plant --episodes 200 --seed 0 \
  --replan-steps 24 --control-mode blocking --max-env-steps 600 \
  --output-dir evaluate_results/dexjoco/water_plant_200/step_006500
```

依赖 `peft`（已写入 `pyproject.toml`，`pip install -e .` 即可）。更多细节见 [`scripts/README.md`](./scripts/README.md#video-lora--actiondit-full-fine-tune-optional)。

### baseline_lora 验证进展

对照组为同数据、同双视角、同 proprio 的全参 run `water_plant_uncond_2cam_384_1e-4`（step_6500）。

| 实验 | Task / 采样 | Run | 状态 |
|------|-------------|-----|------|
| 全参基线 | LeRobot 固定窗口 | `runs/water_plant_uncond_2cam_384_1e-4/` | eval 完成（`evaluate_results/dexjoco/water_plant/step_006500`） |
| **baseline_lora** | LeRobot 固定窗口 + video LoRA | `runs/water_plant_uncond_2cam_384_1e-4_lora/` | **训练中**，训完即跑同配置闭环 eval |
| everobot_full_lora | EveRobot 整 episode + video LoRA | `runs/everobot/everobot_water_plant_full_lora/` | step_475 已 eval（`evaluate_results/dexjoco/everobot_full_lora/step_000475`），待与 baseline_lora 横向对比 |

目标：确认 video LoRA 能否在显著减少 video 侧可训练参数的同时，达到或接近全参基线的闭环成功率，作为 M1 及之后各 milestone 的默认训练后端候选。

---

## 代码地图

```text
configs/          Hydra 配置（model/fastwam.yaml 与 model/fastwam_video_lora.yaml 并行）
src/fastwam/      模型（video_lora.py 为独立 LoRA 模块）
src/fastwam/datasets/eve/  EveRobot v0.1 manifest-driven dataset adapter
scripts/train.py  训练（LeRobot 固定窗口，通用入口）
scripts/everobot/ EveRobot v0.1 sidecar 构造工具
scripts/water_plant/  water_plant 数据准备 / 训练 / DexJoCo eval
scripts/spray_water_gr00tstyle/  真机 spray_water 训练 / deploy
scripts/openloop/ 开环评估引擎
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
    --gpus 0,1 \
    --run-dir runs/pinch_tongs_uncond_2cam_384_1e-4/2026-07-03_10-22-54 \
    --checkpoint runs/pinch_tongs_uncond_2cam_384_1e-4/2026-07-03_10-22-54/checkpoints/weights/step_005000.pt \
    --no-load-text-encoder \
    --task-config-dir third_party/dexjoco/configs/rand_obj \
    --tasks pinch_tongs\
    --episodes 100 --seed 0 \
    --replan-steps 25 --control-mode blocking \
    --max-env-steps 600 \
    --output-dir evaluate_results/dexjoco/pinch_tongs/step_005000    
```
