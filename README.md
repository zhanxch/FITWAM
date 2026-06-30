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
| FastWAM 基线 | success-only 外部参考已知；严格同 pipeline A 待导入或补跑 |
| Failure 闭环 | B 已有 25-episode 关键 checkpoint 结果；C 正在训至 12240，完成后按 C25 → B100 → C100 顺序 rollout 并更新中文报告 |
| 触觉 | 未开始 |
| Event 数据构造 | 未开始 |

### 结果记录

| 组别 | 闭环成功率 | 主要失败模式 | checkpoint 依据 | 报告 |
|------|------------|--------------|-----------------|------|
| B. Text failure | 25-episode 已有初步结果，100-episode 待补 | 待分析 | `6500` / best-val / late-final checkpoints | 中文 HTML 待同步 |
| A. Vanilla success | 外部参考约 70-80% / 25 episodes | 待导入 | 学长 run，非严格同 pipeline | 待导入或补跑 |
| C. Structured failure | 待 12240 后评估 | 待评估 | 保留 `6000/6500` 中间 checkpoint，主结果用 late/final | 待更新 |

---

## 代码地图

```text
configs/          Hydra 配置
src/fastwam/      模型
scripts/train.py  训练
scripts/1/        真机 deploy
scripts/openloop/ 开环评估
scripts/dexjoco_async/ DexJoCo async/LPF 历史消融（结果样本在 results/dexjoco_async_microwave/）
data/  runs/  evaluate_results/
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
