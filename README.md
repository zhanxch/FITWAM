# Interaction-centric WAM

> *WAM learns from state transitions, while physical understanding emerges from interaction transitions.*  
> 世界行动模型从状态转移中学习；物理理解来自交互转移。

[![上游 FastWAM](https://img.shields.io/badge/上游-FastWAM-111111.svg)](./docs/FASTWAM_UPSTREAM.md)

**当前基线：** `spray_water` 真机 · MoT · ZMQ 部署（`scripts/1/`）  
**文档：** [`docs/FASTWAM_UPSTREAM.md`](./docs/FASTWAM_UPSTREAM.md) · [`scripts/README.md`](./scripts/README.md)

---

## 核心贡献

1. **Interaction-centric Event 数据构造** — 以交互边界划分 event，替代均匀 clip，作为训练与评估的基本单元。  
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

当前实验先验证 Failure 数据是否能作为 Interaction-centric WAM 的第一条闭环信号。DexJoCo `water_plant` 使用双视角视频与 proprioception；真机 `spray_water` 保持现有 FastWAM deploy 链路。训练与评估按 **B → A → C** 顺序推进，每个模型完成训练后立即做闭环评估并更新结果报告，再启动下一个模型。

| 组别 | 目标 | 训练数据 | 文本 / metadata | Loss 设计 | 状态 |
|------|------|----------|-----------------|-----------|------|
| B. Text failure | 验证将 failure 作为语言上下文加入视频预训练是否稳定 | Success + Failure | failure 样本在 task text 后追加 `Failed to finish the whole process.` | Success: video + action；Failure: video only，action loss weight = 0 | 训练中 |
| A. Vanilla success | 构造同配置 success-only 对照 | Success only | 原始 task text | video + action | 待训练 |
| C. Structured failure | 验证结构化 outcome 信号是否优于文本拼接 | Success + Failure | 独立 outcome / failure flag | Success: video + action；Failure: video only，action loss weight = 0 | 待训练 |

核心控制变量：

- 三组尽量保持相同模型、数据划分、训练步数、评估脚本、双视角输入与 proprioception 设置。
- Failure 样本不参与 action loss；action loss 的分母只统计启用 action 监督的 success 样本。
- Failure 样本仍参与视频生成目标，用来测试失败轨迹中的视觉交互动态是否能改善或至少不破坏后续动作策略。
- 每组完成后先记录闭环成功率、典型失败模式、验证曲线和 checkpoint 选择依据，再继续下一组。

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
| FastWAM 基线 | Success-only 对照待跑 |
| Failure 闭环 | B 组训练中，完成后先 eval 再启动 A 组 |
| 触觉 | 未开始 |
| Event 数据构造 | 未开始 |

### 结果记录

| 组别 | 闭环成功率 | 主要失败模式 | checkpoint 依据 | 报告 |
|------|------------|--------------|-----------------|------|
| B. Text failure | 待评估 | 待评估 | 待评估 | 待更新 |
| A. Vanilla success | 待评估 | 待评估 | 待评估 | 待更新 |
| C. Structured failure | 待评估 | 待评估 | 待评估 | 待更新 |

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
