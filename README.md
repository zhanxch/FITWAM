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

**推进顺序**（与上文贡献编号无关）：**Failure 数据 → 触觉 → Event 数据构造**。

当前先围绕 Failure 开展设计，首要验证：**在 Success demo 基线上加入 failure 轨迹训练，是否带来可测提升**（如子任务成功率、同类失败复现率；真机与仿真分别做）。

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1 | Failure 数据：采集、入库、Success+Failure 训练对比 | 设计中 |
| 2 | 触觉：接触期模态与预测链路 | 未开始 |
| 3 | Event 数据构造：交互边界切分与评估 | 未开始 |

触觉与 Event 改动建立在 Failure 管线跑通之后；Event 切分与 metadata 主要服务更细粒度的 failure 分析与采样，后置实现。

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
| FastWAM 基线 | - |
| Failure 闭环 | 未开始 |
| 触觉 | 未开始 |
| Event 数据构造 | 未开始 |

---

## 代码地图

```text
configs/          Hydra 配置
src/fastwam/      模型
scripts/train.py  训练
scripts/1/        真机 deploy
scripts/openloop/ 开环评估
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
