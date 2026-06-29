# Interaction-centric WAM

基于 [FastWAM](https://arxiv.org/abs/2603.16666) 的 fork：**从 clip 级世界想象，走向以交互事件为中心的世界行动模型**。

> *WAM learns from state transitions, while physical understanding emerges from interaction transitions.*  
> 世界行动模型从状态转移中学习；物理理解来自交互转移。

[![上游 FastWAM](https://img.shields.io/badge/上游-FastWAM-111111.svg)](./docs/FASTWAM_UPSTREAM.md)

**当前基线：** `spray_water` 真机 · MoT · ZMQ 部署（`scripts/1/`）  
**文档：** [`docs/FASTWAM_UPSTREAM.md`](./docs/FASTWAM_UPSTREAM.md) · [`scripts/README.md`](./scripts/README.md)

---

## 核心贡献

1. **Interaction-centric Event 数据构造** — 用交互事件替代均匀 clip，作为训练与评估的基本单元。  
2. **触觉（Tactile）** — 在接触期引入触觉，补足 RGB 的时滞。  
3. **Failure 闭环** — 把部署中的失败当作监督，而非丢弃：

```text
Success → Deploy → Failure → Retrain
```

metadata（`events.jsonl`、`outcomes.jsonl` 等）服务于事件构造与 failure 入库，是实现手段，不是独立贡献。

---

## 实验设计

（待补充）

---

## 当前进展

| 方向 | 状态 |
|------|------|
| FastWAM + `spray_water` 基线 | 进行中（训练、deploy、开环评估） |
| Event 数据构造 | 未开始 |
| 触觉 | 未开始 |
| Failure 闭环 | 未开始 |

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
