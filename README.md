# Interaction-centric WAM

基于 [FastWAM](https://arxiv.org/abs/2603.16666) 的 fork：**从 clip 级世界想象，走向以交互事件为中心的世界行动模型**。

> **核心动机（Motivation）**  
> *WAM learns from state transitions, while physical understanding emerges from interaction transitions.*  
> 世界行动模型从**状态转移**中学习；而**物理理解**来自**交互转移**——接触、施力、失败与恢复，而非均匀时间窗内的状态平均。

[![上游 FastWAM](https://img.shields.io/badge/上游-FastWAM-111111.svg)](./docs/FASTWAM_UPSTREAM.md)

> **当前基线：** `spray_water` 真机（156 demos + `filtered_out` holdout）、MoT 架构、ZMQ 部署（`scripts/1/`）。  
> **上游安装 / LIBERO / RoboTwin：** [`docs/FASTWAM_UPSTREAM.md`](./docs/FASTWAM_UPSTREAM.md) · **脚本：** [`scripts/README.md`](./scripts/README.md) · **诊断：** [`scripts/diagnose/README.md`](./scripts/diagnose/README.md)

## 目录

- [核心动机](#核心动机)
- [三大核心贡献](#三大核心贡献)
- [实验设计](#实验设计)
- [当前进展](#当前进展)
- [技术支撑（非独立贡献）](#技术支撑非独立贡献)
- [代码地图](#代码地图)
- [致谢与引用](#致谢与引用)

---

## 核心动机

**WAM learns from state transitions, while physical understanding emerges from interaction transitions.**

FastWAM 在固定 clip 内建模状态序列，但真机上的物理理解发生在**交互边界**上。本项目的实验与工程工作，都服务于把建模单元从均匀 clip 推进到交互事件，并把部署中的失败纳入可学习的闭环。

---

## 三大核心贡献

本仓库要实现的**核心贡献**是下面三项。实验设计、里程碑认领、论文叙事均围绕它们展开；其余能力（metadata、解耦 horizon、自适应 replan 等）是**服务这三项的技术路径**，不单独作为贡献点。

| # | 贡献 | 要证明什么 |
|---|------|------------|
| **C1** | **Interaction-centric Event 数据构造** | 以交互事件（非均匀 clip）为训练/评估原子，能更好对齐接触、施力等关键阶段 |
| **C2** | **触觉（Tactile）** | 接触期引入触觉，能补足 RGB 时滞，改善 action 预测与交互理解 |
| **C3** | **Failure 数据使用与再训练闭环** | 部署产生的失败轨迹可作为有效监督，经再训练降低重复失败率 |

### C3 核心闭环（本项目的系统性贡献）

```text
  Success（成功 demo 训练）
           ↓
        Deploy（真机部署）
           ↓
        Failure（失败轨迹采集）
           ↓
        Retrain（failure-aware 微调 / 重训）
           ↓
      更好的模型 → 再次 Deploy …
```

**C3 要交付的不是单次离线模型**，而是可重复运转的 **Success → Deploy → Failure → Retrain** 管线：失败不再丢弃，而是结构化入库并驱动下一轮训练。

### 三项贡献如何衔接

```text
  C1 Event 数据          C2 触觉
  （训练样本怎么切）      （接触期信号从哪来）
         \                    /
          \                  /
           ▼                ▼
         更好的交互期建模与评估
                   │
                   ▼
  C3 Failure 闭环（真机上持续改进）
```

`spray_water` 初版事件类型（C1）：`approach_bottle` · `grasp` · `pump` · `aim_spray` · `spray` · `release`

---

## 实验设计

实验**只围绕 C1 / C2 / C3** 设计。认领时请标明：贡献编号、实验 ID、交付物、成功判据。

### C1 — Interaction-centric Event 数据构造

**核心问题：** 交互事件是否比固定 `frame_stride` clip 更适合作为数据与评估单元？

| 实验 | 对比 | 主指标 | 成功判据 |
|------|------|--------|----------|
| **E1** 事件切分 | 均匀 clip vs event 切分 | contact-phase L1、per-event L1 | 事件切分在接触段显著优于 clip |
| **E1b** 事件级评估 | episode L1 vs event L1 | 分层误差（按 `grasp`/`spray` 等） | 暴露 clip 基线看不到的接触段退化 |

**交付清单**

- [ ] 交互事件定义与切分规范
- [ ] Event 数据集构造 pipeline（替代均匀滑窗采样）
- [ ] 事件级开环评估（扩展 `scripts/openloop/`）
- [ ] E1 对比报告（clip vs event）

> **metadata**（`events.jsonl`、`modality.json` 等）是 C1 的**实现手段**：用于标注事件边界、索引片段、对齐多模态，本身不是独立贡献。

---

### C2 — 触觉（Tactile）

**核心问题：** 接触密集阶段，触觉是否带来可测量的预测与行为改善？

| 实验 | 对比 | 主指标 | 成功判据 |
|------|------|--------|----------|
| **E2** 触觉消融 | RGB+proprio vs +tactile | 接触段 action L1；力峰值/滑移误差 | 接触段指标显著改善 |
| **E2b** 事件×触觉 | E1 最优切分下 ±tactile | per-event L1（重点看 `grasp`/`pump`） | 触觉收益集中在交互事件内 |

**交付清单**

- [ ] 触觉采集、时间对齐、写入 LeRobot
- [ ] Tactile encoder v0 → 接入 Action DiT context
- [ ] E2 / E2b 开环与（可选）闭环比对

**架构（C2 落点）**

```text
  RGB [T_v] ──────► Video DiT
  tactile [T_t] ──► Tactile enc ─┐
  proprio [T_s] ──► proprio tok ─┼──► Action DiT ──► action [H_a]
  event / phase ──► context tok ─┘   （event 来自 C1）
```

---

### C3 — Failure 数据使用：Success → Deploy → Failure → Retrain

**核心问题：** 闭环能否在真机上形成**可重复、可度量**的改进，而非一次性离线训练？

| 实验 | 对比 | 主指标 | 成功判据 |
|------|------|--------|----------|
| **E3** failure 微调 | 仅 Success vs Success+Failure | 子任务成功率；同类失败复现率 | +Failure 后成功率 ↑ 或复现 ↓ |
| **E3b** 多轮闭环 | 1 轮 vs k 轮 Deploy→Retrain | 跨轮次成功率曲线 | k 轮后稳定优于单轮基线 |
| **E3c** failure 比例消融 | 不同 failure 采样/过采样比 | 成功率 vs 数据效率 | 找到可复现的最优配比 |

**交付清单**

- [ ] Deploy 侧 failure 日志协议（与 `scripts/1/` 对接）
- [ ] `failure/` 数据目录 + 入库脚本
- [ ] Retrain pipeline（failure-aware 微调 / 重训）
- [ ] E3 / E3b / E3c 真机 A/B 与轮次报告

> **outcomes.jsonl**、失败类型标签、与 event 的对齐，是 C3 的**实现手段**：用于索引失败片段、按事件/阶段分析、指导过采样——服务于 Failure 使用，不是第四项贡献。

---

### 实验认领速查

| 实验 | 贡献 | 前置 | 可交付 |
|------|------|------|--------|
| E1 / E1b | C1 | E0 基线 ✅ | event pipeline + 对比报告 |
| E2 / E2b | C2 | C1 部分、触觉硬件 | tactile encoder + 接触段指标 |
| E3 / E3b / E3c | C3 | Deploy 稳定 | 闭环脚本 + 多轮成功率 |

**建议顺序：** E0（已完成）→ **E1** → **E2**（可与 E1 后半并行）→ **E3**（依赖真机 failure 采集）。

### 已完成基线（E0，为 C1–C3 提供对照）

| ID | 内容 | 结论摘要 |
|----|------|----------|
| **E0a** | 数据处理 A（绝对 + z-score） | 对照 |
| **E0b** | 数据处理 B（GR00T 相对 + min/max） | 当前主训练基线 |
| **E0c** | ckpt 5k–30k 开环 | pred 互相关 ~0.97；episode L1 ↓约 26%，但接触段是否变好未知 |

**E0c → C1：** episode 级指标不够，必须用 **event 级评估（E1b）** 检验 C1。

---

## 当前进展

| 贡献 | 状态 | 说明 |
|------|------|------|
| **基线（E0）** | ✅ 进行中 | 数据 A/B、训练、ZMQ deploy、`scripts/openloop/` |
| **C1 Event 数据** | 🔲 未开始 | 事件定义、切分 pipeline、E1 |
| **C2 触觉** | 🔲 未开始 | 对齐、encoder、E2 |
| **C3 Failure 闭环** | 🔲 未开始 | Deploy 采集 → Retrain；E3 |

---

## 技术支撑（非独立贡献）

以下能力**服务于** C1 / C3（偶尔辅助 C2 评估），不单独列为贡献，也不单独设计主实验：

| 能力 | 服务对象 | 说明 |
|------|----------|------|
| **Metadata**（`events.jsonl`、`outcomes.jsonl`、`modality.json`） | C1、C3 | 事件边界标注、失败索引、多模态对齐 |
| **Video–Action horizon 解耦** | C1、C2 | 长 RGB 上下文 + 短 action chunk；实现细节 |
| **Adaptive replan** | C3 Deploy 阶段 | 接触期加密 replan；提升 Deploy 质量，非论文主贡献 |
| **Phase / event token** | C1 | 条件化建模；E1 的可选扩展 |

### 出发点：FastWAM 哪里不够用

| 维度 | FastWAM 默认 | 本项目要解决的 |
|------|--------------|----------------|
| 数据单元 | 均匀 clip | **C1：** interaction event |
| 接触期模态 | RGB + proprio | **C2：** tactile |
| 数据闭环 |  mostly Success | **C3：** Failure → Retrain |

---

## 代码地图

```text
FastWAM/
├── README.md
├── docs/FASTWAM_UPSTREAM.md
├── configs/                  # 未来 *_events.yaml、*_tactile.yaml
├── src/fastwam/              # + tactile encoder（C2）
├── scripts/
│   ├── train.py
│   ├── 1/                    # Deploy（C3 入口）
│   ├── openloop/             # 评估（C1 event 指标）
│   └── diagnose/
├── data/                     # + failure/（C3）
├── runs/
└── evaluate_results/
```

| 路径 | 对应贡献 |
|------|----------|
| `configs/data/` | C1 事件数据、C2 触觉字段 |
| `scripts/openloop/` | C1 事件级评估 |
| `scripts/1/` | C3 Deploy + failure 采集 |
| `evaluate_results/openloop_episode_gr00tstyle/` | E0 基线结果 |

---

## 致谢与引用

本仓库基于 **[Fast-WAM](https://arxiv.org/abs/2603.16666)**（Yuan et al., 2026）与 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 评估代码。

```bibtex
@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026}
}
```

| 日期 | 修订 |
|------|------|
| 2026-06-29 | 研究规划作为主 README；开环迁至 `scripts/openloop/` |
| 2026-06-29 | 三大贡献 C1/C2/C3；实验围绕 Event / Tactile / Failure 闭环 |
