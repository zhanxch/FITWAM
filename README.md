# Interaction-centric WAM

基于 [FastWAM](https://arxiv.org/abs/2603.16666) 的 fork：**从 clip 级世界想象，走向以交互事件为中心的世界行动模型**。

[![上游 FastWAM](https://img.shields.io/badge/上游-FastWAM-111111.svg)](./docs/FASTWAM_UPSTREAM.md)

> **当前基线：** `spray_water` 真机（156 demos + `filtered_out` holdout）、MoT 架构、ZMQ 部署（`scripts/1/`）。  
> **上游安装 / LIBERO / RoboTwin：** [`docs/FASTWAM_UPSTREAM.md`](./docs/FASTWAM_UPSTREAM.md) · **脚本：** [`scripts/README.md`](./scripts/README.md) · **诊断：** [`scripts/diagnose/README.md`](./scripts/diagnose/README.md)

## 目录

- [我们在验证什么](#我们在验证什么)
- [当前进展](#当前进展)
- [出发点：FastWAM 哪里不够用](#出发点fastwam-哪里不够用)
- [研究路线图](#研究路线图)
- [代码地图](#代码地图)
- [致谢与引用](#致谢与引用)

---

## 我们在验证什么

本项目不是按周交付的工程排期，而是在真机任务上逐条检验科学假设。合作时优先对齐四件事：

1. **核心假设是什么？**
2. **为验证它要做哪些实验？**
3. **每个实验能推翻或支持什么结论？**
4. **当前做到哪一步？**

| 假设 | 一句话 | 关键对比 | 主指标 |
|------|--------|----------|--------|
| **H1** 交互中心表征 | 交互事件是比固定 clip 更好的建模单元 | 均匀 `frame_stride` vs 事件切分 | per-event / contact-phase L1 |
| **H2** 触觉中心预测 | 触觉让模型在接触期「看得更准」 | 仅 RGB+proprio vs +tactile | 接触段 L1、滑移/力峰值误差 |
| **H3** 失败即监督 | 失败轨迹能提升世界模型，而非应丢弃 | 仅成功 demo vs +failure 微调 | 子任务成功率、失败复现率 |
| **H4** 闭环持续学习 | 部署—采集—再训练能形成稳定改进环 | 单次训练 vs 多轮闭环 | 成功率曲线、failure 占比趋势 |

**核心命题（一句话）：** 世界行动模型应从状态转移中学习，而物理理解来自**交互转移**——接触、施力、失败与恢复，而不是均匀时间窗内的像素平均。

---

## 当前进展

| 模块 | 状态 | 说明 |
|------|------|------|
| FastWAM + `spray_water` 基线 | ✅ 进行中 | 数据管线 A/B、训练、ZMQ 真机部署 |
| 开环评估协议 | ✅ 可用 | `scripts/openloop/`，结果见 `evaluate_results/openloop_episode_gr00tstyle/` |
| 事件 metadata & 切分 | 🔲 未开始 | `events.jsonl`、标注工具、event dataloader |
| 触觉模态 | 🔲 未开始 | 对齐、encoder、进 LeRobot |
| Failure 闭环 | 🔲 未开始 | `failure/` 目录、`outcomes.jsonl`、再训练流程 |
| 自适应推理 | 🔲 未开始 | 空闲稀疏 / 接触密集 replan |

### 已完成基线实验（E0）

| ID | 实验 | 对比 | 结论摘要 |
|----|------|------|----------|
| **E0a** | 数据处理 A | 绝对 action + z-score | 基线对照 |
| **E0b** | 数据处理 B | GR00T 相对 + min/max + clip | 当前主基线 |
| **E0c** | ckpt 5k–30k | 同 episode 开环 | pred 互相关 ~0.97；5k→30k episode L1 ↓约 26%，但曲线形态仍相似 |

**E0c 启示：** 仅靠 episode 级 L1 不足以判断接触段是否变好；**H1 的事件级指标**是下一步必做项。

---

## 出发点：FastWAM 哪里不够用

| 维度 | FastWAM 默认 | `spray_water` 真机暴露的问题 |
|------|--------------|------------------------------|
| **时间切分** | `num_frames=33`、均匀滑窗 | 按压、喷水等阶段短且关键，clip 边界与物理事件不对齐 |
| **模态** | RGB + proprio，视频想象为核心 | 接触力变化快于 RGB；开环各 ckpt 预测高度相关，对 GT 仍有明显偏差 |
| **数据闭环** | 以成功 demo 为主 | 有 `filtered_out` holdout，但缺结构化 failure 与再训练流程 |
| **推理** | 固定 replan 间隔 | 空闲可稀疏、接触需加密；deploy 侧不跑 video diffusion（已确认） |

Interaction-centric WAM **不是否定 FastWAM**，而是在其 MoT（video + action expert）之上：改数据原子、加触觉、解耦时间窗、把失败纳入训练闭环。

```text
         ┌─────────────────────────────────────────┐
         │  metadata：events / outcomes / modality │
         └─────────────────┬───────────────────────┘
                           │
     RGB (慢·长上下文)   tactile (快·接触)   proprio + action
                           │
              interaction event 片段（替代均匀 clip）
```

`spray_water` 初版事件类型：`approach_bottle` · `grasp` · `pump` · `aim_spray` · `spray` · `release`

---

## 研究路线图

以下每条假设对应**可独立认领**的实验模块。认领时请标明：假设编号、实验 ID、预期交付物、成功/失败判据。

---

### 假设 H1 — 交互中心表征

**要回答的问题：** 交互事件是否比固定长度 clip 更适合作为训练与评估的基本单元？

| 实验 | 验证什么 | 变量 | 成功判据 |
|------|----------|------|----------|
| **E1** 事件切分 | clip vs event | `frame_stride=32` vs `events.jsonl` 切分 | contact-phase L1 显著优于 clip 基线 |
| **E6** 事件条件 | phase 是否提供有效归纳偏置 | 无 phase token vs +phase embedding | per-event L1 分层下降 |

**任务清单**

- [ ] 定义交互事件边界与标注规范
- [ ] 事件切分与 `events.jsonl` 生成
- [ ] Event dataloader（替代均匀滑窗采样）
- [ ] 事件级开环评估（扩展 `scripts/openloop/`）
- [ ] Clip 预测 vs 事件预测对比实验

**依赖：** E0 基线完成 ✅ · 需 P1 metadata 工具

---

### 假设 H2 — 触觉中心预测

**要回答的问题：** 在接触密集阶段，触觉能否补足 RGB 的时滞，并改善 action / 未来交互预测？

| 实验 | 验证什么 | 变量 | 成功判据 |
|------|----------|------|----------|
| **E2** 触觉消融 | 接触期是否需要触觉 | 无 tactile vs +tactile encoder | 接触段 L1 ↓；力峰值/滑移对齐改善 |
| **E3** 时间窗解耦 | 长视觉 + 短 action 是否更合理 | 统一 horizon vs `T_v > H_a` | contact L1 不降或下降，且推理延迟可控 |

**任务清单**

- [ ] 触觉数据对齐与 LeRobot 字段扩展
- [ ] Tactile encoder v0（1D conv / MLP → token）
- [ ] 未来 RGB / 触觉 / 交互状态预测（训练侧）
- [ ] 仅视觉 vs 视觉+触觉开环与闭环比对
- [ ] Video–Action horizon 解耦 config

**架构草图**

```text
  RGB [T_v] ──────► Video DiT ──► 短时未来（deploy 可关）
  tactile [T_t] ──► Tactile enc ─┐
  proprio [T_s] ──► proprio tok ─┼──► Action DiT ──► action [H_a]
  phase / event ──► context tok ─┘
```

**远期（与 H2 衔接，非当前阻塞项）**

- [ ] 场景理解 + CoT 规划：何时需要未来预测？
- [ ] 触觉参与规划 vs 仅参与 action 生成

---

### 假设 H3 — 失败改善世界模型

**要回答的问题：** 失败轨迹能否作为有效监督，降低真机重复失败率？

| 实验 | 验证什么 | 变量 | 成功判据 |
|------|----------|------|----------|
| **E4** 失败再训练 | failure 是否有边际收益 | base vs +failure FT | 子任务成功率 ↑；同类失败复现 ↓ |

**任务清单**

- [ ] 真机 failure 采集协议与 `failure/` 目录结构
- [ ] 自动 / 半自动 failure 标注 → `outcomes.jsonl`
- [ ] Failure replay 数据集与过采样策略
- [ ] Failure-aware 微调流程
- [ ] 消融：仅成功 / 成功+失败 / 不同 failure 比例

---

### 假设 H4 — 闭环持续学习

**要回答的问题：** 训练 → 部署 → 采集 → 再训练能否形成可重复的改进闭环，而非一次性离线模型？

| 实验 | 验证什么 | 变量 | 成功判据 |
|------|----------|------|----------|
| **E5** 自适应推理 | 接触期加密 replan 是否更优 | 固定 stride vs adaptive | 成功率 ↑ 或 replan 次数 ↓ |
| **E4′** 多轮闭环 | 迭代再训练是否累积收益 | 单轮 vs k 轮闭环 | 跨轮次成功率单调或阶梯上升 |

**任务清单**

- [ ] Deploy 状态机：`IDLE` 稀疏 replan → `CONTACT` 密集 replan
- [ ] 触觉阈值 / event detector 触发 replan
- [ ] Failure memory 与增量再训练脚本
- [ ] 长程评估协议（跨版本、跨采集批次）
- [ ] 开环 vs 闭环比对；在线 action 修正消融

```text
  训练 v_k → 真机部署 → 日志 + failure → 整理入库 → 微调 v_{k+1}
```

---

### 实验认领速查

| 实验 | 假设 | 难度 | 前置 | 可交付 |
|------|------|------|------|--------|
| E1 | H1 | 中 | E0、事件标注 | event dataloader + 对比报告 |
| E2 | H2 | 中高 | 触觉硬件对齐 | tactile encoder + 接触段指标 |
| E3 | H2 | 中 | E0 | 解耦 config + 延迟 profile |
| E4 | H3 | 中 | failure 采集 | FT 脚本 + 成功率对比 |
| E5 | H4 | 中 | deploy 稳定 | adaptive replan + A/B |
| E6 | H1 | 低中 | E1 部分 | phase token 消融 |

**建议顺序：** E0（已完成）→ **E1** → E2 / E3 → E4 与 E5 并行 → 多轮 H4 验收。

---

### 更远方向（暂无排期）

- 触觉基础模型与跨物体迁移
- VLM 失败推理与交互感知规划
- 视觉–触觉统一世界模型
- 人机协同纠错与交互感知 benchmark

---

## 代码地图

```text
FastWAM/
├── README.md                 # 本文件
├── docs/FASTWAM_UPSTREAM.md  # 上游安装 / LIBERO / RoboTwin
├── configs/                  # Hydra 配置
├── src/fastwam/              # 核心模型
├── scripts/
│   ├── train.py              # 训练入口
│   ├── 1/                    # 真机 ZMQ 部署
│   ├── openloop/             # 开环评估（E0/E1）
│   └── diagnose/             # sim–real 诊断
├── data/                     # 数据集
├── runs/                     # 训练输出
└── evaluate_results/         # 评估结果
```

| 路径 | 角色 |
|------|------|
| `configs/data/` | 数据管线；未来 `*_events.yaml`、`*_tactile.yaml` |
| `scripts/openloop/` | 开环评估入口 |
| `scripts/1/` | 真机 deploy；未来 adaptive replan + failure dump |
| `evaluate_results/openloop_episode_gr00tstyle/` | gr00tstyle 开环结果 |

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
| 2026-06-29 | 按假设 H1–H4 重组路线图，去掉周次排期，突出可认领实验 |
