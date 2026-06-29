# Interaction-centric WAM

基于 [FastWAM](https://arxiv.org/abs/2603.16666) 的 fork：**从 clip 级世界想象，走向以交互事件为中心的世界行动模型**。

[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README.md)
[![上游 FastWAM](https://img.shields.io/badge/Upstream-FastWAM-111111.svg)](./docs/FASTWAM_UPSTREAM.md)

> **当前基线：** `spray_water` 真机（156 demos + `filtered_out` holdout）、MoT 架构、ZMQ 部署（`scripts/1/`）。  
> **上游安装 / LIBERO / RoboTwin 训练：** [`docs/FASTWAM_UPSTREAM.md`](./docs/FASTWAM_UPSTREAM.md) · **本仓库脚本：** [`scripts/README.md`](./scripts/README.md) · **诊断：** [`scripts/diagnose/README.md`](./scripts/diagnose/README.md)

## 目录

- [1. 核心思路](#1-核心思路一句话)
- [2. 问题：FastWAM 哪里不够用？](#2-问题fastwam-哪里不够用)
- [3. 三层设计](#3-三层设计)
- [4. 研究阶段](#4-研究阶段phases)
- [5. 实验设计](#5-实验设计experiment-matrix)
- [6. Schedule](#6-schedule2026-q3-草案)
- [7. 代码地图](#7-仓库内代码地图)
- [8. 致谢与引用](#8-致谢与引用)

---

## 1. 核心思路（一句话）

WAM learns from state transitions, while physical understanding emerges from interaction transitions.

---

## 2. 问题：FastWAM 哪里不够用？

| 维度 | FastWAM 默认设定 | spray_water 真机暴露的问题 |
|------|------------------|---------------------------|
| **时间切分** | 统一 `num_frames=33`、`frame_stride` 滑窗 | 按压、喷水等接触阶段短且关键，均匀 clip 边界与物理事件不对齐 |
| **模态** | RGB + proprio；视频想象是核心 | 接触瞬间力/触觉变化快，RGB 滞后；开环各 ckpt pred 高度相关（~0.97），对 GT 仍偏差大 |
| **数据闭环** | 成功 demo 训练为主 | `filtered_out` 已有 5 条 holdout，但缺少结构化 failure 标签与再训练流程 |
| **推理** | 固定 replan 间隔 | 空闲段可稀疏 replan，接触段需加密；deploy 不跑 video diffusion（C1 已确认） |

**Interaction-centric WAM 不是否定 FastWAM**，而是在其 MoT（video + action expert）之上，改数据原子、加触觉、解耦时间窗、闭环吃失败。

---

## 3. 三层设计

### 3.1 数据层

```text
         ┌─────────────────────────────────────────┐
         │              metadata 层                 │
         │  events.jsonl / outcomes.jsonl / modality │
         └─────────────────┬───────────────────────┘
                           │ 索引 & 切分
     ┌─────────────────────┼─────────────────────┐
     ▼                     ▼                     ▼
  RGB (慢/长上下文)    tactile (快/接触真值)    proprio + action
     │                     │                     │
     └──────────► interaction event 片段 ◄────────┘
                  （替代均匀 clip）
```

| 组件 | 作用 | 与 clip 方案的区别 |
|------|------|-------------------|
| **触觉** | 腕部/指尖力、接触布尔、滑移特征 | 接触期主信号；与 RGB 不同采样率 |
| **Interaction event** | 子任务边界：approach → grasp → pump → spray → release | 训练样本按事件切，而非 `global_sample_stride` 均匀扫 |
| **Failure data** | 真机失败 rollout、超时、子任务未完成 | 单独目录 + `outcomes.jsonl`；再训练时过采样 |
| **Metadata** | 阶段标签、事件类型、成败、对齐质量、机器人版本 | 驱动 event dataloader、分阶段 eval、failure-aware 采样 |

**目标目录布局：**

```text
data/<task>/
├── meta/
│   ├── modality.json
│   ├── events.jsonl
│   ├── outcomes.jsonl
│   ├── stats.json
│   └── relative_stats.json
├── videos/
├── tactile/
├── data/
├── filtered_out/
└── failure/
```

**spray_water 事件类型（初版）：** `approach_bottle` · `grasp` · `pump` · `aim_spray` · `spray` · `release`

### 3.2 架构层

```text
  obs: RGB [T_v long] ──────────────► Video DiT ──► 事件尺度短时未来（deploy 可关）
  obs: tactile [T_t short] ──► Tactile encoder ──┐
  obs: proprio [T_s]       ──► proprio token  ──┼──► Action DiT ──► action [H_a short]
  meta: phase / event id   ──► context token  ──┘
        ▲
        └── T_v > T_t ≈ H_a  （解耦时间窗）
```

| 模块 | 设计要点 | 里程碑 |
|------|----------|--------|
| **触觉模块** | 1D conv / MLP → token，拼入 context 或 action cross-attn | M1: 触觉+proprio 预测接触期 action |
| **Video–Action 解耦** | `num_obs_steps_video` ≠ `action_horizon` | M2: 长 RGB + 短 action chunk |
| **事件条件** | phase embedding 注入 context | M3: 分 phase 开环 L1 下降 |

### 3.3 推理与训练闭环

```text
  训练 v_k → 真机测试 → 日志+failure → 数据整理 → 微调/重训 v_{k+1}
```

| 机制 | 说明 |
|------|------|
| **Adaptive inference** | `IDLE` 稀疏 replan → `CONTACT` 密集 replan；触觉阈值或 event detector 触发 |
| **Train–test–failure retrain** | 真机 batch → `failure/` + `outcomes.jsonl` → failure 过采样微调 |
| **评估** | episode L1 + **per-event** / **per-phase** / **failure 子集**（`scripts/openloop/`） |

---

## 4. 研究阶段（Phases）

| Phase | 名称 | 状态 | 交付物 |
|-------|------|------|--------|
| **P0** | FastWAM + spray_water 基线 | ✅ 进行中 | 数据 A/B、open-loop（`evaluate_results/openloop_episode_gr00tstyle/`）、ZMQ deploy |
| **P1** | Event metadata & 切分 | 🔲 计划 | `events.jsonl`、标注工具、event dataloader POC |
| **P2** | Failure 数据闭环 | 🔲 计划 | failure 日志、`failure/`、outcome 标注 |
| **P3** | 触觉模态 | 🔲 计划 | 触觉对齐、Tactile encoder v0 |
| **P4** | Video–Action 解耦 | 🔲 计划 | 独立 horizon config |
| **P5** | Adaptive inference + 再训练 | 🔲 计划 | deploy 状态机、failure-aware FT |

---

## 5. 实验设计（Experiment Matrix）

### 5.1 已完成 / 进行中

| ID | 实验 | 对比 | 指标 | 结果位置 |
|----|------|------|------|----------|
| **E0a** | 数据处理基线 A | 绝对 action + z-score | open-loop L1 | `spray_water_rot6d_rosbag_ts_filter_*` |
| **E0b** | 数据处理基线 B | GR00T 相对 + min/max + clip | open-loop L1 | `evaluate_results/openloop_episode_gr00tstyle/` |
| **E0c** | ckpt 缩放律 | step 5k–30k | pred 互相关 ~0.97；5k→30k L1 ↓26% | `step_*/gt_window/.../episode_000000_raw_action_series.npz` |

**E0c 结论：** 训练有改善，但各 ckpt 预测曲线视觉相似——需 **事件级指标**（E1）才能看出接触段是否真在变好。

### 5.2 计划实验

| ID | 假设 | 变量 | 主指标 | Phase |
|----|------|------|--------|-------|
| **E1** | 事件切分优于均匀 clip | `event` vs `frame_stride=32` | contact-phase L1 | P1 |
| **E2** | 触觉降低接触期误差 | +tactile vs 无 | contact 内 L1 | P3 |
| **E3** | 解耦时间窗更准更省 | `T_v=65, H_a=8` vs 33 | contact L1 + 延迟 | P4 |
| **E4** | 失败再训练降失败率 | base vs +failure FT | 子任务成功率 | P2+P5 |
| **E5** | 自适应 replan 更优 | adaptive vs stride=32 | 成功率 vs replan 次数 | P5 |
| **E6** | 事件条件 context | +phase token vs 无 | per-event L1 | P1+P4 |

**消融顺序：** E0 → **E1** → E2 → E3；E4/E5 与 P2/P5 并行。

---

## 6. Schedule（2026 Q3 草案）

| 周次 | 日期（约） | 里程碑 | 实验 |
|------|------------|--------|------|
| W1 | 06/30 – 07/06 | P0 收尾：B 基线 ckpt 30k、open-loop 协议 | E0 结题 |
| W2 | 07/07 – 07/13 | P1：`events.jsonl` + 标注工具 | — |
| W3 | 07/14 – 07/20 | P1：event dataloader + **E1** | E1 |
| W4 | 07/21 – 07/27 | P2：真机 failure 日志首批 | — |
| W5 | 07/28 – 08/03 | P2 outcomes；P3 触觉对齐脚本 | — |
| W6 | 08/04 – 08/10 | P3：触觉进 LeRobot + encoder v0 | **E2** |
| W7 | 08/11 – 08/17 | P4：解耦 config | **E3** |
| W8 | 08/18 – 08/24 | P5：adaptive replan 原型 | **E5** |
| W9 | 08/25 – 08/31 | P5：failure-aware FT | **E4** |
| W10 | 09/01 – 09/07 | 组合验收 + 阶段报告 | E1–E6 |

---

## 7. 仓库内代码地图

```text
FastWAM/
├── README.md                 # 本文件（研究规划）
├── docs/FASTWAM_UPSTREAM.md  # 上游官方安装 / LIBERO / RoboTwin
├── configs/                  # Hydra 配置
├── src/fastwam/              # 核心模型（将扩展触觉、解耦 horizon）
├── scripts/
│   ├── train.py              # 训练入口
│   ├── 1/                    # 真机 ZMQ 部署
│   ├── openloop/             # 开环评估（E0/E1）
│   └── diagnose/             # sim–real 诊断
├── data/                     # 数据集
├── runs/                     # 训练输出
└── evaluate_results/         # 开环与评估结果
```

| 路径 | 角色 |
|------|------|
| `src/fastwam/` | 模型；未来 + tactile encoder、解耦 horizon |
| `configs/data/` | 数据管线；未来 `*_events.yaml`、`*_tactile.yaml` |
| `scripts/1/` | 真机 deploy；未来 adaptive replan + failure dump |
| `scripts/openloop/` | 开环评估（E0/E1） |
| `evaluate_results/openloop_episode_gr00tstyle/` | gr00tstyle 开环结果 |

---

## 8. 致谢与引用

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
| 2026-06-29 | 研究规划作为主 README；开环评估迁至 `scripts/openloop/`；`archive/`、`paper/` 不再纳入 git |
