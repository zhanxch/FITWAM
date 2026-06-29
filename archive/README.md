# archive/

本目录存放**已移出主仓库活跃路径**的实验代码与配置，目的是：

1. **保持主线与官方 FastWAM 对齐**（`src/fastwam/`、`scripts/train.py`、`experiments/` 等）
2. **保留历史实验**供参考、复现或移植想法，但不参与当前 `spray_water` 真机默认流程

**这不是废弃垃圾堆** — 其中部分工具仍在使用（尤其是 `openloop/`）。

## 子目录

| 目录 | 原用途 | 典型内容 |
|------|--------|----------|
| **`openloop/`** | 自定义开环 / episode 评估 | `run_openloop.py`、`openloop_eval/core.py`；产出在 `evaluate_results/openloop_episode_*`。**当前仍在用。** |
| **`dexjoco/`** | DexJoCo 双臂仿真 + 子任务 | 数据准备、子任务 HTML 标注器、仿真 ZMQ eval、`手册.md` |
| **`egodex/`** | EgoDex 视频预训练 | fps/resize、LeRobot 导出、预训练 task config |
| **`egovla/`** | EgoVLA 仿真数据 | HDF5→LeRobot converter、`egovla_sim` dataset 代码 |
| **`g1/`** | G1 Unihand | 数据准备与 task config |

## 与主线的关系

```text
  官方 FastWAM 路径          本 fork 活跃主线              archive
  ─────────────────        ──────────────────            ─────────
  third_party/FastWAM  →   spray_water 训练/部署      →  其他机器人/数据集实验
                           scripts/1/ ZMQ 真机
                           scripts/diagnose/
```

## Interaction-centric WAM 会放哪？

| 类型 | 位置 |
|------|------|
| 新模型、数据集、dataloader | `src/fastwam/`、`configs/` |
| 真机 deploy、采集脚本 | `scripts/` |
| 仅开环分析、一次性工具 | 可暂放 `archive/openloop/` 或 `scripts/diagnose/` |
| 已结束、不再维护的实验 | 迁入 `archive/<name>/` |

完整研究规划见 [`docs/INTERACTION_CENTRIC_WAM.md`](../docs/INTERACTION_CENTRIC_WAM.md)。
