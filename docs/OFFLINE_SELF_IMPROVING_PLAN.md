# Offline Self-Improving：方法与实验

## 研究范围

本文档定义 FITWAM 的 offline self-improving 方法与受控实验：利用 rollout 中同一交互阶段的成功/失败动作事件训练 steer token，并与 success replay 和 failure-video baseline 比较。Online RL、test-time world-model update 和 tactile 属于整篇工作的其他模块，不进入本实验。

Offline steer 要求 EveRobot 提供冻结且经过人工复核的 event manifest、数据 provenance 和 episode-disjoint split。自动 event 标注由 EveRobot 数据模块独立定义和验证，不属于 offline steer 的实验变量，也不阻塞该实验。

论文主张限定为：

> Phase-matched failure events provide training-only negative action supervision for an observation-conditioned steer. Failed actions are never imitation targets, and deployment needs neither failure trajectories nor outcome labels.

## 主方法：trajectory teacher，observation student

训练时，用同一 task phase 的 success/failure action event 构造配对。共享 trajectory encoder `E` 将归一化 action chunk 和有效帧 mask 压缩成一个 bottleneck token：

```text
z+ = E(a+_1:H, mask+)        z- = E(a-_1:H, mask-)
```

`E` 是训练期 teacher。一个 observation-conditioned student `G` 从当前首帧双视角 token、task text 和 proprio 预测可部署 steer：

```text
s = G(front, wrist, task, proprio)
observation + task + proprio + s -> Action Expert -> action
```

`z+`、`z-` 只作为训练目标，不进入 policy forward。Pair loss 把 `s` 拉向同阶段成功动作 `z+`，并推离配对失败动作 `z-` 和 batch 内失败负样本；teacher target 在该项中 stop-gradient，避免 student/teacher 共同塌缩。Trajectory encoder 另用 phase-conditioned supervised contrastive loss 学习动作事件结构。

```text
L = L_video(all)
  + L_action(success only)
  + lambda_ctr * L_action_contrast
  + lambda_pair * L_steer_pair
```

Failure sample 的直接 action flow-matching loss 必须严格为零。失败动作只进入 training-only trajectory teacher；student token 只接 Action Expert。首版使用 `B=1`，只有在单 token 已有稳定增益后才增加 bottleneck 数量。

`S = Z+ + alpha(Z+ - Z-)` 不作为主方法：逐 pair 计算会在推理时依赖未来轨迹，全局平均后又退化成静态 soft prompt。它只保留为 prototype-delta ablation。

Sparsh-X 说明少量 bottleneck token 可以压缩多源交互信息；它不直接给出 success/failure steer 的学习规则。实现必须包含 observation-conditioned steer、仅面向 Action Expert 的 token 注入、完整 checkpoint save/load，以及不依赖 outcome context 的部署路径。

## 数据与控制变量

固定 source checkpoint 采 200 条 rollout，成功和失败都来自同一轮、同一 policy 与同一环境协议。先按 episode 和 environment seed 固定 train/validation，再切 event 和做 pair mining，避免同一 episode 的窗口跨 split。人工标出 phase 与 failure boundary；success/failure pair 先限制在同一 task phase，再按 event 起点的 proprio 与冻结视觉特征做近邻匹配，不能使用 outcome frame。

Event manifest 在 split、pair mining 和模型拟合前冻结。Teacher fitting、pair/prototype 构造和 action normalization 只使用 train episode。Student 的输入截止到当前控制时刻；outcome label、failure text、post-outcome frame 和 teacher action 都不能进入 student 或 Action Expert。未来视频和动作只进入 training-only teacher 或 loss target，不进入部署路径。

各方案使用相同的原始 success buffer、同一轮 rollout、global batch、optimizer、scheduler、训练步数和 normalization。新增 rollout 数据采用两路采样：success event 提供 video + action loss，failure event 只提供 video 和辅助表示 loss。Contrastive batch 按 phase/outcome 分层，保证每个有效 anchor 至少有一个正样本；无正样本的 anchor 不计入该 loss。B0 使用同计算量的 success replay，避免把“更多更新步数”误当成 failure 收益。

## 实验矩阵

| 方案 | 训练信号 | 目的 |
|---|---|---|
| S0 | 冻结 source checkpoint | rollout 数据来源与零更新参考 |
| B0 | same-round success replay | compute-matched continuation baseline |
| B1 | B0 + failure video；failure action loss=0 | failure-world-representation baseline |
| T | B1 + observation student；只由普通 success action loss 端到端训练 | 控制 student/token 参数量 |
| C | B1 + action trajectory contrast；steer 不进 Action Expert | 控制辅助 loss |
| M | B1 + trajectory teacher + observation student | 主方法 |
| M-shuffle | M，但 phase 内打乱 success/failure 配对 | 检查语义配对是否真实有效 |

先做 Water Plant。B0/B1/T/C/M 各做 3 个 training seed；M-shuffle 只做 seed 0 的机制检查。单 seed 机制通过后，再只复现 B0/B1/M 到 Hammer Nail。Fold Glasses 是可选第三任务。

### 固定评测协议

- front + wrist，23-d proprio；text 中不出现 failure phrase；
- 显式固定 `max_steps=6500`，checkpoint 为 3000/5000/6500；若 6500 的 validation 仍上升，所有存活方案统一扩到 12000；
- 训练、环境和 policy sampling seed 分开记录；相同环境 seed 在方案间 paired；
- 50 个 validation rollout 只用于 checkpoint 选择；独立的 200-episode development rollout 用于机制 gate；所有架构决定冻结后，B0/B1/M 每个 training seed 的选定 checkpoint 各做一次另一组 environment seed 的 200-episode held-out final；
- 最终报告精确成功数、paired improvement、bootstrap CI、exact McNemar、failure category 和表示空间诊断；
- W&B 从启动即记录，同时保存本地 JSONL/CSV、manifest hash、commit/config、采样比例和 stop reason。

已有 `70/100`、`82/100`、`151/200`、`163/200` pilot 使用了不同训练或评测协议，只作为可行性依据，不进入受控主表。

## Gate

正式长跑前必须满足：train/validation manifest 无 episode overlap；同轮同阶段配对可复核；B1 failure-only batch 的 Action Expert gradient 为零且 Video Expert gradient 非零；M 的 trajectory encoder 和 steer gradient 非零；推理不接 outcome/failure text；DDP save/load 后 token 行为一致。

Water Plant seed 0 的 200-episode development gate：`M - B1 >= 4pp`，paired 90% CI 下界大于 0，并且 M 同时超过 T、C 与 M-shuffle。最终 held-out set 不参与架构、checkpoint 或任务扩展决策。3-seed 结论要求 development gain 的 95% CI 下界大于 0、至少 2/3 seed 为正，且任一 seed 不低于 B1 超过 2pp。未通过则不扩任务，先检查 pair quality、表示塌缩和 token 是否真正影响 action。

## 时间与算力

正式实验以独立 validation manifest、same-round phase matching 和 500-step smoke 为启动条件。训练使用两条 4-GPU lane；配对比较固定硬件、软件环境和并行配置，最终闭环主表由同一环境生成。

| 阶段 | 预计时间 |
|---|---:|
| 实现、单测、500-step smoke | 2-3 天 |
| Water Plant 单 seed：B0/B1/T/C/M/M-shuffle + gate rollout | 4-6 天 |
| Water Plant 三 seed 确认 | 5-7 天 |
| Hammer Nail B0/B1/M 三 seed + 最终 200 rollout | 5-7 天 |

完整受控矩阵为 25 个 continuation run：Water Plant 的 B0/B1/T/C/M 各 3 seed（15）、M-shuffle seed 0（1），以及 Hammer Nail 的 B0/B1/M 各 3 seed（9）。按两条 4-GPU lane 约需 3-4 周。若单 seed gate 不通过，停止后续长跑，不用更多 seed 掩盖架构问题。
