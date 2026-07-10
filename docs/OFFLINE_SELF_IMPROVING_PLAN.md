# Offline Self-Improving：设计与实验

## 当前目标

我负责两件事：

1. 完成 [`EveRobot`](./EVEROBOT_FORMAT.md)，让每轮 rollout 和 event 标注可追溯、可取任意子集；
2. 验证局部失败动作能否为 Fast-WAM 学出一个 success-directed steer。

论文主张先收窄为：

> 离线 failure event 能改善闭环动作；失败动作不作为 imitation target，部署时也不需要 failure/outcome 输入。

Online RL、test-time world-model update 和 tactile 都放在这一步通过之后。

## 架构

从同一 task phase 取 success event 和 failure event。一个小型 trajectory encoder 把归一化 action chunk 编码成 `z+`、`z-`，contrastive loss 用它们学习成功/失败 prototype `P+`、`P-`。Action Expert 接收：

```text
S_improve = LayerNorm(P+ + alpha * (P+ - P-))
```

训练和推理始终使用同一个 `S_improve`，推理不传 outcome。

```text
EveRobot event pair -> trajectory encoder -> contrastive loss -> P+, P-
                                                              |
observation + task + proprio -> Fast-WAM Action Expert <- S_improve
failure sample             -> video loss + contrastive loss
success sample             -> video loss + action loss + contrastive loss
```

Failure sample 的**直接 action flow-matching loss 为零**，但仍会通过 video/contrastive loss 和 shared MoT 间接影响 Action Expert；B1 专门控制这条路径。现有 `outcome_encoder` 会把真实 outcome 放进 shared context，因此不作为主方法。

第一版先用人工定位的 failure window。EveRobot 的 soft subtask score 后续作为权重，并负责匹配同阶段的 success/failure event。首轮不加入 weak localizer、task-specific token bank 或大范围调参。

## 实验

所有方案从同一个 success-only checkpoint 继续训练，保持数据、normalization、双视角、proprio、训练步数和 rollout seed 完全一致。

| 方案 | 改动 | 作用 |
|---|---|---|
| B0：success-only | 不加入 failure update | source policy 和参考点 |
| B1：failure video | 加 failure video，失败动作 loss 为零 | 隔离已有 failure-data 效果 |
| T：token only | B1 + 固定可学习 token | 排除参数量和额外 conditioning |
| C：contrast only | B1 + trajectory contrast，推理无 token | 排除辅助表示 loss |
| M：contrastive steer | B1 + contrast + `S_improve` | 主方法 |

先做 Water Plant。M 如果不能超过 B1，就回头改架构；通过后只复现 B0/B1/M 到 Hammer Nail。Fold Glasses 是两任务结果稳定后的可选第三任务。

Auto event score 单独做一个小实验：

```text
manual event window vs auto soft weight vs shuffled weight
```

只有 auto score 与 held-out 人工标注一致，并且不抹掉 Water Plant 的涨点，才进入主方法。

### 固定协议

- front + wrist，23-d proprioception；
- 所有 continuation run 使用同一训练预算；
- text context 中不出现 failure phrase；
- checkpoint gate 跑 50 个 paired rollout，最终结果跑 200 个；
- checkpoint 只用 validation seed 选择，不能看 final test seed；
- 最终 B0/B1/M 使用 3 个 training seed；
- 报告精确成功数、相对 B1 的 paired improvement、置信区间和 failure-mode 变化。

历史 `70/100`、`82/100`、`151/200`、`163/200` 只作为 motivation，因为训练和 rollout protocol 并未全部统一，不能放进新架构的受控主表。

## Gate

1. **实现正确：** failure text 不进模型；failure sample 的直接 action loss 为零；trajectory encoder/steer 有梯度；推理接口没有 outcome。
2. **Water Plant：** 单 seed 的 M 相对 B1 至少提高 4pp，且 paired result 没有明确退化；否则不扩任务。
3. **可复现：** Water Plant 三个 seed 保留增益，再到 Hammer Nail 复现。
4. **自动标注：** auto score 在人工边界上优于 uniform/shuffled，并且不损害 policy gain。

## 时间

AWS 是 8xA100，按两个 4-GPU lane 使用。历史速度下 6.5k step 约 14-16 小时；第一轮先用 500-step benchmark 校准。

| 时间 | 工作 |
|---|---|
| Day 1-2 | 定稿 EveRobot，标 Water Plant failure interval，验证 manifest |
| Day 3-4 | 实现 trajectory encoder、contrastive loss、action-side token 和梯度测试 |
| Day 5-7 | Water Plant B1/T/C/M 训练与 50-rollout gate |
| Day 8-10 | 三 seed 确认，或根据结果修改架构 |
| Week 3 | Hammer Nail 复现和 200-rollout 最终评估 |

单 seed 的机制结论约一周；两任务、三 seed 的论文证据约两到三周。最晚 8 月底冻结 offline 架构和主结果，9 月只做论文整合与必要复跑。

正式 run 从 step 0 同时记录 W&B 和本地 JSONL/CSV，并保存 EveRobot manifest hash、commit/config、checkpoint、实际 success/failure 采样比例和 stop reason。
