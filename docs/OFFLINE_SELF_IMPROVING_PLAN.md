# Offline Self-Improving：设计与实验

## 当前目标

本阶段交付两项内容：

1. 使用 [`EveRobot`](./EVEROBOT_FORMAT.md) 记录 rollout，并由 state-line score 生成可追溯的候选 interaction window 和 soft weight；
2. 验证局部失败动作能否为 Fast-WAM 学出一个 success-directed steer。

论文主张先收窄为：

> 离线 failure event 能改善闭环动作；失败动作不作为 imitation target，部署时也不需要 failure/outcome 输入。

Online RL、test-time world-model update 和 tactile 都放在这一步通过之后。

## State-line 数据路径

State-line score 只表示机器人状态轨迹的局部转折强度，不直接给出语义 subtask 或准确的 failure cause。当前方法将它作为数据侧的候选窗口信号：

```text
raw state-line score
-> mask 无历史差分的起始帧
-> robust normalization + smoothing
-> hysteresis threshold
-> remove short pulses
-> merge short gaps
-> candidate windows + confidence/soft weight
```

候选窗口数量不要求等于真实 subtask 数量。相邻短窗口可以合并，低置信度窗口不进入 pair loss；长窗口在存在持续低谷时再切分，否则保留为一个粗粒度 event。每个 failure episode 内的候选权重归一化，避免噪声峰较多的 episode 被过度采样。

Success/failure window 在同一 task 内按相近的执行进度和 pre-event robot state 配对。无法可靠配对的 failure window 仍可参与 world/video loss，但不参加 success/failure pair loss。语义 phase 标注若后续可用，只作为额外配对约束，不是第一轮训练的前置条件。

历史 state-line action-prefix probe 直接把连续分数拼到 action 第 0 维，没有执行平滑、切窗或 event 数量校正。该结果只作为辅助预测可能有用的动机，不作为自动 event extraction 的证据。模型侧 event-score auxiliary prediction 放在主方法验证之后单独实验。

## 架构

主方法分两个训练阶段。Trajectory Teacher `E` 读取按窗口截取并 padding/mask 的 action trajectory，得到归一化的成功和失败表示：

```text
z+ = normalize(E(success action window, mask))
z- = normalize(E(failure action window, mask))
```

Teacher 使用 normalized SupCon/InfoNCE 或 fixed-margin objective（在单位球面上分离成功和失败表示，避免距离无限增大）。Teacher 在独立验证集上选择 checkpoint 后冻结；masked-action reconstruction 只作为表示塌缩时的预设消融，不默认加入主方法。

Observation Student `G` 只读取部署时可获得的 observation history、task 和 proprioception，预测 action-side steer：

```text
S = G(observation history, task, proprio)
L_pair = w * [distance(S, stopgrad(z+)) + margin_away(S, stopgrad(z-))]
```

`stopgrad` 表示 Teacher target 不接收 Student 阶段的反向梯度。Steer 接口借鉴 Sparsh-X 用少量 bottleneck token 压缩交互信息的思路，主实验先使用一个 action-side token；多个 token 只在单 token 有效后比较。Steer 通过零初始化 residual projection 注入 Action Expert，使新增分支从零修正量开始学习。基础 Action Expert 与 B0/B1 使用相同训练规则：只由 success sample 的 action loss 更新，不接收 failure action imitation loss。

```text
candidate event pair -> frozen Trajectory Teacher -> z+, z-
observation + task + proprio -> Student -> steer token
                                      -> residual projection
                                      -> Action Expert

failure sample -> world/video loss + weighted pair loss
success sample -> world/video loss + action loss + weighted pair loss
```

Failure sample 的直接 action flow-matching loss 为零；失败动作只作为 Teacher 的表示输入和 pair loss 的 negative。Pair loss 更新 Student representation；零初始化 residual projection 由 success action loss 学习如何使用该 representation。Teacher target 和基础 Fast-WAM feature 在 pair-loss 路径上均停止梯度。World Expert 接收成功和失败视频监督。现有 `outcome_encoder` 会把真实 outcome 放进 shared context，因此不作为主方法；推理接口不接收 outcome、未来 action 或 failure text。

## 实验

所有方案从同一个 success-only checkpoint 继续训练，保持数据、normalization、双视角、proprio、训练步数和 rollout seed 完全一致。

| 方案 | 改动 | 作用 |
|---|---|---|
| B0：success-only | 不加入 failure update | source policy 和参考点 |
| B1：failure video | 加 failure video，失败动作 loss 为零 | 隔离已有 failure-data 效果 |
| T：token only | B1 + 同规模 residual branch + 固定可学习 token | 排除参数量和额外 conditioning |
| C：residual only | B1 + Student + residual steer，只用 success action loss，不训练 Teacher/pair loss | 排除 observation-conditioned residual 架构本身的收益 |
| M：contrastive steer | B1 + frozen Teacher + Student + residual steer | 主方法 |
| M-shuffle | M，但打乱 event weight 或 success/failure pairing | 检查 event 对齐和负例结构是否真正有用 |

先做 Water Plant。M 如果不能超过 B1，就回头改架构；通过后只复现 B0/B1/M 到 Hammer Nail。Fold Glasses 是两任务结果稳定后的可选第三任务。

State-line 首先作为数据侧输入，Water Plant 比较：

```text
whole episode vs state-line soft candidates vs shuffled candidates
```

模型侧 event-score auxiliary prediction 不进入首轮 M。最终只需人工复核一小批 held-out episode，报告 boundary precision/recall、segment IoU 和候选数量误差，并补充 manual window 对照；训练流程不依赖逐 episode 人工切分。

### 固定协议

- front + wrist，23-d proprioception；
- 所有 continuation run 使用同一训练预算；
- 每个 micro-batch 固定相同数量的 success action sample 和 auxiliary video sample，避免加入 failure 后减少 success action update；
- text context 中不出现 failure phrase；
- state-line 的 normalization、smoothing、threshold、最短长度和 merge gap 只在 train/validation split 确定，并写入 EveRobot provenance；
- checkpoint gate 跑 50 个 paired rollout，最终结果跑 200 个；
- checkpoint 只用 validation seed 选择，不能看 final test seed；
- 最终 B0/B1/M 使用 3 个 training seed；
- 报告精确成功数、相对 B1 的 paired improvement、置信区间和 failure-mode 变化。

历史 `70/100`、`82/100`、`151/200`、`163/200` 和 state-line action-prefix `183/200` 只作为 motivation，因为训练和 rollout protocol 并未全部统一。`183/200` 也不提供 event boundary 或 score-prediction accuracy，不能作为自动切分结果。

## Gate

1. **实现正确：** failure text 不进模型；failure sample 的直接 action loss 为零；Teacher 先冻结再训练 Student；Teacher/Student/residual branch 的梯度路径符合设计；推理接口没有 outcome。
2. **Water Plant：** 单 seed 的 M 相对 B1 至少提高 4pp，且 paired result 没有明确退化；否则不扩任务。
3. **可复现：** Water Plant 三个 seed 保留增益，再到 Hammer Nail 复现。
4. **State-line：** 候选窗口的数量、长度和覆盖率无异常；soft candidates 优于 whole-episode/shuffled control。人工边界只用于最终质量审计。
5. **Steer：** Teacher embedding 不塌缩；learned steer 优于 zero/shuffled steer，M-shuffle 不应复现 M 的增益。

## 时间

默认资源是一组 4xA100。临时增加的 GPU 不计入实验排期。历史速度下 6.5k step 约 14-16 小时；第一轮先用 500-step benchmark 校准。

| 时间 | 工作 |
|---|---|
| Day 1-2 | 生成并审计 Water Plant state-line candidate manifest，验证 soft weight 数据接口 |
| Day 3-4 | 实现 Teacher、Student、contrastive/pair loss、residual steer 和梯度测试 |
| Day 5-7 | Water Plant B1/T/C/M 训练与 50-rollout gate |
| Day 8-10 | 三 seed 确认，或根据结果修改架构 |
| Week 3 | Hammer Nail 复现和 200-rollout 最终评估 |

单 seed 的机制结论约一周；两任务、三 seed 的论文证据约两到三周。最晚 8 月底冻结 offline 架构和主结果，9 月只做论文整合与必要复跑。

正式 run 从 step 0 同时记录 W&B 和本地 JSONL/CSV，并保存 EveRobot manifest hash、commit/config、checkpoint、实际 success/failure 采样比例和 stop reason。
