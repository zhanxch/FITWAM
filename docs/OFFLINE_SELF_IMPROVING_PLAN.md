# Offline Self-Improving：设计与实验

## 当前目标

本阶段交付两项内容：

1. 使用 [`EveRobot`](./EVEROBOT_FORMAT.md) 记录 rollout，并由 state-line score 生成可追溯的候选 interaction window 和 soft weight；
2. 验证局部失败动作能否为 Fast-WAM 学出一个 success-directed steer。

本阶段检验的命题为：

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

首轮实现使用 augmented InfoNCE 训练 Teacher（同一 action window 的两次增强作为正对，其他 window 作为负对，temperature 为 0.07）。Teacher 在独立验证集上选择 checkpoint 后冻结；Student pair loss 使用 fixed-margin objective，使 Student 靠近成功 target 并与失败 target 保持 margin。Masked-action reconstruction 只作为表示塌缩时的预设消融，不默认加入主方法。

Observation Student `G` 只读取部署时可获得的 observation history、task 和 proprioception，预测 action-side steer：

```text
S = G(observation history, task, proprio)
L_pair = w * [distance(S, stopgrad(z+)) + margin_away(S, stopgrad(z-))]
```

`stopgrad` 表示 Teacher target 不接收 Student 阶段的反向梯度。Steer 接口借鉴 Sparsh-X 用少量 bottleneck token 压缩交互信息的思路，主实验先使用一个 action-side token；多个 token 只在单 token 有效后比较。Steer 通过零初始化 residual projection 注入 Action Expert，使新增分支从零修正量开始学习。基础 Action Expert 与 B0/B1 使用相同训练规则：failure sample 不接收直接的 action imitation objective；failure video loss 是否通过联合计算间接产生 Action Expert 参数梯度，必须用 failure-only backward 单独审计。

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

所有方案从同一个 success-only source checkpoint `S0` 继续训练。`B0` 是从 S0 开始的
等预算 success-only continuation，两者不能混用。基础 success 数据、样本预算、
normalization、双视角、proprio、训练步数和 rollout seed 保持一致；B0 的 auxiliary
budget 使用 success 样本匹配，B1/C/M 使用 failure event 样本。

| 方案 | 改动 | 作用 |
|---|---|---|
| B0：success-only | success-only continuation，不加入 failure update | 控制继续训练本身的影响 |
| B1：failure video | 加 failure video，失败动作 loss 为零 | 隔离已有 failure-data 效果 |
| T：token only（条件诊断） | B1 + 同规模 residual branch + 固定可学习 token | 排除参数量和静态 residual bias |
| C：residual only | B1 + Student + residual steer，只用 success action loss，不训练 Teacher/pair loss | 排除 observation-conditioned residual 架构本身的收益 |
| M：contrastive steer | B1 + frozen Teacher + Student + residual steer | 主方法 |
| M-pair-shuffle | M，但跨 episode 打乱 success/failure pairing，保持窗口与权重边际不变 | 检查 pair 对齐信息是否真正有用 |

Water Plant 的 B0/B1/C/M 单 seed 训练与 200-seed checkpoint screening 已完成。
M-pair-shuffle 是主因果对照。T 只在 learned/bypass/shuffle、C 和 pair-shuffle 仍无法
排除静态 residual bias 时训练，不进入默认主路径。

当前已执行的 M 使用 `W-state`：一个 failure episode 可以产生多个 state-line
candidate；每个 candidate 在 `core_start_frame` 附近恰好读取一个 33-frame window。它不是整段
failure trajectory。后续数据消融比较：

| 数据版本 | 定义 |
|---|---|
| W-state | 当前方法；保留所有通过质量门的 state-line candidates |
| W-tail-state | 用 periodic-tail cutoff 限制 episode，再复用冻结的 W-state 后处理器生成 candidates |
| W-main-tail | 分支原型；截尾后改用单阈值选择一个 primary state-line transition candidate |
| W-shuffle | 保持窗口长度和 episode 分布，随机平移 candidate |

主数据消融使用 `W-tail-state`，只隔离 periodic-tail 清洗；`W-main-tail` 同时改变平滑、
阈值和 candidate selection，只作为后续 composite ablation。先人工查看至少 20 个差异
最大的 episode；只有 selected-window overlap 明显变化时才训练 B1-tail 和 M-tail。
两者必须克隆当前 v0.2 manifest，仅替换 failure auxiliary units，保持 success identity、
2+2 batch composition 和 `core_start_anchor` 不变。M-tail 从同一 S0 重新训练，并重建
对应的 pair 和 frozen Teacher target，不能复用 W-state 的 M checkpoint。正式比较同时
报告 `B1-tail-B1-state`、`M-tail-M-state`、`M-tail-B1-tail` 以及前两项的
difference-in-differences，避免把数据清洗收益误记为 steer 收益。分支原型的 manifest
builder 不直接用于正式训练。

模型侧 event-score auxiliary prediction 不进入首轮 M。最终只需人工复核一小批 held-out episode，报告 boundary precision/recall、segment IoU 和候选数量误差，并补充 manual window 对照；训练流程不依赖逐 episode 人工切分。

### 固定协议

- front + wrist，23-d proprioception；
- 所有 continuation run 使用同一训练预算；
- 每个 micro-batch 固定相同数量的 success action sample 和 auxiliary video sample，避免加入 failure 后减少 success action update；
- text context 中不出现 failure phrase；
- state-line 的 normalization、smoothing、threshold、最短长度和 merge gap 只在 train/validation split 确定，并写入 EveRobot provenance；
- inference smoke 跑 50 个 paired rollout，确认结果跑 200 个；
- E0 已作为 development set；fresh E1 固定比较共同的 step 6000，不再使用与闭环表现不稳定对齐的 variant-specific `val_base_loss` 选择 checkpoint；
- 额外 training seeds 同样固定比较 step 6000；validation 只用于训练健康诊断，不能看 closed-loop test seeds 选择 checkpoint；
- rollout collection seeds、开发评估 seeds 和最终确认 seeds 必须两两不重叠；
- 最终 B1/M 使用预声明的 3 个 training seeds；B0 不在额外 seeds 上重复；
- 报告精确成功数、相对 B1 的 paired improvement、置信区间和 failure-mode 变化。

历史 `70/100`、`82/100`、`151/200`、`163/200` 和 state-line action-prefix `183/200` 只作为 motivation，因为训练和 rollout protocol 并未全部统一。`183/200` 也不提供 event boundary 或 score-prediction accuracy，不能作为自动切分结果。

## Gate

1. **实现正确：** failure text 不进模型；failure sample 的直接 action loss 为零；Teacher 先冻结再训练 Student；Teacher/Student/residual branch 的梯度路径符合设计；推理接口没有 outcome。
2. **Water Plant 复核：** E1 上 step-6000 的 M 相对 B1 为 `+2.5pp`，95% CI 包含 0。Strict E2 上同一初始化的 M 为 `58.5%`，低于 B1 的 `82.0%`，该 gate 已明确失败。
3. **Steer 因果性：** E1 learned steer 明显优于 bypass，但不优于 cross-episode shuffled steer。Strict E2 中 M 比同初始化 M-pair-shuffle 低 `15.0pp`，95% CI 为 `[-23.0pp, -7.0pp]`。当前 pair supervision 不能作为主方法继续扩展。
4. **数据冻结：** strict E2 或 multi-round 前完成 W-state/W-tail-state 审计。只有发现系统性无效尾段时才训练四臂并切换 extractor；否则 W-state 冻结到 multi-round 完成。
5. **自改进：** R0->R1 的 new-data continuation 必须同时优于 no-update 和 old-data-extra-step；只有 R1->R2 再次提高才称为 multi-round。
6. **加固：** Water Plant 完成 R1->R2 后复现 Hammer Nail，并分阶段增加 training seeds；
   若 Water Plant 无法继续采到足够 failure，则提前把迭代闭环迁移到 Hammer Nail。

## 当前证据

Water Plant 的 B0、B1、C、M 已从同一 S0 continuation。单个 training seed、200 个
paired simulator seeds 的结果如下：

| Checkpoint | B0 | B1 | C | M |
|---|---:|---:|---:|---:|
| step 5000 | 82.0% | 84.0% | 74.5% | **88.0%** |
| step 6000 | 70.5% | 75.5% | 75.5% | **84.5%** |
| step 6500 | 79.5% | 87.0% | 74.5% | **87.5%** |
| validation-best | 72.5% | 80.5% | 74.5% | **88.0%** |

M 相对 B1 在 step 5000、6000、6500 和 validation-best 分别为 `+4.0pp`、
`+9.0pp`、`+0.5pp` 和 `+7.5pp`。step 6000 与 validation-best 的 paired interval
为正，step 6500 基本持平。现有证据是“工程 pipeline 已闭环并出现 checkpoint-sensitive
positive signal”；steer 因果性与稳定增益尚未证明。

这里的 validation-best 由 `val_base_loss` 选择。该指标与闭环成功率并未稳定对齐：
B1 validation-best 为 80.5%，而 B1 step 6500 为 87.0%。因此它不作为 E1 primary
checkpoint rule；E1 固定使用两侧共同的 step 6000，并完整保留 E0 的其他 checkpoint
结果作为开发证据。

训练数据 collection seeds 为 `20260718..20260917`；当前 200 次评估使用
`20261000..20261199`，两者没有重叠。因此现有结果不存在直接的 rollout-seed
训练泄漏。由于多个 checkpoint 已在这组评估 seeds 上被比较，它从此只作为开发评估集。
E1 确认 seeds 为 `20262000..20262199`，已经完成并冻结为只读评估集。

机器可读统计位于
[`results/dexjoco_water_plant_offline_v1/`](../results/dexjoco_water_plant_offline_v1/)。
现有公开 CSV 只含汇总统计；提交前还需导出不含私人路径的 seed-level
outcomes、checkpoint hashes、discordant counts 和 bootstrap seed，使 paired statistics
可独立复算。

E1 fresh seeds `20262000..20262199` 上，B1-6000 为 `170/200 = 85.0%`，
M-6000 为 `175/200 = 87.5%`。Paired delta 为 `+2.5pp`，95% paired bootstrap CI
为 `[-3.5pp, +8.5pp]`，exact McNemar `p=0.511`；M-only success 为 21 条，
B1-only success 为 16 条。该结果没有达到预注册的 `+4pp` gate，且区间包含 0，
因此 E0 的 `+9pp` 没有在 fresh seeds 上复现。当前结论仍是 pipeline 可运行和方向为正，
不是稳定增益或 steer 因果性已经成立。同 seed S0 已完成 `150/200 = 75.0%`；B1-S0
为 `+10.0pp`，95% paired CI `[+2.5pp, +17.5pp]`，McNemar `p=0.0169`；M-S0
为 `+12.5pp`，CI `[+5.5pp, +19.5pp]`，`p=0.00126`。整体 continuation 有效，
但 M 相对 B1 的额外机制增益仍未建立。

zxc 的 soft-event failure-aware 候选在其私有目录报告了两个 200-episode repeat，合计
`356/400 = 89.0%`。当前账号无法读取其 config、seed ledger、baseline、训练数据和
round state，因此该数字只作为并行候选证据，不与 E1 合并，也不作为当前方法选择依据。

Strict E2 使用 seeds `20262200..20262399`、共同的 step-6000 比较规则和
200 个 paired rollout：

| Arm | Success |
|---|---:|
| S0 | `154/200 = 77.0%` |
| B1 | `164/200 = 82.0%` |
| M strict | `117/200 = 58.5%` |
| M pair-shuffle | `147/200 = 73.5%` |

M strict 相对 B1 为 `-23.5pp`（95% CI `[-31.5pp, -15.5pp]`），相对
M pair-shuffle 为 `-15.0pp`（95% CI `[-23.0pp, -7.0pp]`）。这不是新鲜 seed 上的
不显著波动，而是当前 Teacher/pair objective 在严格对照中的负面结果。
当前有效证据收缩为：B1 failure-aware continuation 值得保留；旧 M 的
Teacher/pair 设计不能作为最终 steer 方案。

## 下一阶段

1. **完成 C 诊断：** strict C 使用与 strict M 相同的 common init，但关闭 pair loss。
   它只用来判断负面结果主要来自 Student/residual 架构，还是 Teacher/pair objective；
   不按结果挑 checkpoint，固定评估 step 6000。
2. **冻结旧 M：** 保留 E0/E1/E2 为完整消融，不再向旧 Teacher/pair 方案追加
   training seed、新任务或 multi-round。
3. **独立复现 soft-event/value heads：** 从 `base-20260720` 按模块移植，不整分支合并。
   先复现 soft-event head-only 和 value head-only，再测两者合并；每个 head 保留独立
   loss、gradient scale 和 shuffled-label control。zxc 的 `356/400` 只用于确定这一优先级，
   本分支仍按独立 seed ledger 复现。
4. **重新设计 steer：** 不把两个 head 直接叠到已失败的 M 上。先比较输入端单次
   residual 与 layer-wise zero-init residual；失败窗口继续禁用 action imitation。
   Teacher 只有在能提供可验证的 corrective target 后才恢复为主方法，否则保留为失败消融。
5. **R0->R1 core pilot：** 新 steer 在 fresh-seed efficacy 和 causal control 上通过后才收集新的 deployment experience，比较
   no-update、old-data-extra-step 和 new-data continuation。该 pilot 先于 Hammer Nail 和
   额外 training seeds；若通过，再补 B1 old/new 数据臂做归因。
6. **R1->R2：** 使用 R1 policy 收集下一轮数据并重复同一控制。只有两次连续更新都提高，
   才使用 multi-round self-improving 表述。
7. **Cross-task + seeds：** 随后先做 Hammer Nail B1/新 steer，再训练 Water Plant seed 43/44。
   两个额外 seed 都预注册，只重训 B1/M，不重复 B0；seed 43 若暴露实现问题可先诊断，
   但不能因结果不利而取消或替换 seed 44。seed 43 静态复现通过后，先复现一次 R0->R1，
   再执行 seed 44。
8. **Conditional work：** T 和正式 tail 四臂只在因果结果含混或数据审计显示实质问题时
   执行。Offline RL/value objective 放在 supervised multi-round 饱和以后。

## 时间

默认资源是一组 4xA100。临时增加的 GPU 不计入实验排期。历史速度下 6.5k step 约 14-16 小时；第一轮先用 500-step benchmark 校准。

| 时间 | 工作 |
|---|---|
| Day 1 | 完成 S0、实现并 smoke steer intervention；冻结统一初始化 |
| Day 2-3 | 完成 E1 bypass/shuffled-steer 因果消融 |
| Day 4-7 | 从共同初始化重训 M/M-pair-shuffle 并在 E2 评估 |
| Week 2 | Water Plant R0->R1 core pilot；通过后补 B1 attribution arms |
| Week 3 | R1->R2，或在 R0->R1 未通过时诊断并停止扩展 |
| Week 4 | Hammer Nail B1/M；随后开始 Water Plant seed 43/44 加固 |

Fresh-seed 复核约一到两天；新增一个 6500-step 四卡训练约 14-16 小时。默认只有一条
4xA100 lane，不在排期中假设训练并行。前两周先回答 causal steer 与第一轮
self-improvement，跨任务和多 training-seed 证据随后补齐。

正式 run 从 step 0 同时记录 W&B 和本地 JSONL/CSV，并保存 EveRobot manifest hash、commit/config、checkpoint、实际 success/failure 采样比例和 stop reason。
