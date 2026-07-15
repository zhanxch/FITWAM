# Offline Self-Improving：方法与实验

## 研究范围

本文档定义 FITWAM 的 offline self-improving 方法与受控实验：利用 rollout（策略在环境中闭环执行后保存的一条 episode）中同一交互阶段的成功/失败动作事件训练 steer token，并与 success replay 和 failure-video baseline 比较。Online RL、test-time world-model update 和 tactile 属于整篇工作的其他模块，不进入本实验。

Offline steer 要求 EveRobot 提供冻结且经过人工复核的 event manifest（记录 episode、交互阶段、起止帧和 loss 策略的索引文件）、数据 provenance（数据来自哪一轮、哪个 checkpoint 和哪份配置）和 episode-disjoint split（同一 episode 的任何窗口只能属于一个数据划分）。自动 event 标注由 EveRobot 数据模块独立定义和验证，不属于 offline steer 的实验变量，也不阻塞该实验。

论文主张限定为：

> Phase-matched failure events provide training-only negative action supervision for an observation-conditioned steer. Failed actions are never imitation targets, and deployment needs neither failure trajectories nor outcome labels.

## 主方法：trajectory teacher（轨迹教师），observation student（观测学生）

训练分为两个阶段。第一阶段只训练 trajectory teacher：用同一 task phase（任务中的同一种交互阶段，例如 reach、grasp 或 pour）的 success/failure action event 构造配对，共享 trajectory encoder `E` 将归一化 action chunk（一段连续动作序列）和有效帧 mask（标记真实帧，避免 padding 参与编码）压缩成一个 bottleneck token（用一个固定长度向量概括整段动作）：

```text
z+ = E(a+_1:H, mask+)        z- = E(a-_1:H, mask-)
```

`E` 用 phase-conditioned supervised contrastive loss（同阶段成功动作聚近、失败动作推远的有监督对比损失）训练，并按 held-out event 的同阶段 success/failure 检索与表示分离度选择 checkpoint。第二阶段冻结 `E`，由 observation-conditioned student `G`（只根据当前可观测信息预测 steer 的小网络）从当前决策时刻的双视角 token、task text 和 proprio 预测可部署 steer；它不能读取未来动作、failure label 或最终 outcome：

```text
s = G(front, wrist, task, proprio)
observation + task + proprio + s -> Action Expert -> action
```

`z+`、`z-` 只作为训练目标，不进入 policy forward（policy 真正生成动作的前向计算）。每个 observation anchor（当前决策状态；重点覆盖 failure event 中错误动作开始前的状态）匹配同阶段、近状态的成功动作 `z+` 和失败动作 `z-`。Pair loss（成对表示损失）把 `s` 拉向 `z+`，并推离 `z-` 和 batch 内失败负样本；冻结 teacher 后目标固定，student 不会追逐每步变化的表示。Success anchor 同时接受正常 action loss，使 Action Expert 学会使用 steer；failure anchor 的直接 action loss 保持为零。

```text
Stage A: L_teacher = lambda_ctr * L_action_contrast

Stage B: L_policy = L_video(all)
                  + L_action(success only)
                  + lambda_pair * L_steer_pair
```

Failure sample 的直接 action flow-matching loss（Action Expert 针对该条已记录动作的直接生成监督）必须严格为零；否则模型会把有缺陷的失败动作当作 imitation target 进行模仿。原版 Fast-WAM 的 structured attention mask 允许 future-video token 在 Video Expert 内双向交互，并允许 Action Expert 读取干净首帧；video query 不读取 Action Expert token。Base 配置的 `action_conditioned=false`，所以 B1 的失败动作不会作为 Video Expert 的条件输入：B1 只保留失败视频的 world-modeling loss，M 额外把失败动作交给冻结的 trajectory teacher 作为负例。Failure-only batch 应验证 Action Expert 专属 Q/K/V、FFN 和输出头无直接梯度，Video Expert 有梯度；Vanilla 中被 video/action 共用的 proprio condition encoder 允许由 failure video loss 更新，必须作为独立参数组记录。Video Expert 和共享条件表示的变化仍可通过首帧 K/V 间接改变动作，这正是 B1 控制的 failure-video 表征效应。失败 teacher token `z-` 不进入 policy forward，部署时只有 observation student 生成的 `s` 接入 Action Expert。首版使用 `K=1`（每段轨迹只压缩为一个 token），只有在单 token 已有稳定增益后才增加 bottleneck 数量。

`S = Z+ + alpha(Z+ - Z-)` 不作为主方法：逐 pair 计算会在推理时依赖未来轨迹，全局平均后又退化成静态 soft prompt（对所有状态都使用同一个可学习提示）。它只保留为 prototype-delta ablation（检验“成功原型减失败原型”方向是否有用的对照实验）。

Sparsh-X 用固定的 `K=4` bottleneck token 做多触觉模态的信息交换，但没有比较单 token 与多 token，也没有验证 success/failure action steer。它只支持“少量 latent 可以压缩跨源信息”这一设计动机。若后续使用 `K>1`，trajectory teacher 和 observation student 都输出 `K x d` token，并额外解决 token slot 对齐或集合匹配；该扩展必须单独消融。实现必须包含 observation-conditioned steer、仅面向 Action Expert 的 token 注入、完整 checkpoint save/load，以及不依赖 outcome context 的部署路径。

## 数据与控制变量

固定 source checkpoint 采 200 条 rollout，成功和失败都来自同一轮、同一 policy 与同一环境协议。当前 Water Plant 的 `raw` 与 `collect_shard0-3` 是同一批 200 条 rollout 的合并版和分片版，只保留其中一种；`last_8s/trimmed` 是同一批失败轨迹的派生 event window，不计为新 episode。先按 episode 和 environment seed 固定 train/validation，再切 event 和做 pair mining（从同阶段样本中寻找状态最接近的成功/失败配对），避免重复表示或同一 episode 的窗口跨 split。人工标出 phase 与 failure boundary；success/failure pair 先限制在同一 task phase，再按当前决策时刻的 proprio 与冻结视觉特征（不随本实验更新的视觉表示）做近邻匹配，不能使用 outcome frame（已经暴露最终成功或失败结果的画面）。

Event manifest 在 split、pair mining 和模型拟合前冻结。Teacher fitting、pair/prototype 构造和 action normalization 只使用 train episode。Student 的输入截止到当前控制时刻；outcome label、failure text、post-outcome frame 和 teacher action 都不能进入 student 或 Action Expert，避免 label leakage（训练时偷看部署时不存在的信息）。未来视频和动作只进入 training-only teacher 或 loss target，不进入部署路径。

各方案使用相同的原始 success buffer、同一轮 rollout、global batch、optimizer、scheduler、训练步数和 normalization。Stage A 的 contrastive batch 按 phase/outcome 分层，保证每个有效 anchor（当前作为比较中心的样本）至少有一个正样本；无正样本的 anchor 不计入 teacher loss。Stage B 采用两路采样：success event 提供 video + action + pair loss，failure event 只提供 video + pair loss。B0 使用同计算量的 success replay，避免把“更多更新步数”误当成 failure 收益。

### 没有人工切分时

主方法需要 event 边界，但不要求把原始 LeRobot episode 物理裁成多个新视频；EveRobot 可以在完整 episode 上用 `start_frame/end_frame` 选择训练窗口。数据尚未切分时按以下顺序处理：

1. **主实验路径：** 用人工标注，或先由 Switch/state-line score 提议候选边界，再由人复核边界并赋予 task phase；复核后的 manifest 冻结后即可进入主实验。Switch score 只能提示“这里可能发生了动作变化”，不能单独判断语义 subtask。
2. **兼容性 smoke：** 可以把整条 episode 视为一个 event，再统一采样、截断或 padding；它只验证 loader、loss 和 checkpoint pipeline，不能支持“局部 interaction failure steer”的论文主张。
3. **全自动划分：** 未经人工复核的 Switch score 属于 EveRobot 自动标注实验；在边界和 phase 质量通过独立评估前，不进入 offline steer 主表。

## 实验矩阵

| 方案 | 训练信号 | 目的 |
|---|---|---|
| S0 | 冻结 source checkpoint | rollout 数据来源与零更新参考 |
| B0 | 从 S0 独立启动；same-round success replay | 控制 continuation 与训练预算 |
| B1 | 从 S0 独立启动；使用与 B0 相同的 success replay 并加入 failure video；不加载 B0 权重或 optimizer state；failure action loss=0 | 隔离失败视频表征的收益或伤害 |
| T | 从 S0 独立启动；B1 + observation student/token，只由普通 success action loss 训练 | 控制新增 student/token 的参数量与条件通路 |
| C | 对 M 的 Stage A 冻结 teacher 做离线诊断；steer 不进 Action Expert | 用检索、分离度和塌缩诊断验证 action contrast，不定义额外 policy 训练或闭环 rollout |
| M | 从 S0 独立启动；B1 + 冻结的 trajectory teacher + observation student | 主方法 |
| M-mismatch | M，但把近状态匹配替换为同 phase 的远状态错配 | 检查 state-matched pairing 是否真实有效 |

先做 Water Plant。Trajectory teacher 随每个 M training seed 独立预训练并完成 C 的离线诊断；B0/B1/T/M 各做 3 个 training seed，M-mismatch 只做 seed 0 的机制检查。单 seed 机制通过后，再只复现 B0/B1/M 到 Hammer Nail。Fold Glasses 是可选第三任务。

### 固定评测协议

- front + wrist，23-d proprio；text 中不出现 failure phrase；
- B0/B1/T/M/M-mismatch 均保留原版 structured mask 与 `action_conditioned=false`；steer 只注入 Action Expert；
- 显式固定 `max_steps=6500`，checkpoint 为 3000/5000/6500；若 6500 的 validation 仍上升，所有存活方案统一扩到 12000；
- 训练、环境和 policy sampling seed 分开记录；相同环境 seed 在方案间 paired（各方案面对完全相同的初始环境条件）；
- 50 个 validation rollout 只用于 checkpoint 选择；独立的 200-episode development rollout 用于机制 gate；所有架构决定冻结后，B0/B1/M 每个 training seed 的选定 checkpoint 各做一次另一组 environment seed 的 200-episode held-out final（从未参与选择的最终测试）；
- 最终报告精确成功数、paired improvement、bootstrap CI（对配对结果重采样得到的不确定区间）、exact McNemar（检验同一环境 seed 下两方案成败变化是否显著）、failure category 和表示空间诊断；
- W&B 从启动即记录，同时保存本地 JSONL/CSV、manifest hash、commit/config、采样比例和 stop reason。

已有 `70/100`、`82/100`、`151/200`、`163/200` pilot 使用了不同训练或评测协议，只作为可行性依据，不进入受控主表。

## Gate

正式长跑前必须满足：raw/shard 重复 episode 已去重；train/validation manifest 无 episode overlap；同轮同阶段配对可复核且每个 phase 有足够配对；Stage A 只有 teacher 参数有梯度；Stage B 的 teacher 梯度为零，student/steer adapter 有梯度；B1 failure-only batch 的 Action Expert 专属参数无直接梯度，Video Expert 和共享 condition encoder 的梯度分组记录；M failure-only batch 的 teacher 与 Action Expert 专属参数梯度为零，student 由 pair loss 更新；C 的 teacher 在 held-out event 上优于 chance、未塌缩；推理不接 outcome/failure text；DDP save/load（多卡分布式保存并恢复）后 token 行为一致。

Steer 必须通过 intervention check（干预检查）：固定同一个 observation 和 action 采样噪声，分别使用 learned、zero 和同阶段 shuffled token，先比较预测 action chunk，再做小规模 paired rollout。learned token 与 zero/shuffled token 基本无差异时，说明 Action Expert 没有实际使用 steer，应停止长跑并检查 token 注入位置；action 已变化但成功率不变时，优先检查 event 配对和表示目标。

Water Plant seed 0 的 200-episode development gate：`M - B1 >= 4pp`（pp 表示成功率的百分点差），M 高于 B0、B1 和 T，且 `M-B1`、`M-T` 的 paired 90% CI 下界大于 0。只有先证明 pair loss 对具体匹配关系敏感，才要求 M 同时超过 M-mismatch。最终 held-out set 不参与架构、checkpoint 或任务扩展决策。3-seed 结论要求 development gain 的 95% CI 下界大于 0、至少 2/3 seed 为正，且任一 seed 不低于 B1 超过 2pp。未通过则不扩任务，先检查 pair quality、表示塌缩和 token 是否真正影响 action。

## 执行顺序

1. 冻结 source checkpoint、rollout 协议和 200 条同轮轨迹；建立 episode-disjoint split。
2. 复核 event 边界与 phase，生成 manifest，并检查每个 phase 的样本数、配对数和近邻距离分布。
3. 实现 B0/B1/T/M/M-mismatch 和 M 的 teacher 预训练，完成 action-loss mask、梯度隔离、无信息泄漏、checkpoint 恢复和 steer intervention 单测。
4. 每个 policy 方案先做 500-step smoke；冻结 teacher 先通过 C 的 held-out 表征诊断，再跑 Water Plant seed 0 和 200-episode development gate。
5. seed 0 通过后补 Water Plant 三个 training seed；再次通过后只把 B0/B1/M 扩到 Hammer Nail。
6. 架构、checkpoint 选择规则和超参数冻结后，再运行 held-out final；测试结果不回流到方法选择。

## 时间与算力

正式实验以独立 validation manifest、same-round phase matching 和 500-step smoke 为启动条件。默认资源为一条 4-GPU lane，各方案按优先级串行排队。每个 run 始终保持 `world_size=4`、global batch 和 gradient accumulation 不变，保证恢复训练前后的优化语义一致。临时增加的 GPU 只在另行确认后用于运行独立实验，不纳入默认排程。

启动前生成固定 `RUN_ID`，由四个 rank 共用同一个 output directory，避免各 rank 分别解析时间戳后把 checkpoint 写进不同目录。评测权重长期保留 3000/5000/6500；完整 optimizer/scheduler resume state 至少每 1500 step 保存一次并只保留最新两份。这样进程被终止后可以在任意四张空闲卡上从最近的完整 state 继续，同时避免频繁保存 DeepSpeed state 占满磁盘。

| 阶段 | 预计时间 |
|---|---:|
| 实现、单测、500-step smoke | 2-3 天 |
| Water Plant 单 seed：teacher/C 诊断 + B0/B1/T/M/M-mismatch + gate rollout | 4-6 天 |
| Water Plant 其余 seed 确认 | 6-9 天 |
| Hammer Nail B0/B1/M 三 seed + 最终 200 rollout | 7-10 天 |

完整受控矩阵为 22 个 continuation run：Water Plant 的 B0/B1/T/M 各 3 seed（12）、M-mismatch seed 0（1），以及 Hammer Nail 的 B0/B1/M 各 3 seed（9）；C 是随 M 训练 seed 运行的轻量 teacher 预训练与离线诊断。按一条 4-GPU lane 串行执行约需 4-6 周；若单 seed gate 不通过，停止后续长跑，不用更多 seed 掩盖架构问题。
