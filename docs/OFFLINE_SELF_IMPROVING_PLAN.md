# Offline Self-Improving：方法与实验

## 研究范围

本文档定义 FITWAM 的 offline self-improving 方法与受控实验：利用 rollout（策略在环境中闭环执行后保存的一条 episode）中同一交互阶段的成功/失败动作事件训练 steer token，并与 success replay 和 failure-video baseline 比较。Online RL、test-time world-model update 和 tactile 属于整篇工作的其他模块，不进入本实验。

Offline steer 要求 EveRobot 提供冻结且经过人工复核的 event manifest（记录 episode、交互阶段、起止帧和 loss 策略的索引文件）、数据 provenance（数据来自哪一轮、哪个 checkpoint 和哪份配置）和 episode-disjoint split（同一 episode 的任何窗口只能属于一个数据划分）。自动 event 标注由 EveRobot 数据模块独立定义和验证，不属于 offline steer 的实验变量，也不阻塞该实验。

论文主张限定为：

> Phase-matched failure events provide training-only negative action supervision for an observation-conditioned steer. Failed actions are never imitation targets, and deployment needs neither failure trajectories nor outcome labels.

## 主方法：trajectory teacher（轨迹教师），observation student（观测学生）

训练时，用同一 task phase（任务中的同一种交互阶段，例如 reach、grasp 或 pour）的 success/failure action event 构造配对。共享 trajectory encoder `E` 将归一化 action chunk（一段连续动作序列）和有效帧 mask（标记真实帧，避免 padding 参与编码）压缩成一个 bottleneck token（用一个固定长度向量概括整段动作）：

```text
z+ = E(a+_1:H, mask+)        z- = E(a-_1:H, mask-)
```

`E` 是训练期 teacher（它读取已经发生的动作轨迹来提供学习目标，部署时不运行）。observation-conditioned student `G`（只根据当前可观测信息预测 steer 的小网络）从当前首帧双视角 token、task text 和 proprio 预测可部署 steer；它不能读取未来动作、failure label 或最终 outcome：

```text
s = G(front, wrist, task, proprio)
observation + task + proprio + s -> Action Expert -> action
```

`z+`、`z-` 只作为训练目标，不进入 policy forward（policy 真正生成动作的前向计算）。Pair loss（成对表示损失）把 `s` 拉向同阶段成功动作 `z+`，并推离配对失败动作 `z-` 和 batch 内失败负样本；teacher target 在该项中 stop-gradient（把目标当作常量，不让 pair loss 反向修改 teacher），避免 student 和 teacher 一起退化成相同的无信息向量。Trajectory encoder 另用 phase-conditioned supervised contrastive loss（同阶段成功动作聚近、失败动作推远的有监督对比损失）学习动作事件结构。

```text
L = L_video(all)
  + L_action(success only)
  + lambda_ctr * L_action_contrast
  + lambda_pair * L_steer_pair
```

Failure sample 的直接 action flow-matching loss（Action Expert 针对该条已记录动作的直接生成监督）必须严格为零；否则模型会把有缺陷的失败动作当作 imitation target 进行模仿。这里的“为零”只指这条直接动作损失，failure video 和辅助表示损失仍可训练各自对应的模块。失败动作只进入 training-only trajectory teacher；student token 只接 Action Expert。首版使用 `B=1`（每段轨迹只压缩为一个 token），只有在单 token 已有稳定增益后才增加 bottleneck 数量。

`S = Z+ + alpha(Z+ - Z-)` 不作为主方法：逐 pair 计算会在推理时依赖未来轨迹，全局平均后又退化成静态 soft prompt（对所有状态都使用同一个可学习提示）。它只保留为 prototype-delta ablation（检验“成功原型减失败原型”方向是否有用的对照实验）。

Sparsh-X 说明少量 bottleneck token 可以压缩多源交互信息；它不直接给出 success/failure steer 的学习规则。实现必须包含 observation-conditioned steer、仅面向 Action Expert 的 token 注入、完整 checkpoint save/load，以及不依赖 outcome context 的部署路径。

## 数据与控制变量

固定 source checkpoint 采 200 条 rollout，成功和失败都来自同一轮、同一 policy 与同一环境协议。先按 episode 和 environment seed 固定 train/validation，再切 event 和做 pair mining（从同阶段样本中寻找状态最接近的成功/失败配对），避免同一 episode 的窗口跨 split。人工标出 phase 与 failure boundary；success/failure pair 先限制在同一 task phase，再按 event 起点的 proprio 与冻结视觉特征（不随本实验更新的视觉表示）做近邻匹配，不能使用 outcome frame（已经暴露最终成功或失败结果的画面）。

Event manifest 在 split、pair mining 和模型拟合前冻结。Teacher fitting、pair/prototype 构造和 action normalization 只使用 train episode。Student 的输入截止到当前控制时刻；outcome label、failure text、post-outcome frame 和 teacher action 都不能进入 student 或 Action Expert，避免 label leakage（训练时偷看部署时不存在的信息）。未来视频和动作只进入 training-only teacher 或 loss target，不进入部署路径。

各方案使用相同的原始 success buffer、同一轮 rollout、global batch、optimizer、scheduler、训练步数和 normalization。新增 rollout 数据采用两路采样：success event 提供 video + action loss，failure event 只提供 video 和辅助表示 loss。Contrastive batch 按 phase/outcome 分层，保证每个有效 anchor（当前作为比较中心的样本）至少有一个正样本；无正样本的 anchor 不计入该 loss。B0 使用同计算量的 success replay，避免把“更多更新步数”误当成 failure 收益。

### 没有人工切分时

主方法需要 event 边界，但不要求把原始 LeRobot episode 物理裁成多个新视频；EveRobot 可以在完整 episode 上用 `start_frame/end_frame` 选择训练窗口。数据尚未切分时按以下顺序处理：

1. **主实验路径：** 用人工标注，或先由 Switch/state-line score 提议候选边界，再由人复核边界并赋予 task phase；复核后的 manifest 冻结后即可进入主实验。Switch score 只能提示“这里可能发生了动作变化”，不能单独判断语义 subtask。
2. **兼容性 smoke：** 可以把整条 episode 视为一个 event，再统一采样、截断或 padding；它只验证 loader、loss 和 checkpoint pipeline，不能支持“局部 interaction failure steer”的论文主张。
3. **全自动划分：** 未经人工复核的 Switch score 属于 EveRobot 自动标注实验；在边界和 phase 质量通过独立评估前，不进入 offline steer 主表。

## 实验矩阵

| 方案 | 训练信号 | 目的 |
|---|---|---|
| S0 | 冻结 source checkpoint | rollout 数据来源与零更新参考 |
| B0 | same-round success replay | compute-matched continuation baseline |
| B1 | B0 + failure video；failure action loss=0 | failure-world-representation baseline（只检查失败视频表征带来的收益） |
| T | B1 + observation student；只由普通 success action loss 端到端训练 | 控制 student/token 参数量 |
| C | B1 + action trajectory contrast；steer 不进 Action Expert | 控制辅助 loss |
| M | B1 + trajectory teacher + observation student | 主方法 |
| M-shuffle | M，但 phase 内打乱 success/failure 配对 | 检查语义配对是否真实有效，而不是额外 loss 本身带来的收益 |

先做 Water Plant。B0/B1/T/C/M 各做 3 个 training seed；M-shuffle 只做 seed 0 的机制检查。单 seed 机制通过后，再只复现 B0/B1/M 到 Hammer Nail。Fold Glasses 是可选第三任务。

### 固定评测协议

- front + wrist，23-d proprio；text 中不出现 failure phrase；
- 显式固定 `max_steps=6500`，checkpoint 为 3000/5000/6500；若 6500 的 validation 仍上升，所有存活方案统一扩到 12000；
- 训练、环境和 policy sampling seed 分开记录；相同环境 seed 在方案间 paired（各方案面对完全相同的初始环境条件）；
- 50 个 validation rollout 只用于 checkpoint 选择；独立的 200-episode development rollout 用于机制 gate；所有架构决定冻结后，B0/B1/M 每个 training seed 的选定 checkpoint 各做一次另一组 environment seed 的 200-episode held-out final（从未参与选择的最终测试）；
- 最终报告精确成功数、paired improvement、bootstrap CI（对配对结果重采样得到的不确定区间）、exact McNemar（检验同一环境 seed 下两方案成败变化是否显著）、failure category 和表示空间诊断；
- W&B 从启动即记录，同时保存本地 JSONL/CSV、manifest hash、commit/config、采样比例和 stop reason。

已有 `70/100`、`82/100`、`151/200`、`163/200` pilot 使用了不同训练或评测协议，只作为可行性依据，不进入受控主表。

## Gate

正式长跑前必须满足：train/validation manifest 无 episode overlap；同轮同阶段配对可复核且每个 phase 有足够配对；B1 failure-only batch 的 Action Expert gradient 为零且 Video Expert gradient 非零；M 的 trajectory encoder 和 steer gradient 非零；推理不接 outcome/failure text；DDP save/load（多卡分布式保存并恢复）后 token 行为一致。

Steer 必须通过 intervention check（干预检查）：固定同一个 observation 和 action 采样噪声，分别使用 learned、zero 和同阶段 shuffled token，先比较预测 action chunk，再做小规模 paired rollout。learned token 与 zero/shuffled token 基本无差异时，说明 Action Expert 没有实际使用 steer，应停止长跑并检查 token 注入位置；action 已变化但成功率不变时，优先检查 event 配对和表示目标。

Water Plant seed 0 的 200-episode development gate：`M - B1 >= 4pp`（pp 表示成功率的百分点差），paired 90% CI 下界大于 0，并且 M 同时超过 T、C 与 M-shuffle。最终 held-out set 不参与架构、checkpoint 或任务扩展决策。3-seed 结论要求 development gain 的 95% CI 下界大于 0、至少 2/3 seed 为正，且任一 seed 不低于 B1 超过 2pp。未通过则不扩任务，先检查 pair quality、表示塌缩和 token 是否真正影响 action。

## 执行顺序

1. 冻结 source checkpoint、rollout 协议和 200 条同轮轨迹；建立 episode-disjoint split。
2. 复核 event 边界与 phase，生成 manifest，并检查每个 phase 的样本数、配对数和近邻距离分布。
3. 实现 B0/B1/T/C/M/M-shuffle，完成 action-loss mask、梯度隔离、无信息泄漏、checkpoint 恢复和 steer intervention 单测。
4. 每个方案先做 500-step smoke；随后只跑 Water Plant seed 0 和 200-episode development gate。
5. seed 0 通过后补 Water Plant 三个 training seed；再次通过后只把 B0/B1/M 扩到 Hammer Nail。
6. 架构、checkpoint 选择规则和超参数冻结后，再运行 held-out final；测试结果不回流到方法选择。

## 时间与算力

正式实验以独立 validation manifest、same-round phase matching 和 500-step smoke 为启动条件。8 张卡按两条独立的 4-GPU lane 运行两个实验；后续只剩 4 张卡时按同一队列继续，不把单个 run 从 8 卡改成 4 卡。每个 run 始终保持 `world_size=4`、global batch 和 gradient accumulation 不变，保证中断前后优化语义一致。

启动前生成固定 `RUN_ID`，由四个 rank 共用同一个 output directory，避免各 rank 分别解析时间戳后把 checkpoint 写进不同目录。评测权重长期保留 3000/5000/6500；完整 optimizer/scheduler resume state 至少每 1500 step 保存一次并只保留最新两份。这样进程被终止后可以在任意四张空闲卡上从最近的完整 state 继续，同时避免频繁保存 DeepSpeed state 占满磁盘。

| 阶段 | 预计时间 |
|---|---:|
| 实现、单测、500-step smoke | 2-3 天 |
| Water Plant 单 seed：B0/B1/T/C/M/M-shuffle + gate rollout | 4-6 天 |
| Water Plant 三 seed 确认 | 5-7 天 |
| Hammer Nail B0/B1/M 三 seed + 最终 200 rollout | 5-7 天 |

完整受控矩阵为 25 个 continuation run：Water Plant 的 B0/B1/T/C/M 各 3 seed（15）、M-shuffle seed 0（1），以及 Hammer Nail 的 B0/B1/M 各 3 seed（9）。按两条 4-GPU lane 约需 3-4 周。若单 seed gate 不通过，停止后续长跑，不用更多 seed 掩盖架构问题。
