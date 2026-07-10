# Related Work Coverage Audit

更新日期：2026-07-10

## 1. 审计结论

原稿只有 3 类、24 篇核心文献，能够说明项目位置，但不足以支撑完整论文。问题不只是数量少：它缺少 generalist VLA 与 action decoder 的主干、mixed-quality imitation learning 的成熟先例、弱监督时间定位，以及 tactile policy 到 tactile WAM 的演进关系。

本次重写采用 5 类、13 个综合性段落。正文目标不是逐篇罗列，而是每段都完成三件事：概括一条技术演进、指出与 FITWAM 的关系、收紧可主张的新颖性。

## 2. 榜样论文

下表基于论文 PDF 的 Related Work 部分做近似统计。数字区间引用会放大“被提及工作数”，因此只用于判断覆盖量级，不作为精确 bibliometric 结果。

| 论文 | Related Work 的组织方式 | 近似覆盖 | 对 FITWAM 的约束 |
|---|---|---:|---|
| [RT-2](https://arxiv.org/abs/2307.15818) | VLM、机器人泛化、机器人预训练 | 约 80 次文献提及 | 从基础模型主干逐步收束到 action-token VLA |
| [Open X-Embodiment / RT-X](https://arxiv.org/abs/2310.08864) | 跨 embodiment 迁移、数据集、语言条件策略 | 约 100 篇/组 | 用密集引用说明数据与泛化问题的历史位置 |
| [Octo](https://arxiv.org/abs/2405.12213) | 数据规模、generalist policy、模型设计组件 | 约 55 篇 | 每类最后说明开放性和适配性的缺口 |
| [Diffusion Policy](https://arxiv.org/abs/2303.04137) | 显式策略、隐式策略、diffusion model | 约 50 篇 | Related Work 紧贴方法机制，不写宽泛机器人史 |
| [$\pi_0$](https://arxiv.org/abs/2410.24164) | VLA、flow/diffusion、multimodal LM、大规模机器人学习 | 约 45 篇 | 把 action generator 放进 foundation-policy 谱系 |
| [$\pi_{0.5}$](https://arxiv.org/abs/2504.16054) | generalist policy、非机器人数据共训、语言规划、开放世界泛化 | 约 75 篇 | 分类与论文主张直接对齐 |
| [$\pi_{0.7}$](https://arxiv.org/abs/2604.15483) | generalist policy、跨任务/embodiment、subgoal 与 context steering | 约 100 篇/组 | 已覆盖 metadata/context steering，FITWAM 不能泛称其为首创 |
| [OpenVLA](https://arxiv.org/abs/2406.09246) | VLM、generalist robot policy、VLA | 约 70 篇 | 清晰漏斗结构，最后落到开放模型和 finetuning 缺口 |
| [Sparsh](https://arxiv.org/abs/2410.24090) | tactile sensing、representation learning、下游任务 | 数十篇 | 触觉段应覆盖 sensor、representation、policy，而不只列 tactile WAM |

从这些论文得到的合理目标是：**4-5 个与方法直接相关的类别、约 1,200-1,600 个英文词、45-80 篇正文引用、成熟工作与最新竞争工作并存。** 引用数不是单独的通过标准；如果没有技术关系和差异边界，80 篇引用仍然可能只是 bibliography dump。

## 3. 当前分类

| 类别 | 必须回答的问题 | 代表工作 |
|---|---|---|
| Generalist policies and action generation | backbone、action chunk、diffusion/flow、metadata steering 已经做到什么 | RT-1/2, Open X, Octo, OpenVLA, ACT, Diffusion Policy, RDT, $\pi$ series, GR00T |
| World and world-action models | world supervision 如何进入 control，推理时是否生成 future | visual foresight, DayDreamer, UniPi, Motus, DreamZero, Fast-WAM, WoG, WALL-WM |
| Imperfect data and temporal credit | failure episode 中哪些片段有用，如何避免错误 BC 和错误负标 | T-REX/D-REX, ORIL, DemoDICE, DWBC, SSDF, ContraDiff, AFIL, MIL localization |
| Post-training and self-improvement | offline update、online RL、retry、test-time adaptation 的边界 | DAgger/DART, RoboCat, RLPD, SERL, DPPO, RECAP, RL Token, RISE, AdaJEPA, FAR |
| Tactile learning and WAMs | tactile 表征、策略、预测模型与 failure-local contact 的关系 | DIGIT, Sparsh, AnyTouch, Sparsh-X, FTP-1, RDP, VT-WM, VTAM, Tactile-WAM, VT-WAM |

## 4. 最接近工作与新颖性压力

| 工作 | 已有能力 | FITWAM 必须用实验守住的差异 |
|---|---|---|
| [AFIL](https://arxiv.org/abs/2605.08434) | 成功/失败双 action generator 与 negative guidance | 单 action generator；失败动作不做 BC；离线得到固定 steer；不在推理时运行 failure generator |
| [FAR](https://arxiv.org/abs/2607.01111) | failure-contrastive preference、retry、continual improvement | 当前主结果是一次离线更新；无 retry、online reward 或 test-time adaptation |
| [$\pi_{0.7}$](https://arxiv.org/abs/2604.15483) | metadata、subgoal 和 diverse context steering | 不主张 metadata/steering token 本身；只主张 failure-local negative supervision |
| [WALL-WM](https://arxiv.org/abs/2606.01955) | event-grounded Task/Subtask/Action/Segment 层级 | soft failure localization 和 action-negative steer 是新增部分；event schema 不是首创 |
| [SSDF](https://arxiv.org/abs/2401.08957) | 从 imperfect trajectory 选择高质量片段 | FITWAM 不只过滤/扩充 BC 数据，而把失败局部作为负表示监督 |
| [ContraDiff](https://arxiv.org/abs/2402.02772) | 高/低 return state 的 contrastive planning | 不依赖 reward 或 test-time trajectory planning；直接作用于 Fast-WAM action context |
| [Fast-WAM](https://arxiv.org/abs/2603.16666) | 训练期 video supervision、推理期 action-only | backbone 与效率属性；FITWAM 的增量必须来自 failure-aware action learning |
| [Tactile-WAM](https://arxiv.org/abs/2606.26663) / [VT-WAM](https://arxiv.org/abs/2607.02503) | tactile-aware WAM architecture | 触觉版本必须证明 failure/contact localization 的额外收益，不能只证明“加触觉有用” |

## 5. 来源选择规则

1. 技术结论只引用论文、项目官方页面或正式论文库；不把二手博客作为证据。
2. 成熟主干优先使用已发表或广泛验证的工作；2026 年 WAM/failure/tactile 工作单独视为 contemporaneous preprint。
3. 最新预印本用于说明 novelty pressure，不单独承担方法合理性的证据。
4. 同一工作只在最能说明其作用的段落出现；避免每段重复列 Fast-WAM、AFIL 和 $\pi_{0.7}$。
5. 论文最终若没有 tactile method 和 tactile experiment，正文删除 tactile subsection，避免研究范围与实验不一致。

## 6. 完成度标准

| 检查项 | 目标 | 当前重写 |
|---|---:|---:|
| 分类数 | 4-5 | 5 |
| 综合性段落 | 10-14 | 13 |
| 英文正文 | 1,200-1,600 words | 1,362 |
| 正文独立引用 | 45-60 为常见目标；更高时检查是否堆叠 | 70；逐段检查后保留 |
| 最接近工作 | 至少 6 个并逐轴区分 | 8 |
| 成熟方法主干 | 必须覆盖 | 已覆盖 |
| 2026 同期竞争工作 | 必须单列风险 | 已覆盖 |
| 每类有 FITWAM 边界 | 必须 | 已覆盖 |
| 与实验一一对应 | baseline/ablation 能验证差异 | 需与实验计划共同校验 |

70 篇引用高于预设中心值，但仍处在 OpenVLA、$\pi_{0.5}$、$\pi_{0.7}$ 等榜样论文的覆盖量级内。当前引用以技术演进簇组织，并非逐篇摘要；若最终版受页数限制，优先合并同系列最新预印本，不删成熟主干和最接近工作。

最终论文不应机械追求更多引用。通过标准是：审稿人能从 Related Work 直接看出我们知道最接近的方法、没有把已有概念改名，并且实验表中的每个关键对照都能对应一个明确的文献差异。
