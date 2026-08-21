# EveRobot 数据格式

EveRobot 是 LeRobot 的 self-improving sidecar。原始视频、状态和动作保持不变；EveRobot 只记录：**轨迹从哪里来、发生了什么、某次训练具体用了哪些 interaction event。**

## 目录

```text
<lerobot-dataset>/
  data/  videos/  meta/                 # 原始 LeRobot 数据
  eve/
    schema_version.json
    task_schema.json                      # planned annotation layer
    round_meta.jsonl
    episode_meta.jsonl
    event_meta.jsonl
    annotations/event_scores.parquet      # planned state-line evidence
    annotations/subtask_scores.parquet    # optional semantic annotation
    manifests/<name>.json
```

帧区间统一使用左闭右开 `[start_frame, end_frame)`。sidecar 中的本机路径只是兼容提示，不参与 manifest 和 source-ledger hash；换机器时用 `dataset_root_overrides` 显式重映射。

## 各文件含义

| 文件 | 内容 |
|---|---|
| `task_schema.json`（planned） | 一个 task 的 subtask 词表、顺序约束，以及 `left/right/bimanual/global` 执行者标签 |
| `round_meta.jsonl` | 采集轮次、来源 policy/checkpoint、code/config/dataset hash、时间和父轮次 |
| `episode_meta.jsonl` | 全局 episode ID、task、round、seed、长度、split、outcome 和 outcome 来源 |
| `event_meta.jsonl` | episode ID、event/subtask、左右手、帧区间、outcome/failure type、标注来源、置信度和 action-loss 策略 |
| `event_scores.parquet`（planned） | 逐帧 state-line transition score、平滑分数、candidate weight、生成参数和方法版本 |
| `subtask_scores.parquet`（optional） | 逐帧语义 subtask 分布、boundary score、可选 failure/contact score 和置信度 |
| `manifests/*.json` | 一次训练使用的 round、dataset、split、outcome、event、采样 stride、包含/排除项和内容 hash |

左右手 event 可以重叠。例如左手持续抓取、右手同时开门，不应被强行压成同一个 hard segment。

EveRobot 只保存 event 标签、outcome 和 provenance。模型侧如何消费这些字段不写入数据 metadata。

## 追加与取子集

每轮 rollout 都不可修改。重复写入同一 ID 且内容相同是 no-op；同一 ID 对应不同内容则直接报错。训练不再依赖“最新文件夹”，而由 manifest 明确选择：

```text
base expert success
+ round 0 failure
+ round 1 success/failure
- low-confidence event
```

因此无需复制视频，就能复现 base-only、单轮、累计多轮、按 outcome 平衡或人工挑选的训练集。

builder 默认对 `data/`、`videos/` 和 `meta/` 做内容 SHA-256；大数据集可传入已审计的 `--dataset-fingerprint-sha256`。ledger 更新先统一预检，再在 sidecar 锁内原子替换单个文件，ID 冲突不会留下半轮 metadata。

## State-line Candidate Events

当前 offline 路径先生成非语义的逐帧 transition score：

```text
raw score
-> mask 无历史差分的起始帧
-> robust normalization + smoothing
-> hysteresis threshold
-> remove short pulses + merge short gaps
-> candidate window + confidence/soft weight
```

State-line score 是 interaction-transition cue，不提供经过验证的准确边界或语义 subtask，也不要求每个 episode 产生相同数量的 event。失败 episode 的 outcome 只说明整条轨迹失败，不能把其中每个候选窗口都硬标成 failure cause；训练保留 candidate weight，并允许低置信度窗口不进入 pair loss。

`event_scores.parquet` 至少记录：

```text
episode_id, frame_index, transition_score, smoothed_score,
candidate_weight, method_version, calibration
```

现有仓库汇总报告 action-prefix probe 为 `183/200`，但该实验把连续 score 作为额外 action 维度联合预测，没有进行 event extraction。它是探索性 policy 结果，不等同于自动 event 标注质量；标准 EveRobot 数据仍将 score 保存为 sidecar evidence，不插进 robot action。

训练首先比较 whole episode、state-line soft candidates 和 shuffled candidates。小规模人工边界只用于报告 boundary precision/recall、segment IoU、候选数量误差和参数敏感性，不是每轮 rollout 的训练前置条件。

## Optional Semantic Subtask Annotation

语义层参考 WALL-WM 的 task/subtask/action/segment event 层级，对每帧输出：

```text
p_t(subtask_0 ... subtask_K-1), boundary_score_t, confidence_t
```

每个 task 使用统一 subtask 词表，并结合 video、state 和 action evidence 做顺序约束。该层可在后续用于更严格的 same-phase matching。模型侧 event-score auxiliary prediction 是另一项架构消融，不属于 state-line 数据生成流程。

## 最小验证

- 构造两个 synthetic round，检查 append 幂等和 ID 冲突；
- 更换 dataset 根路径后，canonical manifest hash 保持不变；
- base-only、单轮、累计轮次、outcome/event 子集的数量完全可复核；
- 缺失 episode、越界 interval 直接失败；
- train/validation manifest 不重叠；
- `action_loss=disabled` 的 failure event 对直接 action imitation loss 贡献为零。

## 实现状态

v0.2 已实现不可变 `round/episode/event` ledger、路径无关 manifest hash、round/split/sample 筛选、dataset root 重映射、source-stride-aware window、interval 与 missing-reference 严格校验，以及 synthetic multi-round 测试。loader 继续读取已有 v0.1 manifest；v0.2 builder 不会覆盖 v0.1 sidecar，迁移时需写入新的 `eve_root`。

剩余三项：

1. 实现 `event_scores.parquet`、候选窗口提取和 soft-weight 数据接口；
2. 固化 `task_schema.json`，实现可选的 `subtask_scores.parquet` 与质量评估；
3. 增加独立 train/validation manifest 的 episode-overlap 检查。
