# EveRobot 数据格式

EveRobot 是 LeRobot 的 self-improving sidecar。原始视频、状态和动作保持不变；EveRobot 只记录：**轨迹从哪里来、发生了什么、某次训练具体用了哪些 interaction event。**

## 目录

```text
<lerobot-dataset>/
  data/  videos/  meta/                 # 原始 LeRobot 数据
  eve/
    schema_version.json
    task_schema.json
    round_meta.jsonl
    episode_meta.jsonl
    event_meta.jsonl
    annotations/subtask_scores.parquet
    manifests/<name>.json
```

帧区间统一使用左闭右开 `[start_frame, end_frame)`。sidecar 中的本机路径只是兼容提示，不参与 manifest 和 source-ledger hash；换机器时用 `dataset_root_overrides` 显式重映射。

## 各文件含义

| 文件 | 内容 |
|---|---|
| `task_schema.json` | 一个 task 的 subtask 词表、顺序约束，以及 `left/right/bimanual/global` 执行者标签 |
| `round_meta.jsonl` | 采集轮次、来源 policy/checkpoint、code/config/dataset hash、时间和父轮次 |
| `episode_meta.jsonl` | 全局 episode ID、task、round、seed、长度、split、outcome 和 outcome 来源 |
| `event_meta.jsonl` | episode ID、event/subtask、左右手、帧区间、outcome/failure type、标注来源、置信度和 action-loss 策略 |
| `subtask_scores.parquet` | 逐帧 soft subtask 分布、boundary score，可选 failure/contact score，以及方法版本和置信度 |
| `manifests/*.json` | 一次训练使用的 round、dataset、split、outcome、event、采样权重、排除项和内容 hash |

左右手 event 可以重叠。例如左手持续抓取、右手同时开门，不应被强行压成同一个 hard segment。

`steer_token` 是模型参数，不是数据 metadata。EveRobot 保存学习 token 所需的 event 标签和 provenance；token 本身跟随 checkpoint 保存。

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

## Auto Soft Subtask Annotation

自动标注不输出一个生硬的 clip label，而是对每帧输出：

```text
p_t(subtask_0 ... subtask_K-1), boundary_score_t, confidence_t
```

首版流程：

1. 每个 task 先定义一套统一 subtask 词表，不能让每个 episode 自己发明标签。
2. 从 video、state 和 action 提取逐帧证据，再按任务顺序做单调对齐。
3. 平滑结果只用于提取边界；原始 soft probability 完整保留。
4. 每个 task 人工切 30-50 条 episode，验证通过后再把 auto score 用于训练。

当前 state-line distance 只能作为 boundary cue，不能识别语义 subtask。它应该成为 `subtask_scores.parquet` 的一列，**不能再插进 robot action 维度。**

标注质量报告 boundary F1、segment IoU、subtask macro F1 和 calibration，并与 uniform/shuffled score 对照。随后再用同一训练协议比较 manual hard event 与 auto soft weight。

## 最小验证

- 构造两个 synthetic round，检查 append 幂等和 ID 冲突；
- 更换 dataset 根路径后，canonical manifest hash 保持不变；
- base-only、单轮、累计轮次、outcome/event 子集的数量完全可复核；
- 缺失 episode、越界 interval 直接失败；
- train/validation manifest 不重叠；
- `action_loss=disabled` 的 failure event 对直接 action imitation loss 贡献为零。

## 实现状态

v0.2 已实现不可变 `round/episode/event` ledger、路径无关 manifest hash、round/split/sample 筛选、dataset root 重映射、source-stride-aware window、interval 与 missing-reference 严格校验，以及 synthetic multi-round 测试。loader 继续读取已有 v0.1 manifest；v0.2 builder 不会覆盖 v0.1 sidecar，迁移时需写入新的 `eve_root`。

剩余两项：

1. 固化 `task_schema.json`，实现逐帧 `subtask_scores.parquet` 生成与质量评估；
2. 增加独立 train/validation manifest 的 episode-overlap 检查。
