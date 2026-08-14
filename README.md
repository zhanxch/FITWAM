# FITWAM: Failure-Improvement Tactile WAM

> WAM learns state transitions. Physical understanding comes from interaction transitions.

[![上游 FastWAM](https://img.shields.io/badge/上游-FastWAM-111111.svg)](./docs/FASTWAM_UPSTREAM.md)

**相关文档：** [`docs/DEWOV2.md`](./docs/DEWOV2.md) · [`DEWO.md`](./DEWO.md) · [`docs/EVEROBOT_FORMAT.md`](./docs/EVEROBOT_FORMAT.md) · [`docs/RELATED_WORK.md`](./docs/RELATED_WORK.md) · [`docs/AWS_RUNBOOK.md`](./docs/AWS_RUNBOOK.md) · [`docs/FASTWAM_UPSTREAM.md`](./docs/FASTWAM_UPSTREAM.md) · [`scripts/README.md`](./scripts/README.md) · [`scripts/water_plant/README.md`](./scripts/water_plant/README.md)

本仓库是在 FastWAM 上做的 failure / self-evolution / tactile 方向 fork。当前主线不是重新做一个通用 WAM，而是围绕失败和交互事件回答一个问题：

**成功数据只告诉模型“怎么做对”，失败轨迹和接触反馈能不能让 WAM 学到“为什么会失败、下一轮怎么避免”。**

当前仿真只先考虑 DexJoCo `water_plant` 和 `hammer_nail` 两个任务（front + wrist 双视角，proprio 23d）。真机主战场是 `spray_water`（3cam rot6d，已有 FastWAM server/client deploy 链路）。

## 当前结论

DexJoCo `water_plant` 的历史 pilot 提供了前期可行性信号：

| 结论 | 证据 | 位置 |
|------|------|------|
| **M1：failure video pilot** | Text failure 从 `38/100 -> 81/100 -> 82/100`；failure 样本的直接 action loss 为零 | [`papers/II/experiment_results.md`](./papers/II/experiment_results.md) |
| **success-only 历史参考** | step 6500 为 `70/100`；另一次 source-policy rollout 为 `151/200` | [`papers/II/experiment_results.md`](./papers/II/experiment_results.md) |
| **rollout continuation pilot** | rollout text-failure LoRA continuation 为 `163/200` | [`papers/II/experiment_results.md`](./papers/II/experiment_results.md) |
| **EveRobot sidecar v0.2 已实现** | 不可变 round ledger、可复核 manifest hash、round/sample 子集和路径重映射已覆盖测试；历史 round1 数据仍是 v0.1 | [`docs/EVEROBOT_FORMAT.md`](./docs/EVEROBOT_FORMAT.md) |

注意：前三项是 mixed-protocol 历史 pilot，当前 checkout 没有保存全部 raw summary。`151/200` 和 `163/200` 属于 source-policy rollout 与 LoRA continuation 计数，不能据此单独归因于 EveRobot 或 M4，也不与后面的受控筛选结果混成最终主表。

## 当前 Milestone

| Milestone | 要证明什么 | 当前状态 |
|-----------|------------|----------|
| **M1 Failure video** | failure 轨迹在直接 action loss 为零时仍可通过 video/shared MoT 影响策略 | 200 个配对种子下，B1 在 step 5000/6000/6500 均高于 B0；仍需训练 seed 复现 |
| **M2 EveRobot** | failure rollout 需要结构化记录 outcome、event window、manifest 和 provenance | v0.2 builder/loader 已实现；历史 round1 数据待重建和对齐 |
| **M3 DEWO conditioning** | failure event 的 VideoDiT 使用 `base + FAST(action)`，ActionDiT 只使用 base task | fold_glasses 方案与代码已接入，待正式实验 |
| **M4 Offline self-evolution** | Train -> Test -> append rollout -> retrain 多轮后继续涨点 | 单轮数据闭环已跑通；当前先验证 fold_glasses DEWO |
| **M5 Real tactile** | 真机接触期用触觉区分成功/失败，补 RGB 不可见的接触信息 | 待开展真机实验和触觉模块 |

整体顺序：

```text
M1 证明 failure video 有用
  -> M2 用 EveRobot 把 rollout / event / manifest 管起来
  -> M3 用 FAST(action) 强化 VideoDiT 的 action-video 对齐
  -> M4 多轮 rollout 回灌，验证 offline self-evolution
  -> M5 真机触觉，把 contact-rich failure 接进同一套 event 叙事
```

## 已实现内容

### 1. Failure data training

普通 LeRobot 数据路径支持用文本 marker 控制 loss：

- `action_loss_zero_if_instruction_contains: "Failed to finish the whole process."`
- 命中 marker 的 failure 样本返回 `action_loss_weight=0.0`
- EveRobot 主实验会去掉 failure 后缀；旧 text-failure pilot 保留了该后缀，只作为历史证据

关键实现：

- [`src/fastwam/datasets/lerobot/robot_video_dataset.py`](./src/fastwam/datasets/lerobot/robot_video_dataset.py)
- [`src/fastwam/models/wan22/fastwam.py`](./src/fastwam/models/wan22/fastwam.py)
- [`configs/data/dexjoco_water_plant_2cam_text_failure.yaml`](./configs/data/dexjoco_water_plant_2cam_text_failure.yaml)
- [`configs/data/dexjoco_water_plant_2cam_rollout_text_failure.yaml`](./configs/data/dexjoco_water_plant_2cam_rollout_text_failure.yaml)

### 2. EveRobot sidecar v0.2

EveRobot 不改 LeRobot 原始 `data/`、`videos/`、`meta/`，只在 `eve/` 下增加 sidecar：

| 文件 | 作用 |
|------|------|
| `schema_version.json` | EveRobot 版本和兼容说明 |
| `round_meta.jsonl` | 不可变采集轮次及 checkpoint/config/code/dataset provenance |
| `episode_meta.jsonl` | episode 级 provenance：source policy、round、seed、outcome、length |
| `event_meta.jsonl` | failure/event window、failure type、标注来源和 action loss 策略 |
| `manifests/*.json` | 显式选择 round/split/outcome/event 子集，记录路径无关的内容 hash |

历史 v0.1 round1 数据：

| 项 | 数量 / 策略 |
|----|-------------|
| base success episode | 100 |
| rollout episode | 200 = 151 success + 49 failure |
| round1 train manifest | 100 success episode + 49 failure event |
| 600 帧 timeout failure | trim 8s，event window 变成 `[0, 360)` |
| 短 failure | 使用完整 failure episode window |
| failure action loss | `disabled` |

关键实现：

- [`scripts/everobot/build_eve_sidecar.py`](./scripts/everobot/build_eve_sidecar.py)
- [`scripts/water_plant/build_eve_round1_sidecar.sh`](./scripts/water_plant/build_eve_round1_sidecar.sh)
- [`src/fastwam/datasets/eve/manifest_dataset.py`](./src/fastwam/datasets/eve/manifest_dataset.py)
- [`configs/data/eve_water_plant_round1_failure_events.yaml`](./configs/data/eve_water_plant_round1_failure_events.yaml)

### 3. Steer token

当前代码包含 trajectory Teacher、observation Student、零初始化 residual projection 和 weighted pair loss。旧 outcome plumbing 仍保留：

- dataset 返回 `outcome_flag`
- `FastWAM._append_outcome_to_context()` 将 outcome embedding 追加到 text context
- checkpoint save/load 支持 `outcome_encoder`

当前默认配置里 `model.outcome_num_classes: 0`。M3 不直接打开这条路径，因为真实 outcome 进入 shared context 会造成训练/推理接口不一致。首轮实现采用：

1. 用 state-line soft candidate 配对 success/failure action window
2. 训练 trajectory Teacher，并冻结为 observation Student 的 success/failure target
3. 将 Student steer 通过零初始化 residual branch 接入 action expert；推理不输入 outcome
4. 先跑 `water_plant` 闭环 gate；通过后再扩展到 `hammer_nail`

### 4. Rollout collect / retrain

DexJoCo eval 和 rollout collect 共用 async ZMQ server/client：

- [`scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py`](./scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py)
- [`scripts/dexjoco_async/run_multi_gpu_dexjoco_collect.py`](./scripts/dexjoco_async/run_multi_gpu_dexjoco_collect.py)
- [`scripts/collect_dexjoco_rollouts.py`](./scripts/collect_dexjoco_rollouts.py)
- [`scripts/build_rollout_datasets.py`](./scripts/build_rollout_datasets.py)

默认 water_plant 一键路径：

```bash
bash scripts/water_plant/collect_rollout_200_trim8s_and_train.sh
```

这条路径会采集 rollout、保存 success/failure、trim timeout failure 的尾部，然后用 rollout text failure 配置继续训练。

### 5. Video LoRA

Video LoRA 是独立可选训练后端，不改变 failure/self-evolution 逻辑：

| 维度 | 全参 FastWAM | Video LoRA |
|------|--------------|------------|
| Video DiT | 全参微调 | LoRA adapter，rank 32 |
| ActionDiT / proprio | 全参微调 | 全参微调 |
| 配置 | [`configs/model/fastwam.yaml`](./configs/model/fastwam.yaml) | [`configs/model/fastwam_video_lora.yaml`](./configs/model/fastwam_video_lora.yaml) |
| 实现 | FastWAM 原路径 | [`src/fastwam/models/wan22/video_lora.py`](./src/fastwam/models/wan22/video_lora.py) |

使用：

```bash
bash scripts/water_plant/train_2cam.sh task=water_plant_uncond_2cam_384_1e-4_lora
```

## 复现实验入口

### Water plant baseline

```bash
bash scripts/water_plant/prepare_2cam.sh
bash scripts/water_plant/train_2cam.sh task=water_plant_uncond_2cam_384_1e-4
```

### M1 text failure（当前 6500-step 配置）

```bash
python scripts/precompute_text_embeds.py task=dexjoco/dexjoco_water_plant_text_failure_2cam_proprio_1e-4
bash scripts/train_zero1.sh 4 task=dexjoco/dexjoco_water_plant_text_failure_2cam_proprio_1e-4
```

### M2 EveRobot round1

```bash
bash scripts/water_plant/build_eve_round1_sidecar.sh
bash scripts/water_plant/train_eve_round1.sh
```

### M4 rollout continuation

```bash
bash scripts/water_plant/collect_rollout_200_trim8s_and_train.sh
```

### DexJoCo closed-loop eval

```bash
python scripts/dexjoco_async/run_multi_gpu_dexjoco_eval.py \
  --gpus 0,1,2,3 \
  --run-dir runs/<task_name>/<run_id> \
  --checkpoint runs/<task_name>/<run_id>/checkpoints/weights/step_006500.pt \
  --no-load-text-encoder \
  --task-config-dir third_party/dexjoco/configs/rand_obj \
  --tasks water_plant \
  --episodes 100 --seed 25 \
  --replan-steps 24 --control-mode blocking \
  --max-env-steps 600 \
  --output-dir evaluate_results/dexjoco/<name>/step_006500
```

## 结果记录

### DexJoCo water_plant

| Method / setting | Success | Notes |
|------------------|--------:|-------|
| FastWAM success-only, local same pipeline | 70/100 | step 6500, seed 25, blocking stride 24 |
| FastWAM success-only rollout provenance | 151/200 | round0 rollout used to build failure pool |
| M1 Text failure | 38/100 -> 81/100 -> 82/100 | step 6500 / 11000 / 12240 |
| Historical rollout text-failure LoRA continuation | 163/200 | continued from success-only step 6500 |
| Offline B1 failure-video control | 80.5% | validation-best step 4500, 200 paired seeds |
| Offline B0 success-only control | 72.5% | validation-best step 5500, 200 paired seeds |
| Offline C residual-only | 74.5% | validation-best step 6500, 200 paired seeds |
| Offline M contrastive steer | 88.0% | validation-best step 5000; `M - B1 = +7.5pp`, 95% paired bootstrap CI `[+1pp, +14pp]` |
| Offline B1 fixed-step confirmation | 85.0% | secondary E1 step 6000, `170/200` paired fresh seeds |
| Offline M fixed-step confirmation | 87.5% | secondary E1 step 6000, `175/200`; `M - B1 = +2.5pp`, 95% CI `[-3.5pp, +8.5pp]`, McNemar `p=0.511` |
| FastWAM S0 fresh reference | 75.0% | E1 source step 6500, `150/200`; same seeds and rollout protocol |
| Offline M residual bypass | 10.0% | E1 step 6000, `20/200`; learned steer minus bypass `+77.5pp` |
| Offline M shuffled steer | 88.0% | E1 step 6000, `176/200`; shuffled minus learned `+0.5pp`, 95% CI `[-5.0pp, +6.0pp]` |
| Strict E2 C residual-only validation-best | 85.0% | primary checkpoint diagnostic; step 5000, `170/200`, reused E2 seeds |
| Strict E2 M / pair-shuffle validation-best | 54.5% / 71.0% | steps 6500 / 6500; C-M `+30.5pp`, C-pair-shuffle `+14.0pp` |
| Strict E2 S0 / B1 fixed-step | 77.0% / 82.0% | secondary fixed-step evidence; seeds `20262200..20262399`, 200 paired episodes |
| Strict E2 C residual-only fixed-step | 82.0% | secondary step 6000; common init; C-B1 `0.0pp`, paired 95% CI `[-6.5pp, +6.5pp]` |
| Strict E2 M / pair-shuffle fixed-step | 58.5% / 73.5% | secondary step 6000; common init; M-B1 `-23.5pp`, M-pair-shuffle `-15.0pp` |
| External pi0.5 | 88.7 +/- 3.1 | from DexJoCo rand-obj table, raw trials not tracked here |
| External GR00T N1.5 | 72.7 +/- 1.2 | from DexJoCo rand-obj table, raw trials not tracked here |

当前候选论文结果统一记录在 [`papers/II/experiment_results.md`](./papers/II/experiment_results.md)。

### 其他任务

当前仿真任务范围只保留 `water_plant` 和 `hammer_nail`：

- `hammer_nail` success-only step 6500：`135/200`
- state-line transition probe：[`scripts/probe_state_line_distance.py`](./scripts/probe_state_line_distance.py)；action-prefix converter 只用于旧 probe，不属于 EveRobot 标准格式

## 真机与触觉

真机当前以 `spray_water` 为主：

- 训练配置：[`configs/task/spray_water_rot6d_gr00tstyle_uncond_3cam_384_1e-4.yaml`](./configs/task/spray_water_rot6d_gr00tstyle_uncond_3cam_384_1e-4.yaml)
- 脚本：[`scripts/spray_water_gr00tstyle/`](./scripts/spray_water_gr00tstyle/)
- deploy：[`scripts/spray_water_gr00tstyle/wuji/`](./scripts/spray_water_gr00tstyle/wuji/)
- 开环 smoke eval 由 `scripts/spray_water_gr00tstyle/` 下的对应脚本生成，结果不在 Git 中保存。

M6 的目标不是在仿真里强行加触觉，而是在真机接触期补上 RGB 难以区分的状态：

```text
tactile planning -> future tactile prediction -> tactile-refined action
```

计划接入方式：

1. 把 tactile episode / contact window 写进 EveRobot event metadata
2. 在 MoT 中加 tactile expert 或 tactile token
3. 在接触期预测 future tactile，并用 tactile outcome 改善 action
4. 与 M3 steer token、M4 offline loop 和 M5 online update 汇合

## 代码地图

```text
configs/
  data/                         数据配置，含 text failure / rollout failure / Eve manifest
  model/fastwam*.yaml            FastWAM 与 Video LoRA
  task/dexjoco/                  DexJoCo water_plant / hammer_nail failure/self-evolution 实验

src/fastwam/
  datasets/lerobot/              原始 LeRobot 固定窗口数据集和 processor
  datasets/eve/                  EveRobot manifest-driven dataset adapter
  everobot_schema.py             v0.1/v0.2 校验、manifest hash 和路径重映射
  models/wan22/fastwam.py        WAM 主模型、action loss mask、outcome token plumbing
  models/wan22/video_lora.py     Video LoRA

scripts/
  train.py                       通用训练入口
  probe_state_line_distance.py   唯一保留的 transition-score 探针（state-line）
  everobot/build_state_line_probe_action_dataset.py  旧 probe converter；不用于标准训练数据
  everobot/build_eve_sidecar.py  EveRobot sidecar 构造
  dexjoco_async/                 多卡 DexJoCo eval / collect
  water_plant/                   water_plant 训练、Eve、rollout wrapper
  spray_water_gr00tstyle/        真机训练、开环 eval、deploy

papers/II/
  experiment_results.md            候选论文结果表
```

## 当前 TODO

1. M1：保留 `water_plant` 历史 pilot；B0/B1 的 200-episode 配对筛选与 E1 fresh-seed 复核已完成。
2. M2：用 v0.2 sidecar/manifest 重建并重跑 `water_plant`/`hammer_nail` round1，和旧 rollout continuation 分开报告。
3. M3：共同初始化 M/M-pair-shuffle/C 与独立 E2 已完成；C 恢复到 B1 的观察成功率，旧 Teacher/pair 目标显著伤害闭环成功率。下一步独立复现 soft-event/value head 并重新设计 steer。
4. M4：新 M3 的 efficacy 与 causality gate 通过后，再用冻结协议完成多轮 rollout -> append -> retrain。
5. M5：M3/M4 通过后，再参考 RL Token 更新 online steer，并参考 World Guidance / AdaJEPA 做 adaptive key-frame 或 tactile prediction 与 world-expert update。
6. M6：真机实验和触觉模块。

## 引用

```bibtex
@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026}
}
```
