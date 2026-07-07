# FITWAM: Failure-Improvement Tactile WAM

> WAM learns state transitions. Physical understanding comes from interaction transitions.

[![上游 FastWAM](https://img.shields.io/badge/上游-FastWAM-111111.svg)](./docs/FASTWAM_UPSTREAM.md)

**相关文档：** [`docs/FASTWAM_UPSTREAM.md`](./docs/FASTWAM_UPSTREAM.md) · [`scripts/README.md`](./scripts/README.md) · [`scripts/water_plant/README.md`](./scripts/water_plant/README.md)

本仓库是在 FastWAM 上做的 failure / self-evolution / tactile 方向 fork。当前主线不是重新做一个通用 WAM，而是围绕失败和交互事件回答一个问题：

**成功数据只告诉模型“怎么做对”，失败轨迹和接触反馈能不能让 WAM 学到“为什么会失败、下一轮怎么避免”。**

当前仿真主战场是 DexJoCo `water_plant`（front + wrist 双视角，proprio 23d）。真机主战场是 `spray_water`（3cam rot6d，已有 FastWAM server/client deploy 链路）。

## 当前结论

DexJoCo `water_plant` 已经能支撑前几步叙事：

| 结论 | 证据 | 位置 |
|------|------|------|
| **M1：failure video 有用** | Text failure 从 `38/100 -> 81/100 -> 82/100`，failure 样本只训 video/context，不训 action | [`results/dexjoco_water_plant_failure_ablation/summary.csv`](./results/dexjoco_water_plant_failure_ablation/summary.csv) |
| **success-only 基线可复现** | 同 pipeline step 6500 为 `70/100`；另一次 200-episode rollout 为 `151/200` | [`evaluate_results/dexjoco/water_plant/step_006500/summary.json`](./evaluate_results/dexjoco/water_plant/step_006500/summary.json), `data/water_plant_rollout_200_step6500_raw/collection_summary.json` |
| **M2/M4：rollout 回灌能涨点** | 基于 step 6500 rollout 继续训练后，闭环为 `163/200` | [`evaluate_results/dexjoco/failure-concate-lora/step_006500/summary.json`](./evaluate_results/dexjoco/failure-concate-lora/step_006500/summary.json) |
| **EveRobot sidecar 已落地** | round1 manifest = 100 条 base success episode + 49 个 failure event；49 个 failure 来自 200 次 rollout | [`src/fastwam/datasets/eve/manifest_dataset.py`](./src/fastwam/datasets/eve/manifest_dataset.py), [`scripts/everobot/build_eve_sidecar.py`](./scripts/everobot/build_eve_sidecar.py) |
| **structured failure 需要稳定性诊断** | C 组 `74/100 -> 59/100 -> 4/100`，中期可用但 late checkpoint 崩 | [`results/dexjoco_water_plant_failure_ablation/README.md`](./results/dexjoco_water_plant_failure_ablation/README.md) |

注意：`70/100`、`82/100`、`151/200`、`163/200` 的 episode 数和 seed 起点不完全一致。README 里只把它们作为当前工程证据和趋势，不把它们写成最终 paper 里的严格同 seed A/B 表。

## 五个 Milestone

| Milestone | 要证明什么 | 当前状态 |
|-----------|------------|----------|
| **M1 Failure video** | failure 轨迹即使不学失败动作，也能通过 video/context 监督提升闭环 | 已有阳性结果：Text failure best `82/100`；action loss mask 已接入 |
| **M2 EveRobot** | failure rollout 需要结构化记录 outcome、event window、manifest 和 provenance | sidecar v0.1 已实现；round1 数据闭环已构造；Eve manifest 专项 eval 仍需补 |
| **M3 Steer token + contrastive/RLT** | 模型需要显式消费 success/failure outcome，而不只是靠 text 后缀 | outcome/steer token 代码路径已接入但默认未启用；contrastive/RLT loss 尚未落地 |
| **M4 Iterative self-evolution** | Train -> Test -> append rollout -> retrain 多轮后继续涨点 | round0 -> round1 已跑通；rollout continuation 当前 `163/200` |
| **M5 Real tactile** | 真机接触期用触觉区分成功/失败，补 RGB 不可见的接触信息 | 规划阶段；`spray_water` deploy 和开环链路已在仓库，触觉分支未实现 |

整体顺序：

```text
M1 证明 failure video 有用
  -> M2 用 EveRobot 把 rollout / event / manifest 管起来
  -> M3 用 steer token + contrastive/RLT 让模型显式利用 outcome
  -> M4 多轮 rollout 回灌，验证持续自进化
  -> M5 真机触觉，把 contact-rich failure 接进同一套 event 叙事
```

## 已实现内容

### 1. Failure data training

普通 LeRobot 数据路径支持用文本 marker 控制 loss：

- `action_loss_zero_if_instruction_contains: "Failed to finish the whole process."`
- 命中 marker 的 failure 样本返回 `action_loss_weight=0.0`
- prompt 会去掉 failure 后缀，避免模型只靠句子字面记忆 failure

关键实现：

- [`src/fastwam/datasets/lerobot/robot_video_dataset.py`](./src/fastwam/datasets/lerobot/robot_video_dataset.py)
- [`src/fastwam/models/wan22/fastwam.py`](./src/fastwam/models/wan22/fastwam.py)
- [`configs/data/dexjoco_water_plant_2cam_text_failure.yaml`](./configs/data/dexjoco_water_plant_2cam_text_failure.yaml)
- [`configs/data/dexjoco_water_plant_2cam_rollout_text_failure.yaml`](./configs/data/dexjoco_water_plant_2cam_rollout_text_failure.yaml)

### 2. EveRobot sidecar v0.1

EveRobot 不改 LeRobot 原始 `data/`、`videos/`、`meta/`，只在 `eve/` 下增加 sidecar：

| 文件 | 作用 |
|------|------|
| `schema_version.json` | EveRobot 版本和兼容说明 |
| `episode_meta.jsonl` | episode 级 provenance：source policy、round、seed、outcome、length |
| `event_meta.jsonl` | failure event window、failure type、action loss 策略、预留 `steer_token` |
| `manifests/*.json` | 每轮训练显式选择 success episode / failure event 子集 |

当前 round1：

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

### 3. Steer / outcome token plumbing

模型里已经有 outcome token 的接线：

- dataset 返回 `outcome_flag`
- `FastWAM._append_outcome_to_context()` 将 outcome embedding 追加到 text context
- checkpoint save/load 支持 `outcome_encoder`

当前默认配置里 `model.outcome_num_classes: 0`，所以这条路径还没在主实验中打开。下一步 M3 应该把它正式变成 steer token，并补：

1. `model.outcome_num_classes: 2`
2. Eve `event_meta.steer_token` 到 `outcome_flag` / steer id 的映射
3. success/failure paired event 的 contrastive loss
4. RLT 或 reward-labeled training，把 rollout outcome 转成额外优化信号

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

### M1 text failure

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
  --episodes 100 --seed 0 \
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
| Structured failure C | 74/100 -> 59/100 -> 4/100 | unstable late checkpoint |
| M4 rollout text failure continuation | 163/200 | continued from success-only step 6500 |
| External pi0.5 | 88.7 +/- 3.1 | from DexJoCo rand-obj table, raw trials not tracked here |
| External GR00T N1.5 | 72.7 +/- 1.2 | from DexJoCo rand-obj table, raw trials not tracked here |

更完整的 M1 ablation 见 [`results/dexjoco_water_plant_failure_ablation/`](./results/dexjoco_water_plant_failure_ablation/)。

### 其他任务

仓库里已有 hammer_nail、pinch_tongs、fold_glasses 的 2cam 配置和部分 eval 输出，但主线 README 暂不把它们写成方法结论：

- `hammer_nail` success-only step 6500：`135/200`
- `pinch_tongs` step 5000：结果在 `evaluate_results/dexjoco/pinch_tongs/step_005000/`
- event transition probe：[`results/event_transition_probe/`](./results/event_transition_probe/)

## 真机与触觉

真机当前以 `spray_water` 为主：

- 训练配置：[`configs/task/spray_water_rot6d_gr00tstyle_uncond_3cam_384_1e-4.yaml`](./configs/task/spray_water_rot6d_gr00tstyle_uncond_3cam_384_1e-4.yaml)
- 脚本：[`scripts/spray_water_gr00tstyle/`](./scripts/spray_water_gr00tstyle/)
- deploy：[`scripts/spray_water_gr00tstyle/wuji/`](./scripts/spray_water_gr00tstyle/wuji/)
- 开环 smoke eval：[`evaluate_results/openloop_smoke_gr00tstyle/`](./evaluate_results/openloop_smoke_gr00tstyle/)

M5 的目标不是在仿真里强行加触觉，而是在真机接触期补上 RGB 难以区分的状态：

```text
tactile planning -> future tactile prediction -> tactile-refined action
```

计划接入方式：

1. 把 tactile episode / contact window 写进 EveRobot event metadata
2. 在 MoT 中加 tactile expert 或 tactile token
3. 在接触期预测 future tactile，并用 tactile outcome 改善 action
4. 与 M3 steer token 和 M4 self-evolution 汇合

## 代码地图

```text
configs/
  data/                         数据配置，含 text failure / rollout failure / Eve manifest
  model/fastwam*.yaml            FastWAM 与 Video LoRA
  task/dexjoco/                  DexJoCo water_plant failure/self-evolution 实验

src/fastwam/
  datasets/lerobot/              原始 LeRobot 固定窗口数据集和 processor
  datasets/eve/                  EveRobot manifest-driven dataset adapter
  models/wan22/fastwam.py        WAM 主模型、action loss mask、outcome token plumbing
  models/wan22/video_lora.py     Video LoRA

scripts/
  train.py                       通用训练入口
  everobot/build_eve_sidecar.py  EveRobot sidecar 构造
  dexjoco_async/                 多卡 DexJoCo eval / collect
  water_plant/                   water_plant 训练、Eve、rollout wrapper
  spray_water_gr00tstyle/        真机训练、开环 eval、deploy

results/
  dexjoco_water_plant_failure_ablation/
  event_transition_probe/
```

## 当前 TODO

1. 补严格同 seed、同 episode 数的 M1 success-only vs text failure 表。
2. 跑 EveRobot manifest round1 的闭环 eval，和 rollout text failure continuation 分开报告。
3. 打开 `outcome_num_classes=2`，把 `outcome_flag` 作为 steer token 做 M3 ablation。
4. 加 paired success/failure contrastive loss 与 RLT/reward-labeled objective。
5. 真机 `spray_water` 补 failure collect、touch metadata 和 tactile branch。

## 引用

```bibtex
@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026}
}
```
