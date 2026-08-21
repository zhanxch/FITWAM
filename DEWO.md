# DEWO

# Motivation 

Vision\-language\-action \(VLA\) models need a general\-purpose method to improve through real\-world deployments via reinforcement learning\. Therefore, PI introduces **RECAP** for VLA *post\-deployment training*\.  

> Post training vs Post\-deployment training vs Test\-Time Training/Adaptation
> 
> 

> ## RECAP流程
> 
> RECAP 包含四个顺序执行的阶段：
> 
> ![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NGQ4YjhlZDFlOWUzOThiOTU0Mzk2ZTE4YmIyMmJjODFfMDdlMjdlNTYzYTZhNjg5MzMyODkzMDM2MDdmZGYyZDZfSUQ6NzY2ODk5Mzg0MDU4MTY0MzE5Ml8xNzg1NzM0MzczOjE3ODU4MjA3NzNfVjM)
> 
> **核心思路**
> 
> 1. **Compute Returns**：对数据集中的每条轨迹，按 𝐺𝑡 =𝑟𝑡 \+𝛾 ⋅𝐺𝑡\+1 逆序计算折扣回报，生成 sidecar 文件而不修改原始数据。
> 
> 2. **Value Model SFT**：训练一个价值模型（基于 VLM backbone \+ Value Head），使其从观察（图像 \+ 语言指令）预测归一化回报。
> 
> 3. **Compute Advantages**：利用训练好的价值模型，按 𝐴𝑡 =normalize⁡\(𝑟𝑡:𝑡\+𝑁\) \+𝛾𝑁 ⋅𝑉⁡\(𝑜𝑡\+𝑁\) −𝑉⁡\(𝑜𝑡\) 计算每个时间步的优势，并根据分位数阈值将样本标记为正/负。
> 
> 4. **CFG Training**：使用优势标签训练策略模型——正样本（高优势）作为条件输入，负样本（低优势）作为无条件输入，实现 classifier\-free guidance 策略优化。
> 
> ## RECAP 工作原理
> 
> **RECAP 核心组件**
> 
> 1. **回报计算（Return Computation）**
> 
>     - 对 SFT 数据集（全部成功轨迹）：每步奖励 𝑟𝑡 =−1，终止步 𝑟𝑇 =0
> 
>     - 对 rollout 数据集（含失败轨迹）：失败轨迹终止步 𝑟𝑇 =𝑟fail（如 −300）
> 
>     - 折扣因子 𝛾 默认为 1\.0（无折扣）
> 
> 2. **价值模型（Value Model）**
> 
>     - 基于 SigLIP2 视觉编码器 \+ Gemma3 语言模型 \+ 可学习 Critic Expert
> 
>     - 采用分布式价值预测（Categorical Value Distribution），默认 201 个 bin
> 
>     - 输出范围 \[−1,0\]（归一化后的回报空间）
> 
> 3. **优势估计（Advantage Estimation）**
> 
>     - N 步前瞻优势：𝐴𝑡 =normalize⁡\(𝑟𝑡:𝑡\+𝑁\) \+𝛾𝑁 ⋅𝑉⁡\(𝑜𝑡\+𝑁\) −𝑉⁡\(𝑜𝑡\)
> 
>     - 分位数阈值：top 𝑋% 的样本标记为正样本（默认 𝑋 =30）
> 
>     - 支持多 GPU 分布式推理
> 
> 4. **Classifier\-Free Guidance（CFG）训练**
> 
>     - 基于 OpenPI \(pi0\.5\) 策略模型
> 
>     - `positive_only_conditional` 模式：仅正样本作为条件输入，负样本一律无条件
> 
>     - 正样本以 `unconditional_prob` 概率随机转为无条件（默认 0\.1），实现 dropout 正则化
> 
>     - 推理时通过 `cfgrl_guidance_scale` 控制引导强度
> 
> 



However, VLAs often struggle to adapt to new environments or generalize to novel tasks beyond the distribution of expert demonstrations, without explicitly collecting large\-scale task\- and environment\-specific action data\. **World Action Models** , built upon a pretrained image\-to\-video diffusion backbone,  leverage rich spatiotemporal priors to jointly generate future frames and actions conditioned on language instructions and observations, which shifts action learning from dense state–action imitation to inverse dynamics—aligning motor commands with predicted visual futures\.

WAMs give a promising structure for generalist embodiment foundation models\. This raises the question: 

*Beyond VLA post\-deployment training like RECAP, do WAMs need a new paradigm based on world modeling?* 

It is necessary to do so, but how? During pretraining, WAMs benefit from the world modeling task and enable few\-shot embodiment adaptation\. Therefore, we believe that,

*World modeling is not preparation for policy learning; world modeling is policy learning itself\.* 

We are looking for a solution, maybe just named it **Direct Experience World\-modeling Optimization \(DEWO\)\. **



# Related Works

## VLA \& WAM

## Post\-deployment training \& Self\-improving

World Model post\-training ✔

VLA post training ✔

World Model for VLA post training ✔

World Action Model post training 目前没有

1. Agent 框架

    1. RoboCat: A Self\-Improving Generalist Agent for Robotic Manipulation

2. 多模型框架

    1. TACO: TActile World Model as a Self\-COrrector for Scalable VLA Post\-Training 

    2. WA\-RL: World\-Action Model Reinforcement Learning with Reconstruction Rewards and Online Video SFT

    3. World Action Verifier: Self\-Improving World Models via Forward\-Inverse Asymmetry

3. 单模型

    1. TTT/TTA 实时改进

        1. AdaJEPA: An Adaptive Latent World Model

        2. RoboTTT: Context Scaling for Robot Policies

    2. 持续学习

        1. Towards Long\-Lived Robots: Continual Learning VLA Models via Reinforcement Fine\-Tuning

    3. 后训练 https://zhuanlan\.zhihu\.com/p/2041183022196773671 

        1. $\pi^{*}_{0.6}$: a VLA That Learns From Experience

## Tactile\-Aware Robot Learning





# Experiments

## Probe experiments

成功和失败数据一块训练的world 的latent 更generalist

There are two kinds of traj: success vs failure\. The failure rollouts contain richer unseen interaction vision information, which maybe enable the model to adapt beyond static training data\.



> 这个实验中，B0和B1 是在baseline基础上续训。其中失败数据会提取出failure event，只用于训练videoDiT。而B0为了和B1消融对比，所以B0和B1 不是完全随机抽一组数据构造batch，而是2 primary \+ 2 aff 数据，primary 指的是包含video和action Loss 的成功轨迹，aff指只用于训练video 的轨迹数据。本次实验为了控制变量，提示词均使用一致的，未在输入task 中提供额外标注。
> 
> 

结论：我认为本实验结果可以说明部署后再继续优化Video 是有效的，且用失败数据训练videoDiT 是额外有效的




# 实验计划

目前暂以失败数据截取event 训练video 失败数据做CFG 为DEWO 方案

1. 在多个task，multi\-task上验证并消融 DEWO，优先验证

2. 对比RECAP、RLT、steam等后训练方案

3. 对比Joint Video \& Action Denoising、IDM、FastWAM 这三种WAM架构范式下的DEWO效果 \(FastWAM提供了另外两种，可以直接在FastWAM上做这个我觉得\)

4. 同步进行 相关分析，参考幻觉设计

## DEWO 消融

**分工**：方案必要性（motivation）→ **设计消融**；失败动力学 / world latent 学到了什么 → **表征分析**（见 Analysis），不靠消融单独完成。

实现：[`FastWAM/`](FastWAM/)，文档 [`FastWAM/docs/DEWO.md`](FastWAM/docs/DEWO.md)。**不做全因子交叉**；主链按成分累加。

### 消融因子

| 因子 | 含义 | 验证什么 |
|------|------|----------|
| Post-training | 从 S0 **success-only** 续训（B0） | 涨点是否只是「再训一会儿」 |
| Failure rollouts | 部署失败轨迹写入 world（video） | 失败经验是否必要 |
| Failure events | 失败轨迹截 event，而非整段（仅去尾） | event 截取是否必要 |
| Failure Action loss | 失败样本是否加 action loss | 失败要不要直接模仿 action |
| Success CFG | 成功侧 outcome 条件 + dropout→base | success 条件引导是否必要 |
| Failure CFG | 失败侧 outcome/failure phrase + dropout→base | failure 条件引导是否必要 |

说明：首列 **Post-training** 特指 B0 的 success-only 续训；B1-* 同样从 S0 续训，但不勾此列，以 **Failure rollouts** 区分。Success CFG / Failure CFG 同时打开 = 双后缀 `dual_outcome`（见下）。

### 主消融表

| Setting | Post-training | Failure rollouts | Failure events | Failure Action loss | Success CFG | Failure CFG | Result |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|---|
| FastWAM S0 | | | | | | | 77% |
| B0 | ✓ | | | | | | |
| B1-remap-video | | ✓ | | | | | 79.0% ± 5.4% |
| B1-remap-L_act | | ✓ | ✓ | ✓ | ✓ | ✓ | |
| B1-remap | | ✓ | ✓ | | | | |
| **B1-remap-CFG** | | ✓ | ✓ | | ✓ | ✓ | **86.0% ± 5.7%** |

主表 Result 默认取 **test ckpt**（训练终点 `step_006500`，除非另行注明）。协议：官方 DexJoCo **4×50**（seeds `0–49` × 4 repeats；blocking / `replan_steps=25` / `max_env_steps=1500`），报告 **mean ± std**（跨 4 次独立跑）；`pooled` 为 `successes/200` 参考值。Val-best 按训练日志 `best_checkpoint.json`（metric=`val_base_loss`, mode=`min`）。

#### FastWAM S0

| Role | Ckpt | Result (mean ± std) | Pooled | Status |
|------|------|---------------------|--------|--------|
| Test | `step_006500` | 77% | — | 主表沿用值；官方 4×50 待补 |
| Val-best | — | — | — | 未单独标定 / 未评 |

#### B0

| Role | Ckpt | Result (mean ± std) | Pooled | Status |
|------|------|---------------------|--------|--------|
| Test | — | — | — | 未训 / 未评 |
| Val-best | — | — | — | 未训 / 未评 |

#### B1-remap-video

Run: `2026-08-06_22-18-58_B1-remap-video`。Val-best：`step=5500`（`val_base_loss=0.2223`；权重 `best.pt`，写入时为 `__endqueue__` 状态）。

| Role | Ckpt | Result (mean ± std) | Pooled | Status |
|------|------|---------------------|--------|--------|
| Test | `step_006500` | **79.0% ± 5.4%** | 158/200 | 官方 4×50 完成 |
| Val-best | `step_5500` / `best.pt` | — | — | 官方 4×50 排队中 |
| Other | `step_005000` | ~52%（1/4 runs） | 26/50 | 官方 4×50 进行中 |
| Other | `step_003000` | 55.0% ± 5.7% | 110/200 | 官方 4×50 完成 |
| Other | `step_000500` | 0.5% ± 0.9% | 1/200 | 官方 4×50 完成 |

#### B1-remap-L_act

Run: `2026-08-07_13-43-32_B1-remap-L_act`（训练中）。相对 B1-remap-CFG：failure auxiliary 打开 action loss。

| Role | Ckpt | Result (mean ± std) | Pooled | Status |
|------|------|---------------------|--------|--------|
| Test | `step_006500` | — | — | 训练进行中 |
| Val-best | — | — | — | 训练进行中 |

#### B1-remap

Run: `2026-08-06_11-56-36_B1_frozen20260718_remap`。Val-best：`step=6500`（与 test 重合；`val_base_loss=0.2309`）。

| Role | Ckpt | Result (mean ± std) | Pooled | Status |
|------|------|---------------------|--------|--------|
| Test | `step_006500` | — | — | 官方 4×50 待跑 |
| Val-best | `step_006500`（同 test） | — | — | 官方 4×50 待跑 |
| Smoke（非官方） | `step_006500` | 84%（42/50） | — | seed `28704217`，n=50 |

#### B1-remap-CFG

Run: `2026-08-06_22-08-51_B1-remap-cfg`。Val-best：`step=5000`（`val_base_loss=0.2302`；`best_checkpoint.json` → `step_005000.pt`）。

| Role | Ckpt | Result (mean ± std) | Pooled | Status |
|------|------|---------------------|--------|--------|
| Test | `step_006500` | **86.0% ± 5.7%** | 172/200 | 官方 4×50 完成 |
| Val-best | `step_005000` | ~83.3% ± 0.9%（3/4） | 125/150 | 官方 4×50：84/84/82，run4 进行中 |
| Other | `step_003000` | 19.5% ± 5.4% | 39/200 | 官方 4×50 完成 |
| Other | `step_000500` | 4.0% ± 2.5% | 8/200 | 官方 4×50 完成 |


当前 DexJoCo / DEWO v2 入口是开源 224 / z-score：

```bash
TASK=fold_glasses GPUS=4,5,6,7 bash scripts/dewo_v2/collect_opensource_4x50.sh
```

详见 `docs/DEWOV2.md`。下面 `scripts/dewo/` 流程已过时，不要再指定已删除的 `*_uncond_2cam_384_1e-4` 任务。

# 或分步：
# bash scripts/dewo/collect_baseline_rollouts.sh hammer_nail   # 默认 200 eps, max_env_steps=1500
# bash scripts/dewo/prepare_hammer_nail.sh
# export FASTWAM_DEWO_INIT_CHECKPOINT=${CHECKPOINT}
# bash scripts/dewo/run_hammer_nail_ablation.sh hn_dewo_full
```

**不要**再用 `hammer_nail_rollouts_pi05` 作为 full DEWO 的 rollout；那是 Pi 数据，仅可用于对照。

统一：`max_steps=7000`，`save_every=500`；每个 setting 用 `scripts/dewo/select_checkpoint.py` 选一个评测 ckpt 写入 `selection.json`。

消融（同一份 baseline rollout/manifest）：

```bash
bash scripts/dewo/run_hammer_nail_ablation.sh hn_dewo_full
# 或全部消融：
bash scripts/dewo/run_all_hammer_nail_ablations.sh
```

## FastWAM \- DEWO

本实验表格只需验证DEWO是涨点的即可。实现入口：

```bash
cd FastWAM
TASK=water_plant GPUS=4,5,6,7 bash scripts/dewo_v2/collect_opensource_4x50.sh
```

# hammer 同理（baseline 训完后）
# export RUN_DIR=... CHECKPOINT=...
# bash scripts/dewo/run_full_dewo.sh hammer_nail

bash scripts/dewo/eval_selected.sh hammer_nail baseline   # NUM_EPISODES=5
bash scripts/dewo/eval_selected.sh hammer_nail dewo
```

|ckpt|waterplant<br>||hammernail||Pinch Tongs <br>（优先级中高，属于困难任务，pi和gr00t 表现不行）||Fold Glasses<br>（优先级低，因为论文中gr00t效果明显差于pi，原因不明）||Pick Bucket<br>（优先级低，属于简单任务，和waterplant，hammernail 定位较为重复）||
|---|---|---|---|---|---|---|---|---|---|---|
|对单个setting来说，统一选择ckpt 的方案，不要遍历测试。但是记录好测试了哪个ckpt，并保存好中间其他ckpt|baseline|DEWO|||||||||
|5 eps/ 7K|||||||||||

Pinch Tongs 骨架：`FastWAM/configs/task/dewo/pt_dewo_full.yaml`（需先准备 rollout/baseline）。

Multitasks

|ckpt|waterplant \+ hammer nail<br>||waterplant\+hammer\+PT||multi5||
|---|---|---|---|---|---|---|
|对单个setting来说，统一选择ckpt 的方案，不要遍历测试。但是记录好测试了哪个ckpt，并保存好中间其他ckpt|baseline|DEWO|||||
|5 eps/ 7K \* tasks|||||||

## 

## RECAP \- DSRL \- RLT on Pi and FastWAM

对比其他后训练方案。

RECAP 是Pi0\.6 提出的VLA 的post\-deployment training 代表VLA RL 的类型。

DSRL 是 真机RL 的方案，代表偏robotic RL 的类型。

RLT 是实时改进的方案，代表online RL 的类型。\(优先级相对低，RLT 知名度也低\)



||waterplant<br>||hammernail||waterplant \+ hammer nail||
|---|---|---|---|---|---|---|
|RECAP|pi|Fastwam|||||
|DSRL|||||||
|RLT|||||||
|DEWO|\-||\-||\-||

## DEWO on Joint \- IDM \- FastWAM

对比不同类型WAM，直接基于FastWAM提供的消融架构做

||waterplant|hammernail|waterplant \+ hammer nail|
|---|---|---|---|
|Joint||||
|IDM||||
|FastWAM||||

## Analysis

实证分析「失败 video 写入 world expert」的效果（water_plant S0 vs B1-87%；特征 `video_kv_last_pooled`）。产物：`results/failure_visual_coverage_abcde_20260807/`。

### A. Failure → success gallery 的相对 OOD

- **指标**：failure 帧到 success gallery \(G_s\) 的 kNN / success 自 1NN（跨模型须相对化）。
- **结果**：`fail / self-NN`：S0 **2.54 → B1 1.72**。

![A relative OOD bars](results/failure_visual_coverage_abcde_20260807/A_relative_ood_bars.png)

![A relative OOD hist](results/failure_visual_coverage_abcde_20260807/A_relative_ood_hist.png)

### B. ε-ball 覆盖率

- **指标**：ε = \(G_s\) median 1NN × 1.5；failure 落在 \(G_s\) 内的比例。
- **结果**：约 **53%** 在球内，约 **47%** 在球外。（\(G_{s\cup f}\)→100% / lift 近同义反复，不作证据。）

![B coverage bars](results/failure_visual_coverage_abcde_20260807/B_coverage_bars.png)

### C. PCA

![C PCA S0](results/failure_visual_coverage_abcde_20260807/C_pca_S0_video_kv_last_pooled.png)

![C PCA B1](results/failure_visual_coverage_abcde_20260807/C_pca_B1_video_kv_last_pooled.png)

### D. 沿进度的 OOD 残差（非 PSNR）

- **指标**：progress 分箱下 failure/success 到 \(G_s\) 的 mean kNN。
- **结果**：相对比晚段 **S0 3.12 → B1 2.10**。

![D S0 vs B1](results/failure_visual_coverage_abcde_20260807/D_knn_vs_progress_S0_vs_B1.png)


# DEWO 流程（对照 RECAP）

DEWO（Direct Experience World-modeling Optimization）面向 **World Action Model** 的 post-deployment 优化：不走 RECAP 的 return → value → advantage 链路，而是直接用部署经验中的失败视觉事件强化世界建模，并用 outcome 条件做 CFG 策略引导。

DEWO 包含四个顺序执行的阶段：

**核心思路**

1. **Collect Rollouts**：用已部署的 WAM baseline 在环境中采集 rollout，得到成功与失败轨迹；失败轨迹可截断超时尾部（如 trim 8s），不修改原始专家/SFT 数据。

2. **Prepare Failure Events**：对失败轨迹做 event 定位与标注——截取 soft-subtask / periodic-tail 等 failure event 窗口，写入 sidecar 与 DEWO manifest；并为失败样本附加 failure phrase（成功样本附加 `\nOutcome: success`）。

3. **World Modeling Continue-Train**：以 baseline checkpoint 为初始化，混合训练成功轨迹与失败 event：成功样本同时优化 video + action；失败样本默认仅优化 video（可选再开 failure action loss），把失败交互的时空先验直接写回世界模型。

4. **Dual-Outcome CFG Training**：用 outcome 文本做 classifier-free guidance——成功样本以 success 后缀为条件、失败样本以 failure 短语为条件，两侧均以 `unconditional_prob` dropout 到干净的 base task prompt；推理时用 success（或 base）相对 base 做 guidance。

## DEWO 工作原理

**DEWO 核心组件**

1. **部署数据采集（Rollout Collection）**
    - 输入：同一 task 上训好的 FastWAM baseline checkpoint
    - 输出：含成功/失败的 rollout 数据集（如 `data/<task>_rollout_baseline_<step>_trim8s`）
    - 失败轨迹可按长度/超时规则去尾，保留有效交互段

2. **失败事件准备（Failure Event Preparation）**
    - Hammer：`soft_subtask_value_horizon_v1`（状态 soft-event + 成功时长上界 `α·L*` + 可选 value drop）
    - Water plant：periodic-tail / 简化全段失败视频
    - 生成 event sidecar（parquet）与训练 manifest；失败样本标注 failure phrase

3. **世界建模优化（World Modeling Optimization）**
    - 成功 / 专家窗口：video loss + action loss，条件文本 `base + \nOutcome: success`
    - 失败 event 窗口：默认仅 video loss（action 关），条件文本 `base + Failed to finish...`（或 `\nOutcome: failure`）
    - 核心主张：*World modeling is not preparation for policy learning; world modeling is policy learning itself.*

4. **双后缀 Classifier-Free Guidance（Dual-Outcome CFG）**
    - 模式：`dual_outcome` / `cfg_mode=dual_outcome`
    - 成功侧：主要用 success 后缀；以 `unconditional_prob=0.1` dropout → base
    - 失败侧：主要用 failure 短语；同样以 0.1 dropout → base
    - 无条件上下文始终是干净任务句（如 `Hammer the nail.`），而非“去掉 advantage 标签”
    - 推理：用 success（或 base）相对 base 控制引导强度

**与 RECAP 的对照**

| | RECAP（VLA） | DEWO（WAM） |
|---|---|---|
| 经验信号 | 折扣回报 → 价值模型 → N-step 优势 | 失败视觉 event + outcome 文本 |
| 正/负划分 | 优势分位数（top X% 为正） | 轨迹结果：success vs failure |
| 条件输入 | `\nAdvantage: positive` 等 | `\nOutcome: success` / failure phrase |
| 无条件输入 | 负样本 + dropout | base task prompt（两侧均可 dropout） |
| 优化对象 | 策略（action）CFG | 世界模型（video）为主 + 策略引导 CFG |
| 默认配方 | positive_only_conditional | dual_outcome（两侧都带 outcome） |

实现入口见 [`FastWAM/docs/DEWO.md`](FastWAM/docs/DEWO.md)；默认 full 设置为 `hn_dewo_full` / `wp_dewo_full`。

## Fold-glasses DEWO v2 口径（2026-08-12）

### 训练条件

DEWO v2 使用三类文本条件，训练采样概率为 outcome / FAST / base = 0.4 / 0.2 / 0.4：

- outcome：任务句 + `Successful execution.` 或 `Failed execution.`
- FAST：任务句 + 由该样本真值 action chunk 编码的 FAST token
- base：干净任务句，不带 outcome 或 FAST 后缀

FAST 在当前实现中是 **training-only privileged action-conditioning auxiliary**。它让共享的 VideoDiT/ActionDiT/MoT 在训练时接触 action-token 与视频变化的对应关系，但部署时不提供未来真值 action，也不把 FAST token 用作 CFG 正分支。部署 CFG 使用 success 相对 base：

\[
\hat\epsilon = \hat\epsilon_{base}
+ s\left(\hat\epsilon_{success}-\hat\epsilon_{base}\right).
\]

其中 `s=1` 等价于普通 success 条件推理，`s=0` 等价于 base 条件，`s>1` 才是 success-vs-base guidance。

### FAST 主张边界

可以主张：FAST 是训练期的 privileged auxiliary，用于增强共享 WAM 表征中的 action-video 对齐；推理不依赖未来真值 action，因此没有部署时的真值泄露。

不能直接主张：FAST 已经把部署策略变成 inference-time action-conditioned world model。当前部署没有提出候选 action，再经 VideoDiT 预测未来并选择 action 的闭环。并且当前 FAST 样本仍优化 ActionDiT action loss，ActionDiT 可以从真值 action token 获得 shortcut；因此“FAST 的收益来自 VideoDiT/action-video alignment”必须由消融证明，不能只靠架构解释。

最低限度的机制消融：同 manifest、初始化、batch、训练步数比较 `outcome+base` 与 `outcome+FAST+base`，并加入 shuffled-FAST 控制。如果 shuffled FAST 与真实 FAST 同样有效，证据更支持文本正则化，而不是 action-video 对齐。checkpoint 选择不得只看 FAST-conditioned validation loss。

### 冻结评测协议

同一比较只使用一份训练 manifest 和一个预先选定的 checkpoint。对该 checkpoint 使用相同环境 seeds、diffusion seed、归一化、文本缓存、replan 与最大步数，依次评测：

1. base prompt，`scale=1`
2. success prompt，`scale=1`
3. success-vs-base CFG，预注册一个固定的 `scale>1`

入口为 `scripts/dewo_v2/eval_cfg_ablation.sh`（`TASK=fold_glasses`）；base 与 success/base 成对任务配置分别位于 `configs/eval/dexjoco/fold_glasses_dewo_v2_base/` 和 `configs/eval/dexjoco/fold_glasses_dewo_v2_cfg/`。CFG scale 可以在 validation seeds 上预选一次，最终结果必须在未参与选参的新 test seeds 上报告。
