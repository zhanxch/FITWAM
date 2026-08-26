# DEWO 计划（现行：v7）

栈不变：opensource **224 / z-score**、`replan=24`、官方评测 **4×50**（seeds `0..49` × 4）。无 LoRA。无 384 / min-max。

后缀锁死，不再改词：

- 成功：` Successful execution.`
- 失败：` Failed execution.`

---

## A. 现行版本简谱（v2 / v5 / v6）

仓库、配置和实验资产当前保留 **v2（数据与全参配方）**、**v5（冻本体残差壳）**、**v6（pair 进 batch、残差仍减本体）** 和 **v7（pair 进被放大的残差）**。旧路线不再作为实现、入口或实验资产维护。

| 版本 | 是什么 | 状态 | 为何留下 / 丢掉 |
|---|---|---|---|
| **v2** | recoverability **pair** 流水线 + 全参 DiT。`INIT=scratch` 或 `s0` 全参 `1e-4`。primary = 成功 episode ∪ pair 成功窗（动作损失开）；aux_success = 同窗关动作损失；aux_fail = 失败窗 | **保留：数据协议** | pair 扫描、Eve manifest、opensource collect/prepare 仍是后续版本的数据层。全参 S0 continue 在 fold_glasses 上崩过（约 64.5% → 29.5%），**不要用 v2 全参去 continue 强 S0**。 |
| **v5** | **冻整网 S0**，只训 video/action 文本侧 `cross_attn.k/v` 残差。video 钉 base；动作 CFG dropout `0.9,0,0.1`；video 残差靠动作损失 + identity lock；不抽 fail/aux | **保留：架构** | \(w=1\) 可对齐本体（water_plant mixed 官方约 90%）。正类是「随机成功 + Successful」——与本体同分布，\(w>1\) 掉点（1.5→78%，2→72%）。**壳留下。v6/v7 不换这套 video-context 约束。** |
| **v6** | v5 冻本体 CFG 壳 + v2 pair event；D0 / D+ / D_fail 一个 shuffle 池；残差仍是 \(\varepsilon_+-\varepsilon_0\)；D_fail 只做 video BC | **保留：对照** | pair 只决定谁进 batch，不进入被放大的向量。water_plant 官方最好一格 ≈ S0 86%，引导开多了掉点。 |
| **v7** | 同 v6 壳与池子；残差换成 \(\varepsilon_+-\varepsilon_-\)；失败动作只回归到失败句（\(\varepsilon_-\) 只作减数，永不执行） | **保留：当前主线** | 把 pair 的负支写进 CFG 减数。原点仍是 S0。 |

**v5 在训什么：** 不是 video latent / 世界模型。`lambda_video=0`，主干全冻。动的是 video/action **文本交叉注意 K/V** 上的低秩残差（`uncond_down` / `uncond_up`）。Video adapter 改的是 \(K,V=W\,c+\Delta(c)\)，\(c\) 是文本，不是 latent。v5 等于：拿普通成功 rollout 训动作 CFG adapter；失败数据不用；没有 video BC。

**v5 必须保住的约束：** 推理时 video 两路都是 **base**。\(p(\text{video tokens}\mid c_{\text{base}},\mathrm{obs})\approx p_{S0}\)。Video 作为 action 的 context 分布不能大变。v6 继承这条，不改架构。

**继承关系：** v6 = **v5 的冻本体 CFG 壳与 video-base 约束** + **v2 的 pair event（非对称损失）** + **完整成功 rollout 留在池子里做 base 对齐**。不解冻主干，不把普通成功 rollout 当 CFG 正类，也不做 12:4 配比。

---

## B. DEWO v6（详细）

### B.1 目标

CFG 只在残差指向 **相对 \(\varepsilon_0\) 更好** 时放大：

```
ε_cfg = ε_0 + w · (ε_+ − ε_0)
```

- \(\varepsilon_0\)：adapter **关** + 任务 base = bit-exact S0。
- \(\varepsilon_+\)（动作）：adapter **开** + 成功后缀，数据是 **pair success event**（恢复分支上的成功续行）。**video 仍 base。**
- 另训 pair failure event 上的 **视觉对照**（失败后缀 + video BC，不开动作 BC）。走 \(\Delta(\text{failure})\)，不改 \(\Delta(\text{base})\)。
- \(w=1\)：不混合，只走 \(\varepsilon_0\)。官方默认报这个。
- \(w>1\)：只放大动作支的「更好 − 当前」。残差与本体分不开就不开。

相对性来自 **同一前缀、失败续行 vs 成功续行**，不靠换一个英文词。成功/失败后缀与 v2/v5 相同，锁死。

高成功任务（water_plant 本体 ~90%）：

- **约束**：\(w=1\) 的 4×50 ≥ 本体。
- **改进**：只在可分的恢复态上开小 \(w\)。
- 假阳性有害。完整成功 rollout 不标 \(+\)，只做 base 对齐。

### B.2 三类样本（一个 shuffle 池，无比）

v5 只有一类：普通成功 episode。v6 在同一架构上往池子里加 pair，损失非对称。像 v5 一样 **shuffle，不配 12:4**。丢掉 aux_success 复本（与 \(D_+\) 同窗、关动作损失，不是更严优势）。

| 集合 | 是什么 | video 文本 | action 文本 | video loss | action loss | 作用 |
|---|---|---|---|---|---|---|
| **\(D_0\)** | 完整成功 rollout | **base** | **base** | 开（对齐） | 开（对齐） | adapter 开 + base ≈ 本体；保住 \(p(\text{video}\mid\text{base})\) |
| **\(D_+\)** | pair **success 事件窗** | **base** | ` Successful execution.`，dropout 与 v5 相同 `0.9,0,0.1` | 不开专用 BC（只 identity lock） | 开 | 动作 CFG 正类。相对性在数据（分支点成功续行），不在换词 |
| **\(D_{\text{fail}}\)** | pair **failure 事件窗** | ` Failed execution.` | 不算损失 | **开** | **关** | 在 failure 通道学失败视觉；不污染 base |
| 旧 aux_success | 与 \(D_+\) 同窗、动作损失关 | — | — | — | — | **丢掉** |

切窗仍是 v2 recoverability 33 帧。

**为何 \(D_+\) 的 video 不用成功后缀：** 推理 video 两路都是 base。成功后缀进 video 会训 \(\Delta_{\text{video}}(\text{success})\)，推理用不上，还会把「这段会成功」写进观测。动作 CFG 要的是同一套观测编码、只改动作条件。成功画面的世界模型已由 \(D_0\) 在 **base** 通道对齐。不要给 \(D_+\) 再做一套「video loss + 成功后缀」（那是旧 aux_success）。

**为何 \(D_{\text{fail}}\) 用 failure 后缀而不是 base：** 失败帧 + base + video BC 会直接把 \(p(\text{video}\mid\text{base})\) 推向失败未来，冻住的动作头会 OOD（v5 丢掉 aux 的真正原因）。failure 后缀让残差走 \(\Delta(\text{failure})\)；\(\Delta(\text{base})\) 仍由 \(D_0\)/\(D_+\) 的 identity lock 按住。当前推理 video 仍 base，这条残差暂不参与动作 CFG；它是视觉对照，也给以后视觉 CFG 留通道。

**为何 \(D_{\text{fail}}\) 不要动作 BC：** 失败续行的动作是反例。模仿它会把 \(\varepsilon_+\) 拉向失败动作。

**Identity lock：** 只打在 **video 文本为 base** 的样本上（\(D_0\)、\(D_+\)）。不要打在 \(D_{\text{fail}}\) 上（会把 \(\Delta(\text{failure})\) 抹掉）。

**推理（v6.0）：** 动作 CFG = ` Successful execution.` vs base；**video 两路仍 base**。失败句不进评测观测。以后要做视觉 CFG 再开 video 条件支。

### B.3 训练（v5 壳，换数据）

保留：整网冻；adapter 只挂 video/action `cross_attn.k/v`；推理与 \(D_+\) 的 video 钉 base；**不 BC 失败动作**；**失败 video BC 不用 base 文本**。

相对 v5：

1. **池子：** \(D_0 \cup D_+ \cup D_{\text{fail}}\) 一起 shuffle，**无比**。不要 `primary_per_batch=12`。不要 role-balanced 3:1。
2. **丢掉：** aux_success 复本。不要把普通成功 rollout 当 CFG 正类（那是 v5 的错正类）；它们只当 \(D_0\)。
3. **CFG / dropout：** \(D_+\) 动作与 v5 对齐 `0.9,0,0.1`（成功后缀 vs base）。\(D_0\) 已是 base，不再抽成功后缀。\(D_{\text{fail}}\) 文本 `1.0,0,0` 失败后缀，不向 base dropout。无 FAST。
4. **损失：** \(D_0\) 可开 video + action BC（对齐）。\(D_+\) 开 action BC，video 只 identity lock。\(D_{\text{fail}}\) 只开 video BC。全局 `lambda_video` 可按样本打开，不要对 \(D_+\) 做成功后缀下的 video BC。
5. **动作残差 L2** `action_residual_lock_lambda=0.05`（主要打在 \(D_+\)；\(D_0\) 上 \(\Delta(\text{base})\) 本就该小）。
6. ckpt adapter-only，兼容 v5 loader；`recipe=v6`。

超参可先同 v5：`lr=1e-4`，`max_steps=1500`，`rank=α=16`，`INIT=s0` only。

### B.4 推理与门控

| 设置 | 行为 |
|---|---|
| \(w=1\) | 本体 bypass（adapter 关 + base） |
| \(w>1\) | 建议 **1.2**，不要默认 2.0 |
| 正分支 | 任务句 + ` Successful execution.` |
| video | 两路都是 `cfg_base_prompt` |
| 门控 | NFE0 执行段 RMS \(E\)。\(E>\tau\) 用 \(w\)，否则 mix=0 |

「较高限制」= **更严的何时开 CFG**，不是更大的 \(w\)。正类是局部分支点，残差只在那类状态有意义。

评测顺序：先 \(w=1\) 官方 4×50；再筛 \(w=1.2\) + 校准 \(\tau\) 的 1×50。未校准的 CFG=2 不得当 v6 主结果。评测 yaml 用与训练相同的成功后缀，不要 Recovered。

### B.5 adaptive \(\tau\)：从训练数据估，与 event 筛选同一件事

门控 \(E>\tau\) 才开 CFG，和「只有 recoverability success event 才算 \(D_+\)」是同一把尺子：都是 **只在关键分支点动手**。  
v5 筛到 `tau=0.05` 的 **效果** 对（全程里只有很短一段被引导），那个 **数字** 不能当先验——那是错误残差（全体成功 + Successful）上网格搜出来的。event 只有约 33 帧，先验必须 **尽可能严，同时尽量盖住这些窗**。

**数据（训完后一遍 infer，不跑环境、不网格搜）：**

- \(E_+\)：每个 \(D_+\) 训练窗（成功 recoverability event）的 NFE0 执行段 RMS。样本少，每个窗都要用。
- \(E_0\)：普通成功 episode 上 **非 event** 前缀，同一套 RMS（对照：这些不该开 CFG）。

**规则：在盖住 event 的前提下把 \(\tau\) 提到最高。**

\[
\tau=\max\{\,t:\ \widehat{\mathrm{recall}}_+(t)\ge\rho\,\}
=q_{1-\rho}(E_+)
\]

默认 \(\rho=0.90\) → \(\tau=q_{0.10}(E_+)\)：大约 90% 的 33 帧窗仍触发，其余当残差最弱的 event 放弃。  
若窗更少、更怕漏：\(\rho=0.95\) → \(q_{0.05}(E_+)\)。不要用中点公式（会切掉太多 \(E_+\)）。

然后用 \(E_0\) **检验是否够严**，不拿 \(E_0\) 再去网格搜：

- 记 \(\mathrm{FPR}_0=P(E_0>\tau)\)。目标与 0.05 那档的 **稀疏** 同类：对照前缀几乎不开 CFG，例如 \(\mathrm{FPR}_0\le 0.05\)（可记 `strict_fpr=0.05`）。
- 若 \(\tau=q_{1-\rho}(E_+)\) 仍让 \(\mathrm{FPR}_0\) 过大：残差还不够 event-specific → **禁止 adaptive**，只报 \(w=1\)。不要为了压 FPR 把 \(\tau\) 抬到漏掉大半 event。
- 若 \(\mathrm{FPR}_0\) 已经很小：就用这个 \(\tau\)，不要再往上加码（那会开始漏 33 帧窗）。

禁止：在官方 `0..49` 上扫 tau；把 0.05 / 0.5 当默认；用 val 成功率选 tau。

`ADAPTIVE_CFG_TAU=auto` 只读 `RUN_DIR/adaptive_cfg_tau.json`（写入 \(\tau,\rho,\mathrm{recall}_+,\mathrm{FPR}_0\)）。没有该文件或 `separable=false` → 不启用 adaptive。

### B.6 代码落点（实现时按此改）

| 位置 | 内容 |
|---|---|
| Hydra | `dexjoco/dexjoco_dewo_v6_offline_b1_jump_fast_uncond` |
| `train.sh` | `DEWO_VERSION=v6`，不新建 per-task `.sh` |
| 后缀 | 与 v2/v5 相同：`CFG_SUCCESS_SUFFIX=' Successful execution.'`，`CFG_FAILURE_SUFFIX=' Failed execution.'`。v6 **不要**覆盖成 Recovered |
| Eve dataset | 保留完整成功 episode（\(D_0\)）+ success-event-primary（\(D_+\)）+ failure-event（\(D_{\text{fail}}\)）；丢掉 aux_success。不要用会删掉 episode 的 `unit_filter=recoverability_events` |
| sampler | 与 v5 一样 shuffle，**不要** `primary_per_batch=12` / role-balanced |
| 损失 | \(D_0\)：video+action BC、文本 base。\(D_+\)：action BC + video identity lock、video 钉 base。\(D_{\text{fail}}\)：仅 video BC、failure 后缀、无 action BC、无 identity lock |
| `uncond_adapter` | action lock 配置 + `recommend_adaptive_cfg_tau` |
| trainer / runtime | v6 train_mode；存 adapter；`recipe=v6` |
| eval yaml | `<task>_dewo_v6_cfg`，成功后缀与训练相同（Successful），**不是** Recovered |
| `eval_cfg_official_4x50.sh` | v6 method；`tau=auto` |
| `calibrate_adaptive_cfg_tau.py` | 从 \(E_+/E_0\) 写 json |
| 单测 | 过滤（留 episode、留 fail、丢 aux_success）、\(\tau\) 公式、pin-only-\(D_+\)、failure 不 lock、mode |
| eval-standard 规则 | 统一覆盖 v2 / v5 / v6 |

不改 pair 扫描；不改 `FastWAM-infer-in-DexJoco`；不恢复 LoRA；不新增旧路线入口。

### B.7 启动（实现后）

```bash
TASK=water_plant INIT=s0 DEWO_VERSION=v6 \
  ENV_FILE=data/<task>_dewo_v2_pair_*/eve_v02/protocol/offline_v1_b1_jump_fast.env \
  GPUS=0,1,2,3 \
  bash scripts/dewo_v2/train.sh
```

校准 \(\tau\) 后再评 CFG：

```bash
# w=1 本体（必报）
CFG_SCALE=1.0 ... bash scripts/dewo_v2/eval_cfg_official_4x50.sh

# 仅当 adaptive_cfg_tau.json 可分
CFG_SCALE=1.2 ADAPTIVE_CFG_TAU=auto CFG_TASK_DIR=.../water_plant_dewo_v6_cfg \
  ... bash scripts/dewo_v2/eval_cfg_official_4x50.sh
```

### B.8 验收

1. 池子 = 完整成功 rollout（\(D_0\)）∪ success event（\(D_+\)）∪ failure event；aux_success 不在 batch。无比，shuffle。
2. \(D_0\) 文本 base，video+action 对齐。\(D_+\) video 钉 base、action 成功后缀 + v5 dropout，无「video loss + 成功后缀」。\(D_{\text{fail}}\) 失败后缀 + video BC、无动作 BC、无 identity lock。
3. adapter 关 = 本体；\(w=1\) 走 bypass，4×50 不低于该 S0。推理 video 两路 base。
4. \(\tau=q_{1-\rho}(E_+)\)（默认 \(\rho=0.9\)），由 \(D_+\) 训练窗算出；\(\mathrm{FPR}_0\) 过大则禁用 adaptive，不回退网格搜/0.05。
5. \(w>1\) 若掉出噪声：失败的是门控/正类，不是「再加大 \(w\)」。

### B.9 明确不做

- 给 \(D_+\) 做 **video loss + 成功后缀**（旧 aux_success；推理 video 不用这条残差，还会泄漏结局进观测）。
- 失败窗做 **动作** BC，或失败帧配 **base** 文本做 video BC（污染 \(\Delta(\text{base})\)，破坏 video-as-action-context）。
- 对 \(D_{\text{fail}}\) 打 identity lock（会抹掉 \(\Delta(\text{failure})\)）。
- 把完整成功 rollout 当 CFG 正类（v5 的错正类）；它们只当 \(D_0\)。
- 丢掉全部普通成功 episode（会失去 base 对齐）。
- 人为 12:4 / role-balanced；把 **aux_success** 当更严优势。
- 把成功后缀改成 Recovered 或其它新词。
- 随机 15 条成功 episode 当 \(D_+\)。
- 抄 RECAP 的 30% dropout、value、10% episode 分位（除非以后做效率轴 v6.1）。
- 默认 CFG=2；把 0.05/0.5 当 \(\tau\) 先验；在官方 0–49 或网格上搜 \(\tau\)。
- 解冻主干；LoRA；384 / min-max；新增旧路线入口。

---

## C. DEWO v7（pair 进入被放大的残差）

### C.0 自检（做之前过一遍）

v6 的残差是 \(\varepsilon_+-\varepsilon_0\)。放大它等于问「比本体更成功吗」。water_plant 本体已经 ~86%，官方 4×50 上最好一格也是 86% 且只开约 6% chunk；引导开多了掉到 82%/76%。pair 在数据里，不在梯度里：\(D_+\) 动作 BC、\(D_{\mathrm{fail}}\) video BC、推理减的是 S0。负例对控制是死的。

**该换的是残差对象，不是再加保政策的 anchor，也不是先改成 DPO 单政策。**

选定公式：

```
ε_cfg = ε_0 + w · (ε_+ − ε_-)
```

- \(\varepsilon_0\)：adapter **关** + base = bit-exact S0。\(w=0\) / `text_cfg_scale=1` bypass = 本体。
- \(\varepsilon_+\)：adapter **开** + ` Successful execution.`，video **base**。
- \(\varepsilon_-\)：adapter **开** + ` Failed execution.`，video **base**。只作减数，**永不执行**。禁止 \(\varepsilon_-+w(\varepsilon_+-\varepsilon_-)\)（\(w=0\) 会跑失败政策）。
- 门控能量：NFE0 执行段 \(\mathrm{RMS}(\varepsilon_+-\varepsilon_-)\)，不要再用 \(\mathrm{RMS}(\varepsilon_+-\varepsilon_0)\)。

即便 \(\varepsilon_+\approx\varepsilon_0\)，上式仍是从本体推离失败条件分数。失败 BC 只打在失败句上，不打进 \(\varepsilon_+\) / base。这和 reward hacking 不是一回事。

自检里否掉或推迟的：

| 想法 | 结论 |
|---|---|
| \(D_0\) 上 KL / \(\\|\varepsilon_{\mathrm{new}}-\varepsilon_{\mathrm{base}}\\|^2\) anchor | 不解决残差对象；v6 已有 D0 BC + identity lock。 |
| DPO-first，推理仍减 \(\varepsilon_0\) | 几何还是 v6；和 identity lock 抢同一条 \(\varepsilon_+\)。 |
| 同 batch 配对 + Diffusion-DPO | **v7.1**。v7.0 先让 \(\varepsilon_-\) 存在并被减去。残差在 \(D_+\) vs \(D_0\) 不可分再加配对排序。 |
| 失败动作 BC 进 \(\varepsilon_+\) | 禁止。那是模仿失败。 |
| 推理原点改成 \(\varepsilon_-\) | 禁止。 |

v7.0 仍用独立 shuffle（与 v6 同池）。33 帧窗后半段 \(s_+\neq s_-\)，配对排序现在做了也是假对比。失败 BC 定义的是「失败句下的动作分数」，不是前缀上的精确反事实。

其它已知风险：rank-16 分不开 \(c_+/c_-\)；\(\Delta(c_-)\) 串到 \(\Delta(c_{\mathrm{base}})\)；\(a_+\) 仍是 S0 已恢复的续行，eval 失败可能在支撑外。先看 \(E_{\mathrm{fork}}\gg E_{D_0}\)，不可分就停，别扫 \(w/\tau\)。

### C.1 样本与损失（v7.0）

同一 shuffle 池，无比。后缀锁死，与 v2/v5/v6 相同。

| 集合 | video 文本 | action 文本 | video loss | action loss | identity lock | action residual lock |
|---|---|---|---|---|---|---|
| **\(D_0\)** | base | base | 开 | 开 | 开（\(\Delta(c_{\mathrm{base}})\)） | 关（已有 video_w） |
| **\(D_+\)** | **base**（pin） | ` Successful execution.`，dropout `0.9,0,0.1` | 关 | 开 | 开 | 开 |
| **\(D_{\mathrm{fail}}\)** | **base**（pin；因 action BC>0） | ` Failed execution.`，`1.0,0,0` 不向 base 抽 | **关** | **开**（只训 \(\varepsilon_-\)） | 开（video 已钉 base，锁的是 \(\Delta(c_{\mathrm{base}})\)，不是 \(\Delta(c_{\mathrm{fail}})\)） | **关**（不能按住 \(\Delta_{\mathrm{action}}(c_{\mathrm{fail}})\)） |

相对 v6：失败样本从「video BC、无动作 BC」改成「动作 BC 走失败句、无 video BC」。推理 video 仍两路（三路动作里 video 共用 adapter-on + base）。`pin_video_context_to_base` 仍按 `action_loss_weight>0`，所以 \(D_{\mathrm{fail}}\) 的 video 会被钉到 base——这正是 \(\varepsilon_-\) 的推理条件。

### C.2 推理

`negative_context` **仍是 base / \(\varepsilon_0\)**，不要拿它当失败支。另加 `failure_context`（eval yaml `cfg_failure_prompt`）。

video prefill：adapter 关 + base（给 \(\varepsilon_0\)）；adapter 开 + base（\(\varepsilon_+\) 与 \(\varepsilon_-\) 共用）。动作前向三次：成功句 / 失败句 / base。过门才 mix；bypass（`text_cfg_scale=1`）仍 remap 到 base、adapter 关，不跑失败支。

**术语：** `text_cfg_scale=1` 仍是本体 bypass，不是 mix \(w=1\)。v7 的 mix \(w=1\) 是 \(\varepsilon_0+(\varepsilon_+-\varepsilon_-)\)，**不是** \(\varepsilon_+\)。不要用 v5/v6 的「mix w=1 = ε_posi」来读 v7 的 \(w\)。

### C.3 训练 / 评测落点

| 位置 | 内容 |
|---|---|
| Hydra | `dexjoco/dexjoco_dewo_v7_offline_b1_jump_fast_uncond` |
| `train.sh` | `DEWO_VERSION=v7`，`INIT=s0` only，不新建 per-task `.sh` |
| Eve | `unit_filter=dewo_v7_pool`（单元与 v6 相同；失败样本 `action_loss_weight=1`，`video_loss_weight=0`） |
| `uncond_adapter.recipe` | `v7`；ckpt 格式仍 `dewo_v5_uncond_adapter_v1` |
| infer | `ε_0 + w(ε_+ − ε_-)`；缺 `failure_context` 且 `text_cfg_scale!=1` 则报错，不静默退回 v6 |
| eval yaml | `<task>_dewo_v7_cfg`：`prompt` 成功句，`cfg_base_prompt` 任务句，`cfg_failure_prompt` 失败句 |
| 门控 | 同 v6 公式 \(\tau=q_{0.10}(E_+)\)，但 \(E=\mathrm{RMS}(\varepsilon_+-\varepsilon_-)\) |
| 单测 | 失败样本动作损失开、video 钉 base、action lock 不含 fail、mix 公式、failure 键 |

### C.4 启动

```bash
TASK=water_plant INIT=s0 DEWO_VERSION=v7 \
  ENV_FILE=data/<task>_dewo_v2_pair_*/eve_v02/protocol/offline_v1_b1_jump_fast.env \
  GPUS=0,1,2,3 \
  bash scripts/dewo_v2/train.sh
```

先 dump 训练池 \(E_+\)（D+ 窗）/ \(E_0\)（D0 非 event 前缀）上的 \(\mathrm{RMS}(\varepsilon_+-\varepsilon_-)\)。不可分只报 `CFG_SCALE=1.0`。可分再 `CFG_SCALE=1.2 ADAPTIVE_CFG_TAU=auto` 官方 4×50。

### C.5 验收 / 证伪

1. \(w=1\) bypass 4×50 不低于该 S0。
2. 训练前缀上 \(E_{\mathrm{fork}}\gg E_{D_0}\)（比 v6 的 \(E_+-E_0\) 更可分）。不可分 → 方法失败，停。
3. 可分后官方引导 mean 高于同数据的 v6。负对照：同一 ckpt 推理改回减 \(\varepsilon_0\)（或训练关掉失败动作 BC）分数若不变，负支没起作用。
4. 不要在官方 0–49 上扫 \(\tau\)。

### C.6 明确不做（v7.0）

- 失败动作 BC 进成功句 / base。
- 推理执行 \(\varepsilon_-\)，或 mix 原点改成 \(\varepsilon_-\)。
- 把 `negative_context` 改义为失败句。
- 解冻主干；LoRA；换后缀词；12:4。
- v7.0 不做 pair_id 对齐的 DPO（v7.1）。
- 不把 D0 policy-anchor 当主改动。
