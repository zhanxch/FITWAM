# FastWAM 仿真-真机差距定位 — 诊断结果总览

本目录包含按分析方案执行的 9 项诊断任务（A1-A3, B1-B3, C1-C3）的全部脚本、
配置与结论。下方是结果汇总与推荐修复路径。

> **仓库已对齐官方 FastWAM**（`third_party/FastWAM`）：自定义的
> `state_dit.py` / episode 数据集 / openloop 评估 / DexJoCo·EgoDex·EgoVLA·G1
> 历史实验代码保留在本地 `archive/`（不纳入 git）。当前训练/部署流程全部走官方路径，仅保留
> C3 `skip_dims` 归一化 patch（H2 修复，**已降级为可选验证，见下方"一句话结论"**，
> 详见 `C3_README.md` 的 FORKED BEHAVIOR 说明）和 Wuji 真机 ZMQ 部署栈
>（`scripts/1/`）。后续实验需在对齐后的代码上重训基线，再按下方"修订执行
> 顺序"推进。

## 一句话结论（2026-06-26 修订）

**H1（无 proprio）已被基线吸收（官方默认带 proprio=58）；H2（rot6d 归一化）
经定量复测 + 用户 openpi/rot6d 实测验证，noise amplification 仅 1.03x，
判定为非根因；归一化模式已对齐官方 z-score；H5（deploy 转换）已排除；
H3（action-video 解耦）是结构性事实但非根因。当前根因定位策略改为"两种
数据处理基线对照"：基线 A（FastWAM 官方风格：绝对动作 + z-score）vs 基线
B（GR00T 数据处理：相对动作 + 分组 min/max + clip），两者都用 FastWAM 模型，
通过 B4 开环拟合评估 + 真机 A/B 隔离"数据处理"变量。**

### H2 判定修正依据（2026-06-26）

1. **A2 脚本复测**（`runs/dataset_stats.json`，真实训练 stats）：
   - noise amplification（归一化空间噪声经反归一化后的正交误差放大）=
     **左臂 1.03x，右臂 0.93x**（A2 脚本 `test_noise_amplification` 输出）。
     即 per-dim scale mismatch（2.79x）几乎不放大旋转误差。
   - A2 原 "CONFIRMED" 判定用的 `or` 条件被前两项（normalized-space 正交误差
     1.79、scale ratio 2.79）绑架，但真正表征"输出是否变坏"的 amp=1.03 未超
     1.5 阈值。normalized-space 正交误差大只是表示问题（归一化空间目标流形
     扭曲），不等于模型无法拟合——线性归一化可逆，反归一化后 round-trip 误差 0。
2. **用户实测反证**：用 openpi 训练同一套 rot6d 数据，openpi 对 rot6d 也是
   per-dim z-score 归一化（无 SO(3) 感知），结果"表现略差于 GR00T 但大体趋势
   正确"。若 per-dim 归一化真是 rot6d 的根因，openpi 不可能出正确趋势。
3. **结论**：rot6d 的 per-dim 线性归一化（无论 min/max 还是 z-score）不破坏
   round-trip 正确性，噪声放大可忽略。C3 `skip_dims` 解决的是一个不存在的问题，
   **C3 降级为可选验证项，不再作为主线 A/B 对照**。

## 各任务结论速查

| 任务 | 结论 | 证据 |
|---|---|---|
| **A1** (H5 deploy 转换) | **排除**：rot6d↔quat round-trip 误差 ~0.04°，远低于阈值 | `rot6d_roundtrip_test.py` |
| **A2** (H2 归一化) | **2026-06-26 修订：非根因**。复测真实 stats：noise amplification 左 1.03x/右 0.93x（未超 1.5 阈值），round-trip 误差 0；normalized-space 正交误差 1.79 仅是表示问题（可逆线性变换）。用户 openpi+rot6d 实测趋势正确佐证。C3 降级为可选验证 | `normalization_rot6d_test.py` |
| **A3** (开环 action L1) | 真机 `eval/action_l1=0.0243`，action expert 收敛（loss_action=0.0024 << loss_video=0.061）；sim 无最终 eval 数据。action_l1 是归一化空间标量，需 B4 转成真实物理量才能判断"够不够准" | `extract_wandb_metrics.py` |
| **B1** (dump server/client) | **部分完成**（见下方"真机调试记录 server_debug1"）：已加 `--dump-dir`（server）+ `--dump-raw-actions`（client）；`server_debug1` 已抓 58 chunk 但格式与 B1 计划的 `step_*.npz` 不同，proprio 缺失仅为推断，尚未用 `analyze_deploy_dumps.py` 正式确认 H1（H2 已修订为非根因，仅记录输出正交性作基线参考） | `run_fastwam_server.py`/`run_gr00t_client.py`/`analyze_deploy_dumps.py` |
| **B2** (关后处理) | 已提供 `run_fastwam_client_diagnostic.sh`：execute-horizon=1、无插值/clip/滤波 + dump | `run_fastwam_client_diagnostic.sh` |
| **B3** (H1 启用 proprio) | **已被基线吸收**：官方默认 `proprio_dim` 自动继承=58，基线本就带 proprio；原 `proprio_dim: null` 覆盖已删除。H1 在官方流程下不成立，B3 不再作为独立实验 | `B3_README.md`（仅作历史记录） |
| **C1** (H3 解耦) | **确认结构性**：infer_action 中 action 仅 cross-attend **首帧**视频 token，不 attend 想象的未来视频；deploy 不跑 video diffusion。等机会影响 sim/real，非根因 | `action_video_coupling_analysis.py` |
| **C2** (loss 曲线) | 真机 loss_action 收敛（0.0033→0.0024），loss_video 0.085→0.061；sim 早停无最终值。action expert 能拟合训练目标，失败可能在闭环泛化（B4 需确认开环误差真实物理量） | `extract_wandb_metrics.py` |
| **C3** (H2 修复) | **2026-06-26 降级为可选验证**：`skip_dims` patch 单元测试通过，但 A2 复测显示 H2 非根因，C3 不再作为主线 A/B。patch 保留在 fork 中供需要时启用。**此 patch 为 fork，不在官方 FastWAM 中** | `normalizer.py`/`fastwam_processor.py`/`test_skip_dims_normalizer.py`/`C3_README.md` |

## spray_water 数据集与官方对齐说明

### 数据集

- 路径：`data/spray_water_rot6d_rosbag_ts_filter`（156 episodes）+
  `filtered_out/`（5-episode holdout），LeRobot v2 布局。
- 相机：`head_view` 720×1280、`left_wrist_view`/`right_wrist_view` 360×640，
  训练时 resize 到 384×384（`concat_multi_camera: robotwin` 三相机拼接）。
- 动作/状态：58 维 = 左/右 EEF xyz(3)+rot6d(6) + 左/右手 20 关节，`fps≈30.03`。
- 任务字符串（`meta/tasks.jsonl`）："Pick up the spray bottle, pump it to build
  up pressure, then spray water on the flowers"。

### 数据构建来源

该数据集由**外部** GR00T/rosbag→LeRobot 流水线构建，**本仓库内无构建脚本**
（命名 `*_rosbag_ts_filter` 指向 rosbag 时间戳过滤流程）。`dataset_stats.json`
由首次训练运行生成（数据 config 中 `pretrained_norm_stats` 当前为 `null`，
首跑后写到 `runs/<task>/<run_id>/dataset_stats.json`）。

### 与官方 `robotwin.yaml` 的差异

| 字段 | spray_water | 官方 robotwin | 说明 |
|---|---|---|---|
| `norm_default_mode` | `z-score` | `z-score` | **已对齐官方**（2026-06-26 从 `min/max` 切到 `z-score`，与官方 robotwin 一致）。z-score 与 flow matching 噪声 `N(0,1)` 同尺度，且对 rot6d 的 noise amplification 同样 ~1x（A2 证实非根因） |
| `tolerance_s` | `0.02` | 默认 `1e-4` | rosbag 时间戳容差，对齐 rosbag 数据所需 |
| `video_backend` | `pyav` | 默认 `None` | rosbag mp4 解码后端 |
| `skip_padding_as_possible` | `true` | `false` | 允许跳过 padding 帧 |
| `action/state dim` | 58 | 14 | Wuji 58 维（xyz+rot6d+20 hand joints 双臂）vs 官方 RoboTwin 14 维双臂 qpos（xyz(3)+axis_angle(3)+gripper(1) 每臂）。官方 `PoseRotationTransform` 支持 rot6d↔axis_angle 转换；基线 B 保留 rot6d 但走 GR00T 风格相对动作+分组归一化 |

> **官方格式参考**：官方 `robotwin.yaml` 用 14 维双臂
> qpos + `norm_default_mode: z-score`；官方 `libero_2cam.yaml` 用 7 维
>（eef_pose 6 + gripper 1）。两者都不是 rot6d。官方 `transforms/rotation.py`
> 的 `PoseRotationTransform` 支持 `rotation_6d_to_axis_angle` 等转换，可在
> `action_state_transforms` 中配置，用于把 raw rot6d 转成官方 axis-angle 格式。

> **注意**：`tolerance_s`/`video_backend` 是 `robot_video_dataset.py`/
> `base_lerobot_dataset.py` 上为 rosbag 数据保留的最小定制，不属于官方默认。
> 如从官方重新同步 `src/fastwam/datasets/`，需重新应用这两个 kwargs。

## 真机调试记录（server_debug1）

`runs/server_debug1/`（gitignore，本地保留）记录了一次真机部署调试：

- **58 个连续 chunk**（`request_000001`…`request_000058`），约 108 秒，
  一个完整连续的 spray-water episode（同一 prompt 贯穿全程）。
- 每个 request 含：4 张输入图（head/left_wrist/right_wrist/concat_model）、
  `predicted_video.mp4`、`action_chunk_server_sent.json`+`.npz`、`prompt.txt`。
- 动作 chunk 结构：`left_eef`/`right_eef` `[1,32,9]`、
  `left_hand_joints`/`right_hand_joints` `[1,32,20]`，`action_horizon=32`。
- **proprio 在 dump 中缺失**（与 H1 一致，但仅为推断，未正式确认）。
- 已生成 58 张逐维曲线 PNG + `action_curves/all_dims_grid.png/pdf`
  （`plot_action_curves.py` 默认 `execute_horizon=24`，
  `plot_action_curves_grid.py` 默认 `execute_horizon=16`，**两者不一致，TODO 待统一**）。
- **无书面结论**（`runs/` 下无 `.md`/分析 JSON）。

### B1 剩余工作

`server_debug1` 的 dump 格式与 B1 计划的 `step_*.npz` 不同：无
`use_proprio`/`normalized_proprio_is_none` 标志、无归一化前 raw action
（无法测模型输出 rot6d 正交性）、无 client 侧 `--dump-raw-actions` 输出。
仍需按 B1 标准流程重跑：

```bash
# server 加 --dump-dir，client 加 --dump-raw-actions
python scripts/1/run_fastwam_server.py --run-dir <run> --checkpoint <ckpt> \
    --dump-dir runs/diag_b1/server_dump
bash scripts/1/run_fastwam_client_with_env.sh --host <ip> --port 5560 \
    --dump-raw-actions runs/diag_b1/client_raw
python scripts/diagnose/analyze_deploy_dumps.py \
    --server-dir runs/diag_b1/server_dump --client-dir runs/diag_b1/client_raw
```

## 修订执行顺序（2026-06-26，对齐官方后的流程）

由于 Phase 1-3 移除了自定义 `state_dit` / episode 数据集并把 proprio 切片
回退到官方 `proprio[:-1]`，**当前 checkpoint 是在自定义代码上训的**，需先
重训一个干净基线再继续诊断。

> **重要**：官方 FastWAM 默认 `proprio_dim` 自动从数据 config 的
> `proprio_output_dim` 继承（=58），即官方训练**本来就带 proprio**。之前
> spray_water 基线显式写 `proprio_dim: null` 主动关掉 proprio，那是 H1 假设
> 的来源，**不是官方默认**。现已删除该覆盖，基线走官方默认带 proprio=58。
> 因此 **H1（无 proprio）在官方流程下本就不成立**。

> **H2 判定已修订**（见上方"一句话结论"）：rot6d per-dim 归一化经定量复测 +
> 用户 openpi 实测，noise amplification 仅 1.03x，**非根因**。C3 从主线降级
> 为可选验证。归一化模式已从 `min/max` 切到官方 `z-score`（2026-06-26）。
> 当前根因定位策略：**两种数据处理基线对照**（基线 A = FastWAM 官方风格
> 绝对动作+z-score，基线 B = GR00T 数据处理相对动作+分组 min/max+clip），
> 两者都用 FastWAM 模型，隔离"数据处理"变量。详见下方"修订执行顺序"。

1. **重训基线 A：FastWAM 官方风格（1-2 天，需重训，已就绪）**：
   `bash scripts/train_spray_water_rot6d.sh`
   （task=`spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4`，
   `proprio_dim` 自动继承 58，全 58 维 **z-score** 归一化，与官方 robotwin
   对齐），得到与官方代码对齐的参考 checkpoint。
   - 数据格式：58 维 rot6d（绝对动作）
   - 归一化：z-score，per-dim，全 58 维合并 stats
   - 动作类型：绝对动作（`action_state_transforms: null`）
   - **代表"FastWAM 官方默认数据处理路径"**

2. **实现基线 B：GR00T 数据处理 + FastWAM 模型（1-2 天实现 + 1-2 天重训）**：
   数据处理完全复刻 GR00T，仅把 GR00T 的 DiT 模型换成 FastWAM 的 Wan2.2。
   - **核心流程（GR00T `launch_finetune.py` + `state_action_processor.py`）**：
     a. **相对动作转换**（SE(3) 几何正确）：
        - EEF（rot6d 9 维）：rot6d→齐次矩阵→`T_relative = inv(T_state) @ T_action`→转回 rot6d
          （`RelativePoseTransform` 的逻辑，但官方只接受 7 维 xyz+quat，需写 rot6d 版）
        - hand joints（20 维）：简单相减 `action - state[0]`（`RelativeJointTransform`，官方已有）
        - reference = 当前 state 最后一帧
     b. **分组 min/max 归一化到 `[-1,1]` + clip**：
        - EEF / hand joints 各自独立 stats、独立归一化（GR00T 的 per-modality）
        - 用 FastWAM 的 `norm_exception_mode` 给不同 key 设 min/max，或拆成多个 key
     c. **stats 对相对动作算**：FastWAM 的 `get_dataset_stats` 已在 `action_state_transform`
        之后算 stats（`base_lerobot_dataset.py:275`），自动对相对动作算，无需改代码
   - **实现工作量**：
     - 新增 `transforms/relative_action.py` 的 rot6d 版 `RelativePoseRot6dTransform`
      （官方 `RelativePoseTransform` 只支持 7 维 quat，需扩展；rot6d→齐次矩阵已验证无损）
     - 新增数据 config（`spray_water_rot6d_gr00tstyle.yaml`）：配置
       `action_state_transforms`（rot6d 相对 + joint 相对）+ 分组 min/max
     - 新增任务 config + 训练脚本
   - **代表"GR00T 数据处理路径 + FastWAM 模型"，隔离数据处理变量**
   - **注意**：deploy 侧需对应实现相对动作的 `backward`（相对→绝对），当前
     `wuji_fastwam_adapter.py` 假设绝对动作，需适配

3. **B4 开环 action chunk 拟合评估（半天，无需重训，对两个基线都做）**：
   用基线 A 和基线 B 的 checkpoint 在**训练/测试同布局同任务**的场景上做开环推理，
   结合视频逐帧分析 action chunk 预测误差的**真实物理距离**（xyz L2 in meters、
   rot6d 经 Gram-Schmidt 恢复后的测地距离 in degrees、hand joint 角度差），
   对照 GT action 评估：
   - 两个基线在训练分布内的拟合误差对比（A3 显示 `eval/action_l1=0.0243`，但
     L1 是归一化空间标量，需转成真实物理量判断是否"够准"）
   - 误差主要在 xyz / rot6d / hand joints 哪一段？两个基线差异在哪段？
   - 误差是否随 chunk 步数累积（前几步准、后几步偏）？
   - 结合视频：模型想象的未来视频帧与真实视频是否吻合？action 是否与视频一致？
   脚本：复用 `runs/server_debug1` 的 dump 格式 + 新增 `scripts/diagnose/
   eval_openloop_chunk.py`（待编写：读 action chunk + GT，转真实物理量误差）。

4. **B1+B2（半天，无需重训）**：用基线 A 的干净 checkpoint 跑标准 B1 dumps
   + `analyze_deploy_dumps.py`：确认 `use_proprio=True`（H1 在官方默认下已消
   除）、记录模型输出 rot6d 正交性作为基线参考；再用
   `run_fastwam_client_diagnostic.sh` 关后处理对照，定位是输出错还是后处理
   弄坏。

5. **真机 A/B 对照（按需，半天）**：基线 A vs 基线 B 真机部署对比，验证
   GR00T 数据处理（相对动作 + 分组 min/max）是否改善 sim-vs-real 差距。
   若 B 明显改善 → 数据处理是根因；若两者相当 → 数据处理非根因，根因在
   模型架构（action-video 解耦 C1）或闭环泛化。

6. **B3 已被基线吸收**：原 B3（`proprio_dim=58`）与现基线 A 完全相同，不再
   作为独立实验。

7. **C3 可选验证（低优先级）**：若 B4 后仍怀疑归一化，可跑
   `bash scripts/train_spray_water_rot6d_skip_rot6d.sh` 做 A/B。但基于 A2
   复测，预期改善有限。

## 文件索引

### 诊断脚本（`scripts/diagnose/`）
- `rot6d_roundtrip_test.py` — A1：deploy 旋转转换 round-trip
- `normalization_rot6d_test.py` — A2：min/max 归一化对 rot6d 正交性的破坏
- `extract_wandb_metrics.py` — A3/C2：从 wandb/output.log 提取 loss 与 eval 指标
- `action_video_coupling_analysis.py` — C1：infer_action 中 action-video 耦合分析
- `analyze_deploy_dumps.py` — B1：分析 server/client dump
- `test_skip_dims_normalizer.py` — C3：skip_dims normalizer patch 单元测试
- `metrics_output.json` — A3/C2 提取的指标输出
- `B3_README.md` / `C3_README.md` — 实验协议文档

### 源码改动（fork，向后兼容）
- `src/fastwam/datasets/lerobot/utils/normalizer.py` — `SingleFieldLinearNormalizer(skip_dims)` + `LinearNormalizer(skip_dims)` + 原子写 `dataset_stats.json`（C3，**fork**）
- `src/fastwam/datasets/lerobot/processors/fastwam_processor.py` — `norm_skip_dims` 配置字段（C3，**fork**）
- `src/fastwam/datasets/lerobot/robot_video_dataset.py` — `tolerance_s`/`video_backend` kwargs + `dataset_stats.json` 防自拷贝（rosbag 所需，非官方默认）
- `src/fastwam/datasets/lerobot/base_lerobot_dataset.py` — `tolerance_s`/`video_backend` 透传（rosbag 所需）
- `src/fastwam/models/wan22/fastwam.py` — `load_checkpoint(experts=...)` 部分 MoT 专家加载（ops，向后兼容）
- `scripts/1/run_fastwam_server.py` — `--dump-dir` B1 诊断
- `scripts/1/run_gr00t_client.py` — `--dump-raw-actions` B1 诊断

### 配置 / 脚本
- `configs/data/spray_water_rot6d_rosbag_ts_filter.yaml` — 基线数据 config
- `configs/data/spray_water_rot6d_rosbag_ts_filter_filtered_out.yaml` — 5-episode holdout
- `configs/data/spray_water_rot6d_rosbag_ts_filter_skip_rot6d.yaml` — C3 数据 config（`norm_skip_dims`）
- `configs/task/spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4.yaml` — 基线任务 config（proprio 自动继承 58）
- `configs/task/spray_water_rot6d_rosbag_ts_filter_uncond_3cam_384_1e-4_skip_rot6d.yaml` — C3 任务 config（proprio 58 + rot6d 不归一化）
- `scripts/train_spray_water_rot6d.sh` — 基线训练脚本
- `scripts/train_spray_water_rot6d_skip_rot6d.sh` — C3 训练脚本
- `scripts/1/run_fastwam_client_diagnostic.sh` — B2 关后处理部署脚本
