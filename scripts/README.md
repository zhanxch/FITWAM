# Scripts

**Research direction:** [`README.md`](../README.md) · **Upstream setup:** [`docs/FASTWAM_UPSTREAM.md`](../docs/FASTWAM_UPSTREAM.md)

## 目录结构

```text
scripts/
├── train.py, train_zero1.sh, train_zero2.sh   # 通用 Hydra 训练入口
├── precompute_text_embeds.py                   # T5 embedding 预计算
├── preprocess_action_dit_backbone.py           # ActionDiT backbone 准备
├── fastwam_policy_server.py, policy_io.py      # ZMQ policy server 基础设施
├── openloop/                                   # 开环 eval 引擎（多数据集共用）
├── water_plant/                                # DexJoCo water_plant（2cam + proprio）
├── spray_water_gr00tstyle/                     # 真机 spray_water GR00T-style
├── diagnose/                                   # 本地诊断（gitignore，不上传）
├── accelerate_configs/  ds_configs/
└── archive/  (repo 外)                         # 历史实验
```

## 按数据集

| 数据集 | 目录 | 说明 |
|--------|------|------|
| **water_plant** | [`water_plant/`](water_plant/) | 双视角 + proprio 23d；LeRobot 窗口 / EveRobot 整 episode；DexJoCo 闭环 |
| **spray_water_gr00tstyle** | [`spray_water_gr00tstyle/`](spray_water_gr00tstyle/) | 真机 3cam rot6d；训练、开环 eval、Wuji deploy |

## 通用训练流程

```bash
# 1. 预计算 text embed（task 名见 configs/task/）
python scripts/precompute_text_embeds.py task=<task_name>

# 2. 训练（ZeRO-1）
bash scripts/train_zero1.sh 4 task=<task_name>
```

数据集专用的 prepare / train launcher 见各子目录 README。

## Video LoRA

在 task 配置中 `override /model: fastwam_video_lora`，或：

```bash
bash scripts/water_plant/train_2cam.sh task=water_plant_uncond_2cam_384_1e-4_lora
bash scripts/water_plant/train_everobot.sh task=everobot_water_plant_full_lora
```

详见根目录 [`README.md`](../README.md#video-lora独立可选路径) 与 `configs/model/fastwam_video_lora.yaml`。

## 哪些是临时 / 历史脚本？

| 位置 | 状态 |
|------|------|
| `scripts/diagnose/` | 本地 sim-vs-real 诊断，**gitignore**，不上传 |
| `water_plant/dexjoco_async/run_dexjoco_async_lpf_eval_clients.sh` | 早期 microwave LPF 消融 launcher，保留参考 |
| `openloop/run_robotwin_openloop*.py` | 已弃用 shim，请用 `run_openloop.py` + configs |
| `wait_and_train_spray_water_rot6d.sh` | 已删除（旧 rosbag 临时排队） |
