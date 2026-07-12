# spray_water_gr00tstyle（真机）

真机 `spray_water` 任务，使用 **GR00T 风格 rot6d 动作空间 + 3 cam robotwin mosaic**。

## 数据

| 路径 | 说明 |
|------|------|
| `data/spray_water_rot6d_rosbag_ts_filter` | 原始 rosbag 过滤数据（本地） |
| `configs/data/spray_water_rot6d_gr00tstyle.yaml` | 训练数据配置 |
| `configs/task/spray_water_rot6d_gr00tstyle_uncond_3cam_384_1e-4.yaml` | Task 配置 |

## 训练

```bash
python scripts/precompute_text_embeds.py \
  task=spray_water_rot6d_gr00tstyle_uncond_3cam_384_1e-4

bash scripts/spray_water_gr00tstyle/train.sh
```

## 开环评估

```bash
bash scripts/spray_water_gr00tstyle/run_openloop_filtered_out.sh
bash scripts/spray_water_gr00tstyle/run_openloop_checkpoints.sh
```

底层引擎：`scripts/openloop/run_openloop.py`（共享，非数据集专用）。

## 真机 deploy（Wuji/Astribot）

`run_*_with_env.sh` 不包含机器私有路径。运行前设置 `ASTRIBOT_SDK_ROOT`、`ASTRIBOT_PYTHON_SHIMS` 和 `WUJI_HAND_SETUP`；GR00T client 还需设置 `GR00T_REPO_ROOT`。

```bash
# GPU 机器
bash scripts/spray_water_gr00tstyle/wuji/run_fastwam_server_with_env.sh \
  --run-dir runs/spray_water_rot6d_gr00tstyle_uncond_3cam_384_1e-4/<run_id> \
  --checkpoint runs/.../step_XXXX.pt

# 机器人端
bash scripts/spray_water_gr00tstyle/wuji/run_fastwam_client_with_env.sh \
  --policy-host <gpu-ip> --policy-port 5560
```

| 文件 | 用途 |
|------|------|
| `train.sh` | 训练 launcher |
| `run_openloop_*.sh` | 开环 batch eval |
| `wuji_fastwam_adapter.py` | GR00T obs ↔ FastWAM policy obs |
| `robotwin_camera_utils.py` | 3-cam mosaic 工具 |
| `wuji/` | ZMQ server/client + ROS 环境 launcher |

## 已移除的临时脚本

- `wait_and_train_spray_water_rot6d.sh` — 旧 rosbag 训练排队脚本，依赖已删除的 `train_spray_water_rot6d.sh`，已清理。
