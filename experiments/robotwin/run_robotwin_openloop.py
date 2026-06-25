import csv
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging
from fastwam.utils.video_io import save_mp4
from fastwam.utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

register_default_resolvers()
logger = get_logger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _resolve_path(path_str: str, *, base: Path = PROJECT_ROOT, must_exist: bool = True) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return path


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 < len(parts):
            return f"{parts[runs_idx + 1]}_{parts[runs_idx + 2]}"
    return ckpt_path.stem


def _to_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _to_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _to_optional_int_list(value: Any) -> Optional[list[int]]:
    if _is_none_like(value):
        return None
    if isinstance(value, ListConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1]
        return [int(item.strip()) for item in stripped.split(",") if item.strip() != ""]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _instantiate_openloop_dataset(cfg: DictConfig):
    split = str(cfg.OPENLOOP.split).strip().lower()
    if split not in {"train", "val"}:
        raise ValueError(f"OPENLOOP.split must be 'train' or 'val', got: {cfg.OPENLOOP.split}")

    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data[split], resolve=True))
    stats_path = cfg.OPENLOOP.get("dataset_stats_path")
    if not _is_none_like(stats_path):
        dataset_cfg.pretrained_norm_stats = str(_resolve_path(str(stats_path)))
    elif _is_none_like(dataset_cfg.get("pretrained_norm_stats")):
        raise ValueError(
            "Dataset normalization stats are required for open-loop eval. "
            "Set OPENLOOP.dataset_stats_path or data.<split>.pretrained_norm_stats."
        )
    return instantiate(dataset_cfg)


def _first_prompt(sample: dict[str, Any]) -> Optional[str]:
    prompt = sample.get("prompt")
    if prompt is None:
        return None
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, (list, tuple)):
        return str(prompt[0])
    raise TypeError(f"Unsupported prompt type: {type(prompt)}")


def _first_idx(sample: dict[str, Any]) -> int:
    idx = sample.get("idx", -1)
    if isinstance(idx, torch.Tensor):
        return int(idx.reshape(-1)[0].item())
    if isinstance(idx, (list, tuple)):
        return int(idx[0])
    return int(idx)


def _denormalize_action(processor: Any, action_btd: torch.Tensor, proprio_btd: torch.Tensor) -> torch.Tensor:
    if action_btd.ndim == 2:
        action_btd = action_btd.unsqueeze(0)
    if proprio_btd.ndim == 2:
        proprio_btd = proprio_btd.unsqueeze(0)
    if action_btd.ndim != 3 or proprio_btd.ndim != 3:
        raise ValueError(
            f"Expected action/proprio as [B,T,D], got {tuple(action_btd.shape)} and {tuple(proprio_btd.shape)}"
        )

    batch = {
        "action": action_btd.detach().to(device="cpu", dtype=torch.float32),
        "state": proprio_btd.detach().to(device="cpu", dtype=torch.float32),
    }
    batch = processor.action_state_merger.backward(batch)
    batch = processor.normalizer.backward(batch)

    merged_batch = {
        "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in processor.shape_meta["action"]},
        "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in processor.shape_meta["state"]},
    }
    merged_batch = processor.action_state_merger.forward(merged_batch)
    return merged_batch["action"].unsqueeze(0).detach().to(device="cpu", dtype=torch.float32)


def _denormalize_state(processor: Any, proprio_btd: torch.Tensor, action_shape_ref: torch.Tensor) -> torch.Tensor:
    """Denormalize proprio trajectory to the same merged vector space as `action` (RobotWin: dim=14)."""
    if proprio_btd.ndim == 2:
        proprio_btd = proprio_btd.unsqueeze(0)
    if action_shape_ref.ndim == 2:
        action_shape_ref = action_shape_ref.unsqueeze(0)
    if proprio_btd.ndim != 3 or action_shape_ref.ndim != 3:
        raise ValueError(
            f"Expected proprio/action ref as [B,T,D], got {tuple(proprio_btd.shape)} and {tuple(action_shape_ref.shape)}"
        )
    if proprio_btd.shape[:2] != action_shape_ref.shape[:2]:
        raise ValueError(
            f"proprio/action time length mismatch: {tuple(proprio_btd.shape)} vs {tuple(action_shape_ref.shape)}"
        )
    zeros = torch.zeros_like(action_shape_ref, device="cpu", dtype=torch.float32)
    batch = {
        "action": zeros,
        "state": proprio_btd.detach().to(device="cpu", dtype=torch.float32),
    }
    batch = processor.action_state_merger.backward(batch)
    batch = processor.normalizer.backward(batch)
    merged_batch = {
        "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in processor.shape_meta["action"]},
        "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in processor.shape_meta["state"]},
    }
    merged_batch = processor.action_state_merger.forward(merged_batch)
    return merged_batch["state"].unsqueeze(0).detach().to(device="cpu", dtype=torch.float32)


def _action_metrics(
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    action_is_pad: Optional[torch.Tensor],
) -> dict[str, Any]:
    if pred_action.shape != gt_action.shape:
        raise ValueError(
            "Predicted action/GT action shape mismatch: "
            f"pred={tuple(pred_action.shape)} vs gt={tuple(gt_action.shape)}"
        )
    diff = pred_action - gt_action
    if action_is_pad is not None:
        if action_is_pad.ndim == 1:
            action_is_pad = action_is_pad.unsqueeze(0)
        valid = ~action_is_pad.to(dtype=torch.bool, device=diff.device)
        if valid.shape != diff.shape[:2]:
            raise ValueError(
                f"action_is_pad shape mismatch: mask={tuple(valid.shape)} vs action={tuple(diff.shape)}"
            )
        diff_valid = diff[valid]
    else:
        diff_valid = diff.reshape(-1, diff.shape[-1])
    if diff_valid.numel() == 0:
        raise ValueError("No valid action steps available for metrics.")

    abs_diff = diff_valid.abs()
    sq_diff = diff_valid.pow(2)
    mse = float(sq_diff.mean().item())
    return {
        "action_l1": float(abs_diff.mean().item()),
        "action_mse": mse,
        "action_rmse": float(math.sqrt(mse)),
        "action_max_abs": float(abs_diff.max().item()),
        "action_l1_per_dim": abs_diff.mean(dim=0),
        "action_mse_per_dim": sq_diff.mean(dim=0),
        "num_action_steps": int(diff_valid.shape[0]),
    }


def _video_metrics(
    pred_frames: list[Image.Image],
    gt_video: torch.Tensor,
    image_is_pad: Optional[torch.Tensor],
) -> dict[str, float]:
    pred_video = pil_frames_to_video_tensor(pred_frames)
    gt_video = ((gt_video.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
    if pred_video.shape != gt_video.shape:
        raise ValueError(
            f"Predicted video/GT video shape mismatch: pred={tuple(pred_video.shape)} vs gt={tuple(gt_video.shape)}"
        )

    if image_is_pad is not None:
        if image_is_pad.ndim == 2:
            image_is_pad = image_is_pad[0]
        valid = ~image_is_pad.to(dtype=torch.bool, device="cpu")
        if valid.shape[0] != pred_video.shape[1]:
            raise ValueError(
                f"image_is_pad length mismatch: mask={tuple(valid.shape)} vs video={tuple(pred_video.shape)}"
            )
        if bool(valid.any().item()):
            pred_video = pred_video[:, valid]
            gt_video = gt_video[:, valid]

    diff = pred_video - gt_video
    return {
        "video_psnr": video_psnr(pred=pred_video, target=gt_video),
        "video_ssim": video_ssim(pred=pred_video, target=gt_video),
        "video_l1": float(diff.abs().mean().item()),
        "video_mse": float(diff.pow(2).mean().item()),
    }


def _first_action_mask(action_is_pad: Optional[torch.Tensor], action_len: int) -> Optional[np.ndarray]:
    if action_is_pad is None:
        return None
    if action_is_pad.ndim == 2:
        action_is_pad = action_is_pad[0]
    valid = ~action_is_pad.detach().to(dtype=torch.bool, device="cpu")
    if valid.shape[0] != action_len:
        raise ValueError(f"action_is_pad length mismatch: mask={tuple(valid.shape)} vs action_len={action_len}")
    return valid.numpy()


def _polyline_xy(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], *, fill: tuple[int, int, int], width: int = 2) -> None:
    if len(points) < 2:
        return
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=fill, width=width)


def _draw_action_curves_panel(
    gt_action: torch.Tensor,
    pred_action: torch.Tensor,
    *,
    frame_idx: int,
    num_video_frames: int,
    width: int,
    height: int,
    action_is_pad: Optional[torch.Tensor] = None,
    state_joints: Optional[torch.Tensor] = None,
    inference_mark_stride: int,
) -> Image.Image:
    """Per-dim time-series GT vs pred (and optional state), similar to GR00T plot_trajectory_results."""
    if gt_action.ndim == 3:
        gt_action = gt_action[0]
    if pred_action.ndim == 3:
        pred_action = pred_action[0]
    if gt_action.shape != pred_action.shape:
        raise ValueError(
            f"GT/pred action shape mismatch for plotting: gt={tuple(gt_action.shape)} pred={tuple(pred_action.shape)}"
        )

    gt_np = gt_action.detach().to(device="cpu", dtype=torch.float32).numpy()
    pred_np = pred_action.detach().to(device="cpu", dtype=torch.float32).numpy()
    state_np: Optional[np.ndarray] = None
    if state_joints is not None:
        sj = state_joints[0] if state_joints.ndim == 3 else state_joints
        state_np = sj.detach().to(device="cpu", dtype=torch.float32).numpy()
        if state_np.shape != gt_np.shape:
            state_np = None

    action_len, action_dim = gt_np.shape
    valid_mask = _first_action_mask(action_is_pad, action_len)
    valid_steps = np.flatnonzero(valid_mask) if valid_mask is not None else np.arange(action_len)
    if valid_steps.size == 0:
        valid_steps = np.arange(action_len)

    current_step = 0 if num_video_frames <= 1 else round(frame_idx * (action_len - 1) / (num_video_frames - 1))
    stride = max(int(inference_mark_stride), 1)

    panel = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 4), "Action vs time  GT=blue  Pred=red  State=green (if same shape)", fill=(20, 20, 20))

    title_h = 22
    avail_h = max(height - title_h - 8, 1)
    row_h = max(avail_h // max(action_dim, 1), 22)
    left_margin = 44
    for dim in range(action_dim):
        y0 = title_h + dim * row_h
        y1 = y0 + row_h - 4
        x0 = left_margin
        x1 = width - 8
        if y1 <= y0 + 8 or x1 <= x0:
            continue
        draw.rectangle((x0, y0, x1, y1), outline=(210, 210, 210))

        vals = np.concatenate([gt_np[valid_steps, dim], pred_np[valid_steps, dim]])
        if state_np is not None:
            vals = np.concatenate([vals, state_np[valid_steps, dim]])
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if math.isclose(vmin, vmax):
            pad = max(abs(vmin) * 0.05, 1e-3)
            vmin -= pad
            vmax += pad
        else:
            pad = (vmax - vmin) * 0.06
            vmin -= pad
            vmax += pad

        def step_to_x(step: int) -> int:
            if action_len <= 1:
                return int((x0 + x1) * 0.5)
            return int(round(x0 + (x1 - x0) * step / (action_len - 1)))

        def val_to_y(v: float) -> int:
            return int(round(y1 - (y1 - y0 - 6) * (float(v) - vmin) / (vmax - vmin))) - 3

        gt_pts = [(step_to_x(int(s)), val_to_y(float(gt_np[s, dim]))) for s in valid_steps]
        pred_pts = [(step_to_x(int(s)), val_to_y(float(pred_np[s, dim]))) for s in valid_steps]
        _polyline_xy(draw, gt_pts, fill=(20, 90, 220), width=2)
        _polyline_xy(draw, pred_pts, fill=(220, 40, 40), width=2)
        if state_np is not None:
            st_pts = [(step_to_x(int(s)), val_to_y(float(state_np[s, dim]))) for s in valid_steps]
            _polyline_xy(draw, st_pts, fill=(30, 160, 60), width=2)

        for j in range(0, action_len, stride):
            gx, gy = step_to_x(j), val_to_y(float(gt_np[j, dim]))
            draw.ellipse((gx - 4, gy - 4, gx + 4, gy + 4), fill=(200, 30, 30))

        cx = step_to_x(current_step)
        draw.line((cx, y0, cx, y1), fill=(120, 120, 120), width=1)

        draw.text((8, y0 + 2), f"a{dim:02d}", fill=(40, 40, 40))

    return panel


def _project_xyz_iso(points_xyz: np.ndarray) -> np.ndarray:
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]
    return np.stack([(x - y) * 0.8660254, (x + y) * 0.5 - z], axis=1)


def _action_xyz_slices(action_dim: int) -> list[tuple[str, slice]]:
    if action_dim >= 14:
        return [("left arm xyz", slice(0, 3)), ("right arm xyz", slice(7, 10))]
    if action_dim >= 7:
        return [("xyz", slice(0, 3))]
    if action_dim >= 3:
        return [("xyz", slice(0, 3))]
    return []


def _draw_single_3d_trajectory(
    draw: ImageDraw.ImageDraw,
    *,
    title: str,
    gt_xyz: np.ndarray,
    pred_xyz: np.ndarray,
    valid_steps: np.ndarray,
    current_step: int,
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(215, 215, 215), fill=(250, 250, 250))
    draw.text((x0 + 8, y0 + 6), title, fill=(35, 35, 35))

    gt_rel = gt_xyz - gt_xyz[valid_steps[0]]
    pred_rel = pred_xyz - pred_xyz[valid_steps[0]]
    all_rel = np.concatenate([gt_rel[valid_steps], pred_rel[valid_steps], np.zeros((1, 3), dtype=np.float32)], axis=0)
    proj_all = _project_xyz_iso(all_rel)
    p_min = proj_all.min(axis=0)
    p_max = proj_all.max(axis=0)
    span = np.maximum(p_max - p_min, 1e-6)

    plot_x0 = x0 + 28
    plot_y0 = y0 + 34
    plot_x1 = x1 - 16
    plot_y1 = y1 - 22
    plot_w = max(plot_x1 - plot_x0, 1)
    plot_h = max(plot_y1 - plot_y0, 1)
    scale = min(plot_w / span[0], plot_h / span[1]) * 0.82
    center = np.array([(plot_x0 + plot_x1) * 0.5, (plot_y0 + plot_y1) * 0.5], dtype=np.float32)
    proj_center = (p_min + p_max) * 0.5

    def to_xy(points: np.ndarray) -> list[tuple[int, int]]:
        proj = _project_xyz_iso(points)
        xy = (proj - proj_center) * scale + center
        return [(int(round(px)), int(round(py))) for px, py in xy]

    origin = to_xy(np.zeros((1, 3), dtype=np.float32))[0]
    axes = {
        "x": np.array([[0.0, 0.0, 0.0], [0.06, 0.0, 0.0]], dtype=np.float32),
        "y": np.array([[0.0, 0.0, 0.0], [0.0, 0.06, 0.0]], dtype=np.float32),
        "z": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.06]], dtype=np.float32),
    }
    for axis_name, axis_points in axes.items():
        axis_xy = to_xy(axis_points)
        draw.line(axis_xy, fill=(185, 185, 185), width=1)
        draw.text(axis_xy[-1], axis_name, fill=(120, 120, 120))

    cur_valid = valid_steps[valid_steps <= current_step]
    if cur_valid.size == 0:
        cur_valid = valid_steps[:1]

    gt_full = to_xy(gt_rel[valid_steps])
    pred_full = to_xy(pred_rel[valid_steps])
    gt_cur = to_xy(gt_rel[cur_valid])
    pred_cur = to_xy(pred_rel[cur_valid])

    if len(gt_full) >= 2:
        draw.line(gt_full, fill=(150, 185, 245), width=2)
        draw.line(pred_full, fill=(245, 165, 165), width=2)
    if len(gt_cur) >= 2:
        draw.line(gt_cur, fill=(20, 90, 220), width=4)
        draw.line(pred_cur, fill=(220, 40, 40), width=4)

    draw.ellipse((origin[0] - 5, origin[1] - 5, origin[0] + 5, origin[1] + 5), fill=(40, 40, 40))
    for xy, color in ((gt_cur[-1], (20, 90, 220)), (pred_cur[-1], (220, 40, 40))):
        draw.ellipse((xy[0] - 5, xy[1] - 5, xy[0] + 5, xy[1] + 5), fill=color)

    dist = float(np.linalg.norm(gt_rel[min(current_step, gt_rel.shape[0] - 1)] - pred_rel[min(current_step, pred_rel.shape[0] - 1)]))
    final_dist = float(np.linalg.norm(gt_rel[valid_steps[-1]] - pred_rel[valid_steps[-1]]))
    draw.text((x0 + 8, y1 - 18), f"cur diff {dist:.3g} | final diff {final_dist:.3g}", fill=(70, 70, 70))


def _draw_action_3d_panel(
    gt_action: torch.Tensor,
    pred_action: torch.Tensor,
    *,
    frame_idx: int,
    num_video_frames: int,
    width: int,
    height: int,
    action_is_pad: Optional[torch.Tensor] = None,
) -> Image.Image:
    if gt_action.ndim == 3:
        gt_action = gt_action[0]
    if pred_action.ndim == 3:
        pred_action = pred_action[0]
    if gt_action.shape != pred_action.shape:
        raise ValueError(
            f"GT/pred action shape mismatch for plotting: gt={tuple(gt_action.shape)} pred={tuple(pred_action.shape)}"
        )

    gt_np = gt_action.detach().to(device="cpu", dtype=torch.float32).numpy()
    pred_np = pred_action.detach().to(device="cpu", dtype=torch.float32).numpy()
    action_len, action_dim = gt_np.shape
    valid_mask = _first_action_mask(action_is_pad, action_len)
    valid_steps = np.flatnonzero(valid_mask) if valid_mask is not None else np.arange(action_len)
    if valid_steps.size == 0:
        valid_steps = np.arange(action_len)

    panel = Image.new("RGB", (width, height), color=(250, 250, 250))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 5), "3D action trajectory (relative start)  GT=blue  Pred=red", fill=(20, 20, 20))

    xyz_slices = _action_xyz_slices(action_dim)
    if len(xyz_slices) == 0:
        draw.text((8, 28), f"Need >=3 action dims for XYZ trajectory, got {action_dim}", fill=(160, 30, 30))
        return panel

    current_step = 0 if num_video_frames <= 1 else round(frame_idx * (action_len - 1) / (num_video_frames - 1))

    top = 26
    gap = 8
    box_h = max((height - top - gap * (len(xyz_slices) - 1)) // len(xyz_slices), 1)
    for idx, (title, xyz_slice) in enumerate(xyz_slices):
        y0 = top + idx * (box_h + gap)
        y1 = height - 6 if idx == len(xyz_slices) - 1 else y0 + box_h
        _draw_single_3d_trajectory(
            draw,
            title=title,
            gt_xyz=gt_np[:, xyz_slice],
            pred_xyz=pred_np[:, xyz_slice],
            valid_steps=valid_steps,
            current_step=current_step,
            box=(8, y0, width - 8, y1),
        )

    return panel


def _save_stitched_video(
    pred_frames: list[Image.Image],
    gt_video: torch.Tensor,
    path: Path,
    fps: int,
    *,
    gt_action: Optional[torch.Tensor] = None,
    pred_action: Optional[torch.Tensor] = None,
    action_is_pad: Optional[torch.Tensor] = None,
    action_panel: str = "curves",
    state_joints: Optional[torch.Tensor] = None,
    inference_mark_stride: int = 1,
) -> None:
    pred_video = pil_frames_to_video_tensor(pred_frames)
    gt_video = ((gt_video.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()
    # Tensor layout is [3, T, H, W]: dim=2 is height (vertical stack), dim=3 is width (horizontal).
    stitched = torch.cat([gt_video, pred_video], dim=3).contiguous()
    frames = []
    mode = str(action_panel).strip().lower()
    for t in range(stitched.shape[1]):
        arr = (
            stitched[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0
        ).astype(np.uint8)
        frame = Image.fromarray(arr)
        if mode not in {"", "none"} and gt_action is not None and pred_action is not None:
            panel_width = max(480, min(int(gt_video.shape[3]), 960))
            h = int(gt_video.shape[2])
            if mode == "3d":
                action_panel_img = _draw_action_3d_panel(
                    gt_action=gt_action,
                    pred_action=pred_action,
                    frame_idx=t,
                    num_video_frames=int(stitched.shape[1]),
                    width=panel_width,
                    height=h,
                    action_is_pad=action_is_pad,
                )
            elif mode == "curves":
                action_panel_img = _draw_action_curves_panel(
                    gt_action=gt_action,
                    pred_action=pred_action,
                    frame_idx=t,
                    num_video_frames=int(stitched.shape[1]),
                    width=panel_width,
                    height=h,
                    action_is_pad=action_is_pad,
                    state_joints=state_joints,
                    inference_mark_stride=inference_mark_stride,
                )
            else:
                raise ValueError(
                    f"Unsupported OPENLOOP.video_action_panel={action_panel!r}. "
                    "Expected one of: ['none', 'curves', '3d']."
                )
            combined = Image.new("RGB", (frame.width + action_panel_img.width, frame.height), color=(255, 255, 255))
            combined.paste(frame, (0, 0))
            combined.paste(action_panel_img, (frame.width, 0))
            frame = combined
        frames.append(frame)
    save_mp4(frames, str(path), fps=fps)


def _dataset_index_for_sample(dataloader: DataLoader, sample_i: int, sample: dict[str, Any]) -> int:
    dataset = dataloader.dataset
    if isinstance(dataset, Subset):
        return int(dataset.indices[sample_i])
    dataset_index = _first_idx(sample)
    if dataset_index >= 0:
        return dataset_index
    return int(sample_i)


def _get_openloop_dataset(dataloader: DataLoader) -> Any:
    dataset = dataloader.dataset
    if isinstance(dataset, Subset):
        return dataset.dataset
    return dataset


def _episode_id_for_dataset_index(dataset: Any, dataset_index: int) -> int:
    episode_data_index = dataset.lerobot_dataset.episode_data_index
    starts = episode_data_index["from"]
    ends = episode_data_index["to"]
    for episode_id in range(int(starts.shape[0])):
        start = int(starts[episode_id].item())
        end = int(ends[episode_id].item())
        if start <= dataset_index < end:
            return episode_id
    raise ValueError(f"dataset_index {dataset_index} is not contained in any episode.")


def _normalized_gt_video_tensor(gt_video: torch.Tensor) -> torch.Tensor:
    return ((gt_video.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()


def _pil_frame_to_input_image(frame: Image.Image, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    arr = np.array(frame.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device=device, dtype=dtype)


def _normalize_video_obs_source(value: Any) -> str:
    key = str(value).strip().lower()
    if key not in {"gt_window", "autoregressive"}:
        raise ValueError(
            f"Unsupported OPENLOOP.video_obs_source={value!r}. Expected one of: ['gt_window', 'autoregressive']."
        )
    return key


def _normalize_video_action_source(value: Any) -> str:
    key = str(value).strip().lower()
    if key not in {"gt", "pred"}:
        raise ValueError(
            f"Unsupported OPENLOOP.video_action_source={value!r}. Expected one of: ['gt', 'pred']."
        )
    return key


def _episode_video_stem(episode_id: int, *, video_obs_source: str, video_action_source: str) -> str:
    parts = [f"episode_{episode_id:06d}_gt_pred"]
    if video_obs_source == "autoregressive":
        parts.append("obs_ar")
    if video_action_source == "pred":
        parts.append("act_pred")
    return "_".join(parts)


def _append_rollout_video_frames(
    accumulated: list[torch.Tensor],
    chunk: torch.Tensor,
    *,
    is_first_chunk: bool,
    skip_conditioning_frame: bool,
) -> None:
    if chunk.ndim != 4:
        raise ValueError(f"Expected rollout chunk shape [3, T, H, W], got {tuple(chunk.shape)}")
    start = 0 if is_first_chunk or not skip_conditioning_frame else 1
    for frame_idx in range(start, chunk.shape[1]):
        accumulated.append(chunk[:, frame_idx].clone())


def _save_episode_gt_pred_video(
    gt_frames: list[torch.Tensor],
    pred_frames: list[torch.Tensor],
    path: Path,
    fps: int,
) -> None:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        raise ValueError("Cannot save episode video from empty GT/pred frame lists.")
    if len(gt_frames) != len(pred_frames):
        raise ValueError(
            f"GT/pred episode frame count mismatch: gt={len(gt_frames)} vs pred={len(pred_frames)}"
        )
    gt_video = torch.stack(gt_frames, dim=1).contiguous()
    pred_video = torch.stack(pred_frames, dim=1).contiguous()
    stitched = torch.cat([gt_video, pred_video], dim=3).contiguous()
    frames = []
    for frame_idx in range(stitched.shape[1]):
        arr = (
            stitched[:, frame_idx].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0
        ).astype(np.uint8)
        frames.append(Image.fromarray(arr))
    save_mp4(frames, str(path), fps=fps)


def _call_action_infer(model: Any, infer_kwargs: dict[str, Any], num_video_frames: int) -> dict[str, Any]:
    if not hasattr(model, "infer_action"):
        raise AttributeError(f"{type(model).__name__} does not implement infer_action().")
    signature = inspect.signature(model.infer_action)
    if "num_video_frames" in signature.parameters:
        infer_kwargs["num_video_frames"] = num_video_frames
    return model.infer_action(**infer_kwargs)


def _episode_sample_indices(
    dataset: Any,
    episode_indices: list[int],
    frame_stride: int,
    max_samples: Optional[int],
) -> list[int]:
    episode_data_index = dataset.lerobot_dataset.episode_data_index
    starts = episode_data_index["from"]
    ends = episode_data_index["to"]
    num_episodes = int(starts.shape[0])
    stride = max(int(frame_stride), 1)

    sample_indices: list[int] = []
    for episode_index in episode_indices:
        if not (0 <= episode_index < num_episodes):
            raise ValueError(
                f"Episode index {episode_index} is out of range for split "
                f"{cfg_split_name(dataset)}: valid range is [0, {num_episodes})."
            )
        start = int(starts[episode_index].item())
        end = int(ends[episode_index].item())
        sample_indices.extend(range(start, end, stride))

    if max_samples is not None:
        sample_indices = sample_indices[: int(max_samples)]
    if len(sample_indices) == 0:
        raise ValueError("No samples selected for open-loop evaluation.")
    return sample_indices


def cfg_split_name(dataset: Any) -> str:
    return "train" if bool(dataset.lerobot_dataset.is_training_set) else "val"


def _mean(values: list[float]) -> Optional[float]:
    if len(values) == 0:
        return None
    return float(sum(values) / len(values))


def load_openloop_model_for_eval(cfg: DictConfig) -> tuple[Any, Path, Path]:
    """Load checkpoint and build the model. Safe to call once and reuse while iterating on eval code."""
    if _is_none_like(cfg.ckpt):
        raise ValueError("`ckpt` must be provided.")

    ckpt_path = _resolve_path(str(cfg.ckpt))
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)
    output_dir = _resolve_path(str(cfg.OPENLOOP.output_dir), must_exist=False) / ckpt_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    # RobotVideoDataset mirrors loaded normalization stats into misc.get_work_dir().
    # In standalone eval scripts the default is ./runs, so make sure it exists.
    (PROJECT_ROOT / "runs").mkdir(parents=True, exist_ok=True)

    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    device = str(cfg.OPENLOOP.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; falling back to CPU.")
        device = "cpu"

    logger.info("Loading model from %s", ckpt_path)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.load_checkpoint(str(ckpt_path))
    model.eval()
    return model, ckpt_path, output_dir


def prepare_openloop_dataloader(cfg: DictConfig) -> tuple[DataLoader, Any]:
    """Build dataset + dataloader. Re-run when changing OPENLOOP episode / stride / split fields."""
    device = str(cfg.OPENLOOP.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    dataset = _instantiate_openloop_dataset(cfg)
    max_samples = _to_optional_int(cfg.OPENLOOP.get("max_samples"))
    episode_indices = _to_optional_int_list(cfg.OPENLOOP.get("episode_indices"))
    eval_dataset = dataset
    if episode_indices is not None:
        selected_indices = _episode_sample_indices(
            dataset=dataset,
            episode_indices=episode_indices,
            frame_stride=int(cfg.OPENLOOP.get("frame_stride", 1)),
            max_samples=max_samples,
        )
        eval_dataset = Subset(dataset, selected_indices)
        logger.info(
            "Open-loop episode subset: split=%s episodes=%s frame_stride=%d samples=%d",
            str(cfg.OPENLOOP.split),
            episode_indices,
            int(cfg.OPENLOOP.get("frame_stride", 1)),
            len(selected_indices),
        )

    dataloader = DataLoader(
        eval_dataset,
        batch_size=int(cfg.OPENLOOP.batch_size),
        shuffle=False,
        num_workers=int(cfg.OPENLOOP.num_workers),
        pin_memory=device.startswith("cuda"),
    )
    if int(cfg.OPENLOOP.batch_size) != 1:
        raise ValueError("OPENLOOP.batch_size must be 1 because model inference is single-sample.")

    processor = dataset.lerobot_dataset.processor
    return dataloader, processor


def run_openloop_evaluation(
    cfg: DictConfig,
    model: Any,
    dataloader: DataLoader,
    processor: Any,
    *,
    output_dir: Path,
    ckpt_path: Path,
) -> None:
    max_samples = _to_optional_int(cfg.OPENLOOP.get("max_samples"))
    predict_video = bool(cfg.OPENLOOP.predict_video)
    save_video_samples = int(cfg.OPENLOOP.save_video_samples)
    save_episode_video = bool(cfg.OPENLOOP.get("save_episode_video", False))
    episode_indices = _to_optional_int_list(cfg.OPENLOOP.get("episode_indices"))
    skip_conditioning_frame = bool(cfg.OPENLOOP.get("episode_video_skip_conditioning_frame", True))
    if save_episode_video and not predict_video:
        raise ValueError("OPENLOOP.save_episode_video=true requires OPENLOOP.predict_video=true.")
    if save_episode_video and episode_indices is None:
        raise ValueError("OPENLOOP.save_episode_video=true requires OPENLOOP.episode_indices to be set.")
    video_obs_source = _normalize_video_obs_source(cfg.OPENLOOP.get("video_obs_source", "gt_window"))
    video_action_source = _normalize_video_action_source(cfg.OPENLOOP.get("video_action_source", "gt"))

    rows = []
    action_metric_keys = ["action_l1", "action_mse", "action_rmse", "action_max_abs"]
    video_metric_keys = ["video_psnr", "video_ssim", "video_l1", "video_mse"]
    aggregate: dict[str, list[float]] = {key: [] for key in action_metric_keys + video_metric_keys}
    per_dim_l1_sum = None
    per_dim_mse_sum = None
    per_dim_count = 0
    episode_gt_frames: dict[int, list[torch.Tensor]] = {}
    episode_pred_frames: dict[int, list[torch.Tensor]] = {}
    episode_chunk_count: dict[int, int] = {}
    episode_next_input_image: dict[int, Optional[torch.Tensor]] = {}
    openloop_dataset = _get_openloop_dataset(dataloader)

    start_time = time.perf_counter()
    for sample_i, sample in enumerate(dataloader):
        if max_samples is not None and sample_i >= max_samples:
            break

        video = sample["video"][0]  # [3, T_video, H, W], range [-1, 1]
        gt_action_norm = sample["action"][0].detach().to(device="cpu", dtype=torch.float32)
        proprio_norm = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
        proprio0 = sample["proprio"][0, 0].detach().to(device="cpu", dtype=torch.float32)
        num_video_frames = int(video.shape[1])
        action_horizon = int(gt_action_norm.shape[0])
        episode_id: Optional[int] = None
        if predict_video and (save_episode_video or video_obs_source == "autoregressive"):
            dataset_index = _dataset_index_for_sample(dataloader, sample_i, sample)
            episode_id = _episode_id_for_dataset_index(openloop_dataset, dataset_index)

        if video_obs_source == "autoregressive":
            if episode_id is None:
                raise ValueError("OPENLOOP.video_obs_source=autoregressive requires episode_indices to be set.")
            cached_input = episode_next_input_image.get(episode_id)
            if cached_input is None:
                input_image = video[:, 0].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
            else:
                input_image = cached_input
        else:
            input_image = video[:, 0].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)

        infer_kwargs = {
            "prompt": None,
            "input_image": input_image,
            "action_horizon": action_horizon,
            "proprio": proprio0,
            "num_inference_steps": int(cfg.OPENLOOP.num_inference_steps),
            "sigma_shift": _to_optional_float(cfg.OPENLOOP.get("sigma_shift")),
            "seed": _to_optional_int(cfg.OPENLOOP.get("seed")),
            "rand_device": str(cfg.OPENLOOP.rand_device),
            "tiled": bool(cfg.OPENLOOP.tiled),
        }
        if "context" in sample and "context_mask" in sample:
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = _first_prompt(sample)

        if predict_video:
            action_for_video = gt_action_norm
            if video_action_source == "pred":
                action_only = _call_action_infer(model, dict(infer_kwargs), num_video_frames=num_video_frames)
                pred_action_for_video = action_only.get("action")
                if pred_action_for_video is None:
                    raise ValueError("infer_action did not return `action` for video_action_source=pred.")
                if pred_action_for_video.ndim == 2:
                    pred_action_for_video = pred_action_for_video.unsqueeze(0)
                action_for_video = pred_action_for_video[0].detach().to(device="cpu", dtype=torch.float32)

            joint_kwargs = dict(infer_kwargs)
            joint_kwargs.update(
                {
                    "num_frames": num_video_frames,
                    "action": action_for_video,
                    "text_cfg_scale": float(cfg.OPENLOOP.text_cfg_scale),
                    "action_cfg_scale": 1.0,
                    "negative_prompt": str(cfg.OPENLOOP.negative_prompt),
                }
            )
            pred = model.infer(**joint_kwargs)

            if video_obs_source == "autoregressive":
                assert episode_id is not None
                episode_next_input_image[episode_id] = _pil_frame_to_input_image(
                    pred["video"][-1],
                    device=model.device,
                    dtype=model.torch_dtype,
                )
        else:
            pred = _call_action_infer(model, infer_kwargs, num_video_frames=num_video_frames)

        pred_action = pred.get("action")
        if pred_action is None:
            raise ValueError("Model inference did not return `action`.")
        if pred_action.ndim == 2:
            pred_action = pred_action.unsqueeze(0)
        gt_action_btd = gt_action_norm.unsqueeze(0)

        pred_action_denorm = _denormalize_action(processor, pred_action, proprio_norm)
        gt_action_denorm = _denormalize_action(processor, gt_action_btd, proprio_norm)
        action_metrics = _action_metrics(
            pred_action=pred_action_denorm,
            gt_action=gt_action_denorm,
            action_is_pad=sample.get("action_is_pad"),
        )

        per_dim_l1 = action_metrics.pop("action_l1_per_dim")
        per_dim_mse = action_metrics.pop("action_mse_per_dim")
        per_dim_l1_sum = per_dim_l1 if per_dim_l1_sum is None else per_dim_l1_sum + per_dim_l1
        per_dim_mse_sum = per_dim_mse if per_dim_mse_sum is None else per_dim_mse_sum + per_dim_mse
        per_dim_count += 1

        row = {
            "sample_index": sample_i,
            "dataset_index": _dataset_index_for_sample(dataloader, sample_i, sample),
            "video_obs_source": video_obs_source,
            "video_action_source": video_action_source,
            **{k: action_metrics[k] for k in action_metric_keys},
            "num_action_steps": action_metrics["num_action_steps"],
        }
        for key in action_metric_keys:
            aggregate[key].append(float(row[key]))

        if predict_video:
            if "video" not in pred:
                raise ValueError("OPENLOOP.predict_video=true but model inference did not return `video`.")
            video_metrics = _video_metrics(
                pred_frames=pred["video"],
                gt_video=video,
                image_is_pad=sample.get("image_is_pad"),
            )
            row.update(video_metrics)
            for key in video_metric_keys:
                aggregate[key].append(float(video_metrics[key]))

            if sample_i < save_video_samples:
                panel_mode = str(cfg.OPENLOOP.get("video_action_panel", "curves")).strip().lower()
                mark_stride = _to_optional_int(cfg.OPENLOOP.get("video_action_inference_mark_stride"))
                if mark_stride is None:
                    mark_stride = action_horizon

                state_denorm: Optional[torch.Tensor] = None
                if panel_mode == "curves":
                    try:
                        sd = _denormalize_state(processor, proprio_norm, gt_action_btd)
                        if sd.shape == gt_action_denorm.shape:
                            state_denorm = sd
                    except Exception as exc:
                        logger.debug("State trajectory overlay skipped: %s", exc)

                if panel_mode == "none":
                    video_stem = f"sample_{sample_i:06d}_gt_pred"
                elif panel_mode == "3d":
                    video_stem = f"sample_{sample_i:06d}_gt_pred_action3d"
                elif panel_mode == "curves":
                    video_stem = f"sample_{sample_i:06d}_gt_pred_action"
                else:
                    raise ValueError(
                        f"Unsupported OPENLOOP.video_action_panel={panel_mode!r}. "
                        "Expected one of: ['none', 'curves', '3d']."
                    )
                video_path = output_dir / f"{video_stem}.mp4"
                _save_stitched_video(
                    pred_frames=pred["video"],
                    gt_video=video,
                    path=video_path,
                    fps=int(cfg.OPENLOOP.video_fps),
                    gt_action=gt_action_denorm,
                    pred_action=pred_action_denorm,
                    action_is_pad=sample.get("action_is_pad"),
                    action_panel=panel_mode,
                    state_joints=state_denorm,
                    inference_mark_stride=int(mark_stride),
                )
                row["video_path"] = str(video_path)

            if save_episode_video:
                if episode_id is None:
                    dataset_index = _dataset_index_for_sample(dataloader, sample_i, sample)
                    episode_id = _episode_id_for_dataset_index(openloop_dataset, dataset_index)
                is_first_chunk = episode_chunk_count.get(episode_id, 0) == 0
                gt_chunk = _normalized_gt_video_tensor(video)
                pred_chunk = pil_frames_to_video_tensor(pred["video"])
                _append_rollout_video_frames(
                    episode_gt_frames.setdefault(episode_id, []),
                    gt_chunk,
                    is_first_chunk=is_first_chunk,
                    skip_conditioning_frame=skip_conditioning_frame,
                )
                _append_rollout_video_frames(
                    episode_pred_frames.setdefault(episode_id, []),
                    pred_chunk,
                    is_first_chunk=is_first_chunk,
                    skip_conditioning_frame=skip_conditioning_frame,
                )
                episode_chunk_count[episode_id] = episode_chunk_count.get(episode_id, 0) + 1

        rows.append(row)
        if (sample_i + 1) % int(cfg.OPENLOOP.log_every) == 0:
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Processed %d samples | action_l1=%.6f | elapsed=%.1fs",
                sample_i + 1,
                _mean(aggregate["action_l1"]) or float("nan"),
                elapsed,
            )

    if len(rows) == 0:
        raise RuntimeError("No samples were evaluated.")

    episode_video_paths: dict[int, str] = {}
    if save_episode_video:
        for episode_id in sorted(episode_gt_frames.keys()):
            episode_video_path = output_dir / f"{_episode_video_stem(episode_id, video_obs_source=video_obs_source, video_action_source=video_action_source)}.mp4"
            _save_episode_gt_pred_video(
                gt_frames=episode_gt_frames[episode_id],
                pred_frames=episode_pred_frames[episode_id],
                path=episode_video_path,
                fps=int(cfg.OPENLOOP.video_fps),
            )
            episode_video_paths[episode_id] = str(episode_video_path)
            logger.info(
                "Saved episode rollout video: episode=%d frames=%d path=%s",
                episode_id,
                len(episode_gt_frames[episode_id]),
                episode_video_path,
            )

    summary = {
        "ckpt": str(ckpt_path),
        "split": str(cfg.OPENLOOP.split),
        "num_samples": len(rows),
        "predict_video": predict_video,
        "save_episode_video": save_episode_video,
        "video_obs_source": video_obs_source,
        "video_action_source": video_action_source,
        "metrics": {key: _mean(values) for key, values in aggregate.items() if len(values) > 0},
        "elapsed_sec": float(time.perf_counter() - start_time),
    }
    if episode_video_paths:
        summary["episode_video_paths"] = {str(k): v for k, v in episode_video_paths.items()}
    if per_dim_l1_sum is not None and per_dim_mse_sum is not None and per_dim_count > 0:
        summary["metrics"]["action_l1_per_dim"] = _jsonable(per_dim_l1_sum / per_dim_count)
        summary["metrics"]["action_mse_per_dim"] = _jsonable(per_dim_mse_sum / per_dim_count)

    rows_path = output_dir / "per_sample.csv"
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, default=_jsonable), encoding="utf-8")

    config_path = output_dir / "config.yaml"
    OmegaConf.save(config=cfg, f=str(config_path), resolve=True)

    logger.info("Open-loop summary saved to %s", summary_path)
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=_jsonable))


def run_openloop(cfg: DictConfig) -> None:
    setup_logging()
    model, ckpt_path, output_dir = load_openloop_model_for_eval(cfg)
    dataloader, processor = prepare_openloop_dataloader(cfg)
    run_openloop_evaluation(
        cfg,
        model,
        dataloader,
        processor,
        output_dir=output_dir,
        ckpt_path=ckpt_path,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="openloop_robotwin.yaml")
def main(cfg: DictConfig) -> None:
    run_openloop(cfg)


if __name__ == "__main__":
    main()
