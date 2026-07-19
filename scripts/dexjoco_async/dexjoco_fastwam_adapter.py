"""DexJoCo sim ↔ FastWAM policy-server adapter (closed-loop eval)."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hashlib

import numpy as np
import yaml
from PIL import Image

KEY_INPUT_IMAGE = "input_image"
KEY_PROPRIO = "proprio"
KEY_PROMPT = "prompt"
KEY_CONTEXT = "context"
KEY_CONTEXT_MASK = "context_mask"
KEY_ACTION = "action"

DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)

EVEROOBOT_FULL_EPISODE_DATASET = "EveRobotFullEpisodeDataset"
DEFAULT_SLIDING_WINDOW_ACTION_HORIZON = 32


def is_everobot_full_episode_train(train_data: dict[str, Any]) -> bool:
    """True when the run was trained with variable-length full-episode EveRobot data."""
    target = str(train_data.get("_target_", ""))
    return EVEROOBOT_FULL_EPISODE_DATASET in target


def resolve_eval_action_horizon(
    train_data: dict[str, Any],
    *,
    action_horizon_override: int | None = None,
) -> int:
    """Resolve closed-loop action chunk size for DexJoCo eval.

    - Sliding-window (LeRobot) runs: use ``num_frames - 1`` from training config.
    - EveRobot full-episode runs: require ``action_horizon_override`` (no fixed train T).
    """
    if action_horizon_override is not None:
        return int(action_horizon_override)

    num_frames = train_data.get("num_frames")
    if num_frames is not None:
        return int(num_frames) - 1

    if is_everobot_full_episode_train(train_data):
        raise ValueError(
            "EveRobot full-episode training config has no fixed `num_frames`. "
            "Pass --action-horizon to specify inference chunk size (e.g. 32 or 180)."
        )

    return DEFAULT_SLIDING_WINDOW_ACTION_HORIZON


def _path_from_config(value: Any, *, run_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in (Path.cwd(), run_dir):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def _as_path_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_train_dataset_roots(train_data: dict[str, Any], *, run_dir: Path) -> list[Path]:
    roots = [_path_from_config(path, run_dir=run_dir) for path in _as_path_list(train_data.get("dataset_dirs"))]
    if roots:
        return roots

    manifest_path = train_data.get("manifest_path")
    if manifest_path:
        path = _path_from_config(manifest_path, run_dir=run_dir)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            return [
                _path_from_config(root, run_dir=run_dir)
                for root in manifest.get("dataset_roots", {}).values()
            ]
    return []


def _load_eve_action_schema(train_data: dict[str, Any], *, run_dir: Path) -> dict[str, Any]:
    for root in _resolve_train_dataset_roots(train_data, run_dir=run_dir):
        schema_path = root / "meta" / "eve" / "action_schema.json"
        if schema_path.exists():
            with schema_path.open("r", encoding="utf-8") as f:
                schema = json.load(f)
            schema["_schema_path"] = str(schema_path)
            return schema
    return {}

os.environ.setdefault("MUJOCO_GL", "egl")

CLICK_MOUSE_ALIGN_ROTVEC = np.array(
    [
        -4.4294e-01,
        1.3729e-06,
        1.5170e00,
        -3.14156462e00,
        -6.91584035e-05,
        -1.40317984e-03,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.263,
        0,
        0,
        0,
    ],
    dtype=np.float64,
)
CLICK_MOUSE_ALIGN_STEPS = 30

DEFAULT_TASK_CONFIG_DIR = Path("third_party/dexjoco/configs/rand_obj")

# Robotwin 3-cam mosaic (matches robot_video_dataset.concat_multi_camera="robotwin").
ROBOTWIN_TOP_SIZE_WH = (320, 256)
ROBOTWIN_WRIST_SIZE_WH = (160, 128)


def resize_rgb(rgb: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    return np.asarray(image.resize(size_wh, resample=Image.BILINEAR), dtype=np.uint8)


def hwc_rgb_to_input_image_np(rgb: np.ndarray) -> np.ndarray:
    """HWC uint8 → [1,3,H,W] float32 in [-1, 1] (no resize)."""
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    tensor = rgb.transpose(2, 0, 1).astype(np.float32)
    tensor = tensor * (2.0 / 255.0) - 1.0
    return tensor[np.newaxis, ...]


def rgb_to_input_image_np(rgb: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """HWC uint8 → [1,3,H,W] float32 in [-1, 1]."""
    return hwc_rgb_to_input_image_np(resize_rgb(rgb, size_wh))


def concat_robotwin_rgb(top: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Stack ego + wrist cameras into a 384x320 RGB mosaic (HWC uint8)."""
    head = resize_rgb(top, ROBOTWIN_TOP_SIZE_WH)
    wrist_left = resize_rgb(left, ROBOTWIN_WRIST_SIZE_WH)
    wrist_right = resize_rgb(right, ROBOTWIN_WRIST_SIZE_WH)
    bottom = np.concatenate([wrist_left, wrist_right], axis=1)
    return np.ascontiguousarray(np.concatenate([head, bottom], axis=0), dtype=np.uint8)


def resolve_env_camera_keys(
    image_keys: list[str],
    camera_mapping: dict[str, str],
) -> list[str]:
    """Map training shape_meta image keys to DexJoCo env observation keys."""
    env_keys: list[str] = []
    for key in image_keys:
        if key in camera_mapping:
            env_keys.append(camera_mapping[key])
        elif key == "ego" and "base" in camera_mapping:
            env_keys.append(camera_mapping["base"])
        elif key in camera_mapping.values():
            env_keys.append(key)
        else:
            raise ValueError(
                f"Cannot map training image key {key!r} via camera_mapping {camera_mapping}"
            )
    return env_keys


def load_text_context_arrays(
    instruction: str,
    *,
    text_embedding_cache_dir: str | Path,
    context_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached T5 context as numpy arrays (torch-free, .npz only).

    Requires pre-exported ``.npz`` caches produced by
    ``scripts/export_text_embed_cache_npz.py`` (run in the ``fastwam`` env).
    The ``.pt`` format is intentionally unsupported here to keep the dexjoco
    eval client free of any torch dependency.
    """
    cache_dir = Path(text_embedding_cache_dir)
    hashed = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    npz_path = cache_dir / f"{hashed}.t5_len{context_len}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing torch-free text embedding cache (.npz): {npz_path}. "
            "The dexjoco eval client does not import torch, so .pt caches are "
            "not supported. Run in the fastwam env:\n"
            f"  python scripts/export_text_embed_cache_npz.py --cache-dir {cache_dir}"
        )
    payload = np.load(npz_path)
    context = payload["context"].astype(np.float32)
    context_mask = payload["mask"].astype(bool)

    context = context.copy()
    context[~context_mask] = 0.0
    context_mask = np.ones_like(context_mask, dtype=bool)
    return context, context_mask


def load_dexjoco_eval_settings(
    run_dir: Path,
    *,
    action_horizon_override: int | None = None,
    text_embedding_cache_dir_override: str | Path | None = None,
) -> dict[str, Any]:
    """Training-run settings for DexJoCo closed-loop eval."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing training config: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train_data = cfg["data"]["train"]
    image_size = tuple(int(x) for x in train_data["video_size"])
    processor = train_data["processor"]
    shape_meta = train_data["shape_meta"]
    image_keys = [str(item["key"]) for item in shape_meta["images"]]
    image_sizes_wh = [(int(item["shape"][2]), int(item["shape"][1])) for item in shape_meta["images"]]
    everobot_full_episode = is_everobot_full_episode_train(train_data)
    action_schema = _load_eve_action_schema(train_data, run_dir=run_dir)
    policy_action_output_dim = int(action_schema.get("policy_action_dim", processor["action_output_dim"]))
    control_action_slice = action_schema.get("control_action_slice")
    if control_action_slice is None:
        prefix_dim = int(action_schema.get("policy_action_prefix_dim", 0))
        control_action_slice = [prefix_dim, policy_action_output_dim]
    return {
        "image_size_wh": (image_size[1], image_size[0]),
        "action_horizon": resolve_eval_action_horizon(
            train_data,
            action_horizon_override=action_horizon_override,
        ),
        "everobot_full_episode": everobot_full_episode,
        "action_output_dim": policy_action_output_dim,
        "policy_action_prefix_dim": int(action_schema.get("policy_action_prefix_dim", 0)),
        "policy_action_control_slice": [int(control_action_slice[0]), int(control_action_slice[1])],
        "eve_action_schema_path": action_schema.get("_schema_path"),
        "proprio_output_dim": int(processor["proprio_output_dim"]),
        "text_embedding_cache_dir": (
            str(Path(text_embedding_cache_dir_override).expanduser().resolve())
            if text_embedding_cache_dir_override is not None
            else train_data.get("text_embedding_cache_dir")
        ),
        "context_len": int(train_data.get("context_len", 128)),
        "load_text_encoder": bool(cfg.get("model", {}).get("load_text_encoder", False)),
        "concat_multi_camera": train_data.get("concat_multi_camera"),
        "image_keys": image_keys,
        "image_sizes_wh": image_sizes_wh,
    }


def load_task_configs(config_dir: Path) -> list[dict[str, Any]]:
    """Load all DexJoCo rand_obj task YAML configs (11 tasks)."""
    config_dir = config_dir.resolve()
    if not config_dir.exists():
        raise FileNotFoundError(f"Task config directory not found: {config_dir}")
    configs: list[dict[str, Any]] = []
    for path in sorted(config_dir.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["_config_path"] = str(path)
        configs.append(cfg)
    if not configs:
        raise FileNotFoundError(f"No task YAML files under {config_dir}")
    return configs


def _rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rotvec / angle
    half = 0.5 * angle
    sin_half = np.sin(half)
    return np.array(
        [np.cos(half), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half],
        dtype=np.float64,
    )


def _quat_wxyz_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    quat = quat / norm
    w = float(np.clip(quat[0], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float64)
    axis = quat[1:4] / sin_half
    return axis * angle


def rotvec_action_to_env_quat(action_rotvec: np.ndarray, *, dual_arm: bool) -> np.ndarray:
    """Convert policy rotvec action (22/44-dim) to DexJoCo quat action (23/46-dim)."""
    action_rotvec = np.asarray(action_rotvec, dtype=np.float64)
    if dual_arm:
        r_xyz = action_rotvec[:3]
        r_rotvec = action_rotvec[3:6]
        r_hand = action_rotvec[6:22]
        l_xyz = action_rotvec[22:25]
        l_rotvec = action_rotvec[25:28]
        l_hand = action_rotvec[28:44]
        r_quat = _rotvec_to_quat_wxyz(r_rotvec)
        l_quat = _rotvec_to_quat_wxyz(l_rotvec)
        return np.concatenate([r_xyz, r_quat, l_xyz, l_quat, r_hand, l_hand])

    xyz = action_rotvec[:3]
    rotvec = action_rotvec[3:6]
    hand = action_rotvec[6:22]
    quat = _rotvec_to_quat_wxyz(rotvec)
    return np.concatenate([xyz, quat, hand])


def clamp_rotvec_action_to_state(
    action_rotvec: np.ndarray,
    state: np.ndarray,
    *,
    dual_arm: bool,
    max_displacement: float,
    max_dz_down: float | None = None,
) -> np.ndarray:
    """Limit absolute xyz targets to stay near the current proprio state."""
    action = np.asarray(action_rotvec, dtype=np.float64).copy()
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    arm_specs = [(slice(0, 3), 0), (slice(22, 25), 7)] if dual_arm else [(slice(0, 3), 0)]
    for xyz_sl, state_start in arm_specs:
        state_xyz = state[state_start : state_start + 3]
        delta = action[xyz_sl] - state_xyz
        if max_dz_down is not None and delta[2] < -max_dz_down:
            delta[2] = -max_dz_down
        norm = float(np.linalg.norm(delta))
        if norm > max_displacement:
            delta = delta * (max_displacement / norm)
        action[xyz_sl] = state_xyz + delta
    return action.astype(np.float32)


def state_to_rotvec_reference(state: np.ndarray, *, dual_arm: bool) -> np.ndarray:
    """Convert env proprio state to the 22/44-dim rotvec action reference frame."""
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if dual_arm:
        r_arm = state[:7]
        l_arm = state[7:14]
        r_hand = state[14:30]
        l_hand = state[30:46]
        return np.concatenate(
            [
                r_arm[:3],
                _quat_wxyz_to_rotvec(r_arm[3:7]),
                r_hand,
                l_arm[:3],
                _quat_wxyz_to_rotvec(l_arm[3:7]),
                l_hand,
            ]
        ).astype(np.float32)
    arm = state[:7]
    hand = state[7:23]
    return np.concatenate([arm[:3], _quat_wxyz_to_rotvec(arm[3:7]), hand]).astype(np.float32)


@dataclass(frozen=True)
class ActionConstraintConfig:
    """Per-step action amplitude limits for closed-loop eval."""

    max_xyz_step: float = 0.05
    max_rot_step: float = 0.0
    max_hand_step: float = 0.0
    max_dz_down: float | None = 0.03
    clip_to_dataset_bounds: bool = False


def _load_action_bounds(stats_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    import json

    payload = json.loads(Path(stats_path).expanduser().read_text(encoding="utf-8"))
    action_stats = payload["action"]["default"]
    action_min = np.asarray(action_stats["global_min"], dtype=np.float64)
    action_max = np.asarray(action_stats["global_max"], dtype=np.float64)
    return action_min, action_max


def constrain_rotvec_action(
    action_rotvec: np.ndarray,
    state: np.ndarray,
    *,
    dual_arm: bool,
    config: ActionConstraintConfig,
    action_min: np.ndarray | None = None,
    action_max: np.ndarray | None = None,
) -> np.ndarray:
    """Clamp one policy action to a normal per-step range around current state."""
    action = np.asarray(action_rotvec, dtype=np.float64).copy()
    reference = np.asarray(state_to_rotvec_reference(state, dual_arm=dual_arm), dtype=np.float64)

    if dual_arm:
        arm_specs = [
            (slice(0, 3), slice(3, 6), slice(6, 22)),
            (slice(22, 25), slice(25, 28), slice(28, 44)),
        ]
    else:
        arm_specs = [(slice(0, 3), slice(3, 6), slice(6, 22))]

    for xyz_sl, rot_sl, hand_sl in arm_specs:
        ref_xyz = reference[xyz_sl]
        delta_xyz = action[xyz_sl] - ref_xyz
        if config.max_dz_down is not None and delta_xyz[2] < -config.max_dz_down:
            delta_xyz[2] = -config.max_dz_down
        xyz_norm = float(np.linalg.norm(delta_xyz))
        if xyz_norm > config.max_xyz_step:
            delta_xyz = delta_xyz * (config.max_xyz_step / xyz_norm)
        action[xyz_sl] = ref_xyz + delta_xyz

        if config.max_rot_step > 0.0:
            delta_rot = action[rot_sl] - reference[rot_sl]
            rot_norm = float(np.linalg.norm(delta_rot))
            if rot_norm > config.max_rot_step:
                delta_rot = delta_rot * (config.max_rot_step / rot_norm)
            action[rot_sl] = reference[rot_sl] + delta_rot

        if config.max_hand_step > 0.0:
            delta_hand = action[hand_sl] - reference[hand_sl]
            hand_norm = float(np.linalg.norm(delta_hand))
            if hand_norm > config.max_hand_step:
                delta_hand = delta_hand * (config.max_hand_step / hand_norm)
            action[hand_sl] = reference[hand_sl] + delta_hand

    if config.clip_to_dataset_bounds and action_min is not None and action_max is not None:
        action = np.clip(action, action_min, action_max)

    return action.astype(np.float32)


def _safe_rgb_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=2)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if np.nanmax(arr) <= 1.0:
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:
                arr = np.clip(arr, 0.0, 255.0)
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


@dataclass
class DexJoCoTaskConfig:
    env_name: str
    prompt: str
    dual_arm: bool
    camera_key: str
    camera_mapping: dict[str, str]
    password: list[int] | None = None

    @classmethod
    def from_yaml(cls, cfg: dict[str, Any]) -> DexJoCoTaskConfig:
        camera_mapping = {str(k): str(v) for k, v in cfg["camera_mapping"].items()}
        base_key = camera_mapping.get("base")
        if base_key is None:
            raise ValueError(f"camera_mapping must contain 'base' key: {cfg}")
        return cls(
            env_name=str(cfg["env_name"]),
            prompt=str(cfg["prompt"]),
            dual_arm=str(cfg.get("robot_type", "single_arm")) == "dual_arm",
            camera_key=str(base_key),
            camera_mapping=camera_mapping,
            password=cfg.get("password"),
        )


class DexJoCoFastWAMAdapter:
    """Maps DexJoCo env observations/actions ↔ FastWAM policy-server I/O."""

    def __init__(self, eval_settings: dict[str, Any]) -> None:
        self.image_size_wh: tuple[int, int] = eval_settings["image_size_wh"]
        self.action_horizon = int(eval_settings["action_horizon"])
        self.policy_action_output_dim = int(eval_settings["action_output_dim"])
        self.policy_action_prefix_dim = int(eval_settings.get("policy_action_prefix_dim", 0))
        control_slice = eval_settings.get("policy_action_control_slice")
        if control_slice is None:
            control_slice = [self.policy_action_prefix_dim, self.policy_action_output_dim]
        if not isinstance(control_slice, (list, tuple)) or len(control_slice) != 2:
            raise ValueError(f"Invalid policy_action_control_slice: {control_slice}")
        self.policy_action_control_start = int(control_slice[0])
        self.policy_action_control_end = int(control_slice[1])
        if not (0 <= self.policy_action_control_start < self.policy_action_control_end <= self.policy_action_output_dim):
            raise ValueError(
                "Invalid policy action control slice: "
                f"{control_slice}, policy_action_output_dim={self.policy_action_output_dim}"
            )
        self.action_output_dim = self.policy_action_control_end - self.policy_action_control_start
        if self.action_output_dim <= 0:
            raise ValueError(
                "Invalid action dims: "
                f"policy_action_output_dim={self.policy_action_output_dim}, "
                f"policy_action_control_slice={control_slice}"
            )
        self.proprio_output_dim = int(eval_settings["proprio_output_dim"])
        self.text_embedding_cache_dir = eval_settings.get("text_embedding_cache_dir")
        self.context_len = int(eval_settings["context_len"])
        self.use_prompt = bool(eval_settings["load_text_encoder"])
        self.concat_multi_camera = eval_settings.get("concat_multi_camera")
        self.image_keys: list[str] = list(eval_settings.get("image_keys") or [])
        self.image_sizes_wh: list[tuple[int, int]] = [
            tuple(map(int, item)) for item in eval_settings.get("image_sizes_wh", [])
        ]

    def task_prompt(self, task_prompt: str) -> str:
        return DEFAULT_PROMPT.format(task=task_prompt)

    def _build_input_image(
        self,
        env_obs: dict[str, Any],
        *,
        camera_key: str,
        camera_mapping: dict[str, str],
    ) -> np.ndarray:
        if self.concat_multi_camera == "robotwin":
            if len(self.image_keys) != 3:
                raise ValueError(
                    f"concat_multi_camera='robotwin' requires 3 image keys, got {self.image_keys}"
                )
            env_cam_keys = resolve_env_camera_keys(self.image_keys, camera_mapping)
            top = _safe_rgb_uint8(env_obs[env_cam_keys[0]])
            left = _safe_rgb_uint8(env_obs[env_cam_keys[1]])
            right = _safe_rgb_uint8(env_obs[env_cam_keys[2]])
            rgb = concat_robotwin_rgb(top, left, right)
            return hwc_rgb_to_input_image_np(rgb)

        if self.concat_multi_camera in {"horizontal", "vertical"} and len(self.image_keys) > 1:
            if len(self.image_sizes_wh) != len(self.image_keys):
                raise ValueError(
                    "Multi-camera eval requires one shape_meta image size per image key, "
                    f"got keys={self.image_keys}, sizes={self.image_sizes_wh}"
                )
            env_cam_keys = resolve_env_camera_keys(self.image_keys, camera_mapping)
            tiles = [
                resize_rgb(_safe_rgb_uint8(env_obs[env_key]), size_wh)
                for env_key, size_wh in zip(env_cam_keys, self.image_sizes_wh)
            ]
            axis = 1 if self.concat_multi_camera == "horizontal" else 0
            rgb = np.ascontiguousarray(np.concatenate(tiles, axis=axis), dtype=np.uint8)
            if (rgb.shape[1], rgb.shape[0]) != self.image_size_wh:
                rgb = resize_rgb(rgb, self.image_size_wh)
            return hwc_rgb_to_input_image_np(rgb)

        rgb = _safe_rgb_uint8(env_obs[camera_key])
        return rgb_to_input_image_np(rgb, self.image_size_wh)

    def env_obs_to_policy_obs(
        self,
        env_obs: dict[str, Any],
        *,
        camera_key: str,
        camera_mapping: dict[str, str],
        task_prompt: str,
    ) -> dict[str, Any]:
        policy_obs: dict[str, Any] = {
            KEY_INPUT_IMAGE: self._build_input_image(
                env_obs,
                camera_key=camera_key,
                camera_mapping=camera_mapping,
            ),
            KEY_PROPRIO: self._extract_proprio(env_obs).astype(np.float32),
        }
        instruction = self.task_prompt(task_prompt)
        if self.use_prompt or self.text_embedding_cache_dir is None:
            policy_obs[KEY_PROMPT] = instruction
        else:
            context, context_mask = load_text_context_arrays(
                instruction,
                text_embedding_cache_dir=self.text_embedding_cache_dir,
                context_len=self.context_len,
            )
            policy_obs[KEY_CONTEXT] = context
            policy_obs[KEY_CONTEXT_MASK] = context_mask
        return policy_obs

    def _extract_proprio(self, env_obs: dict[str, Any]) -> np.ndarray:
        state = np.asarray(env_obs["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] >= self.proprio_output_dim:
            return state[: self.proprio_output_dim]
        padded = np.zeros(self.proprio_output_dim, dtype=np.float32)
        padded[: state.shape[0]] = state
        return padded

    def parse_policy_response(self, response: Any) -> np.ndarray:
        if isinstance(response, (list, tuple)) and len(response) >= 1:
            action_dict = response[0]
        elif isinstance(response, dict):
            action_dict = response
        else:
            raise RuntimeError(f"Unexpected policy response type: {type(response)}")
        chunk = np.asarray(action_dict[KEY_ACTION], dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        if chunk.shape[-1] != self.policy_action_output_dim:
            raise ValueError(
                f"Policy action dim {chunk.shape[-1]} != expected {self.policy_action_output_dim}"
            )
        if self.policy_action_control_start != 0 or self.policy_action_control_end != self.policy_action_output_dim:
            chunk = chunk[:, self.policy_action_control_start : self.policy_action_control_end]
        return chunk

    def rotvec_to_env_action(self, rotvec_action: np.ndarray, *, dual_arm: bool) -> np.ndarray:
        return rotvec_action_to_env_quat(rotvec_action, dual_arm=dual_arm)


class DexJoCoFastWAMEvalEnv:
    """Thin DexJoCo wrapper for synchronous FastWAM closed-loop evaluation."""

    def __init__(
        self,
        task: DexJoCoTaskConfig,
        *,
        seed: int,
        randomize: bool = False,
        randomize_dynamics: bool = False,
        render_mode: str = "rgb_array",
    ) -> None:
        from dexjoco.tasks.mappings import CONFIG_MAPPING

        self.task = task
        self.seed = seed
        self._done = False
        self._success = False
        self._last_stay_state: np.ndarray | None = None
        self._latest_obs: dict[str, Any] = {}

        config = CONFIG_MAPPING[task.env_name]()
        env_kwargs: dict[str, Any] = {}
        if task.env_name == "bimanual_unlock_ipad" and task.password is not None:
            env_kwargs["password"] = task.password

        self.env = config.get_environment(
            policy_mode=True,
            render_mode=render_mode,
            randomize=randomize,
            seed=seed,
            randomize_dynamics=randomize_dynamics,
            **env_kwargs,
        )

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def reset(self) -> dict[str, Any]:
        obs, _ = self.env.reset()
        self._done = False
        self._success = False
        self._last_stay_state = None
        self._latest_obs = copy.deepcopy(obs)
        return self._latest_obs

    def get_camera_frame(self) -> np.ndarray:
        return _safe_rgb_uint8(self._latest_obs[self.task.camera_key])

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def is_success(self) -> bool:
        return self._success

    def click_mouse_warmup(self) -> None:
        if self.task.env_name != "click_mouse":
            return
        align = rotvec_action_to_env_quat(
            CLICK_MOUSE_ALIGN_ROTVEC,
            dual_arm=self.task.dual_arm,
        )
        for _ in range(CLICK_MOUSE_ALIGN_STEPS):
            self._step_env(align)

    def step_rotvec(self, rotvec_action: np.ndarray) -> None:
        env_action = rotvec_action_to_env_quat(rotvec_action, dual_arm=self.task.dual_arm)
        self._step_env(env_action)

    def stay(self, *, continue_stay: bool = False) -> np.ndarray:
        if continue_stay and self._last_stay_state is not None:
            stay_state = self._last_stay_state
        else:
            state = np.asarray(self._latest_obs["state"], dtype=np.float64).reshape(-1)
            if self.task.dual_arm:
                stay_state = state[:46]
            else:
                stay_state = state[:23]
            self._last_stay_state = stay_state

        if self.task.dual_arm:
            r_arm = stay_state[:7]
            l_arm = stay_state[7:14]
            r_hand = stay_state[14:30]
            l_hand = stay_state[30:46]
            rotvec_action = np.concatenate(
                [
                    r_arm[:3],
                    _quat_wxyz_to_rotvec(r_arm[3:7]),
                    r_hand,
                    l_arm[:3],
                    _quat_wxyz_to_rotvec(l_arm[3:7]),
                    l_hand,
                ]
            )
        else:
            arm = stay_state[:7]
            hand = stay_state[7:23]
            rotvec_action = np.concatenate([arm[:3], _quat_wxyz_to_rotvec(arm[3:7]), hand])

        self.step_rotvec(rotvec_action)
        return rotvec_action.astype(np.float32)

    def build_policy_obs(self, adapter: DexJoCoFastWAMAdapter) -> dict[str, Any]:
        return adapter.env_obs_to_policy_obs(
            self._latest_obs,
            camera_key=self.task.camera_key,
            camera_mapping=self.task.camera_mapping,
            task_prompt=self.task.prompt,
        )

    def _step_env(self, env_action: np.ndarray) -> None:
        obs, _reward, terminated, truncated, info = self.env.step(env_action)
        self._done = bool(terminated or truncated)
        self._success = bool(info.get("succeed", False))
        self._latest_obs = copy.deepcopy(obs)
