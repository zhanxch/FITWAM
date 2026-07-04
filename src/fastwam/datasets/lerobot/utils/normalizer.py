from typing import Literal, Dict, Annotated, Union, Any, List, Tuple, Optional
import os
import torch
import json
from collections import defaultdict
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hashlib
from pathlib import Path
from git import Repo
from fastwam.utils.logging_config import get_logger

from fastwam.utils.pytorch_utils import dict_apply

logger = get_logger(__name__)

ConstConstStr = Annotated[str, "format: 'const_min/const_max', where const_min and const_max give the constant range"]
NormMode = Union[Literal["min/max", "q01/q99", "z-score"], ConstConstStr]

class LinearNormalizer:
    def __init__(
            self, 
            shape_meta, 
            use_stepwise_action_norm,
            default_mode: NormMode, 
            exception_mode: Dict[str, Dict[str, NormMode]], 
            stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
            skip_dims: Dict[str, Dict[str, List[int]]] | None = None,
            per_dim_modes: Dict[str, Dict[str, Dict[str, List[int]]]] | None = None,
            clip_to_unit: bool = False,
        ):
        super().__init__()
        self.normalizers = {"action": {}, "state": {}}
        self.stats = stats
        skip_dims = skip_dims or {}
        per_dim_modes = per_dim_modes or {}

        for meta in shape_meta["action"]:
            key = meta["key"]
            
            if use_stepwise_action_norm:
                cur_stats = {k.removeprefix("stepwise_"): v for k, v in stats["action"][key].items() if k.startswith("stepwise_")}
            else:
                cur_stats = {k.removeprefix("global_"): v for k, v in stats["action"][key].items() if k.startswith("global_")}

            if exception_mode is not None and "action" in exception_mode and key in exception_mode["action"]:
                cur_mode = exception_mode["action"][key]
            else:
                cur_mode = default_mode

            cur_skip = skip_dims.get("action", {}).get(key)
            cur_per_dim = per_dim_modes.get("action", {}).get(key)

            self.normalizers["action"][key] = SingleFieldLinearNormalizer(
                stats=cur_stats, 
                mode=cur_mode,
                skip_dims=cur_skip,
                per_dim_modes=cur_per_dim,
                clip_to_unit=clip_to_unit,
            )

        for meta in shape_meta["state"]:
            key = meta["key"]
            cur_stats = {k.removeprefix("global_"): v for k, v in stats["state"][key].items() if k.startswith("global_")}

            if exception_mode is not None and "state" in exception_mode and key in exception_mode["state"]:
                cur_mode = exception_mode["state"][key]
            else:
                cur_mode = default_mode

            cur_skip = skip_dims.get("state", {}).get(key)
            cur_per_dim = per_dim_modes.get("state", {}).get(key)

            self.normalizers["state"][key] = SingleFieldLinearNormalizer(
                stats=cur_stats, 
                mode=cur_mode,
                skip_dims=cur_skip,
                per_dim_modes=cur_per_dim,
                clip_to_unit=clip_to_unit,
            )

    def get_stats(self):
        stats = {
            "action": {key: norm.get_stats() for key, norm in self.normalizers["action"].items()},
            "state": {key: norm.get_stats() for key, norm in self.normalizers["state"].items()}
        }
        return stats

    @classmethod
    def from_modality_stats(
        cls,
        shape_meta,
        modality_meta,
        stats_json_path,
        relative_stats_json_path=None,
        use_stepwise_action_norm=False,
        default_mode="min/max",
        exception_mode=None,
        skip_dims=None,
        per_dim_modes=None,
        clip_to_unit=False,
        relative_action_keys=None,
    ):
        """Build a LinearNormalizer from GR00T-style meta/stats.json + modality.json.

        Args:
            shape_meta: FastWAM shape_meta with per-key action/state entries.
            modality_meta: contents of meta/modality.json ({state/action: {key: {start, end}}}).
            stats_json_path: path to meta/stats.json (flat {action: {min:[58], ...}, observation.state: {...}}).
            relative_stats_json_path: optional path to meta/relative_stats.json
                (per-key stepwise {key: {min:[T,D], ...}}). Used for action keys listed in
                `relative_action_keys`; flattened over the time axis to produce global stats.
            relative_action_keys: action keys whose stats should come from relative_stats
                (e.g. ["left_eef", "right_eef"] when those use SE(3) relative transforms).
        """
        import json as _json
        from pathlib import Path as _Path

        with open(stats_json_path, "r") as f:
            meta_stats = _json.load(f)

        rel_stats = {}
        if relative_stats_json_path and _Path(relative_stats_json_path).exists():
            with open(relative_stats_json_path, "r") as f:
                rel_stats = _json.load(f)

        relative_action_keys = set(relative_action_keys or [])

        # Map FastWAM modality type -> GR00T parquet column name.
        col_map = {"action": "action", "state": "observation.state"}

        # Build FastWAM-format stats dict: {action/state: {key: {global_min, global_max, ...}}}
        fw_stats = {"action": {}, "state": {}}
        for modality in ("action", "state"):
            col = col_map[modality]
            col_stats = meta_stats[col]
            mod_slices = modality_meta.get(modality, {})
            for meta in shape_meta[modality]:
                key = meta["key"]
                if key not in mod_slices:
                    # Fall back to treating the key as the full column (single-key mode).
                    s, e = 0, meta["raw_shape"]
                else:
                    s, e = mod_slices[key]["start"], mod_slices[key]["end"]

                # For relative action keys, use relative_stats (flatten time axis to global).
                if modality == "action" and key in relative_action_keys and key in rel_stats:
                    rs = rel_stats[key]
                    # rs[stat] has shape (T, D); flatten to global (D,).
                    g_min = torch.as_tensor(rs["min"], dtype=torch.float32).min(0).values
                    g_max = torch.as_tensor(rs["max"], dtype=torch.float32).max(0).values
                    g_mean_t = torch.as_tensor(rs["mean"], dtype=torch.float32)
                    g_std_t = torch.as_tensor(rs["std"], dtype=torch.float32)
                    g_mean = g_mean_t.mean(0)
                    g_std = torch.sqrt(
                        (g_std_t ** 2 + (g_mean_t - g_mean) ** 2).mean(0)
                    )
                    # q01/q99 not in relative_stats; derive from min/max as a safe fallback.
                    g_q01 = g_min.clone()
                    g_q99 = g_max.clone()
                else:
                    g_min = torch.as_tensor(col_stats["min"][s:e], dtype=torch.float32)
                    g_max = torch.as_tensor(col_stats["max"][s:e], dtype=torch.float32)
                    g_mean = torch.as_tensor(col_stats["mean"][s:e], dtype=torch.float32)
                    g_std = torch.as_tensor(col_stats["std"][s:e], dtype=torch.float32)
                    # q01/q99 may be absent in some stats.json; fall back to min/max.
                    if "q01" in col_stats and "q99" in col_stats:
                        g_q01 = torch.as_tensor(col_stats["q01"][s:e], dtype=torch.float32)
                        g_q99 = torch.as_tensor(col_stats["q99"][s:e], dtype=torch.float32)
                    else:
                        g_q01 = g_min.clone()
                        g_q99 = g_max.clone()

                fw_stats[modality][key] = {
                    "global_min": g_min,
                    "global_max": g_max,
                    "global_mean": g_mean,
                    "global_std": g_std,
                    "global_q01": g_q01,
                    "global_q99": g_q99,
                    # stepwise_* mirror global_* for compatibility (not used when
                    # use_stepwise_action_norm=False).
                    "stepwise_min": g_min,
                    "stepwise_max": g_max,
                    "stepwise_mean": g_mean,
                    "stepwise_std": g_std,
                    "stepwise_q01": g_q01,
                    "stepwise_q99": g_q99,
                }

        fw_stats["num_episodes"] = 0
        fw_stats["num_transition"] = 0
        return cls(
            shape_meta=shape_meta,
            use_stepwise_action_norm=use_stepwise_action_norm,
            default_mode=default_mode,
            exception_mode=exception_mode,
            stats=fw_stats,
            skip_dims=skip_dims,
            per_dim_modes=per_dim_modes,
            clip_to_unit=clip_to_unit,
        )

                
    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if "action" in batch:
            for key, norm in self.normalizers["action"].items():
                batch["action"][key] = norm.forward(batch["action"][key])

        for key, norm in self.normalizers["state"].items():
            batch["state"][key] = norm.forward(batch["state"][key])

        return batch
    
    def backward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        for key, norm in self.normalizers["action"].items():
            batch["action"][key] = norm.backward(batch["action"][key])

        for key, norm in self.normalizers["state"].items():
            batch["state"][key] = norm.backward(batch["state"][key])
        
        return batch


class SingleFieldLinearNormalizer:
    std_reg = 1e-8
    range_tol = 1e-4
    output_max = 1.0
    output_min = -1.0
    def __init__(self, stats, mode: NormMode="min/max", skip_dims: List[int] | None = None,
                 per_dim_modes: Dict[str, List[int]] | None = None,
                 clip_to_unit: bool = False):
        """Per-field linear normalizer.

        Args:
            stats: dict with min/max/mean/std/q01/q99 tensors.
            mode: normalization mode for dims NOT covered by skip_dims or
                per_dim_modes.
            skip_dims: optional list of dimension indices that should NOT be
                normalized (identity: scale=1, offset=0). Used for rot6d
                components whose SO(3) geometry is destroyed by per-dim
                min/max scaling (see H2 in the sim-vs-real analysis). Skipped
                dims are expected to already live in a roughly [-1,1] range
                (rot6d entries are in [-1,1] by construction).
            per_dim_modes: optional mapping {NormMode: [dim_indices]} giving
                different normalization modes to different dimension segments
                of a single key. Used for GR00T-style per-modality
                normalization on a merged action vector (e.g. EEF dims use
                min/max, hand-joint dims use min/max with independent stats
                computed only over those dims). Takes precedence over `mode`
                for the listed dims; `skip_dims` takes precedence over both.
            clip_to_unit: if True, forward() clips to [-1, 1] (matching GR00T's
                normalize_values_minmax + clip_outliers behavior) instead of
                the default [-5, 5] safety clamp.
        """
        self.stats = stats
        self.mode = mode
        self.clip_to_unit = clip_to_unit

        # Start from the global `mode` for all dims, then override per-segment.
        scale, offset = self._compute_scale_offset(stats, mode)

        # per_dim_modes: override specific dims with their own mode.
        if per_dim_modes:
            for dim_mode, dim_indices in per_dim_modes.items():
                idx = torch.as_tensor(dim_indices, dtype=torch.long)
                seg_scale, seg_offset = self._compute_scale_offset(stats, dim_mode)
                scale[idx] = seg_scale[idx]
                offset[idx] = seg_offset[idx]
        self.per_dim_modes = per_dim_modes

        # H2 fix: identity (no normalization) for selected dims (e.g. rot6d).
        if skip_dims:
            skip_idx = torch.as_tensor(skip_dims, dtype=torch.long)
            scale[skip_idx] = 1.0
            offset[skip_idx] = 0.0
        self.skip_dims = skip_dims

        self.scale = scale
        self.offset = offset

    @classmethod
    def _compute_scale_offset(cls, stats, mode: NormMode):
        """Compute per-dim scale/offset for the given mode over all dims."""
        if mode == "z-score":
            input_mean, input_std = stats["mean"], stats["std"]
            scale = 1.0 / (input_std + cls.std_reg)
            offset = -input_mean / (input_std + cls.std_reg)
        else:
            if mode == "min/max":
                input_min, input_max = stats["min"], stats["max"]
            elif mode == "q01/q99":
                input_min, input_max = stats["q01"], stats["q99"]
            else:
                input_min, input_max = map(float, mode.split("/"))
                input_min = torch.full_like(stats["min"], input_min)
                input_max = torch.full_like(stats["max"], input_max)

            input_range = input_max - input_min
            ignore_dim = input_range < cls.range_tol
            input_range[ignore_dim] = cls.output_max - cls.output_min
            scale = (cls.output_max - cls.output_min) / input_range
            offset = cls.output_min - scale * input_min
            offset[ignore_dim] = (cls.output_max + cls.output_min) / 2 - input_min[ignore_dim]
        return scale, offset

    def get_stats(self):
        return self.stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale + self.offset
        if self.clip_to_unit:
            x = torch.clamp(x, self.output_min, self.output_max)
        else:
            x = torch.clamp(x, -5.0, 5.0)
        return x
    def backward(self, x: torch.Tensor) -> torch.Tensor:
        if self.clip_to_unit:
            x = torch.clamp(x, self.output_min, self.output_max)
        x = (x - self.offset) / self.scale
        return x

def save_dataset_stats_to_json(dataset_stats: dict, file_path: str):

    def convert_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (defaultdict, dict)):
            return {k: convert_tensor(v) for k, v in dict(obj).items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_tensor(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    serializable_stats = convert_tensor(dataset_stats)

    dir_name = os.path.dirname(os.path.abspath(file_path)) or "."
    os.makedirs(dir_name, exist_ok=True)
    tmp_path = f"{file_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(serializable_stats, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, file_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def load_dataset_stats_from_json(file_path: str, 
                                 try_convert_tensor: bool = True) -> Dict[str, Any]:

    def is_numeric_list(obj):
        if isinstance(obj, list):
            if not obj:
                return True  
            first = obj[0]
            if isinstance(first, (int, float)):
                return all(isinstance(x, (int, float)) for x in obj)
            elif isinstance(first, list):
                return all(is_numeric_list(item) for item in obj)
            else:
                return False
        return False

    def convert_back_to_tensor(obj):
        if isinstance(obj, dict):
            return {k: convert_back_to_tensor(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            if is_numeric_list(obj):
                try:
                    arr = np.array(obj)
                    return torch.from_numpy(arr)
                except Exception:
                    return [convert_back_to_tensor(item) for item in obj]
            else:
                return [convert_back_to_tensor(item) for item in obj]
        else:
            return obj

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if try_convert_tensor:
        data = convert_back_to_tensor(data)

    data = dict_apply(
        data,
        lambda x: x.to(torch.float32) if isinstance(x, torch.Tensor) else x,
    )

    return data


def search_dataset_stats_cache_json(cache_dir: str | Path, data_config: DictConfig) -> Tuple[bool, str | None]:
    if isinstance(cache_dir, str):
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def get_git_hash() -> Optional[str]:
        repo = Repo(__file__, search_parent_directories=True)
        return repo.head.commit.hexsha

    def to_plain(value: Any) -> Any:
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return value

    def normalize_str_list(value: Any) -> List[str]:
        value = to_plain(value)
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        return [str(item) for item in value if item is not None]

    def normalize_transforms(value: Any) -> Any:
        value = to_plain(value)
        if isinstance(value, dict):
            return [value]
        return value

    def normalize_dataset_dirs(cfg: DictConfig) -> Any:
        dataset_cfg = cfg.get("dataset")
        if dataset_cfg is None:
            return None
        embodiment_datasets = dataset_cfg.get("embodiment_datasets")
        if embodiment_datasets is not None:
            emb_dirs: Dict[str, List[str]] = {}
            for emb, emb_cfg in embodiment_datasets.items():
                dataset_groups = emb_cfg.get("dataset_groups")
                if dataset_groups is None:
                    emb_dirs[emb] = []
                    continue
                dirs: List[str] = []
                for group in dataset_groups:
                    group_dirs = group.get("dataset_dirs")
                    if group_dirs is None:
                        continue
                    dirs.extend(normalize_str_list(group_dirs))
                emb_dirs[emb] = sorted(dirs)
            return emb_dirs

        dataset_dirs = dataset_cfg.get("dataset_dirs")
        return sorted(normalize_str_list(dataset_dirs))

    def normalize_action_state_transforms(cfg: DictConfig) -> Any:
        processor_cfg = cfg.get("processor")
        if processor_cfg is None:
            return None
        embodiment_processors = processor_cfg.get("embodiment_processors")
        if embodiment_processors is not None:
            emb_transforms: Dict[str, Any] = {}
            for emb, emb_cfg in embodiment_processors.items():
                transforms = emb_cfg.get("action_state_transforms")
                emb_transforms[emb] = normalize_transforms(transforms)
            return emb_transforms

        transforms = processor_cfg.get("action_state_transforms")
        return normalize_transforms(transforms)

    signature = {
        "action_size": data_config.dataset.action_size, 
        "dataset_dirs": normalize_dataset_dirs(data_config),
        "action_state_transforms": normalize_action_state_transforms(data_config),
    }
    signature_json = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    dataset_hash = hashlib.sha256(signature_json.encode("utf-8"), usedforsecurity=False).hexdigest()

    git_hash = get_git_hash()
    precise_name = f"dataset_stats_{dataset_hash}_{git_hash}.json"
    precise = cache_dir / precise_name
    if precise.exists():
        logger.info(f"Found dataset stats cache with precisely matching dataset and git hash: {precise_name}.")
        return True, str(precise)
    
    candidates = sorted(cache_dir.glob(f"dataset_stats_{dataset_hash}_*.json"))
    if not candidates:
        logger.info(f"No dataset stats cache found for dataset hash {dataset_hash}")
        return False, str(precise) # return precise cache path for saving cache

    picked = candidates[0]
    prefix = f"dataset_stats_{dataset_hash}_"
    picked_git_hash = picked.name[len(prefix):-5]
    assert picked_git_hash != git_hash
    logger.warning(f"Found substitute dataset stats cache {picked.name} which mismatch current git hash {git_hash}.")
    return True, str(picked)
