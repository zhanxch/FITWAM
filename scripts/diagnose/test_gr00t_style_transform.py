#!/usr/bin/env python3
"""Unit tests for Gr00tStyleRelativeTransform and per_dim_modes normalizer.

Verifies:
  1. SE(3) relative pose round-trip (forward -> backward == identity) for rot6d.
  2. Joint relative round-trip.
  3. Gr00tStyleRelativeTransform on a 58-dim single-key batch.
  4. SingleFieldLinearNormalizer with per_dim_modes + clip_to_unit.

Run:
    PYTHONPATH=src python scripts/diagnose/test_gr00t_style_transform.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fastwam.datasets.lerobot.transforms.relative_action import (
    Gr00tStyleRelativeTransform,
    RelativePoseRot6dTransform,
    RelativeJointTransformLastFrame,
)
from fastwam.datasets.lerobot.utils.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from fastwam.datasets.lerobot.utils.rotation import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def random_pose_9d(batch_shape, device="cpu"):
    """Random xyz + rot6d pose tensor with ORTHOGONAL rot6d (valid rotation)."""
    *leading, _ = batch_shape
    pos = torch.randn(*leading, 3, device=device)
    R = torch.linalg.qr(torch.randn(*leading, 3, 3, device=device)).Q
    rot6d = matrix_to_rotation_6d(R)
    return torch.cat([pos, rot6d], dim=-1)


def test_rot6d_se3_roundtrip():
    """forward -> backward should recover the original pose."""
    torch.manual_seed(0)
    pose = random_pose_9d((4, 32, 9))
    base = random_pose_9d((4, 1, 9))
    trans = RelativePoseRot6dTransform(keys=["default"])
    rel = trans._forward(pose, base)
    rec = trans._backward(rel, base)
    err = (pose - rec).abs().max().item()
    assert err < 1e-4, f"rot6d SE(3) round-trip error {err} too large"
    print(f"[PASS] rot6d SE(3) round-trip: max err = {err:.2e}")


def test_gr00t_style_58d_roundtrip():
    """Gr00tStyleRelativeTransform on 58-dim: forward -> backward == identity.

    Uses orthogonal rot6d in EEF segments (valid rotations) since real data
    rot6d is always orthonormal; Gram-Schmidt projects non-orthogonal input
    to the nearest rotation, so only orthogonal input round-trips exactly.
    """
    torch.manual_seed(1)
    B, T = 2, 32
    action = torch.randn(B, T, 58)
    state = torch.randn(B, T + 1, 58)
    for b in range(B):
        for t in range(T):
            action[b, t, 0:9] = random_pose_9d((1, 9))[0]
            action[b, t, 9:18] = random_pose_9d((1, 9))[0]
        for t in range(T + 1):
            state[b, t, 0:9] = random_pose_9d((1, 9))[0]
            state[b, t, 9:18] = random_pose_9d((1, 9))[0]
    trans = Gr00tStyleRelativeTransform(key="default")
    batch = {"action": {"default": action}, "state": {"default": state}}
    import copy
    batch_copy = copy.deepcopy(batch)
    batch_rel = trans.forward(batch_copy)
    assert not torch.allclose(batch_rel["action"]["default"], action)
    batch_rec = trans.backward(batch_rel)
    err = (batch_rec["action"]["default"] - action).abs().max().item()
    assert err < 1e-4, f"58-dim round-trip error {err} too large"
    assert batch_rel["action"]["default"].shape[-1] == 58
    print(f"[PASS] Gr00tStyleRelativeTransform 58-dim round-trip: max err = {err:.2e}")


def test_per_dim_modes_normalizer():
    """per_dim_modes gives different modes to different dims + clip_to_unit."""
    torch.manual_seed(2)
    stats = {
        "min": torch.tensor([0.0, 0.0, 0.0, -1.0, -1.0, -1.0] + [0.0] * 52),
        "max": torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0] + [1.0] * 52),
        "mean": torch.tensor([0.5] * 58),
        "std": torch.tensor([0.25] * 58),
        "q01": torch.tensor([0.01] * 58),
        "q99": torch.tensor([0.99] * 58),
    }
    # dims 0:6 use z-score, rest use min/max (default)
    norm = SingleFieldLinearNormalizer(
        stats=stats,
        mode="min/max",
        per_dim_modes={"z-score": [0, 1, 2, 3, 4, 5]},
        clip_to_unit=True,
    )
    x = torch.tensor([[2.0] * 58])  # out-of-range value
    y = norm.forward(x)
    # z-score dims: (2 - 0.5)/0.25 = 6 -> clipped to 1
    assert y[0, 0].item() == 1.0, f"z-score dim not clipped: {y[0, 0]}"
    # min/max dims: (2 - 0)/1 * 2 - 1 = 3 -> clipped to 1
    assert y[0, 6].item() == 1.0, f"min/max dim not clipped: {y[0, 6]}"
    # backward round-trip (clipped values won't be exact, but in-range should be)
    x_in = torch.tensor([[0.5] * 58])
    y_in = norm.forward(x_in)
    x_rec = norm.backward(y_in)
    err = (x_in - x_rec).abs().max().item()
    assert err < 1e-5, f"normalizer round-trip error {err}"
    print(f"[PASS] per_dim_modes + clip_to_unit: in-range round-trip err = {err:.2e}")


def test_per_dim_modes_independent_stats():
    """Verify per_dim_modes uses the SAME stats dict but applies different scale/offset per dim."""
    stats = {
        "min": torch.tensor([0.0, 0.0, 10.0, 10.0]),
        "max": torch.tensor([1.0, 1.0, 20.0, 20.0]),
        "mean": torch.tensor([0.5, 0.5, 15.0, 15.0]),
        "std": torch.tensor([0.25, 0.25, 2.0, 2.0]),
        "q01": torch.tensor([0.01, 0.01, 10.1, 10.1]),
        "q99": torch.tensor([0.99, 0.99, 19.9, 19.9]),
    }
    # dims 0:2 -> z-score, dims 2:4 -> min/max
    norm = SingleFieldLinearNormalizer(
        stats=stats,
        mode="min/max",
        per_dim_modes={"z-score": [0, 1]},
    )
    # dim 0 (z-score): (0.5 - 0.5)/0.25 = 0
    # dim 2 (min/max): (10 - 10)/(20-10)*2-1 = -1
    x = torch.tensor([[0.5, 0.5, 10.0, 10.0]])
    y = norm.forward(x)
    assert abs(y[0, 0].item() - 0.0) < 1e-5, f"z-score dim 0 wrong: {y[0, 0]}"
    assert abs(y[0, 2].item() - (-1.0)) < 1e-5, f"min/max dim 2 wrong: {y[0, 2]}"
    print(f"[PASS] per_dim_modes independent: dim0(z-score)={y[0,0]:.3f} dim2(min/max)={y[0,2]:.3f}")


# ---- Multi-modality (modality.json alignment) tests ----

DATASET_DIR = "data/spray_water_rot6d_rosbag_ts_filter"
MODALITY_KEYS = ("left_eef", "right_eef", "left_hand_joints", "right_hand_joints")


def _multi_key_shape_meta():
    return {
        "action": [
            {"key": "left_eef", "raw_shape": 9, "shape": 9},
            {"key": "right_eef", "raw_shape": 9, "shape": 9},
            {"key": "left_hand_joints", "raw_shape": 20, "shape": 20},
            {"key": "right_hand_joints", "raw_shape": 20, "shape": 20},
        ],
        "state": [
            {"key": "left_eef", "raw_shape": 9, "shape": 9},
            {"key": "right_eef", "raw_shape": 9, "shape": 9},
            {"key": "left_hand_joints", "raw_shape": 20, "shape": 20},
            {"key": "right_hand_joints", "raw_shape": 20, "shape": 20},
        ],
    }


def test_multi_key_modality_roundtrip():
    """Per-key transforms (RelativePoseRot6dTransform + RelativeJointTransformLastFrame)
    forward -> backward round-trip on a 4-key batch matches the original per-key action.
    """
    import json as _json
    from pathlib import Path as _Path
    if not _Path(DATASET_DIR).exists():
        print("[SKIP] multi-key round-trip (dataset dir absent)")
        return
    torch.manual_seed(3)
    shape_meta = _multi_key_shape_meta()
    with open(f"{DATASET_DIR}/meta/modality.json") as f:
        mm = _json.load(f)

    # Build in-range absolute action/state using the dataset's absolute stats.
    with open(f"{DATASET_DIR}/meta/stats.json") as f:
        meta_stats = _json.load(f)
    a_min = torch.tensor(meta_stats["action"]["min"])
    a_max = torch.tensor(meta_stats["action"]["max"])
    s_min = torch.tensor(meta_stats["observation.state"]["min"])
    s_max = torch.tensor(meta_stats["observation.state"]["max"])

    def in_range(lo, hi, shape):
        # Midpoint + small noise -> stays in [lo, hi].
        mid = (lo + hi) / 2
        span = (hi - lo) / 4  # keep within quarter-span
        return mid + (torch.rand(shape) - 0.5) * span

    T_act, T_obs = 32, 33
    action_flat = in_range(a_min, a_max, (T_act, 58))
    state_flat = in_range(s_min, s_max, (T_obs, 58))
    # Force EEF rot6d to be orthogonal for a clean SE(3) round-trip.
    for t in range(T_act):
        action_flat[t, 3:9] = matrix_to_rotation_6d(torch.linalg.qr(torch.randn(3, 3)).Q)
        action_flat[t, 12:18] = matrix_to_rotation_6d(torch.linalg.qr(torch.randn(3, 3)).Q)
    for t in range(T_obs):
        state_flat[t, 3:9] = matrix_to_rotation_6d(torch.linalg.qr(torch.randn(3, 3)).Q)
        state_flat[t, 12:18] = matrix_to_rotation_6d(torch.linalg.qr(torch.randn(3, 3)).Q)

    def split(flat, modality_type):
        out = {}
        for k in MODALITY_KEYS:
            s, e = mm[modality_type][k]["start"], mm[modality_type][k]["end"]
            out[k] = flat[..., s:e].clone()
        return out

    batch = {"action": split(action_flat, "action"), "state": split(state_flat, "state")}
    transforms = [
        RelativePoseRot6dTransform(keys=["left_eef", "right_eef"]),
        RelativeJointTransformLastFrame(keys=["left_hand_joints", "right_hand_joints"]),
    ]
    import copy
    fwd = copy.deepcopy(batch)
    for tr in transforms:
        fwd = tr.forward(fwd)
    rec = copy.deepcopy(fwd)
    for tr in reversed(transforms):
        rec = tr.backward(rec)
    for k in MODALITY_KEYS:
        err = (rec["action"][k] - batch["action"][k]).abs().max().item()
        assert err < 1e-4, f"multi-key round-trip key {k} err {err}"
    print(f"[PASS] multi-key per-key transforms round-trip: max err = "
          f"{max((rec['action'][k]-batch['action'][k]).abs().max().item() for k in MODALITY_KEYS):.2e}")


def test_from_modality_stats_matches_meta():
    """LinearNormalizer.from_modality_stats slices stats.json per modality key and
    the absolute (non-relative) key stats match meta/stats.json slices exactly.
    """
    import json as _json
    from pathlib import Path as _Path
    if not _Path(DATASET_DIR).exists():
        print("[SKIP] from_modality_stats (dataset dir absent)")
        return
    shape_meta = _multi_key_shape_meta()
    with open(f"{DATASET_DIR}/meta/modality.json") as f:
        mm = _json.load(f)
    norm = LinearNormalizer.from_modality_stats(
        shape_meta=shape_meta, modality_meta=mm,
        stats_json_path=f"{DATASET_DIR}/meta/stats.json",
        relative_stats_json_path=f"{DATASET_DIR}/meta/relative_stats.json",
        default_mode="min/max", clip_to_unit=True,
        relative_action_keys=["left_eef", "right_eef"],
    )
    with open(f"{DATASET_DIR}/meta/stats.json") as f:
        meta_stats = _json.load(f)

    # Absolute state keys must match meta/stats.json slices exactly.
    for k in ("left_eef", "right_eef", "left_hand_joints", "right_hand_joints"):
        s, e = mm["state"][k]["start"], mm["state"][k]["end"]
        st = norm.normalizers["state"][k].get_stats()
        err_min = (st["min"] - torch.tensor(meta_stats["observation.state"]["min"][s:e])).abs().max().item()
        err_max = (st["max"] - torch.tensor(meta_stats["observation.state"]["max"][s:e])).abs().max().item()
        assert err_min < 1e-5, f"state {k} min mismatch {err_min}"
        assert err_max < 1e-5, f"state {k} max mismatch {err_max}"

    # Absolute hand-joint action keys match meta/stats.json slices.
    for k in ("left_hand_joints", "right_hand_joints"):
        s, e = mm["action"][k]["start"], mm["action"][k]["end"]
        st = norm.normalizers["action"][k].get_stats()
        err_min = (st["min"] - torch.tensor(meta_stats["action"]["min"][s:e])).abs().max().item()
        err_max = (st["max"] - torch.tensor(meta_stats["action"]["max"][s:e])).abs().max().item()
        assert err_min < 1e-5, f"action {k} min mismatch {err_min}"
        assert err_max < 1e-5, f"action {k} max mismatch {err_max}"

    # Relative EEF action keys use relative_stats.json (flattened over time).
    with open(f"{DATASET_DIR}/meta/relative_stats.json") as f:
        rel_stats = _json.load(f)
    for k in ("left_eef", "right_eef"):
        st = norm.normalizers["action"][k].get_stats()
        gmin = torch.tensor(rel_stats[k]["min"]).min(0).values
        gmax = torch.tensor(rel_stats[k]["max"]).max(0).values
        err_min = (st["min"] - gmin).abs().max().item()
        err_max = (st["max"] - gmax).abs().max().item()
        assert err_min < 1e-5, f"relative action {k} min mismatch {err_min}"
        assert err_max < 1e-5, f"relative action {k} max mismatch {err_max}"
    print("[PASS] from_modality_stats: state + hand-joint action == meta/stats.json slices, "
          "EEF action == relative_stats.json (time-flattened)")


if __name__ == "__main__":
    test_rot6d_se3_roundtrip()
    test_gr00t_style_58d_roundtrip()
    test_per_dim_modes_normalizer()
    test_per_dim_modes_independent_stats()
    test_multi_key_modality_roundtrip()
    test_from_modality_stats_matches_meta()
    print("\nAll GR00T-style transform tests PASSED.")
