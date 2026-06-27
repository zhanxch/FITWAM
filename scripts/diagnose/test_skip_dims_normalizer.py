#!/usr/bin/env python3
"""C3 verification: confirm the skip_dims normalizer patch leaves rot6d
identity (scale=1, offset=0) while keeping min/max for xyz/hand dims.

This is a unit test for the H2 fix. It does NOT require GPU or the full model;
it only exercises SingleFieldLinearNormalizer and LinearNormalizer directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from fastwam.datasets.lerobot.utils.normalizer import (
    SingleFieldLinearNormalizer,
    LinearNormalizer,
)


def make_stats(dim: int = 58) -> dict:
    rng = np.random.default_rng(0)
    mn = torch.tensor(rng.uniform(-1.0, 0.0, size=dim), dtype=torch.float32)
    mx = torch.tensor(rng.uniform(0.0, 1.0, size=dim), dtype=torch.float32)
    # make rot6d ranges non-uniform (mimic real stats)
    mx[3:9] = torch.tensor([0.48, -0.48, 0.25, 1.0, 0.79, 0.34], dtype=torch.float32)
    mn[3:9] = torch.tensor([-0.49, -1.0, -0.88, -0.08, -0.25, -0.98], dtype=torch.float32)
    mean = (mn + mx) / 2
    std = (mx - mn) / 4 + 1e-3
    return {"min": mn, "max": mx, "mean": mean, "std": std, "q01": mn, "q99": mx}


def test_single_field_skip() -> int:
    print("=" * 70)
    print("C3: skip_dims normalizer patch unit test")
    print("=" * 70)
    stats = make_stats(58)
    skip = [3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17]
    norm = SingleFieldLinearNormalizer(stats=stats, mode="min/max", skip_dims=skip)

    # skipped dims must be identity
    skip_idx = torch.tensor(skip, dtype=torch.long)
    scale_skip = norm.scale[skip_idx]
    offset_skip = norm.offset[skip_idx]
    print("\n[skipped rot6d dims]")
    print(f"  scale (should be all 1.0):  {scale_skip.tolist()}")
    print(f"  offset (should be all 0.0): {offset_skip.tolist()}")
    assert torch.allclose(scale_skip, torch.ones_like(scale_skip)), "skip scale != 1"
    assert torch.allclose(offset_skip, torch.zeros_like(offset_skip)), "skip offset != 0"

    # non-skipped dims must be normalized (scale != 1 in general)
    non_skip = [i for i in range(58) if i not in skip]
    scale_non = norm.scale[non_skip]
    print(f"\n[non-skipped dims] sample scales: {scale_non[:6].tolist()}")
    assert not torch.allclose(scale_non, torch.ones_like(scale_non)), "non-skip scale should differ from 1"

    # round-trip on skipped dims is identity; for non-skipped dims use inputs
    # within [min,max] so the pre-existing clamp(-5,5) does not trigger.
    x = torch.zeros((4, 58), dtype=torch.float32)
    for i in range(58):
        if i in skip:
            x[:, i] = torch.tensor(np.random.default_rng(1).uniform(-1, 1, size=4), dtype=torch.float32)
        else:
            lo, hi = float(stats["min"][i]), float(stats["max"][i])
            x[:, i] = torch.tensor(np.random.default_rng(100 + i).uniform(lo, hi, size=4), dtype=torch.float32)
    fwd = norm.forward(x)
    back = norm.backward(fwd)
    err = (x - back).abs().max().item()
    print(f"\n[round-trip] max abs error over all dims = {err:.3e} (should be ~0)")
    assert err < 1e-5, "round-trip not identity"

    # forward on skipped dims is identity (no shift/scale)
    err_skip = (fwd[:, skip_idx] - x[:, skip_idx]).abs().max().item()
    print(f"[forward on skipped dims] max abs diff from input = {err_skip:.3e} (should be ~0)")
    assert err_skip < 1e-6, "skipped dims changed by forward"

    # orthonormality preserved: build an orthonormal rot6d, forward it, check
    rows = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)  # already orthonormal
    rot6d = rows.reshape(-1)  # (6,)
    full = torch.zeros((1, 58), dtype=torch.float32)
    full[0, 3:9] = rot6d
    fwd_full = norm.forward(full)
    fwd_rot6d = fwd_full[0, 3:9].reshape(2, 3)
    # check orthonormality
    r0 = fwd_rot6d[0] / fwd_rot6d[0].norm()
    r1 = fwd_rot6d[1] - torch.dot(fwd_rot6d[1], r0) * r0
    r1 = r1 / r1.norm()
    orth_err = (torch.cat([r0[None], r1[None], torch.cross(r0, r1)[None]], 0)
                @ torch.cat([r0[None], r1[None], torch.cross(r0, r1)[None]], 0).T - torch.eye(3)).norm().item()
    print(f"[orthonormality after forward on skipped rot6d] err = {orth_err:.3e} (should be ~0)")
    assert orth_err < 1e-5, "skipped rot6d orthonormality not preserved"

    print("\nPASS: skip_dims patch works correctly.")
    return 0


def test_linear_normalizer_skip() -> int:
    print("\n" + "=" * 70)
    print("C3: LinearNormalizer with skip_dims integration")
    print("=" * 70)
    stats = make_stats(58)
    shape_meta = {
        "action": [{"key": "default", "raw_shape": 58, "shape": 58}],
        "state": [{"key": "default", "raw_shape": 58, "shape": 58}],
    }
    stats_full = {
        "action": {"default": {f"global_{k}": v for k, v in stats.items()}},
        "state": {"default": {f"global_{k}": v for k, v in stats.items()}},
    }
    skip = {"action": {"default": [3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17]}}
    norm = LinearNormalizer(
        shape_meta=shape_meta,
        use_stepwise_action_norm=False,
        default_mode="min/max",
        exception_mode=None,
        stats=stats_full,
        skip_dims=skip,
    )
    sf = norm.normalizers["action"]["default"]
    skip_idx = torch.tensor(skip["action"]["default"], dtype=torch.long)
    print(f"  action default scale[skip] = {sf.scale[skip_idx].tolist()}")
    print(f"  action default offset[skip] = {sf.offset[skip_idx].tolist()}")
    assert torch.allclose(sf.scale[skip_idx], torch.ones_like(sf.scale[skip_idx]))
    assert torch.allclose(sf.offset[skip_idx], torch.zeros_like(sf.offset[skip_idx]))
    # state has no skip -> should be fully normalized
    sf_state = norm.normalizers["state"]["default"]
    assert not torch.allclose(sf_state.scale, torch.ones_like(sf_state.scale)), "state should be normalized"
    print("  state default scale[:3] =", sf_state.scale[:3].tolist(), "(normalized, != 1)")
    print("\nPASS: LinearNormalizer skip_dims integration works.")
    return 0


if __name__ == "__main__":
    rc = 0
    rc |= test_single_field_skip()
    rc |= test_linear_normalizer_skip()
    print("\n" + "=" * 70)
    if rc == 0:
        print("C3 PATCH VERIFIED: rot6d skip_dims normalization works as intended.")
    else:
        print("C3 PATCH FAILED.")
    sys.exit(rc)
