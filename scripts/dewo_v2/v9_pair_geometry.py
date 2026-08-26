"""Geometry helpers for DEWO v9 full-horizon pair materialization."""

from __future__ import annotations

import numpy as np

MIN_EVENT_FRAMES = 33
FAIL_CLIFF_POST = 24


def fail_cliff_span(
    t_star: int,
    m_first_zero: int,
    fail_len: int,
    *,
    min_len: int = MIN_EVENT_FRAMES,
    post: int = FAIL_CLIFF_POST,
) -> tuple[int, int]:
    """Half-open ``[lo, hi)`` on the factual fail episode.

    Prefer expanding into the timeout tail rather than back into the
    shared prefix ``0..t``.
    """

    lo = int(t_star)
    hi = min(int(fail_len), int(m_first_zero) + int(post))
    if hi < lo:
        raise ValueError(f"fail cliff inverted: t={lo} M+post={hi} len={fail_len}")
    if hi - lo < min_len:
        hi = min(int(fail_len), lo + min_len)
    if hi - lo < min_len:
        lo = max(0, hi - min_len)
    if hi - lo < min_len:
        raise ValueError(
            f"fail episode length {fail_len} cannot host {min_len} cliff frames"
        )
    return lo, hi


def stitch_prefix_plus_continuation(
    prefix: np.ndarray,
    continuation: np.ndarray,
) -> np.ndarray:
    """``prefix`` is ``[0, t)``; ``continuation`` starts at frame ``t``."""

    prefix = np.asarray(prefix)
    continuation = np.asarray(continuation)
    if prefix.ndim != continuation.ndim:
        raise ValueError(
            f"prefix/continuation rank mismatch: {prefix.shape} vs {continuation.shape}"
        )
    if prefix.shape[0] < 1:
        return continuation
    return np.concatenate([prefix, continuation], axis=0)
