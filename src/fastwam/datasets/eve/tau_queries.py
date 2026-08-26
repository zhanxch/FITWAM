"""Select NFE0 RMS query frames from the DEWO v6 training pool (not collect-200).

D+ = success-event primaries (recoverability windows).
D0 = success episodes in the v6 pool; query non-event prefixes at replan stride.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Sequence

from fastwam.datasets.eve.manifest_dataset import EveManifestRobotVideoDataset

Kind = Literal["plus", "zero"]


@dataclass(frozen=True)
class TauQuery:
    kind: Kind
    sample_id: str
    dataset_root: str
    episode_index: int
    frame_index: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _in_v6_pool(unit: dict[str, Any]) -> bool:
    return EveManifestRobotVideoDataset._passes_dewo_v6_pool_filter(
        EveManifestRobotVideoDataset, unit
    )


def is_v6_d_plus(unit: dict[str, Any]) -> bool:
    if str(unit.get("sample_type")) != "event":
        return False
    if not _in_v6_pool(unit):
        return False
    return EveManifestRobotVideoDataset._sampling_role(unit) == "primary"


def is_v6_d_zero(unit: dict[str, Any]) -> bool:
    if str(unit.get("sample_type")) != "episode":
        return False
    if not _in_v6_pool(unit):
        return False
    return EveManifestRobotVideoDataset._sampling_role(unit) == "primary"


def collect_v6_tau_queries(
    units: Sequence[dict[str, Any]],
    *,
    replan_steps: int = 24,
    prefix_fraction: float = 0.5,
    max_zero_per_episode: int | None = None,
) -> list[TauQuery]:
    """One query per D+ event window; replan-stride prefixes on D0 episodes."""

    if replan_steps < 1:
        raise ValueError(f"replan_steps must be >= 1, got {replan_steps}")
    if not 0.0 < float(prefix_fraction) <= 1.0:
        raise ValueError(f"prefix_fraction must be in (0, 1], got {prefix_fraction}")

    plus_units = [unit for unit in units if is_v6_d_plus(unit)]
    zero_units = [unit for unit in units if is_v6_d_zero(unit)]

    event_spans: dict[tuple[str, int], list[tuple[int, int]]] = {}
    queries: list[TauQuery] = []
    seen_plus: set[tuple[str, int, int]] = set()
    for unit in plus_units:
        root = str(unit["dataset_root"])
        episode_index = int(unit["episode_index"])
        start = int(unit.get("core_start_frame", unit.get("start_frame", 0)))
        end = int(unit.get("core_end_frame", unit.get("end_frame", start + 1)))
        event_spans.setdefault((root, episode_index), []).append((start, end))
        key = (root, episode_index, start)
        if key in seen_plus:
            continue
        seen_plus.add(key)
        queries.append(
            TauQuery(
                kind="plus",
                sample_id=str(unit.get("sample_id", "")),
                dataset_root=root,
                episode_index=episode_index,
                frame_index=start,
            )
        )

    for unit in zero_units:
        root = str(unit["dataset_root"])
        episode_index = int(unit["episode_index"])
        start = int(unit.get("start_frame", 0))
        end = int(unit.get("end_frame", start + 1))
        length = max(0, end - start)
        prefix_end = start + max(1, int(length * float(prefix_fraction)))
        prefix_end = min(prefix_end, end)
        spans = event_spans.get((root, episode_index), [])
        n_kept = 0
        frame = start
        while frame < prefix_end:
            if not _frame_in_spans(frame, spans):
                queries.append(
                    TauQuery(
                        kind="zero",
                        sample_id=str(unit.get("sample_id", "")),
                        dataset_root=root,
                        episode_index=episode_index,
                        frame_index=frame,
                    )
                )
                n_kept += 1
                if max_zero_per_episode is not None and n_kept >= int(max_zero_per_episode):
                    break
            frame += int(replan_steps)
    return queries


def shard_queries(queries: Sequence[TauQuery], *, shard_index: int, num_shards: int) -> list[TauQuery]:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    return [query for i, query in enumerate(queries) if i % num_shards == shard_index]


def _frame_in_spans(frame: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(start <= frame < end for start, end in spans)
