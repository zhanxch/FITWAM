#!/usr/bin/env python3
"""Promote observed schema-v2 steer recordings into shuffled replay caches."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, Sequence


FORMAT = "FastWAMShuffledSteerCache"
FORMAT_VERSION = "2.0"
CACHE_SCHEMA_VERSION = 2
PROTOCOL_SCHEMA_VERSION = 1
COVERAGE_OBSERVED = "observed_contiguous"
COVERAGE_FULL = "full_horizon"
HEADER_KEYS = (
    "type",
    "schema_version",
    "checkpoint_sha256",
    "config_sha256",
    "embedding_dim",
    "protocol",
    "protocol_sha256",
    "coverage_policy",
)


@dataclass(frozen=True)
class CacheEntry:
    shard_index: int
    entry_index: int
    episode: int
    request: int
    embedding: tuple[float, ...]
    embedding_sha256: str

    @property
    def identity(self) -> tuple[int, int]:
        return self.shard_index, self.entry_index


@dataclass(frozen=True)
class CacheShard:
    path: Path
    file_sha256: str
    source_header: dict[str, Any]
    replay_header: dict[str, Any]
    source_footer: dict[str, Any]
    replay_footer: dict[str, Any]
    protocol: dict[str, Any]
    protocol_sha256: str
    request_counts: dict[str, int]
    source_keyset_sha256: str
    replay_keyset_sha256: str
    entries: tuple[CacheEntry, ...]


@dataclass(frozen=True)
class EpisodeTrajectory:
    shard_index: int
    episode: int
    entries: tuple[CacheEntry, ...]

    @property
    def observed_count(self) -> int:
        return len(self.entries)

    @property
    def last(self) -> CacheEntry:
        return self.entries[-1]


@dataclass(frozen=True)
class RequestAssignment:
    shard_index: int
    target_episode: int
    target_request: int
    donor_episode: int
    donor_entry: CacheEntry
    repeat_last: bool


def _canonical_json_bytes(payload: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": True,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
        return (json.dumps(payload, **options) + "\n").encode("utf-8")
    options["separators"] = (",", ":")
    return json.dumps(payload, **options).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(payload))


def _normalize_sha256(value: Any, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA256")
    return normalized


def _require_int(container: Mapping[str, Any], key: str, *, label: str) -> int:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}.{key} must be an integer")
    return value


def _normalize_protocol(protocol: Any, *, label: str) -> tuple[dict[str, Any], str]:
    """Mirror the strict protocol checks used by the schema-v2 replay loader."""

    if not isinstance(protocol, dict):
        raise ValueError(f"{label} must be a JSON object")
    canonical = json.loads(_canonical_json_bytes(protocol))
    if canonical.get("schema") != "fastwam.steer_protocol":
        raise ValueError(f"{label}.schema must be 'fastwam.steer_protocol'")
    if canonical.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(
            f"{label}.schema_version must be {PROTOCOL_SCHEMA_VERSION}"
        )
    task = canonical.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError(f"{label}.task must be a non-empty string")

    named_objects: dict[str, dict[str, Any]] = {}
    for name in (
        "environment_seeds",
        "episodes",
        "inference",
        "environment_options",
        "model",
    ):
        value = canonical.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"{label}.{name} must be an object")
        named_objects[name] = value
    seeds = named_objects["environment_seeds"]
    episodes = named_objects["episodes"]
    inference = named_objects["inference"]
    options = named_objects["environment_options"]
    model = named_objects["model"]

    for container, keys, object_name in (
        (
            seeds,
            (
                "global_base",
                "global_end_exclusive",
                "shard_base",
                "shard_end_exclusive",
            ),
            "environment_seeds",
        ),
        (
            episodes,
            (
                "global_start",
                "global_end_exclusive",
                "shard_global_start",
                "shard_global_end_exclusive",
                "local_start",
                "local_end_exclusive",
                "shard_id",
            ),
            "episodes",
        ),
        (
            inference,
            ("replan_steps", "max_env_steps", "max_requests_per_episode"),
            "inference",
        ),
    ):
        for key in keys:
            _require_int(container, key, label=f"{label}.{object_name}")

    inference_seed = inference.get("seed")
    if inference_seed is not None and (
        isinstance(inference_seed, bool) or not isinstance(inference_seed, int)
    ):
        raise ValueError(f"{label}.inference.seed must be an integer or null")
    if inference.get("control_mode") != "blocking":
        raise ValueError(f"{label}.inference.control_mode must be 'blocking'")
    if inference.get("async_fallback") not in ("wait", "hold_last"):
        raise ValueError(f"{label}.inference.async_fallback is invalid")
    for key in ("action_horizon_override", "num_inference_steps_override"):
        value = inference.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{label}.inference.{key} must be positive or null")

    if episodes["global_start"] != 0 or episodes["local_start"] != 0:
        raise ValueError(f"{label} global/local episode ranges must start at zero")
    local_count = episodes["local_end_exclusive"]
    shard_count = (
        episodes["shard_global_end_exclusive"]
        - episodes["shard_global_start"]
    )
    seed_count = seeds["shard_end_exclusive"] - seeds["shard_base"]
    if local_count <= 0 or local_count != shard_count or local_count != seed_count:
        raise ValueError(f"{label} shard episode/seed ranges have inconsistent sizes")
    if (
        seeds["global_end_exclusive"] - seeds["global_base"]
        != episodes["global_end_exclusive"]
    ):
        raise ValueError(f"{label} global seed/episode ranges have inconsistent sizes")
    if (
        seeds["shard_base"]
        != seeds["global_base"] + episodes["shard_global_start"]
    ):
        raise ValueError(f"{label} shard seed base does not match episode offset")
    if inference["replan_steps"] <= 0 or inference["max_env_steps"] <= 0:
        raise ValueError(f"{label} replan_steps/max_env_steps must be positive")
    expected_requests = math.ceil(
        inference["max_env_steps"] / inference["replan_steps"]
    )
    if inference["max_requests_per_episode"] != expected_requests:
        raise ValueError(
            f"{label}.inference.max_requests_per_episode must equal "
            "ceil(max_env_steps / replan_steps)"
        )
    for key in ("randomize", "randomize_dynamics", "action_clip"):
        if not isinstance(options.get(key), bool):
            raise ValueError(f"{label}.environment_options.{key} must be boolean")
    for key in ("clip_max_xyz_step", "clip_max_dz_down"):
        value = options.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{label}.environment_options.{key} must be finite")
    if not isinstance(options.get("task_config_dir"), str) or not options[
        "task_config_dir"
    ]:
        raise ValueError(f"{label}.environment_options.task_config_dir is required")
    for key in ("checkpoint_path", "config_path"):
        if not isinstance(model.get(key), str) or not model[key]:
            raise ValueError(f"{label}.model.{key} must be a non-empty path")
    for key in ("checkpoint_sha256", "config_sha256"):
        normalized = _normalize_sha256(
            model.get(key), label=f"{label}.model.{key}"
        )
        if model[key] != normalized:
            raise ValueError(f"{label}.model.{key} must use canonical lowercase hex")
    return canonical, _json_sha256(canonical)


def _float32_bytes(values: Sequence[float]) -> bytes:
    try:
        return struct.pack(f"={len(values)}f", *values)
    except (OverflowError, struct.error) as error:
        raise ValueError("embedding values must be representable as float32") from error


def _embedding_sha256(values: Sequence[float]) -> str:
    return _sha256_bytes(_float32_bytes(values))


def _parse_embedding(value: Any, *, dim: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != dim:
        raise ValueError(f"{label} must be a list with length {dim}")
    embedding: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label}[{index}] must be finite")
        embedding.append(number)
    _float32_bytes(embedding)
    return tuple(embedding)


def _cache_keyset_sha256(keys: set[tuple[int, int]]) -> str:
    return _json_sha256(
        [
            {"episode": episode, "request": request}
            for episode, request in sorted(keys)
        ]
    )


def _request_counts_and_expected_keys(
    keys: set[tuple[int, int]],
    *,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, int], set[tuple[int, int]]]:
    episodes = protocol["episodes"]
    inference = protocol["inference"]
    local_ids = range(episodes["local_start"], episodes["local_end_exclusive"])
    valid_episode_ids = set(local_ids)
    unexpected = sorted({episode for episode, _ in keys} - valid_episode_ids)
    if unexpected:
        raise ValueError(f"Cache contains out-of-range local episodes {unexpected}")
    request_counts: dict[str, int] = {}
    expected_keys: set[tuple[int, int]] = set()
    for episode in local_ids:
        requests = sorted(request for ep, request in keys if ep == episode)
        if not requests:
            raise ValueError(f"Cache is missing local episode {episode}")
        if requests != list(range(requests[-1] + 1)):
            raise ValueError(f"Cache episode {episode} has non-contiguous requests")
        if len(requests) > inference["max_requests_per_episode"]:
            raise ValueError(
                f"Cache episode {episode} exceeds protocol max_requests_per_episode"
            )
        request_counts[str(episode)] = len(requests)
        expected_keys.update(
            (episode, request)
            for request in range(inference["max_requests_per_episode"])
        )
    return request_counts, expected_keys


def _load_json_line(path: Path, line_number: int, line: str) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path}:{line_number} must contain a JSON object")
    return payload


def _load_cache(path: Path, *, shard_index: int) -> CacheShard:
    if not path.is_file():
        raise ValueError(f"Input cache does not exist or is not a file: {path}")
    with path.open("r", encoding="utf-8") as stream:
        lines = [
            (line_number, line)
            for line_number, line in enumerate(stream, 1)
            if line.strip()
        ]
    if len(lines) < 3:
        raise ValueError(f"Cache must contain header, entries, and footer: {path}")
    header = _load_json_line(path, *lines[0])
    footer = _load_json_line(path, *lines[-1])
    missing = [key for key in HEADER_KEYS if key not in header]
    if missing:
        raise ValueError(f"Cache header is missing required keys {missing}: {path}")
    if header["type"] != "header" or header["schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError(f"Cache must use a schema-v2 header: {path}")
    if header["coverage_policy"] != COVERAGE_OBSERVED:
        raise ValueError(
            f"Input cache coverage_policy must be {COVERAGE_OBSERVED!r}: {path}"
        )
    embedding_dim = header["embedding_dim"]
    if (
        isinstance(embedding_dim, bool)
        or not isinstance(embedding_dim, int)
        or embedding_dim <= 0
    ):
        raise ValueError(f"Cache embedding_dim must be a positive integer: {path}")
    checkpoint_sha256 = _normalize_sha256(
        header["checkpoint_sha256"], label=f"{path} checkpoint_sha256"
    )
    config_sha256 = _normalize_sha256(
        header["config_sha256"], label=f"{path} config_sha256"
    )
    if header["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(f"Cache checkpoint_sha256 must be canonical: {path}")
    if header["config_sha256"] != config_sha256:
        raise ValueError(f"Cache config_sha256 must be canonical: {path}")
    protocol, protocol_sha256 = _normalize_protocol(
        header["protocol"], label=f"{path} protocol"
    )
    if header["protocol"] != protocol:
        raise ValueError(f"Cache protocol is not canonical: {path}")
    if header["protocol_sha256"] != protocol_sha256:
        raise ValueError(f"Cache protocol_sha256 mismatch: {path}")
    if protocol["model"]["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(f"Protocol/checkpoint header hash mismatch: {path}")
    if protocol["model"]["config_sha256"] != config_sha256:
        raise ValueError(f"Protocol/config header hash mismatch: {path}")

    if footer.get("type") != "footer":
        raise ValueError(f"Cache is incomplete: missing completion footer: {path}")
    if footer.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"Cache footer must use schema version 2: {path}")
    if footer.get("complete") is not True:
        raise ValueError(f"Cache completion footer must have complete=true: {path}")
    if footer.get("coverage_policy") != COVERAGE_OBSERVED:
        raise ValueError(f"Cache footer coverage_policy mismatch: {path}")
    if footer.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"Cache footer protocol_sha256 mismatch: {path}")
    if footer.get("error") is not None:
        raise ValueError(f"Complete cache footer must have error=null: {path}")

    entries: list[CacheEntry] = []
    keys: set[tuple[int, int]] = set()
    for entry_index, (line_number, line) in enumerate(lines[1:-1]):
        payload = _load_json_line(path, line_number, line)
        if payload.get("type") != "entry":
            raise ValueError(f"{path}:{line_number} must have type='entry'")
        episode = payload.get("episode")
        request = payload.get("request")
        if (
            isinstance(episode, bool)
            or isinstance(request, bool)
            or not isinstance(episode, int)
            or not isinstance(request, int)
            or episode < 0
            or request < 0
        ):
            raise ValueError(f"{path}:{line_number} has an invalid episode/request key")
        key = (episode, request)
        if key in keys:
            raise ValueError(f"Duplicate cache key {key} in {path}")
        embedding = _parse_embedding(
            payload.get("embedding"),
            dim=embedding_dim,
            label=f"{path}:{line_number} embedding",
        )
        embedding_sha256 = _embedding_sha256(embedding)
        recorded_sha256 = _normalize_sha256(
            payload.get("embedding_sha256"),
            label=f"{path}:{line_number} embedding_sha256",
        )
        if recorded_sha256 != embedding_sha256:
            raise ValueError(f"Embedding SHA256 mismatch at {path}:{line_number}")
        entries.append(
            CacheEntry(
                shard_index=shard_index,
                entry_index=entry_index,
                episode=episode,
                request=request,
                embedding=embedding,
                embedding_sha256=embedding_sha256,
            )
        )
        keys.add(key)
    if not entries:
        raise ValueError(f"Input cache contains no entries: {path}")

    request_counts, expected_keys = _request_counts_and_expected_keys(
        keys, protocol=protocol
    )
    source_keyset_sha256 = _cache_keyset_sha256(keys)
    expected_source_footer = {
        "entry_count": len(keys),
        "episode_request_counts": request_counts,
        "keyset_sha256": source_keyset_sha256,
    }
    for key, expected in expected_source_footer.items():
        if footer.get(key) != expected:
            raise ValueError(
                f"Cache source footer mismatch for {key}: expected {expected!r}, "
                f"got {footer.get(key)!r}: {path}"
            )
    replay_header = dict(header)
    replay_header["coverage_policy"] = COVERAGE_FULL
    replay_request_counts = {
        str(episode): protocol["inference"]["max_requests_per_episode"]
        for episode in range(
            protocol["episodes"]["local_start"],
            protocol["episodes"]["local_end_exclusive"],
        )
    }
    replay_keyset_sha256 = _cache_keyset_sha256(expected_keys)
    replay_footer = {
        "type": "footer",
        "schema_version": CACHE_SCHEMA_VERSION,
        "complete": True,
        "protocol_sha256": protocol_sha256,
        "coverage_policy": COVERAGE_FULL,
        "entry_count": len(expected_keys),
        "episode_request_counts": replay_request_counts,
        "keyset_sha256": replay_keyset_sha256,
        "error": None,
    }
    return CacheShard(
        path=path,
        file_sha256=_sha256_file(path),
        source_header=header,
        replay_header=replay_header,
        source_footer=footer,
        replay_footer=replay_footer,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        request_counts=request_counts,
        source_keyset_sha256=source_keyset_sha256,
        replay_keyset_sha256=replay_keyset_sha256,
        entries=tuple(entries),
    )


def _seeded_rank(seed: int, *parts: Any) -> str:
    return _json_sha256({"seed": int(seed), "parts": list(parts)})


def _episode_trajectories(shard: CacheShard) -> dict[int, EpisodeTrajectory]:
    grouped: dict[int, list[CacheEntry]] = defaultdict(list)
    for entry in shard.entries:
        grouped[entry.episode].append(entry)
    trajectories: dict[int, EpisodeTrajectory] = {}
    for episode in range(
        shard.protocol["episodes"]["local_start"],
        shard.protocol["episodes"]["local_end_exclusive"],
    ):
        entries = tuple(sorted(grouped.get(episode, []), key=lambda item: item.request))
        if not entries:
            raise ValueError(f"Cache is missing local episode {episode}")
        if [entry.request for entry in entries] != list(range(len(entries))):
            raise ValueError(f"Cache episode {episode} has non-contiguous requests")
        trajectories[episode] = EpisodeTrajectory(
            shard_index=entries[0].shard_index,
            episode=episode,
            entries=entries,
        )
    return trajectories


def _build_shard_mapping(
    shard: CacheShard, *, seed: int
) -> dict[int, EpisodeTrajectory]:
    """Derange complete episode trajectories inside one shard."""

    trajectories = _episode_trajectories(shard)
    if len(trajectories) < 2:
        raise ValueError("Cannot derange a shard with fewer than two episodes")
    namespace = f"shard:{shard.protocol['episodes']['shard_id']}:{shard.protocol_sha256}"
    ordered = sorted(
        trajectories.values(),
        key=lambda trajectory: (
            _seeded_rank(seed, namespace, "episode", trajectory.episode),
            trajectory.episode,
        ),
    )
    shift = int(_seeded_rank(seed, namespace, "shift"), 16) % (len(ordered) - 1) + 1
    donors = ordered[shift:] + ordered[:shift]
    mapping = {
        target.episode: donor
        for target, donor in zip(ordered, donors, strict=True)
    }
    if set(mapping) != set(trajectories):
        raise RuntimeError("Internal error: episode derangement is not total")
    if {donor.episode for donor in mapping.values()} != set(trajectories):
        raise RuntimeError("Internal error: donor episode assignment is not bijective")
    if any(target == donor.episode for target, donor in mapping.items()):
        raise RuntimeError("Internal error: an episode retained itself as donor")
    if any(donor.shard_index != ordered[0].shard_index for donor in mapping.values()):
        raise RuntimeError("Internal error: donor crossed a shard boundary")
    return mapping


def _request_assignments(
    shard: CacheShard,
    mapping: Mapping[int, EpisodeTrajectory],
) -> list[RequestAssignment]:
    assignments: list[RequestAssignment] = []
    max_requests = shard.protocol["inference"]["max_requests_per_episode"]
    for target_episode in range(
        shard.protocol["episodes"]["local_start"],
        shard.protocol["episodes"]["local_end_exclusive"],
    ):
        donor = mapping[target_episode]
        for target_request in range(max_requests):
            repeat_last = target_request >= donor.observed_count
            donor_entry = donor.last if repeat_last else donor.entries[target_request]
            assignments.append(
                RequestAssignment(
                    shard_index=donor.shard_index,
                    target_episode=target_episode,
                    target_request=target_request,
                    donor_episode=donor.episode,
                    donor_entry=donor_entry,
                    repeat_last=repeat_last,
                )
            )
    return assignments


def _entry_payload(assignment: RequestAssignment) -> dict[str, Any]:
    return {
        "type": "entry",
        "episode": assignment.target_episode,
        "request": assignment.target_request,
        "embedding": list(assignment.donor_entry.embedding),
        "embedding_sha256": assignment.donor_entry.embedding_sha256,
    }


def _render_cache(shard: CacheShard, assignments: Sequence[RequestAssignment]) -> bytes:
    lines = [_canonical_json_bytes(shard.replay_header)]
    lines.extend(_canonical_json_bytes(_entry_payload(item)) for item in assignments)
    lines.append(_canonical_json_bytes(shard.replay_footer))
    return b"\n".join(lines) + b"\n"


def _multiset_payload(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"embedding_sha256": digest, "count": counter[digest]}
        for digest in sorted(counter)
    ]


def _assert_distinct_paths(
    input_paths: Sequence[Path], output_paths: Sequence[Path], proof_path: Path
) -> None:
    inputs = [path.resolve() for path in input_paths]
    outputs = [path.resolve() for path in output_paths]
    if len(set(inputs)) != len(inputs):
        raise ValueError("Input cache paths must be unique")
    if len(set(outputs)) != len(outputs):
        raise ValueError("Output cache paths must be unique")
    if set(inputs) & set(outputs):
        raise ValueError("Output cache paths must not overwrite input caches")
    if proof_path.resolve() in set(inputs) | set(outputs):
        raise ValueError("Proof output path must be distinct from cache paths")


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _verify_rendered_replay(
    shard: CacheShard,
    payload: bytes,
    mapping: Mapping[int, EpisodeTrajectory],
    assignments: Sequence[RequestAssignment],
) -> dict[str, Any]:
    rows = [json.loads(line) for line in payload.splitlines()]
    header = rows[0]
    footer = rows[-1]
    entries = rows[1:-1]
    target_keys = [
        (episode, request)
        for episode in range(
            shard.protocol["episodes"]["local_start"],
            shard.protocol["episodes"]["local_end_exclusive"],
        )
        for request in range(
            shard.protocol["inference"]["max_requests_per_episode"]
        )
    ]
    output_keys = [(row["episode"], row["request"]) for row in entries]
    expected_header = dict(shard.source_header)
    expected_header["coverage_policy"] = COVERAGE_FULL
    expected_footer = shard.replay_footer
    source_embedding_counter = Counter(
        entry.embedding_sha256 for entry in shard.entries
    )
    direct_output_counter = Counter(
        item.donor_entry.embedding_sha256
        for item in assignments
        if not item.repeat_last
    )
    checks = {
        "protocol_and_header_preserved": header == expected_header,
        "full_horizon_footer_exact": footer == expected_footer,
        "target_key_order_preserved": output_keys == target_keys,
        "target_keyset_preserved": set(output_keys) == set(target_keys),
        "complete_coverage": len(output_keys) == len(target_keys),
        "donor_within_shard": all(
            donor.shard_index == shard.entries[0].shard_index
            for donor in mapping.values()
        ),
        "no_donor_from_same_episode": all(
            target_episode != donor.episode
            for target_episode, donor in mapping.items()
        ),
        "donor_episode_bijection": len({donor.episode for donor in mapping.values()})
        == len(mapping),
        "observed_donor_prefix_multiset_equal": source_embedding_counter
        == direct_output_counter,
        "repeat_last_only_after_donor_end": all(
            item.repeat_last
            == (item.target_request >= mapping[item.target_episode].observed_count)
            and item.donor_entry.request
            == min(
                item.target_request,
                mapping[item.target_episode].observed_count - 1,
            )
            for item in assignments
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Rendered replay cache invariant failed: {checks}")
    return checks


def build(
    input_paths: Sequence[Path],
    output_paths: Sequence[Path],
    proof_path: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    if not input_paths:
        raise ValueError("At least one input cache is required")
    if len(input_paths) != len(output_paths):
        raise ValueError("Provide exactly one output cache for each input cache")
    input_paths = [Path(path) for path in input_paths]
    output_paths = [Path(path) for path in output_paths]
    proof_path = Path(proof_path)
    _assert_distinct_paths(input_paths, output_paths, proof_path)
    existing = [path for path in [*output_paths, proof_path] if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing)
        )

    shards = [
        _load_cache(path, shard_index=index)
        for index, path in enumerate(input_paths)
    ]
    model_contract = {
        key: shards[0].source_header[key]
        for key in ("schema_version", "checkpoint_sha256", "config_sha256", "embedding_dim")
    }
    for shard in shards[1:]:
        observed = {key: shard.source_header[key] for key in model_contract}
        if observed != model_contract:
            raise ValueError(
                "All input shards must share schema/checkpoint/config/embedding_dim"
            )

    mappings = [_build_shard_mapping(shard, seed=int(seed)) for shard in shards]
    assignments = [
        _request_assignments(shard, mapping)
        for shard, mapping in zip(shards, mappings, strict=True)
    ]
    rendered = [
        _render_cache(shard, shard_assignments)
        for shard, shard_assignments in zip(shards, assignments, strict=True)
    ]
    shard_checks = [
        _verify_rendered_replay(shard, payload, mapping, shard_assignments)
        for shard, payload, mapping, shard_assignments in zip(
            shards, rendered, mappings, assignments, strict=True
        )
    ]

    source_counter = Counter(
        entry.embedding_sha256 for shard in shards for entry in shard.entries
    )
    direct_output_counter = Counter(
        item.donor_entry.embedding_sha256
        for shard_assignments in assignments
        for item in shard_assignments
        if not item.repeat_last
    )
    full_output_counter = Counter(
        item.donor_entry.embedding_sha256
        for shard_assignments in assignments
        for item in shard_assignments
    )
    observed_entries = sum(len(shard.entries) for shard in shards)
    total_entries = sum(len(items) for items in assignments)
    extension_count = sum(
        item.repeat_last for shard_assignments in assignments for item in shard_assignments
    )
    global_checks = {
        "all_sources_finalized_observed_contiguous": all(
            shard.source_footer.get("complete") is True
            and shard.source_footer.get("coverage_policy") == COVERAGE_OBSERVED
            for shard in shards
        ),
        "all_outputs_full_horizon": all(
            shard.replay_footer["coverage_policy"] == COVERAGE_FULL
            for shard in shards
        ),
        "all_target_keys_covered_exactly_once": all(
            checks["complete_coverage"] and checks["target_keyset_preserved"]
            for checks in shard_checks
        ),
        "no_cross_shard_donors": all(
            checks["donor_within_shard"] for checks in shard_checks
        ),
        "no_donor_from_same_episode": all(
            checks["no_donor_from_same_episode"] for checks in shard_checks
        ),
        "observed_donor_prefix_multiset_equal": source_counter
        == direct_output_counter,
        "repeat_last_extension_count_consistent": observed_entries + extension_count
        == total_entries,
    }
    if not all(global_checks.values()):
        raise RuntimeError(f"Global steer-cache invariant failed: {global_checks}")

    output_hashes = [_sha256_bytes(payload) for payload in rendered]
    source_multiset = _multiset_payload(source_counter)
    direct_output_multiset = _multiset_payload(direct_output_counter)
    full_output_multiset = _multiset_payload(full_output_counter)
    shard_multiset_proofs = []
    for shard, shard_assignments in zip(shards, assignments, strict=True):
        shard_source = Counter(entry.embedding_sha256 for entry in shard.entries)
        shard_direct = Counter(
            item.donor_entry.embedding_sha256
            for item in shard_assignments
            if not item.repeat_last
        )
        shard_multiset_proofs.append(
            {
                "source_sha256": _json_sha256(_multiset_payload(shard_source)),
                "observed_prefix_output_sha256": _json_sha256(
                    _multiset_payload(shard_direct)
                ),
                "equal": shard_source == shard_direct,
                "source_entry_count": sum(shard_source.values()),
                "observed_prefix_output_entry_count": sum(shard_direct.values()),
            }
        )
    proof: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "seed": int(seed),
        "donor_scope": "within_shard",
        "model_contract": model_contract,
        "source": {
            "cache_count": len(shards),
            "entry_count": observed_entries,
            "files": [
                {
                    "shard_index": index,
                    "protocol_shard_id": shard.protocol["episodes"]["shard_id"],
                    "path": str(shard.path.resolve()),
                    "sha256": shard.file_sha256,
                    "entry_count": len(shard.entries),
                    "header_sha256": _json_sha256(shard.source_header),
                    "footer_sha256": _json_sha256(shard.source_footer),
                    "protocol_sha256": shard.protocol_sha256,
                    "keyset_sha256": shard.source_keyset_sha256,
                    "episode_observed_counts": shard.request_counts,
                }
                for index, shard in enumerate(shards)
            ],
        },
        "output": {
            "cache_count": len(shards),
            "entry_count": total_entries,
            "files": [
                {
                    "shard_index": index,
                    "protocol_shard_id": shard.protocol["episodes"]["shard_id"],
                    "path": str(path.resolve()),
                    "sha256": output_hashes[index],
                    "entry_count": len(shard.entries),
                    "header_sha256": _json_sha256(shard.replay_header),
                    "footer_sha256": _json_sha256(shard.replay_footer),
                    "protocol_sha256": shard.protocol_sha256,
                    "keyset_sha256": shard.replay_keyset_sha256,
                }
                for index, (shard, path) in enumerate(
                    zip(shards, output_paths, strict=True)
                )
            ],
        },
        "extension": {
            "observed_prefix_entry_count": observed_entries,
            "repeat_last_entry_count": extension_count,
            "full_horizon_entry_count": total_entries,
            "repeat_last_fraction": extension_count / total_entries,
        },
        "observed_donor_prefix_embedding_multiset": {
            "source_sha256": _json_sha256(source_multiset),
            "output_sha256": _json_sha256(direct_output_multiset),
            "equal": source_counter == direct_output_counter,
            "unique_embedding_count": len(source_counter),
            "items": source_multiset,
        },
        "full_output_embedding_multiset": {
            "sha256": _json_sha256(full_output_multiset),
            "entry_count": total_entries,
            "includes_repeat_last_extensions": extension_count > 0,
            "source_equality_is_not_an_invariant": True,
        },
        "global_invariant_checks": global_checks,
        "shard_invariant_checks": [
            {
                "shard_index": index,
                "protocol_shard_id": shard.protocol["episodes"]["shard_id"],
                **checks,
                "observed_prefix_embedding_multiset": shard_multiset_proofs[index],
            }
            for index, (shard, checks) in enumerate(
                zip(shards, shard_checks, strict=True)
            )
        ],
        "episode_donor_mapping": [
            {
                "target": {
                    "shard_index": shard_index,
                    "episode": target_episode,
                    "observed_count": shard.request_counts[str(target_episode)],
                },
                "donor": {
                    "shard_index": donor.shard_index,
                    "episode": donor.episode,
                    "observed_count": donor.observed_count,
                },
                "full_horizon_count": shard.protocol["inference"][
                    "max_requests_per_episode"
                ],
                "extension_count": shard.protocol["inference"][
                    "max_requests_per_episode"
                ]
                - donor.observed_count,
                "repeat_last_mapping": {
                    "donor_request": donor.observed_count - 1,
                    "target_request_start": donor.observed_count,
                    "target_request_end_exclusive": shard.protocol["inference"][
                        "max_requests_per_episode"
                    ],
                },
            }
            for shard_index, (shard, mapping) in enumerate(
                zip(shards, mappings, strict=True)
            )
            for target_episode, donor in sorted(mapping.items())
        ],
        "request_mapping": [
            {
                "target": {
                    "shard_index": item.shard_index,
                    "episode": item.target_episode,
                    "request": item.target_request,
                },
                "donor": {
                    "shard_index": item.donor_entry.shard_index,
                    "episode": item.donor_episode,
                    "request": item.donor_entry.request,
                    "embedding_sha256": item.donor_entry.embedding_sha256,
                },
                "repeat_last": item.repeat_last,
            }
            for shard_assignments in assignments
            for item in shard_assignments
        ],
    }
    proof["proof_sha256"] = _json_sha256(proof)

    written: list[Path] = []
    try:
        for path, payload in zip(output_paths, rendered, strict=True):
            _write_exclusive(path, payload)
            written.append(path)
        _write_exclusive(proof_path, _canonical_json_bytes(proof, pretty=True))
        written.append(proof_path)
    except Exception:
        for path in reversed(written):
            path.unlink(missing_ok=True)
        raise
    for path, expected_hash in zip(output_paths, output_hashes, strict=True):
        if _sha256_file(path) != expected_hash:
            raise RuntimeError(f"Post-write SHA256 verification failed: {path}")
    return proof


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-cache",
        action="append",
        type=Path,
        required=True,
        help="Observed schema-v2 JSONL cache. Repeat in shard order.",
    )
    parser.add_argument(
        "--output-cache",
        action="append",
        type=Path,
        required=True,
        help="Full-horizon shuffled replay cache. Repeat once per input shard.",
    )
    parser.add_argument("--proof-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    proof = build(
        args.input_cache,
        args.output_cache,
        args.proof_output,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "proof_output": str(args.proof_output),
                "proof_sha256": proof["proof_sha256"],
                "entry_count": proof["source"]["entry_count"],
                "donor_scope": proof["donor_scope"],
                "extension": proof["extension"],
                "output_sha256": [
                    item["sha256"] for item in proof["output"]["files"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
