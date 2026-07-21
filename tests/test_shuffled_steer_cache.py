import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "everobot"
    / "build_shuffled_steer_cache.py"
)
SPEC = importlib.util.spec_from_file_location("build_shuffled_steer_cache", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
shuffle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = shuffle
SPEC.loader.exec_module(shuffle)


CHECKPOINT_SHA = "a" * 64
CONFIG_SHA = "b" * 64


def make_protocol(
    *,
    shard_id: int,
    shard_count: int = 2,
    episodes_per_shard: int = 100,
    max_env_steps: int = 50,
    replan_steps: int = 25,
) -> dict:
    global_episodes = shard_count * episodes_per_shard
    shard_global_start = shard_id * episodes_per_shard
    global_seed_base = 20262000
    return {
        "schema": "fastwam.steer_protocol",
        "schema_version": 1,
        "task": "water_plant",
        "environment_seeds": {
            "global_base": global_seed_base,
            "global_end_exclusive": global_seed_base + global_episodes,
            "shard_base": global_seed_base + shard_global_start,
            "shard_end_exclusive": (
                global_seed_base + shard_global_start + episodes_per_shard
            ),
        },
        "episodes": {
            "global_start": 0,
            "global_end_exclusive": global_episodes,
            "shard_id": shard_id,
            "shard_global_start": shard_global_start,
            "shard_global_end_exclusive": shard_global_start + episodes_per_shard,
            "local_start": 0,
            "local_end_exclusive": episodes_per_shard,
        },
        "inference": {
            "seed": 314159,
            "replan_steps": replan_steps,
            "max_env_steps": max_env_steps,
            "max_requests_per_episode": math.ceil(
                max_env_steps / replan_steps
            ),
            "control_mode": "blocking",
            "async_fallback": "wait",
            "action_horizon_override": None,
            "num_inference_steps_override": None,
        },
        "environment_options": {
            "randomize": False,
            "randomize_dynamics": False,
            "action_clip": False,
            "clip_max_xyz_step": 0.05,
            "clip_max_dz_down": 0.03,
            "task_config_dir": "/tmp/rand_obj",
        },
        "model": {
            "checkpoint_path": "/tmp/step_006000.pt",
            "checkpoint_sha256": CHECKPOINT_SHA,
            "config_path": "/tmp/config.yaml",
            "config_sha256": CONFIG_SHA,
        },
    }


def full_horizon_rows(protocol: dict, *, offset: float = 0.0):
    rows = []
    request_count = protocol["inference"]["max_requests_per_episode"]
    episode_count = protocol["episodes"]["local_end_exclusive"]
    for episode in range(episode_count):
        for request in range(request_count):
            value = offset + episode * request_count + request
            rows.append((episode, request, [value, value + 0.25]))
    return rows


def variable_observed_rows(protocol: dict, *, offset: float = 0.0):
    rows = []
    max_requests = protocol["inference"]["max_requests_per_episode"]
    episode_count = protocol["episodes"]["local_end_exclusive"]
    for episode in range(episode_count):
        observed_count = 1 + episode % max_requests
        for request in range(observed_count):
            value = offset + episode * max_requests + request
            rows.append((episode, request, [value, value + 0.25]))
    return rows


def write_observed_cache(
    path: Path,
    protocol: dict,
    rows: list[tuple[int, int, list[float]]],
    *,
    footer_overrides: dict | None = None,
):
    protocol_sha = shuffle._json_sha256(protocol)
    header = {
        "type": "header",
        "schema_version": 2,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "config_sha256": CONFIG_SHA,
        "embedding_dim": 2,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "coverage_policy": "observed_contiguous",
    }
    entries = []
    keys = set()
    for episode, request, embedding in rows:
        keys.add((episode, request))
        entries.append(
            {
                "type": "entry",
                "episode": episode,
                "request": request,
                "embedding": embedding,
                "embedding_sha256": shuffle._embedding_sha256(embedding),
            }
        )
    counts = {
        str(episode): sum(1 for ep, _ in keys if ep == episode)
        for episode in range(protocol["episodes"]["local_end_exclusive"])
        if any(ep == episode for ep, _ in keys)
    }
    footer = {
        "type": "footer",
        "schema_version": 2,
        "complete": True,
        "protocol_sha256": protocol_sha,
        "coverage_policy": "observed_contiguous",
        "entry_count": len(keys),
        "episode_request_counts": counts,
        "keyset_sha256": shuffle._cache_keyset_sha256(keys),
        "error": None,
    }
    if footer_overrides:
        footer.update(footer_overrides)
    path.write_text(
        "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            for item in [header, *entries, footer]
        ),
        encoding="utf-8",
    )
    return header, footer


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def assert_strict_replay_cache(
    test: unittest.TestCase,
    path: Path,
    *,
    source_header: dict,
    protocol: dict,
):
    rows = read_jsonl(path)
    header, footer = rows[0], rows[-1]
    entries = rows[1:-1]
    expected_header = dict(source_header)
    expected_header["coverage_policy"] = "full_horizon"
    test.assertEqual(header, expected_header)
    test.assertEqual(footer["type"], "footer")
    test.assertEqual(footer["schema_version"], 2)
    test.assertIs(footer["complete"], True)
    test.assertEqual(footer["coverage_policy"], "full_horizon")
    test.assertEqual(footer["protocol_sha256"], shuffle._json_sha256(protocol))
    test.assertIsNone(footer["error"])
    expected_keys = {
        (episode, request)
        for episode in range(protocol["episodes"]["local_end_exclusive"])
        for request in range(protocol["inference"]["max_requests_per_episode"])
    }
    keys = {(row["episode"], row["request"]) for row in entries}
    test.assertEqual(keys, expected_keys)
    test.assertEqual(len(entries), len(expected_keys))
    test.assertEqual(footer["entry_count"], len(expected_keys))
    test.assertEqual(footer["keyset_sha256"], shuffle._cache_keyset_sha256(keys))
    expected_counts = {
        str(episode): protocol["inference"]["max_requests_per_episode"]
        for episode in range(protocol["episodes"]["local_end_exclusive"])
    }
    test.assertEqual(footer["episode_request_counts"], expected_counts)
    for row in entries:
        test.assertEqual(
            row["embedding_sha256"],
            shuffle._embedding_sha256(row["embedding"]),
        )


class ShuffledSteerCacheV2Test(unittest.TestCase):
    def test_two_formal_shards_are_deterministic_and_deranged_within_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocols = [make_protocol(shard_id=index) for index in range(2)]
            sources = [root / f"observed-{index}.jsonl" for index in range(2)]
            headers = []
            for index, (source, protocol) in enumerate(zip(sources, protocols)):
                header, _ = write_observed_cache(
                    source,
                    protocol,
                    variable_observed_rows(protocol, offset=index * 10000.0),
                )
                headers.append(header)
            outputs_a = [root / f"replay-a-{index}.jsonl" for index in range(2)]
            outputs_b = [root / f"replay-b-{index}.jsonl" for index in range(2)]

            proof_a = shuffle.build(
                sources, outputs_a, root / "proof-a.json", seed=1729
            )
            proof_b = shuffle.build(
                sources, outputs_b, root / "proof-b.json", seed=1729
            )

            self.assertEqual(
                proof_a["episode_donor_mapping"], proof_b["episode_donor_mapping"]
            )
            self.assertEqual(proof_a["request_mapping"], proof_b["request_mapping"])
            self.assertEqual(proof_a["donor_scope"], "within_shard")
            self.assertTrue(all(proof_a["global_invariant_checks"].values()))
            self.assertEqual(proof_a["source"]["entry_count"], 300)
            self.assertEqual(proof_a["extension"]["repeat_last_entry_count"], 100)
            self.assertEqual(proof_a["output"]["entry_count"], 400)
            for index in range(2):
                self.assertEqual(outputs_a[index].read_bytes(), outputs_b[index].read_bytes())
                assert_strict_replay_cache(
                    self,
                    outputs_a[index],
                    source_header=headers[index],
                    protocol=protocols[index],
                )
            for row in proof_a["episode_donor_mapping"]:
                self.assertEqual(
                    row["target"]["shard_index"], row["donor"]["shard_index"]
                )
                self.assertNotEqual(
                    row["target"]["episode"], row["donor"]["episode"]
                )
                self.assertEqual(
                    row["extension_count"],
                    row["full_horizon_count"] - row["donor"]["observed_count"],
                )
            self.assertEqual(
                proof_a["observed_donor_prefix_embedding_multiset"]["source_sha256"],
                proof_a["observed_donor_prefix_embedding_multiset"]["output_sha256"],
            )
            self.assertTrue(
                proof_a["full_output_embedding_multiset"][
                    "source_equality_is_not_an_invariant"
                ]
            )

    def test_variable_lengths_repeat_donor_last_token_after_observed_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = make_protocol(
                shard_id=0,
                shard_count=1,
                episodes_per_shard=3,
                max_env_steps=75,
            )
            source = root / "source.jsonl"
            rows = variable_observed_rows(protocol)
            header, _ = write_observed_cache(source, protocol, rows)
            output = root / "output.jsonl"
            proof = shuffle.build([source], [output], root / "proof.json", seed=9)

            source_rows = read_jsonl(source)[1:-1]
            output_rows = read_jsonl(output)[1:-1]
            assert_strict_replay_cache(
                self, output, source_header=header, protocol=protocol
            )
            source_by_key = {
                (row["episode"], row["request"]): row for row in source_rows
            }
            output_by_key = {
                (row["episode"], row["request"]): row for row in output_rows
            }
            for episode_pair in proof["episode_donor_mapping"]:
                target_episode = episode_pair["target"]["episode"]
                donor_episode = episode_pair["donor"]["episode"]
                donor_count = episode_pair["donor"]["observed_count"]
                for request in range(
                    protocol["inference"]["max_requests_per_episode"]
                ):
                    donor_request = min(request, donor_count - 1)
                    self.assertEqual(
                        output_by_key[(target_episode, request)]["embedding_sha256"],
                        source_by_key[(donor_episode, donor_request)][
                            "embedding_sha256"
                        ],
                    )
            self.assertEqual(proof["source"]["entry_count"], 6)
            self.assertEqual(proof["extension"]["repeat_last_entry_count"], 3)
            self.assertEqual(proof["output"]["entry_count"], 9)
            repeat_rows = [row for row in proof["request_mapping"] if row["repeat_last"]]
            self.assertEqual(len(repeat_rows), 3)
            for row in repeat_rows:
                donor_meta = next(
                    item
                    for item in proof["episode_donor_mapping"]
                    if item["target"]["episode"] == row["target"]["episode"]
                )
                self.assertEqual(
                    row["donor"]["request"],
                    donor_meta["repeat_last_mapping"]["donor_request"],
                )
            self.assertEqual(
                proof["output"]["files"][0]["sha256"],
                shuffle._sha256_file(output),
            )

    def test_zero_entry_or_non_contiguous_episode_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = make_protocol(
                shard_id=0,
                shard_count=1,
                episodes_per_shard=2,
                max_env_steps=75,
            )
            missing = root / "missing-episode.jsonl"
            rows = [row for row in variable_observed_rows(protocol) if row[0] == 0]
            write_observed_cache(missing, protocol, rows)
            with self.assertRaisesRegex(ValueError, "missing local episode 1"):
                shuffle.build(
                    [missing], [root / "output.jsonl"], root / "proof.json", seed=1
                )

            non_contiguous = root / "non-contiguous.jsonl"
            rows = [
                (0, 0, [0.0, 0.25]),
                (0, 2, [2.0, 2.25]),
                (1, 0, [3.0, 3.25]),
            ]
            write_observed_cache(non_contiguous, protocol, rows)
            with self.assertRaisesRegex(ValueError, "non-contiguous requests"):
                shuffle.build(
                    [non_contiguous],
                    [root / "output-two.jsonl"],
                    root / "proof-two.json",
                    seed=1,
                )

    def test_incomplete_or_inconsistent_footer_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = make_protocol(
                shard_id=0, shard_count=1, episodes_per_shard=2
            )
            source = root / "bad-footer.jsonl"
            write_observed_cache(
                source,
                protocol,
                full_horizon_rows(protocol),
                footer_overrides={"complete": False},
            )
            with self.assertRaisesRegex(ValueError, "complete=true"):
                shuffle.build(
                    [source], [root / "output.jsonl"], root / "proof.json", seed=1
                )

            source_two = root / "bad-keyset.jsonl"
            write_observed_cache(
                source_two,
                protocol,
                full_horizon_rows(protocol),
                footer_overrides={"keyset_sha256": "0" * 64},
            )
            with self.assertRaisesRegex(ValueError, "footer mismatch for keyset"):
                shuffle.build(
                    [source_two],
                    [root / "output-two.jsonl"],
                    root / "proof-two.json",
                    seed=1,
                )

    def test_protocol_or_embedding_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = make_protocol(
                shard_id=0, shard_count=1, episodes_per_shard=2
            )
            source = root / "bad-protocol.jsonl"
            write_observed_cache(source, protocol, full_horizon_rows(protocol))
            rows = read_jsonl(source)
            rows[0]["protocol_sha256"] = "0" * 64
            source.write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "protocol_sha256 mismatch"):
                shuffle.build(
                    [source], [root / "output.jsonl"], root / "proof.json", seed=1
                )

            bad_embedding = root / "bad-embedding.jsonl"
            write_observed_cache(
                bad_embedding, protocol, full_horizon_rows(protocol)
            )
            rows = read_jsonl(bad_embedding)
            rows[1]["embedding_sha256"] = "0" * 64
            bad_embedding.write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Embedding SHA256 mismatch"):
                shuffle.build(
                    [bad_embedding],
                    [root / "output-two.jsonl"],
                    root / "proof-two.json",
                    seed=1,
                )

    def test_no_overwrite_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = make_protocol(
                shard_id=0, shard_count=1, episodes_per_shard=2
            )
            source = root / "source.jsonl"
            write_observed_cache(source, protocol, full_horizon_rows(protocol))
            output = root / "output.jsonl"
            proof = root / "proof.json"
            output.write_text("sentinel", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                shuffle.build([source], [output], proof, seed=1)
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel")
            self.assertFalse(proof.exists())


if __name__ == "__main__":
    unittest.main()
