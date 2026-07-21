from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastwam.everobot_schema import SCHEMA_VERSION, with_manifest_hash
from scripts.everobot import preflight_offline_run


def sample(
    sample_id: str,
    *,
    split: str,
    episode_index: int,
    outcome: str,
    role: str,
    event_id: str | None = None,
    pair_id: str | None = None,
    pair_weight: float = 0.0,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "sample_type": "event" if event_id else "episode",
        "sample_id": sample_id,
        "dataset_id": "dataset",
        "dataset_root": "/data/dataset",
        "episode_id": f"dataset:episode:{episode_index:06d}",
        "episode_index": episode_index,
        "round_id": "dataset:round:0",
        "collection_round": 0,
        "start_frame": 0,
        "end_frame": 40,
        "sample_stride": 1,
        "action_loss": "enabled" if role == "primary" else "disabled",
        "batch_role": role,
        "episode_outcome": outcome,
        "event_outcome": outcome,
        "split": split,
    }
    if event_id is not None:
        payload.update(
            {
                "event_id": event_id,
                "effector": "global",
            }
        )
    if pair_id is not None:
        payload["pair_id"] = pair_id
        payload["pair_weight"] = pair_weight
    return payload


def manifest(samples: list[dict[str, object]]) -> dict[str, object]:
    return with_manifest_hash(
        {
            "format": "EveRobotTrainManifest",
            "schema_version": SCHEMA_VERSION,
            "manifest_name": "offline-test",
            "frame_interval": "half_open",
            "selection": {"splits": ["train", "val"]},
            "dataset_roots": {"dataset": "/data/dataset"},
            "source_round_ids": ["dataset:round:0"],
            "source_hashes": {
                "round_meta_sha256": "1" * 64,
                "episode_meta_sha256": "2" * 64,
                "event_meta_sha256": "3" * 64,
            },
            "num_samples": len(samples),
            "samples": samples,
        }
    )


def control_samples(auxiliary_outcome: str) -> list[dict[str, object]]:
    return [
        sample(
            "train-primary",
            split="train",
            episode_index=0,
            outcome="success",
            role="primary",
        ),
        sample(
            "train-aux",
            split="train",
            episode_index=1,
            outcome=auxiliary_outcome,
            role="auxiliary",
        ),
    ]


def two_auxiliary_samples(auxiliary_outcome: str) -> list[dict[str, object]]:
    rows = control_samples(auxiliary_outcome)
    rows.append(
        sample(
            "train-aux-2",
            split="train",
            episode_index=2,
            outcome=auxiliary_outcome,
            role="auxiliary",
        )
    )
    return rows


def selection_samples(*, episode_index: int = 10) -> list[dict[str, object]]:
    return [
        sample(
            "val-primary",
            split="val",
            episode_index=episode_index,
            outcome="success",
            role="primary",
        )
    ]


def write_deepspeed_state_payload(
    state: Path,
    *,
    ranks: tuple[int, ...] = (0, 1, 2, 3),
) -> None:
    model_dir = state / "pytorch_model"
    model_dir.mkdir()
    for rank in ranks:
        (
            model_dir
            / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        ).write_bytes(f"optimizer-{rank}".encode("ascii"))
        (state / f"random_states_{rank}.pkl").write_bytes(
            f"random-{rank}".encode("ascii")
        )
    (model_dir / "mp_rank_00_model_states.pt").write_bytes(b"model")
    (state / "scheduler.bin").write_bytes(b"scheduler")


@dataclass
class _Target:
    split: str
    failure_event_id: str
    pair_id: str = "pair"
    success_event_id: str = "success-event"
    pair_weight: float = 0.8
    z_plus: tuple[float, ...] = (1.0, 2.0)
    z_minus: tuple[float, ...] = (3.0, 4.0)
    teacher_sha256: str = "a" * 64


class _TargetStore:
    def __init__(self, targets: dict[str, _Target]):
        self.targets = targets

    def __contains__(self, pair_id: object) -> bool:
        return pair_id in self.targets

    def __len__(self) -> int:
        return len(self.targets)

    @property
    def pair_ids(self) -> tuple[str, ...]:
        return tuple(self.targets)

    def get(self, pair_id: str) -> _Target:
        return self.targets[pair_id]


def pair_control_fixture() -> tuple[
    dict[str, object],
    dict[str, object],
    _TargetStore,
    _TargetStore,
]:
    base_rows = [control_samples("failure")[0]]
    source_targets: dict[str, _Target] = {}
    shuffled_targets: dict[str, _Target] = {}
    source_rows = list(base_rows)
    shuffled_rows = list(base_rows)
    failures = ("failure-a", "failure-b")
    successes = ("success-a", "success-b")
    for index, failure_event_id in enumerate(failures, start=1):
        source_pair_id = f"source-pair-{index}"
        shuffled_pair_id = f"shuffled-pair-{index}"
        common = sample(
            f"train-aux-{index}",
            split="train",
            episode_index=index,
            outcome="failure",
            role="auxiliary",
            event_id=failure_event_id,
        )
        common["pair_weight"] = 0.8
        source_row = dict(common)
        source_row["pair_id"] = source_pair_id
        shuffled_row = dict(common)
        shuffled_row["pair_id"] = shuffled_pair_id
        source_rows.append(source_row)
        shuffled_rows.append(shuffled_row)
        source_targets[source_pair_id] = _Target(
            pair_id=source_pair_id,
            success_event_id=successes[index - 1],
            failure_event_id=failure_event_id,
            split="train",
            z_plus=(float(index), float(index + 1)),
            z_minus=(float(index + 10), float(index + 11)),
        )
        shuffled_success_index = 1 - (index - 1)
        shuffled_targets[shuffled_pair_id] = _Target(
            pair_id=shuffled_pair_id,
            success_event_id=successes[shuffled_success_index],
            failure_event_id=failure_event_id,
            split="train",
            z_plus=(
                float(shuffled_success_index + 1),
                float(shuffled_success_index + 2),
            ),
            z_minus=(float(index + 10), float(index + 11)),
        )
    return (
        manifest(source_rows),
        manifest(shuffled_rows),
        _TargetStore(source_targets),
        _TargetStore(shuffled_targets),
    )


class PreflightOfflineRunTest(unittest.TestCase):
    def test_code_snapshot_is_deterministic_and_content_bound(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "a.py").write_text("a = 1\n", encoding="utf-8")
            (root / "nested" / "b.yaml").write_text("b: 2\n", encoding="utf-8")
            relative_paths = ("nested/b.yaml", "a.py")

            first = preflight_offline_run.build_code_snapshot(
                root, relative_paths=relative_paths
            )
            second = preflight_offline_run.build_code_snapshot(
                root, relative_paths=reversed(relative_paths)
            )
            self.assertEqual(first, second)
            self.assertEqual(
                [item["path"] for item in first["files"]],
                ["a.py", "nested/b.yaml"],
            )

            (root / "a.py").write_text("a = 2\n", encoding="utf-8")
            changed = preflight_offline_run.build_code_snapshot(
                root, relative_paths=relative_paths
            )
            self.assertNotEqual(
                first["snapshot_sha256"], changed["snapshot_sha256"]
            )

    def test_source_bundle_binds_checkpoint_config_and_normalization(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "step_006500.pt"
            config = root / "config.yaml"
            meta = root / "norm_stats_meta"
            cache = root / "text_cache"
            meta.mkdir()
            cache.mkdir()
            checkpoint.write_bytes(b"weights")
            config.write_bytes(b"config")
            (meta / "stats.json").write_bytes(b"stats")
            (meta / "modality.json").write_bytes(b"modality")
            (cache / "prompt.npz").write_bytes(b"cache")
            normalization_hash = preflight_offline_run.sha256_json(
                {
                    "kind": "meta",
                    "artifacts": {
                        "stats.json": preflight_offline_run.sha256_file(
                            meta / "stats.json"
                        ),
                        "modality.json": preflight_offline_run.sha256_file(
                            meta / "modality.json"
                        ),
                    },
                }
            )
            artifacts = (
                checkpoint,
                config,
                meta / "stats.json",
                meta / "modality.json",
                cache / "prompt.npz",
            )
            lines = [
                "normalization_kind=meta",
                f"normalization_bundle_sha256={normalization_hash}",
                "",
                "sha256",
            ]
            lines.extend(
                f"{preflight_offline_run.sha256_file(path)}  "
                f"./{path.relative_to(root).as_posix()}"
                for path in artifacts
            )
            bundle_manifest = root / "bundle_manifest.txt"
            bundle_manifest.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )
            report = preflight_offline_run.validate_source_bundle_manifest(
                bundle_manifest,
                source_checkpoint=checkpoint,
                source_config=config,
                normalization_bundle_sha256=normalization_hash,
            )
            self.assertEqual(report["artifact_count"], 5)
            with self.assertRaisesRegex(
                ValueError, "normalization SHA-256"
            ):
                preflight_offline_run.validate_source_bundle_manifest(
                    bundle_manifest,
                    source_checkpoint=checkpoint,
                    source_config=config,
                    normalization_bundle_sha256="0" * 64,
                )

    def test_b0_requires_success_auxiliary(self) -> None:
        report = preflight_offline_run.validate_manifest_protocol(
            manifest(control_samples("success")),
            variant="B0",
            targets=None,
        )
        self.assertEqual(report["positive_pair_samples"], 0)
        with self.assertRaisesRegex(ValueError, "success auxiliary"):
            preflight_offline_run.validate_manifest_protocol(
                manifest(control_samples("failure")),
                variant="B0",
                targets=None,
            )

    def test_b1_and_c_reject_pair_supervision(self) -> None:
        for variant in ("B1", "C"):
            report = preflight_offline_run.validate_manifest_protocol(
                manifest(control_samples("failure")),
                variant=variant,
                targets=None,
            )
            self.assertEqual(report["counts"]["train"]["failure"], 1)

    def test_m_accepts_failure_only_pair_targets(self) -> None:
        rows = control_samples("failure")
        targets: dict[str, _Target] = {}
        event_id = "train-failure-event"
        pair_id = "train-pair"
        rows[1] = sample(
            str(rows[1]["sample_id"]),
            split="train",
            episode_index=int(rows[1]["episode_index"]),
            outcome="failure",
            role="auxiliary",
            event_id=event_id,
            pair_id=pair_id,
            pair_weight=0.8,
        )
        targets[pair_id] = _Target(
            split="train", failure_event_id=event_id
        )

        report = preflight_offline_run.validate_manifest_protocol(
            manifest(rows),
            variant="M",
            targets=_TargetStore(targets),
        )
        self.assertEqual(report["positive_pair_samples"], 1)

    def test_pair_shuffle_control_proves_marginals_and_changed_relation(self) -> None:
        source, shuffled, source_targets, shuffled_targets = pair_control_fixture()
        report = preflight_offline_run.validate_pair_shuffle_control(
            source_manifest=source,
            shuffled_manifest=shuffled,
            source_targets=source_targets,
            shuffled_targets=shuffled_targets,
        )
        self.assertEqual(report["pair_count"], 2)
        self.assertEqual(report["changed_relation_count"], 2)
        self.assertTrue(report["all_relations_changed"])
        self.assertTrue(report["all_success_episodes_changed"])

        unchanged_targets = _TargetStore(
            {
                f"shuffled-pair-{index}": _Target(
                    pair_id=f"shuffled-pair-{index}",
                    success_event_id=f"success-{'a' if index == 1 else 'b'}",
                    failure_event_id=f"failure-{'a' if index == 1 else 'b'}",
                    split="train",
                    z_plus=(float(index), float(index + 1)),
                    z_minus=(float(index + 10), float(index + 11)),
                )
                for index in (1, 2)
            }
        )
        with self.assertRaisesRegex(ValueError, "derange every"):
            preflight_offline_run.validate_pair_shuffle_control(
                source_manifest=source,
                shuffled_manifest=shuffled,
                source_targets=source_targets,
                shuffled_targets=unchanged_targets,
            )

        changed_manifest = json.loads(json.dumps(shuffled))
        changed_manifest["samples"][1]["end_frame"] = 41
        changed_manifest = manifest(changed_manifest["samples"])
        with self.assertRaisesRegex(ValueError, "non-pair marginals"):
            preflight_offline_run.validate_pair_shuffle_control(
                source_manifest=source,
                shuffled_manifest=changed_manifest,
                source_targets=source_targets,
                shuffled_targets=shuffled_targets,
            )

        changed_targets = dict(shuffled_targets.targets)
        original = changed_targets["shuffled-pair-1"]
        changed_targets["shuffled-pair-1"] = _Target(
            pair_id=original.pair_id,
            success_event_id=original.success_event_id,
            failure_event_id=original.failure_event_id,
            split=original.split,
            pair_weight=original.pair_weight,
            z_plus=original.z_plus,
            z_minus=(99.0, 100.0),
        )
        with self.assertRaisesRegex(ValueError, "z_minus_sha256"):
            preflight_offline_run.validate_pair_shuffle_control(
                source_manifest=source,
                shuffled_manifest=shuffled,
                source_targets=source_targets,
                shuffled_targets=_TargetStore(changed_targets),
            )

    def test_pair_shuffle_rejects_same_episode_different_candidates(self) -> None:
        source, shuffled, source_targets, shuffled_targets = pair_control_fixture()
        source_targets.targets["source-pair-1"] = _Target(
            pair_id="source-pair-1",
            success_event_id="rollout_ep000001_candidate_000",
            failure_event_id="failure-a",
            split="train",
            z_plus=(1.0, 2.0),
            z_minus=(11.0, 12.0),
        )
        shuffled_targets.targets["shuffled-pair-1"] = _Target(
            pair_id="shuffled-pair-1",
            success_event_id="rollout_ep000001_candidate_001",
            failure_event_id="failure-a",
            split="train",
            z_plus=(2.0, 3.0),
            z_minus=(11.0, 12.0),
        )
        with self.assertRaisesRegex(ValueError, "another episode"):
            preflight_offline_run.validate_pair_shuffle_control(
                source_manifest=source,
                shuffled_manifest=shuffled,
                source_targets=source_targets,
                shuffled_targets=shuffled_targets,
            )

    def test_pair_shuffle_proof_is_content_and_artifact_bound(self) -> None:
        reduced_mapping = [
            {
                "failure_event_id": "failure-a",
                "source_success_event_id": "rollout_ep000001_candidate_000",
                "shuffled_success_event_id": "rollout_ep000002_candidate_000",
                "source_success_episode_id": "rollout_ep000001",
                "shuffled_success_episode_id": "rollout_ep000002",
            },
            {
                "failure_event_id": "failure-b",
                "source_success_event_id": "rollout_ep000002_candidate_000",
                "shuffled_success_event_id": "rollout_ep000001_candidate_000",
                "source_success_episode_id": "rollout_ep000002",
                "shuffled_success_episode_id": "rollout_ep000001",
            },
        ]
        control_report = {
            "pair_count": 2,
            "changed_relation_count": 2,
            "target_row_count": 2,
            "passthrough_unreferenced_target_count": 0,
            "relation_mapping_sha256": preflight_offline_run.sha256_json(
                reduced_mapping
            ),
        }
        payload = {
            "format": preflight_offline_run.PAIR_SHUFFLE_PROOF_FORMAT,
            "version": preflight_offline_run.PAIR_SHUFFLE_PROOF_VERSION,
            "shuffle_seed": 20260721,
            "source": {
                "manifest_file_sha256": "1" * 64,
                "pair_targets_file_sha256": "2" * 64,
                "referenced_pair_count": 2,
                "target_row_count": 2,
                "passthrough_unreferenced_target_count": 0,
            },
            "output": {
                "manifest_file_sha256": "3" * 64,
                "pair_targets_file_sha256": "4" * 64,
            },
            "mapping": [
                {
                    "failure_event_id": row["failure_event_id"],
                    "original_success_event_id": row["source_success_event_id"],
                    "shuffled_success_event_id": row["shuffled_success_event_id"],
                    "original_success_episode_id": row[
                        "source_success_episode_id"
                    ],
                    "shuffled_success_episode_id": row[
                        "shuffled_success_episode_id"
                    ],
                }
                for row in reduced_mapping
            ],
            "invariant_checks": {
                "sample_order_and_non_pair_fields_preserved": True,
                "all_referenced_success_identities_deranged": True,
                "all_referenced_success_episodes_deranged": True,
                "failure_event_ids_preserved_rowwise": True,
                "z_minus_preserved_rowwise": True,
                "success_event_id_z_plus_multiset_preserved": True,
            },
        }
        payload["proof_sha256"] = preflight_offline_run.sha256_json(payload)
        report = preflight_offline_run.validate_pair_shuffle_proof(
            payload,
            seed=20260721,
            source_manifest_sha256="1" * 64,
            source_pair_targets_sha256="2" * 64,
            output_manifest_sha256="3" * 64,
            output_pair_targets_sha256="4" * 64,
            control_report=control_report,
        )
        self.assertEqual(report["proof_sha256"], payload["proof_sha256"])
        tampered = dict(payload)
        tampered["shuffle_seed"] = 7
        with self.assertRaisesRegex(ValueError, "proof hash"):
            preflight_offline_run.validate_pair_shuffle_proof(
                tampered,
                seed=20260721,
                source_manifest_sha256="1" * 64,
                source_pair_targets_sha256="2" * 64,
                output_manifest_sha256="3" * 64,
                output_pair_targets_sha256="4" * 64,
                control_report=control_report,
            )

    def test_common_init_arguments_are_variant_local(self) -> None:
        values = ("weights", "proof", "config", 42, "a", "b", "c", "d")
        preflight_offline_run.validate_common_init_arguments(
            variant="M",
            strict_mode=False,
            common_init_values=(None,) * len(values),
        )
        preflight_offline_run.validate_common_init_arguments(
            variant="B0",
            strict_mode=False,
            common_init_values=(None,) * len(values),
        )
        preflight_offline_run.validate_common_init_arguments(
            variant="M",
            strict_mode=True,
            common_init_values=values,
        )
        preflight_offline_run.validate_common_init_arguments(
            variant="M_PAIR_SHUFFLE",
            strict_mode=True,
            common_init_values=values,
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            preflight_offline_run.validate_common_init_arguments(
                variant="M_PAIR_SHUFFLE",
                strict_mode=False,
                common_init_values=(None,) * len(values),
            )
        with self.assertRaisesRegex(ValueError, "only valid"):
            preflight_offline_run.validate_common_init_arguments(
                variant="B1",
                strict_mode=True,
                common_init_values=values,
            )
        with self.assertRaisesRegex(ValueError, "require"):
            preflight_offline_run.validate_common_init_arguments(
                variant="M",
                strict_mode=True,
                common_init_values=(None,) * len(values),
            )

    def test_common_init_proof_binds_all_inputs_and_output(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "common.pt"
            config = root / "resolved.yaml"
            baseline = root / "s0.pt"
            proof_path = root / "proof.json"
            weights.write_bytes(b"complete-common-init")
            config.write_text("seed: 42\n", encoding="utf-8")
            baseline.write_bytes(b"frozen-s0")
            weights_hash = preflight_offline_run.sha256_file(weights)
            config_hash = preflight_offline_run.sha256_file(config)
            baseline_hash = preflight_offline_run.sha256_file(baseline)
            payload = {
                "format": preflight_offline_run.COMMON_INIT_PROOF_FORMAT,
                "schema_version": preflight_offline_run.COMMON_INIT_PROOF_VERSION,
                "seed": 42,
                "inputs": {
                    "resolved_config": {
                        "path": str(config.resolve()),
                        "sha256": config_hash,
                    },
                    "baseline_checkpoint": {
                        "path": str(baseline.resolve()),
                        "sha256": baseline_hash,
                    },
                },
                "output": {
                    "checkpoint": {
                        "path": str(weights.resolve()),
                        "sha256": weights_hash,
                        "step": 0,
                        "top_level_keys": [
                            "mot",
                            "offline_steer_student",
                            "offline_steer_residual",
                            "offline_steer_config",
                            "proprio_encoder",
                        ],
                    }
                },
                "invariants": {
                    "seeded_before_model_instantiation": True,
                    "full_s0_load": True,
                    "baseline_exact_structure_match": True,
                    "baseline_is_steer_free": True,
                    "steer_unchanged_by_s0_load": True,
                    "residual_is_zero_initialized": True,
                    "complete_weight_only_checkpoint": True,
                    "no_overwrite": True,
                },
            }
            payload["proof_sha256"] = preflight_offline_run.sha256_json(payload)
            proof_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            proof_file_hash = preflight_offline_run.sha256_file(proof_path)
            report = preflight_offline_run.validate_common_init_proof(
                payload,
                proof_path=proof_path,
                common_init_weights=weights,
                common_init_config=config,
                baseline_s0=baseline,
                seed=42,
                expected_weights_sha256=weights_hash,
                expected_proof_file_sha256=proof_file_hash,
                expected_baseline_sha256=baseline_hash,
                expected_config_sha256=config_hash,
            )
            self.assertEqual(report["weights"]["sha256"], weights_hash)
            self.assertEqual(report["proof"]["sha256"], proof_file_hash)
            self.assertEqual(report["seed"], 42)

            expected = {
                "expected_weights_sha256": weights_hash,
                "expected_proof_file_sha256": proof_file_hash,
                "expected_baseline_sha256": baseline_hash,
                "expected_config_sha256": config_hash,
            }
            for field, label in (
                ("expected_weights_sha256", "weights"),
                ("expected_proof_file_sha256", "proof_file"),
                ("expected_baseline_sha256", "baseline"),
                ("expected_config_sha256", "config"),
            ):
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, f"{label} SHA-256 mismatch"
                ):
                    bad_expected = dict(expected)
                    bad_expected[field] = "0" * 64
                    preflight_offline_run.validate_common_init_proof(
                        payload,
                        proof_path=proof_path,
                        common_init_weights=weights,
                        common_init_config=config,
                        baseline_s0=baseline,
                        seed=42,
                        **bad_expected,
                    )
            with self.assertRaisesRegex(ValueError, "common-init seed"):
                preflight_offline_run.validate_common_init_proof(
                    payload,
                    proof_path=proof_path,
                    common_init_weights=weights,
                    common_init_config=config,
                    baseline_s0=baseline,
                    seed=7,
                    **expected,
                )

    def test_training_manifest_rejects_validation_samples(self) -> None:
        rows = control_samples("failure")
        rows.append(selection_samples()[0])
        with self.assertRaisesRegex(ValueError, "split='train'"):
            preflight_offline_run.validate_manifest_protocol(
                manifest(rows), variant="B1", targets=None
            )

    def test_selection_manifest_is_shared_and_episode_disjoint(self) -> None:
        manifests = {
            "B0": manifest(control_samples("success")),
            "B1": manifest(control_samples("failure")),
            "C": manifest(control_samples("failure")),
            "M": manifest(control_samples("failure")),
        }
        report, identities = preflight_offline_run.validate_selection_manifest(
            manifest(selection_samples()),
            training_manifests=manifests,
        )
        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(len(identities), 1)
        with self.assertRaisesRegex(ValueError, "leakage"):
            preflight_offline_run.validate_selection_manifest(
                manifest(selection_samples(episode_index=0)),
                training_manifests=manifests,
            )

    def test_source_config_requires_two_views_and_proprio(self) -> None:
        payload = {
            "data": {
                "train": {
                    "shape_meta": {
                        "images": [
                            {"key": "front"},
                            {"key": "wrist"},
                        ]
                    },
                    "video_size": [384, 768],
                    "processor": {
                        "action_output_dim": 22,
                        "proprio_output_dim": 23,
                    },
                }
            },
            "model": {"proprio_dim": 23},
        }
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text(json.dumps(payload), encoding="utf-8")
            contract = preflight_offline_run.validate_source_config(path)
        self.assertEqual(contract["camera_keys"], ["front", "wrist"])
        self.assertEqual(contract["proprio_dim"], 23)

    def test_training_data_contract_accepts_frozen_dataset_stats(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "dataset"
            text_cache_dir = root / "text_cache"
            dataset_stats = root / "dataset_stats.json"
            (dataset_root / "meta").mkdir(parents=True)
            text_cache_dir.mkdir()
            dataset_stats.write_bytes(b"dataset-stats")
            (dataset_root / "meta" / "episodes.jsonl").write_text(
                json.dumps(
                    {
                        "episode_index": 0,
                        "tasks": [preflight_offline_run.WATER_PLANT_TASK],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            text_cache_file = (
                text_cache_dir
                / preflight_offline_run.WATER_PLANT_TEXT_CACHE_BASENAME
            )
            text_cache_file.write_bytes(b"text-cache")
            text_cache_hash = preflight_offline_run.sha256_file(
                text_cache_file
            )
            split = {
                "dataset_dirs": [str(dataset_root)],
                "pretrained_norm_stats": str(dataset_stats),
                "context_len": 128,
                "text_embedding_cache_dir": str(text_cache_dir),
                "processor": {
                    "norm_stats_source": "compute",
                    "norm_stats_meta_dir": None,
                },
            }
            report = preflight_offline_run.validate_training_data_contract(
                {"data": {"train": split, "val": dict(split)}},
                expected_dataset_roots=[str(dataset_root)],
                expected_normalization_kind="compute",
                referenced_samples=[
                    {
                        "dataset_root": str(dataset_root),
                        "episode_index": 0,
                        "task": preflight_offline_run.WATER_PLANT_TASK,
                    }
                ],
            )
        self.assertEqual(report["normalization_kind"], "compute")
        self.assertEqual(
            set(report["normalization_artifacts"]),
            {"dataset_stats.json"},
        )
        self.assertEqual(
            report["text_embedding_cache"]["bundle_sha256"],
            preflight_offline_run.sha256_json(
                {
                    "artifacts": {
                        preflight_offline_run.WATER_PLANT_TEXT_CACHE_BASENAME: (
                            text_cache_hash
                        )
                    }
                }
            ),
        )

    def test_protocol_matrix_requires_identical_primary_samples(self) -> None:
        m, shuffled, targets, shuffled_targets = pair_control_fixture()
        manifests = {
            "B0": manifest(two_auxiliary_samples("success")),
            "B1": manifest(two_auxiliary_samples("failure")),
            "C": manifest(two_auxiliary_samples("failure")),
            "M": m,
            "M_PAIR_SHUFFLE": shuffled,
        }
        reports, identities = preflight_offline_run.validate_protocol_matrix(
            manifests,
            targets=targets,
            pair_shuffle_targets=shuffled_targets,
            include_pair_shuffle_control=True,
        )
        self.assertEqual(len(identities), 1)
        self.assertEqual(
            {report["primary_identity_sha256"] for report in reports.values()},
            {reports["B0"]["primary_identity_sha256"]},
        )

        drifted_rows = control_samples("failure")
        drifted_rows[0]["end_frame"] = 41
        manifests["B1"] = manifest(drifted_rows)
        with self.assertRaisesRegex(ValueError, "Primary sample identity mismatch"):
            preflight_offline_run.validate_protocol_matrix(
                manifests,
                targets=targets,
                pair_shuffle_targets=shuffled_targets,
                include_pair_shuffle_control=True,
            )

    def test_base_protocol_matrix_does_not_require_pair_shuffle(self) -> None:
        m, _, targets, _ = pair_control_fixture()
        manifests = {
            "B0": manifest(two_auxiliary_samples("success")),
            "B1": manifest(two_auxiliary_samples("failure")),
            "C": manifest(two_auxiliary_samples("failure")),
            "M": m,
        }
        reports, identities = preflight_offline_run.validate_protocol_matrix(
            manifests,
            targets=targets,
            pair_shuffle_targets=None,
        )
        self.assertEqual(set(reports), preflight_offline_run.BASE_VARIANTS)
        self.assertEqual(len(identities), 1)

    def test_optional_pair_shuffle_control_requires_fifth_artifacts(self) -> None:
        m, _, targets, _ = pair_control_fixture()
        manifests = {
            "B0": manifest(two_auxiliary_samples("success")),
            "B1": manifest(two_auxiliary_samples("failure")),
            "C": manifest(two_auxiliary_samples("failure")),
            "M": m,
        }
        with self.assertRaisesRegex(ValueError, "M_PAIR_SHUFFLE"):
            preflight_offline_run.validate_protocol_matrix(
                manifests,
                targets=targets,
                pair_shuffle_targets=None,
                include_pair_shuffle_control=True,
            )

    def test_protocol_matrix_requires_equal_auxiliary_budgets(self) -> None:
        b0_rows = two_auxiliary_samples("success")
        b0_rows.append(
            sample(
                "train-aux-extra",
                split="train",
                episode_index=2,
                outcome="success",
                role="auxiliary",
            )
        )
        m, shuffled, targets, shuffled_targets = pair_control_fixture()
        manifests = {
            "B0": manifest(b0_rows),
            "B1": manifest(two_auxiliary_samples("failure")),
            "C": manifest(two_auxiliary_samples("failure")),
            "M": m,
            "M_PAIR_SHUFFLE": shuffled,
        }
        with self.assertRaisesRegex(ValueError, "Auxiliary sample budget mismatch"):
            preflight_offline_run.validate_protocol_matrix(
                manifests,
                targets=targets,
                pair_shuffle_targets=shuffled_targets,
                include_pair_shuffle_control=True,
            )

    def test_resume_state_must_be_complete_directory(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            weight_file = root / "step.pt"
            weight_file.write_bytes(b"weights")
            with self.assertRaisesRegex(ValueError, "must be a directory"):
                preflight_offline_run.validate_resume_state(weight_file)

            state = root / "step_001500"
            state.mkdir()
            (state / "checkpoint_meta.json").write_text(
                json.dumps({"complete": True, "global_step": 1500}),
                encoding="utf-8",
            )
            (state / "trainer_state.json").write_text(
                json.dumps({"global_step": 1500}),
                encoding="utf-8",
            )
            write_deepspeed_state_payload(state)
            report = preflight_offline_run.validate_resume_state(state)
        self.assertEqual(report["global_step"], 1500)
        self.assertEqual(report["deepspeed_optimizer_ranks"], [0, 1, 2, 3])

    def test_resume_state_rejects_missing_deepspeed_rank(self) -> None:
        with TemporaryDirectory() as temporary:
            state = Path(temporary) / "step_001500"
            state.mkdir()
            (state / "checkpoint_meta.json").write_text(
                json.dumps({"complete": True, "global_step": 1500}),
                encoding="utf-8",
            )
            (state / "trainer_state.json").write_text(
                json.dumps({"global_step": 1500}),
                encoding="utf-8",
            )
            write_deepspeed_state_payload(state, ranks=(0, 1, 2))
            with self.assertRaisesRegex(ValueError, "ranks 0-3"):
                preflight_offline_run.validate_resume_state(state)

    def test_resume_state_is_bound_to_experiment_provenance(self) -> None:
        provenance = {
            "protocol": "fitwam_offline_self_improving_v1",
            "variant": "M",
            "manifest_sha256": "a" * 64,
            "code_snapshot_sha256": "b" * 64,
        }
        with TemporaryDirectory() as temporary:
            state = Path(temporary) / "step_001500"
            state.mkdir()
            (state / "checkpoint_meta.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "global_step": 1500,
                        "experiment_provenance": provenance,
                        "experiment_provenance_sha256": (
                            preflight_offline_run.sha256_json(provenance)
                        ),
                    }
                ),
                encoding="utf-8",
            )
            (state / "trainer_state.json").write_text(
                json.dumps({"global_step": 1500}),
                encoding="utf-8",
            )
            write_deepspeed_state_payload(state)

            report = preflight_offline_run.validate_resume_state(
                state, expected_provenance=provenance
            )
            self.assertEqual(report["experiment_provenance"], provenance)
            preflight_offline_run.validate_resume_state(
                state,
                expected_provenance=provenance,
                expected_global_step=1500,
            )
            with self.assertRaisesRegex(ValueError, "global_step must be 20"):
                preflight_offline_run.validate_resume_state(
                    state,
                    expected_provenance=provenance,
                    expected_global_step=20,
                )
            with self.assertRaisesRegex(
                ValueError, "does not match the selected variant"
            ):
                preflight_offline_run.validate_resume_state(
                    state,
                    expected_provenance={
                        **provenance,
                        "code_snapshot_sha256": "c" * 64,
                    },
                )

    def test_preformal_resume_request_is_only_smoke20_to_smoke500(self) -> None:
        preflight_offline_run.validate_execution_resume_request(
            execution_mode="smoke20",
            resume_state_dir=None,
            expected_resume_step=None,
        )
        preflight_offline_run.validate_execution_resume_request(
            execution_mode="smoke500",
            resume_state_dir=Path("/tmp/step_000020"),
            expected_resume_step=20,
        )
        with self.assertRaisesRegex(
            ValueError, "requires a complete step-20 state"
        ):
            preflight_offline_run.validate_execution_resume_request(
                execution_mode="smoke500",
                resume_state_dir=None,
                expected_resume_step=None,
            )
        with self.assertRaisesRegex(ValueError, "must start from INIT_WEIGHTS"):
            preflight_offline_run.validate_execution_resume_request(
                execution_mode="smoke20",
                resume_state_dir=Path("/tmp/step_000020"),
                expected_resume_step=20,
            )
        with self.assertRaisesRegex(ValueError, "complete step-20 state"):
            preflight_offline_run.validate_execution_resume_request(
                execution_mode="smoke500",
                resume_state_dir=Path("/tmp/step_000100"),
                expected_resume_step=100,
            )
        with self.assertRaisesRegex(ValueError, "must not set"):
            preflight_offline_run.validate_execution_resume_request(
                execution_mode="formal",
                resume_state_dir=Path("/tmp/step_001500"),
                expected_resume_step=1500,
            )

    def test_smoke_state_can_resume_smoke500_but_not_formal(self) -> None:
        formal_provenance = {
            "protocol": "fitwam_offline_self_improving_v1",
            "variant": "M",
            "manifest_sha256": "a" * 64,
            "code_snapshot_sha256": "b" * 64,
        }
        smoke_provenance = {
            **formal_provenance,
            "run_mode": "preformal_smoke",
        }
        with TemporaryDirectory() as temporary:
            state = Path(temporary) / "step_000020"
            state.mkdir()
            (state / "checkpoint_meta.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "global_step": 20,
                        "experiment_provenance": smoke_provenance,
                        "experiment_provenance_sha256": (
                            preflight_offline_run.sha256_json(smoke_provenance)
                        ),
                    }
                ),
                encoding="utf-8",
            )
            (state / "trainer_state.json").write_text(
                json.dumps({"global_step": 20}),
                encoding="utf-8",
            )
            write_deepspeed_state_payload(state)

            report = preflight_offline_run.validate_resume_state(
                state,
                expected_provenance=smoke_provenance,
                expected_global_step=20,
            )
            self.assertEqual(report["global_step"], 20)
            with self.assertRaisesRegex(
                ValueError, "does not match the selected variant"
            ):
                preflight_offline_run.validate_resume_state(
                    state,
                    expected_provenance=formal_provenance,
                )

    def test_resolved_config_binds_formal_contract(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            init_weights = root / "s0.pt"
            source_config = root / "source.yaml"
            manifest_path = root / "manifest.json"
            selection_manifest_path = root / "selection.json"
            pair_targets = root / "pairs.npz"
            dataset_root = root / "dataset"
            normalization_meta = root / "normalization_meta"
            text_cache_dir = root / "text_cache"
            (dataset_root / "meta").mkdir(parents=True)
            normalization_meta.mkdir()
            text_cache_dir.mkdir()
            (dataset_root / "meta" / "episodes.jsonl").write_text(
                json.dumps(
                    {
                        "episode_index": 0,
                        "tasks": [preflight_offline_run.WATER_PLANT_TASK],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (normalization_meta / "stats.json").write_bytes(b"stats")
            (normalization_meta / "modality.json").write_bytes(b"modality")
            text_cache_file = (
                text_cache_dir
                / preflight_offline_run.WATER_PLANT_TEXT_CACHE_BASENAME
            )
            text_cache_file.write_bytes(b"text-cache")
            for path, content in (
                (init_weights, b"s0"),
                (source_config, b"source"),
                (manifest_path, b"manifest"),
                (selection_manifest_path, b"selection"),
                (pair_targets, b"pairs"),
            ):
                path.write_bytes(content)
            init_hash = preflight_offline_run.sha256_file(init_weights)
            common_weights = root / "common-init.pt"
            common_proof = root / "common-init.proof.json"
            common_config = root / "common-init.yaml"
            common_weights.write_bytes(b"complete-common-init")
            common_proof.write_text("{}\n", encoding="utf-8")
            common_config.write_text("seed: 42\n", encoding="utf-8")
            common_hash = preflight_offline_run.sha256_file(common_weights)
            common_report = {
                "weights": preflight_offline_run.artifact_record(common_weights),
                "proof": preflight_offline_run.artifact_record(common_proof),
                "config": preflight_offline_run.artifact_record(common_config),
                "baseline": preflight_offline_run.artifact_record(init_weights),
                "payload_proof_sha256": "e" * 64,
                "seed": 42,
            }
            source_hash = preflight_offline_run.sha256_file(source_config)
            manifest_hash = preflight_offline_run.sha256_file(manifest_path)
            selection_hash = preflight_offline_run.sha256_file(
                selection_manifest_path
            )
            pair_hash = preflight_offline_run.sha256_file(pair_targets)
            teacher_hash = "a" * 64
            code_snapshot_hash = "b" * 64
            normalization_bundle_hash = preflight_offline_run.sha256_json(
                {
                    "kind": "meta",
                    "artifacts": {
                        "modality.json": preflight_offline_run.sha256_file(
                            normalization_meta / "modality.json"
                        ),
                        "stats.json": preflight_offline_run.sha256_file(
                            normalization_meta / "stats.json"
                        ),
                    },
                }
            )
            text_cache_hash = preflight_offline_run.sha256_json(
                {
                    "artifacts": {
                        preflight_offline_run.WATER_PLANT_TEXT_CACHE_BASENAME: (
                            preflight_offline_run.sha256_file(
                                text_cache_file
                            )
                        )
                    }
                }
            )
            payload = {
                "batch_size": 4,
                "role_balanced_sampling": {
                    "enabled": True,
                    "primary_per_batch": 2,
                },
                "learning_rate": 1.0e-4,
                "max_steps": 6500,
                "gradient_accumulation_steps": 1,
                "mixed_precision": "bf16",
                "seed": 42,
                "eval_seed": 20260717,
                "eval_every": 500,
                "best_metric": "val_base_loss",
                "save_weights_every": 0,
                "save_weight_steps": [500, 1000, 3000, 5000, 6000, 6500],
                "save_state_every": 1500,
                "state_keep_last": 1,
                "resume": str(init_weights),
                "model": {
                    "offline_steer": {
                        "enabled": True,
                        "pair_loss_weight": 0.1,
                        "pair_loss_warmup_steps": 500,
                    }
                },
                "data": {
                    "train": {
                        "dataset_dirs": [str(dataset_root)],
                        "manifest_path": str(manifest_path),
                        "pair_targets_path": str(pair_targets),
                        "expected_teacher_sha256": teacher_hash,
                        "pretrained_norm_stats": None,
                        "context_len": 128,
                        "text_embedding_cache_dir": str(text_cache_dir),
                        "processor": {
                            "norm_stats_source": "meta",
                            "norm_stats_meta_dir": str(normalization_meta),
                        },
                    },
                    "val": {
                        "dataset_dirs": [str(dataset_root)],
                        "manifest_path": str(selection_manifest_path),
                        "pair_targets_path": None,
                        "expected_teacher_sha256": None,
                        "pretrained_norm_stats": None,
                        "context_len": 128,
                        "text_embedding_cache_dir": str(text_cache_dir),
                        "processor": {
                            "norm_stats_source": "meta",
                            "norm_stats_meta_dir": str(normalization_meta),
                        },
                    },
                },
                "experiment_provenance": {
                    "protocol": preflight_offline_run.PROTOCOL_NAME,
                    "variant": "M",
                    "source_checkpoint_sha256": init_hash,
                    "source_config_sha256": source_hash,
                    "manifest_sha256": manifest_hash,
                    "selection_manifest_sha256": selection_hash,
                    "pair_targets_sha256": pair_hash,
                    "teacher_sha256": teacher_hash,
                    "code_snapshot_sha256": code_snapshot_hash,
                    "normalization_kind": "meta",
                    "normalization_bundle_sha256": normalization_bundle_hash,
                    "text_embedding_cache_sha256": text_cache_hash,
                },
                "wandb": {"enabled": True, "mode": "online"},
            }
            validation_kwargs = {
                "variant": "M",
                "manifest_path": manifest_path,
                "selection_manifest_path": selection_manifest_path,
                "init_weights": init_weights,
                "init_weights_sha256": init_hash,
                "source_config_sha256": source_hash,
                "manifest_sha256": manifest_hash,
                "selection_manifest_sha256": selection_hash,
                "pair_targets": pair_targets,
                "pair_targets_sha256": pair_hash,
                "teacher_sha256": teacher_hash,
                "code_snapshot_sha256": code_snapshot_hash,
                "expected_dataset_roots": [str(dataset_root)],
                "expected_normalization_kind": "meta",
                "expected_normalization_bundle_sha256": (
                    normalization_bundle_hash
                ),
                "expected_text_cache_sha256": text_cache_hash,
                "referenced_samples": [
                    {
                        "dataset_root": str(dataset_root),
                        "episode_index": 0,
                        "task": preflight_offline_run.WATER_PLANT_TASK,
                    }
                ],
            }
            formal_payload = json.loads(json.dumps(payload))
            config = root / "M.yaml"
            config.write_text(json.dumps(payload), encoding="utf-8")
            report = preflight_offline_run.validate_resolved_config(
                config,
                **validation_kwargs,
            )
            self.assertEqual(report["variant"], "M")

            strict_payload = json.loads(json.dumps(formal_payload))
            strict_payload["resume"] = str(common_weights)
            strict_payload["experiment_provenance"].update(
                {
                    "source_checkpoint_sha256": common_hash,
                    "comparison_mode": preflight_offline_run.STRICT_COMMON_INIT_MODE,
                    "common_init_checkpoint_sha256": common_hash,
                    "common_init_proof_file_sha256": common_report["proof"]["sha256"],
                    "common_init_payload_proof_sha256": common_report[
                        "payload_proof_sha256"
                    ],
                    "common_init_baseline_sha256": init_hash,
                    "common_init_config_sha256": common_report["config"]["sha256"],
                    "common_init_seed": 42,
                }
            )
            config.write_text(json.dumps(strict_payload), encoding="utf-8")
            strict_kwargs = dict(validation_kwargs)
            strict_kwargs.update(
                {
                    "init_weights": common_weights,
                    "init_weights_sha256": common_hash,
                    "strict_common_init": True,
                    "common_init_report": common_report,
                }
            )
            strict_m_report = preflight_offline_run.validate_resolved_config(
                config,
                **strict_kwargs,
            )
            self.assertEqual(strict_m_report["initialization_mode"], "common_init")

            shuffle_payload = json.loads(json.dumps(strict_payload))
            shuffle_payload["experiment_provenance"]["variant"] = "M_PAIR_SHUFFLE"
            shuffle_payload["experiment_provenance"].update(
                {
                    "pair_shuffle_proof_sha256": "f" * 64,
                    "pair_shuffle_seed": 20260721,
                    "pair_shuffle_source_pair_targets_sha256": pair_hash,
                }
            )
            config.write_text(json.dumps(shuffle_payload), encoding="utf-8")
            shuffle_kwargs = dict(strict_kwargs)
            shuffle_kwargs.update(
                {
                    "variant": "M_PAIR_SHUFFLE",
                    "pair_shuffle_proof_sha256": "f" * 64,
                    "pair_shuffle_seed": 20260721,
                    "pair_shuffle_source_pair_targets_sha256": pair_hash,
                }
            )
            shuffle_report = preflight_offline_run.validate_resolved_config(
                config,
                **shuffle_kwargs,
            )
            self.assertEqual(shuffle_report["variant"], "M_PAIR_SHUFFLE")
            self.assertEqual(
                shuffle_report["init_weights"],
                strict_m_report["init_weights"],
            )

            smoke_payload = json.loads(json.dumps(payload))
            smoke_payload["max_steps"] = 20
            smoke_payload["eval_every"] = 10
            smoke_payload["save_weights_every"] = 10
            smoke_payload["save_weight_steps"] = None
            smoke_payload["save_state_every"] = 20
            smoke_payload["lr_scheduler_total_steps"] = 500
            smoke_payload["experiment_provenance"][
                "run_mode"
            ] = "preformal_smoke"
            config.write_text(json.dumps(smoke_payload), encoding="utf-8")
            smoke_report = preflight_offline_run.validate_resolved_config(
                config,
                execution_mode="smoke20",
                **validation_kwargs,
            )
            self.assertEqual(smoke_report["execution_mode"], "smoke20")
            formal_with_smoke_provenance = json.loads(
                json.dumps(smoke_payload)
            )
            formal_with_smoke_provenance["max_steps"] = 6500
            formal_with_smoke_provenance["eval_every"] = 500
            formal_with_smoke_provenance["save_weights_every"] = 0
            formal_with_smoke_provenance["save_weight_steps"] = [
                500,
                1000,
                3000,
                5000,
                6000,
                6500,
            ]
            formal_with_smoke_provenance["save_state_every"] = 1500
            formal_with_smoke_provenance["lr_scheduler_total_steps"] = None
            config.write_text(
                json.dumps(formal_with_smoke_provenance),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must not contain run_mode"):
                preflight_offline_run.validate_resolved_config(
                    config,
                    execution_mode="formal",
                    **validation_kwargs,
                )

            payload = json.loads(json.dumps(formal_payload))
            payload["max_steps"] = 6501
            config.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "max_steps"):
                preflight_offline_run.validate_resolved_config(
                    config,
                    **validation_kwargs,
                )

            payload["max_steps"] = 6500
            payload["experiment_provenance"]["code_snapshot_sha256"] = "c" * 64
            config.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "code_snapshot_sha256"):
                preflight_offline_run.validate_resolved_config(
                    config,
                    **validation_kwargs,
                )

    def test_protocol_bundle_is_immutable(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "protocol.json"
            payload = {
                "format": preflight_offline_run.BUNDLE_FORMAT,
                "version": preflight_offline_run.BUNDLE_VERSION,
                "artifact": {"sha256": "1" * 64},
            }
            payload["bundle_sha256"] = preflight_offline_run.sha256_json(payload)
            self.assertEqual(
                preflight_offline_run.write_or_validate_protocol_bundle(
                    path, payload
                ),
                "created",
            )
            self.assertEqual(
                preflight_offline_run.write_or_validate_protocol_bundle(
                    path, payload
                ),
                "validated",
            )
            changed = dict(payload)
            changed["artifact"] = {"sha256": "2" * 64}
            changed.pop("bundle_sha256")
            changed["bundle_sha256"] = preflight_offline_run.sha256_json(changed)
            with self.assertRaisesRegex(ValueError, "does not match"):
                preflight_offline_run.write_or_validate_protocol_bundle(
                    path, changed
                )


if __name__ == "__main__":
    unittest.main()
