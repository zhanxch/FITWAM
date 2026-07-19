from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT / "scripts" / "everobot" / "train_offline_steer_teacher.py"
)
SPEC = importlib.util.spec_from_file_location(
    "train_offline_steer_teacher", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
teacher_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = teacher_script
SPEC.loader.exec_module(teacher_script)

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    np = None
    pa = None
    pq = None


def episode(
    dataset_id: str,
    index: int,
    outcome: str,
    root: str = "/synthetic",
    split: str = "train",
) -> dict[str, object]:
    return {
        "episode_id": f"{dataset_id}:ep:{index}",
        "dataset_id": dataset_id,
        "dataset_root": root,
        "episode_index": index,
        "episode_outcome": outcome,
        "length": 20,
        "split": split,
    }


def event(
    event_id: str,
    episode_row: dict[str, object],
    outcome: str,
    *,
    start: int = 2,
    end: int = 10,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "episode_id": episode_row["episode_id"],
        "dataset_id": episode_row["dataset_id"],
        "episode_index": episode_row["episode_index"],
        "event_outcome": outcome,
        "start_frame": start,
        "end_frame": end,
        "split": episode_row.get("split", "train"),
    }


def pair(
    pair_id: str,
    success_event_id: str,
    failure_event_id: str,
    split: str = "train",
) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "success_event_id": success_event_id,
        "failure_event_id": failure_event_id,
        "pair_weight": 0.8,
        "split": split,
    }


def metric_row(epoch: int, offset: float = 0.0) -> dict[str, float | int]:
    return {
        "epoch": epoch,
        "train_loss": 1.0 + offset,
        "val_loss": 2.0 + offset,
        "val_positive_cosine": 0.8 + offset,
        "val_negative_cosine": 0.2 + offset,
        "val_cosine_gap": 0.6 + offset,
        "val_embedding_variance": 0.1 + offset,
        "val_top1_paired_retrieval_accuracy": 0.5 + offset,
        "learning_rate": 3e-4,
    }


class OfflineSteerTeacherPureFunctionTest(unittest.TestCase):
    def test_default_temperature_is_point_zero_seven(self) -> None:
        args = teacher_script.parse_args(
            [
                "--eve-root",
                "/synthetic/eve",
                "--output-dir",
                "/synthetic/output",
            ]
        )
        self.assertEqual(args.temperature, 0.07)
        self.assertFalse(args.resume)

    def test_resume_flag_is_boolean(self) -> None:
        args = teacher_script.parse_args(
            [
                "--eve-root",
                "/synthetic/eve",
                "--output-dir",
                "/synthetic/output",
                "--resume",
            ]
        )
        self.assertTrue(args.resume)

    def test_protocol_rejects_parameter_and_ledger_drift(self) -> None:
        args = teacher_script.parse_args(
            [
                "--eve-root",
                "/synthetic/eve",
                "--output-dir",
                "/synthetic/output",
                "--epochs",
                "10",
            ]
        )
        protocol = teacher_script._protocol_payload(
            args=args,
            teacher_config=teacher_script.TeacherConfig(),
            episode_ledger_sha256="a" * 64,
            event_ledger_sha256="b" * 64,
            pair_ledger_sha256="c" * 64,
            split_sha256="d" * 64,
            action_mean=[0.0] * 22,
            action_std=[1.0] * 22,
            resolved_device="cpu",
        )
        protocol_hash = teacher_script._sha256_json(protocol)
        stored_config = {
            "protocol": protocol,
            "protocol_sha256": protocol_hash,
        }
        config_hash = teacher_script._sha256_json(stored_config)
        stored_config["config_sha256"] = config_hash
        checkpoint = {
            "protocol_sha256": protocol_hash,
            "config_sha256": config_hash,
            "artifact_hashes": {
                "episode_ledger_sha256": "a" * 64,
                "event_ledger_sha256": "b" * 64,
                "pair_ledger_sha256": "c" * 64,
                "split_sha256": "d" * 64,
            },
        }
        self.assertEqual(
            teacher_script._require_protocol_match(
                stored_config=stored_config,
                checkpoint=checkpoint,
                current_protocol=protocol,
                current_config_sha256=config_hash,
            ),
            protocol_hash,
        )

        changed_parameter = json.loads(json.dumps(protocol))
        changed_parameter["optimization"]["batch_size"] += 1
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            teacher_script._require_protocol_match(
                stored_config=stored_config,
                checkpoint=checkpoint,
                current_protocol=changed_parameter,
                current_config_sha256=config_hash,
            )

        changed_ledger = json.loads(json.dumps(protocol))
        changed_ledger["ledgers"]["pair_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            teacher_script._require_protocol_match(
                stored_config=stored_config,
                checkpoint=checkpoint,
                current_protocol=changed_ledger,
                current_config_sha256=config_hash,
            )

    def test_metric_resume_repairs_final_write_without_duplicate_epochs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            jsonl_path = root / "metrics.jsonl"
            csv_path = root / "metrics.csv"
            rows = [metric_row(1), metric_row(2, 0.1)]
            teacher_script._write_jsonl_atomic(jsonl_path, rows[:1])
            teacher_script._write_metrics_csv_atomic(csv_path, rows)
            reconciled = teacher_script._reconcile_metrics_for_resume(
                metrics_jsonl=jsonl_path,
                metrics_csv=csv_path,
                checkpoint_epoch=2,
                checkpoint_metric_row=rows[1],
            )
            self.assertEqual([row["epoch"] for row in reconciled], [1, 2])

            repeated = teacher_script._reconcile_metrics_for_resume(
                metrics_jsonl=jsonl_path,
                metrics_csv=csv_path,
                checkpoint_epoch=2,
                checkpoint_metric_row=rows[1],
            )
            self.assertEqual([row["epoch"] for row in repeated], [1, 2])
            self.assertEqual(
                [row["epoch"] for row in teacher_script.read_jsonl(jsonl_path)],
                [1, 2],
            )

    def test_metric_resume_rejects_noncontiguous_history(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            jsonl_path = root / "metrics.jsonl"
            csv_path = root / "metrics.csv"
            rows = [metric_row(1), metric_row(3, 0.1)]
            teacher_script._write_jsonl_atomic(jsonl_path, rows)
            teacher_script._write_metrics_csv_atomic(csv_path, rows)
            with self.assertRaisesRegex(ValueError, "contiguous"):
                teacher_script._reconcile_metrics_for_resume(
                    metrics_jsonl=jsonl_path,
                    metrics_csv=csv_path,
                    checkpoint_epoch=2,
                    checkpoint_metric_row=metric_row(2, 0.2),
                )

    def test_core_interval_is_preferred_and_validated(self) -> None:
        row = {
            "event_id": "event",
            "start_frame": 2,
            "end_frame": 20,
            "core_start_frame": 5,
            "core_end_frame": 12,
        }
        self.assertEqual(
            teacher_script.resolve_event_interval(row, prefer_core=True),
            (5, 12),
        )
        self.assertEqual(
            teacher_script.resolve_event_interval(row, prefer_core=False),
            (2, 20),
        )
        with self.assertRaisesRegex(ValueError, "both core"):
            teacher_script.resolve_event_interval(
                {**row, "core_end_frame": None}, prefer_core=True
            )

    def test_ledgers_join_and_validate_outcomes(self) -> None:
        success_episode = episode("success", 0, "success")
        failure_episode = episode("failure", 0, "failure")
        success_event = event("success-event", success_episode, "unknown")
        failure_event = event("failure-event", failure_episode, "failure")
        records = teacher_script.build_pair_records(
            [success_episode, failure_episode],
            [success_event, failure_event],
            [pair("pair-0", "success-event", "failure-event")],
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["success_episode_key"], "success:episode:0")
        self.assertEqual(records[0]["failure_episode_key"], "failure:episode:0")

        with self.assertRaisesRegex(ValueError, "not marked failure"):
            teacher_script.build_pair_records(
                [success_episode, {**failure_episode, "episode_outcome": "success"}],
                [success_event, failure_event],
                [pair("pair-0", "success-event", "failure-event")],
            )

    def test_split_is_episode_disjoint_with_reused_events(self) -> None:
        records: list[dict[str, object]] = []
        for component in range(4):
            for reuse in range(2):
                records.append(
                    {
                        "pair_id": f"pair-{component}-{reuse}",
                        "success_episode_key": f"success-{component}",
                        "failure_episode_key": (
                            f"failure-{component}"
                            if reuse == 0
                            else f"failure-{component}-extra"
                        ),
                    }
                )
        train, validation = teacher_script.split_episode_disjoint(
            records, val_fraction=0.25, seed=7
        )
        train_episodes = {
            row[key]
            for row in train
            for key in ("success_episode_key", "failure_episode_key")
        }
        val_episodes = {
            row[key]
            for row in validation
            for key in ("success_episode_key", "failure_episode_key")
        }
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertFalse(train_episodes & val_episodes)
        repeat = teacher_script.split_episode_disjoint(
            records, val_fraction=0.25, seed=7
        )
        self.assertEqual(
            [row["pair_id"] for row in validation],
            [row["pair_id"] for row in repeat[1]],
        )

    def test_declared_split_is_authoritative_and_episode_disjoint(self) -> None:
        train_success = episode("success-train", 0, "success", split="train")
        train_failure = episode("failure-train", 0, "failure", split="train")
        val_success = episode("success-val", 0, "success", split="val")
        val_failure = episode("failure-val", 0, "failure", split="val")
        events = [
            event("train-success", train_success, "success"),
            event("train-failure", train_failure, "failure"),
            event("val-success", val_success, "success"),
            event("val-failure", val_failure, "failure"),
        ]
        records = teacher_script.build_pair_records(
            [train_success, train_failure, val_success, val_failure],
            events,
            [
                pair(
                    "train-pair",
                    "train-success",
                    "train-failure",
                    split="train",
                ),
                pair(
                    "val-pair",
                    "val-success",
                    "val-failure",
                    split="val",
                ),
            ],
        )

        train, validation = teacher_script.split_by_declared_ledger(records)
        self.assertEqual([row["pair_id"] for row in train], ["train-pair"])
        self.assertEqual([row["pair_id"] for row in validation], ["val-pair"])

    def test_declared_split_rejects_cross_split_pair(self) -> None:
        success_episode = episode("success", 0, "success", split="train")
        failure_episode = episode("failure", 0, "failure", split="val")
        records = teacher_script.build_pair_records(
            [success_episode, failure_episode],
            [
                event("success-event", success_episode, "success"),
                event("failure-event", failure_episode, "failure"),
            ],
            [
                pair(
                    "cross-split",
                    "success-event",
                    "failure-event",
                    split="train",
                )
            ],
        )
        with self.assertRaisesRegex(ValueError, "crosses event splits"):
            teacher_script.split_by_declared_ledger(records)

    def test_single_connected_component_cannot_leak_into_validation(self) -> None:
        records = [
            {
                "pair_id": "pair-0",
                "success_episode_key": "success",
                "failure_episode_key": "failure-a",
            },
            {
                "pair_id": "pair-1",
                "success_episode_key": "success",
                "failure_episode_key": "failure-b",
            },
        ]
        with self.assertRaisesRegex(ValueError, "connected episode component"):
            teacher_script.split_episode_disjoint(
                records, val_fraction=0.5, seed=1
            )

    def test_action_statistics_use_all_valid_train_frames(self) -> None:
        samples = [
            {
                "success_actions": [[0.0, 2.0], [2.0, 4.0]],
                "failure_actions": [[4.0, 6.0], [6.0, 8.0]],
            }
        ]
        mean, std = teacher_script.compute_action_statistics(
            samples, action_dim=2
        )
        self.assertEqual(mean, [3.0, 5.0])
        self.assertAlmostEqual(std[0], 5.0**0.5)
        self.assertAlmostEqual(std[1], 5.0**0.5)

    def test_locate_episode_parquet_uses_lerobot_template(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "chunks_size": 100,
                        "data_path": (
                            "data/chunk-{episode_chunk:03d}/"
                            "episode_{episode_index:06d}.parquet"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            parquet_path = root / "data" / "chunk-001" / "episode_000123.parquet"
            parquet_path.parent.mkdir(parents=True)
            parquet_path.write_bytes(b"synthetic")
            located = teacher_script.locate_episode_parquet(
                {
                    "dataset_root": str(root),
                    "episode_index": 123,
                }
            )
            self.assertEqual(located, parquet_path.resolve())


@unittest.skipUnless(pa is not None, "PyArrow is not installed locally")
class OfflineSteerTeacherParquetTest(unittest.TestCase):
    def test_event_action_reader_uses_half_open_core_interval(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            parquet_path = root / "data" / "chunk-000" / "episode_000000.parquet"
            parquet_path.parent.mkdir(parents=True)
            (root / "meta").mkdir()
            (root / "meta" / "info.json").write_text(
                json.dumps(
                    {
                        "chunks_size": 1000,
                        "data_path": (
                            "data/chunk-{episode_chunk:03d}/"
                            "episode_{episode_index:06d}.parquet"
                        ),
                    }
                ),
                encoding="utf-8",
            )
            actions = [
                [float(frame * 100 + dimension) for dimension in range(22)]
                for frame in range(8)
            ]
            pq.write_table(pa.table({"action": actions}), parquet_path)
            trajectory = teacher_script.read_event_actions(
                {
                    "event_id": "event",
                    "start_frame": 0,
                    "end_frame": 8,
                    "core_start_frame": 2,
                    "core_end_frame": 5,
                },
                {
                    "dataset_root": str(root),
                    "episode_index": 0,
                },
            )
            self.assertEqual(trajectory, actions[2:5])


@unittest.skipUnless(torch is not None, "PyTorch is not installed locally")
class OfflineSteerTeacherTorchTest(unittest.TestCase):
    def test_checkpoint_restores_training_and_all_rng_state(self) -> None:
        random.seed(11)
        torch.manual_seed(11)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(11)
        loader_generator = torch.Generator().manual_seed(29)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=4
        )
        loss = model(torch.ones(2, 3)).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        checkpoint = {
            "checkpoint_schema_version": teacher_script.CHECKPOINT_SCHEMA_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": 1,
            "val_loss": 0.7,
            "val_representation_metrics": {},
            "best_val_loss": 0.7,
            "best_epoch": 1,
            "teacher_config": {},
            "action_mean": [],
            "action_std": [],
            "protocol_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "artifact_hashes": {},
            "metric_row": metric_row(1),
            "rng_state": teacher_script._capture_rng_state(loader_generator),
            "started_at": "2026-07-17T00:00:00+00:00",
            "wandb_run_id": None,
        }

        expected_python = random.random()
        expected_torch = torch.rand(4)
        expected_loader = torch.rand(4, generator=loader_generator)
        random.seed(99)
        torch.manual_seed(99)
        loader_generator.manual_seed(99)

        restored_model = torch.nn.Linear(3, 2)
        restored_optimizer = torch.optim.AdamW(
            restored_model.parameters(), lr=0.5
        )
        restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            restored_optimizer, T_max=4
        )
        restored = teacher_script._restore_training_checkpoint(
            checkpoint=checkpoint,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            loader_generator=loader_generator,
        )
        self.assertEqual(restored, (1, 0.7, 1, checkpoint["started_at"], None))
        for expected, actual in zip(
            model.parameters(), restored_model.parameters()
        ):
            self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(
            restored_scheduler.state_dict(), scheduler.state_dict()
        )
        self.assertEqual(random.random(), expected_python)
        self.assertTrue(torch.equal(torch.rand(4), expected_torch))
        self.assertTrue(
            torch.equal(
                torch.rand(4, generator=loader_generator),
                expected_loader,
            )
        )

    def test_existing_best_checkpoint_is_retained(self) -> None:
        loader_generator = torch.Generator().manual_seed(4)
        checkpoint = {
            "checkpoint_schema_version": teacher_script.CHECKPOINT_SCHEMA_VERSION,
            "model": {"weight": torch.tensor([1.0])},
            "optimizer": {},
            "scheduler": {},
            "epoch": 2,
            "val_loss": 0.5,
            "val_representation_metrics": {},
            "best_val_loss": 0.5,
            "best_epoch": 2,
            "teacher_config": {},
            "action_mean": [],
            "action_std": [],
            "protocol_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "artifact_hashes": {},
            "metric_row": metric_row(2),
            "rng_state": teacher_script._capture_rng_state(loader_generator),
            "started_at": "2026-07-17T00:00:00+00:00",
            "wandb_run_id": None,
        }
        with TemporaryDirectory() as temporary:
            best_path = Path(temporary) / "best_teacher.pt"
            teacher_script._torch_save_atomic(best_path, checkpoint)
            before = teacher_script.sha256_file(best_path)
            teacher_script._ensure_best_checkpoint(
                best_path=best_path,
                last_checkpoint={
                    **checkpoint,
                    "epoch": 3,
                    "val_loss": 0.8,
                    "metric_row": metric_row(3),
                },
            )
            self.assertEqual(teacher_script.sha256_file(best_path), before)

    def test_missing_best_is_recovered_only_from_matching_last_epoch(self) -> None:
        loader_generator = torch.Generator().manual_seed(4)
        checkpoint = {
            "checkpoint_schema_version": teacher_script.CHECKPOINT_SCHEMA_VERSION,
            "model": {"weight": torch.tensor([1.0])},
            "optimizer": {},
            "scheduler": {},
            "epoch": 2,
            "val_loss": 0.5,
            "val_representation_metrics": {},
            "best_val_loss": 0.5,
            "best_epoch": 2,
            "teacher_config": {},
            "action_mean": [],
            "action_std": [],
            "protocol_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "artifact_hashes": {},
            "metric_row": metric_row(2),
            "rng_state": teacher_script._capture_rng_state(loader_generator),
            "started_at": "2026-07-17T00:00:00+00:00",
            "wandb_run_id": None,
        }
        with TemporaryDirectory() as temporary:
            best_path = Path(temporary) / "best_teacher.pt"
            teacher_script._ensure_best_checkpoint(
                best_path=best_path,
                last_checkpoint=checkpoint,
            )
            recovered = torch.load(
                best_path, map_location="cpu", weights_only=False
            )
            self.assertEqual(recovered["epoch"], 2)
            best_path.unlink()
            with self.assertRaisesRegex(ValueError, "missing or inconsistent"):
                teacher_script._ensure_best_checkpoint(
                    best_path=best_path,
                    last_checkpoint={
                        **checkpoint,
                        "epoch": 3,
                        "val_loss": 0.8,
                        "metric_row": metric_row(3),
                    },
                )

    def test_representation_metrics_are_weighted_and_detached(self) -> None:
        success_one = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]], requires_grad=True
        )
        success_two = success_one.detach().clone().requires_grad_(True)
        failure_one = (-success_one.detach().clone()).requires_grad_(True)
        failure_two = failure_one.detach().clone().requires_grad_(True)
        metrics = teacher_script.representation_metrics(
            success_one,
            success_two,
            failure_one,
            failure_two,
            torch.tensor([1.0, 3.0]),
        )
        self.assertAlmostEqual(metrics["positive_cosine"], 1.0)
        self.assertAlmostEqual(metrics["negative_cosine"], -1.0)
        self.assertAlmostEqual(metrics["cosine_gap"], 2.0)
        self.assertAlmostEqual(metrics["embedding_variance"], 0.5)
        self.assertAlmostEqual(
            metrics["top1_paired_retrieval_accuracy"], 1.0
        )
        self.assertIsNone(success_one.grad)
        self.assertIsNone(success_two.grad)
        self.assertIsNone(failure_one.grad)
        self.assertIsNone(failure_two.grad)

    def test_representation_metrics_handle_empty_and_single_pair(self) -> None:
        empty = torch.empty(0, 3)
        empty_metrics = teacher_script.representation_metrics(
            empty,
            empty,
            empty,
            empty,
            torch.empty(0),
        )
        self.assertEqual(set(empty_metrics.values()), {0.0})

        success = torch.tensor([[1.0, 0.0]])
        failure = torch.tensor([[0.0, 1.0]])
        single_metrics = teacher_script.representation_metrics(
            success,
            success,
            failure,
            failure,
            torch.tensor([0.0]),
        )
        self.assertTrue(all(
            math.isfinite(value) for value in single_metrics.values()
        ))
        self.assertEqual(set(single_metrics.values()), {0.0})

    def test_synthetic_teacher_optimizer_step(self) -> None:
        from fastwam.models.wan22.offline_steer import TrajectoryTeacher

        batch = [
            {
                "pair_id": "p0",
                "success_event_id": "s0",
                "failure_event_id": "f0",
                "pair_weight": 1.0,
                "success_actions": [[0.0, 0.2], [0.1, 0.3], [0.2, 0.4]],
                "failure_actions": [[0.0, -0.2], [-0.1, -0.3]],
            },
            {
                "pair_id": "p1",
                "success_event_id": "s1",
                "failure_event_id": "f1",
                "pair_weight": 0.7,
                "success_actions": [[0.4, 0.2], [0.5, 0.3]],
                "failure_actions": [[-0.4, -0.2], [-0.5, -0.3], [-0.6, -0.4]],
            },
        ]
        collated = teacher_script.collate_action_pairs(
            batch,
            action_mean=[0.0, 0.0],
            action_std=[1.0, 1.0],
            max_steps=None,
        )
        model = TrajectoryTeacher(
            action_dim=2,
            hidden_dim=8,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        loss = teacher_script._teacher_loss(
            model,
            collated,
            mask_probability=0.1,
            jitter_std=0.01,
            temperature=0.2,
            hard_negative_bias=0.5,
            generator=torch.Generator().manual_seed(4),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradient_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_norm, 0.0)
        optimizer.step()

    def test_variable_length_collate_builds_padding_masks(self) -> None:
        batch = [
            {
                "pair_id": "p0",
                "success_event_id": "s0",
                "failure_event_id": "f0",
                "pair_weight": 1.0,
                "success_actions": [[0.0, 1.0], [1.0, 2.0]],
                "failure_actions": [[2.0, 3.0]],
            },
            {
                "pair_id": "p1",
                "success_event_id": "s1",
                "failure_event_id": "f1",
                "pair_weight": 0.5,
                "success_actions": [[3.0, 4.0]],
                "failure_actions": [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
            },
        ]
        result = teacher_script.collate_action_pairs(
            batch,
            action_mean=[0.0, 0.0],
            action_std=[1.0, 1.0],
            max_steps=None,
        )
        self.assertEqual(tuple(result["success_actions"].shape), (2, 2, 2))
        self.assertEqual(tuple(result["failure_actions"].shape), (2, 3, 2))
        self.assertEqual(
            result["success_mask"].tolist(),
            [[True, True], [True, False]],
        )
        self.assertEqual(
            result["failure_mask"].tolist(),
            [[True, False, False], [True, True, True]],
        )

    def test_augmented_infonce_has_gradients_and_uses_hard_negative(self) -> None:
        torch.manual_seed(2)
        success_one = torch.randn(3, 8, requires_grad=True)
        success_two = success_one.detach().clone().requires_grad_(True)
        failure_one = torch.randn(3, 8, requires_grad=True)
        failure_two = failure_one.detach().clone().requires_grad_(True)
        weights = torch.tensor([1.0, 0.7, 0.5])
        without_bias = teacher_script.paired_augmented_infonce(
            success_one,
            success_two,
            failure_one,
            failure_two,
            weights,
            temperature=0.2,
            hard_negative_bias=0.0,
        )
        with_bias = teacher_script.paired_augmented_infonce(
            success_one,
            success_two,
            failure_one,
            failure_two,
            weights,
            temperature=0.2,
            hard_negative_bias=0.5,
        )
        self.assertTrue(torch.isfinite(with_bias))
        self.assertGreaterEqual(float(with_bias), float(without_bias))
        with_bias.backward()
        self.assertTrue(torch.isfinite(success_one.grad).all())
        self.assertGreater(float(success_one.grad.abs().sum()), 0.0)

    def test_mask_augmentation_preserves_at_least_one_valid_step(self) -> None:
        actions = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        mask = torch.tensor([[True, False, False], [True, True, True]])
        augmented, augmented_mask = teacher_script.augment_trajectory(
            actions,
            mask,
            mask_probability=0.999,
            jitter_std=0.0,
            generator=torch.Generator().manual_seed(0),
        )
        self.assertTrue(augmented_mask.any(dim=1).all())
        self.assertTrue((augmented[~augmented_mask] == 0).all())

    @unittest.skipUnless(
        pa is not None and np is not None,
        "PyArrow and NumPy are required for target export",
    )
    def test_pair_target_export_contains_required_fields(self) -> None:
        from fastwam.models.wan22.offline_steer import TrajectoryTeacher

        samples = [
            {
                "pair_id": "pair-0",
                "success_event_id": "success-0",
                "failure_event_id": "failure-0",
                "pair_weight": 0.8,
                "success_actions": [[0.0, 0.1], [0.2, 0.3]],
                "failure_actions": [[-0.1, -0.2], [-0.3, -0.4]],
            }
        ]
        model = TrajectoryTeacher(
            action_dim=2,
            hidden_dim=8,
            embedding_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            rows = teacher_script.export_pair_targets(
                model=model,
                samples_by_split={"train": samples},
                action_mean=[0.0, 0.0],
                action_std=[1.0, 1.0],
                max_steps=None,
                batch_size=1,
                device=torch.device("cpu"),
                teacher_hash="a" * 64,
                output_dir=output_dir,
                export_format="both",
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]["z_plus"]), 8)
            self.assertEqual(len(rows[0]["z_minus"]), 8)
            parquet_row = pq.read_table(
                output_dir / "pair_targets.parquet"
            ).to_pylist()[0]
            self.assertEqual(parquet_row["pair_id"], "pair-0")
            self.assertEqual(parquet_row["success_event_id"], "success-0")
            self.assertEqual(parquet_row["failure_event_id"], "failure-0")
            self.assertEqual(parquet_row["teacher_sha256"], "a" * 64)
            with np.load(output_dir / "pair_targets.npz") as archive:
                self.assertEqual(archive["z_plus"].shape, (1, 8))
                self.assertEqual(archive["z_minus"].shape, (1, 8))
                self.assertEqual(archive["pair_id"].tolist(), ["pair-0"])


if __name__ == "__main__":
    unittest.main()
