from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.everobot import build_episode_split


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_dataset(
    root: Path,
    outcomes: list[str],
    *,
    write_outcome_ledger: bool = False,
    mark_failures_in_task: bool = True,
) -> None:
    write_json(root / "meta" / "info.json", {"fps": 30})
    write_jsonl(
        root / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": index,
                "length": 40,
                "tasks": [
                    "Water plant"
                    + (
                        ". " + build_episode_split.build_eve_sidecar.FAILURE_PHRASE
                        if outcome == "failure" and mark_failures_in_task
                        else ""
                    )
                ],
            }
            for index, outcome in enumerate(outcomes)
        ],
    )
    data = root / "data" / "chunk-000" / "episode_000000.parquet"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_bytes(b"synthetic")
    if write_outcome_ledger:
        write_jsonl(
            root / "meta" / "episode_outcomes.jsonl",
            [
                {
                    "episode_index": index,
                    "outcome": outcome,
                    "success": outcome == "success",
                    "attempt_index": index,
                    "seed": 1000 + index,
                    "source": "dexjoco_env",
                }
                for index, outcome in enumerate(outcomes)
            ],
        )


class BuildEpisodeSplitTest(unittest.TestCase):
    def test_split_is_deterministic_and_stratified(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rollout = root / "rollout"
            make_dataset(
                rollout,
                ["success"] * 8 + ["failure"] * 4,
            )
            rows_a, report_a = build_episode_split.build_split_map(
                [("rollout", rollout)],
                force_success_dataset_ids=set(),
                failure_phrase=build_episode_split.build_eve_sidecar.FAILURE_PHRASE,
                val_fraction=0.25,
                seed=7,
            )
            rows_b, report_b = build_episode_split.build_split_map(
                [("rollout", rollout)],
                force_success_dataset_ids=set(),
                failure_phrase=build_episode_split.build_eve_sidecar.FAILURE_PHRASE,
                val_fraction=0.25,
                seed=7,
            )

        self.assertEqual(rows_a, rows_b)
        self.assertEqual(
            report_a["split_map_sha256"], report_b["split_map_sha256"]
        )
        self.assertEqual(
            report_a["counts"]["rollout:success"], {"train": 6, "val": 2}
        )
        self.assertEqual(
            report_a["counts"]["rollout:failure"], {"train": 3, "val": 1}
        )

    def test_force_success_overrides_task_marker(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "base"
            make_dataset(root, ["failure", "failure"])
            rows, _ = build_episode_split.build_split_map(
                [("base", root)],
                force_success_dataset_ids={"base"},
                failure_phrase=build_episode_split.build_eve_sidecar.FAILURE_PHRASE,
                val_fraction=0.5,
                seed=3,
            )

        self.assertEqual(
            {row["episode_outcome"] for row in rows}, {"success"}
        )
        self.assertEqual(
            {row["split"] for row in rows}, {"train", "val"}
        )

    def test_singleton_stratum_stays_in_train(self) -> None:
        rows = [
            {
                "dataset_id": "rollout",
                "episode_index": 0,
                "episode_id": "ep-0",
                "episode_outcome": "failure",
                "outcome_source": "task_marker",
            }
        ]
        assigned = build_episode_split.assign_splits(
            rows, val_fraction=0.2, seed=1
        )
        self.assertEqual(assigned[0]["split"], "train")

    def test_required_structured_outcomes_support_clean_task_text(self) -> None:
        with TemporaryDirectory() as temporary:
            rollout = Path(temporary) / "rollout"
            make_dataset(
                rollout,
                ["success", "failure", "success", "failure"],
                write_outcome_ledger=True,
                mark_failures_in_task=False,
            )
            rows, report = build_episode_split.build_split_map(
                [("rollout", rollout)],
                force_success_dataset_ids=set(),
                failure_phrase=build_episode_split.build_eve_sidecar.FAILURE_PHRASE,
                val_fraction=0.5,
                seed=11,
                require_explicit_outcome_dataset_ids={"rollout"},
            )

        self.assertEqual(
            [row["episode_outcome"] for row in rows],
            ["success", "failure", "success", "failure"],
        )
        self.assertEqual(
            {row["outcome_source"] for row in rows},
            {"structured_outcome_ledger"},
        )
        self.assertEqual(
            report["require_explicit_outcome_dataset_ids"], ["rollout"]
        )

    def test_required_structured_outcomes_reject_missing_ledger(self) -> None:
        with TemporaryDirectory() as temporary:
            rollout = Path(temporary) / "rollout"
            make_dataset(
                rollout,
                ["success", "failure"],
                mark_failures_in_task=False,
            )
            with self.assertRaisesRegex(
                FileNotFoundError, "structured outcome ledger"
            ):
                build_episode_split.build_split_map(
                    [("rollout", rollout)],
                    force_success_dataset_ids=set(),
                    failure_phrase=(
                        build_episode_split.build_eve_sidecar.FAILURE_PHRASE
                    ),
                    val_fraction=0.5,
                    seed=11,
                    require_explicit_outcome_dataset_ids={"rollout"},
                )


if __name__ == "__main__":
    unittest.main()
