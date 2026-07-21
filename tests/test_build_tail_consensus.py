from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "water_plant"
    / "build_tail_consensus.py"
)
SPEC = importlib.util.spec_from_file_location("build_tail_consensus", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
consensus = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = consensus
SPEC.loader.exec_module(consensus)


MANIFEST_SHA256 = "a" * 64


def _write_input(
    root: Path,
    *,
    label: str,
    periodic_window_frames: int,
    rows: list[dict],
    manifest_sha256: str = MANIFEST_SHA256,
) -> consensus.InputSpec:
    input_root = root / label
    input_root.mkdir()
    episodes_path = input_root / "episodes.csv"
    with episodes_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = input_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "manifest_sha256": manifest_sha256,
                "failure_episodes": len(rows),
                "tail_config": {
                    "periodic_window_frames": periodic_window_frames,
                },
                "materiality": {
                    "episodes_trimmed": sum(
                        str(row["should_trim"]).lower() == "true" for row in rows
                    ),
                    "material_episodes": sum(
                        str(row["material"]).lower() == "true" for row in rows
                    ),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return consensus.InputSpec(
        label=label,
        expected_periodic_window_frames=periodic_window_frames,
        episodes_csv=episodes_path,
        summary_json=summary_path,
    )


def _row(
    episode_index: int,
    *,
    num_frames: int,
    cutoff_frame: int,
    should_trim: bool,
    material: bool,
) -> dict:
    return {
        "dataset_id": "water_plant_rollout",
        "episode_index": episode_index,
        "num_frames": num_frames,
        "cutoff_frame": cutoff_frame,
        "should_trim": should_trim,
        "material": material,
    }


class BuildTailConsensusTest(unittest.TestCase):
    def test_conservative_consensus_and_deterministic_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = [
                _write_input(
                    root,
                    label="window-150",
                    periodic_window_frames=150,
                    rows=[
                        _row(0, num_frames=100, cutoff_frame=82, should_trim=True, material=True),
                        _row(1, num_frames=120, cutoff_frame=95, should_trim=True, material=True),
                        _row(
                            2,
                            num_frames=200,
                            cutoff_frame=200,
                            should_trim=False,
                            material=False,
                        ),
                        _row(3, num_frames=180, cutoff_frame=150, should_trim=True, material=True),
                    ],
                ),
                _write_input(
                    root,
                    label="window-100",
                    periodic_window_frames=100,
                    rows=[
                        _row(0, num_frames=100, cutoff_frame=80, should_trim=True, material=True),
                        _row(1, num_frames=120, cutoff_frame=90, should_trim=True, material=True),
                        _row(
                            2,
                            num_frames=200,
                            cutoff_frame=200,
                            should_trim=False,
                            material=False,
                        ),
                        _row(3, num_frames=180, cutoff_frame=150, should_trim=True, material=True),
                    ],
                ),
                _write_input(
                    root,
                    label="window-125",
                    periodic_window_frames=125,
                    rows=[
                        _row(0, num_frames=100, cutoff_frame=85, should_trim=True, material=True),
                        _row(
                            1,
                            num_frames=120,
                            cutoff_frame=120,
                            should_trim=False,
                            material=False,
                        ),
                        _row(
                            2,
                            num_frames=200,
                            cutoff_frame=200,
                            should_trim=False,
                            material=False,
                        ),
                        _row(3, num_frames=180, cutoff_frame=150, should_trim=True, material=False),
                    ],
                ),
            ]

            report, records = consensus.build_consensus(specs)
            self.assertEqual(report["expected_periodic_window_frames"], [100, 125, 150])
            self.assertEqual([record["episode_index"] for record in records], [0, 1, 2, 3])
            self.assertTrue(records[0]["should_trim"])
            self.assertEqual(records[0]["cutoff_frame"], 85)
            self.assertFalse(records[1]["should_trim"])
            self.assertEqual(records[1]["cutoff_frame"], 120)
            self.assertEqual(
                records[1]["instability_flags"],
                [
                    "decision_disagreement",
                    "cutoff_disagreement",
                    "materiality_disagreement",
                ],
            )
            self.assertEqual(records[2]["instability_flags"], [])
            self.assertTrue(records[3]["should_trim"])
            self.assertFalse(records[3]["consensus_material"])
            self.assertEqual(records[3]["instability_flags"], ["materiality_disagreement"])
            self.assertEqual(report["aggregate"]["agreement"]["unanimous_trim_episodes"], 2)
            self.assertEqual(report["aggregate"]["agreement"]["unanimous_no_trim_episodes"], 1)
            self.assertEqual(report["aggregate"]["materiality"]["consensus_material_episodes"], 1)

            first = root / "output-a"
            second = root / "output-b"
            first_paths = consensus.write_outputs(first, report, records)
            second_paths = consensus.write_outputs(second, report, records)
            self.assertEqual(first_paths[0].read_bytes(), second_paths[0].read_bytes())
            self.assertEqual(first_paths[1].read_bytes(), second_paths[1].read_bytes())
            written_report = json.loads(first_paths[0].read_text(encoding="utf-8"))
            self.assertEqual(written_report["outputs"]["cutoff_records_count"], 4)
            self.assertNotIn(str(root), first_paths[0].read_text(encoding="utf-8"))

    def test_fails_closed_on_provenance_or_episode_mismatch(self) -> None:
        scenarios = ("manifest", "episode_set", "num_frames", "window")
        for scenario in scenarios:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base_rows = [
                    _row(0, num_frames=100, cutoff_frame=80, should_trim=True, material=True),
                    _row(1, num_frames=100, cutoff_frame=100, should_trim=False, material=False),
                ]
                specs = [
                    _write_input(root, label="a", periodic_window_frames=100, rows=base_rows),
                    _write_input(root, label="b", periodic_window_frames=125, rows=base_rows),
                    _write_input(
                        root,
                        label="c",
                        periodic_window_frames=150,
                        rows=base_rows,
                        manifest_sha256=("b" * 64 if scenario == "manifest" else MANIFEST_SHA256),
                    ),
                ]
                if scenario == "episode_set":
                    specs[2] = _write_input(
                        root,
                        label="c-replacement",
                        periodic_window_frames=150,
                        rows=[base_rows[0]],
                    )
                elif scenario == "num_frames":
                    specs[2] = _write_input(
                        root,
                        label="c-replacement",
                        periodic_window_frames=150,
                        rows=[
                            base_rows[0],
                            _row(
                                1,
                                num_frames=101,
                                cutoff_frame=101,
                                should_trim=False,
                                material=False,
                            ),
                        ],
                    )
                elif scenario == "window":
                    specs[2] = consensus.InputSpec(
                        label=specs[2].label,
                        expected_periodic_window_frames=151,
                        episodes_csv=specs[2].episodes_csv,
                        summary_json=specs[2].summary_json,
                    )

                output_dir = root / "output"
                with self.assertRaises(ValueError):
                    report, records = consensus.build_consensus(specs)
                    consensus.write_outputs(output_dir, report, records)
                self.assertFalse(output_dir.exists())

    def test_rejects_fewer_than_three_inputs_and_inconsistent_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_rows = [
                _row(0, num_frames=100, cutoff_frame=80, should_trim=False, material=False)
            ]
            specs = [
                _write_input(root, label="a", periodic_window_frames=100, rows=invalid_rows),
                _write_input(root, label="b", periodic_window_frames=125, rows=invalid_rows),
            ]
            with self.assertRaisesRegex(ValueError, "at least three inputs"):
                consensus.build_consensus(specs)
            third = _write_input(
                root, label="c", periodic_window_frames=150, rows=invalid_rows
            )
            with self.assertRaisesRegex(ValueError, "should_trim=false"):
                consensus.build_consensus([*specs, third])

    def test_cli_help(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--input", result.stdout)
        self.assertIn("EXPECTED_WINDOW", result.stdout)
        self.assertIn("modify an EveRobot manifest", result.stdout)


if __name__ == "__main__":
    unittest.main()
