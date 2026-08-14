from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from scripts.fold_glasses import build_seedpair_width_candidates as candidates
from scripts.fold_glasses import discover_seedpair_branch_events as branches
from scripts.fold_glasses import build_seedpair_probe_shortlist as shortlist
from scripts.fold_glasses import probe_seedpair_action_distributions as probe
from scripts.fold_glasses import score_seedpair_action_support as support
from scripts.fold_glasses import run_seedpair_block_interventions as intervention


def unit_phase_features(
    *,
    episode_index: int,
    phases: np.ndarray,
    offset: float = 0.0,
    diverge_after: int | None = None,
    transient_at: int | None = None,
) -> dict[str, np.ndarray]:
    phases = np.asarray(phases, dtype=np.float32)
    angles = 0.15 * phases + float(offset)
    if diverge_after is not None:
        angles[diverge_after:] += 1.0
    if transient_at is not None:
        angles[transient_at] += 1.0
    visual = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(
        np.float32
    )
    states = np.repeat((0.1 * phases + offset)[:, None], 3, axis=1)
    if diverge_after is not None:
        states[diverge_after:] += 1.0
    if transient_at is not None:
        states[transient_at] += 1.0
    length = len(phases)
    return {
        "episode_index": np.asarray(episode_index, dtype=np.int32),
        "stride": np.asarray(24, dtype=np.int32),
        "frame_indices": np.arange(length, dtype=np.int32) * 24,
        "front_visual": visual,
        "wrist_visual": visual.copy(),
        "states": states.astype(np.float32),
        "action_blocks": np.zeros((length, 24, 22), dtype=np.float32),
        "action_valid": np.ones((length, 24), dtype=bool),
    }


def synthetic_population(
    *, transient_failure: bool = False
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[int, dict]]:
    phases = np.arange(8, dtype=np.float32)
    outcomes: list[dict[str, object]] = [
        {"episode_index": 0, "seed": 10, "success": True},
        {"episode_index": 1, "seed": 10, "success": True},
        {"episode_index": 2, "seed": 10, "success": False},
        {"episode_index": 3, "seed": 20, "success": True},
        {"episode_index": 4, "seed": 20, "success": True},
        {"episode_index": 5, "seed": 30, "success": False},
        {"episode_index": 6, "seed": 30, "success": False},
    ]
    features = {
        0: unit_phase_features(episode_index=0, phases=phases),
        1: unit_phase_features(
            episode_index=1, phases=phases, offset=0.01
        ),
        2: unit_phase_features(
            episode_index=2,
            phases=phases,
            diverge_after=None if transient_failure else 4,
            transient_at=4 if transient_failure else None,
        ),
        # Deliberately large all-success variation. It must not calibrate the
        # mixed-seed support threshold.
        3: unit_phase_features(episode_index=3, phases=phases, offset=0.5),
        4: unit_phase_features(episode_index=4, phases=phases, offset=1.0),
        5: unit_phase_features(episode_index=5, phases=phases, offset=0.2),
        6: unit_phase_features(episode_index=6, phases=phases, offset=0.3),
    }
    events: list[dict[str, object]] = [
        {
            "episode_index": 2,
            "event_center_frame": 72,
            "jump_ratio_obs": 2.5,
        }
    ]
    return outcomes, events, features


class SeedPairCandidateTest(unittest.TestCase):
    def test_dtw_alignment_is_monotone_and_uses_expected_endpoint(self) -> None:
        failure_length = 7
        success_length = 9
        expected = [0, 1, 2, 4, 5, 6, 8]
        cost = np.full((failure_length, success_length), 8.0, dtype=np.float32)
        for row, col in enumerate(expected):
            cost[row, col] = 0.0
        result = candidates.dtw_prefix_endpoint(
            cost,
            event_index=5,
            warp_fraction=0.6,
            min_warp_blocks=1,
        )

        self.assertEqual(result["success_index"], expected[5])
        path = result["path"]
        self.assertEqual(path[0], [0, 0])
        self.assertEqual(path[-1], [5, expected[5]])
        self.assertTrue(
            all(
                right[0] >= left[0] and right[1] >= left[1]
                for left, right in zip(path, path[1:])
            )
        )

    def test_mixed_seed_candidate_excludes_all_success_and_all_failure(self) -> None:
        outcomes, events, features = synthetic_population()
        rows, seed_audit, diagnostics = candidates.build_candidates(
            outcomes,
            events,
            features,
            future_horizons=(1, 2, 3),
        )

        self.assertEqual(diagnostics["calibration"]["seed_population"], "mixed_only")
        self.assertEqual(
            diagnostics["calibration"]["same_seed_success_pair_count"], 1
        )
        by_seed = {row["seed"]: row for row in seed_audit}
        self.assertEqual(by_seed[20]["classification"], "all_success")
        self.assertFalse(by_seed[20]["use_for_training"])
        self.assertEqual(by_seed[30]["classification"], "all_failure")
        self.assertFalse(by_seed[30]["use_for_training"])
        self.assertTrue(by_seed[30]["use_for_evaluation"])
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["probe_eligible"])
        summary = rows[0]["alignment_summary"]
        self.assertTrue(summary["future_diverged_after_action"])
        self.assertGreaterEqual(
            summary["future_divergence_persistence"]["run_length"], 2
        )

    def test_transient_visual_spike_is_not_future_divergence(self) -> None:
        outcomes, events, features = synthetic_population(transient_failure=True)
        rows, _, _ = candidates.build_candidates(
            outcomes,
            events,
            features,
            future_horizons=(1, 2, 3),
        )

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["probe_eligible"])
        self.assertFalse(rows[0]["observational_event_supported"])
        self.assertIn(
            "future_not_persistently_diverged", rows[0]["rejection_reasons"]
        )

    def test_head_width_event_is_rejected(self) -> None:
        outcomes, events, features = synthetic_population()
        events[0]["event_center_frame"] = 24
        rows, _, _ = candidates.build_candidates(
            outcomes,
            events,
            features,
            future_horizons=(1, 2, 3),
        )

        self.assertFalse(rows[0]["probe_eligible"])
        self.assertIn("event_in_head_guard", rows[0]["rejection_reasons"])


class _AffineNormalizer:
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(value * 2.0 - 1.0, -5.0, 5.0)


class _FakeProcessor:
    shape_meta = {"action": [{"key": "default"}]}

    class _Container:
        normalizers = {"action": {"default": _AffineNormalizer()}}

    normalizer = _Container()


class _FakePolicy:
    processor = _FakeProcessor()


class ActionProbeTest(unittest.TestCase):
    def test_actions_are_compared_in_training_normalization_space(self) -> None:
        robot_actions = np.asarray([[0.0, 1.0], [3.0, -4.0]], dtype=np.float32)
        normalized = probe.normalize_actions_for_training(
            _FakePolicy(), robot_actions
        )
        np.testing.assert_allclose(
            normalized,
            np.asarray([[-1.0, 1.0], [5.0, -5.0]], dtype=np.float32),
        )

    def test_probe_contexts_include_only_eligible_candidates_by_default(self) -> None:
        rows = [
            {
                "candidate_id": "keep",
                "probe_eligible": True,
                "failure_episode_index": 2,
                "failure_frame": 72,
                "success_alignments": [
                    {"success_episode_index": 0, "success_frame": 72}
                ],
            },
            {
                "candidate_id": "drop",
                "probe_eligible": False,
                "failure_episode_index": 3,
                "failure_frame": 96,
                "success_alignments": [],
            },
        ]
        contexts = probe.build_contexts(rows)
        self.assertEqual(
            [row["context_id"] for row in contexts],
            ["ep000000_f0072", "ep000002_f0072"],
        )


class ActionSupportTest(unittest.TestCase):
    @staticmethod
    def write_probe(
        root: Path,
        episode: int,
        frame: int,
        *,
        actual: float,
        samples: list[float],
    ) -> None:
        path = root / "contexts" / f"ep{episode:06d}_f{frame:04d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        actual_block = np.full((4, 2), actual, dtype=np.float32)
        sample_blocks = np.stack(
            [np.full((4, 2), value, dtype=np.float32) for value in samples]
        )
        np.savez_compressed(
            path,
            action_samples_normalized=sample_blocks,
            actual_action_block_normalized=actual_block,
            actual_action_valid=np.ones(4, dtype=bool),
        )

    def test_support_score_detects_mixed_sampled_branches(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_probe(
                root, 0, 48, actual=0.0, samples=[0.01, 0.02, 0.98, 1.01]
            )
            self.write_probe(
                root, 1, 48, actual=1.0, samples=[0.97, 1.02, 0.99, 1.01]
            )
            candidate = {
                "candidate_id": "mixed",
                "seed": 7,
                "failure_episode_index": 0,
                "failure_frame": 48,
                "observational_event_supported": True,
                "success_alignments": [
                    {"success_episode_index": 1, "success_frame": 48}
                ],
            }
            row = support.score_candidate(
                candidate, root, support_radius_multiplier=2.0
            )

        self.assertTrue(
            row["sampled_distribution_contains_success_and_failure_branches"]
        )
        self.assertEqual(row["failure_policy_any_success_support_hits"], 2)
        self.assertEqual(row["failure_policy_any_success_support_mass"], 0.5)
        self.assertFalse(row["training_eligible"])
        low, high = row["failure_policy_any_success_support_mass_wilson95"]
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_support_score_keeps_multiple_success_modes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_probe(root, 0, 48, actual=0.0, samples=[0.0] * 4)
            self.write_probe(root, 1, 48, actual=1.0, samples=[1.0] * 4)
            self.write_probe(root, 2, 48, actual=3.0, samples=[3.0] * 4)
            candidate = {
                "candidate_id": "two-modes",
                "seed": 7,
                "failure_episode_index": 0,
                "failure_frame": 48,
                "success_alignments": [
                    {"success_episode_index": 1, "success_frame": 48},
                    {"success_episode_index": 2, "success_frame": 48},
                ],
            }
            row = support.score_candidate(candidate, root)

        self.assertEqual(len(row["success_modes"]), 2)
        self.assertEqual(
            row["success_mode_pairwise_actual_distance"]["p50"], 2.0
        )
        self.assertEqual(row["failure_policy_any_success_support_mass"], 0.0)


class BlockInterventionTest(unittest.TestCase):
    def test_orthogonal_control_matches_target_rms(self) -> None:
        factual = np.zeros((4, 3), dtype=np.float32)
        success_a = np.zeros_like(factual)
        success_a[:, 0] = 1.0
        success_b = np.zeros_like(factual)
        success_b[:, 1] = 2.0
        basis = np.zeros_like(factual)
        basis[:, 2] = 0.5

        control = intervention.orthogonal_matched_control(
            factual,
            success_a,
            basis,
            [success_a, success_b],
        )

        self.assertAlmostEqual(
            support.block_rms(control, factual),
            support.block_rms(success_a, factual),
        )
        delta = control.reshape(-1)
        self.assertAlmostEqual(float(delta @ success_a.reshape(-1)), 0.0)
        self.assertAlmostEqual(float(delta @ success_b.reshape(-1)), 0.0)

    def test_progress_delta_uses_task_state_not_visual_distance(self) -> None:
        before = {
            "hinge_0": 0.2,
            "hinge_1": 0.5,
            "hinge_min": 0.2,
            "glass_minus_box_xyz": [0.3, 0.4, 0.0],
            "trigger_active": False,
        }
        after = {
            "hinge_0": 0.7,
            "hinge_1": 0.6,
            "hinge_min": 0.6,
            "glass_minus_box_xyz": [0.0, 0.1, 0.0],
            "trigger_active": True,
        }
        delta = intervention.progress_delta(before, after)

        self.assertAlmostEqual(delta["hinge_min"], 0.4)
        self.assertAlmostEqual(delta["glass_to_box_xy_distance"], -0.4)
        self.assertTrue(delta["trigger_active_changed"])


class BranchDiscoveryTest(unittest.TestCase):
    @staticmethod
    def population(*, failure_matches_success_mode: bool = False):
        phases = np.arange(10, dtype=np.float32)
        outcomes = [
            {"episode_index": 0, "seed": 77, "success": True},
            {"episode_index": 1, "seed": 77, "success": True},
            {"episode_index": 2, "seed": 77, "success": False},
            {"episode_index": 3, "seed": 88, "success": True},
            {"episode_index": 4, "seed": 99, "success": False},
        ]
        features = {
            0: unit_phase_features(episode_index=0, phases=phases),
            1: unit_phase_features(episode_index=1, phases=phases, offset=0.005),
            2: unit_phase_features(
                episode_index=2, phases=phases, diverge_after=5
            ),
            3: unit_phase_features(episode_index=3, phases=phases, offset=0.5),
            4: unit_phase_features(episode_index=4, phases=phases, offset=0.5),
        }
        # Two distinct successful modes are preserved instead of averaged.
        features[0]["action_blocks"][3] = -1.0
        features[1]["action_blocks"][3] = 1.0
        features[2]["action_blocks"][3] = (
            -1.0 if failure_matches_success_mode else 3.0
        )
        return outcomes, features

    def test_global_zscore_matches_checkpoint_normalizer(self) -> None:
        values = np.asarray([[-10.0] + [2.0] * 21], dtype=np.float32)
        mean = np.zeros(22, dtype=np.float32)
        std = np.full(22, 2.0, dtype=np.float32)
        normalized = branches.normalize_actions(values, mean, std)
        self.assertEqual(float(normalized[0, 0]), -5.0)
        np.testing.assert_allclose(normalized[0, 1:], 1.0)

    def test_monotone_endpoint_map_never_moves_backward(self) -> None:
        cost = np.full((8, 10), 5.0, dtype=np.float32)
        expected = [0, 1, 2, 4, 5, 6, 8, 9]
        for left, right in enumerate(expected):
            cost[left, right] = 0.0
        mapping = branches.monotone_endpoint_map(cost)
        self.assertTrue(np.all(np.diff(mapping) >= 0))
        self.assertEqual(int(mapping[0]), 0)

    def test_future_context_uses_explicit_success_terminal_proxy(self) -> None:
        cost = np.arange(24, dtype=np.float32).reshape(6, 4)
        evidence = branches.future_context_evidence(
            cost, left_index=1, right_index=2, horizon=3
        )
        assert evidence is not None
        self.assertEqual(evidence["failure_future_replan_index"], 4)
        self.assertEqual(evidence["success_future_replan_index"], 3)
        self.assertTrue(evidence["success_terminal_proxy"])
        self.assertEqual(evidence["cost"], float(cost[4, 3]))

    def test_discovers_action_branch_before_future_visual_divergence(self) -> None:
        outcomes, features = self.population()
        selected, _, seed_audit, diagnostics = branches.discover_events(
            outcomes,
            features,
            action_mean=np.zeros(22, dtype=np.float32),
            action_std=np.ones(22, dtype=np.float32),
            head_guard_blocks=2,
            min_action_run_steps=8,
            future_horizons=(1, 2, 3, 4),
        )
        self.assertEqual(len(selected), 1)
        event = selected[0]
        self.assertEqual(event["failure_replan_index"], 3)
        self.assertTrue(event["observational_event_supported"])
        self.assertFalse(event["training_eligible"])
        self.assertEqual(
            event["future_context"]["persistence"]["onset_horizon"], 2
        )
        # The successes disagree at this event, so low success width cannot be a gate.
        distances = [
            row["executed_action_block_rms"]
            for row in event["success_alignments"]
        ]
        self.assertEqual(len(distances), 2)
        by_seed = {row["seed"]: row for row in seed_audit}
        self.assertFalse(by_seed[88]["use_for_discovery"])
        self.assertTrue(by_seed[99]["use_for_evaluation"])
        self.assertFalse(diagnostics["failure_width_jump_used_for_discovery"])

    def test_rejects_failure_action_that_matches_any_success_mode(self) -> None:
        outcomes, features = self.population(failure_matches_success_mode=True)
        selected, scored, _, _ = branches.discover_events(
            outcomes,
            features,
            action_mean=np.zeros(22, dtype=np.float32),
            action_std=np.ones(22, dtype=np.float32),
            head_guard_blocks=2,
            min_action_run_steps=8,
            future_horizons=(1, 2, 3, 4),
        )
        self.assertEqual(selected, [])
        anchor = next(
            row for row in scored if row["failure_replan_index"] == 3
        )
        self.assertIn(
            "failure_action_not_separated_from_observed_success_blocks",
            anchor["rejection_reasons"],
        )


class ProbeShortlistTest(unittest.TestCase):
    def test_width_strata_use_evidence_rank_and_distinct_seeds(self) -> None:
        with TemporaryDirectory() as temporary:
            width_root = Path(temporary)
            candidates_rows = []
            specifications = [
                (1, 1, 2.0, 4.0),
                (2, 2, 1.6, 3.0),
                (3, 3, 0.8, 5.0),
                (4, 4, 0.6, 2.0),
                # Same seed as the strongest high-width candidate: excluded
                # within that stratum despite a high evidence score.
                (5, 1, 3.0, 6.0),
            ]
            for episode, seed, ratio, score in specifications:
                path = width_root / "npz" / f"ep{episode:06d}_widths.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    path,
                    probe_steps=np.asarray([48]),
                    probe_widths=np.asarray([ratio], dtype=np.float32),
                    baseline_median=np.asarray([1.0], dtype=np.float32),
                    found_event=np.asarray(False),
                    event_center_frame=np.asarray(-1),
                )
                candidates_rows.append(
                    {
                        "candidate_id": f"candidate-{episode}",
                        "failure_episode_index": episode,
                        "failure_frame": 48,
                        "seed": seed,
                        "selection_score": score,
                        "global_observational_rank": episode,
                        "shared_context": {"supported_success_count": 2},
                        "future_context": {
                            "persistence": {"onset_horizon": 2},
                            "evidence_source_counts": {"2": 2},
                        },
                    }
                )
            selected = shortlist.build_shortlist(
                candidates_rows,
                width_root,
                per_stratum=2,
                high_ratio_min=1.5,
                low_ratio_max=1.0,
            )

        by_stratum = {}
        for row in selected:
            by_stratum.setdefault(row["width_stratum"], []).append(row)
        self.assertEqual(
            [row["candidate_id"] for row in by_stratum["high_failure_width"]],
            ["candidate-5", "candidate-2"],
        )
        self.assertEqual(
            [row["candidate_id"] for row in by_stratum["low_failure_width_control"]],
            ["candidate-3", "candidate-4"],
        )


if __name__ == "__main__":
    unittest.main()
