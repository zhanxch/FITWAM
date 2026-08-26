from __future__ import annotations

import unittest

from fastwam.datasets.eve.tau_queries import collect_v6_tau_queries, shard_queries


def _episode(index: int, *, end: int = 200) -> dict:
    return {
        "sample_type": "episode",
        "sample_id": f"ep{index:03d}",
        "sample_role": "success_episode",
        "batch_role": "primary",
        "action_loss": "enabled",
        "episode_outcome": "success",
        "event_outcome": "success",
        "dataset_root": "/data/success_rollouts",
        "episode_index": index,
        "start_frame": 0,
        "end_frame": end,
        "split": "train",
    }


def _event(index: int, *, primary: bool = True, start: int = 0) -> dict:
    return {
        "sample_type": "event",
        "sample_id": f"ev{index:03d}_{'p' if primary else 'a'}",
        "sample_role": "success_event_primary" if primary else "success_auxiliary",
        "batch_role": "primary" if primary else "auxiliary",
        "action_loss": "enabled" if primary else "disabled",
        "episode_outcome": "success",
        "event_outcome": "success",
        "event_type": "success_event",
        "dataset_root": "/data/pair_events",
        "episode_index": index,
        "start_frame": start,
        "end_frame": start + 33,
        "core_start_frame": start,
        "core_end_frame": start + 33,
        "split": "train",
    }


class CollectV6TauQueriesTest(unittest.TestCase):
    def test_plus_keeps_primary_events_not_aux_or_collect200(self) -> None:
        units = [
            _episode(0),
            _event(0, primary=True),
            _event(0, primary=False),
            {
                **_episode(1),
                "sample_role": "success_episode",
                "dataset_root": "/data/collect200",
                "episode_index": 99,
            },
        ]
        # Second episode is still a v6 D0 success episode if it is in the manifest.
        queries = collect_v6_tau_queries(units, replan_steps=24, prefix_fraction=0.5)
        plus = [q for q in queries if q.kind == "plus"]
        self.assertEqual(len(plus), 1)
        self.assertEqual(plus[0].episode_index, 0)
        self.assertEqual(plus[0].frame_index, 0)
        self.assertTrue(all(q.kind != "plus" or q.dataset_root.endswith("pair_events") for q in queries))

    def test_zero_uses_replan_prefixes_not_every_frame(self) -> None:
        queries = collect_v6_tau_queries([_episode(7, end=120)], replan_steps=24, prefix_fraction=0.5)
        zero = [q for q in queries if q.kind == "zero"]
        self.assertEqual([q.frame_index for q in zero], [0, 24, 48])
        self.assertTrue(all(q.episode_index == 7 for q in zero))

    def test_aux_success_dropped(self) -> None:
        queries = collect_v6_tau_queries([_event(3, primary=False)])
        self.assertEqual(queries, [])

    def test_shard_splits_round_robin(self) -> None:
        queries = collect_v6_tau_queries(
            [_event(i, primary=True) for i in range(4)],
            replan_steps=24,
        )
        self.assertEqual(len(queries), 4)
        shard0 = shard_queries(queries, shard_index=0, num_shards=4)
        shard1 = shard_queries(queries, shard_index=1, num_shards=4)
        self.assertEqual(len(shard0), 1)
        self.assertEqual(len(shard1), 1)
        self.assertNotEqual(shard0[0].episode_index, shard1[0].episode_index)


if __name__ == "__main__":
    unittest.main()
