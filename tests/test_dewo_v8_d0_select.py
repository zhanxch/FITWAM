from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_SELECT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "dewo_v2"
    / "select_success_rollout_primary.py"
)
_spec = importlib.util.spec_from_file_location("select_success_rollout_primary", _SELECT)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
select_one_per_all_success_seed = _mod.select_one_per_all_success_seed


class DewoV8D0SelectTests(unittest.TestCase):
    def test_one_episode_per_all_success_seed(self) -> None:
        outcomes = []
        lengths = {}
        ep = 0
        # seed 10: 4/4 success
        for attempt in range(4):
            outcomes.append(
                {
                    "episode_index": ep,
                    "seed": 10,
                    "attempt_index": attempt,
                    "success": True,
                    "outcome": "success",
                }
            )
            lengths[ep] = 200
            ep += 1
        # seed 11: mixed
        for attempt, ok in enumerate([True, False, True, False]):
            outcomes.append(
                {
                    "episode_index": ep,
                    "seed": 11,
                    "attempt_index": attempt,
                    "success": ok,
                    "outcome": "success" if ok else "failure",
                }
            )
            lengths[ep] = 200 if ok else 1000
            ep += 1
        primary, leftover, failures = select_one_per_all_success_seed(
            outcomes=outcomes, lengths=lengths
        )
        self.assertEqual(len(primary), 1)
        self.assertEqual(int(primary[0]["seed"]), 10)
        self.assertEqual(int(primary[0]["attempt_index"]), 0)
        self.assertEqual(len(leftover), 5)  # 3 leftover all-success + 2 mixed success
        self.assertEqual(len(failures), 2)


if __name__ == "__main__":
    unittest.main()
