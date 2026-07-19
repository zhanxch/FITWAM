from __future__ import annotations

import unittest

from fastwam.utils.role_balanced_sampler import RoleBalancedBatchSampler


class RoleBalancedBatchSamplerTest(unittest.TestCase):
    @staticmethod
    def _composition(batch: list[int], roles: list[str]) -> tuple[int, int]:
        success = sum(roles[index] == "success" for index in batch)
        return success, len(batch) - success

    def test_four_ranks_have_exact_composition_and_no_overlap(self) -> None:
        roles = ["success"] * 96 + ["auxiliary"] * 160
        samplers = [
            RoleBalancedBatchSampler(
                roles,
                batch_size=8,
                success_per_batch=3,
                num_replicas=4,
                rank=rank,
                seed=17,
            )
            for rank in range(4)
        ]
        rank_batches = [list(sampler) for sampler in samplers]

        self.assertTrue(all(len(batches) == 8 for batches in rank_batches))
        for batch_index in range(8):
            batches = [rank_batches[rank][batch_index] for rank in range(4)]
            for batch in batches:
                self.assertEqual(len(batch), 8)
                self.assertEqual(self._composition(batch, roles), (3, 5))

            flattened = [index for batch in batches for index in batch]
            self.assertEqual(len(flattened), len(set(flattened)))

        for role in ("success", "auxiliary"):
            per_rank = [
                {
                    index
                    for batch in rank_batches[rank]
                    for index in batch
                    if roles[index] == role
                }
                for rank in range(4)
            ]
            for left in range(4):
                for right in range(left + 1, 4):
                    self.assertTrue(per_rank[left].isdisjoint(per_rank[right]))

    def test_seed_and_epoch_are_deterministic(self) -> None:
        roles = ["success"] * 24 + ["failure"] * 32

        first = RoleBalancedBatchSampler(
            roles,
            batch_size=4,
            success_per_batch=2,
            num_replicas=2,
            rank=1,
            seed=123,
        )
        second = RoleBalancedBatchSampler(
            roles,
            batch_size=4,
            success_per_batch=2,
            num_replicas=2,
            rank=1,
            seed=123,
        )

        self.assertEqual(list(first), list(second))
        first.set_epoch(3)
        second.set_epoch(3)
        self.assertEqual(list(first), list(second))

        epoch_three = list(first)
        first.set_epoch(4)
        self.assertNotEqual(list(first), epoch_three)

    def test_shorter_role_cycles_and_preserves_exact_batches(self) -> None:
        roles = ["success"] * 2 + ["auxiliary"] * 20
        sampler = RoleBalancedBatchSampler(
            roles,
            batch_size=4,
            success_per_batch=1,
            seed=9,
        )

        batches = list(sampler)

        self.assertEqual(len(batches), 7)
        self.assertTrue(
            all(self._composition(batch, roles) == (1, 3) for batch in batches)
        )
        self.assertEqual(
            {batch[0] for batch in batches},
            {0, 1},
        )

    def test_cycled_role_stays_in_disjoint_rank_shards_when_possible(self) -> None:
        roles = ["success"] * 4 + ["auxiliary"] * 80
        samplers = [
            RoleBalancedBatchSampler(
                roles,
                batch_size=4,
                success_per_batch=1,
                num_replicas=4,
                rank=rank,
                seed=27,
            )
            for rank in range(4)
        ]
        rank_batches = [list(sampler) for sampler in samplers]
        success_sets = [
            {
                index
                for batch in rank_batches[rank]
                for index in batch
                if roles[index] == "success"
            }
            for rank in range(4)
        ]

        self.assertTrue(all(len(indices) == 1 for indices in success_sets))
        self.assertEqual(len(set.union(*success_sets)), 4)
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertTrue(success_sets[left].isdisjoint(success_sets[right]))

    def test_resume_offset_reproduces_unconsumed_suffix(self) -> None:
        roles = ["success"] * 12 + ["auxiliary"] * 18
        full_sampler = RoleBalancedBatchSampler(
            roles,
            batch_size=5,
            success_per_batch=2,
            seed=31,
        )
        full_batches = list(full_sampler)

        resumed = RoleBalancedBatchSampler(
            roles,
            batch_size=5,
            success_per_batch=2,
            seed=31,
        )
        resumed.set_epoch(2)
        epoch_two_batches = list(resumed)
        resumed.set_resume_batch_offset(2)

        self.assertEqual(len(resumed), len(epoch_two_batches) - 2)
        self.assertEqual(list(resumed), epoch_two_batches[2:])

        state = resumed.state_dict()
        restored = RoleBalancedBatchSampler(
            roles,
            batch_size=5,
            success_per_batch=2,
            seed=31,
        )
        restored.load_state_dict(state)
        self.assertEqual(list(restored), epoch_two_batches[2:])
        self.assertNotEqual(full_batches, epoch_two_batches)

    def test_new_epoch_clears_old_resume_offset(self) -> None:
        sampler = RoleBalancedBatchSampler(
            ["success"] * 8 + ["auxiliary"] * 8,
            batch_size=4,
            success_per_batch=2,
        )
        sampler.set_resume_batch_offset(1)
        sampler.set_epoch(1)

        self.assertEqual(sampler.batch_offset, 0)
        self.assertEqual(len(list(sampler)), sampler.num_batches_per_epoch)

    def test_explicit_auxiliary_roles_reject_uncovered_dataset_roles(self) -> None:
        with self.assertRaisesRegex(ValueError, "not covered"):
            RoleBalancedBatchSampler(
                ["success", "failure", "ignored"],
                batch_size=2,
                success_per_batch=1,
                auxiliary_roles=("failure",),
            )

    def test_invalid_configuration_is_rejected(self) -> None:
        invalid_calls = (
            lambda: RoleBalancedBatchSampler(
                ["success", "auxiliary"],
                batch_size=0,
                success_per_batch=1,
            ),
            lambda: RoleBalancedBatchSampler(
                ["success", "auxiliary"],
                batch_size=2,
                success_per_batch=2,
            ),
            lambda: RoleBalancedBatchSampler(
                ["auxiliary"],
                batch_size=2,
                success_per_batch=1,
            ),
            lambda: RoleBalancedBatchSampler(
                ["success"],
                batch_size=2,
                success_per_batch=1,
            ),
            lambda: RoleBalancedBatchSampler(
                ["success", "auxiliary"],
                batch_size=2,
                success_per_batch=1,
                num_replicas=4,
                rank=4,
            ),
        )

        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()

    def test_resume_offset_is_range_checked(self) -> None:
        sampler = RoleBalancedBatchSampler(
            ["success"] * 4 + ["auxiliary"] * 4,
            batch_size=2,
            success_per_batch=1,
        )

        with self.assertRaises(ValueError):
            sampler.set_resume_batch_offset(-1)
        with self.assertRaises(ValueError):
            sampler.set_resume_batch_offset(sampler.num_batches_per_epoch + 1)


if __name__ == "__main__":
    unittest.main()
