from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from torch.utils.data import Sampler


class RoleBalancedBatchSampler(Sampler[list[int]]):
    """Build deterministic, role-balanced batches for one distributed rank.

    Every yielded local batch contains ``primary_per_batch`` primary samples and
    ``batch_size - primary_per_batch`` auxiliary samples. Each shuffled role is
    partitioned into disjoint rank-local shards before batching. A shorter role
    is reshuffled and cycled only inside its shard after the current permutation
    has been exhausted.

    The sampler already performs distributed sharding. A DataLoader using it
    must not be sharded again by Accelerate.
    """

    def __init__(
        self,
        roles: Sequence[str],
        batch_size: int,
        success_per_batch: int | None = None,
        *,
        primary_per_batch: int | None = None,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        primary_role: str = "success",
        auxiliary_roles: str | Sequence[str] | None = None,
        ignore_roles: str | Sequence[str] | None = None,
    ) -> None:
        self.roles = tuple(roles)
        self.batch_size = self._positive_int("batch_size", batch_size)
        if primary_per_batch is None:
            primary_per_batch = success_per_batch
        elif (
            success_per_batch is not None
            and int(success_per_batch) != int(primary_per_batch)
        ):
            raise ValueError(
                "`success_per_batch` and `primary_per_batch` disagree. "
                "Use `primary_per_batch`; `success_per_batch` is retained only "
                "for backward compatibility."
            )
        if primary_per_batch is None:
            raise ValueError("`primary_per_batch` must be provided.")
        self.primary_per_batch = self._positive_int(
            "primary_per_batch", primary_per_batch
        )
        self.num_replicas = self._positive_int("num_replicas", num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.primary_role = self._role_name("primary_role", primary_role)
        self.ignore_roles = self._normalize_ignore_roles(ignore_roles)
        self.auxiliary_roles = self._normalize_auxiliary_roles(auxiliary_roles)
        if self.primary_role in self.ignore_roles:
            raise ValueError("`ignore_roles` cannot contain `primary_role`.")
        if self.auxiliary_roles is not None:
            overlap = sorted(set(self.auxiliary_roles) & set(self.ignore_roles))
            if overlap:
                raise ValueError(
                    "`ignore_roles` cannot overlap `auxiliary_roles`: "
                    f"{overlap!r}."
                )

        if self.primary_per_batch > self.batch_size:
            raise ValueError(
                "`primary_per_batch` cannot exceed `batch_size`, got "
                f"primary_per_batch={self.primary_per_batch} batch_size={self.batch_size}."
            )
        if self.auxiliary_roles and self.primary_per_batch >= self.batch_size:
            raise ValueError(
                "`primary_per_batch` must be smaller than `batch_size` when "
                "auxiliary roles are requested."
            )
        if (
            self.auxiliary_roles is not None
            and len(self.auxiliary_roles) == 0
            and self.primary_per_batch != self.batch_size
        ):
            raise ValueError(
                "Empty `auxiliary_roles` requires `primary_per_batch == batch_size` "
                "(all-primary batches)."
            )
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"`rank` must be in [0, {self.num_replicas}), got {self.rank}."
            )
        if not self.roles:
            raise ValueError("`roles` must contain at least one dataset role.")
        if any(not isinstance(role, str) or not role for role in self.roles):
            raise ValueError("Every entry in `roles` must be a non-empty string.")

        self._primary_indices = tuple(
            index for index, role in enumerate(self.roles) if role == self.primary_role
        )
        self._auxiliary_indices = tuple(
            index for index, role in enumerate(self.roles) if self._is_auxiliary(role)
        )
        self._auxiliary_indices_by_role = self._partition_auxiliary_indices()
        self._auxiliary_draws_by_role = self._draws_per_auxiliary_role()
        self._validate_role_partition()

        primary_global = self.primary_per_batch * self.num_replicas
        self.num_batches_per_epoch = math.ceil(
            len(self._primary_indices) / primary_global
        )
        for role, indices in self._auxiliary_indices_by_role.items():
            draws = self._auxiliary_draws_by_role.get(role, 0) * self.num_replicas
            if draws < 1:
                continue
            self.num_batches_per_epoch = max(
                self.num_batches_per_epoch,
                math.ceil(len(indices) / draws),
            )

        self.epoch = 0
        self.resume_batch_offset = 0

    @staticmethod
    def _positive_int(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"`{name}` must be a positive integer, got {value!r}.")
        return value

    @staticmethod
    def _role_name(name: str, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"`{name}` must be a non-empty string.")
        return value

    def _normalize_auxiliary_roles(
        self, auxiliary_roles: str | Sequence[str] | None
    ) -> tuple[str, ...] | None:
        if auxiliary_roles is None:
            return None
        if isinstance(auxiliary_roles, str):
            normalized = (self._role_name("auxiliary_roles", auxiliary_roles),)
        else:
            normalized = tuple(auxiliary_roles)
            if any(not isinstance(role, str) or not role for role in normalized):
                raise ValueError(
                    "`auxiliary_roles` must contain non-empty role strings."
                )
        if self.primary_role in normalized:
            raise ValueError(
                "`primary_role` cannot also be listed in `auxiliary_roles`."
            )
        return normalized

    def _normalize_ignore_roles(
        self, ignore_roles: str | Sequence[str] | None
    ) -> tuple[str, ...]:
        if ignore_roles is None:
            return ()
        if isinstance(ignore_roles, str):
            if not ignore_roles.strip():
                return ()
            return (self._role_name("ignore_roles", ignore_roles),)
        return tuple(
            dict.fromkeys(
                self._role_name("ignore_roles", role) for role in ignore_roles
            )
        )

    def _is_auxiliary(self, role: str) -> bool:
        if role in self.ignore_roles:
            return False
        if self.auxiliary_per_batch == 0:
            return False
        if self.auxiliary_roles is None:
            return role != self.primary_role
        return role in self.auxiliary_roles

    def _partition_auxiliary_indices(self) -> dict[str, tuple[int, ...]]:
        if self.auxiliary_per_batch == 0:
            return {}
        if self.auxiliary_roles is None or len(self.auxiliary_roles) <= 1:
            key = (
                self.auxiliary_roles[0]
                if self.auxiliary_roles is not None
                else "_aux"
            )
            return {key: self._auxiliary_indices}
        return {
            role: tuple(
                index for index, value in enumerate(self.roles) if value == role
            )
            for role in self.auxiliary_roles
        }

    def _draws_per_auxiliary_role(self) -> dict[str, int]:
        roles = tuple(self._auxiliary_indices_by_role)
        if not roles or self.auxiliary_per_batch == 0:
            return {}
        if len(roles) <= 1:
            return {roles[0]: self.auxiliary_per_batch}
        if self.auxiliary_per_batch % len(roles) != 0:
            raise ValueError(
                "`auxiliary_per_batch` must be divisible by the number of "
                f"auxiliary roles ({len(roles)}); got "
                f"auxiliary_per_batch={self.auxiliary_per_batch}."
            )
        each = self.auxiliary_per_batch // len(roles)
        if each < 1:
            raise ValueError(
                "Each auxiliary role must receive at least one slot per batch."
            )
        return {role: each for role in roles}

    def _validate_role_partition(self) -> None:
        if not self._primary_indices:
            raise ValueError(
                f"No samples have the primary role {self.primary_role!r}."
            )
        if self.auxiliary_per_batch == 0:
            if self.auxiliary_roles is not None:
                recognized = {
                    self.primary_role,
                    *self.auxiliary_roles,
                    *self.ignore_roles,
                }
                unknown = sorted(set(self.roles) - recognized)
                if unknown:
                    raise ValueError(
                        "Dataset roles are not covered by `primary_role`, "
                        f"`auxiliary_roles`, and `ignore_roles`: {unknown!r}."
                    )
            return
        if not self._auxiliary_indices:
            expected = (
                "a role different from the primary role"
                if self.auxiliary_roles is None
                else f"one of {self.auxiliary_roles!r}"
            )
            raise ValueError(f"No samples have an auxiliary role ({expected}).")

        if self.auxiliary_roles is not None:
            recognized = {
                self.primary_role,
                *self.auxiliary_roles,
                *self.ignore_roles,
            }
            unknown = sorted(set(self.roles) - recognized)
            if unknown:
                raise ValueError(
                    "Dataset roles are not covered by `primary_role`, "
                    f"`auxiliary_roles`, and `ignore_roles`: {unknown!r}."
                )
            missing = [
                role
                for role, indices in self._auxiliary_indices_by_role.items()
                if not indices
            ]
            if missing:
                raise ValueError(
                    "No samples have auxiliary role(s) "
                    f"{missing!r} required by `auxiliary_roles`."
                )

    @property
    def auxiliary_per_batch(self) -> int:
        return self.batch_size - self.primary_per_batch

    @property
    def success_per_batch(self) -> int:
        """Backward-compatible alias for older experiment code."""

        return self.primary_per_batch

    @property
    def batch_offset(self) -> int:
        return self.resume_batch_offset

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError(f"`epoch` must be non-negative, got {epoch}.")
        if epoch != self.epoch:
            self.epoch = epoch
            self.resume_batch_offset = 0

    def set_resume_batch_offset(self, batch_offset: int) -> None:
        batch_offset = int(batch_offset)
        if not 0 <= batch_offset <= self.num_batches_per_epoch:
            raise ValueError(
                "`batch_offset` must be between 0 and "
                f"{self.num_batches_per_epoch}, got {batch_offset}."
            )
        self.resume_batch_offset = batch_offset

    def set_batch_offset(self, batch_offset: int) -> None:
        self.set_resume_batch_offset(batch_offset)

    def clear_resume_batch_offset(self) -> None:
        self.resume_batch_offset = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "epoch": self.epoch,
            "batch_offset": self.resume_batch_offset,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if "epoch" not in state or "batch_offset" not in state:
            raise ValueError("Sampler state must contain `epoch` and `batch_offset`.")
        self.set_epoch(int(state["epoch"]))
        self.set_resume_batch_offset(int(state["batch_offset"]))

    @staticmethod
    def _draw_global_batches(
        indices: Sequence[int],
        draws_per_batch: int,
        num_batches: int,
        rng: random.Random,
    ) -> list[list[int]]:
        """Cycle shuffled permutations without avoidable within-batch overlap."""

        pool = list(indices)
        cycle: list[int] = []
        cursor = 0
        batches: list[list[int]] = []

        for _ in range(num_batches):
            selected: list[int] = []
            while len(selected) < draws_per_batch:
                if cursor == len(cycle):
                    cycle = pool.copy()
                    rng.shuffle(cycle)
                    cursor = 0

                    # At a cycle boundary, consume unseen-in-this-batch items
                    # first whenever the role has enough distinct samples.
                    if selected and len(pool) >= draws_per_batch:
                        selected_set = set(selected)
                        cycle = [
                            index for index in cycle if index not in selected_set
                        ] + [index for index in cycle if index in selected_set]

                needed = draws_per_batch - len(selected)
                available = len(cycle) - cursor
                take = min(needed, available)
                selected.extend(cycle[cursor : cursor + take])
                cursor += take

            batches.append(selected)

        return batches

    def _rank_pool(
        self,
        indices: Sequence[int],
        rng: random.Random,
    ) -> list[int]:
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if len(shuffled) >= self.num_replicas:
            return shuffled[self.rank :: self.num_replicas]

        # Cross-rank reuse is unavoidable when a role has fewer samples than
        # ranks. Give each rank one deterministic sample so iteration remains
        # finite and every local batch can still satisfy its composition.
        return [shuffled[self.rank % len(shuffled)]]

    def _local_batches(self) -> tuple[list[list[int]], list[list[int]]]:
        rng = random.Random(self.seed + self.epoch)
        primary_pool = self._rank_pool(self._primary_indices, rng)
        primary_batches = self._draw_global_batches(
            primary_pool,
            self.primary_per_batch,
            self.num_batches_per_epoch,
            rng,
        )
        auxiliary_parts: list[list[list[int]]] = []
        for role, indices in self._auxiliary_indices_by_role.items():
            draws = self._auxiliary_draws_by_role.get(role, 0)
            if draws < 1:
                continue
            pool = self._rank_pool(indices, rng)
            auxiliary_parts.append(
                self._draw_global_batches(
                    pool,
                    draws,
                    self.num_batches_per_epoch,
                    rng,
                )
            )
        if not auxiliary_parts:
            auxiliary_batches = [[] for _ in range(self.num_batches_per_epoch)]
        else:
            auxiliary_batches = [
                [index for part in auxiliary_parts for index in part[batch_index]]
                for batch_index in range(self.num_batches_per_epoch)
            ]
        return primary_batches, auxiliary_batches

    def __iter__(self) -> Iterator[list[int]]:
        primary_batches, auxiliary_batches = self._local_batches()

        for batch_index in range(
            self.resume_batch_offset, self.num_batches_per_epoch
        ):
            yield primary_batches[batch_index] + auxiliary_batches[batch_index]

    def __len__(self) -> int:
        return self.num_batches_per_epoch - self.resume_batch_offset


DistributedRoleBalancedBatchSampler = RoleBalancedBatchSampler


__all__ = [
    "DistributedRoleBalancedBatchSampler",
    "RoleBalancedBatchSampler",
]
