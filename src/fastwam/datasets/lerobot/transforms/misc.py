from typing import List

import torch


class WrapStateAngle:
    def __init__(self, keys: List[str]):
        self.keys = keys
    
    @staticmethod
    def _wrap(x):
        return torch.atan2(torch.sin(x), torch.cos(x))

    def forward(self, batch):
        for k in self.keys:
            batch["state"][k] = self._wrap(batch["state"][k])
        return batch
    
    def backward(self, batch):
        return batch


class ConcatStateToAction:
    """Append selected state fields to an action field along the feature axis."""

    def __init__(
        self,
        action_key: str = "default",
        state_keys: List[str] | None = None,
        raw_action_dim: int | None = None,
        future_state: bool = True,
        state_first: bool = False,
    ):
        self.action_key = action_key
        self.state_keys = state_keys
        self.raw_action_dim = raw_action_dim
        self.future_state = future_state
        self.state_first = state_first

    def forward(self, batch):
        if "action" not in batch:
            return batch
        if self.action_key not in batch["action"]:
            raise KeyError(f"Missing action key `{self.action_key}`.")

        action = batch["action"][self.action_key]
        if action.ndim not in (2, 3):
            raise ValueError(
                f"`action[{self.action_key}]` must be [T,D] or [N,T,D], got {tuple(action.shape)}."
            )

        state_keys = self.state_keys or list(batch["state"].keys())
        state_parts = [self._align_state(batch["state"][key], action) for key in state_keys]
        if self.state_first:
            batch["action"][self.action_key] = torch.cat([*state_parts, action], dim=-1)
        else:
            batch["action"][self.action_key] = torch.cat([action, *state_parts], dim=-1)
        return batch

    def backward(self, batch):
        if "action" not in batch or self.action_key not in batch["action"]:
            return batch
        if self.raw_action_dim is None:
            raise ValueError("`raw_action_dim` is required to invert ConcatStateToAction.")
        if self.state_first:
            batch["action"][self.action_key] = batch["action"][self.action_key][..., -self.raw_action_dim :]
        else:
            batch["action"][self.action_key] = batch["action"][self.action_key][..., : self.raw_action_dim]
        return batch

    def _align_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.ndim != action.ndim:
            raise ValueError(
                f"State/action rank mismatch for ConcatStateToAction: {tuple(state.shape)} vs {tuple(action.shape)}."
            )

        action_steps = action.shape[-2]
        state_steps = state.shape[-2]
        if self.future_state and state_steps >= action_steps + 1:
            return state[..., 1 : action_steps + 1, :]
        if state_steps >= action_steps:
            return state[..., :action_steps, :]
        if state_steps == 1:
            return state.expand(*state.shape[:-2], action_steps, state.shape[-1])
        raise ValueError(
            "Cannot align state to action horizon: "
            f"state steps={state_steps}, action steps={action_steps}."
        )