from typing import Any, Dict, Optional

import torch

from .action_dit import ActionDiT


class StateDiT(ActionDiT):
    """DiT expert for low-dimensional state/proprioception condition tokens.

    StateDiT intentionally mirrors ActionDiT's low-dimensional token processing
    so it can join the same MoT stack. FastWAM uses the current state as a
    condition branch, analogous to current-frame video conditioning; it does not
    denoise or predict state targets.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__(
            action_dim=state_dim,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            text_dim=text_dim,
            freq_dim=freq_dim,
            eps=eps,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            num_layers=num_layers,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.state_dim = int(state_dim)

    @classmethod
    def from_pretrained(
        cls,
        state_dit_config: dict[str, Any],
        state_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "StateDiT":
        if state_dit_config is None:
            raise ValueError("`state_dit_config` is required for StateDiT.from_pretrained().")
        cfg = dict(state_dit_config)
        if "state_dim" not in cfg:
            raise ValueError("`state_dit_config['state_dim']` is required.")

        action_cfg = dict(cfg)
        action_cfg["action_dim"] = action_cfg.pop("state_dim")
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_cfg,
            action_dit_pretrained_path=state_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )

        state_expert = cls(**cfg).to(device=device, dtype=torch_dtype)
        state_expert.load_state_dict(action_expert.state_dict(), strict=True)
        return state_expert

    def pre_dit(
        self,
        state_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        return super().pre_dit(
            action_tokens=state_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
