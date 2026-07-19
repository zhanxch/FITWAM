from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_feature_tensor(
    name: str,
    value: torch.Tensor,
    *,
    feature_dim: int,
) -> tuple[int, int]:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"`{name}` must be a torch.Tensor, got {type(value)}")
    if value.ndim != 3:
        raise ValueError(
            f"`{name}` must have shape [batch, tokens, features], got {tuple(value.shape)}"
        )
    if value.shape[-1] != feature_dim:
        raise ValueError(
            f"`{name}` feature dimension must be {feature_dim}, got {value.shape[-1]}"
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"`{name}` must contain only finite values")
    return int(value.shape[0]), int(value.shape[1])


def _valid_mask(
    name: str,
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    if mask is None:
        return torch.ones(
            batch_size,
            num_tokens,
            dtype=torch.bool,
            device=device,
        )
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"`{name}` must be a torch.Tensor, got {type(mask)}")
    if mask.shape != (batch_size, num_tokens):
        raise ValueError(
            f"`{name}` must have shape {(batch_size, num_tokens)}, got {tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        raise TypeError(f"`{name}` must have dtype torch.bool, got {mask.dtype}")
    return mask.to(device=device)


def _sinusoidal_positions(
    length: int,
    hidden_dim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, hidden_dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / hidden_dim)
    )
    angles = positions * frequencies.unsqueeze(0)
    encoding = torch.zeros(length, hidden_dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


class TrajectoryTeacher(nn.Module):
    """Encode a masked action trajectory into one normalized target embedding."""

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        *,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if action_dim <= 0 or hidden_dim <= 0 or embedding_dim <= 0:
            raise ValueError("`action_dim`, `hidden_dim`, and `embedding_dim` must be positive")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                f"`hidden_dim` ({hidden_dim}) must be divisible by `num_heads` ({num_heads})"
            )
        if num_layers <= 0:
            raise ValueError(f"`num_layers` must be positive, got {num_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout}")

        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.eps = float(eps)

        self.action_projection = nn.Linear(action_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim or hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trajectory_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
            enable_nested_tensor=False,
        )
        self.bottleneck_query = nn.Parameter(
            torch.randn(1, 1, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.pooler = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(
        self,
        actions: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_steps = _validate_feature_tensor(
            "actions",
            actions,
            feature_dim=self.action_dim,
        )
        if num_steps == 0:
            raise ValueError("`actions` must contain at least one timestep")
        mask = _valid_mask(
            "valid_mask",
            valid_mask,
            batch_size=batch_size,
            num_tokens=num_steps,
            device=actions.device,
        )
        if not mask.any(dim=1).all():
            raise ValueError("Each trajectory must contain at least one valid timestep")

        masked_actions = torch.where(mask.unsqueeze(-1), actions, torch.zeros_like(actions))
        tokens = self.action_projection(masked_actions)
        tokens = tokens + _sinusoidal_positions(
            num_steps,
            self.hidden_dim,
            device=tokens.device,
            dtype=tokens.dtype,
        ).unsqueeze(0)
        encoded = self.trajectory_encoder(
            tokens,
            src_key_padding_mask=~mask,
        )

        query = self.bottleneck_query.expand(batch_size, -1, -1)
        pooled, _ = self.pooler(
            query,
            encoded,
            encoded,
            key_padding_mask=~mask,
            need_weights=False,
        )
        embedding = self.output_projection(self.output_norm(pooled[:, 0]))
        return F.normalize(embedding, dim=-1, eps=self.eps)


class ObservationSteerStudent(nn.Module):
    """Predict one steer embedding from current video and context tokens."""

    def __init__(
        self,
        video_dim: int,
        context_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if min(video_dim, context_dim, hidden_dim, embedding_dim) <= 0:
            raise ValueError(
                "`video_dim`, `context_dim`, `hidden_dim`, and `embedding_dim` must be positive"
            )
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                f"`hidden_dim` ({hidden_dim}) must be divisible by `num_heads` ({num_heads})"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout}")

        self.video_dim = int(video_dim)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.eps = float(eps)

        self.video_projection = nn.Linear(video_dim, hidden_dim)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.token_type_embedding = nn.Parameter(torch.zeros(2, hidden_dim))
        self.bottleneck_query = nn.Parameter(
            torch.randn(1, 1, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(
        self,
        video_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        *,
        video_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, num_video_tokens = _validate_feature_tensor(
            "video_tokens",
            video_tokens,
            feature_dim=self.video_dim,
        )
        context_batch_size, num_context_tokens = _validate_feature_tensor(
            "context_tokens",
            context_tokens,
            feature_dim=self.context_dim,
        )
        if context_batch_size != batch_size:
            raise ValueError(
                "`video_tokens` and `context_tokens` must have the same batch size, "
                f"got {batch_size} and {context_batch_size}"
            )
        if num_video_tokens + num_context_tokens == 0:
            raise ValueError("At least one video or context token is required")

        valid_video = _valid_mask(
            "video_mask",
            video_mask,
            batch_size=batch_size,
            num_tokens=num_video_tokens,
            device=video_tokens.device,
        )
        valid_context = _valid_mask(
            "context_mask",
            context_mask,
            batch_size=batch_size,
            num_tokens=num_context_tokens,
            device=context_tokens.device,
        )
        valid_tokens = torch.cat([valid_video, valid_context], dim=1)
        if not valid_tokens.any(dim=1).all():
            raise ValueError("Each sample must contain at least one valid video or context token")

        masked_video = torch.where(
            valid_video.unsqueeze(-1),
            video_tokens,
            torch.zeros_like(video_tokens),
        )
        masked_context = torch.where(
            valid_context.unsqueeze(-1),
            context_tokens,
            torch.zeros_like(context_tokens),
        )
        video_hidden = (
            self.video_projection(masked_video) + self.token_type_embedding[0]
        )
        context_hidden = (
            self.context_projection(masked_context) + self.token_type_embedding[1]
        )
        source = torch.cat([video_hidden, context_hidden], dim=1)

        query = self.bottleneck_query.expand(batch_size, -1, -1)
        pooled, _ = self.cross_attention(
            query,
            source,
            source,
            key_padding_mask=~valid_tokens,
            need_weights=False,
        )
        embedding = self.output_projection(self.output_norm(pooled[:, 0]))
        return F.normalize(embedding, dim=-1, eps=self.eps)


class ZeroInitSteerResidual(nn.Module):
    """Project a steer embedding into an exact-zero additive action residual."""

    def __init__(self, embedding_dim: int, action_hidden_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0 or action_hidden_dim <= 0:
            raise ValueError("`embedding_dim` and `action_hidden_dim` must be positive")
        self.embedding_dim = int(embedding_dim)
        self.action_hidden_dim = int(action_hidden_dim)
        self.projection = nn.Linear(embedding_dim, action_hidden_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, steer_embedding: torch.Tensor) -> torch.Tensor:
        if steer_embedding.ndim != 2:
            raise ValueError(
                "`steer_embedding` must have shape [batch, embedding_dim], "
                f"got {tuple(steer_embedding.shape)}"
            )
        if steer_embedding.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"`steer_embedding` last dimension must be {self.embedding_dim}, "
                f"got {steer_embedding.shape[-1]}"
            )
        if not torch.isfinite(steer_embedding).all():
            raise ValueError("`steer_embedding` must contain only finite values")
        return self.projection(steer_embedding)

    def add_to_action_tokens(
        self,
        action_tokens: torch.Tensor,
        steer_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if action_tokens.ndim != 3:
            raise ValueError(
                "`action_tokens` must have shape [batch, tokens, hidden_dim], "
                f"got {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[-1] != self.action_hidden_dim:
            raise ValueError(
                f"`action_tokens` last dimension must be {self.action_hidden_dim}, "
                f"got {action_tokens.shape[-1]}"
            )
        if action_tokens.shape[0] != steer_embedding.shape[0]:
            raise ValueError(
                "`action_tokens` and `steer_embedding` must have the same batch size"
            )
        if not torch.isfinite(action_tokens).all():
            raise ValueError("`action_tokens` must contain only finite values")
        residual = self(steer_embedding)
        return action_tokens + residual.unsqueeze(1)


def weighted_pair_loss(
    student_embedding: torch.Tensor,
    success_target: torch.Tensor,
    failure_target: torch.Tensor,
    sample_weight: torch.Tensor,
    *,
    margin: float = 0.2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pull the student toward success and keep it a margin away from failure."""

    embeddings = {
        "student_embedding": student_embedding,
        "success_target": success_target,
        "failure_target": failure_target,
    }
    for name, value in embeddings.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"`{name}` must be a torch.Tensor, got {type(value)}")
        if value.ndim != 2:
            raise ValueError(
                f"`{name}` must have shape [batch, embedding_dim], got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"`{name}` must contain only finite values")
    if success_target.shape != student_embedding.shape:
        raise ValueError(
            "`success_target` must match `student_embedding`, got "
            f"{tuple(success_target.shape)} and {tuple(student_embedding.shape)}"
        )
    if failure_target.shape != student_embedding.shape:
        raise ValueError(
            "`failure_target` must match `student_embedding`, got "
            f"{tuple(failure_target.shape)} and {tuple(student_embedding.shape)}"
        )
    if not isinstance(sample_weight, torch.Tensor):
        raise TypeError(
            f"`sample_weight` must be a torch.Tensor, got {type(sample_weight)}"
        )
    if sample_weight.ndim != 1 or sample_weight.shape[0] != student_embedding.shape[0]:
        raise ValueError(
            "`sample_weight` must have shape [batch], got "
            f"{tuple(sample_weight.shape)} for batch {student_embedding.shape[0]}"
        )
    if not torch.isfinite(sample_weight).all():
        raise ValueError("`sample_weight` must contain only finite values")
    if (sample_weight < 0).any():
        raise ValueError("`sample_weight` must be non-negative")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(f"`margin` must be finite and non-negative, got {margin}")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"`eps` must be finite and positive, got {eps}")

    weights = sample_weight.detach().to(
        device=student_embedding.device,
        dtype=student_embedding.dtype,
    )
    weight_sum = weights.sum()
    if weight_sum.item() <= 0:
        raise ValueError("`sample_weight` must contain at least one positive value")

    student = F.normalize(student_embedding, dim=-1, eps=eps)
    success = F.normalize(success_target.detach(), dim=-1, eps=eps)
    failure = F.normalize(failure_target.detach(), dim=-1, eps=eps)

    success_similarity = (student * success).sum(dim=-1)
    failure_similarity = (student * failure).sum(dim=-1)
    pull_loss = 1.0 - success_similarity
    margin_loss = F.relu(failure_similarity - success_similarity + margin)
    return (weights * (pull_loss + margin_loss)).sum() / weight_sum


__all__ = [
    "ObservationSteerStudent",
    "TrajectoryTeacher",
    "ZeroInitSteerResidual",
    "weighted_pair_loss",
]
