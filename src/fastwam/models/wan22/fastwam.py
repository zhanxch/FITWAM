from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .offline_steer import (
    ObservationSteerStudent,
    ZeroInitSteerResidual,
    weighted_pair_loss,
)
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .state_dit import StateDiT

logger = get_logger(__name__)


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        state_expert: Optional[StateDiT] = None,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        outcome_num_classes: int = 0,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        state_train_shift: float = 5.0,
        state_infer_shift: float = 5.0,
        state_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_state: float = 1.0,
        offline_steer_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.state_expert = state_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None
        self.outcome_num_classes = int(outcome_num_classes or 0)
        if self.outcome_num_classes > 0:
            self.outcome_encoder = nn.Embedding(self.outcome_num_classes, self.text_dim).to(torch_dtype)
        else:
            self.outcome_encoder = None
        offline_steer_config = dict(offline_steer_config or {})
        supported_steer_keys = {
            "enabled",
            "hidden_dim",
            "embedding_dim",
            "num_heads",
            "dropout",
            "detach_backbone_inputs",
            "pair_loss_weight",
            "pair_loss_margin",
            "pair_loss_warmup_steps",
        }
        unknown_steer_keys = set(offline_steer_config) - supported_steer_keys
        if unknown_steer_keys:
            raise ValueError(
                f"Unsupported `offline_steer` keys: {sorted(unknown_steer_keys)}"
            )
        self.offline_steer_enabled = bool(offline_steer_config.get("enabled", False))
        self.offline_steer_config = {
            "enabled": self.offline_steer_enabled,
            "hidden_dim": int(offline_steer_config.get("hidden_dim", 256)),
            "embedding_dim": int(offline_steer_config.get("embedding_dim", 256)),
            "num_heads": int(offline_steer_config.get("num_heads", 4)),
            "dropout": float(offline_steer_config.get("dropout", 0.0)),
            "detach_backbone_inputs": bool(
                offline_steer_config.get("detach_backbone_inputs", True)
            ),
            "pair_loss_weight": float(
                offline_steer_config.get("pair_loss_weight", 0.0)
            ),
            "pair_loss_margin": float(
                offline_steer_config.get("pair_loss_margin", 0.2)
            ),
            "pair_loss_warmup_steps": int(
                offline_steer_config.get("pair_loss_warmup_steps", 0)
            ),
        }
        if self.offline_steer_config["pair_loss_weight"] < 0.0:
            raise ValueError("`offline_steer.pair_loss_weight` must be non-negative.")
        if self.offline_steer_config["pair_loss_margin"] < 0.0:
            raise ValueError("`offline_steer.pair_loss_margin` must be non-negative.")
        if self.offline_steer_config["pair_loss_warmup_steps"] < 0:
            raise ValueError(
                "`offline_steer.pair_loss_warmup_steps` must be non-negative."
            )
        if self.offline_steer_enabled:
            if not hasattr(self.video_expert, "hidden_dim"):
                raise ValueError(
                    "Enabled `offline_steer` requires `video_expert.hidden_dim`."
                )
            if not hasattr(self.action_expert, "hidden_dim"):
                raise ValueError(
                    "Enabled `offline_steer` requires `action_expert.hidden_dim`."
                )
            self.offline_steer_student = ObservationSteerStudent(
                video_dim=int(self.video_expert.hidden_dim),
                context_dim=self.text_dim,
                hidden_dim=self.offline_steer_config["hidden_dim"],
                embedding_dim=self.offline_steer_config["embedding_dim"],
                num_heads=self.offline_steer_config["num_heads"],
                dropout=self.offline_steer_config["dropout"],
            ).to(dtype=torch_dtype)
            self.offline_steer_residual = ZeroInitSteerResidual(
                embedding_dim=self.offline_steer_config["embedding_dim"],
                action_hidden_dim=int(self.action_expert.hidden_dim),
            ).to(dtype=torch_dtype)
        else:
            self.offline_steer_student = None
            self.offline_steer_residual = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        if self.state_expert is not None:
            self.train_state_scheduler = WanContinuousFlowMatchScheduler(
                num_train_timesteps=state_num_train_timesteps,
                shift=state_train_shift,
            )
            self.infer_state_scheduler = WanContinuousFlowMatchScheduler(
                num_train_timesteps=state_num_train_timesteps,
                shift=state_infer_shift,
            )
        else:
            self.train_state_scheduler = None
            self.infer_state_scheduler = None
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.loss_lambda_state = float(loss_lambda_state)

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        outcome_num_classes: int = 0,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        state_dit_config: dict[str, Any] | None = None,
        state_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        state_train_shift: float = 5.0,
        state_infer_shift: float = 5.0,
        state_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_state: float = 1.0,
        offline_steer_config: Optional[dict[str, Any]] = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        state_expert = None
        if state_dit_config is not None:
            state_expert = StateDiT.from_pretrained(
                state_dit_config=state_dit_config,
                state_dit_pretrained_path=state_dit_pretrained_path,
                skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
                device=device,
                torch_dtype=torch_dtype,
            )
            if int(state_expert.num_heads) != int(video_expert.num_heads):
                raise ValueError("StateDiT `num_heads` must match video expert for MoT mixed attention.")
            if int(state_expert.attn_head_dim) != int(video_expert.attn_head_dim):
                raise ValueError("StateDiT `attn_head_dim` must match video expert for MoT mixed attention.")
            if int(len(state_expert.blocks)) != int(len(video_expert.blocks)):
                raise ValueError("StateDiT `num_layers` must match video expert.")

        mixtures = {"video": video_expert, "action": action_expert}
        if state_expert is not None:
            mixtures["state"] = state_expert
        mot = MoT(
            mixtures=mixtures,
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            state_expert=state_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            outcome_num_classes=outcome_num_classes,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            state_train_shift=state_train_shift,
            state_infer_shift=state_infer_shift,
            state_num_train_timesteps=state_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            loss_lambda_state=loss_lambda_state,
            offline_steer_config=offline_steer_config,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        if state_expert is not None:
            model.model_paths["state_dit_backbone"] = (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else state_dit_pretrained_path
            )
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_outcome_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        outcome_flag: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.outcome_encoder is None:
            return context, context_mask
        if outcome_flag is None:
            outcome_flag = torch.zeros((context.shape[0],), dtype=torch.long, device=context.device)
        if not isinstance(outcome_flag, torch.Tensor):
            outcome_flag = torch.as_tensor(outcome_flag, dtype=torch.long, device=context.device)
        outcome_flag = outcome_flag.to(device=context.device, dtype=torch.long).view(-1)
        if outcome_flag.shape[0] != context.shape[0]:
            raise ValueError(
                f"`outcome_flag` must have one value per batch element, got {tuple(outcome_flag.shape)} for batch={context.shape[0]}"
            )
        if bool((outcome_flag < 0).any().item()) or bool((outcome_flag >= self.outcome_num_classes).any().item()):
            raise ValueError(
                f"`outcome_flag` values must be in [0, {self.outcome_num_classes - 1}], got {outcome_flag.detach().cpu().tolist()}"
            )
        outcome_token = self.outcome_encoder(outcome_flag).to(dtype=context.dtype).unsqueeze(1)
        outcome_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, outcome_token], dim=1),
            torch.cat([context_mask, outcome_mask], dim=1),
        )

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    def _compute_offline_steer_embedding(
        self,
        *,
        video_tokens: torch.Tensor,
        video_tokens_per_frame: int,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.offline_steer_enabled:
            return None
        if self.offline_steer_student is None:
            raise RuntimeError("`offline_steer` is enabled but the Student is missing.")
        if video_tokens_per_frame <= 0 or video_tokens.shape[1] < video_tokens_per_frame:
            raise ValueError(
                "`video_tokens_per_frame` must select a non-empty current observation, "
                f"got {video_tokens_per_frame} for {video_tokens.shape[1]} video tokens."
            )
        current_video_tokens = video_tokens[:, :video_tokens_per_frame]
        if self.offline_steer_config["detach_backbone_inputs"]:
            current_video_tokens = current_video_tokens.detach()
            context = context.detach()
        current_video_mask = torch.ones(
            current_video_tokens.shape[:2],
            dtype=torch.bool,
            device=current_video_tokens.device,
        )
        return self.offline_steer_student(
            current_video_tokens,
            context,
            video_mask=current_video_mask,
            context_mask=context_mask,
        )

    def _validate_explicit_steer_embedding(
        self,
        steer_embedding: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        if not self.offline_steer_enabled:
            raise ValueError(
                "`steer_embedding` was provided while `offline_steer.enabled=false`."
            )
        expected_dim = self.offline_steer_config["embedding_dim"]
        if steer_embedding.ndim != 2 or tuple(steer_embedding.shape) != (
            batch_size,
            expected_dim,
        ):
            raise ValueError(
                "`steer_embedding` must have shape "
                f"[{batch_size}, {expected_dim}], got {tuple(steer_embedding.shape)}."
            )
        if not torch.isfinite(steer_embedding).all():
            raise ValueError("`steer_embedding` must contain only finite values.")
        return steer_embedding.to(device=self.device, dtype=self.torch_dtype)

    @staticmethod
    def _resolve_steer_inference_mode(
        steer_inference_mode: Optional[str],
        *,
        steer_embedding: Optional[torch.Tensor],
    ) -> str:
        """Resolve inference-only steer control without changing legacy callers."""
        if steer_inference_mode is None:
            return "explicit" if steer_embedding is not None else "learned"
        mode = str(steer_inference_mode).strip().lower()
        if mode not in {"learned", "bypass", "explicit"}:
            raise ValueError(
                "`steer_inference_mode` must be one of "
                "['learned', 'bypass', 'explicit'], "
                f"got {steer_inference_mode!r}."
            )
        if mode == "bypass" and steer_embedding is not None:
            raise ValueError("`bypass` mode does not accept `steer_embedding`.")
        if mode == "explicit" and steer_embedding is None:
            raise ValueError("`explicit` mode requires `steer_embedding`.")
        if mode == "learned" and steer_embedding is not None:
            raise ValueError(
                "`learned` mode computes its own embedding; use `explicit` for a "
                "provided embedding."
            )
        return mode

    def _inject_offline_steer(
        self,
        action_pre: dict[str, Any],
        *,
        steer_embedding: Optional[torch.Tensor],
        steer_inference_mode: str = "learned",
    ) -> None:
        if steer_inference_mode == "bypass":
            if steer_embedding is not None:
                raise ValueError("`bypass` mode does not accept `steer_embedding`.")
            return
        if steer_inference_mode not in {"learned", "explicit"}:
            raise ValueError(
                "Internal steer mode must be learned, bypass, or explicit; "
                f"got {steer_inference_mode!r}."
            )
        if not self.offline_steer_enabled:
            if steer_embedding is not None:
                raise ValueError(
                    "`steer_embedding` was provided while `offline_steer.enabled=false`."
                )
            return
        if self.offline_steer_residual is None or steer_embedding is None:
            raise RuntimeError(
                "Enabled `offline_steer` requires both a residual module and steer embedding."
            )
        action_pre["tokens"] = self.offline_steer_residual.add_to_action_tokens(
            action_pre["tokens"],
            steer_embedding,
        )

    def _compute_inference_steer_embedding(
        self,
        *,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        steer_embedding: Optional[torch.Tensor] = None,
        steer_inference_mode: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        mode = self._resolve_steer_inference_mode(
            steer_inference_mode,
            steer_embedding=steer_embedding,
        )
        if mode == "bypass":
            return None
        if mode == "explicit":
            return self._validate_explicit_steer_embedding(
                steer_embedding,
                batch_size=first_frame_latents.shape[0],
            )
        if not self.offline_steer_enabled:
            return None
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        return self._compute_offline_steer_embedding(
            video_tokens=video_pre["tokens"],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            context=context,
            context_mask=context_mask,
        )

    def _prepare_state_condition(self, proprio: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self.state_expert is None or proprio is None:
            return None
        if proprio.ndim == 1:
            proprio = proprio.view(1, 1, -1)
        elif proprio.ndim == 2:
            if proprio.shape[0] == 1:
                proprio = proprio.unsqueeze(1)
            else:
                proprio = proprio[:1].unsqueeze(0)
        elif proprio.ndim == 3:
            proprio = proprio[:, :1, :]
        else:
            raise ValueError(f"`proprio` must be [D], [T,D], [1,D], or [B,T,D], got {tuple(proprio.shape)}")
        if proprio.shape[0] != 1:
            raise ValueError(f"StateDiT inference currently expects batch size 1, got {proprio.shape[0]}")
        if proprio.shape[2] != self.state_expert.state_dim:
            raise ValueError(f"`proprio` last dim must be {self.state_expert.state_dim}, got {proprio.shape[2]}")
        return proprio.to(device=self.device, dtype=self.torch_dtype)

    def _state_condition_pre_dit(
        self,
        state: Optional[torch.Tensor],
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> Optional[dict[str, Any]]:
        if self.state_expert is None or state is None:
            return None
        timestep_state = torch.zeros((state.shape[0],), device=self.device, dtype=state.dtype)
        return self.state_expert.pre_dit(
            state_tokens=state,
            timestep=timestep_state,
            context=context,
            context_mask=context_mask,
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        state_is_pad = sample.get("proprio_is_pad", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        steer_context = context if self.offline_steer_enabled else None
        steer_context_mask = context_mask if self.offline_steer_enabled else None
        context, context_mask = self._append_outcome_to_context(
            context=context,
            context_mask=context_mask,
            outcome_flag=sample.get("outcome_flag", None),
        )
        proprio_seq = proprio
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio_token_source = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio_token_source.to(device=self.device, dtype=self.torch_dtype),
            )
            if steer_context is not None and steer_context_mask is not None:
                steer_context, steer_context_mask = self._append_proprio_to_context(
                    context=steer_context,
                    context_mask=steer_context_mask,
                    proprio=proprio_token_source.to(
                        device=self.device,
                        dtype=self.torch_dtype,
                    ),
                )
        if self.state_expert is not None:
            if proprio_seq is None:
                raise ValueError("`sample['proprio']` is required when `state_expert` is enabled.")
            if proprio_seq.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio_seq.shape)}")
            if proprio_seq.shape[2] != self.state_expert.state_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.state_expert.state_dim}, got {proprio_seq.shape[2]}"
                )
            if state_is_pad is not None:
                if state_is_pad.ndim != 2:
                    raise ValueError(
                        f"`sample['proprio_is_pad']` must be 2D [B, T], got shape {tuple(state_is_pad.shape)}"
                    )
                if state_is_pad.shape[0] != batch_size or state_is_pad.shape[1] != proprio_seq.shape[1]:
                    raise ValueError(
                        "`sample['proprio_is_pad']` shape mismatch: "
                        f"got {tuple(state_is_pad.shape)} vs expected ({batch_size}, {proprio_seq.shape[1]})"
                    )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        state = None
        if self.state_expert is not None:
            state = proprio_seq.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if state_is_pad is not None:
            state_is_pad = state_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "steer_context": steer_context,
            "steer_context_mask": steer_context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "state": state,
            "action_is_pad": action_is_pad,
            "state_is_pad": state_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    @torch.no_grad()
    def _build_conditioned_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        state_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        base_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        if state_seq_len is None or state_seq_len <= 0:
            return base_mask

        base_seq_len = int(base_mask.shape[0])
        total_seq_len = base_seq_len + int(state_seq_len)
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)
        mask[:base_seq_len, :base_seq_len] = base_mask
        # Video branch remains identical to original FastWAM: it cannot read state.
        # Action only gets the current/first state token, never noisy future state.
        action_start = int(video_seq_len)
        state_start = base_seq_len
        mask[action_start:base_seq_len, state_start : state_start + 1] = True

        state_mask = torch.ones((state_seq_len, state_seq_len), dtype=torch.bool, device=device)
        state_mask[:1, 1:] = False
        mask[state_start:, state_start:] = state_mask
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _compute_offline_pair_loss(
        self,
        *,
        steer_embedding: Optional[torch.Tensor],
        sample: dict[str, Any],
        require_pair_fields: bool = True,
    ) -> tuple[Optional[torch.Tensor], float, float]:
        pair_loss_weight = self.offline_steer_config["pair_loss_weight"]
        if not self.offline_steer_enabled or pair_loss_weight <= 0.0:
            return None, 0.0, 0.0
        if steer_embedding is None:
            raise RuntimeError("Offline pair loss requires a Student steer embedding.")

        sample_weight = sample.get("pair_weight")
        success_target = sample.get("steer_success_target")
        failure_target = sample.get("steer_failure_target")
        if sample_weight is None:
            if not require_pair_fields:
                return None, 0.0, 0.0
            raise ValueError(
                "Enabled offline pair loss requires `sample['pair_weight']`."
            )
        sample_weight = sample_weight.to(
            device=steer_embedding.device,
            dtype=steer_embedding.dtype,
            non_blocking=True,
        ).view(-1)
        if sample_weight.shape[0] != steer_embedding.shape[0]:
            raise ValueError(
                "`sample['pair_weight']` must have one value per batch element."
            )
        positive = sample_weight > 0
        if not positive.any():
            zero = steer_embedding.float().sum() * 0.0
            return zero, 0.0, 0.0
        if success_target is None or failure_target is None:
            raise ValueError(
                "Positive pair weights require `steer_success_target` and "
                "`steer_failure_target`."
            )

        success_target = success_target.to(
            device=steer_embedding.device,
            dtype=steer_embedding.dtype,
            non_blocking=True,
        )
        failure_target = failure_target.to(
            device=steer_embedding.device,
            dtype=steer_embedding.dtype,
            non_blocking=True,
        )
        if success_target.shape != steer_embedding.shape:
            raise ValueError(
                "`steer_success_target` must match the Student embedding shape, "
                f"got {tuple(success_target.shape)} and "
                f"{tuple(steer_embedding.shape)}."
            )
        if failure_target.shape != steer_embedding.shape:
            raise ValueError(
                "`steer_failure_target` must match the Student embedding shape, "
                f"got {tuple(failure_target.shape)} and "
                f"{tuple(steer_embedding.shape)}."
            )

        loss = weighted_pair_loss(
            steer_embedding[positive],
            success_target[positive],
            failure_target[positive],
            sample_weight[positive],
            margin=self.offline_steer_config["pair_loss_margin"],
        )
        return (
            loss,
            float(sample_weight[positive].detach().sum().item()),
            float(positive.float().detach().mean().item()),
        )

    def training_loss(
        self,
        sample,
        tiled: bool = False,
        *,
        pair_loss_scale: float = 1.0,
        pair_loss_ddp_scale: float = 1.0,
    ):
        if not 0.0 <= float(pair_loss_scale) <= 1.0:
            raise ValueError(
                f"`pair_loss_scale` must be in [0, 1], got {pair_loss_scale}."
            )
        if float(pair_loss_ddp_scale) < 0.0:
            raise ValueError(
                "`pair_loss_ddp_scale` must be non-negative, got "
                f"{pair_loss_ddp_scale}."
            )
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        steer_context = inputs["steer_context"]
        steer_context_mask = inputs["steer_context_mask"]
        action = inputs["action"]
        state = inputs["state"]
        action_is_pad = inputs["action_is_pad"]
        state_is_pad = inputs["state_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        noisy_state = None
        target_state = None
        timestep_state = None
        if self.state_expert is not None:
            if state is None or self.train_state_scheduler is None:
                raise ValueError("StateDiT is enabled but state inputs or scheduler are missing.")
            if state.shape[1] <= 1:
                raise ValueError(f"`sample['proprio']` must include current and future states, got {state.shape[1]} steps.")
            noise_state = torch.randn_like(state)
            timestep_state = self.train_state_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=state.dtype,
            )
            noisy_state = self.train_state_scheduler.add_noise(state, noise_state, timestep_state)
            noisy_state[:, :1, :] = state[:, :1, :]
            target_state = self.train_state_scheduler.training_target(state, noise_state, timestep_state)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        steer_embedding = None
        if self.offline_steer_enabled:
            if steer_context is None or steer_context_mask is None:
                raise RuntimeError("Enabled `offline_steer` requires clean context.")
            steer_embedding = self._compute_offline_steer_embedding(
                video_tokens=video_pre["tokens"],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                context=steer_context,
                context_mask=steer_context_mask,
            )
        self._inject_offline_steer(
            action_pre,
            steer_embedding=steer_embedding,
        )
        state_pre = self._state_condition_pre_dit(
            state=noisy_state,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]
        state_tokens = None if state_pre is None else state_pre["tokens"]

        attention_mask = self._build_conditioned_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
            state_seq_len=None if state_tokens is None else state_tokens.shape[1],
        )
        embeds_all = {
            "video": video_tokens,
            "action": action_tokens,
        }
        freqs_all = {
            "video": video_pre["freqs"],
            "action": action_pre["freqs"],
        }
        context_all = {
            "video": {
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            "action": {
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
        }
        t_mod_all = {
            "video": video_pre["t_mod"],
            "action": action_pre["t_mod"],
        }
        if state_pre is not None:
            embeds_all["state"] = state_tokens
            freqs_all["state"] = state_pre["freqs"]
            context_all["state"] = {
                "context": state_pre["context"],
                "mask": state_pre["context_mask"],
            }
            t_mod_all["state"] = state_pre["t_mod"]
        tokens_out = self.mot(
            embeds_all=embeds_all,
            attention_mask=attention_mask,
            freqs_all=freqs_all,
            context_all=context_all,
            t_mod_all=t_mod_all,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        pred_state = None
        if state_pre is not None:
            pred_state = self.state_expert.post_dit(tokens_out["state"], state_pre)
        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        action_loss_sample_weight = sample.get("action_loss_weight", None)
        action_loss_enabled_frac = None
        if action_loss_sample_weight is not None:
            action_loss_sample_weight = action_loss_sample_weight.to(
                device=action_loss_per_sample.device, dtype=action_loss_per_sample.dtype, non_blocking=True
            ).view(-1)
            if action_loss_sample_weight.shape[0] != action_loss_per_sample.shape[0]:
                raise ValueError("`sample['action_loss_weight']` must have one value per batch element.")
            action_loss_enabled_frac = action_loss_sample_weight.detach().mean()
            loss_action = (
                action_loss_per_sample * action_weight * action_loss_sample_weight
            ).sum() / action_loss_sample_weight.sum().clamp(min=1.0)
        else:
            loss_action = (action_loss_per_sample * action_weight).mean()

        loss_state = None
        if pred_state is not None and target_state is not None and timestep_state is not None:
            state_loss_token = F.mse_loss(
                pred_state[:, 1:].float(),
                target_state[:, 1:].float(),
                reduction="none",
            ).mean(dim=2)
            if state_is_pad is not None:
                valid = (~state_is_pad[:, 1:]).to(device=state_loss_token.device, dtype=state_loss_token.dtype)
                valid_sum = valid.sum(dim=1).clamp(min=1.0)
                state_loss_per_sample = (state_loss_token * valid).sum(dim=1) / valid_sum
            else:
                state_loss_per_sample = state_loss_token.mean(dim=1)
            state_weight = self.train_state_scheduler.training_weight(timestep_state).to(
                state_loss_per_sample.device, dtype=state_loss_per_sample.dtype
            )
            loss_state = (state_loss_per_sample * state_weight).mean()

        loss_pair, pair_weight_sum, pair_enabled_frac = (
            self._compute_offline_pair_loss(
                steer_embedding=steer_embedding,
                sample=sample,
                require_pair_fields=float(pair_loss_scale) > 0.0,
            )
        )

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        if loss_state is not None:
            loss_total = loss_total + self.loss_lambda_state * loss_state
        scaled_pair_weight = (
            self.offline_steer_config["pair_loss_weight"]
            * float(pair_loss_scale)
            * float(pair_loss_ddp_scale)
        )
        if loss_pair is not None and scaled_pair_weight > 0.0:
            loss_total = loss_total + scaled_pair_weight * loss_pair
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        if loss_pair is not None:
            loss_dict["loss_pair_raw"] = float(loss_pair.detach().item())
            loss_dict["loss_pair"] = scaled_pair_weight * float(
                loss_pair.detach().item()
            )
            loss_dict["pair_loss_scale"] = float(pair_loss_scale)
            loss_dict["pair_loss_ddp_scale"] = float(pair_loss_ddp_scale)
            loss_dict["pair_weight_sum"] = pair_weight_sum
            loss_dict["pair_enabled_frac"] = pair_enabled_frac
        if steer_embedding is not None:
            loss_dict["steer_embedding_norm"] = float(
                steer_embedding.detach().float().norm(dim=-1).mean().item()
            )
            if self.offline_steer_residual is not None:
                loss_dict["steer_residual_norm"] = float(
                    self.offline_steer_residual(steer_embedding)
                    .detach()
                    .float()
                    .norm(dim=-1)
                    .mean()
                    .item()
                )
        if action_loss_enabled_frac is not None:
            loss_dict["action_loss_enabled_frac"] = float(action_loss_enabled_frac.item())
        outcome_flag = sample.get("outcome_flag", None)
        if outcome_flag is not None:
            loss_dict["outcome_failure_frac"] = float(
                outcome_flag.to(device=loss_total.device, dtype=torch.float32, non_blocking=True).view(-1).mean().detach().item()
            )
        if loss_state is not None:
            loss_dict["loss_state"] = self.loss_lambda_state * float(loss_state.detach().item())
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        steer_embedding: Optional[torch.Tensor] = None,
        steer_inference_mode: str = "learned",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        if (
            steer_inference_mode == "learned"
            and self.offline_steer_enabled
            and steer_embedding is None
        ):
            steer_embedding = self._compute_offline_steer_embedding(
                video_tokens=video_pre["tokens"],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                context=context,
                context_mask=context_mask,
            )
        self._inject_offline_steer(
            action_pre,
            steer_embedding=steer_embedding,
            steer_inference_mode=steer_inference_mode,
        )
        state_pre = self._state_condition_pre_dit(
            state=state,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_conditioned_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            state_seq_len=None if state_pre is None else state_pre["tokens"].shape[1],
        )
        embeds_all = {
            "video": video_pre["tokens"],
            "action": action_pre["tokens"],
        }
        freqs_all = {
            "video": video_pre["freqs"],
            "action": action_pre["freqs"],
        }
        context_all = {
            "video": {
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            "action": {
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
        }
        t_mod_all = {
            "video": video_pre["t_mod"],
            "action": action_pre["t_mod"],
        }
        if state_pre is not None:
            embeds_all["state"] = state_pre["tokens"]
            freqs_all["state"] = state_pre["freqs"]
            context_all["state"] = {
                "context": state_pre["context"],
                "mask": state_pre["context_mask"],
            }
            t_mod_all["state"] = state_pre["t_mod"]

        tokens_out = self.mot(
            embeds_all=embeds_all,
            attention_mask=attention_mask,
            freqs_all=freqs_all,
            context_all=context_all,
            t_mod_all=t_mod_all,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        state: Optional[torch.Tensor] = None,
        steer_embedding: Optional[torch.Tensor] = None,
        steer_inference_mode: str = "learned",
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        if (
            steer_inference_mode == "learned"
            and self.offline_steer_enabled
            and steer_embedding is None
        ):
            steer_embedding = self._compute_offline_steer_embedding(
                video_tokens=video_pre["tokens"],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                context=context,
                context_mask=context_mask,
            )
        self._inject_offline_steer(
            action_pre,
            steer_embedding=steer_embedding,
            steer_inference_mode=steer_inference_mode,
        )
        state_pre = self._state_condition_pre_dit(
            state=state,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_conditioned_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            state_seq_len=None if state_pre is None else state_pre["tokens"].shape[1],
        )
        embeds_all = {
            "video": video_pre["tokens"],
            "action": action_pre["tokens"],
        }
        freqs_all = {
            "video": video_pre["freqs"],
            "action": action_pre["freqs"],
        }
        context_all = {
            "video": {
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            "action": {
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
        }
        t_mod_all = {
            "video": video_pre["t_mod"],
            "action": action_pre["t_mod"],
        }
        if state_pre is not None:
            embeds_all["state"] = state_pre["tokens"]
            freqs_all["state"] = state_pre["freqs"]
            context_all["state"] = {
                "context": state_pre["context"],
                "mask": state_pre["context_mask"],
            }
            t_mod_all["state"] = state_pre["t_mod"]
        tokens_out = self.mot(
            embeds_all=embeds_all,
            attention_mask=attention_mask,
            freqs_all=freqs_all,
            context_all=context_all,
            t_mod_all=t_mod_all,
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        steer_embedding: Optional[torch.Tensor] = None,
        steer_inference_mode: str = "learned",
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        self._inject_offline_steer(
            action_pre,
            steer_embedding=steer_embedding,
            steer_inference_mode=steer_inference_mode,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        outcome_flag: Optional[torch.Tensor] = None,
        steer_embedding: Optional[torch.Tensor] = None,
        steer_inference_mode: Optional[str] = None,
        return_steer_embedding: bool = False,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        resolved_steer_mode = self._resolve_steer_inference_mode(
            steer_inference_mode,
            steer_embedding=steer_embedding,
        )
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                outcome_flag=outcome_flag.clone() if isinstance(outcome_flag, torch.Tensor) else outcome_flag,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
                steer_embedding=(
                    steer_embedding.clone()
                    if isinstance(steer_embedding, torch.Tensor)
                    else steer_embedding
                ),
                steer_inference_mode=resolved_steer_mode,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        state_cond = self._prepare_state_condition(proprio)
        proprio_context = None
        if proprio is not None and self.proprio_encoder is not None:
            if proprio.ndim == 1:
                proprio_context = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                proprio_context = proprio
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D] for proprio context, got shape {tuple(proprio.shape)}")
            if proprio_context.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio_context.shape[1]}")
            proprio_context = proprio_context.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        steer_context = context
        steer_context_mask = context_mask
        context, context_mask = self._append_outcome_to_context(
            context=context,
            context_mask=context_mask,
            outcome_flag=outcome_flag,
        )
        if proprio_context is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio_context,
            )
            steer_context, steer_context_mask = self._append_proprio_to_context(
                context=steer_context,
                context_mask=steer_context_mask,
                proprio=proprio_context,
            )

        steer_embedding = self._compute_inference_steer_embedding(
            first_frame_latents=first_frame_latents,
            context=steer_context,
            context_mask=steer_context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
            steer_embedding=steer_embedding,
            steer_inference_mode=resolved_steer_mode,
        )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
                state=state_cond,
                steer_embedding=steer_embedding,
                steer_inference_mode=resolved_steer_mode,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        result = {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }
        if return_steer_embedding:
            result["steer_embedding"] = (
                None
                if steer_embedding is None
                else steer_embedding.detach().to(device="cpu", dtype=torch.float32)
            )
        return result

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        outcome_flag: Optional[torch.Tensor] = None,
        steer_embedding: Optional[torch.Tensor] = None,
        steer_inference_mode: Optional[str] = None,
        return_steer_embedding: bool = False,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        resolved_steer_mode = self._resolve_steer_inference_mode(
            steer_inference_mode,
            steer_embedding=steer_embedding,
        )
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        state_cond = self._prepare_state_condition(proprio)
        proprio_context = None
        if proprio is not None and self.proprio_encoder is not None:
            if proprio.ndim == 1:
                proprio_context = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                proprio_context = proprio
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D] for proprio context, got shape {tuple(proprio.shape)}")
            if proprio_context.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio_context.shape[1]}")
            proprio_context = proprio_context.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        steer_context = context
        steer_context_mask = context_mask
        context, context_mask = self._append_outcome_to_context(
            context=context,
            context_mask=context_mask,
            outcome_flag=outcome_flag,
        )
        if proprio_context is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio_context,
            )
            steer_context, steer_context_mask = self._append_proprio_to_context(
                context=steer_context,
                context_mask=steer_context_mask,
                proprio=proprio_context,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        if resolved_steer_mode == "explicit":
            steer_embedding = self._validate_explicit_steer_embedding(
                steer_embedding,
                batch_size=first_frame_latents.shape[0],
            )
        elif resolved_steer_mode == "learned" and self.offline_steer_enabled:
            steer_embedding = self._compute_offline_steer_embedding(
                video_tokens=video_pre["tokens"],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                context=steer_context,
                context_mask=steer_context_mask,
            )
        else:
            steer_embedding = None
        video_seq_len = int(video_pre["tokens"].shape[1])
        if state_cond is None:
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=latents_action.shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            )
            video_kv_cache = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            )
        else:
            attention_mask = None
            video_kv_cache = None

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            if state_cond is None:
                pred_action_posi = self._predict_action_noise_with_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                    steer_embedding=steer_embedding,
                    steer_inference_mode=resolved_steer_mode,
                )
            else:
                pred_action_posi = self._predict_action_noise(
                    first_frame_latents=first_frame_latents,
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    state=state_cond,
                    steer_embedding=steer_embedding,
                    steer_inference_mode=resolved_steer_mode,
                )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        result = {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
        if return_steer_embedding:
            result["steer_embedding"] = (
                None
                if steer_embedding is None
                else steer_embedding.detach().to(device="cpu", dtype=torch.float32)
            )
        return result

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        outcome_flag: Optional[torch.Tensor] = None,
        steer_embedding: Optional[torch.Tensor] = None,
        steer_inference_mode: Optional[str] = None,
        return_steer_embedding: bool = False,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            outcome_flag=outcome_flag,
            steer_embedding=steer_embedding,
            steer_inference_mode=steer_inference_mode,
            return_steer_embedding=return_steer_embedding,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.outcome_encoder is not None:
            payload["outcome_encoder"] = self.outcome_encoder.state_dict()
        if self.offline_steer_enabled:
            if (
                self.offline_steer_student is None
                or self.offline_steer_residual is None
            ):
                raise RuntimeError("Enabled `offline_steer` modules are missing.")
            payload["offline_steer_student"] = (
                self.offline_steer_student.state_dict()
            )
            payload["offline_steer_residual"] = (
                self.offline_steer_residual.state_dict()
            )
            payload["offline_steer_config"] = dict(self.offline_steer_config)
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    @staticmethod
    def _select_mot_state_dict(
        mot_state_dict: dict[str, torch.Tensor],
        *,
        experts: Optional[Sequence[str]] = None,
    ) -> dict[str, torch.Tensor]:
        if not experts:
            return mot_state_dict
        prefixes = tuple(f"mixtures.{name}." for name in experts)
        return {
            key: value
            for key, value in mot_state_dict.items()
            if key.startswith(prefixes)
        }

    @staticmethod
    def _filter_state_dict_by_shape(
        module: torch.nn.Module,
        state_dict: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
        current = module.state_dict()
        filtered: dict[str, torch.Tensor] = {}
        skipped_shape: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        for key, value in state_dict.items():
            if key not in current:
                continue
            if tuple(value.shape) != tuple(current[key].shape):
                skipped_shape.append((key, tuple(value.shape), tuple(current[key].shape)))
                continue
            filtered[key] = value
        return filtered, skipped_shape

    def load_checkpoint(
        self,
        path,
        optimizer=None,
        experts: Optional[Sequence[str]] = None,
    ):
        payload = torch.load(path, map_location="cpu")
        steer_keys = {
            "offline_steer_student",
            "offline_steer_residual",
        }
        present_steer_keys = steer_keys.intersection(payload)
        if present_steer_keys and not self.offline_steer_enabled:
            raise ValueError(
                "Checkpoint contains `offline_steer` weights but current model has "
                "`offline_steer.enabled=false`. Enable the module before loading "
                "this checkpoint."
            )
        if (
            self.offline_steer_enabled
            and present_steer_keys
            and present_steer_keys != steer_keys
        ):
            raise ValueError(
                "Checkpoint has incomplete `offline_steer` state: "
                f"present={sorted(present_steer_keys)}."
            )

        if "mot" in payload:
            mot_state = self._select_mot_state_dict(payload["mot"], experts=experts)
            filtered, skipped_shape = self._filter_state_dict_by_shape(self.mot, mot_state)
            incompatible = self.mot.load_state_dict(filtered, strict=False)
            expert_desc = ",".join(experts) if experts else "all"
            logger.info(
                "Loaded MoT checkpoint from %s (experts=%s): matched=%d skipped_shape=%d missing_in_ckpt=%d unexpected_in_ckpt=%d",
                path,
                expert_desc,
                len(filtered),
                len(skipped_shape),
                len(incompatible.missing_keys),
                len(incompatible.unexpected_keys),
            )
            for key, src_shape, dst_shape in skipped_shape:
                logger.warning(
                    "Skipped MoT key due to shape mismatch: %s ckpt=%s model=%s",
                    key,
                    src_shape,
                    dst_shape,
                )
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")
        if self.outcome_encoder is not None:
            if "outcome_encoder" in payload:
                self.outcome_encoder.load_state_dict(payload["outcome_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `outcome_encoder` weights; keeping current `outcome_encoder` params.")
        elif "outcome_encoder" in payload:
            logger.warning("Checkpoint contains `outcome_encoder` weights but current model has `outcome_num_classes=0`; ignoring.")
        if self.offline_steer_enabled:
            if present_steer_keys == steer_keys:
                if (
                    self.offline_steer_student is None
                    or self.offline_steer_residual is None
                ):
                    raise RuntimeError("Enabled `offline_steer` modules are missing.")
                self.offline_steer_student.load_state_dict(
                    payload["offline_steer_student"],
                    strict=True,
                )
                self.offline_steer_residual.load_state_dict(
                    payload["offline_steer_residual"],
                    strict=True,
                )
            else:
                logger.warning(
                    "Checkpoint has no `offline_steer` weights; keeping the current "
                    "Student and zero-initialized residual parameters."
                )

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
