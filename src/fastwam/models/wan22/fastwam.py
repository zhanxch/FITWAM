import math
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .state_dit import StateDiT

logger = get_logger(__name__)


def action_cfg_residual_energy(delta: torch.Tensor) -> torch.Tensor:
    """Per-token RMS of a CFG residual ``ε_posi − ε_base``.

    ``delta`` is ``[B, H, D]``; returns ``[B, H]``.
    """
    if delta.ndim != 3:
        raise ValueError(f"`delta` must be [B, H, D], got {tuple(delta.shape)}")
    return delta.float().square().mean(dim=-1).sqrt()


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
        # Optional train-time VAE cache fill (miss → encode → atomic save).
        self.fill_vae_latent_cache = False
        self.vae_latent_cache_dir = None

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
        load_vae: bool = True,
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
            load_vae=load_vae,
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
        if self.vae is not None:
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
        if self.vae is None:
            raise ValueError(
                "VAE is not loaded (`load_vae=False`). Provide precomputed `sample['input_latents']` "
                "or set `model.load_vae=true`."
            )
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @staticmethod
    def _is_placeholder_video(video: torch.Tensor) -> bool:
        # Dataset drops real pixels on cache hit to a 16x16 stub.
        return int(video.shape[-1]) < 64 or int(video.shape[-2]) < 64

    def _normalize_precomputed_latent_list(self, precomputed_latents, *, batch_size: int | None):
        if precomputed_latents is None:
            return None
        if torch.is_tensor(precomputed_latents):
            if precomputed_latents.ndim == 4:
                items = [precomputed_latents]
            elif precomputed_latents.ndim == 5:
                items = [precomputed_latents[i] for i in range(int(precomputed_latents.shape[0]))]
            else:
                raise ValueError(
                    f"`sample['input_latents']` must be [B,C,T,H,W] or [C,T,H,W], "
                    f"got {tuple(precomputed_latents.shape)}"
                )
        elif isinstance(precomputed_latents, (list, tuple)):
            items = list(precomputed_latents)
        else:
            raise TypeError(
                f"`sample['input_latents']` must be a tensor or list, got {type(precomputed_latents)}"
            )
        if batch_size is not None and len(items) != int(batch_size):
            raise ValueError(
                f"`input_latents` batch mismatch: got {len(items)} entries, expected {batch_size}"
            )
        for i, z in enumerate(items):
            if z is None:
                continue
            if not torch.is_tensor(z):
                raise TypeError(f"`input_latents[{i}]` must be a tensor, got {type(z)}")
            if z.ndim != 4:
                raise ValueError(f"`input_latents[{i}]` must be [C,T,H,W], got {tuple(z.shape)}")
        return items

    def _maybe_save_filled_vae_latent(
        self,
        *,
        sample,
        batch_index: int,
        latents_chw: torch.Tensor,
        video_chw: torch.Tensor | None,
    ) -> None:
        if not bool(getattr(self, "fill_vae_latent_cache", False)):
            return
        cache_dir = getattr(self, "vae_latent_cache_dir", None)
        if cache_dir is None:
            return
        sample_ids = sample.get("eve_sample_id", None)
        window_starts = sample.get("eve_window_start", None)
        if sample_ids is None or window_starts is None:
            return
        if isinstance(sample_ids, str):
            sample_id = sample_ids
        else:
            sample_id = str(sample_ids[batch_index])
        if torch.is_tensor(window_starts):
            window_start = int(window_starts[batch_index].item())
        elif isinstance(window_starts, (list, tuple)):
            window_start = int(window_starts[batch_index])
        else:
            window_start = int(window_starts)
        from fastwam.datasets.vae_latent_cache import save_vae_latent_cache

        save_vae_latent_cache(
            cache_dir,
            sample_id=sample_id,
            window_start=window_start,
            latents=latents_chw,
            video_shape=None if video_chw is None else list(video_chw.shape),
        )

    def _resolve_training_input_latents(self, sample, tiled: bool = False):
        """Resolve [B,C,T,H,W] latents from cache hits and/or online VAE encode."""
        video = sample.get("video", None)
        batch_size_hint = None
        if torch.is_tensor(video) and video.ndim == 5:
            batch_size_hint = int(video.shape[0])
        elif isinstance(video, (list, tuple)):
            batch_size_hint = len(video)
        elif torch.is_tensor(sample.get("action", None)) and sample["action"].ndim == 3:
            batch_size_hint = int(sample["action"].shape[0])

        latent_list = self._normalize_precomputed_latent_list(
            sample.get("input_latents", None),
            batch_size=batch_size_hint,
        )

        if latent_list is not None and all(z is not None for z in latent_list):
            input_latents = torch.stack(
                [
                    z.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                    for z in latent_list
                ],
                dim=0,
            )
            latent_t = int(input_latents.shape[2])
            num_frames = 1 + (latent_t - 1) * 4
            return input_latents, int(input_latents.shape[0]), num_frames

        if video is None:
            raise ValueError(
                "`sample['video']` is required when cached `input_latents` are incomplete."
            )

        video_list: list[torch.Tensor] | None = None
        if isinstance(video, (list, tuple)):
            video_list = list(video)
            batch_size = len(video_list)
            if batch_size < 1:
                raise ValueError("`sample['video']` list is empty.")
            for i, video_i in enumerate(video_list):
                if not torch.is_tensor(video_i) or video_i.ndim != 4 or int(video_i.shape[0]) != 3:
                    raise ValueError(
                        f"`sample['video'][{i}]` must be [3,T,H,W], got "
                        f"{None if video_i is None else tuple(getattr(video_i, 'shape', ()))}"
                    )
            num_frames = int(video_list[0].shape[1])
            height = int(video_list[0].shape[-2])
            width = int(video_list[0].shape[-1])
        elif torch.is_tensor(video) and video.ndim == 5:
            if video.shape[1] != 3:
                raise ValueError(
                    f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}"
                )
            batch_size, _, num_frames, height, width = video.shape
        else:
            raise ValueError(
                f"`sample['video']` must be 5D [B,3,T,H,W] or list of [3,T,H,W], got "
                f"{None if video is None else type(video)}"
            )

        if latent_list is None:
            latent_list = [None] * batch_size
        if len(latent_list) != batch_size:
            raise ValueError(
                f"`input_latents` length {len(latent_list)} != video batch {batch_size}"
            )

        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        def _video_at(index: int) -> torch.Tensor:
            if video_list is not None:
                return video_list[index]
            return video[index]

        if all(z is None for z in latent_list):
            if video_list is not None:
                # Heterogeneous list should not appear for all-miss (all full-res).
                stacked = torch.stack(video_list, dim=0)
            else:
                stacked = video
            if self._is_placeholder_video(stacked[0]):
                raise ValueError(
                    "Cannot encode placeholder video; missing VAE latent cache and no real pixels."
                )
            if int(stacked.shape[-2]) % 16 != 0 or int(stacked.shape[-1]) % 16 != 0:
                raise ValueError(
                    f"Video spatial dims must be multiples of 16, got "
                    f"H={int(stacked.shape[-2])}, W={int(stacked.shape[-1])}"
                )
            if self.vae is None:
                raise ValueError(
                    "VAE is not loaded and `sample['input_latents']` is missing. "
                    "Precompute VAE latents or set `model.load_vae=true`."
                )
            input_video = stacked.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            input_latents = self._encode_video_latents(input_video, tiled=tiled)
            if isinstance(input_latents, list):
                input_latents = torch.stack(input_latents, dim=0)
            for i in range(batch_size):
                self._maybe_save_filled_vae_latent(
                    sample=sample,
                    batch_index=i,
                    latents_chw=input_latents[i],
                    video_chw=stacked[i],
                )
            return input_latents, batch_size, num_frames

        if self.vae is None:
            raise ValueError(
                "VAE is not loaded but this batch has VAE latent cache misses. "
                "Set `model.load_vae=true` (fill mode) or finish precompute."
            )
        resolved: list[torch.Tensor] = []
        for i, z in enumerate(latent_list):
            if z is not None:
                resolved.append(z.to(device=self.device, dtype=self.torch_dtype, non_blocking=True))
                continue
            video_i = _video_at(i)
            if self._is_placeholder_video(video_i):
                raise ValueError(
                    f"Batch index {i} is a VAE cache miss but video is a placeholder stub; "
                    "dataset must decode real pixels on miss."
                )
            h_i, w_i = int(video_i.shape[-2]), int(video_i.shape[-1])
            if h_i % 16 != 0 or w_i % 16 != 0:
                raise ValueError(
                    f"Video spatial dims must be multiples of 16, got H={h_i}, W={w_i}"
                )
            enc = self._encode_video_latents(
                video_i.unsqueeze(0).to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
                tiled=tiled,
            )
            if isinstance(enc, list):
                enc = enc[0]
            if enc.ndim == 5 and enc.shape[0] == 1:
                enc = enc[0]
            if enc.ndim != 4:
                raise ValueError(f"VAE encode expected [C,T,H,W], got {tuple(enc.shape)}")
            resolved.append(enc)
            self._maybe_save_filled_vae_latent(
                sample=sample,
                batch_index=i,
                latents_chw=enc,
                video_chw=video_i,
            )
        input_latents = torch.stack(resolved, dim=0)
        latent_t = int(input_latents.shape[2])
        num_frames = 1 + (latent_t - 1) * 4
        return input_latents, batch_size, num_frames

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
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        action_context = sample.get("action_context", context)
        action_context_mask = sample.get("action_context_mask", context_mask)
        if bool(getattr(self, "pin_video_context_to_base", False)):
            if "base_context" not in sample or "base_context_mask" not in sample:
                raise ValueError(
                    "`pin_video_context_to_base` requires `sample['base_context']` "
                    "and `sample['base_context_mask']`."
                )
            action_loss_weight = sample.get("action_loss_weight", None)
            if action_loss_weight is None:
                context = sample["base_context"]
                context_mask = sample["base_context_mask"]
            else:
                from fastwam.models.wan22.uncond_adapter import pin_video_context_per_sample

                context, context_mask = pin_video_context_per_sample(
                    context,
                    context_mask,
                    sample["base_context"],
                    sample["base_context_mask"],
                    action_loss_weight,
                )
        proprio = sample.get("proprio", None)
        state_is_pad = sample.get("proprio_is_pad", None)

        input_latents, batch_size, num_frames = self._resolve_training_input_latents(
            sample, tiled=tiled
        )

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

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        if action_context.ndim != 3 or action_context_mask.ndim != 2:
            raise ValueError(
                "`action_context/action_context_mask` must be [B,L,D]/[B,L], "
                f"got {tuple(action_context.shape)} and {tuple(action_context_mask.shape)}"
            )
        if action_context.shape[0] != batch_size or action_context_mask.shape[0] != batch_size:
            raise ValueError(
                "Action text context batch size must match the training batch, "
                f"got {action_context.shape[0]}/{action_context_mask.shape[0]} vs {batch_size}."
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        action_context = action_context.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        action_context_mask = action_context_mask.to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )
        context, context_mask = self._append_outcome_to_context(
            context=context,
            context_mask=context_mask,
            outcome_flag=sample.get("outcome_flag", None),
        )
        action_context, action_context_mask = self._append_outcome_to_context(
            context=action_context,
            context_mask=action_context_mask,
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
            action_context, action_context_mask = self._append_proprio_to_context(
                context=action_context,
                context_mask=action_context_mask,
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
            "action_context": action_context,
            "action_context_mask": action_context_mask,
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

        temporal_factor = int(
            getattr(self.vae, "temporal_downsample_factor", None)
            or getattr(self, "vae_temporal_downsample_factor", 4)
        )
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

    @staticmethod
    def _cfg_channels_from_sample(sample, batch_size: int) -> Optional[list[str]]:
        raw = sample.get("cfg_channel") if isinstance(sample, dict) else None
        if raw is None:
            return None
        if isinstance(raw, str):
            channels = [raw]
        else:
            channels = [str(item) for item in raw]
        if len(channels) != int(batch_size):
            raise ValueError(
                "`sample['cfg_channel']` must have one value per batch element, "
                f"got {len(channels)} vs batch={batch_size}."
            )
        return channels

    def _denoise_joint(
        self,
        *,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor],
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_expert = self.video_expert
        action_expert = self.action_expert
        mot = self.mot
        device = next(video_expert.parameters()).device

        def _to(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            return tensor.to(device=device, non_blocking=True)

        latents_video = _to(latents_video)
        latents_action = _to(latents_action)
        timestep_video = _to(timestep_video)
        timestep_action = _to(timestep_action)
        video_context = _to(video_context)
        video_context_mask = _to(video_context_mask)
        action_context = _to(action_context)
        action_context_mask = _to(action_context_mask)
        gt_action = _to(gt_action)
        state = _to(state)

        video_pre = video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=video_context,
            context_mask=video_context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=action_context,
            context_mask=action_context_mask,
        )
        state_pre = None
        if self.state_expert is not None and state is not None:
            state_pre = self._state_condition_pre_dit(
                state=state,
                context=action_context,
                context_mask=action_context_mask,
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
        embeds_all = {"video": video_tokens, "action": action_tokens}
        freqs_all = {"video": video_pre["freqs"], "action": action_pre["freqs"]}
        context_all = {
            "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
            "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
        }
        t_mod_all = {"video": video_pre["t_mod"], "action": action_pre["t_mod"]}
        if state_pre is not None:
            embeds_all["state"] = state_tokens
            freqs_all["state"] = state_pre["freqs"]
            context_all["state"] = {
                "context": state_pre["context"],
                "mask": state_pre["context_mask"],
            }
            t_mod_all["state"] = state_pre["t_mod"]
        tokens_out = mot(
            embeds_all=embeds_all,
            attention_mask=attention_mask,
            freqs_all=freqs_all,
            context_all=context_all,
            t_mod_all=t_mod_all,
        )
        pred_video = video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = action_expert.post_dit(tokens_out["action"], action_pre)
        pred_state = None
        if state_pre is not None:
            pred_state = self.state_expert.post_dit(tokens_out["state"], state_pre)
        if pred_video.device != self.device:
            pred_video = pred_video.to(self.device, non_blocking=True)
        if pred_action.device != self.device:
            pred_action = pred_action.to(self.device, non_blocking=True)
        if pred_state is not None and pred_state.device != self.device:
            pred_state = pred_state.to(self.device, non_blocking=True)
        return pred_video, pred_action, pred_state

    def training_loss(
        self,
        sample,
        tiled: bool = False,
    ):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action_context = inputs["action_context"]
        action_context_mask = inputs["action_context_mask"]
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

        pred_video, pred_action, pred_state = self._denoise_joint(
            latents_video=latents,
            latents_action=noisy_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            video_context=context,
            video_context_mask=context_mask,
            action_context=action_context,
            action_context_mask=action_context_mask,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            gt_action=action,
            state=noisy_state,
        )
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
        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        channels = self._cfg_channels_from_sample(sample, batch_size)

        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        action_loss_sample_weight = sample.get("action_loss_weight", None)
        action_loss_enabled_frac = None
        primary_lock_w = None
        if action_loss_sample_weight is not None:
            action_loss_sample_weight = action_loss_sample_weight.to(
                device=action_loss_per_sample.device, dtype=action_loss_per_sample.dtype, non_blocking=True
            ).view(-1)
            if action_loss_sample_weight.shape[0] != action_loss_per_sample.shape[0]:
                raise ValueError("`sample['action_loss_weight']` must have one value per batch element.")
            primary_lock_w = action_loss_sample_weight.detach()
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

        video_loss_sample_weight = sample.get("video_loss_weight", None)
        video_bc_on_zero_action = bool(getattr(self, "video_bc_on_zero_action", False))
        video_term = self.loss_lambda_video * loss_video
        video_w = None
        if video_loss_sample_weight is not None:
            video_w = video_loss_sample_weight.to(
                device=loss_video_per_sample.device,
                dtype=loss_video_per_sample.dtype,
                non_blocking=True,
            ).view(-1)
            if video_w.shape[0] != loss_video_per_sample.shape[0]:
                raise ValueError(
                    "`sample['video_loss_weight']` must have one value per batch element."
                )
            if float(video_w.sum().item()) > 0:
                loss_video = (
                    loss_video_per_sample * video_weight * video_w
                ).sum() / video_w.sum().clamp(min=1.0)
                video_term = loss_video
            else:
                loss_video = loss_video_per_sample.new_zeros(())
                video_term = loss_video
        elif video_bc_on_zero_action and primary_lock_w is not None:
            fail_w = (1.0 - primary_lock_w.to(dtype=loss_video_per_sample.dtype)).clamp(min=0)
            if float(fail_w.sum().item()) > 0:
                loss_video = (
                    loss_video_per_sample * video_weight * fail_w
                ).sum() / fail_w.sum().clamp(min=1.0)
                video_term = loss_video
            else:
                loss_video = loss_video_per_sample.new_zeros(())
                video_term = loss_video

        loss_total = video_term + self.loss_lambda_action * loss_action
        if loss_state is not None:
            loss_total = loss_total + self.loss_lambda_state * loss_state

        identity_lambda = float(getattr(self, "video_identity_lock_lambda", 0.0) or 0.0)
        identity_mse = None
        if identity_lambda > 0.0 and bool(getattr(self, "uncond_adapter_injected", False)):
            from fastwam.models.wan22.uncond_adapter import uncond_adapter_residual_mse

            identity_mse = uncond_adapter_residual_mse(
                self, expert="video", sample_weight=primary_lock_w
            )
            if identity_mse is not None and bool(
                torch.isfinite(identity_mse.detach()).all().item()
            ):
                loss_total = loss_total + identity_lambda * identity_mse

        action_lock_lambda = float(getattr(self, "action_residual_lock_lambda", 0.0) or 0.0)
        action_lock_mse = None
        action_lock_w = primary_lock_w
        adapter_recipe = str(
            (getattr(self, "uncond_adapter_config", {}) or {}).get("recipe") or "v5"
        )
        if adapter_recipe in {"v8", "v9"}:
            # Idle the residual on D0 only (action BC and video BC both on).
            action_w = sample.get("action_loss_weight", None)
            if action_w is not None and video_w is not None:
                action_lock_w = (
                    action_w.to(device=video_w.device, dtype=video_w.dtype).view(-1)
                    * video_w
                ).clamp(min=0)
        elif video_w is not None and primary_lock_w is not None:
            action_lock_w = (primary_lock_w * (1.0 - video_w.detach())).clamp(min=0)
        outcome_for_lock = sample.get("outcome_flag", None)
        if outcome_for_lock is not None:
            fail_row = (
                outcome_for_lock.to(device=loss_total.device, dtype=torch.float32)
                .view(-1)
                == 1
            ).to(dtype=loss_total.dtype)
            if action_lock_w is None:
                action_lock_w = (1.0 - fail_row).clamp(min=0)
            else:
                action_lock_w = (
                    action_lock_w.to(device=fail_row.device, dtype=fail_row.dtype)
                    * (1.0 - fail_row)
                ).clamp(min=0)
        if action_lock_lambda > 0.0 and bool(getattr(self, "uncond_adapter_injected", False)):
            from fastwam.models.wan22.uncond_adapter import uncond_adapter_residual_mse

            action_lock_mse = uncond_adapter_residual_mse(
                self, expert="action", sample_weight=action_lock_w
            )
            if action_lock_mse is not None and bool(
                torch.isfinite(action_lock_mse.detach()).all().item()
            ):
                loss_total = loss_total + action_lock_lambda * action_lock_mse

        value_loss = None
        cliff_loss = None
        head = getattr(self, "value_head", None)
        value_target = sample.get("value_target", None)
        lambda_value = float(getattr(self, "value_head_lambda", 1.0) or 0.0)
        lambda_cliff = float(getattr(self, "value_cliff_lambda", 0.0) or 0.0)
        if (
            head is not None
            and value_target is not None
            and (lambda_value > 0.0 or lambda_cliff > 0.0)
        ):
            from fastwam.models.wan22.value_head import (
                VALUE_ENCODER_DIT,
                VALUE_LOSS_HUBER,
                recoverability_cliff_loss,
            )

            # Keep the critic off the (noisy) DiT graph and out of bf16
            # autocast. A NaN in this branch previously poisoned ZeRO buckets.
            device_type = input_latents.device.type
            amp_off = (
                torch.autocast(device_type=device_type, enabled=False)
                if device_type in {"cuda", "cpu"}
                else torch.autocast(device_type="cpu", enabled=False)
            )
            encoder = str(
                getattr(self, "value_head_encoder", "vae_latents") or "vae_latents"
            )
            loss_kind = str(getattr(self, "value_head_loss", "bce") or "bce")
            with amp_off:
                if encoder == VALUE_ENCODER_DIT:
                    frame = inputs.get("first_frame_latents")
                    if frame is None:
                        frame = input_latents[:, :, 0:1]
                    with torch.no_grad():
                        tokens = self._encode_value_video_tokens(
                            frame.detach(),
                            context,
                            context_mask,
                            bool(inputs.get("fuse_vae_embedding_in_latents", False)),
                        )
                    value_logits = head.logits(tokens.detach())
                else:
                    value_logits = head.logits(input_latents.detach().clone())
                pred_value = torch.sigmoid(value_logits)
                target_value = value_target.to(
                    device=pred_value.device, dtype=torch.float32, non_blocking=True
                ).view(-1)
                if pred_value.shape[0] != target_value.shape[0]:
                    raise ValueError(
                        "`value_target` batch mismatch: "
                        f"pred={tuple(pred_value.shape)} target={tuple(target_value.shape)}"
                    )
                if loss_kind == VALUE_LOSS_HUBER:
                    value_loss = F.smooth_l1_loss(pred_value, target_value)
                else:
                    value_loss = F.binary_cross_entropy_with_logits(
                        value_logits, target_value
                    )
                value_loss = torch.nan_to_num(
                    value_loss,
                    nan=0.0,
                    posinf=16.0,
                    neginf=16.0,
                )
                # Keep the critic on every rank's graph even if a term is skipped.
                graph_lock = value_logits.sum() * 0.0
            loss_total = loss_total + graph_lock.to(dtype=loss_total.dtype)
            if lambda_value > 0.0:
                loss_total = loss_total + lambda_value * value_loss.to(
                    dtype=loss_total.dtype
                )
            if lambda_cliff > 0.0:
                pair_ids = sample.get("pair_id", None)
                if pair_ids is None:
                    cliff_loss = pred_value.sum() * 0.0
                else:
                    cliff_loss = recoverability_cliff_loss(
                        pred_value.float(),
                        target_value.float(),
                        pair_ids,
                        margin=float(getattr(self, "value_cliff_margin", 0.2) or 0.2),
                    )
                cliff_loss = torch.nan_to_num(
                    cliff_loss, nan=0.0, posinf=1.0, neginf=0.0
                )
                loss_total = loss_total + lambda_cliff * cliff_loss.to(
                    dtype=loss_total.dtype
                )

        loss_dict = {
            "loss_video": float(video_term.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        if action_loss_enabled_frac is not None:
            loss_dict["action_loss_enabled_frac"] = float(action_loss_enabled_frac.item())
        if video_w is not None:
            loss_dict["video_bc_enabled_frac"] = float(video_w.mean().item())
        outcome_flag = sample.get("outcome_flag", None)
        if outcome_flag is not None:
            loss_dict["outcome_failure_frac"] = float(
                outcome_flag.to(device=loss_total.device, dtype=torch.float32, non_blocking=True).view(-1).mean().detach().item()
            )
        if loss_state is not None:
            loss_dict["loss_state"] = self.loss_lambda_state * float(loss_state.detach().item())
        if identity_mse is not None:
            loss_dict["loss_video_identity"] = identity_lambda * float(
                identity_mse.detach().item()
            )
        if action_lock_mse is not None:
            loss_dict["loss_action_residual_lock"] = action_lock_lambda * float(
                action_lock_mse.detach().item()
            )
        if value_loss is not None:
            loss_dict["loss_value"] = float(
                (getattr(self, "value_head_lambda", 1.0) or 0.0) * value_loss.detach().item()
            )
            loss_dict["value_pred_mean"] = float(pred_value.detach().mean().item())
            loss_dict["value_target_mean"] = float(target_value.detach().mean().item())
        if cliff_loss is not None:
            loss_dict["loss_value_cliff"] = float(
                (getattr(self, "value_cliff_lambda", 0.0) or 0.0) * cliff_loss.detach().item()
            )
        if primary_lock_w is not None:
            loss_dict["primary_frac"] = float(primary_lock_w.mean().item())
        loss_dict["finite_video_pred"] = float(torch.isfinite(pred_video).all().item())
        loss_dict["finite_action_pred"] = float(torch.isfinite(pred_action).all().item())
        loss_dict["finite_total"] = float(torch.isfinite(loss_total).all().item())
        if torch.is_tensor(pred_video) and pred_video.numel():
            loss_dict["pred_video_absmax"] = float(
                pred_video.detach().float().abs().max().item()
            )
        if torch.is_tensor(pred_action) and pred_action.numel():
            loss_dict["pred_action_absmax"] = float(
                pred_action.detach().float().abs().max().item()
            )
        if channels is not None:
            loss_dict["cfg_base_frac"] = float(
                sum(1.0 if str(ch) == "base" else 0.0 for ch in channels) / max(len(channels), 1)
            )
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

    def _encode_value_video_tokens(
        self,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_flag: bool,
    ) -> torch.Tensor:
        """Frozen-S0 current-obs VideoDiT tokens for the v9 value head.

        Adapter is forced off. Text is whatever the caller passed (S0 base
        when ``pin_video_context_to_base`` is on). Tokens are detached from
        the action-denoising graph.
        """

        from fastwam.models.wan22.uncond_adapter import uncond_adapter_enabled

        frame = first_frame_latents
        if frame.ndim == 4:
            frame = frame.unsqueeze(2)
        timestep_video = torch.zeros(
            (frame.shape[0],),
            dtype=frame.dtype,
            device=frame.device,
        )
        with uncond_adapter_enabled(self, False):
            video_pre = self.video_expert.pre_dit(
                x=frame,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            video_seq_len = int(video_pre["tokens"].shape[1])
            video_mask = self.video_expert.build_video_to_video_mask(
                video_seq_len=video_seq_len,
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_pre["tokens"].device,
            )
            _kv, tokens = self.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=video_mask,
                return_tokens=True,
            )
        return tokens

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
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
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
        negative_prompt: Optional[str] = None,
        negative_context: Optional[torch.Tensor] = None,
        negative_context_mask: Optional[torch.Tensor] = None,
        failure_prompt: Optional[str] = None,
        failure_context: Optional[torch.Tensor] = None,
        failure_context_mask: Optional[torch.Tensor] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        return_cfg_residual: bool = False,
        cfg_exec_horizon: int = 24,
        adaptive_cfg_tau: Optional[float] = None,
        cfg_epsilon_l: Optional[float] = None,
        cfg_residual_clip_mode: str = "rms",
        cfg_gate_mode: Optional[str] = None,
        cfg_value_prev: Optional[float] = None,
        cfg_gate_fired: bool = False,
        cfg_v_high: Optional[float] = None,
        cfg_drop_delta: Optional[float] = None,
        cfg_replan_index: Optional[int] = None,
        cfg_growth_tau: Optional[float] = None,
        cfg_growth_start_replan: Optional[int] = None,
    ) -> dict[str, Any]:
        self.eval()
        text_cfg_scale = float(text_cfg_scale)
        if not math.isfinite(text_cfg_scale):
            raise ValueError(f"`text_cfg_scale` must be finite, got {text_cfg_scale}.")
        # text_cfg_scale=1 is the 本体 bypass (no mix). Mix w=1 is recipe-dependent:
        # v5/v6 → ε_posi; v7 → ε_base+(ε_posi-ε_fail). Neither is this bypass.
        use_text_cfg = text_cfg_scale != 1.0
        adaptive_tau = None if adaptive_cfg_tau is None else float(adaptive_cfg_tau)
        if adaptive_tau is not None:
            if not math.isfinite(adaptive_tau) or adaptive_tau < 0.0:
                raise ValueError(
                    f"`adaptive_cfg_tau` must be finite and >= 0, got {adaptive_cfg_tau}."
                )
            if not use_text_cfg:
                raise ValueError(
                    "adaptive CFG requires a guided mix (`text_cfg_scale != 1`). "
                    "text_cfg_scale=1 is the 本体 bypass; low-E fallback is mix weight 0."
                )
        epsilon_l = None if cfg_epsilon_l is None else float(cfg_epsilon_l)
        if epsilon_l is not None:
            if not math.isfinite(epsilon_l) or epsilon_l < 0.0:
                raise ValueError(
                    f"`cfg_epsilon_l` must be finite and >= 0, got {cfg_epsilon_l}."
                )
            # Validate the mode once, before model execution.  The helper
            # repeats no global state and is also used directly in tests.
            from fastwam.models.wan22.uncond_adapter import normalize_cfg_residual_clip_mode

            residual_clip_mode = normalize_cfg_residual_clip_mode(cfg_residual_clip_mode)
        else:
            residual_clip_mode = str(cfg_residual_clip_mode)
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
        cfg_value = None
        cfg_gate_g = None
        cfg_value_rel = None
        value_head = getattr(self, "value_head", None)
        value_encoder = str(
            getattr(self, "value_head_encoder", "vae_latents") or "vae_latents"
        )
        if value_head is not None and value_encoder != "video_dit":
            cfg_value = float(value_head(first_frame_latents).reshape(-1)[0].item())
        gate_mode = str(cfg_gate_mode or "").strip().lower()

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        use_negative_context = negative_context is not None or negative_context_mask is not None
        if use_negative_context and negative_prompt is not None:
            raise ValueError(
                "`negative_prompt` and `negative_context/negative_context_mask` are mutually exclusive."
            )
        if use_negative_context and (
            negative_context is None or negative_context_mask is None
        ):
            raise ValueError(
                "`negative_context` and `negative_context_mask` must be both provided together."
            )
        use_failure_context = failure_context is not None or failure_context_mask is not None
        if use_failure_context and failure_prompt is not None:
            raise ValueError(
                "`failure_prompt` and `failure_context/failure_context_mask` are mutually exclusive."
            )
        if use_failure_context and (
            failure_context is None or failure_context_mask is None
        ):
            raise ValueError(
                "`failure_context` and `failure_context_mask` must be both provided together."
            )
        if use_text_cfg:
            if use_prompt and negative_prompt is None:
                raise ValueError(
                    "`text_cfg_scale != 1` with prompt input requires an explicit `negative_prompt`."
                )
            if use_context and not use_negative_context:
                raise ValueError(
                    "`text_cfg_scale != 1` with cached context requires both "
                    "`negative_context` and `negative_context_mask`."
                )

        def prepare_cached_context(
            value: torch.Tensor,
            mask: torch.Tensor,
            *,
            label: str,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if value.ndim == 2:
                value = value.unsqueeze(0)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if value.ndim != 3 or mask.ndim != 2:
                raise ValueError(
                    f"`{label}/{label}_mask` must be [B,L,D]/[B,L], "
                    f"got {tuple(value.shape)} and {tuple(mask.shape)}"
                )
            if value.shape[:2] != mask.shape or value.shape[0] != 1:
                raise ValueError(
                    f"`{label}/{label}_mask` shape mismatch or batch size is not 1: "
                    f"got {tuple(value.shape)} and {tuple(mask.shape)}"
                )
            return (
                value.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
                mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
            )

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
            if negative_prompt:
                negative_context, negative_context_mask = self.encode_prompt(negative_prompt)
            if failure_prompt:
                failure_context, failure_context_mask = self.encode_prompt(failure_prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            context, context_mask = prepare_cached_context(
                context,
                context_mask,
                label="context",
            )
            if use_negative_context:
                negative_context, negative_context_mask = prepare_cached_context(
                    negative_context,
                    negative_context_mask,
                    label="negative_context",
                )
            if use_failure_context:
                failure_context, failure_context_mask = prepare_cached_context(
                    failure_context,
                    failure_context_mask,
                    label="failure_context",
                )
        have_negative = negative_context is not None and negative_context_mask is not None
        have_failure = failure_context is not None and failure_context_mask is not None
        context, context_mask = self._append_outcome_to_context(
            context=context,
            context_mask=context_mask,
            outcome_flag=outcome_flag,
        )
        if have_negative:
            negative_context, negative_context_mask = self._append_outcome_to_context(
                context=negative_context,
                context_mask=negative_context_mask,
                outcome_flag=outcome_flag,
            )
        if have_failure:
            failure_context, failure_context_mask = self._append_outcome_to_context(
                context=failure_context,
                context_mask=failure_context_mask,
                outcome_flag=outcome_flag,
            )
        if proprio_context is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio_context,
            )
            if have_negative:
                negative_context, negative_context_mask = self._append_proprio_to_context(
                    context=negative_context,
                    context_mask=negative_context_mask,
                    proprio=proprio_context,
                )
            if have_failure:
                failure_context, failure_context_mask = self._append_proprio_to_context(
                    context=failure_context,
                    context_mask=failure_context_mask,
                    proprio=proprio_context,
                )

        from fastwam.models.wan22.uncond_adapter import (
            adaptive_cfg_mix_weight,
            bound_cfg_residual,
            cfg_mix_subtract_branch,
            CFG_MIX_SUBTRACT_FAIL,
            mix_guided_action_epsilon,
            uncond_adapter_enabled,
            v5_infer_remap_to_base_context,
            v5_infer_use_adapter,
            v5_infer_video_uses_base_context,
        )

        adapter_injected = bool(getattr(self, "uncond_adapter_injected", False))
        adapter_recipe = str(
            (getattr(self, "uncond_adapter_config", {}) or {}).get("recipe") or "v5"
        )
        mix_subtract = cfg_mix_subtract_branch(adapter_recipe)
        if mix_subtract == CFG_MIX_SUBTRACT_FAIL and use_text_cfg and not have_failure:
            raise ValueError(
                "DEWO v7 CFG mix requires `failure_prompt` or "
                "`failure_context`/`failure_context_mask`. "
                "Do not fall back to subtracting ε_base."
            )
        # S0 本体 = adapter off + base text. CFG posi = adapter on + success.
        # text_cfg_scale=1 skips the mix and remaps onto cfg_base_prompt (本体).
        # v5/v6 mix w=1 would be ε_posi; v7 mix w=1 is ε_base+(ε_posi-ε_fail).
        if v5_infer_remap_to_base_context(
            adapter_injected=adapter_injected,
            use_text_cfg=use_text_cfg,
            has_negative_context=have_negative,
        ):
            context = negative_context
            context_mask = negative_context_mask
        posi_use_adapter = adapter_injected and v5_infer_use_adapter(
            branch="posi",
            use_text_cfg=use_text_cfg,
        )
        base_use_adapter = adapter_injected and v5_infer_use_adapter(
            branch="base",
            use_text_cfg=use_text_cfg,
        )
        pin_video_to_base = v5_infer_video_uses_base_context(
            adapter_injected=adapter_injected,
            pin_video_context_to_base=bool(
                getattr(self, "pin_video_context_to_base", False)
            ),
            has_negative_context=have_negative,
        )
        if (
            adapter_injected
            and bool(getattr(self, "pin_video_context_to_base", False))
            and not have_negative
            and use_text_cfg
        ):
            logger.warning(
                "pin_video_context_to_base is set but cfg_base_prompt / "
                "negative_context is missing; video prefill will follow the "
                "action prompt and may leave the S0 observation encoding."
            )
        video_posi_context, video_posi_mask = context, context_mask
        video_base_context, video_base_mask = negative_context, negative_context_mask
        if pin_video_to_base:
            video_posi_context, video_posi_mask = negative_context, negative_context_mask

        if value_head is not None and value_encoder == "video_dit":
            video_ctx, video_msk = video_posi_context, video_posi_mask
            if video_ctx is None:
                video_ctx, video_msk = context, context_mask
            value_tokens = self._encode_value_video_tokens(
                first_frame_latents,
                video_ctx,
                video_msk,
                fuse_flag,
            )
            cfg_value = float(value_head(value_tokens).reshape(-1)[0].item())

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        with uncond_adapter_enabled(self, posi_use_adapter):
            video_pre = self.video_expert.pre_dit(
                x=first_frame_latents,
                timestep=timestep_video,
                context=video_posi_context,
                context_mask=video_posi_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
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
            negative_video_kv_cache = None
            if use_text_cfg:
                with uncond_adapter_enabled(self, base_use_adapter):
                    negative_video_pre = self.video_expert.pre_dit(
                        x=first_frame_latents,
                        timestep=timestep_video,
                        context=video_base_context,
                        context_mask=video_base_mask,
                        action=None,
                        fuse_vae_embedding_in_latents=fuse_flag,
                    )
                    if int(negative_video_pre["tokens"].shape[1]) != video_seq_len:
                        raise ValueError(
                            "Positive and base CFG branches produced different video sequence lengths."
                        )
                    negative_video_kv_cache = self.mot.prefill_video_cache(
                        video_tokens=negative_video_pre["tokens"],
                        video_freqs=negative_video_pre["freqs"],
                        video_t_mod=negative_video_pre["t_mod"],
                        video_context_payload={
                            "context": negative_video_pre["context"],
                            "mask": negative_video_pre["context_mask"],
                        },
                        video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
                    )
        else:
            attention_mask = None
            video_kv_cache = None
            negative_video_kv_cache = None

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        cfg_token_rms_steps: list[torch.Tensor] = []
        exec_horizon = max(1, min(int(cfg_exec_horizon), int(latents_action.shape[1])))
        frozen_mix_weight: Optional[float] = None
        gate_exec_rms: Optional[float] = None
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            with uncond_adapter_enabled(self, posi_use_adapter):
                if state_cond is None:
                    pred_action_posi = self._predict_action_noise_with_cache(
                        latents_action=latents_action,
                        timestep_action=timestep_action,
                        context=context,
                        context_mask=context_mask,
                        video_kv_cache=video_kv_cache,
                        attention_mask=attention_mask,
                        video_seq_len=video_seq_len,
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
                    )
            if use_text_cfg:
                with uncond_adapter_enabled(self, base_use_adapter):
                    if state_cond is None:
                        pred_action_base = self._predict_action_noise_with_cache(
                            latents_action=latents_action,
                            timestep_action=timestep_action,
                            context=negative_context,
                            context_mask=negative_context_mask,
                            video_kv_cache=negative_video_kv_cache,
                            attention_mask=attention_mask,
                            video_seq_len=video_seq_len,
                        )
                    else:
                        pred_action_base = self._predict_action_noise(
                            first_frame_latents=first_frame_latents,
                            latents_action=latents_action,
                            timestep_action=timestep_action,
                            context=negative_context,
                            context_mask=negative_context_mask,
                            fuse_vae_embedding_in_latents=fuse_flag,
                            state=state_cond,
                        )
                pred_action_fail = None
                if mix_subtract == CFG_MIX_SUBTRACT_FAIL:
                    # Same adapter-on video cache as ε_+; only the action text changes.
                    with uncond_adapter_enabled(self, posi_use_adapter):
                        if state_cond is None:
                            pred_action_fail = self._predict_action_noise_with_cache(
                                latents_action=latents_action,
                                timestep_action=timestep_action,
                                context=failure_context,
                                context_mask=failure_context_mask,
                                video_kv_cache=video_kv_cache,
                                attention_mask=attention_mask,
                                video_seq_len=video_seq_len,
                            )
                        else:
                            pred_action_fail = self._predict_action_noise(
                                first_frame_latents=first_frame_latents,
                                latents_action=latents_action,
                                timestep_action=timestep_action,
                                context=failure_context,
                                context_mask=failure_context_mask,
                                fuse_vae_embedding_in_latents=fuse_flag,
                                state=state_cond,
                            )
                _, raw_delta = mix_guided_action_epsilon(
                    pred_action_base,
                    pred_action_posi,
                    mix_weight=1.0,
                    subtract=mix_subtract,
                    epsilon_fail=pred_action_fail,
                )
                # ``epsilon_l`` is a trust-region bound on the learned CFG
                # correction.  It acts before the scalar guidance weight, so
                # increasing ``text_cfg_scale`` cannot undo the bound.
                delta = bound_cfg_residual(
                    raw_delta,
                    epsilon_l,
                    mode=residual_clip_mode,
                )
                # Origin is always ε_base. v7 mix w=1 is ε_base+(ε_posi-ε_fail),
                # not ε_posi. text_cfg_scale=1 never reaches this branch.
                mix_weight = text_cfg_scale
                if gate_mode == "value":
                    from fastwam.models.wan22.value_head import (
                        DEFAULT_VALUE_DROP_DELTA,
                        drop_edge_gate,
                    )

                    if cfg_value is None:
                        raise ValueError(
                            "`cfg_gate_mode=value` requires a trained value_head."
                        )
                    if frozen_mix_weight is None:
                        resolved_v_high = (
                            cfg_v_high
                            if cfg_v_high is not None
                            else getattr(self, "value_v_high", None)
                        )
                        cfg_gate_g = drop_edge_gate(
                            None if cfg_value_prev is None else float(cfg_value_prev),
                            float(cfg_value),
                            v_high=resolved_v_high,
                            delta=float(
                                getattr(self, "value_drop_delta", DEFAULT_VALUE_DROP_DELTA)
                                if cfg_drop_delta is None
                                else cfg_drop_delta
                            ),
                            fired=bool(cfg_gate_fired),
                        )
                        frozen_mix_weight = float(cfg_gate_g) * float(text_cfg_scale)
                    mix_weight = frozen_mix_weight
                elif gate_mode in {"value_growth", "growth"}:
                    from fastwam.models.wan22.value_head import (
                        DEFAULT_VALUE_GROWTH_START_REPLAN,
                        DEFAULT_VALUE_GROWTH_TAU,
                        relative_growth,
                        relative_growth_gate,
                    )

                    if cfg_value is None:
                        raise ValueError(
                            "`cfg_gate_mode=value_growth` requires a trained value_head."
                        )
                    if frozen_mix_weight is None:
                        v_prev = (
                            None
                            if cfg_value_prev is None
                            else float(cfg_value_prev)
                        )
                        tau = (
                            DEFAULT_VALUE_GROWTH_TAU
                            if cfg_growth_tau is None
                            else float(cfg_growth_tau)
                        )
                        start_replan = (
                            DEFAULT_VALUE_GROWTH_START_REPLAN
                            if cfg_growth_start_replan is None
                            else int(cfg_growth_start_replan)
                        )
                        replan_index = (
                            0 if cfg_replan_index is None else int(cfg_replan_index)
                        )
                        cfg_value_rel = relative_growth(v_prev, float(cfg_value))
                        cfg_gate_g = relative_growth_gate(
                            v_prev,
                            float(cfg_value),
                            tau=tau,
                            replan_index=replan_index,
                            start_replan=start_replan,
                        )
                        frozen_mix_weight = float(cfg_gate_g) * float(text_cfg_scale)
                    mix_weight = frozen_mix_weight
                elif adaptive_tau is not None:
                    token_e = action_cfg_residual_energy(delta)
                    step_exec_rms = float(token_e[0, :exec_horizon].mean().item())
                    if frozen_mix_weight is None:
                        frozen_mix_weight = adaptive_cfg_mix_weight(
                            exec_rms=step_exec_rms,
                            tau=adaptive_tau,
                            guided_scale=text_cfg_scale,
                        )
                        gate_exec_rms = step_exec_rms
                    mix_weight = frozen_mix_weight
                pred_action = pred_action_base + mix_weight * delta
                if return_cfg_residual:
                    cfg_token_rms_steps.append(action_cfg_residual_energy(delta)[0].detach())
            else:
                pred_action = pred_action_posi
                if return_cfg_residual:
                    cfg_token_rms_steps.append(
                        torch.zeros(
                            latents_action.shape[1],
                            device=latents_action.device,
                            dtype=torch.float32,
                        )
                    )

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        result = {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
        if return_cfg_residual:
            if not cfg_token_rms_steps:
                horizon = int(latents_action.shape[1])
                stacked = torch.zeros((1, horizon), dtype=torch.float32)
            else:
                stacked = torch.stack(cfg_token_rms_steps, dim=0).to(
                    device="cpu", dtype=torch.float32
                )
            result["cfg_token_rms_nfe"] = stacked
            result["cfg_chunk_rms_nfe"] = stacked.mean(dim=-1)
            result["cfg_token_rms"] = stacked.mean(dim=0)
            result["cfg_chunk_rms"] = stacked.mean()
            result["cfg_exec_rms"] = stacked[:, :exec_horizon].mean()
        if adaptive_tau is not None:
            result["cfg_mix_weight"] = (
                0.0 if frozen_mix_weight is None else float(frozen_mix_weight)
            )
            if gate_exec_rms is not None:
                result["cfg_gate_exec_rms"] = float(gate_exec_rms)
        if cfg_value is not None:
            result["cfg_value"] = float(cfg_value)
        if cfg_value_rel is not None:
            result["cfg_value_rel"] = float(cfg_value_rel)
        if cfg_gate_g is not None:
            result["cfg_gate_g"] = float(cfg_gate_g)
            result["cfg_mix_weight"] = (
                0.0 if frozen_mix_weight is None else float(frozen_mix_weight)
            )
        if epsilon_l is not None:
            result["cfg_epsilon_l"] = float(epsilon_l)
            result["cfg_residual_clip_mode"] = residual_clip_mode
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
        from fastwam.models.wan22.uncond_adapter import is_uncond_adapter_checkpoint

        if is_uncond_adapter_checkpoint(payload):
            raise ValueError(
                f"{path} is a DEWO v5 uncond-adapter file, not a backbone MoT. "
                "Load the frozen 本体 checkpoint first, then "
                "load_uncond_adapter_state_dict()."
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
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
