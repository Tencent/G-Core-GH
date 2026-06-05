"""
Shared forward utilities for WGOv3 and Bagel top-level models.

Extracts duplicated logic from the training forward pass:
  - VAE latent preparation (patchify, flow matching noise, embedding)
  - Loss computation (MSE + CE with dummy loss injection for FSDP sync)
  - Attention mask construction (flash mask index or flex_attention block mask)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    from gpatch_v4.models.bagel.data.data_utils import build_flash_mask_index, create_sparse_mask
except ImportError:
    build_flash_mask_index = None
    create_sparse_mask = None


@dataclass
class VaeLatentResult:
    """Intermediate results from VAE latent preparation, consumed by loss computation."""

    packed_sequence: torch.Tensor
    packed_latent_clean: Optional[torch.Tensor] = None
    noise: Optional[torch.Tensor] = None
    packed_timesteps: Optional[torch.Tensor] = None


def prepare_vae_latent(
    packed_sequence: torch.Tensor,
    latent_patch_size: int,
    latent_channel: int,
    patch_latent_dim: int,
    timestep_shift: float,
    vae2llm: nn.Module,
    time_embedder: nn.Module,
    latent_pos_embed: nn.Module,
    packed_latent: Optional[torch.Tensor],
    padded_latent: Optional[torch.Tensor],
    patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]],
    packed_latent_position_ids: Optional[torch.Tensor],
    packed_vae_token_indexes: Optional[torch.Tensor],
    packed_timesteps: Optional[torch.Tensor],
) -> VaeLatentResult:
    """Process VAE latent data with flow matching noise schedule.

    Handles three mutually exclusive cases:
      1. packed_latent provided (RL inference): reshape and embed directly.
      2. padded_latent provided (training): patchify, add flow-matching noise, embed.
      3. Neither provided: dummy forward to keep VAE params in FSDP computation graph.

    Modifies packed_sequence in-place (writes to packed_vae_token_indexes positions).
    Returns VaeLatentResult with intermediate tensors needed for loss computation.
    """
    p = latent_patch_size
    packed_latent_clean = None
    noise = None

    if packed_latent is not None:
        packed_latent_tmp = packed_latent.reshape(-1, p * p * latent_channel)

    elif padded_latent is not None:
        packed_latent_tmp = []
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * latent_channel)
            packed_latent_tmp.append(latent)
        packed_latent_clean = torch.cat(packed_latent_tmp, dim=0)

        noise = torch.randn_like(packed_latent_clean)
        packed_timesteps = torch.sigmoid(packed_timesteps)
        packed_timesteps = timestep_shift * packed_timesteps / (
            1 + (timestep_shift - 1) * packed_timesteps
        )
        packed_latent_tmp = (
            (1 - packed_timesteps[:, None]) * packed_latent_clean +
            packed_timesteps[:, None] * noise
        )

    else:
        device = packed_sequence.device
        dtype = packed_sequence.dtype
        dummy_latent = torch.zeros(1, patch_latent_dim, device=device, dtype=dtype)
        dummy_t = torch.zeros(1, device=device, dtype=torch.float)
        dummy_pos = torch.zeros(1, device=device, dtype=torch.long)
        d1 = vae2llm(dummy_latent)
        d2 = time_embedder(dummy_t)
        d3 = latent_pos_embed(dummy_pos)
        packed_sequence = packed_sequence + (d1.sum() + d2.sum() + d3.sum()) * 0.0
        packed_latent_tmp = None

    if packed_latent_tmp is not None:
        # VAE encode can yield fp32 latents even when the bridge weights are bf16,
        # so align inputs to the bridge layer before the linear op.
        vae2llm_dtype = vae2llm.weight.dtype
        packed_latent_tmp = packed_latent_tmp.to(dtype=vae2llm_dtype)
        packed_timestep_embeds = time_embedder(packed_timesteps).to(dtype=vae2llm_dtype)
        latent_pos_emb = latent_pos_embed(packed_latent_position_ids).to(dtype=vae2llm_dtype)
        packed_latent_tmp = (vae2llm(packed_latent_tmp) + packed_timestep_embeds + latent_pos_emb)
        if packed_latent_tmp.dtype != packed_sequence.dtype or packed_latent_tmp.device != packed_sequence.device:
            packed_latent_tmp = packed_latent_tmp.to(
                device=packed_sequence.device,
                dtype=packed_sequence.dtype,
            )
        packed_sequence[packed_vae_token_indexes] = packed_latent_tmp

    return VaeLatentResult(
        packed_sequence=packed_sequence,
        packed_latent_clean=packed_latent_clean,
        noise=noise,
        packed_timesteps=packed_timesteps,
    )


def compute_losses(
    last_hidden_state: torch.Tensor,
    packed_sequence: torch.Tensor,
    hidden_size: int,
    llm2vae: nn.Module,
    lm_head: nn.Module,
    packed_latent: Optional[torch.Tensor],
    padded_latent: Optional[torch.Tensor],
    mse_loss_indexes: Optional[torch.BoolTensor],
    ce_loss_indexes: Optional[torch.BoolTensor],
    packed_label_ids: Optional[torch.LongTensor],
    packed_latent_clean: Optional[torch.Tensor],
    noise: Optional[torch.Tensor],
    packed_timesteps: Optional[torch.Tensor],
) -> Dict[str, Any]:
    """Compute MSE and CE losses with dummy loss injection for FSDP sync.

    MSE branch (flow matching velocity prediction):
      - RL infer (packed_latent): just produce predictions, no loss.
      - Training (padded_latent + mse_loss_indexes): v_t prediction MSE.
      - Otherwise: dummy forward through llm2vae.

    CE branch (next-token prediction):
      - ce_loss_indexes has True entries: cross-entropy loss.
      - Otherwise: dummy forward through lm_head.

    Dummy outputs are summed * 0.0 and injected into real losses to ensure all
    parameters participate in the FSDP backward pass.

    Returns:
        dict with keys: mse, ce, model_preds (packed_mse_preds).
    """
    mse = None
    dummy_llm2vae_output = None
    packed_mse_preds = None

    if packed_latent is not None:
        packed_mse_preds = llm2vae(last_hidden_state[mse_loss_indexes])
    elif (
        packed_latent is None and mse_loss_indexes is not None and padded_latent is not None and
        torch.any(mse_loss_indexes)
    ):
        packed_mse_preds = llm2vae(last_hidden_state[mse_loss_indexes])
        target = noise - packed_latent_clean
        has_mse = packed_timesteps > 0
        mse = (packed_mse_preds - target[has_mse])**2
    else:
        if llm2vae is not None:
            dummy_h = torch.zeros(
                1,
                hidden_size,
                device=packed_sequence.device,
                dtype=packed_sequence.dtype,
            )
            dummy_llm2vae_output = llm2vae(dummy_h)

    ce = None
    dummy_lm_head_output = None

    if ce_loss_indexes is not None and torch.any(ce_loss_indexes):
        packed_ce_preds = lm_head(last_hidden_state[ce_loss_indexes])
        ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")
    else:
        dummy_h = torch.zeros(
            1,
            hidden_size,
            device=packed_sequence.device,
            dtype=packed_sequence.dtype,
        )
        dummy_lm_head_output = lm_head(dummy_h)

    dummy_loss = torch.tensor(0.0, device=packed_sequence.device, dtype=packed_sequence.dtype)
    if dummy_llm2vae_output is not None:
        dummy_loss = dummy_loss + dummy_llm2vae_output.sum() * 0.0
    if dummy_lm_head_output is not None:
        dummy_loss = dummy_loss + dummy_lm_head_output.sum() * 0.0

    if mse is not None:
        mse = mse + dummy_loss
    if ce is not None:
        ce = ce + dummy_loss

    if mse is None and ce is None:
        ce = dummy_loss.unsqueeze(0)

    return dict(mse=mse, ce=ce, model_preds=packed_mse_preds)


def build_attention_mask(
    nested_attention_masks: Optional[List[torch.Tensor]],
    use_flash_mask: bool,
    sample_lens: List[int],
    split_lens: List[int],
    attn_modes: List[str],
    num_key_value_heads: int,
    num_heads: int,
    device: torch.device,
    packed_vae_token_indexes: Optional[torch.Tensor] = None,
    vae_mask: bool = False,
):
    """Build attention mask for the LLM forward pass.

    Args:
        nested_attention_masks: Pre-built masks; returned as-is if not None.
        use_flash_mask: Whether to use flash mask index (caller resolves model-specific
            conditions, e.g. Bagel checks ``use_flash_mask and not vae_mask``).
        vae_mask: Passed through to create_sparse_mask as enable_vae_mask.
    """
    if nested_attention_masks is not None:
        return nested_attention_masks

    if use_flash_mask:
        return build_flash_mask_index(sample_lens, split_lens, attn_modes, num_key_value_heads)

    from torch.nn.attention.flex_attention import create_block_mask

    seqlen = sum(sample_lens)
    sparse_mask = create_sparse_mask(
        sample_lens,
        split_lens,
        attn_modes,
        device,
        vae_token_indexes=packed_vae_token_indexes,
        total_length=seqlen,
        enable_vae_mask=vae_mask,
    )
    block_mask = create_block_mask(
        sparse_mask,
        B=1,
        H=num_heads,
        Q_LEN=seqlen,
        KV_LEN=seqlen,
        device=device,
        BLOCK_SIZE=128,
        _compile=True,
    )
    return block_mask
