from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core import mpu

from gpatch_v4.utils import get_tensor_on_this_cp_rank


class OnlineMtpSftMixin:
    """Shared helpers for feeding MTP labels into the model forward.

    When ``training.online_mtp_sft`` is enabled the model computes the MTP loss
    internally from ``labels``/``loss_mask`` and then resets ``labels`` to
    ``None`` so the forward still returns logits (the main loss is computed
    outside). Mix this into a ``PrepareDataForward`` subclass to share the
    (otherwise duplicated) MTP label preparation logic across the grpo / opd /
    sft data-prep paths.
    """
    def _maybe_set_mtp_sft_fwd_kwargs(
        self,
        fwd_kwargs: Dict[str, Any],
        labels: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
    ) -> None:
        """Attach already CP-split ``labels``/``loss_mask`` to ``fwd_kwargs``.

        No-op when ``online_mtp_sft`` is disabled, so callers keep passing
        ``labels=None``.

        Parameters
        ----------
        fwd_kwargs : dict
            Forward keyword arguments to be updated in place.
        labels : torch.Tensor or None
            Per-position labels, already shifted and CP-split.
        loss_mask : torch.Tensor or None
            Per-position loss mask, already CP-split.
        """
        if self.config.training.online_mtp_sft:
            fwd_kwargs["labels"] = labels
            fwd_kwargs["loss_mask"] = loss_mask

    def _build_online_mtp_labels(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        do_cp_split: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Build ``(mtp_labels, mtp_loss_mask)`` for online MTP during RL/distill.

        Returns ``(None, None)`` when ``online_mtp_sft`` is disabled. Otherwise
        the MTP labels are the input tokens rolled left by one (the next token
        for each position) and the loss mask is the response ``mask`` padded by
        one. When context parallel is active they are zigzag-split onto the
        current CP rank.

        Parameters
        ----------
        tokens : torch.Tensor
            Full-sequence input tokens of shape ``(batch, seqlen)``.
        mask : torch.Tensor
            Response mask of shape ``(batch, seqlen - 1)``.
        do_cp_split : bool, default True
            Whether to slice the tensors onto the current CP rank.

        Returns
        -------
        tuple of (torch.Tensor or None, torch.Tensor or None)
            The MTP labels and loss mask.
        """
        if not self.config.training.online_mtp_sft:
            return None, None
        mtp_labels = torch.roll(tokens, shifts=-1, dims=1)
        mtp_loss_mask = torch.nn.functional.pad(mask, (0, 1), value=0).to(tokens.device)
        if do_cp_split and dist.get_world_size(mpu.get_context_parallel_group()) > 1:
            mtp_labels = get_tensor_on_this_cp_rank(mtp_labels, 1, key_name="labels")
            mtp_loss_mask = get_tensor_on_this_cp_rank(mtp_loss_mask, 1, key_name="loss_mask")
        return mtp_labels, mtp_loss_mask
