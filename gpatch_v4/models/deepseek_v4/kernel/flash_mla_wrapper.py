"""
FlashMLA forward + cuDNN DSA backward for DeepSeek-V4 sparse attention.

Uses ``flash_mla.flash_mla_sparse_fwd`` (from https://github.com/deepseek-ai/FlashMLA)
for the forward pass and ``cudnn.DSA.sparse_attention_backward_wrapper`` for
gradients — the same pairing as Megatron-LM ``SparseAttnFunc``.

FlashMLA expects flat (unbatched) tensors with a multi-head-KV signature
``(total_S, h_kv, D)``; this wrapper adapts the gcore batched layout
``[B, S, H, D] / [B, S_kv, D]`` to that convention, then reshapes grads back.

Usage:
    from .kernel.flash_mla_wrapper import sparse_attn_flash_mla
    output = sparse_attn_flash_mla(q, kv, attn_sink, topk_idxs, sm_scale=...)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from flash_mla import flash_mla_sparse_fwd as _flash_mla_sparse_fwd

_DSA = None


def _ensure_dsa_namespace():
    """Lazily import the cudnn-frontend DSA namespace."""
    global _DSA
    if _DSA is not None:
        return
    try:
        from cudnn import DSA as _ns
    except ImportError as e:
        raise ImportError(
            "cudnn-frontend DSA namespace not available. Install with "
            "`pip install nvidia-cudnn-frontend[cutedsl]`."
        ) from e
    _DSA = _ns


def _get_topk_alignment() -> int:
    """Minimum TopK alignment required by FlashMLA for the current GPU.

    SM90: dual-warpgroup loop steps by 2 blocks → 2 * B_TOPK = 128
    SM100: single-pipeline loop steps by 1 block → B_TOPK = 64
    """
    sm = torch.cuda.get_device_capability()
    if sm[0] >= 10:
        return 64
    return 128


class FlashMLAForwardCudnnDSABackward(torch.autograd.Function):
    """Sparse attention: FlashMLA forward + cuDNN DSA backward.

    Matches Megatron ``SparseAttnFunc`` kernel pairing.  LSE is kept in
    FlashMLA's native convention (natural log, attn_sink NOT folded in) —
    that is what cuDNN DSA backward expects.  Do NOT convert to TileLang
    log2 / sink-included LSE.
    """

    @staticmethod
    def forward(
        ctx,
        q: Tensor,           # [B, S, H, D]  bf16
        kv: Tensor,          # [B, S_kv, D]  bf16
        attn_sink: Tensor,   # [H]  fp32
        topk_idxs: Tensor,   # [B, S, topk]  int32
        sm_scale: Optional[float] = None,
    ) -> Tensor:
        B, S, H, D = q.shape
        _, S_kv, _ = kv.shape
        TopK = topk_idxs.shape[-1]

        if sm_scale is None:
            sm_scale = D ** (-0.5)

        # ---- pad TopK to FlashMLA alignment ----
        topk_align = _get_topk_alignment()
        TopK_padded = (TopK + topk_align - 1) // topk_align * topk_align
        if TopK_padded != TopK:
            topk_idxs_padded = torch.nn.functional.pad(
                topk_idxs, (0, TopK_padded - TopK), value=-1
            )
        else:
            topk_idxs_padded = topk_idxs

        # ---- batched → flat: per-batch local indices → global flat indices ----
        # KV layout is B-major: segment b occupies [b * S_kv, (b+1) * S_kv).
        # Invalid (-1) stays -1.  Consistent with FlashMLA flat indexing.
        if B == 1:
            topk_idxs_global = topk_idxs_padded
        else:
            offsets = (
                torch.arange(B, device=q.device, dtype=torch.int32).view(B, 1, 1) * S_kv
            )
            topk_idxs_global = torch.where(
                topk_idxs_padded >= 0,
                topk_idxs_padded + offsets,
                topk_idxs_padded,
            )

        q_flat = q.reshape(B * S, H, D).contiguous()
        kv_flat = kv.reshape(B * S_kv, D).contiguous()
        # Save the same padded global indices FlashMLA / cuDNN both see.
        indices_flat = topk_idxs_global.reshape(B * S, TopK_padded).contiguous()

        kv_3d = kv_flat.unsqueeze(1)              # (total_S_kv, 1, D)
        indices_3d = indices_flat.unsqueeze(1)    # (total_S_q, 1, TopK_padded)

        with torch.cuda.nvtx.range("flash_mla_sparse_fwd"):
            out_flat, _max_logits, lse_flat = _flash_mla_sparse_fwd(
                q_flat,
                kv_3d,
                indices_3d,
                sm_scale,
                d_v=D,
                attn_sink=attn_sink,
            )

        # Keep FlashMLA native LSE (ln, no sink) for cuDNN DSA bwd.
        ctx.save_for_backward(q_flat, kv_flat, attn_sink, indices_flat, out_flat, lse_flat)
        ctx.sm_scale = sm_scale
        ctx.B = B
        ctx.S = S
        ctx.S_kv = S_kv
        ctx.H = H
        ctx.D = D

        return out_flat.reshape(B, S, H, D).contiguous()

    @staticmethod
    def backward(ctx, do: Tensor) -> Tuple:
        q_flat, kv_flat, attn_sink, indices_flat, out_flat, lse_flat = ctx.saved_tensors
        B, S, S_kv, H, D = ctx.B, ctx.S, ctx.S_kv, ctx.H, ctx.D

        _ensure_dsa_namespace()
        do_flat = do.reshape(B * S, H, D).contiguous()

        with torch.cuda.nvtx.range("cudnn_dsa_sparse_attn_bwd"):
            result = _DSA.sparse_attention_backward_wrapper(
                q_flat,
                kv_flat,
                out_flat,
                do_flat,
                lse_flat,
                attn_sink,
                indices_flat,
                softmax_scale=ctx.sm_scale,
                topk_length=None,
            )

        dq = result["dq"].reshape(B, S, H, D)
        dkv = result["dkv"].reshape(B, S_kv, D)
        d_attn_sink = result["d_sink"]
        return dq, dkv, d_attn_sink, None, None


def sparse_attn_flash_mla(
    q: Tensor,
    kv: Tensor,
    attn_sink: Tensor,
    topk_idxs: Tensor,
    sm_scale: Optional[float] = None,
) -> Tensor:
    """Sparse attention with FlashMLA forward + cuDNN DSA backward.

    Args:
        q:         [B, S, H, D]  bf16
        kv:        [B, S_kv, D]  bf16
        attn_sink: [H]  fp32
        topk_idxs: [B, S, topk]  int32, -1 for invalid positions
        sm_scale:  float or None (defaults to 1/sqrt(D))

    Returns:
        out: [B, S, H, D]  bf16
    """
    return FlashMLAForwardCudnnDSABackward.apply(q, kv, attn_sink, topk_idxs, sm_scale)
