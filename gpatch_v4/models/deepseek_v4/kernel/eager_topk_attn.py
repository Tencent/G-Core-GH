import torch
import torch.nn.functional as F


def eager_topk_attention_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    sinks: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Eager reference for the sparse top-k MQA path.

    Mirrors :func:`sparse_attn_tilelang` semantics exactly (per-query gather
    over ``topk_idxs``, ``-1`` = masked slot, per-head sink added to the
    softmax denominator). The dtype flow intentionally matches HF dense eager:
    q/k scores are computed in the input dtype, then concatenating fp32 sinks
    promotes logits before softmax, and probabilities are cast back to the KV
    dtype before the value matmul.

    Parameters
    ----------
    q : torch.Tensor
        Shape ``[B, H, S, D]``.
    kv : torch.Tensor
        Shape ``[B, 1, S_kv, D]``; read as both key and value.
    sinks : torch.Tensor
        Shape ``[H]``; per-head pre-scaled sink logit (same space as
        ``score * sm_scale``).
    topk_idxs : torch.Tensor
        Shape ``[B, S, topk]`` int; absolute indices into ``kv``'s seq axis.
        ``-1`` marks an inactive slot (contributes nothing).
    sm_scale : float
        Softmax logit scale (``head_dim ** -0.5``).

    Returns
    -------
    torch.Tensor
        Shape ``[B, S, H, D]``, ``q.dtype`` (matches the fused branch layout).
    """
    B, H, S, D = q.shape
    S_kv = kv.shape[2]
    topk = topk_idxs.shape[-1]

    kv_f = kv[:, 0]  # [B, S_kv, D]
    valid = topk_idxs >= 0  # [B, S, topk]
    safe = topk_idxs.clamp(min=0).long()  # [B, S, topk]
    kv_g = torch.gather(
        kv_f.unsqueeze(1).expand(B, S, S_kv, D),
        2,
        safe.unsqueeze(-1).expand(B, S, topk, D),
    )
    scores = torch.einsum("bhsd,bstd->bhst", q, kv_g) * sm_scale  # [B,H,S,topk]
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))

    sink_logits = sinks.reshape(1, H, 1, 1).expand(B, H, S, 1)
    combined_logits = torch.cat([scores, sink_logits], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    attn = probs[..., :-1].to(kv_g.dtype)
    out = torch.einsum("bhst,bstd->bhsd", attn, kv_g)  # [B,H,S,D]
    return out.transpose(1, 2).contiguous().to(q.dtype)  # [B,S,H,D]
