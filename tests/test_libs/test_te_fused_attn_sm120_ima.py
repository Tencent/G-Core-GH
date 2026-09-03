# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.

"""TE fused-attn CUDA IMA on RTX PRO 5000 (sm_120).

Qwen3-Omni vision DPA: THD packed, head_dim=72, 16 heads, padding mask.
Fused is forced; flash/unfused are off. CUDA_LAUNCH_BLOCKING stays 0.

Isolated uniform THD and a single ragged packed call can PASS. The IMA
showed up on the same 2 DPA modules after changing packed seqlens.
This is round=0 of seed 20260901; fused_attn_bwd IMA'd at step 4::

    [1824, 481, 112, 687, 166, 275, 249, 5]

Run::

    python3 -B tests/test_libs/test_te_fused_attn_sm120_ima.py
    pytest -v -s tests/test_libs/test_te_fused_attn_sm120_ima.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["NVTE_FLASH_ATTN"] = "0"
os.environ["NVTE_FUSED_ATTN"] = "1"
os.environ["NVTE_UNFUSED_ATTN"] = "0"

import pytest
import torch

pytest.importorskip("transformer_engine")
from transformer_engine.pytorch.attention import DotProductAttention

HEADS = 16
HEAD_DIM = 72
N_LAYERS = 2
PAD_MAX = 4096
SCHEDULE = [
    [2153, 259, 146, 15, 25, 1, 2],
    [582],
    [29, 5, 20],
    [1000, 39, 511, 191, 6, 2, 1],
    [1824, 481, 112, 687, 166, 275, 249, 5],
]


def _cu_seqlens(seqlens, device):
    offsets = [0]
    for seqlen in seqlens:
        offsets.append(offsets[-1] + int(seqlen))
    return torch.tensor(offsets, device=device, dtype=torch.int32)


def _one_step(layers, seqlens, device, dtype):
    total = int(sum(seqlens))
    max_s = int(max(seqlens))
    max_s_pad = max(max_s, PAD_MAX)
    cu = _cu_seqlens(seqlens, device)
    q = torch.randn(
        total, HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        total, HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        total, HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True
    )
    loss = None
    for dpa in layers:
        out = dpa(
            q,
            k,
            v,
            attention_mask=None,
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=max_s_pad,
            max_seqlen_kv=max_s_pad,
            attn_mask_type="padding",
            core_attention_bias_type="no_bias",
            pad_between_seqs=False,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        step_loss = out.float().pow(2).mean()
        loss = step_loss if loss is None else loss + step_loss
    loss.backward()
    return total, max_s, max_s_pad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_te_fused_attn_sm120_ragged_thd():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cap = torch.cuda.get_device_capability()
    print(
        f"gpu={torch.cuda.get_device_name(0)} cap={cap} "
        f"torch={torch.__version__} cudnn={torch.backends.cudnn.version()}",
        flush=True,
    )

    torch.manual_seed(1000)
    torch.cuda.manual_seed_all(1000)
    layers = []
    for _ in range(N_LAYERS):
        dpa = DotProductAttention(
            num_attention_heads=HEADS,
            kv_channels=HEAD_DIM,
            num_gqa_groups=HEADS,
            attention_dropout=0.0,
            qkv_format="thd",
            attn_mask_type="padding",
            window_size=(-1, -1),
            softmax_scale=HEAD_DIM ** -0.5,
        ).to(device=device, dtype=dtype)
        dpa.train()
        layers.append(dpa)

    print(
        f"heads={HEADS} head_dim={HEAD_DIM} layers={N_LAYERS} pad_max={PAD_MAX} "
        f"NVTE_FUSED_ATTN={os.environ['NVTE_FUSED_ATTN']} "
        f"CUDA_LAUNCH_BLOCKING={os.environ['CUDA_LAUNCH_BLOCKING']}",
        flush=True,
    )
    for step, seqlens in enumerate(SCHEDULE):
        total, max_s, max_s_pad = _one_step(layers, seqlens, device, dtype)
        print(
            f"ok step={step} seqlens={seqlens} "
            f"total={total} max_s={max_s} max_s_pad={max_s_pad}",
            flush=True,
        )
    torch.cuda.synchronize()


if __name__ == "__main__":
    test_te_fused_attn_sm120_ragged_thd()
