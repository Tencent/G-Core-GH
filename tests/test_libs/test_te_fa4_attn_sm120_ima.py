# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.

"""TE flash-attn THD packed CUDA IMA (FA3 or FA4).

Qwen3-Omni vision DPA: THD packed, head_dim=72, 16 heads, padding mask.
Flash is forced; fused/unfused are off. CUDA_LAUNCH_BLOCKING stays 0.

On RTX PRO 5000, FA4 4.0.0b27 IMA'd after ``seq_length=2944``. Older
images without FA4 still run this path on FA3/FA2.

Run::

    python3 -B tests/test_libs/test_te_fa4_attn_sm120_ima.py
    pytest -v -s tests/test_libs/test_te_fa4_attn_sm120_ima.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ["NVTE_FLASH_ATTN"] = "1"
os.environ["NVTE_FUSED_ATTN"] = "0"
os.environ["NVTE_UNFUSED_ATTN"] = "0"

import pytest
import torch

pytest.importorskip("transformer_engine")
from transformer_engine.pytorch.attention import DotProductAttention

try:
    from transformer_engine.pytorch.attention.dot_product_attention import (
        utils as dpa_utils,
    )
except ImportError:
    dpa_utils = None

HEADS = 16
HEAD_DIM = 72
N_LAYERS = 2
PAD_MAX = 4096
# Training crashed after seq=2944; also replay the fused ragged packed case.
SCHEDULE = [
    [256],
    [1664],
    [2944],
    [1024, 1920],
    [1824, 481, 112, 687, 166, 275, 249, 5],
]


def _cu_seqlens(seqlens, device):
    offsets = [0]
    for seqlen in seqlens:
        offsets.append(offsets[-1] + int(seqlen))
    return torch.tensor(offsets, device=device, dtype=torch.int32)


def _dpa_call(dpa, q, k, v, cu, max_s_pad):
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
    return out


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
        out = torch.utils.checkpoint.checkpoint(
            _dpa_call,
            dpa,
            q,
            k,
            v,
            cu,
            max_s_pad,
            use_reentrant=False,
        )
        step_loss = out.float().pow(2).mean()
        loss = step_loss if loss is None else loss + step_loss
    loss.backward()
    return total, max_s, max_s_pad


def _fa4_available():
    """True when TE reports FlashAttention 4 is installed.

    Do not import ``flash_attn.cute``: FA2 2.8.3 also ships that package,
    and importing it can raise ``AttributeError`` (cutlass ``ThrMma``)
    instead of ``ImportError``.
    """
    if dpa_utils is None or not hasattr(dpa_utils, "FlashAttentionUtils"):
        return False
    fa = dpa_utils.FlashAttentionUtils
    if not hasattr(fa, "v4_is_installed"):
        return False
    return bool(fa.v4_is_installed)


def _backend_name():
    if dpa_utils is None or not hasattr(dpa_utils, "FlashAttentionUtils"):
        return "flash=unknown"
    fa = dpa_utils.FlashAttentionUtils
    fa3_installed = fa.v3_is_installed if hasattr(fa, "v3_is_installed") else False
    fa3_ver = fa.fa3_version if hasattr(fa, "fa3_version") else None
    fa4_installed = _fa4_available()
    fa4_ver = fa.fa4_version if hasattr(fa, "fa4_version") else None
    use_v3 = fa.use_v3 if hasattr(fa, "use_v3") else None
    use_v4 = fa.use_v4 if hasattr(fa, "use_v4") else None
    fa2_ver = fa.version if hasattr(fa, "version") else None
    return (
        f"fa2={fa2_ver} fa3_installed={fa3_installed} fa3={fa3_ver} "
        f"use_v3={use_v3} fa4_installed={fa4_installed} fa4={fa4_ver} "
        f"use_v4={use_v4}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_te_fa4_attn_sm120_ragged_thd():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cap = torch.cuda.get_device_capability()
    print(
        f"gpu={torch.cuda.get_device_name(0)} cap={cap} "
        f"torch={torch.__version__} cudnn={torch.backends.cudnn.version()} "
        f"{_backend_name()}",
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
        f"NVTE_FLASH_ATTN={os.environ['NVTE_FLASH_ATTN']} "
        f"NVTE_FUSED_ATTN={os.environ['NVTE_FUSED_ATTN']} "
        f"CUDA_LAUNCH_BLOCKING={os.environ['CUDA_LAUNCH_BLOCKING']}",
        flush=True,
    )
    for step, seqlens in enumerate(SCHEDULE):
        try:
            total, max_s, max_s_pad = _one_step(layers, seqlens, device, dtype)
        except Exception:
            print(f"FAIL step={step} seqlens={seqlens}", flush=True)
            raise
        print(
            f"ok step={step} seqlens={seqlens} "
            f"total={total} max_s={max_s} max_s_pad={max_s_pad} "
            f"{_backend_name()}",
            flush=True,
        )
    torch.cuda.synchronize()


if __name__ == "__main__":
    test_te_fa4_attn_sm120_ragged_thd()
