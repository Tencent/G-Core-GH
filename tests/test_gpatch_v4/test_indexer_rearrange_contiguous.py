"""Indexer fused 路径里 rearrange → contiguous 的显存行为。

对齐 modeling_deepseek_v4.py::
  rearrange(q, 'b s h d -> s b h d').contiguous()
  rearrange(compressed_kv, 'b t d -> t b d').contiguous()
  rearrange(weights, 'b s h -> s b h').contiguous()
"""

import torch
from einops import rearrange


def _assert_no_alloc(src: torch.Tensor, layout: str) -> None:
    viewed = rearrange(src, layout)
    assert viewed.is_contiguous()
    contig = viewed.contiguous()
    assert contig.data_ptr() == viewed.data_ptr()
    assert contig is viewed


def _assert_allocates(src: torch.Tensor, layout: str) -> None:
    viewed = rearrange(src, layout)
    assert not viewed.is_contiguous()
    contig = viewed.contiguous()
    assert contig.data_ptr() != viewed.data_ptr()


def test_b1_rearrange_already_contiguous_no_alloc():
    # 准备数据：与 indexer fused 路径同 layout
    s, t, h, d = 8, 4, 64, 128
    q = torch.randn(1, s, h, d)
    k = torch.randn(1, t, d)
    w = torch.randn(1, s, h)

    _assert_no_alloc(q, 'b s h d -> s b h d')
    _assert_no_alloc(k, 'b t d -> t b d')
    _assert_no_alloc(w, 'b s h -> s b h')


def test_b_gt1_rearrange_needs_contiguous_alloc():
    s, t, h, d = 8, 4, 64, 128
    q = torch.randn(2, s, h, d)
    k = torch.randn(2, t, d)
    w = torch.randn(2, s, h)

    _assert_allocates(q, 'b s h d -> s b h d')
    _assert_allocates(k, 'b t d -> t b d')
    _assert_allocates(w, 'b s h -> s b h')
