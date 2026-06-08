import torch
from apex.normalization import FusedLayerNorm


def test_apex():
    norm = FusedLayerNorm(128).to(dtype=torch.bfloat16, device='cuda')
    x = torch.rand(16, 2, 128, device='cuda', dtype=torch.bfloat16)
    y = norm(x)
