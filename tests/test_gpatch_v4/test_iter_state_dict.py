# coding=utf-8
"""确认 torch.Module 的 register_buffer 会进入 state_dict。"""
import torch
import torch.nn as nn


class _MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 8))
        self.register_buffer("e_score_correction_bias", torch.zeros(4), persistent=True)
        self.register_buffer("inv_freq", torch.zeros(16), persistent=False)


def test_state_dict_includes_persistent_buffer():
    m = _MiniModel()
    sd = m.state_dict()
    # persistent buffer 在 state_dict 里
    assert "e_score_correction_bias" in sd
    # non-persistent buffer 不在 state_dict 里
    assert "inv_freq" not in sd
