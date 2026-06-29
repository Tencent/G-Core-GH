"""Tests for per-token loss weight support in SFT training."""
import torch
import pytest
from transformers import AutoTokenizer

from gpatch_v4.utils.training_utils import build_token_loss_weights_from_spans, format_token_weight_table

_TOKENIZER = AutoTokenizer.from_pretrained("hf-hub/Qwen/Qwen2.5-Math-1.5B/", use_fast=True)



def test_build_token_loss_weights_prompt_zero():
    """Prompt tokens get weight 0, target tokens get default weight 1.0."""
    prompt = "Hello world"
    target = "This is the answer."
    full_text = prompt + target
    target_start = len(prompt)  # 11

    weights = build_token_loss_weights_from_spans(
        _TOKENIZER, full_text, target_start_char=target_start,
        weight_spans=[], default_target_weight=1.0,
    )

    # 打印详情，肉眼确认
    print(format_token_weight_table(_TOKENIZER, full_text, weights, target_start))

    # 结构性断言
    encoded = _TOKENIZER(full_text, return_offsets_mapping=True, add_special_tokens=False)
    assert len(weights) == len(encoded["input_ids"]), "weights 数量 != token 数量"
    assert all(w >= 0 for w in weights), "不应有负权重"

    # prompt 部分全 0（通过原文子串独立判断：前 target_start 个字符）
    prompt_chars = set(range(0, target_start))
    target_chars = set(range(target_start, len(full_text)))
    offsets = encoded["offset_mapping"]

    for i, (cs, ce) in enumerate(offsets):
        tok_chars = set(range(cs, ce))
        if tok_chars.issubset(prompt_chars):
            assert weights[i] == 0.0, f"token {i} ({full_text[cs:ce]!r}) 全在 prompt 内，应为 0"
        if tok_chars.issubset(target_chars):
            assert weights[i] == 1.0, f"token {i} ({full_text[cs:ce]!r}) 全在 target 内，应为 1.0"


def test_build_token_loss_weights_span_override():
    """weight_spans 覆盖 default_target_weight。"""
    prompt = "Q: "
    target = "AABBCCDD"
    full_text = prompt + target
    target_start = len(prompt)  # 3

    # target 内 [0,4)='AABB' -> 0.5,  [4,8)='CCDD' -> 0.1
    # default_target_weight=0.0 (被 span 覆盖的区域用 span weight)
    spans = [
        {"start_char": 0, "end_char": 4, "weight": 0.5},
        {"start_char": 4, "end_char": 8, "weight": 0.1},
    ]
    weights = build_token_loss_weights_from_spans(
        _TOKENIZER, full_text, target_start_char=target_start,
        weight_spans=spans, default_target_weight=0.0,
    )

    print(format_token_weight_table(_TOKENIZER, full_text, weights, target_start, spans))

    encoded = _TOKENIZER(full_text, return_offsets_mapping=True, add_special_tokens=False)
    assert len(weights) == len(encoded["input_ids"])
    assert all(w >= 0 for w in weights)

    # prompt 全 0
    offsets = encoded["offset_mapping"]
    for i, (cs, ce) in enumerate(offsets):
        if ce <= target_start:
            assert weights[i] == 0.0, f"token {i} 在 prompt 中，应为 0"

    # target 内不应有 default (0.0) 以外的"意外"权重
    # 所有 target token 的权重只能是 0.5 或 0.1（因为 spans 覆盖了整个 target）
    for i, (cs, ce) in enumerate(offsets):
        if cs >= target_start:
            assert weights[i] in (0.5, 0.1), (
                f"token {i} ({full_text[cs:ce]!r}) chars=[{cs},{ce}) "
                f"应为 0.5 或 0.1，实际 {weights[i]}"
            )


def testprepare_loss_weights_pad_and_shift():
    """prepare_loss_weights should pad to seq_len+1 then shift (drop first)."""
    from gpatch_v4.extended_model.llm import PrepareDataForwardLLM

    class StubConfig:
        pass

    pdf = PrepareDataForwardLLM.__new__(PrepareDataForwardLLM)
    pdf.config = StubConfig()

    # 1D weights [w0, w1, w2, w3] with seq_len=6
    lw = torch.tensor([0.0, 0.5, 0.8, 1.0])
    result = pdf.prepare_loss_weights(lw, seq_len=6)
    # pad to 7: [0.0, 0.5, 0.8, 1.0, 0.0, 0.0, 0.0]
    # shift (drop first): [0.5, 0.8, 1.0, 0.0, 0.0, 0.0]
    assert result.shape == (6,)
    assert torch.equal(result, torch.tensor([0.5, 0.8, 1.0, 0.0, 0.0, 0.0]))


def testprepare_loss_weights_truncate():
    """prepare_loss_weights should truncate when input exceeds seq_len."""
    from gpatch_v4.extended_model.llm import PrepareDataForwardLLM

    class StubConfig:
        pass

    pdf = PrepareDataForwardLLM.__new__(PrepareDataForwardLLM)
    pdf.config = StubConfig()

    # 1D weights longer than seq_len=3
    lw = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    result = pdf.prepare_loss_weights(lw, seq_len=3)
    # shift first: [0.1, 0.2, 0.3, 0.4, 0.5]
    # then take last 3: [0.3, 0.4, 0.5]
    assert result.shape == (3,)
    assert torch.equal(result, torch.tensor([0.3, 0.4, 0.5]))
