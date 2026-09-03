"""QSA（Qwen Sparse Attention）向量化选择器的 CPU 测试。

最关键的一条是 parity：向量化选择器必须和上游那个逐 (batch, query) 的 Python
双层循环选出同样最优的 key 集合；只有 top-k 边界没有并列时才要求集合完全相同。
其余覆盖稀疏起始行、causal tail、padding 行、以及 indexer 冻结。
"""
import math

import pytest
import torch

from gpatch_v4.models.qwen4_exp import Qwen4ExpTextConfig
from gpatch_v4.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextQSAIndexer,
    apply_rotary_pos_emb,
)
from gpatch_v4.models.qwen4_exp.qsa import (
    Qwen4ExpQSAAttention,
    Qwen4ExpQSAIndexer,
    dense_sparse_gqa_attention,
    flex_sparse_gqa_attention,
    select_qsa_membership,
)

INDEXER_KWARGS = dict(
    vocab_size=256,
    hidden_size=64,
    num_hidden_layers=4,
    full_attention_interval=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    output_gate_type="sigmoid",
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=16,
    shared_expert_intermediate_size=16,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    hc_count=4,
    hc_lowrank=8,
    eos_token_id=1,
    tie_word_embeddings=False,
    use_cache=False,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
    },
)


def _config(**overrides) -> Qwen4ExpTextConfig:
    kwargs = {**INDEXER_KWARGS, **overrides}
    kwargs.setdefault("indexer_budget", 16)
    kwargs.setdefault("indexer_compress_ratio", 4)
    return Qwen4ExpTextConfig(**kwargs)


def _rope_cos_sin(seq_len: int, rot_dim: int, dtype=torch.float32):
    inv_freq = 1.0 / (10000.0**(torch.arange(0, rot_dim, 2, dtype=torch.float) / rot_dim))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None].to(dtype), emb.sin()[None].to(dtype)


def _causal_bool_mask(batch_size: int, q_len: int, seq_lens=None) -> torch.Tensor:
    """右侧 padding 的 4D causal bool mask，形状 [B, 1, S, S]。"""
    positions = torch.arange(q_len)
    mask = positions[None, :, None] >= positions[None, None, :]
    mask = mask.expand(batch_size, q_len, q_len).clone()
    if seq_lens is not None:
        key_valid = positions[None, None, :] < torch.as_tensor(seq_lens)[:, None, None]
        mask &= key_valid
    return mask.unsqueeze(1)


@torch.no_grad()
def _upstream_block_scores(indexer, hidden_states, cos, sin, mask, batch_idx, query_idx):
    """按上游的算法重算某一行的 per-block 分数，用于判断 top-k 边界是否有并列。"""
    batch_size, q_len, _ = hidden_states.shape
    head_dim = indexer.index_head_dim
    qk = indexer.index_qk_proj(hidden_states)
    query_states, token_k = torch.split(
        qk, [indexer.index_n_heads * head_dim, indexer.index_kv_heads * head_dim], dim=-1
    )
    shape = (batch_size, q_len, -1, head_dim)
    query_states = indexer.q_layernorm(query_states.reshape(*shape))
    query_states = apply_rotary_pos_emb(
        query_states, cos=cos[:, -q_len:, :], sin=sin[:, -q_len:, :], unsqueeze_dim=2
    )
    raw_keys = token_k.reshape(*shape).squeeze(2)

    visible = torch.nonzero(mask[batch_idx, 0, query_idx]).flatten()
    num_blocks = visible.shape[-1] // indexer.compress_ratio
    block_tokens = visible[:num_blocks * indexer.compress_ratio].view(-1, indexer.compress_ratio)
    groups = raw_keys[batch_idx].index_select(0, block_tokens.flatten())
    groups = groups.view(*block_tokens.shape, head_dim)
    pooled = indexer.k_layernorm(groups.float().mean(dim=1).to(raw_keys.dtype))
    block_keys = apply_rotary_pos_emb(
        pooled.unsqueeze(1),
        cos=cos[batch_idx].index_select(0, block_tokens[:, 0]),
        sin=sin[batch_idx].index_select(0, block_tokens[:, 0]),
    ).squeeze(1)
    scores = torch.matmul(
        query_states[batch_idx, query_idx].float(), block_keys.float().transpose(-1, -2)
    ).transpose(-1, -2)
    return torch.relu(scores).sum(dim=-1) / math.sqrt(head_dim), num_blocks


def _selected_blocks(membership_row, num_blocks, ratio):
    return {b for b in range(num_blocks) if membership_row[b * ratio:(b + 1) * ratio].all()}


@pytest.mark.parametrize(
    "seed, batch_size, q_len, seq_lens",
    [(0, 2, 37, None), (1, 3, 29, [29, 21, 9])],
)
def test_vectorized_selection_agrees_with_upstream_loop(seed, batch_size, q_len, seq_lens) -> None:
    """核心 parity。

    分数是 ``relu(q·k).sum(heads)``，所有 index head 的点积都为负时分数**恰好为 0**，
    因此 0 并列很常见。并列跨过 top-k 边界时，选中的集合本身就是二义的（上游 topk、
    我们的 top-k、参考 CUDA kernel 会各选一个同样最优的集合）。所以这里断言：
    无并列的行必须逐元素相同；所有行的选中集合总分必须等于最优值。
    """
    torch.manual_seed(seed)
    config = _config()
    ratio = config.indexer_compress_ratio
    block_topk = config.indexer_budget // ratio

    upstream = Qwen4ExpTextQSAIndexer(config, layer_idx=3).eval()
    ours = Qwen4ExpQSAIndexer(config, layer_idx=3).eval()
    ours.load_state_dict(upstream.state_dict())

    hidden_states = torch.randn(batch_size, q_len, config.hidden_size)
    rot_dim = int(config.head_dim * config.rope_parameters["partial_rotary_factor"])
    cos, sin = _rope_cos_sin(q_len, rot_dim)
    cos, sin = cos.expand(batch_size, -1, -1), sin.expand(batch_size, -1, -1)
    attention_mask = _causal_bool_mask(batch_size, q_len, seq_lens=seq_lens)

    with torch.no_grad():
        expected = upstream(hidden_states, (cos, sin), attention_mask, None).squeeze(1)
        got = ours(hidden_states, (cos, sin), attention_mask, None)

    assert got.dtype == torch.bool
    assert got.shape == expected.shape

    tied_rows = 0
    for batch_idx in range(batch_size):
        for query_idx in range(q_len):
            scores, num_blocks = _upstream_block_scores(
                upstream, hidden_states, cos, sin, attention_mask, batch_idx, query_idx
            )
            k = min(block_topk, num_blocks)
            ordered = scores.sort(descending=True).values
            untied = k == num_blocks or not torch.isclose(ordered[k - 1], ordered[k])
            row_expected = expected[batch_idx, query_idx]
            row_got = got[batch_idx, query_idx]

            if untied:
                torch.testing.assert_close(
                    row_got, row_expected, rtol=0, atol=0,
                    msg=f"untied row (b={batch_idx}, q={query_idx}) must match exactly"
                )
                continue

            tied_rows += 1
            # 并列行：集合可以不同，但必须同样最优，且块数与 tail 一致
            ours_blocks = _selected_blocks(row_got, num_blocks, ratio)
            upstream_blocks = _selected_blocks(row_expected, num_blocks, ratio)
            assert len(ours_blocks) == len(upstream_blocks) == k
            assert math.isclose(
                sum(float(scores[b]) for b in ours_blocks),
                float(ordered[:k].sum()),
                rel_tol=1e-6,
                abs_tol=1e-6,
            ), f"(b={batch_idx}, q={query_idx}) selection is not score-optimal"
            assert int(row_got.sum()) == int(row_expected.sum())

    # 这个 tiny config 只有 2 个 index head，并列应当真实出现过（否则测试没覆盖到）
    assert tied_rows > 0


def _membership_for(q_len: int, budget: int, ratio: int, n_heads: int = 4, head_dim: int = 16):
    torch.manual_seed(7)
    k_layernorm = torch.nn.Identity()
    index_q = torch.randn(1, q_len, n_heads, head_dim)
    raw_k = torch.randn(1, q_len, head_dim)
    cos, sin = _rope_cos_sin(q_len, head_dim)
    visible = _causal_bool_mask(1, q_len).squeeze(1)
    return select_qsa_membership(
        index_q, raw_k, visible, cos, sin, k_layernorm, ratio, budget // ratio
    ), visible


def test_first_sparse_row_is_2051_for_released_budget() -> None:
    """真实配置 budget=2048 / ratio=4：第一个真正稀疏的 query 位置是 2051。

    q=2050 时 visible_blocks = 2051//4 = 512 == block_topk，全选，等于 causal；
    q=2051 时 visible_blocks = 2052//4 = 513 > 512，开始丢块。
    """
    membership, visible = _membership_for(q_len=2053, budget=2048, ratio=4)
    dense_rows = (membership == visible).all(dim=-1)[0]
    first_sparse = int((~dense_rows).nonzero()[0])
    assert first_sparse == 2051
    assert dense_rows[:2051].all()


def test_below_budget_selection_equals_plain_causal_mask() -> None:
    membership, visible = _membership_for(q_len=16, budget=2048, ratio=4)
    torch.testing.assert_close(membership, visible, rtol=0, atol=0)


def test_sparse_membership_respects_budget_tail_and_causality() -> None:
    budget, ratio = 16, 4
    q_len = 64
    membership, visible = _membership_for(q_len=q_len, budget=budget, ratio=ratio)
    assert int(membership.sum(dim=-1).max()) <= budget + ratio - 1
    assert not (membership & ~visible).any()

    key_ids = torch.arange(q_len)
    for query_idx in range(q_len):
        visible_blocks = (query_idx + 1) // ratio
        tail = (key_ids >= visible_blocks * ratio) & (key_ids <= query_idx)
        assert membership[0, query_idx][tail].all(), query_idx


def test_left_padding_is_rejected() -> None:
    q_len = 12
    visible = _causal_bool_mask(1, q_len).squeeze(1).clone()
    visible[0, :, 0] = False  # 把最前面的 key 挖掉 -> 不再是 [0, n) 前缀
    index_q = torch.randn(1, q_len, 2, 16)
    raw_k = torch.randn(1, q_len, 16)
    cos, sin = _rope_cos_sin(q_len, 16)
    with pytest.raises(ValueError, match="contiguous prefix"):
        select_qsa_membership(index_q, raw_k, visible, cos, sin, torch.nn.Identity(), 4, 4)


def test_dense_attention_zeroes_rows_without_routes() -> None:
    batch_size, n_q, n_kv, q_len, head_dim = 1, 4, 2, 8, 16
    query_states = torch.randn(batch_size, n_q, q_len, head_dim, requires_grad=True)
    key_states = torch.randn(batch_size, n_kv, q_len, head_dim)
    value_states = torch.randn(batch_size, n_kv, q_len, head_dim)
    membership = torch.zeros(batch_size, q_len, q_len, dtype=torch.bool)
    membership[0, 1:, :] = True  # 第 0 行完全没有可选 key（模拟纯 padding query 行）

    out = dense_sparse_gqa_attention(query_states, key_states, value_states, membership, 0.125)
    assert torch.isfinite(out).all()
    assert (out[:, :, 0] == 0).all()

    out.float().pow(2).mean().backward()
    assert torch.isfinite(query_states.grad).all()


def test_qsa_attention_indexer_is_frozen_and_rest_trains() -> None:
    torch.manual_seed(3)
    config = _config()
    attn = Qwen4ExpQSAAttention(config, layer_idx=3, attn_backend="dense")
    attn.train()

    assert not any(p.requires_grad for p in attn.indexer.parameters())

    q_len = 20
    hidden_states = torch.randn(1, q_len, config.hidden_size, requires_grad=True)
    rot_dim = int(config.head_dim * config.rope_parameters["partial_rotary_factor"])
    cos, sin = _rope_cos_sin(q_len, rot_dim)
    attention_mask = _causal_bool_mask(1, q_len)

    out, weights = attn(hidden_states, (cos, sin), attention_mask)
    assert weights is None
    assert out.shape == (1, q_len, config.hidden_size)
    out.float().pow(2).mean().backward()

    assert all(p.grad is None for p in attn.indexer.parameters())
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        param = getattr(attn, name).weight
        assert param.grad is not None and torch.isfinite(param.grad).all(), name


def test_flex_backend_refuses_to_run_on_cpu() -> None:
    config = _config()
    attn = Qwen4ExpQSAAttention(config, layer_idx=3, attn_backend="flex")
    q_len = 12
    hidden_states = torch.randn(1, q_len, config.hidden_size)
    rot_dim = int(config.head_dim * config.rope_parameters["partial_rotary_factor"])
    cos, sin = _rope_cos_sin(q_len, rot_dim)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        attn(hidden_states, (cos, sin), _causal_bool_mask(1, q_len))


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="attn_backend"):
        Qwen4ExpQSAAttention(_config(), layer_idx=3, attn_backend="sdpa")


def test_indexer_rejects_kv_cache() -> None:
    config = _config()
    indexer = Qwen4ExpQSAIndexer(config, layer_idx=3)
    q_len = 8
    hidden_states = torch.randn(1, q_len, config.hidden_size)
    cos, sin = _rope_cos_sin(q_len, 8)
    with pytest.raises(NotImplementedError, match="KV cache"):
        indexer(hidden_states, (cos, sin), _causal_bool_mask(1, q_len), past_key_values=object())


def test_attention_rejects_kv_cache() -> None:
    config = _config()
    attention = Qwen4ExpQSAAttention(config, layer_idx=3, attn_backend="dense")
    q_len = 8
    hidden_states = torch.randn(1, q_len, config.hidden_size)
    cos, sin = _rope_cos_sin(q_len, 8)
    with pytest.raises(NotImplementedError, match="KV cache"):
        attention(
            hidden_states,
            (cos, sin),
            _causal_bool_mask(1, q_len),
            past_key_values=object(),
        )


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")


@requires_cuda
@pytest.mark.parametrize("q_len", [256, 512])
def test_flex_matches_dense_oracle_on_cuda(q_len: int) -> None:
    """flex 路径必须和 dense oracle 数值一致（真实 QSA 头型：24 Q / 2 KV / head_dim 256）。"""
    torch.manual_seed(0)
    n_q, n_kv, head_dim = 24, 2, 256
    scaling = head_dim**-0.5
    query_states = torch.randn(1, n_q, q_len, head_dim, device="cuda", dtype=torch.bfloat16)
    key_states = torch.randn(1, n_kv, q_len, head_dim, device="cuda", dtype=torch.bfloat16)
    value_states = torch.randn(1, n_kv, q_len, head_dim, device="cuda", dtype=torch.bfloat16)

    membership, _ = _membership_for(q_len=q_len, budget=64, ratio=4, n_heads=4, head_dim=16)
    membership = membership.to("cuda")

    dense = dense_sparse_gqa_attention(
        query_states, key_states, value_states, membership, scaling
    )
    flex = flex_sparse_gqa_attention(query_states, key_states, value_states, membership, scaling)

    assert flex.shape == dense.shape
    assert torch.isfinite(flex).all()
    torch.testing.assert_close(flex.float(), dense.float(), rtol=2e-2, atol=2e-2)


@requires_cuda
def test_flex_backward_is_finite_on_cuda() -> None:
    torch.manual_seed(0)
    q_len, n_q, n_kv, head_dim = 256, 24, 2, 256
    tensors = [
        torch.randn(1, heads, q_len, head_dim, device="cuda", dtype=torch.bfloat16,
                    requires_grad=True) for heads in (n_q, n_kv, n_kv)
    ]
    membership, _ = _membership_for(q_len=q_len, budget=64, ratio=4, n_heads=4, head_dim=16)
    out = flex_sparse_gqa_attention(*tensors, membership.to("cuda"), head_dim**-0.5)
    out.float().pow(2).mean().backward()
    assert all(torch.isfinite(t.grad).all() for t in tensors)


@requires_cuda
def test_flex_zeroes_rows_without_routes_on_cuda() -> None:
    q_len, n_q, n_kv, head_dim = 128, 4, 2, 64
    query_states = torch.randn(1, n_q, q_len, head_dim, device="cuda", dtype=torch.bfloat16)
    key_states = torch.randn(1, n_kv, q_len, head_dim, device="cuda", dtype=torch.bfloat16)
    value_states = torch.randn(1, n_kv, q_len, head_dim, device="cuda", dtype=torch.bfloat16)
    membership = torch.zeros(1, q_len, q_len, dtype=torch.bool, device="cuda")
    membership[0, 1:, :] = True

    out = flex_sparse_gqa_attention(
        query_states, key_states, value_states, membership, head_dim**-0.5
    )
    assert torch.isfinite(out).all()
    assert (out[:, :, 0] == 0).all()


def test_scores_use_relu_sum_over_heads_scaled_by_sqrt_head_dim() -> None:
    """按定义手算一行分数，确认 relu -> sum(heads) -> /sqrt(D) 的顺序。"""
    ratio, budget, head_dim, n_heads = 4, 8, 16, 3
    q_len = 16
    torch.manual_seed(5)
    index_q = torch.randn(1, q_len, n_heads, head_dim)
    raw_k = torch.randn(1, q_len, head_dim)
    cos = torch.ones(1, q_len, head_dim)
    sin = torch.zeros(1, q_len, head_dim)  # RoPE 退化为恒等，便于手算
    visible = _causal_bool_mask(1, q_len).squeeze(1)

    membership = select_qsa_membership(
        index_q, raw_k, visible, cos, sin, torch.nn.Identity(), ratio, budget // ratio
    )

    query_idx = q_len - 1
    num_blocks = (query_idx + 1) // ratio
    pooled = raw_k[0, :num_blocks * ratio].view(num_blocks, ratio, head_dim).mean(dim=1)
    scores = torch.relu(index_q[0, query_idx].float() @ pooled.float().T)
    scores = scores.sum(dim=0) / math.sqrt(head_dim)
    expected_blocks = set(scores.topk(budget // ratio).indices.tolist())

    got_blocks = {
        b
        for b in range(num_blocks) if membership[0, query_idx, b * ratio:(b + 1) * ratio].all()
    }
    assert got_blocks == expected_blocks
