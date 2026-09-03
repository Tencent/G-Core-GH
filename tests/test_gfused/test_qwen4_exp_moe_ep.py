"""Expert-parallel MoE 的测试。

核心性质：EP 是**精确**的，不是近似——把 expert 切到多个 rank 上算，输出必须等于
单 rank 全量 expert 的结果；expert 梯度必须只落在 owner 上，且等于"所有 rank 的
token 合起来对全量 expert 求梯度"后的对应切片。

多 rank 部分用 gloo/CPU，本地就能跑。
"""
import datetime
import os
import pathlib
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gpatch_v4.models.qwen4_exp import Qwen4ExpTextConfig
from gpatch_v4.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextExperts,
    Qwen4ExpTextTopKRouter,
)
from gpatch_v4.models.qwen4_exp.moe import Qwen4ExpEPExperts

NUM_EXPERTS = 8
TOP_K = 2
HIDDEN = 16
INTERMEDIATE = 8

TINY_KWARGS = dict(
    vocab_size=128,
    hidden_size=HIDDEN,
    num_hidden_layers=4,
    full_attention_interval=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    linear_num_key_heads=1,
    linear_num_value_heads=2,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=4,
    output_gate_type="sigmoid",
    num_experts=NUM_EXPERTS,
    num_experts_per_tok=TOP_K,
    moe_intermediate_size=INTERMEDIATE,
    shared_expert_intermediate_size=INTERMEDIATE,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=16,
    indexer_compress_ratio=4,
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
    return Qwen4ExpTextConfig(**{**TINY_KWARGS, **overrides})


def _routing(config, hidden_states, seed: int):
    """用真实 router 产生 top-k 路由，避免手工构造出不真实的分布。"""
    torch.manual_seed(seed)
    router = Qwen4ExpTextTopKRouter(config)
    router.weight.data.normal_()
    with torch.no_grad():
        _, weights, indices = router(hidden_states)
    return indices, weights


# ---------------------------------------------------------------------------
# 单进程
# ---------------------------------------------------------------------------


def test_ep_size_one_matches_upstream_dense_forward() -> None:
    torch.manual_seed(0)
    config = _config()
    upstream = Qwen4ExpTextExperts(config)
    ours = Qwen4ExpEPExperts(config)
    upstream.gate_up_proj.data.normal_(std=0.05)
    upstream.down_proj.data.normal_(std=0.05)
    ours.load_state_dict(upstream.state_dict())
    ours.configure_ep(None)

    assert ours.ep_size == 1 and ours.num_local_experts == NUM_EXPERTS
    assert ours.ep_rank == 0 and ours.ep_group is None

    hidden_states = torch.randn(12, HIDDEN)
    indices, weights = _routing(config, hidden_states, seed=1)
    with torch.no_grad():
        expected = upstream(hidden_states, indices, weights)
        got = ours(hidden_states, indices, weights)
    torch.testing.assert_close(got, expected)


def test_empty_input_keeps_expert_params_in_graph() -> None:
    """零 token 时 expert 参数必须仍在 autograd 图里，否则 FSDP/EP 会失步。"""
    torch.manual_seed(0)
    ours = Qwen4ExpEPExperts(_config())
    ours.gate_up_proj.data.normal_(std=0.05)
    ours.down_proj.data.normal_(std=0.05)

    hidden_states = torch.empty(0, HIDDEN, requires_grad=True)
    out = ours._local_experts(hidden_states, torch.empty(0, dtype=torch.long))

    assert out.shape == hidden_states.shape
    out.sum().backward()
    assert ours.gate_up_proj.grad is not None
    assert ours.down_proj.grad is not None
    assert torch.count_nonzero(ours.gate_up_proj.grad) == 0
    assert torch.count_nonzero(ours.down_proj.grad) == 0


# ---------------------------------------------------------------------------
# 多进程（gloo / CPU）
# ---------------------------------------------------------------------------


def _run_workers(fn, world_size: int) -> None:
    # 线程数降到 1：多个 worker 在同一台机器上跑，避免 CPU 线程超订。
    previous_threads = torch.get_num_threads()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = str(pathlib.Path(tmpdir) / "pg_init")
            mp.start_processes(
                fn,
                args=(world_size, init_file),
                nprocs=world_size,
                join=True,
                start_method="spawn",
            )
    finally:
        torch.set_num_threads(previous_threads)


def _per_rank_tokens(world_size: int, num_tokens: int = 10):
    torch.manual_seed(4242)
    return [torch.randn(num_tokens, HIDDEN) for _ in range(world_size)]


def _init_pg(rank: int, world_size: int, init_file: str) -> None:
    """带超时地建 gloo group。

    超时很重要：某个 rank 抛异常时，其余 rank 还卡在集合通信上，默认 30 分钟超时会让
    测试看起来是"挂住"而不是"失败"。
    """
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )


def _ep_worker(rank: int, world_size: int, init_file: str) -> None:
    _init_pg(rank, world_size, init_file)
    try:
        config = _config()
        torch.manual_seed(0)
        dense = Qwen4ExpTextExperts(config)
        dense.gate_up_proj.data.normal_(std=0.05)
        dense.down_proj.data.normal_(std=0.05)

        local_count = NUM_EXPERTS // world_size
        expert_slice = slice(rank * local_count, (rank + 1) * local_count)

        ep_experts = Qwen4ExpEPExperts(config)
        ep_experts.configure_ep(dist.group.WORLD)
        assert ep_experts.ep_size == world_size
        assert ep_experts.num_local_experts == local_count
        # apply_hp 会做真正的切片；这里手工切，等价
        ep_experts.gate_up_proj = torch.nn.Parameter(
            dense.gate_up_proj.data[expert_slice].clone()
        )
        ep_experts.down_proj = torch.nn.Parameter(dense.down_proj.data[expert_slice].clone())

        all_tokens = _per_rank_tokens(world_size)
        hidden_states = all_tokens[rank].clone().requires_grad_(True)
        indices, weights = _routing(config, all_tokens[rank], seed=1)
        weights = weights.clone().requires_grad_(True)

        got = ep_experts(hidden_states, indices, weights)

        # 1) 前向必须等于全量 expert 的稠密结果
        reference_input = all_tokens[rank].clone().requires_grad_(True)
        reference_weights = weights.detach().clone().requires_grad_(True)
        expected = dense(reference_input, indices, reference_weights)
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)

        # 避免把 dense 的参数梯度累加一份，污染下面的 owner 梯度参考。
        reference_input_grad, reference_weight_grad = torch.autograd.grad(
            expected.sum(), (reference_input, reference_weights)
        )

        # 2) 反向：本 rank 的 expert 梯度 == 用所有 rank 的 token 对全量 expert 求梯度后的切片
        got.sum().backward()
        total = 0.0
        for other_rank in range(world_size):
            other_indices, other_weights = _routing(config, all_tokens[other_rank], seed=1)
            total = total + dense(all_tokens[other_rank], other_indices, other_weights).sum()
        total.backward()

        torch.testing.assert_close(
            ep_experts.gate_up_proj.grad, dense.gate_up_proj.grad[expert_slice],
            rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            ep_experts.down_proj.grad, dense.down_proj.grad[expert_slice], rtol=1e-5, atol=1e-6
        )
        # 3) 输入梯度也必须对（combine 的逆置换写错就会挂）
        assert hidden_states.grad is not None
        torch.testing.assert_close(
            hidden_states.grad, reference_input_grad, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            weights.grad, reference_weight_grad, rtol=1e-5, atol=1e-6
        )
    finally:
        dist.destroy_process_group()


def test_expert_parallel_matches_dense_and_owns_gradients() -> None:
    _run_workers(_ep_worker, 4)


def _skewed_routing_worker(rank: int, world_size: int, init_file: str) -> None:
    """所有 token 都路由到 rank 0 的 expert：其余 rank 收到 0 个 token。"""
    _init_pg(rank, world_size, init_file)
    try:
        config = _config()
        torch.manual_seed(0)
        dense = Qwen4ExpTextExperts(config)
        dense.gate_up_proj.data.normal_(std=0.05)
        dense.down_proj.data.normal_(std=0.05)

        local_count = NUM_EXPERTS // world_size
        expert_slice = slice(rank * local_count, (rank + 1) * local_count)
        ep_experts = Qwen4ExpEPExperts(config)
        ep_experts.configure_ep(dist.group.WORLD)
        ep_experts.gate_up_proj = torch.nn.Parameter(dense.gate_up_proj.data[expert_slice].clone())
        ep_experts.down_proj = torch.nn.Parameter(dense.down_proj.data[expert_slice].clone())

        torch.manual_seed(7)
        hidden_states = torch.randn(6, HIDDEN, requires_grad=True)
        # 只用 expert 0 和 1，两者都归 rank 0
        indices = torch.zeros(6, TOP_K, dtype=torch.long)
        indices[:, 1] = 1
        weights = torch.full((6, TOP_K), 0.5)

        got = ep_experts(hidden_states, indices, weights)
        expected = dense(hidden_states.detach().clone().requires_grad_(True), indices, weights)
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)

        got.sum().backward()
        if rank == 0:
            assert torch.count_nonzero(ep_experts.gate_up_proj.grad) > 0
        else:
            # 没收到任何 token，但梯度必须存在且为 0（不能是 None，否则 FSDP 归约会失步）
            assert ep_experts.gate_up_proj.grad is not None
            assert torch.count_nonzero(ep_experts.gate_up_proj.grad) == 0
    finally:
        dist.destroy_process_group()


def test_skewed_routing_leaves_idle_ranks_with_zero_grads() -> None:
    _run_workers(_skewed_routing_worker, 2)


def _indivisible_worker(rank: int, world_size: int, init_file: str) -> None:
    _init_pg(rank, world_size, init_file)
    try:
        experts = Qwen4ExpEPExperts(_config(num_experts=6, num_experts_per_tok=2))
        try:
            experts.configure_ep(dist.group.WORLD)
        except ValueError as error:
            assert "divisible" in str(error)
        else:
            raise AssertionError("expected ValueError for indivisible num_experts")
    finally:
        dist.destroy_process_group()


def test_indivisible_expert_count_is_rejected() -> None:
    _run_workers(_indivisible_worker, 4)
