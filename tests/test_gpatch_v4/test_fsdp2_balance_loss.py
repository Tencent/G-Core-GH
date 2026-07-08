# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""load_balancing_loss_func 单元测试。

校验：
1. 非 CP 路径与 HF switch-loss 参考实现数值一致；
2. 梯度只经 router_logits 回传且有限；
3. CP 路径（gloo 多进程，CPU）与单进程全局结果一致：loss 值相等、且
   各 rank 对本地 shard 的梯度拼起来等于单进程全局逐 token 梯度——验证
   CpMean 的全局均值与 1/T 梯度，以及框架对 CP 求和的约定。
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint

from gpatch_v4.training_backend.loss_factory import load_balancing_loss_func


def _ref_switch_loss(gate_logits, num_experts, top_k):
    """HF load_balancing_loss_func（attention_mask=None 分支）参考实现。"""
    all_logits = torch.cat([r for r in gate_logits], dim=0)
    routing_weights = torch.softmax(all_logits, dim=-1)
    selected = torch.topk(routing_weights, top_k, dim=-1).indices
    expert_mask = torch.nn.functional.one_hot(selected, num_experts).float()
    tokens_per_expert = expert_mask.mean(dim=0)
    prob_per_expert = routing_weights.mean(dim=0)
    return (tokens_per_expert * prob_per_expert.unsqueeze(0)).sum() * num_experts


def _make_logits(num_layers, n_tokens, num_experts, seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(n_tokens, num_experts, generator=g, dtype=torch.float64)
        for _ in range(num_layers)
    )


def test_matches_reference():
    top_k, num_experts, n_tokens, num_layers = 2, 8, 16, 3
    logits = _make_logits(num_layers, n_tokens, num_experts)

    got = load_balancing_loss_func(gate_logits=logits, num_experts=num_experts, top_k=top_k)
    ref = _ref_switch_loss(logits, num_experts, top_k)
    assert torch.allclose(got, ref, atol=1e-10), f"{got=} {ref=}"


def test_gradient_flows():
    top_k, num_experts, n_tokens, num_layers = 2, 8, 16, 2
    logits = tuple(
        torch.randn(n_tokens, num_experts, dtype=torch.float64, requires_grad=True)
        for _ in range(num_layers)
    )

    loss = load_balancing_loss_func(gate_logits=logits, num_experts=num_experts, top_k=top_k)
    loss.backward()
    for r in logits:
        assert r.grad is not None
        assert torch.isfinite(r.grad).all()


def test_captured_logits_grad_under_gradient_checkpointing():
    """recompute 下 forward-hook 捕获的中间激活（router_logits 类比）能否反传。

    这是 balance loss 兼容 recompute 的前提：balance loss 读的 router_logits
    是 hook 从 MoE 层内部抓出来的中间激活。结论：

    - use_reentrant=False（transformers 默认、本仓库实际用的）：捕获张量保留
      grad_fn，梯度与不开 ckpt 完全一致；
    - use_reentrant=True：hook 抓到 no-grad 首次前向的张量，aux 梯度被**静默
      丢弃**（当 layer 输入 requires_grad 时不会报错），所以 recompute 必须保持
      non-reentrant。
    """
    w = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)

    def run(mode, with_aux):
        captured = []

        def block(x):
            h = x @ w                 # router_logits 类比（ckpt 区域内中间激活）
            captured.append(h)        # forward-hook 捕获到区域外
            return torch.relu(h) @ w  # 下游 layer 输出

        # layer 输入 requires_grad=True，对齐真实场景（前一层激活）
        x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        if mode == "none":
            y = block(x)
        else:
            y = checkpoint(block, x, use_reentrant=(mode == "reentrant"))
        loss = y.sum()
        if with_aux:
            loss = loss + captured[0].pow(2).sum()  # aux loss 走捕获张量
        loss.backward()
        return w.grad.clone()

    def grad(mode, with_aux):
        torch.manual_seed(1)  # 固定 x
        w.grad = None
        return run(mode, with_aux)

    g_main = grad("none", with_aux=False)
    g_full = grad("none", with_aux=True)
    assert not torch.allclose(g_main, g_full), "aux 应当改变梯度，否则测试无意义"

    # non-reentrant：与含 aux 的完整梯度一致
    assert torch.allclose(grad("nonreentrant", with_aux=True), g_full, atol=1e-10)
    # reentrant：aux 被静默丢弃，退化为 main-only 梯度
    assert torch.allclose(grad("reentrant", with_aux=True), g_main, atol=1e-10)


def _cp_worker(rank, world, shard_splits, full_layers, num_experts, top_k):
    """单个 CP rank：跑 cp_group 路径，rank 0 与单进程全局结果对拍。

    shard_splits: list[int]，每个 rank 拥有的 token 数（允许各 rank 不同）。
    full_layers: tuple，每层 [T, e]（T = sum(shard_splits)）。各 rank 按
    shard_splits 切出本地 shard。
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29577"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        group = dist.group.WORLD
        offsets = [sum(shard_splits[:i]) for i in range(world + 1)]

        local_layers = tuple(
            layer[offsets[rank]:offsets[rank + 1]].clone().detach().requires_grad_(True)
            for layer in full_layers
        )
        loss = load_balancing_loss_func(
            gate_logits=local_layers, num_experts=num_experts, top_k=top_k, cp_group=group
        )
        loss.backward()

        # 收集各 rank 梯度。由于各 rank 的 shard 长度不同，用 all_gather 需
        # padding 到最大长度；这里简单地让 rank 0 gather 各 rank 的梯度。
        max_n = max(shard_splits)
        num_e = full_layers[0].shape[1]
        cat_grads = []
        for li, lyr in enumerate(local_layers):
            padded = torch.zeros(max_n, num_e, dtype=lyr.grad.dtype)
            padded[:shard_splits[rank]] = lyr.grad
            bucket = [torch.zeros(max_n, num_e, dtype=lyr.grad.dtype) for _ in range(world)]
            dist.all_gather(bucket, padded)
            # 按各 rank 的真实长度裁切后拼接
            cat_grads.append(torch.cat([bucket[r][:shard_splits[r]] for r in range(world)], dim=0))

        if rank == 0:
            ref_layers = tuple(l.clone().detach().requires_grad_(True) for l in full_layers)
            ref_loss = load_balancing_loss_func(
                gate_logits=ref_layers, num_experts=num_experts, top_k=top_k, cp_group=None
            )
            ref_loss.backward()

            assert torch.allclose(loss, ref_loss, rtol=1e-6, atol=1e-9), \
                f"CP loss {loss.item()} != single-proc {ref_loss.item()}"
            for li, ref_lyr in enumerate(ref_layers):
                assert torch.allclose(cat_grads[li], ref_lyr.grad, rtol=1e-6, atol=1e-8), \
                    f"layer {li} grad mismatch, max diff " \
                    f"{(cat_grads[li] - ref_lyr.grad).abs().max().item():.3e}"
    finally:
        dist.destroy_process_group()


def test_cp_matches_single_process():
    """cp_size>1 各 rank 不等长序列：全局 balance loss 与单进程一致（值 + 梯度）。"""
    world, top_k, num_experts, num_layers = 4, 2, 8, 2
    # 各 rank 的 token 数故意不同
    shard_splits = [5, 9, 3, 7]
    total_tokens = sum(shard_splits)
    full_layers = _make_logits(num_layers, total_tokens, num_experts, seed=11)
    mp.spawn(
        _cp_worker,
        args=(world, shard_splits, full_layers, num_experts, top_k),
        nprocs=world,
        join=True,
    )
