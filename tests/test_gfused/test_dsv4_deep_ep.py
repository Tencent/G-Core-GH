# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.

import importlib.util
import os
import unittest

import ray
import torch
import torch.distributed as dist
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

from gpatch_v4.models.deepseek_v4.deepep_a2a import (
    fused_combine,
    fused_dispatch,
    set_deepep_num_sms,
)

_DEEPEP_AVAILABLE = importlib.util.find_spec("deep_ep") is not None
requires_deepep = unittest.skipUnless(_DEEPEP_AVAILABLE, "deep_ep not installed")


@ray.remote(num_cpus=0, num_gpus=0)
def _get_node_ip():
    return ray.util.get_node_ip_address()


@ray.remote(num_cpus=0, num_gpus=0)
def _get_node_info():
    ctx = ray.get_runtime_context()
    return {"node_id": ctx.get_node_id(), "ip": ray.util.get_node_ip_address()}


@ray.remote(num_gpus=1)
def _deepep_worker(rank: int, world_size: int, master_addr: str, master_port: int):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    set_deepep_num_sms(24)

    hidden_dim = 8
    num_tokens = 6
    num_experts = world_size * 2
    top_k = 2
    tokens = (
        torch.arange(num_tokens * hidden_dim, dtype=torch.bfloat16, device="cuda")
        .reshape(num_tokens, hidden_dim)
        .add_(rank * 100)
    )
    tokens.requires_grad_(True)
    generator = torch.Generator(device="cuda").manual_seed(20260612 + rank)
    topk_idx = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        generator=generator,
        dtype=torch.int64,
        device="cuda",
    )
    topk_weights = torch.rand(
        num_tokens, top_k, generator=generator, dtype=torch.float32, device="cuda"
    )
    topk_weights = torch.softmax(topk_weights, dim=-1)
    topk_weights.requires_grad_(True)

    recv_tokens, recv_topk_idx, recv_weights, tokens_per_expert, handle = fused_dispatch(
        tokens,
        topk_idx,
        topk_weights,
        num_experts,
        dist.group.WORLD,
    )
    valid = recv_topk_idx >= 0
    row_idx, slot_idx = valid.nonzero(as_tuple=True)
    recv_out = recv_tokens.new_zeros(recv_tokens.shape)
    if row_idx.numel() > 0:
        expert_out = recv_tokens.index_select(0, row_idx).float() * 2.0
        expert_out = expert_out * recv_weights[row_idx, slot_idx].unsqueeze(-1)
        recv_out.index_add_(0, row_idx, expert_out.to(recv_out.dtype))

    combined = fused_combine(recv_out, dist.group.WORLD, handle)
    combined.float().sum().backward()
    result = {
        "rank": rank,
        "combined_shape": list(combined.shape),
        "combined_finite": bool(torch.isfinite(combined).all()),
        "tokens_grad_finite": bool(torch.isfinite(tokens.grad).all()),
        "tokens_grad_norm": tokens.grad.float().norm().item(),
        "weights_grad_finite": bool(torch.isfinite(topk_weights.grad).all()),
        "weights_grad_norm": topk_weights.grad.float().norm().item(),
        "tokens_per_expert": tokens_per_expert.cpu().tolist(),
    }
    dist.destroy_process_group()
    return result


@requires_deepep
class TestDeepEPHelper(unittest.TestCase):
    def setUp(self):
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < 2:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(f"need >= 2 GPUs in Ray cluster, only {total_gpus}")

    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    def test_dispatch_combine_autograd(self):
        pg = placement_group([{"GPU": 1, "CPU": 1}] * 2, strategy="PACK")
        ray.get(pg.ready())
        master_addr = ray.get(
            _get_node_ip.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=0,
                )
            ).remote()
        )
        futures = [
            _deepep_worker.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=rank,
                )
            ).remote(rank, 2, master_addr, 12920)
            for rank in range(2)
        ]
        results = ray.get(futures)
        remove_placement_group(pg)

        for res in results:
            self.assertEqual(res["combined_shape"], [6, 8])
            self.assertTrue(res["combined_finite"])
            self.assertTrue(res["tokens_grad_finite"])
            self.assertGreater(res["tokens_grad_norm"], 0.0)
            self.assertTrue(res["weights_grad_finite"])
            self.assertGreater(res["weights_grad_norm"], 0.0)
            self.assertEqual(len(res["tokens_per_expert"]), 2)

    def test_dispatch_combine_autograd_ep16(self):
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < 16:
            raise unittest.SkipTest(f"need >= 16 GPUs in Ray cluster, only {total_gpus}")

        nodes = []
        seen = set()
        for node in ray.nodes():
            if not node.get("Alive") or node.get("Resources", {}).get("GPU", 0) < 8:
                continue
            info = ray.get(
                _get_node_info.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node["NodeID"],
                        soft=False,
                    )
                ).remote()
            )
            if info["node_id"] in seen:
                continue
            seen.add(info["node_id"])
            nodes.append(info)
        if len(nodes) < 2:
            raise unittest.SkipTest(f"need >= 2 nodes with 8 GPUs, got {len(nodes)}")

        world_size = 16
        futures = [
            _deepep_worker.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=nodes[rank // 8]["node_id"],
                    soft=False,
                )
            ).remote(rank, world_size, nodes[0]["ip"], 12940)
            for rank in range(world_size)
        ]
        results = ray.get(futures)
        for res in results:
            self.assertEqual(res["combined_shape"], [6, 8])
            self.assertTrue(res["combined_finite"])
            self.assertTrue(res["tokens_grad_finite"])
            self.assertGreater(res["tokens_grad_norm"], 0.0)
            self.assertTrue(res["weights_grad_finite"])
            self.assertGreater(res["weights_grad_norm"], 0.0)
            self.assertEqual(len(res["tokens_per_expert"]), 2)
