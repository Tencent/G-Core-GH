"""zigzag CP 下 opd_topk_logprobs_from_linear_ce 的多 rank 正确性测试。

起 2 个 Ray actor（各 1 GPU），``TP=1, CP=2``，用真实 Megatron parallel_state +
真实 linear_cross_entropy kernel，验证融合版在 zigzag CP（``reorder_target_for_cp``
分片 + ``all_gather_from_context_parallel_region`` 还原）下：

1. forward 与非融合版 ``from_parallel_logits_to_opd_topk_logprobs`` 及全序列参考一致；
2. backward 的 ``d/d hidden`` / ``d/d weight`` 与非融合版一致（CP all-gather 的反向 scatter 正确）。

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=300 tests/test_gpatch_v4/test_opd_topk_linear_ce_cp.py
"""

import os
import socket
import unittest
from types import SimpleNamespace
from typing import List, Tuple

import ray
import torch
import torch.nn.functional as F

from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

WORLD_SIZE = 2
TP, CP = 1, 2
BATCH, SEQ, HIDDEN, VOCAB, K = 2, 32, 128, 256, 8
SEED = 1234


@ray.remote(num_gpus=1)
class OpdTopkCpWorker:
    """Single-GPU worker: fused vs 非融合 vs 全序列参考，在 zigzag CP 下对拍。"""

    def get_master_addr_and_port(self) -> Tuple[str, int]:
        ip = socket.gethostbyname(socket.gethostname())
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return ip, s.getsockname()[1]

    def init_dist(self, rank: int, world_size: int, master_addr: str, master_port: int):
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        from megatron.core import parallel_state as ps
        ps.initialize_model_parallel(
            tensor_model_parallel_size=TP,
            pipeline_model_parallel_size=1,
            context_parallel_size=CP,
        )
        self.cp_rank = ps.get_context_parallel_rank()
        self.cp_size = ps.get_context_parallel_world_size()
        self.tp_group = ps.get_tensor_model_parallel_group()
        self.device = torch.device("cuda:0")

    def destroy(self):
        torch.distributed.destroy_process_group()

    # ------------------------------------------------------------------

    def _make_full_data(self):
        """Deterministic hidden/weight/ids，各 rank 完全一致。"""
        g = torch.Generator(device=self.device).manual_seed(SEED)
        hidden = torch.randn(BATCH, SEQ, HIDDEN, device=self.device, dtype=torch.bfloat16, generator=g)
        weight = torch.randn(VOCAB, HIDDEN, device=self.device, dtype=torch.bfloat16, generator=g)
        ids = torch.randint(0, VOCAB, (BATCH, SEQ, K), device=self.device, generator=g)
        return hidden, weight, ids

    def _zigzag_shard_bsh(self, full_bsh):
        """按 zigzag CP 切出本 rank 的 [B, local_s, H] shard（与函数内部 reorder 对齐）。"""
        from gpatch_v4.utils.training_utils import reorder_target_for_cp
        reordered = reorder_target_for_cp(full_bsh, seq_dim=1)
        local_s = full_bsh.shape[1] // self.cp_size
        return reordered[:, self.cp_rank * local_s:(self.cp_rank + 1) * local_s, :]

    def _fused(self, hidden_shard_seqfirst, weight, ids_full):
        from gpatch_v4.utils.training_utils import opd_topk_logprobs_from_linear_ce
        output_layer = SimpleNamespace(
            tp_group=self.tp_group, sequence_parallel=False, weight=None
        )
        lce_out = {
            "hidden_states": hidden_shard_seqfirst.contiguous(),
            "weight": weight,
            "output_layer": output_layer,
        }
        return opd_topk_logprobs_from_linear_ce(
            linear_ce_backend="split_n",
            linear_ce_output=lce_out,
            target_ids=ids_full,
            ignore_cp=False,
            temperature=1.0,
        )

    def _non_fused(self, hidden_shard_bsh, weight, ids_full):
        from gpatch_v4.utils.training_utils import from_parallel_logits_to_opd_topk_logprobs
        logits_shard = torch.einsum(
            'bsh,vh->bsv', hidden_shard_bsh.float(), weight.float()
        )  # [B, local_s, V]
        return from_parallel_logits_to_opd_topk_logprobs(
            vocab_parallel_logits=logits_shard,
            target_ids=ids_full,
            ignore_cp=False,
            temperature=1.0,
        )

    def test_forward(self) -> List[float]:
        """fused / 非融合 / 全序列参考三者一致（all_gather 后原始 seq 顺序）。"""
        hidden, weight, ids = self._make_full_data()
        shard_bsh = self._zigzag_shard_bsh(hidden)  # [B, local_s, H]

        fused = self._fused(shard_bsh.transpose(0, 1), weight, ids)  # [B, S, K]
        non_fused = self._non_fused(shard_bsh, weight, ids)  # [B, S, K]

        # 全序列参考（原始顺序）
        logits_full = torch.einsum('bsh,vh->bsv', hidden.float(), weight.float())
        ref = torch.gather(F.log_softmax(logits_full, dim=-1), dim=-1, index=ids)

        torch.cuda.synchronize()
        assert fused.shape == non_fused.shape == ref.shape == (BATCH, SEQ, K)
        return [
            (fused - non_fused).abs().max().item(),
            (fused - ref).abs().max().item(),
            (non_fused - ref).abs().max().item(),
        ]

    def test_backward(self) -> List[float]:
        """fused 与非融合的 d/d hidden、d/d weight 在 zigzag CP 下一致。"""
        hidden, weight, ids = self._make_full_data()
        shard_bsh = self._zigzag_shard_bsh(hidden)  # [B, local_s, H]

        # fused：leaf 为 seq-first shard
        hs_a = shard_bsh.transpose(0, 1).contiguous().detach().clone().requires_grad_(True)
        w_a = weight.detach().clone().requires_grad_(True)
        self._fused(hs_a, w_a, ids).sum().backward()

        # 非融合：leaf 为 [B, local_s, H]
        hs_b = shard_bsh.detach().clone().requires_grad_(True)
        w_b = weight.detach().clone().requires_grad_(True)
        self._non_fused(hs_b, w_b, ids).sum().backward()

        torch.cuda.synchronize()
        grad_h_a = hs_a.grad.transpose(0, 1).float()  # -> [B, local_s, H]

        # fused 的 d_hidden 从 bf16 kernel 出来，non_fused 走 fp32 einsum；用 Frobenius
        # 相对误差对 bf16 舍入稳健（真正的 CP scatter bug 会是 O(1) 相对误差）。
        def _rel(a, b):
            return (a - b).norm().item() / (b.norm().item() + 1e-8)

        grad_h_rel = _rel(grad_h_a, hs_b.grad.float())
        grad_w_rel = _rel(w_a.grad.float(), w_b.grad.float())
        finite = float(
            torch.isfinite(hs_a.grad).all() and torch.isfinite(w_a.grad).all()
        )
        return [grad_h_rel, grad_w_rel, finite]


def _create_and_init_workers(world_size: int) -> List[ray.actor.ActorHandle]:
    workers = [OpdTopkCpWorker.remote() for _ in range(world_size)]
    master_addr, master_port = ray.get(workers[0].get_master_addr_and_port.remote())
    ray.get([
        w.init_dist.remote(rank, world_size, master_addr, master_port)
        for rank, w in enumerate(workers)
    ])
    return workers


class OpdTopkCpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < WORLD_SIZE:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(f"Need >= {WORLD_SIZE} GPUs, only {total_gpus} available")
        cls.workers = _create_and_init_workers(WORLD_SIZE)

    @classmethod
    def tearDownClass(cls):
        ray.get([w.destroy.remote() for w in cls.workers])
        kill_all_actors_and_shutdown_ray()

    def test_forward(self):
        results = ray.get([w.test_forward.remote() for w in self.workers])
        for rank, (fvn, fvr, nvr) in enumerate(results):
            # fused 走 bf16 kernel，non_fused/ref 走 fp32，容忍 bf16 级误差。
            self.assertLess(fvn, 5e-2, f"rank {rank}: fused vs non_fused max_diff={fvn}")
            self.assertLess(fvr, 5e-2, f"rank {rank}: fused vs ref max_diff={fvr}")
            # non_fused 与全序列参考都是 fp32，唯一差异是 zigzag CP 分片/还原，应几乎 bit 一致。
            self.assertLess(nvr, 1e-4, f"rank {rank}: non_fused vs ref max_diff={nvr}")

    def test_backward(self):
        results = ray.get([w.test_backward.remote() for w in self.workers])
        for rank, (gh, gw, finite) in enumerate(results):
            self.assertGreater(finite, 0.5, f"rank {rank}: grad not finite")
            # fp32 grad：Frobenius 相对误差应到 ~1e-3 级别。
            self.assertLess(gh, 5e-3, f"rank {rank}: d/d hidden rel_diff={gh}")
            self.assertLess(gw, 5e-3, f"rank {rank}: d/d weight rel_diff={gw}")
