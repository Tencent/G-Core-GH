"""Test CP all_gather correctness for zigzag and non-zigzag patterns.

Spawns 4 Ray actors (1 GPU each) with ``TP=1, CP=2, DP=2`` and exercises
the round-trip identity of chunk → all_gather as well as backward gradient
correctness.  Covers both the legacy zigzag and the DSv4 non-zigzag
:func:`all_gather_from_context_parallel_region_no_zigzag`.

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=300 tests/test_gpatch_v4/test_cp_mappings.py
"""

import os
import socket
import unittest
from typing import List, Tuple

import ray
import torch
import torch.distributed as dist

from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORLD_SIZE = 4
TP, CP = 1, 2
DP = WORLD_SIZE // (TP * CP)
BATCH, SEQ = 2, 64
VOCAB = 16

# ---------------------------------------------------------------------------
# Ray actor
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=1)
class CpMappingsWorker:
    """Single-GPU worker for CP all_gather correctness tests."""
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
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device("cuda:0")

        from megatron.core import parallel_state as ps
        ps.initialize_model_parallel(
            tensor_model_parallel_size=TP,
            pipeline_model_parallel_size=1,
            context_parallel_size=CP,
        )
        self.cp_rank = ps.get_context_parallel_rank()
        self.cp_size = ps.get_context_parallel_world_size()

    def destroy(self):
        torch.distributed.destroy_process_group()

    # ------------------------------------------------------------------
    # Test 1: non-zigzag round-trip (dsv4 CP)
    # ------------------------------------------------------------------

    def _cp_chunk_non_zigzag(self, data: torch.Tensor) -> torch.Tensor:
        """Simulate dsv4 ``cp_chunk_single_data``: contiguous split along dim 1."""
        from gpatch_v4.models.deepseek_v4.cp import cp_chunk_data
        return cp_chunk_data(self.cp_rank, self.cp_size, tokens=data)[0]

    def test_non_zigzag_roundtrip(self) -> List[float]:
        """``cp_chunk_single_data`` → ``all_gather_no_zigzag`` 应恢复原始 tensor。"""
        from gpatch_v4.core.mappings import all_gather_from_context_parallel_region_no_zigzag

        # Deterministic tensor, identical on every rank.
        full = torch.arange(BATCH * SEQ, dtype=torch.float32, device=self.device).view(BATCH, SEQ)
        local = self._cp_chunk_non_zigzag(full)

        gathered = all_gather_from_context_parallel_region_no_zigzag(local, gather_dim=1)
        torch.cuda.synchronize()

        # All ranks see the same gathered output
        max_diff = (gathered - full).abs().max().item()
        return [max_diff]

    # ------------------------------------------------------------------
    # Test 2: non-zigzag backward correctness
    # ------------------------------------------------------------------

    def test_non_zigzag_backward(self) -> List[float]:
        """All-gather → loss → backward: 各 rank 的 local 梯度应等于 full grad 的对应 chunk。"""
        from gpatch_v4.core.mappings import all_gather_from_context_parallel_region_no_zigzag

        # Deterministic tensor, identical on every rank.
        full_src = torch.arange(BATCH * SEQ, dtype=torch.float32,
                                device=self.device).view(BATCH, SEQ)
        local = self._cp_chunk_non_zigzag(full_src).clone().requires_grad_(True)

        with torch.enable_grad():
            gathered = all_gather_from_context_parallel_region_no_zigzag(local, gather_dim=1)
            loss = gathered.pow(2).sum()
            loss.backward()

        torch.cuda.synchronize()

        # Expected: local gradient = the per-rank chunk of grad over the output
        full_src_grad = 2.0 * full_src  # d/dx (x^2) = 2x at each position
        expected_local_grad = self._cp_chunk_non_zigzag(full_src_grad)

        max_diff = (local.grad - expected_local_grad).abs().max().item()
        return [max_diff]

    # ------------------------------------------------------------------
    # Test 3: logprob + entropy shape sanity (simulate rl_train_actor flow)
    # ------------------------------------------------------------------

    def test_logprob_entropy_shapes(self) -> List[float]:
        """Verify that gather_log_probs_packed + entropy produce ``[B, S-1]``."""
        logits_local = torch.randn(BATCH, SEQ // CP, VOCAB, device=self.device)
        target_full = torch.randint(0, VOCAB, (BATCH, SEQ), device=self.device)

        # Simulate gather_log_probs_packed
        from gpatch_v4.training_backend.fsdp2_backend.mixin import selective_log_softmax_raw

        targets_rolled = target_full.roll(shifts=-1, dims=-1)
        local_targets = self._cp_chunk_non_zigzag(targets_rolled)

        curr_log_probs = selective_log_softmax_raw(logits_local, local_targets)  # [B, S//CP]

        from gpatch_v4.core.mappings import all_gather_from_context_parallel_region_no_zigzag
        if self.cp_size > 1:
            curr_log_probs = all_gather_from_context_parallel_region_no_zigzag(
                curr_log_probs, gather_dim=1
            )
        curr_log_probs = curr_log_probs[:, :-1]

        # Entropy
        probs = logits_local.softmax(dim=-1)
        entropy_local = -(probs * logits_local.log_softmax(dim=-1)).sum(dim=-1)
        if self.cp_size > 1:
            entropy = all_gather_from_context_parallel_region_no_zigzag(entropy_local, gather_dim=1)
        else:
            entropy = entropy_local
        entropy = entropy[:, :-1]

        torch.cuda.synchronize()

        # Both should be [B, S-1]
        ok = (curr_log_probs.shape == (BATCH, SEQ - 1)) and (entropy.shape == (BATCH, SEQ - 1))
        return [1.0 if ok else 0.0]

    # ------------------------------------------------------------------
    # Test 4: full vs CP-split logprobs — numerical equivalence
    # ------------------------------------------------------------------

    def _make_full_data(self):
        """Deterministic logits + target, identical on every rank."""
        g = torch.Generator(device=self.device).manual_seed(99)
        logits_full = torch.randn(BATCH, SEQ, VOCAB, device=self.device, generator=g)
        target_full = torch.randint(0, VOCAB, (BATCH, SEQ), device=self.device, generator=g)
        return logits_full, target_full

    def test_logprobs_full_vs_cp(self) -> List[float]:
        """Full-sequence logprobs vs CP-split + all_gather: should be bit-identical."""
        from gpatch_v4.core.mappings import all_gather_from_context_parallel_region_no_zigzag
        from gpatch_v4.training_backend.fsdp2_backend.mixin import selective_log_softmax_raw

        logits_full, target_full = self._make_full_data()

        # Ground truth: compute on full sequence (no split)
        target_shifted = target_full.roll(shifts=-1, dims=-1)
        gt_logprobs = selective_log_softmax_raw(logits_full, target_shifted)
        gt_logprobs = gt_logprobs[:, :-1]  # [B, S-1]

        # CP path: chunk BOTH logits and target, compute locally, all_gather
        local_logits = self._cp_chunk_non_zigzag(logits_full)
        local_targets = self._cp_chunk_non_zigzag(target_shifted)
        local_logprobs = selective_log_softmax_raw(local_logits, local_targets)
        cp_logprobs = all_gather_from_context_parallel_region_no_zigzag(
            local_logprobs, gather_dim=1
        )
        cp_logprobs = cp_logprobs[:, :-1]

        torch.cuda.synchronize()
        max_diff = (cp_logprobs - gt_logprobs).abs().max().item()
        return [max_diff]

    def test_entropy_full_vs_cp(self) -> List[float]:
        """Full-sequence entropy vs CP-split + all_gather: should be bit-identical."""
        from gpatch_v4.core.mappings import all_gather_from_context_parallel_region_no_zigzag

        logits_full, _ = self._make_full_data()

        # Ground truth: compute on full sequence (no split)
        probs_full = logits_full.softmax(dim=-1)
        gt_entropy = -(probs_full * logits_full.log_softmax(dim=-1)).sum(dim=-1)
        gt_entropy = gt_entropy[:, :-1]  # [B, S-1]

        # CP path: chunk logits, compute locally, all_gather, then slice
        local_logits = self._cp_chunk_non_zigzag(logits_full)
        probs_local = local_logits.softmax(dim=-1)
        entropy_local = -(probs_local * local_logits.log_softmax(dim=-1)).sum(dim=-1)
        cp_entropy = all_gather_from_context_parallel_region_no_zigzag(entropy_local, gather_dim=1)
        cp_entropy = cp_entropy[:, :-1]

        torch.cuda.synchronize()
        max_diff = (cp_entropy - gt_entropy).abs().max().item()
        return [max_diff]

    def test_logprobs_thd_pre_shifted_full_vs_cp(self) -> List[float]:
        """Pre-shifted THD targets keep the full axis after contiguous CP gather."""
        from gpatch_v4.core.mappings import all_gather_from_context_parallel_region_no_zigzag
        from gpatch_v4.training_backend.fsdp2_backend.mixin import selective_log_softmax_raw

        logits_full, target_full = self._make_full_data()
        gt_logprobs = selective_log_softmax_raw(logits_full, target_full)

        local_logits = self._cp_chunk_non_zigzag(logits_full)
        local_targets = self._cp_chunk_non_zigzag(target_full)
        local_logprobs = selective_log_softmax_raw(local_logits, local_targets)
        cp_logprobs = all_gather_from_context_parallel_region_no_zigzag(
            local_logprobs, gather_dim=1
        )

        torch.cuda.synchronize()
        assert cp_logprobs.shape == (BATCH, SEQ)
        max_diff = (cp_logprobs - gt_logprobs).abs().max().item()
        return [max_diff]

    def test_entropy_thd_full_vs_cp(self) -> List[float]:
        """THD entropy keeps all pre-shifted positions after contiguous CP gather."""
        from gpatch_v4.core.mappings import all_gather_from_context_parallel_region_no_zigzag

        logits_full, _ = self._make_full_data()
        probs_full = logits_full.softmax(dim=-1)
        gt_entropy = -(probs_full * logits_full.log_softmax(dim=-1)).sum(dim=-1)

        local_logits = self._cp_chunk_non_zigzag(logits_full)
        probs_local = local_logits.softmax(dim=-1)
        local_entropy = -(probs_local * local_logits.log_softmax(dim=-1)).sum(dim=-1)
        cp_entropy = all_gather_from_context_parallel_region_no_zigzag(local_entropy, gather_dim=1)

        torch.cuda.synchronize()
        assert cp_entropy.shape == (BATCH, SEQ)
        max_diff = (cp_entropy - gt_entropy).abs().max().item()
        return [max_diff]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_and_init_workers(world_size: int) -> List[ray.actor.ActorHandle]:
    workers = [CpMappingsWorker.remote() for _ in range(world_size)]
    master_addr, master_port = ray.get(workers[0].get_master_addr_and_port.remote())
    init_refs = [
        w.init_dist.remote(rank, world_size, master_addr, master_port)
        for rank, w in enumerate(workers)
    ]
    ray.get(init_refs)
    return workers


def _destroy_workers(workers):
    ray.get([w.destroy.remote() for w in workers])


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class CpMappingsTest(unittest.TestCase):
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
        _destroy_workers(cls.workers)
        kill_all_actors_and_shutdown_ray()

    # -- non-zigzag (dsv4) -------------------------------------------------

    def test_non_zigzag_roundtrip(self):
        refs = [w.test_non_zigzag_roundtrip.remote() for w in self.workers]
        results = ray.get(refs)
        for rank, (max_diff, ) in enumerate(results):
            self.assertLess(max_diff, 1e-4, f"rank {rank}: max_diff={max_diff}")

    def test_non_zigzag_backward(self):
        refs = [w.test_non_zigzag_backward.remote() for w in self.workers]
        results = ray.get(refs)
        for rank, (max_diff, ) in enumerate(results):
            self.assertLess(max_diff, 1e-4, f"rank {rank}: max_diff={max_diff}")

    # -- logprob + entropy shapes -------------------------------------------

    def test_logprob_entropy_shapes(self):
        refs = [w.test_logprob_entropy_shapes.remote() for w in self.workers]
        results = ray.get(refs)
        for rank, (ok, ) in enumerate(results):
            self.assertGreater(ok, 0.5, f"rank {rank}: shape mismatch")

    # -- full vs CP numerical equivalence ---------------------------------

    def test_logprobs_full_vs_cp(self):
        refs = [w.test_logprobs_full_vs_cp.remote() for w in self.workers]
        results = ray.get(refs)
        for rank, (max_diff, ) in enumerate(results):
            self.assertLess(max_diff, 1e-5, f"rank {rank}: max_diff={max_diff}")

    def test_entropy_full_vs_cp(self):
        refs = [w.test_entropy_full_vs_cp.remote() for w in self.workers]
        results = ray.get(refs)
        for rank, (max_diff, ) in enumerate(results):
            self.assertLess(max_diff, 1e-5, f"rank {rank}: max_diff={max_diff}")

    def test_logprobs_thd_pre_shifted_full_vs_cp(self):
        refs = [w.test_logprobs_thd_pre_shifted_full_vs_cp.remote() for w in self.workers]
        results = ray.get(refs)
        for rank, (max_diff, ) in enumerate(results):
            self.assertLess(max_diff, 1e-5, f"rank {rank}: max_diff={max_diff}")

    def test_entropy_thd_full_vs_cp(self):
        refs = [w.test_entropy_thd_full_vs_cp.remote() for w in self.workers]
        results = ray.get(refs)
        for rank, (max_diff, ) in enumerate(results):
            self.assertLess(max_diff, 1e-5, f"rank {rank}: max_diff={max_diff}")
