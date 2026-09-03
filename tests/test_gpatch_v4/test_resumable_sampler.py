#!/usr/bin/env python
"""
PYTHONPATH="$PWD:/root/Megatron-LM" python3 tests/test_utils/test_resumable_sampler.py

测试 ResumableDistributedSampler 跳过样本的正确性

测试逻辑：
1. 模拟从头开始训练，记录每一步的样本索引
2. 模拟从中间步数续训，记录续训后的样本索引
3. 验证续训后的样本索引与从头训练时对应步数后的索引完全一致
"""

import unittest

import pytest
import torch
import ray
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from gpatch_v4.utils.resumable_distributed_sampler import ResumableDistributedSampler


class DummyDataset(Dataset):
    """简单的测试数据集"""
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"idx": idx, "data": f"sample_{idx}"}


def collate_fn(batch):
    return {
        "indices": [item["idx"] for item in batch],
        "data": [item["data"] for item in batch],
    }


def run_training_simulation(
    dataset, batch_size, dp_rank, dp_size, seed, num_steps=None, resume_step=0
):
    """
    模拟训练过程，返回每一步的样本索引

    Args:
        dataset: 数据集
        batch_size: 批次大小
        dp_rank: 数据并行 rank
        dp_size: 数据并行 world size
        seed: 随机种子
        num_steps: 训练步数（None 表示训练完整 epoch）
        resume_step: 续训起始步数

    Returns:
        list: 每一步的样本索引列表
    """
    sampler = ResumableDistributedSampler(
        dataset,
        rank=dp_rank,
        num_replicas=dp_size,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )

    if resume_step > 0:
        sampler.set_start_index(resume_step, batch_size)

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=True,
    )

    all_indices = []
    step = 0
    for batch in dataloader:
        all_indices.append(batch["indices"])
        step += 1
        if num_steps is not None and step >= num_steps:
            break

    return all_indices


class ResumableSamplerTest(unittest.TestCase):
    def test_resume_correctness(self):
        """测试续训跳过样本的正确性"""
        print("=" * 60)
        print("测试 ResumableDistributedSampler 续训正确性")
        print("=" * 60)

        # 测试参数
        dataset_size = 100
        batch_size = 4
        dp_rank = 0
        dp_size = 2  # 模拟 2 个 DP rank
        seed = 42
        resume_step = 5  # 从第 5 步续训

        dataset = DummyDataset(size=dataset_size)

        # ========== 场景1: 从头训练，记录所有步骤的样本 ==========
        print("\n[场景1] 从头开始训练:")
        full_training_indices = run_training_simulation(
            dataset, batch_size, dp_rank, dp_size, seed, num_steps=None, resume_step=0
        )

        print(f"总共 {len(full_training_indices)} 步")
        for i, indices in enumerate(full_training_indices):
            print(f"  Step {i}: {indices}")

        # ========== 场景2: 从 resume_step 续训 ==========
        print(f"\n[场景2] 从 step {resume_step} 续训:")
        resumed_indices = run_training_simulation(
            dataset, batch_size, dp_rank, dp_size, seed, num_steps=None, resume_step=resume_step
        )

        print(f"续训 {len(resumed_indices)} 步")
        for i, indices in enumerate(resumed_indices):
            actual_step = resume_step + i
            print(f"  Step {actual_step} (续训后第 {i} 步): {indices}")

        # ========== 验证正确性 ===
        print("\n[验证] 对比续训后的样本与从头训练的样本:")

        expected_indices = full_training_indices[resume_step:]

        if len(resumed_indices) != len(expected_indices):
            print(f"❌ 步数不一致: 续训 {len(resumed_indices)} 步, 期望 {len(expected_indices)} 步")
            assert False

        all_match = True
        for i, (resumed, expected) in enumerate(zip(resumed_indices, expected_indices)):
            actual_step = resume_step + i
            if resumed == expected:
                print(f"  ✅ Step {actual_step}: {resumed} == {expected}")
            else:
                print(f"  ❌ Step {actual_step}: {resumed} != {expected}")
                all_match = False

        assert all_match, f"测试失败！续训后的样本与从头训练不完全一致"

    def test_cross_epoch_resume(self):
        """测试跨 epoch 续训的正确性"""
        print("\n" + "=" * 60)
        print("测试跨 Epoch 续训正确性")
        print("=" * 60)

        # 测试参数：小数据集，确保能跨 epoch
        dataset_size = 20
        batch_size = 4
        dp_rank = 0
        dp_size = 2  # 每个 rank 10 个样本，每 epoch 2 步
        seed = 42

        dataset = DummyDataset(size=dataset_size)

        # 计算每 epoch 步数
        samples_per_rank = dataset_size // dp_size  # 10
        steps_per_epoch = samples_per_rank // batch_size  # 2

        print(f"Dataset size: {dataset_size}, DP size: {dp_size}")
        print(f"Samples per rank: {samples_per_rank}, Steps per epoch: {steps_per_epoch}")

        # 模拟训练 2 个完整 epoch
        print("\n[场景1] 训练 2 个完整 epoch:")

        sampler = ResumableDistributedSampler(
            dataset, rank=dp_rank, num_replicas=dp_size, shuffle=True, seed=seed, drop_last=True
        )
        dataloader = DataLoader(
            dataset, sampler=sampler, collate_fn=collate_fn, batch_size=batch_size, drop_last=True
        )

        all_indices = []
        for epoch in range(2):
            sampler.set_epoch(epoch)
            sampler.start_index = 0
            print(f"\n  Epoch {epoch}:")
            for step_in_epoch, batch in enumerate(dataloader):
                global_step = epoch * steps_per_epoch + step_in_epoch
                all_indices.append(batch["indices"])
                print(
                    f"    Global Step {global_step} (Epoch {epoch}, Step {step_in_epoch}): {batch['indices']}"
                )

        # 测试从第 3 步续训（即第 2 个 epoch 的第 1 步）
        resume_step = 3
        print(f"\n[场景2] 从 global step {resume_step} 续训 (应该在 epoch 1, step 1):")

        resumed_indices = run_training_simulation(
            dataset, batch_size, dp_rank, dp_size, seed, num_steps=1, resume_step=resume_step
        )

        print(f"续训得到的样本: {resumed_indices[0]}")
        print(f"期望的样本: {all_indices[resume_step]}")

        assert resumed_indices[0] == all_indices[resume_step], f"跨 Epoch 续训测试失败！"

    def test_multi_rank(self):
        """测试多 rank 场景下的一致性"""
        print("\n" + "=" * 60)
        print("测试多 Rank 场景")
        print("=" * 60)

        dataset_size = 100
        batch_size = 4
        dp_size = 4
        seed = 42
        resume_step = 3

        dataset = DummyDataset(size=dataset_size)

        print(f"Dataset size: {dataset_size}, DP size: {dp_size}, Batch size: {batch_size}")
        print(f"Resume from step: {resume_step}")

        for rank in range(dp_size):
            print(f"\n--- Rank {rank} ---")

            full_indices = run_training_simulation(
                dataset, batch_size, rank, dp_size, seed, num_steps=5, resume_step=0
            )

            resumed_indices = run_training_simulation(
                dataset, batch_size, rank, dp_size, seed, num_steps=2, resume_step=resume_step
            )

            expected = full_indices[resume_step:resume_step + 2]
            match = resumed_indices == expected
            assert match, f"Rank {rank} 续训结果不一致"

    @staticmethod
    def _rebuild_global_shuffled(dataset, dp_size, seed, epoch):
        """从 sampler 各 rank 的输出重建全局 shuffled 序列。

        DistributedSampler 的分片逻辑: indices[rank::num_replicas]
        所以 global[rank + j * dp_size] = rank_indices[j]
        """
        all_rank_indices = {}
        for rank in range(dp_size):
            sampler = ResumableDistributedSampler(
                dataset, rank=rank, num_replicas=dp_size,
                shuffle=True, seed=seed, drop_last=True,
            )
            sampler.set_epoch(epoch)
            all_rank_indices[rank] = list(sampler)

        per_rank_len = len(all_rank_indices[0])
        total_size = dp_size * per_rank_len
        global_shuffled = [None] * total_size
        for rank in range(dp_size):
            for j, val in enumerate(all_rank_indices[rank]):
                global_shuffled[rank + j * dp_size] = val
        return global_shuffled

    def test_cross_dp_size_shuffle_consistency(self):
        """验证不同 dp_size 下 sampler 产出的全局 shuffled 序列完全一致。

        这是跨 dp_size 续训正确性的前提：shuffle 只依赖 seed+epoch
        和 len(dataset)，与 num_replicas 无关。
        """
        print("\n" + "=" * 60)
        print("测试: 不同 dp_size 下 sampler 全局 shuffle 序列一致性")
        print("=" * 60)

        dataset_size = 128
        seed = 42
        dataset = DummyDataset(size=dataset_size)

        dp_sizes = [2, 4, 8, 16]
        print(f"Dataset size: {dataset_size}, seed: {seed}")
        print(f"dp_sizes: {dp_sizes}")

        for epoch in range(3):
            results = {}
            for dp in dp_sizes:
                if dataset_size % dp != 0:
                    continue
                results[dp] = self._rebuild_global_shuffled(
                    dataset, dp, seed, epoch,
                )

            ref_dp = dp_sizes[0]
            all_match = True
            for dp in dp_sizes[1:]:
                if dp not in results:
                    continue
                match = results[dp] == results[ref_dp]
                if not match:
                    all_match = False
                assert match, (
                    f"epoch={epoch}: dp_size={ref_dp} 与 dp_size={dp} "
                    f"的全局 shuffled 序列不一致"
                )
            print(f"  epoch={epoch}: dp_sizes {list(results.keys())} "
                  f"全局序列一致 ✓  (前8个: {results[ref_dp][:8]}...)")

    def test_cross_dp_size_resume(self):
        """dp_size 变化时（如 8->4），验证跳过的全局样本集合与原始消费集合完全一致。

        验证方法：直接从 sampler 取各 rank 的 indices，重建全局 shuffled
        序列，然后比较前 K 个元素（K = 全局消费样本数）。
        """
        print("\n" + "=" * 60)
        print("测试: 跨 dp_size 续训 sampler indices 跳过一致性")
        print("=" * 60)

        dataset_size = 128
        rollout_mbs = 1
        seed = 42
        dataset = DummyDataset(size=dataset_size)

        test_cases = [
            # (orig_dp, new_dp, rollout_gbs, train_steps)
            (8, 4, 32, 3),
            (4, 8, 32, 2),
            (8, 2, 32, 2),
            (4, 2, 16, 3),
            (16, 4, 64, 1),
        ]

        print(f"Dataset size: {dataset_size}, seed: {seed}")

        for orig_dp, new_dp, rollout_gbs, train_steps in test_cases:
            orig_gas = rollout_gbs // (orig_dp * rollout_mbs)
            new_gas = rollout_gbs // (new_dp * rollout_mbs)
            global_consumed = train_steps * rollout_gbs

            print(f"\n  --- dp {orig_dp} -> {new_dp}, gbs={rollout_gbs}, "
                  f"{train_steps} steps (global consumed={global_consumed}) ---")

            # 从 sampler 重建全局 shuffled 序列
            shuffled_orig = self._rebuild_global_shuffled(
                dataset, orig_dp, seed, epoch=0,
            )
            shuffled_new = self._rebuild_global_shuffled(
                dataset, new_dp, seed, epoch=0,
            )

            assert shuffled_orig == shuffled_new, (
                f"dp {orig_dp}->{new_dp}: 全局 shuffled 序列不一致"
            )
            print(f"    shuffled 序列一致 ✓")

            # 原始训练: 每 rank 消费前 (train_steps * gas) 个
            orig_consumed = set()
            for rank in range(orig_dp):
                sampler = ResumableDistributedSampler(
                    dataset, rank=rank, num_replicas=orig_dp,
                    shuffle=True, seed=seed, drop_last=True,
                )
                sampler.set_epoch(0)
                rank_indices = list(sampler)
                consumed_per_rank = train_steps * orig_gas
                orig_consumed.update(rank_indices[:consumed_per_rank])

            # 续训: 每 rank 跳过前 (train_steps * new_gas) 个
            new_skipped = set()
            for rank in range(new_dp):
                sampler = ResumableDistributedSampler(
                    dataset, rank=rank, num_replicas=new_dp,
                    shuffle=True, seed=seed, drop_last=True,
                )
                sampler.set_epoch(0)
                rank_indices = list(sampler)
                skip_per_rank = train_steps * new_gas
                new_skipped.update(rank_indices[:skip_per_rank])

            # 参考值: shuffled 数组的前 K 个
            expected = set(shuffled_orig[:global_consumed])

            print(f"    orig consumed (dp={orig_dp}, {orig_gas} gas, "
                  f"{train_steps * orig_gas}/rank): {len(orig_consumed)} 样本")
            print(f"    new  skipped  (dp={new_dp}, {new_gas} gas, "
                  f"{train_steps * new_gas}/rank): {len(new_skipped)} 样本")
            print(f"    shuffled[0:{global_consumed}]: {len(expected)} 样本")

            assert orig_consumed == expected, (
                f"dp {orig_dp}: consumed != shuffled[0:{global_consumed}]"
            )
            assert new_skipped == expected, (
                f"dp {new_dp}: skipped != shuffled[0:{global_consumed}]"
            )
            assert orig_consumed == new_skipped, (
                f"dp {orig_dp}->{new_dp}: consumed != skipped\n"
                f"  extra: {new_skipped - orig_consumed}\n"
                f"  miss:  {orig_consumed - new_skipped}"
            )
            print(f"    三者一致 ✓  (consumed == skipped == shuffled[0:K])")

    @staticmethod
    def _simulate_resume_per_rank(
        dataset, dp_size, rollout_gbs, rollout_mbs, seed, resume_step,
    ):
        """模拟 simple_dataset.py 的 resume 逻辑，按 rank 返回每个 rank
        从 DataLoader 迭代出的有序样本列表。

        Returns
        -------
        dict[int, list[list[int]]]
            {rank: [[batch0_indices], [batch1_indices], ...]}
        """
        rollout_gas = rollout_gbs // (dp_size * rollout_mbs)
        consumed_batches = resume_step * rollout_gas

        per_rank_batches = {}
        for rank in range(dp_size):
            sampler = ResumableDistributedSampler(
                dataset, rank=rank, num_replicas=dp_size,
                shuffle=True, seed=seed, drop_last=True,
            )
            sampler.set_start_index(consumed_batches, rollout_mbs)

            dataloader = DataLoader(
                dataset, sampler=sampler, collate_fn=collate_fn,
                batch_size=rollout_mbs, drop_last=True,
            )
            per_rank_batches[rank] = [batch["indices"] for batch in dataloader]

        return per_rank_batches

    @staticmethod
    def _collect_global_set(per_rank_batches):
        """从 per-rank 有序列表中收集全局样本集合。"""
        result = set()
        for batches in per_rank_batches.values():
            for batch in batches:
                result.update(batch)
        return result

    def test_cross_dp_size_resume_with_dataloader(self):
        """端到端验证: 完整模拟 simple_dataset.py 的 resume 路径。

        验证三个层面：
        1. per-rank 顺序: 同 dp_size 下 resume 后每个 rank 的 DataLoader
           迭代顺序，与从头跑到同一位置后剩余的顺序完全一致
        2. 全局集合: 不同 dp_size resume 后的全局剩余样本集合一致
        3. 覆盖完整: consumed ∪ remaining = 整个 epoch 的有效样本
        """
        print("\n" + "=" * 60)
        print("测试: 端到端 DataLoader 跨 dp_size 续训 (模拟 simple_dataset.py)")
        print("=" * 60)

        dataset_size = 64
        seed = 42

        test_cases = [
            # (orig_dp, new_dp, rollout_gbs, rollout_mbs, resume_step)
            (8, 4, 16, 1, 2),
            (4, 8, 16, 1, 2),
            (8, 2, 32, 1, 1),
            (4, 2, 8, 1, 3),
            (8, 4, 32, 2, 1),   # rollout_mbs=2
            (16, 4, 16, 1, 1),
            (16, 8, 32, 1, 1),
        ]

        dataset = DummyDataset(size=dataset_size)
        print(f"Dataset size: {dataset_size}, seed: {seed}")

        for orig_dp, new_dp, rollout_gbs, rollout_mbs, resume_step in test_cases:
            global_consumed = resume_step * rollout_gbs
            assert global_consumed < dataset_size, (
                f"测试前提: 全局消费 {global_consumed} 必须 < "
                f"dataset_size {dataset_size}（否则需要跨 epoch 测试）"
            )

            print(f"\n  --- dp {orig_dp} -> {new_dp}, gbs={rollout_gbs}, "
                  f"mbs={rollout_mbs}, resume_step={resume_step} "
                  f"(global consumed={global_consumed}) ---")

            orig_gas = rollout_gbs // (orig_dp * rollout_mbs)
            consumed_steps = resume_step * orig_gas

            # ====== 验证 1: 同 dp_size resume 的 per-rank 顺序正确性 ======
            # 从头跑，按 rank 分别取 consumed / remaining 的有序列表
            orig_remaining_per_rank = {}
            orig_consumed_set = set()
            orig_remaining_set = set()

            for rank in range(orig_dp):
                batches = run_training_simulation(
                    dataset, rollout_mbs, rank, orig_dp, seed,
                    num_steps=None, resume_step=0,
                )
                for batch in batches[:consumed_steps]:
                    orig_consumed_set.update(batch)
                remaining_batches = batches[consumed_steps:]
                orig_remaining_per_rank[rank] = remaining_batches
                for batch in remaining_batches:
                    orig_remaining_set.update(batch)

            # 同 dp_size resume，验证 per-rank 顺序完全一致
            orig_resume_per_rank = self._simulate_resume_per_rank(
                dataset, orig_dp, rollout_gbs, rollout_mbs, seed, resume_step,
            )

            print(f"    [验证1] 同 dp_size={orig_dp} resume per-rank 顺序:")
            for rank in range(orig_dp):
                expected = orig_remaining_per_rank[rank]
                actual = orig_resume_per_rank[rank]
                assert actual == expected, (
                    f"dp={orig_dp} rank={rank}: resume 后顺序不一致\n"
                    f"  expected: {expected}\n"
                    f"  actual:   {actual}"
                )
            print(f"      所有 {orig_dp} 个 rank 的迭代顺序完全一致 ✓")

            # ====== 验证 2: new_dp per-rank 跳过正确性 ======
            # 对 new_dp 的每个 rank：从头跑取全部样本，按 skip 切分，
            # 验证 resume 后的输出 == 跳过前 C 个后的剩余部分（list 精确匹配）
            new_resume_per_rank = self._simulate_resume_per_rank(
                dataset, new_dp, rollout_gbs, rollout_mbs, seed, resume_step,
            )

            new_gas = rollout_gbs // (new_dp * rollout_mbs)
            new_skip_per_rank = resume_step * new_gas

            print(f"    [验证2] new dp_size={new_dp} per-rank 跳过正确性 "
                  f"(skip {new_skip_per_rank}/rank):")
            for rank in range(new_dp):
                # 从头跑，取该 rank 的全部 batch
                full_batches = run_training_simulation(
                    dataset, rollout_mbs, rank, new_dp, seed,
                    num_steps=None, resume_step=0,
                )
                expected_remaining = full_batches[new_skip_per_rank:]
                actual_remaining = new_resume_per_rank[rank]

                assert actual_remaining == expected_remaining, (
                    f"new dp={new_dp} rank={rank}: resume 后顺序不一致\n"
                    f"  expected ({len(expected_remaining)} batches): "
                    f"{expected_remaining[:3]}...\n"
                    f"  actual   ({len(actual_remaining)} batches): "
                    f"{actual_remaining[:3]}..."
                )
            print(f"      所有 {new_dp} 个 rank 的迭代顺序完全一致 ✓")

            # ====== 验证 3: 跨 dp_size resume 全局迭代顺序正确性 ======
            # resume 后每个全局 step 的 batch
            # 应该对应 shuffled[K + step*dp_size : K + (step+1)*dp_size]

            # 重建全局 shuffled 序列
            global_shuffled = self._rebuild_global_shuffled(
                dataset, new_dp, seed, epoch=0,
            )

            # 组装全局 batch 顺序：step j 的 batch = {rank_i 的第 j 个 batch}
            # 由于 mbs 可能 > 1，每个 rank 每步产出 mbs 个样本
            num_remaining_steps = len(new_resume_per_rank[0])
            print(f"    [验证3] 跨 dp_size 全局迭代顺序 (dp={new_dp}):")
            print(f"      global_consumed={global_consumed}, "
                  f"remaining_steps/rank={num_remaining_steps}")

            order_correct = True
            for step_j in range(num_remaining_steps):
                # 实际：各 rank 第 j 步的样本拼接
                actual_batch = []
                for rank in range(new_dp):
                    actual_batch.extend(new_resume_per_rank[rank][step_j])

                # 期望：shuffled 数组中对应位置
                # rank i 跳过 skip_per_rank 个后第 j 个 =
                #   shuffled[i + (skip_per_rank + j) * dp_size]
                # 全局 batch = shuffled[K + step_j*dp_size*mbs :
                #                       K + (step_j+1)*dp_size*mbs]
                start = global_consumed + step_j * new_dp * rollout_mbs
                end = start + new_dp * rollout_mbs
                expected_batch = global_shuffled[start:end]

                # actual_batch 的顺序是 [rank0_samples, rank1_samples, ...]
                # expected 的顺序是 shuffled 中交错排列:
                #   [rank0_sample, rank1_sample, ..., rankN_sample] (per mbs)
                # 它们应该包含相同元素（顺序就是交错顺序）
                assert sorted(actual_batch) == sorted(expected_batch), (
                    f"dp={new_dp} step {step_j}: batch 内容不匹配\n"
                    f"  actual:   {actual_batch}\n"
                    f"  expected: {expected_batch}"
                )

                # 更强的验证：逐 rank 验证顺序
                for rank in range(new_dp):
                    rank_samples = new_resume_per_rank[rank][step_j]
                    for k, sample in enumerate(rank_samples):
                        pos = global_consumed + rank + (step_j * rollout_mbs + k) * new_dp
                        if sample != global_shuffled[pos]:
                            order_correct = False
                            print(f"      step {step_j} rank {rank}[{k}]: "
                                  f"got {sample}, expected shuffled[{pos}]="
                                  f"{global_shuffled[pos]}")

            assert order_correct, (
                f"dp={new_dp}: resume 后 per-rank 迭代顺序与 shuffled 不一致"
            )
            preview_steps = min(3, num_remaining_steps)
            for step_j in range(preview_steps):
                batch = []
                for rank in range(new_dp):
                    batch.extend(new_resume_per_rank[rank][step_j])
                print(f"      step {step_j}: {batch}")
            if num_remaining_steps > preview_steps:
                print(f"      ... ({num_remaining_steps - preview_steps} more steps)")
            print(f"      全局迭代顺序正确 ✓  "
                  f"(逐 rank 逐 step 对齐 shuffled[{global_consumed}:])")

            # ====== 验证 4: consumed ∪ remaining 覆盖完整 ======
            new_remaining_set = self._collect_global_set(new_resume_per_rank)
            actual_total = orig_consumed_set | orig_remaining_set
            assert actual_total == (orig_consumed_set | new_remaining_set), (
                f"dp {orig_dp}->{new_dp}: consumed ∪ remaining 不一致"
            )
            assert len(orig_consumed_set & new_remaining_set) == 0, (
                f"dp {orig_dp}->{new_dp}: consumed 与 new remaining 有重叠: "
                f"{orig_consumed_set & new_remaining_set}"
            )
            print(f"    [验证4] consumed({len(orig_consumed_set)}) ∪ "
                  f"remaining({len(new_remaining_set)}) = "
                  f"{len(actual_total)} 样本, 无重叠 ✓")


class TestAlignSamplerNumSamples(unittest.TestCase):
    """align_sampler_num_samples 截断后必须能安全 __iter__（复现 itao resume 挂）。"""

    def test_align_drop_last_false_then_iter(self):
        from gpatch_v4.utils.training_utils import align_sampler_num_samples

        # 对齐 03 日志量级：N=100310, dp=2, step_per_epoch=3134, gas=16, mbs=1
        dataset = DummyDataset(size=100310)
        sampler = ResumableDistributedSampler(
            dataset,
            rank=0,
            num_replicas=2,
            shuffle=False,
            drop_last=False,
        )
        assert sampler.num_samples == 50155
        assert sampler.total_size == 100310

        align_sampler_num_samples(sampler, train_step_per_epoch=3134, mbs=1, gas=16)
        assert sampler.num_samples == 50144
        assert sampler.total_size == 100288
        assert sampler.drop_last is True

        indices = list(sampler)
        assert len(indices) == sampler.num_samples
