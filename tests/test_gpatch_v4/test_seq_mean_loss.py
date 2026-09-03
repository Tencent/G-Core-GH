"""seq-mean-token-mean 归一化的核心不变性测试。

验证 masked_sum_per_seq 作为 seq-mean 分子的性质：
- 跨 micro-batch 切分不变性（先求和、最后除全局条数 → 与切分无关）；
- 死样本（sample_mask=0）不计入分子；
- 与 token-weighted mean 在序列长度不等时确有区别。

同时验证 legacy ``masked_sum_and_count_per_sample_or_token``（返回 3 对值）：
1. 无梯度 token-sum / token-count
2. 无梯度 sample-sum / sample-count
3. 有梯度 bwd-sum / bwd-count（由 calculate_per_token_loss 选择）

New-loss ``agg`` is ``[B, S]`` only and is covered lightly below.
"""
import torch

from gpatch_v4.training_backend.loss.utils import agg
from gpatch_v4.training_backend.loss_factory import (
    _thd_cu_seqlens_for_values,
    _thd_per_sample_sum_and_count_local,
    masked_mean_per_sample_or_token,
    masked_sum_and_count_per_sample_or_token,
)
from gpatch_v4.utils.training_utils import masked_mean, masked_sum, masked_sum_per_seq


def _simulate_te_thd_local_indices(cu_seqlens_padded, cp_size, cp_rank):
    """Python mirror of TE ``thd_partition_indices_kernel`` local layout."""
    cu = [int(x) for x in cu_seqlens_padded.tolist()]
    local_cu = [c // cp_size for c in cu]
    local_len = cu[-1] // cp_size
    indices = []
    for token_id in range(local_len):
        # binary search seq_id in local_cu
        seq_id = 0
        for i in range(len(local_cu) - 1):
            if local_cu[i] <= token_id < local_cu[i + 1]:
                seq_id = i
                break
        seq_len = local_cu[seq_id + 1] - local_cu[seq_id]
        index = token_id - local_cu[seq_id]
        offset = cp_rank if index < seq_len // 2 else (cp_size - 1) * 2 - cp_rank
        index = index + local_cu[seq_id] * cp_size + (seq_len // 2) * offset
        indices.append(index)
    return torch.tensor(indices, dtype=torch.long)


def _seq_mean(values, mask, sample_mask=None):
    """直接算 seq-mean-token-mean：sum_of_per_seq_means / 存活条数。"""
    s = masked_sum_per_seq(values, mask, sample_mask)
    n = sample_mask.sum() if sample_mask is not None else float(values.size(0))
    return s / n


class TestNewLossAggRejectsThd:
    def test_agg_rejects_cu_seqlens(self):
        values = torch.randn(2, 4)
        mask = torch.ones(2, 4)
        try:
            agg(values, mask, cu_seqlens_padded=torch.tensor([0, 4, 8]))
            raise AssertionError("expected assert")
        except AssertionError as exc:
            assert "response-padded" in str(exc)

    def test_agg_bshd_per_token(self):
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        bwd_sum, bwd_count = agg(values, mask, calculate_per_token_loss=True)
        assert torch.allclose(bwd_sum, torch.tensor(8.0))
        assert torch.allclose(bwd_count, torch.tensor(3.0))


def _unpack_bwd(result):
    """取第 3 对（有梯度的 bwd sum/count）。"""
    return result[4], result[5]


def _unpack_token_det(result):
    return result[0], result[1]


def _unpack_sample_det(result):
    return result[2], result[3]


class TestPartitionInvariance:
    def test_nan_mask_entries_are_excluded(self):
        values = torch.tensor([[2.0, 100.0, 4.0], [10.0, 20.0, 30.0]])
        mask = torch.tensor([[1.0, float("nan"), 1.0], [float("nan"), 0.0, 1.0]])

        result = masked_sum_per_seq(values, mask)

        # (2 + 4) / 2 + 30 / 1
        assert torch.allclose(result, torch.tensor(33.0))

    def test_sum_invariant_to_microbatch_split(self):
        # 把 8 条序列任意切成多个 mb，各 mb 的 S_b 求和应等于整批 S。
        torch.manual_seed(0)
        b, s = 8, 16
        values = torch.randn(b, s)
        mask = (torch.rand(b, s) > 0.3).float()
        mask[:, 0] = 1.0  # 保证每条至少一个有效 token

        full = masked_sum_per_seq(values, mask)
        for splits in ([4, 4], [2, 3, 3], [1, 1, 1, 1, 1, 1, 1, 1]):
            chunks_v = torch.split(values, splits)
            chunks_m = torch.split(mask, splits)
            partitioned = sum(
                masked_sum_per_seq(v, m) for v, m in zip(chunks_v, chunks_m)
            )
            assert torch.allclose(full, partitioned, atol=1e-5), splits

    def test_seq_mean_equals_mean_of_per_seq_means(self):
        torch.manual_seed(1)
        b, s = 5, 10
        values = torch.randn(b, s)
        mask = torch.ones(b, s)

        per_seq = (values * mask).sum(-1) / mask.sum(-1)
        assert torch.allclose(_seq_mean(values, mask), per_seq.mean(), atol=1e-6)


class TestDeadSample:
    def test_dead_rows_dropped(self):
        torch.manual_seed(2)
        b, s = 6, 12
        values = torch.randn(b, s)
        mask = torch.ones(b, s)
        sample_mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])

        with_mask = masked_sum_per_seq(values, mask, sample_mask)
        alive_only = masked_sum_per_seq(values[sample_mask.bool()], mask[sample_mask.bool()])
        assert torch.allclose(with_mask, alive_only, atol=1e-6)

    def test_seq_mean_over_alive_only(self):
        torch.manual_seed(3)
        b, s = 4, 8
        values = torch.randn(b, s)
        mask = torch.ones(b, s)
        sample_mask = torch.tensor([1.0, 1.0, 0.0, 0.0])

        per_seq = (values * mask).sum(-1) / mask.sum(-1)
        expected = per_seq[:2].mean()  # 只对前两条活样本平均
        assert torch.allclose(_seq_mean(values, mask, sample_mask), expected, atol=1e-6)


class TestDiffersFromTokenMean:
    def test_unequal_lengths_seq_mean_differs_from_token_mean(self):
        # 第 0 条长、第 1 条短：seq-mean 等权，token-mean 偏向长序列。
        values = torch.zeros(2, 10)
        values[0, :8] = 1.0   # 长序列 8 个 token，值 1
        values[1, :2] = 5.0   # 短序列 2 个 token，值 5
        mask = torch.zeros(2, 10)
        mask[0, :8] = 1.0
        mask[1, :2] = 1.0

        seq_mean = _seq_mean(values, mask)        # (1 + 5) / 2 = 3
        token_mean = masked_mean(values, mask)    # (8*1 + 2*5) / 10 = 1.8
        assert torch.allclose(seq_mean, torch.tensor(3.0), atol=1e-6)
        assert torch.allclose(token_mean, torch.tensor(1.8), atol=1e-6)
        assert not torch.allclose(seq_mean, token_mean)


class TestMaskedSumAndCountPerSampleOrToken:
    """覆盖 3 对返回值：detached token / detached sample / with-grad bwd。"""

    def test_per_token_matches_masked_sum(self):
        torch.manual_seed(10)
        b, s = 4, 8
        values = torch.randn(b, s, requires_grad=True)
        mask = (torch.rand(b, s) > 0.3).float()

        result = masked_sum_and_count_per_sample_or_token(
            values, mask, cu_seqlens_padded=None, calculate_per_token_loss=True
        )
        token_sum_det, token_count_det = _unpack_token_det(result)
        bwd_sum, bwd_count = _unpack_bwd(result)

        assert torch.allclose(bwd_sum, masked_sum(values, mask))
        assert torch.allclose(bwd_count, mask.sum())
        assert torch.allclose(token_sum_det, bwd_sum.detach())
        assert torch.allclose(token_count_det, bwd_count.detach())
        assert not token_sum_det.requires_grad
        assert bwd_sum.requires_grad

        fake_boundaries = torch.tensor([0, s, 2 * s, 3 * s, 4 * s])
        result2 = masked_sum_and_count_per_sample_or_token(
            values, mask, cu_seqlens_padded=fake_boundaries, calculate_per_token_loss=True
        )
        bwd_sum2, bwd_count2 = _unpack_bwd(result2)
        assert torch.allclose(bwd_sum, bwd_sum2)
        assert torch.allclose(bwd_count, bwd_count2)

    def test_per_token_partition_invariance(self):
        torch.manual_seed(11)
        b, s = 8, 16
        values = torch.randn(b, s)
        mask = (torch.rand(b, s) > 0.3).float()

        full_num, full_cnt = _unpack_bwd(
            masked_sum_and_count_per_sample_or_token(
                values, mask, calculate_per_token_loss=True
            )
        )
        for splits in ([4, 4], [2, 3, 3], [1] * 8):
            chunks_v = torch.split(values, splits)
            chunks_m = torch.split(mask, splits)
            nums, cnts = zip(
                *[
                    _unpack_bwd(
                        masked_sum_and_count_per_sample_or_token(
                            v, m, calculate_per_token_loss=True
                        )
                    ) for v, m in zip(chunks_v, chunks_m)
                ]
            )
            assert torch.allclose(full_num, sum(nums), atol=1e-5), splits
            assert torch.allclose(full_cnt, sum(cnts), atol=1e-5), splits

    def test_plain_2d_per_sample_matches_masked_sum_per_seq(self):
        torch.manual_seed(12)
        b, s = 6, 12
        values = torch.randn(b, s, requires_grad=True)
        mask = torch.ones(b, s)
        sample_mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])

        result = masked_sum_and_count_per_sample_or_token(
            values,
            mask,
            cu_seqlens_padded=None,
            calculate_per_token_loss=False,
            sample_mask=sample_mask,
        )
        sample_sum_det, sample_count_det = _unpack_sample_det(result)
        bwd_sum, bwd_count = _unpack_bwd(result)

        expected = masked_sum_per_seq(values, mask, sample_mask)
        assert torch.allclose(bwd_sum, expected)
        assert torch.allclose(bwd_count, sample_mask.sum())
        assert torch.allclose(sample_sum_det, bwd_sum.detach())
        assert torch.allclose(sample_count_det, bwd_count.detach())
        assert not sample_sum_det.requires_grad
        assert bwd_sum.requires_grad
        assert torch.allclose(bwd_sum / bwd_count, _seq_mean(values, mask, sample_mask))

    def test_both_aggregates_always_populated(self):
        # 无论 calculate_per_token_loss 取何值，前两对 detached 都应有正确数值。
        torch.manual_seed(15)
        b, s = 4, 6
        values = torch.randn(b, s)
        mask = torch.ones(b, s)
        sample_mask = torch.tensor([1.0, 1.0, 0.0, 1.0])

        for per_token in (True, False):
            result = masked_sum_and_count_per_sample_or_token(
                values,
                mask,
                calculate_per_token_loss=per_token,
                sample_mask=sample_mask,
            )
            token_sum, token_count = _unpack_token_det(result)
            sample_sum, sample_count = _unpack_sample_det(result)
            bwd_sum, bwd_count = _unpack_bwd(result)

            assert torch.allclose(token_sum, masked_sum(values, mask))
            assert torch.allclose(token_count, mask.sum())
            assert torch.allclose(sample_sum, masked_sum_per_seq(values, mask, sample_mask))
            assert torch.allclose(sample_count, sample_mask.sum())
            if per_token:
                assert torch.allclose(bwd_sum, token_sum)
                assert torch.allclose(bwd_count, token_count)
            else:
                assert torch.allclose(bwd_sum, sample_sum)
                assert torch.allclose(bwd_count, sample_count)

    def test_thd_per_sample_matches_legacy_mean(self):
        torch.manual_seed(13)
        seg_lens = [5, 3, 8, 2]
        total = sum(seg_lens)
        values = torch.randn(1, total)
        mask = (torch.rand(1, total) > 0.2).float()
        cu_seqlens_padded = torch.tensor([0] + list(torch.tensor(seg_lens).cumsum(0).tolist()))

        legacy_mean = masked_mean_per_sample_or_token(
            values, mask, cu_seqlens_padded, calculate_per_token_loss=False, local_cp_size=1
        )
        bwd_sum, bwd_count = _unpack_bwd(
            masked_sum_and_count_per_sample_or_token(
                values, mask, cu_seqlens_padded, calculate_per_token_loss=False, local_cp_size=1
            )
        )
        assert torch.allclose(bwd_count, torch.tensor(float(len(seg_lens))))
        assert torch.allclose(bwd_sum / bwd_count, legacy_mean, atol=1e-6)

    def test_thd_per_sample_counts_empty_segments_like_legacy_mean(self):
        values = torch.tensor([[1.0, 2.0, 10.0, 20.0, 30.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]])
        cu_seqlens_padded = torch.tensor([0, 2, 5])

        legacy_mean = masked_mean_per_sample_or_token(
            values, mask, cu_seqlens_padded, calculate_per_token_loss=False, local_cp_size=1
        )
        bwd_sum, bwd_count = _unpack_bwd(
            masked_sum_and_count_per_sample_or_token(
                values, mask, cu_seqlens_padded, calculate_per_token_loss=False, local_cp_size=1
            )
        )

        assert torch.allclose(bwd_sum, torch.tensor(1.5))
        assert torch.allclose(bwd_count, torch.tensor(2.0))
        assert torch.allclose(bwd_sum / bwd_count, legacy_mean, atol=1e-6)

    def test_thd_per_sample_partition_invariance(self):
        torch.manual_seed(14)
        seg_lens = [4, 6, 3, 5, 7]
        total = sum(seg_lens)
        values = torch.randn(1, total)
        mask = torch.ones(1, total)
        cu_seqlens_padded = torch.tensor([0] + list(torch.tensor(seg_lens).cumsum(0).tolist()))

        full_num, full_cnt = _unpack_bwd(
            masked_sum_and_count_per_sample_or_token(
                values, mask, cu_seqlens_padded, calculate_per_token_loss=False, local_cp_size=1
            )
        )

        split_at = cu_seqlens_padded[2].item()
        mb1_boundaries = cu_seqlens_padded[:3]
        mb2_boundaries = cu_seqlens_padded[2:] - split_at

        num1, cnt1 = _unpack_bwd(
            masked_sum_and_count_per_sample_or_token(
                values[:, :split_at], mask[:, :split_at], mb1_boundaries,
                calculate_per_token_loss=False, local_cp_size=1
            )
        )
        num2, cnt2 = _unpack_bwd(
            masked_sum_and_count_per_sample_or_token(
                values[:, split_at:], mask[:, split_at:], mb2_boundaries,
                calculate_per_token_loss=False, local_cp_size=1
            )
        )
        assert torch.allclose(full_num, num1 + num2, atol=1e-5)
        assert torch.allclose(full_cnt, cnt1 + cnt2, atol=1e-5)

    def test_thd_local_cu_matches_te_layout(self):
        # After CP shard, values length is global/cp; local boundaries are cu/cp.
        cu = torch.tensor([0, 8, 16, 32])  # each % (cp*2)==0 for cp=2
        cp = 2
        local_values_len = int(cu[-1].item()) // cp
        local_cu = _thd_cu_seqlens_for_values(cu, local_values_len, cp)
        assert local_cu.tolist() == [0, 4, 8, 16]

        # Full packed length still uses global cu.
        full_cu = _thd_cu_seqlens_for_values(cu, int(cu[-1].item()), 1)
        assert full_cu.tolist() == cu.tolist()

    def test_thd_cp_sharded_per_sample_matches_full(self):
        """Simulate dyn-cp local shards + CP sum-reduce of per-sample stats."""
        torch.manual_seed(20)
        # Lengths divisible by cp_size*2 (TE requirement).
        seg_lens = [8, 16, 8, 24]
        cp_size = 2
        total = sum(seg_lens)
        values = torch.randn(1, total)
        mask = (torch.rand(1, total) > 0.2).float()
        mask[0, 0] = 1.0
        cu = torch.tensor([0] + list(torch.tensor(seg_lens).cumsum(0).tolist()))

        ref_sum, ref_cnt = _unpack_bwd(
            masked_sum_and_count_per_sample_or_token(
                values, mask, cu, calculate_per_token_loss=False, local_cp_size=1
            )
        )
        ref_sample_sum, _ = _unpack_sample_det(
            masked_sum_and_count_per_sample_or_token(
                values, mask, cu, calculate_per_token_loss=True, local_cp_size=1
            )
        )

        # Per-rank local compact shards (TE layout); aggregate with local cu,
        # then sum-reduce sample sum/count across ranks (no real dist).
        values_flat = torch.where(mask > 0, values, 0.0).reshape(-1)
        mask_flat = mask.reshape(-1)
        reduced_sums = None
        reduced_counts = None
        for rank in range(cp_size):
            idx = _simulate_te_thd_local_indices(cu, cp_size, rank)
            local_v = values_flat.index_select(0, idx)
            local_m = mask_flat.index_select(0, idx)
            local_cu = _thd_cu_seqlens_for_values(cu, local_v.numel(), cp_size)
            sums, counts = _thd_per_sample_sum_and_count_local(local_v, local_m, local_cu)
            reduced_sums = sums if reduced_sums is None else reduced_sums + sums
            reduced_counts = counts if reduced_counts is None else reduced_counts + counts

        per_sample_mean = reduced_sums / reduced_counts.clamp(min=1)
        sharded_sample_sum = per_sample_mean.sum()
        sharded_sample_cnt = torch.tensor(float(len(seg_lens)))

        assert torch.allclose(sharded_sample_sum, ref_sum, atol=1e-5)
        assert torch.allclose(sharded_sample_cnt, ref_cnt, atol=1e-5)
        assert torch.allclose(sharded_sample_sum, ref_sample_sum, atol=1e-5)

        # Bug regression: slicing local shard with *global* cu must NOT match.
        buggy_sums = []
        buggy_counts = []
        idx0 = _simulate_te_thd_local_indices(cu, cp_size, 0)
        local_v0 = values_flat.index_select(0, idx0)
        local_m0 = mask_flat.index_select(0, idx0)
        for i in range(len(seg_lens)):
            s = cu[i].item()
            e = cu[i + 1].item()
            buggy_sums.append(local_v0[s:e].sum())
            buggy_counts.append(local_m0[s:e].sum())
        buggy_mean = (torch.stack(buggy_sums) / torch.stack(buggy_counts).clamp(min=1)).sum()
        assert not torch.allclose(buggy_mean, ref_sum, atol=1e-4)
