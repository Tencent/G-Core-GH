"""
Tests for reward redistribution via LCS alignment.

Run:
    PYTHONPATH="$PWD" python -m pytest tests/test_gpatch_v4/test_reward_redist.py -v
"""

import asyncio
import os.path
import random
import time
import unittest

import torch

from gpatch_v4.utils.reward_redist import (
    AlignedChunk,
    lcs_match,
    naive_dp_lcs_match,
    redistribute_rewards,
    redistribute_rewards_batch,
    redistribute_rewards_batch_mp,
)


def _blocks_to_indices(blocks: list) -> tuple:
    """Extract flat (oi, ri) index lists from AlignedBlock list (match blocks only)."""
    oi, ri = [], []
    for blk in blocks:
        if blk.match:
            for k in range(blk.x_end - blk.x_start):
                oi.append(blk.x_start + k)
                ri.append(blk.y_start + k)
    return oi, ri


def _assert_valid_blocks(test: unittest.TestCase, x, y, blocks):
    """Validate that blocks form a valid, complete alignment of (x, y).

    Checks:
      1. Blocks cover [0, len(x)) and [0, len(y)) completely and without gaps/overlaps.
      2. For match blocks: x_end - x_start == y_end - y_start > 0 and values match.
      3. For mismatch blocks: at least one span is non-empty.
      4. Adjacent blocks do not have the same match flag (no redundant splits).
    """
    x_pos, y_pos = 0, 0
    for i, blk in enumerate(blocks):
        test.assertEqual(blk.x_start, x_pos, f"Block {i}: x gap/overlap at {x_pos}")
        test.assertEqual(blk.y_start, y_pos, f"Block {i}: y gap/overlap at {y_pos}")
        test.assertGreaterEqual(blk.x_end, blk.x_start, f"Block {i}: x_end < x_start")
        test.assertGreaterEqual(blk.y_end, blk.y_start, f"Block {i}: y_end < y_start")
        if blk.match:
            x_len = blk.x_end - blk.x_start
            y_len = blk.y_end - blk.y_start
            test.assertEqual(x_len, y_len, f"Block {i}: match block has unequal spans")
            test.assertGreater(x_len, 0, f"Block {i}: empty match block")
            for k in range(x_len):
                test.assertEqual(
                    x[blk.x_start + k], y[blk.y_start + k],
                    f"Block {i}: value mismatch at offset {k}"
                )
        else:
            test.assertTrue(
                blk.x_end > blk.x_start or blk.y_end > blk.y_start,
                f"Block {i}: empty mismatch block"
            )
        if i > 0:
            test.assertNotEqual(
                blk.match, blocks[i - 1].match, f"Block {i}: adjacent blocks have same match flag"
            )
        x_pos = blk.x_end
        y_pos = blk.y_end
    test.assertEqual(x_pos, len(x), f"Blocks don't cover full x: ended at {x_pos}")
    test.assertEqual(y_pos, len(y), f"Blocks don't cover full y: ended at {y_pos}")


def _assert_valid_lcs(test: unittest.TestCase, x, y, blocks):
    """Validate that blocks form a valid LCS alignment of (x, y)."""
    _assert_valid_blocks(test, x, y, blocks)


def _lcs_len_from_blocks(blocks) -> int:
    """Sum of match block lengths."""
    return sum(blk.x_end - blk.x_start for blk in blocks if blk.match)


def _assert_lcs_optimal(test: unittest.TestCase, x, y, blocks):
    """Validate that blocks produce an optimal-length LCS.

    Uses naive DP to compute the true LCS length, then checks that Myers
    produces the same length. Does NOT require index-level equality (since
    LCS may not be unique).
    """
    _assert_valid_lcs(test, x, y, blocks)
    dp_blocks = naive_dp_lcs_match(x, y)
    dp_lcs_len = _lcs_len_from_blocks(dp_blocks)
    myers_lcs_len = _lcs_len_from_blocks(blocks)
    test.assertEqual(
        myers_lcs_len, dp_lcs_len, f"LCS length mismatch: myers={myers_lcs_len}, dp={dp_lcs_len}"
    )


class TestLcsMatch(unittest.TestCase):

    # ------------------------------------------------------------------ #
    # Basic / deterministic cases (LCS is unique → exact index match)     #
    # ------------------------------------------------------------------ #

    def test_identical(self):
        orig = [100, 101, 102]
        reenc = [100, 101, 102]
        blocks = lcs_match(orig, reenc)
        oi, ri = _blocks_to_indices(blocks)
        self.assertEqual(oi, [0, 1, 2])
        self.assertEqual(ri, [0, 1, 2])
        _assert_lcs_optimal(self, orig, reenc, blocks)

    def test_empty(self):
        blocks = lcs_match([], [])
        self.assertEqual(blocks, [])

    def test_one_empty_x(self):
        blocks = lcs_match([1, 2], [])
        oi, ri = _blocks_to_indices(blocks)
        self.assertEqual(oi, [])
        self.assertEqual(ri, [])
        _assert_valid_blocks(self, [1, 2], [], blocks)

    def test_one_empty_y(self):
        blocks = lcs_match([], [1, 2])
        oi, ri = _blocks_to_indices(blocks)
        self.assertEqual(oi, [])
        self.assertEqual(ri, [])
        _assert_valid_blocks(self, [], [1, 2], blocks)

    def test_single_element_match(self):
        blocks = lcs_match([42], [42])
        oi, ri = _blocks_to_indices(blocks)
        self.assertEqual(oi, [0])
        self.assertEqual(ri, [0])

    def test_single_element_no_match(self):
        blocks = lcs_match([1], [2])
        oi, ri = _blocks_to_indices(blocks)
        self.assertEqual(oi, [])
        self.assertEqual(ri, [])

    def test_split_middle(self):
        """orig [100,101,104,105,106] vs reenc [100,101,300,106]
        LCS = [100, 101, 106] at orig[0,1,4] and reenc[0,1,3]."""
        orig = [100, 101, 104, 105, 106]
        reenc = [100, 101, 300, 106]
        blocks = lcs_match(orig, reenc)
        oi, ri = _blocks_to_indices(blocks)
        self.assertEqual(oi, [0, 1, 4])
        self.assertEqual(ri, [0, 1, 3])
        _assert_lcs_optimal(self, orig, reenc, blocks)

    def test_merge(self):
        """orig [100,101,300,106] vs reenc [100,101,104,105,106]
        LCS = [100, 101, 106]."""
        orig = [100, 101, 300, 106]
        reenc = [100, 101, 104, 105, 106]
        blocks = lcs_match(orig, reenc)
        oi, ri = _blocks_to_indices(blocks)
        self.assertEqual(oi, [0, 1, 3])
        self.assertEqual(ri, [0, 1, 4])
        _assert_lcs_optimal(self, orig, reenc, blocks)

    def test_no_common(self):
        blocks = lcs_match([1, 2, 3], [4, 5, 6])
        oi, ri = _blocks_to_indices(blocks)
        self.assertEqual(oi, [])
        self.assertEqual(ri, [])

    def test_single_match(self):
        blocks = lcs_match([1, 2, 3], [4, 2, 5])
        oi, ri = _blocks_to_indices(blocks)
        self.assertEqual(oi, [1])
        self.assertEqual(ri, [1])

    def test_multiple_diff_regions(self):
        """Two separate diff regions with matches between them."""
        orig = [1, 10, 11, 2, 20, 21, 3]
        reenc = [1, 99, 2, 88, 3]
        blocks = lcs_match(orig, reenc)
        oi, ri = _blocks_to_indices(blocks)
        self.assertEqual(oi, [0, 3, 6])
        self.assertEqual(ri, [0, 2, 4])
        _assert_lcs_optimal(self, orig, reenc, blocks)

    # ------------------------------------------------------------------ #
    # Edge cases: prefix / suffix / boundary                              #
    # ------------------------------------------------------------------ #

    def test_prefix_match_only(self):
        """Common prefix, then completely different tails."""
        orig = [1, 2, 3, 10, 11]
        reenc = [1, 2, 3, 20, 21, 22]
        blocks = lcs_match(orig, reenc)
        oi, ri = _blocks_to_indices(blocks)
        self.assertEqual(oi, [0, 1, 2])
        self.assertEqual(ri, [0, 1, 2])
        _assert_lcs_optimal(self, orig, reenc, blocks)

    def test_suffix_match_only(self):
        """Completely different heads, then common suffix."""
        orig = [10, 11, 1, 2, 3]
        reenc = [20, 21, 1, 2, 3]
        blocks = lcs_match(orig, reenc)
        _assert_lcs_optimal(self, orig, reenc, blocks)
        self.assertEqual(_lcs_len_from_blocks(blocks), 3)  # LCS = [1, 2, 3]

    def test_diff_only_in_middle(self):
        """Common prefix + suffix, diff only in the middle."""
        orig = [1, 2, 10, 11, 3, 4]
        reenc = [1, 2, 20, 3, 4]
        blocks = lcs_match(orig, reenc)
        _assert_lcs_optimal(self, orig, reenc, blocks)
        # LCS must include the prefix [1,2] and suffix [3,4]
        self.assertEqual(_lcs_len_from_blocks(blocks), 4)

    def test_x_is_subsequence_of_y(self):
        """x is entirely contained as a subsequence of y."""
        orig = [1, 2, 3]
        reenc = [9, 1, 8, 2, 7, 3, 6]
        blocks = lcs_match(orig, reenc)
        _assert_lcs_optimal(self, orig, reenc, blocks)
        self.assertEqual(_lcs_len_from_blocks(blocks), 3)  # Full match

    def test_y_is_subsequence_of_x(self):
        """y is entirely contained as a subsequence of x."""
        orig = [9, 1, 8, 2, 7, 3, 6]
        reenc = [1, 2, 3]
        blocks = lcs_match(orig, reenc)
        _assert_lcs_optimal(self, orig, reenc, blocks)
        self.assertEqual(_lcs_len_from_blocks(blocks), 3)

    def test_single_diff_at_start(self):
        """One token differs at position 0."""
        orig = [99, 2, 3, 4, 5]
        reenc = [88, 2, 3, 4, 5]
        blocks = lcs_match(orig, reenc)
        oi, ri = _blocks_to_indices(blocks)
        self.assertEqual(oi, [1, 2, 3, 4])
        self.assertEqual(ri, [1, 2, 3, 4])
        _assert_lcs_optimal(self, orig, reenc, blocks)

    def test_single_diff_at_end(self):
        """One token differs at the last position."""
        orig = [1, 2, 3, 4, 99]
        reenc = [1, 2, 3, 4, 88]
        blocks = lcs_match(orig, reenc)
        oi, ri = _blocks_to_indices(blocks)
        self.assertEqual(oi, [0, 1, 2, 3])
        self.assertEqual(ri, [0, 1, 2, 3])
        _assert_lcs_optimal(self, orig, reenc, blocks)

    def test_alternating_diff(self):
        """Every other token differs."""
        orig = [1, 10, 2, 20, 3, 30]
        reenc = [1, 11, 2, 21, 3, 31]
        blocks = lcs_match(orig, reenc)
        oi, ri = _blocks_to_indices(blocks)
        self.assertEqual(oi, [0, 2, 4])
        self.assertEqual(ri, [0, 2, 4])
        _assert_lcs_optimal(self, orig, reenc, blocks)

    # ------------------------------------------------------------------ #
    # Non-unique LCS cases (validate length & validity, not exact indices)#
    # ------------------------------------------------------------------ #

    def test_all_same_elements(self):
        """All elements identical — LCS length = min(len(x), len(y)),
        but index assignment is non-unique."""
        orig = [1, 1, 1, 1]
        reenc = [1, 1, 1]
        blocks = lcs_match(orig, reenc)
        _assert_valid_lcs(self, orig, reenc, blocks)
        self.assertEqual(_lcs_len_from_blocks(blocks), 3)  # LCS length = min(4, 3) = 3
        _assert_lcs_optimal(self, orig, reenc, blocks)

    def test_repeated_with_diffs(self):
        """Repeated values mixed with diffs — LCS is non-unique.
        x = [1, 2, 1, 2], y = [2, 1, 2, 1]
        LCS length = 3 (e.g. [1,2,1] or [2,1,2]), but exact indices vary."""
        orig = [1, 2, 1, 2]
        reenc = [2, 1, 2, 1]
        blocks = lcs_match(orig, reenc)
        _assert_lcs_optimal(self, orig, reenc, blocks)
        self.assertEqual(_lcs_len_from_blocks(blocks), 3)

    def test_symmetric_non_unique(self):
        """x = [A, B, A], y = [A, B, A]. LCS is unique here (full match)."""
        orig = [10, 20, 10]
        reenc = [10, 20, 10]
        blocks = lcs_match(orig, reenc)
        oi, ri = _blocks_to_indices(blocks)
        self.assertEqual(oi, [0, 1, 2])
        self.assertEqual(ri, [0, 1, 2])

    def test_many_repeated_values(self):
        """Lots of repeated values → lots of valid LCS alignments."""
        orig = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        reenc = [1, 2, 3, 1, 2, 3]
        blocks = lcs_match(orig, reenc)
        _assert_lcs_optimal(self, orig, reenc, blocks)
        self.assertEqual(_lcs_len_from_blocks(blocks), 6)  # LCS = entire reenc

    # ------------------------------------------------------------------ #
    # Fuzz / randomised tests                                             #
    # ------------------------------------------------------------------ #

    def test_fuzz_small_sequences(self):
        """Fuzz test with many small random sequences.
        Validates LCS validity and optimal length against DP."""
        rng = random.Random(12345)
        for _ in range(500):
            n = rng.randint(0, 20)
            m = rng.randint(0, 20)
            # Small alphabet to encourage repeated values (non-unique LCS)
            x = [rng.randint(1, 5) for _ in range(n)]
            y = [rng.randint(1, 5) for _ in range(m)]
            blocks = lcs_match(x, y)
            _assert_lcs_optimal(self, x, y, blocks)

    def test_fuzz_medium_sequences(self):
        """Fuzz test with medium sequences and realistic diff rates."""
        rng = random.Random(67890)
        for _ in range(50):
            n = rng.randint(50, 200)
            base = [rng.randint(0, 10000) for _ in range(n)]
            # Mutate ~5% of positions
            y = list(base)
            for i in range(n):
                if rng.random() < 0.05:
                    y[i] = 200000 + i
            # Possibly insert/delete a few tokens
            if rng.random() < 0.3:
                pos = rng.randint(0, len(y))
                y.insert(pos, 300000)
            if rng.random() < 0.3 and len(y) > 0:
                pos = rng.randint(0, len(y) - 1)
                y.pop(pos)

            blocks = lcs_match(base, y)
            _assert_lcs_optimal(self, base, y, blocks)

    def test_fuzz_spm_realistic(self):
        """Simulate realistic SPM re-encode drift: mostly identical with
        rare split/merge diffs. Validates LCS length against DP."""
        rng = random.Random(42424)
        for _ in range(30):
            n = rng.randint(100, 500)
            x = [rng.randint(0, 50000) for _ in range(n)]
            y = list(x)
            # Simulate ~1% split: replace one token with two
            i = 0
            while i < len(y):
                if rng.random() < 0.01:
                    old_val = y[i]
                    y[i] = 100000 + old_val
                    y.insert(i + 1, 100001 + old_val)
                    i += 2
                else:
                    i += 1
            # Simulate ~1% merge: replace two tokens with one
            i = 0
            while i < len(y) - 1:
                if rng.random() < 0.01:
                    y[i] = 200000 + y[i]
                    y.pop(i + 1)
                i += 1

            blocks = lcs_match(x, y)
            _assert_lcs_optimal(self, x, y, blocks)

    # ------------------------------------------------------------------ #
    # Performance benchmarks                                              #
    # ------------------------------------------------------------------ #

    def _run_perf(self, seq_len, diff_rate=0.01, seed=42):
        """Helper: benchmark lcs_match on sequences of given length."""
        rng = random.Random(seed)
        base = [rng.randint(0, 100000) for _ in range(seq_len)]

        reenc = list(base)
        num_diffs = 0
        for i in range(seq_len):
            if rng.random() < diff_rate:
                reenc[i] = 200000 + i
                num_diffs += 1

        print(f"\n[perf] seq_len={seq_len}, diff_rate={diff_rate}, num_diffs={num_diffs}")

        t0 = time.perf_counter()
        blocks = lcs_match(base, reenc)
        elapsed = time.perf_counter() - t0

        lcs_len = _lcs_len_from_blocks(blocks)
        print(f"[perf] LCS length={lcs_len}, elapsed={elapsed:.3f}s")

        # Validate
        _assert_valid_lcs(self, base, reenc, blocks)
        self.assertGreaterEqual(lcs_len, seq_len - num_diffs)
        self.assertLessEqual(lcs_len, seq_len)
        return elapsed

    def test_long_sequence_16k_perf(self):
        """Benchmark on 16k sequences, ~1% diffs."""
        self._run_perf(16 * 1024)

    def test_long_sequence_64k_perf(self):
        """Benchmark on 64k sequences, ~1% diffs."""
        self._run_perf(64 * 1024)

    def test_long_sequence_128k_perf(self):
        """Benchmark on 128k sequences, ~1% diffs."""
        self._run_perf(128 * 1024)


class TestRedistributeRewards(unittest.TestCase):
    def test_identical_copy(self):
        """Identical sequences -> rewards copied 1:1."""
        orig = [100, 101, 102]
        reenc = [100, 101, 102]
        rewards = torch.tensor([1.0, 2.0, 3.0])
        result = redistribute_rewards(orig, reenc, rewards)
        self.assertTrue(torch.allclose(result, rewards))

    def test_empty(self):
        result = redistribute_rewards([], [], torch.tensor([], dtype=torch.float32))
        self.assertEqual(result.shape[0], 0)

    def test_split_mean_broadcast(self):
        """orig = [100, 101, 104, 105, 106]
        reenc = [100, 101, 300, 106]
        LCS = [100, 101, 106] at orig[0,1,4] / reenc[0,1,3]

        Rewards for reenc: [1.0, 2.0, 6.0, 4.0]
        Expected for orig:
          orig[0] = 1.0  (matched -> copy rewards[0])
          orig[1] = 2.0  (matched -> copy rewards[1])
          orig[2..3] = mean(rewards[2]) = 6.0  (gap: reenc has [300], orig has [104,105])
          orig[4] = 4.0  (matched -> copy rewards[3])
        """
        orig = [100, 101, 104, 105, 106]
        reenc = [100, 101, 300, 106]
        rewards = torch.tensor([1.0, 2.0, 6.0, 4.0])
        result = redistribute_rewards(orig, reenc, rewards)
        expected = torch.tensor([1.0, 2.0, 6.0, 6.0, 4.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_merge_broadcast(self):
        """orig = [100, 101, 300, 106]
        reenc = [100, 101, 104, 105, 106]
        LCS = [100, 101, 106]

        Rewards for reenc: [1.0, 2.0, 3.0, 5.0, 4.0]
        Expected for orig:
          orig[0] = 1.0
          orig[1] = 2.0
          orig[2] = mean(3.0, 5.0) = 4.0  (gap: orig has [300], reenc has [104,105])
          orig[3] = 4.0
        """
        orig = [100, 101, 300, 106]
        reenc = [100, 101, 104, 105, 106]
        rewards = torch.tensor([1.0, 2.0, 3.0, 5.0, 4.0])
        result = redistribute_rewards(orig, reenc, rewards)
        expected = torch.tensor([1.0, 2.0, 4.0, 4.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_multiple_diff_regions(self):
        """Two diff regions with matches between them.
        orig  = [1, 10, 11, 2, 20, 21, 3]
        reenc = [1, 99,     2, 88,     3]
        LCS = [1, 2, 3]

        Rewards for reenc: [0.5, 3.0, 1.0, 7.0, 2.0]
        Expected for orig:
          orig[0] = 0.5      (matched)
          orig[1] = 3.0      (gap)
          orig[2] = 3.0
          orig[3] = 1.0      (matched)
          orig[4] = 7.0      (gap)
          orig[5] = 7.0
          orig[6] = 2.0      (matched)
        """
        orig = [1, 10, 11, 2, 20, 21, 3]
        reenc = [1, 99, 2, 88, 3]
        rewards = torch.tensor([0.5, 3.0, 1.0, 7.0, 2.0])
        result = redistribute_rewards(orig, reenc, rewards)
        expected = torch.tensor([0.5, 3.0, 3.0, 1.0, 7.0, 7.0, 2.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_no_common_tokens(self):
        """No common tokens -> all orig get mean of all reenc rewards."""
        orig = [1, 2, 3]
        reenc = [4, 5]
        rewards = torch.tensor([2.0, 4.0])
        result = redistribute_rewards(orig, reenc, rewards)
        expected = torch.tensor([3.0, 3.0, 3.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_diff_at_start(self):
        """Diff region at the beginning before first match."""
        orig = [10, 11, 1, 2]
        reenc = [99, 1, 2]
        rewards = torch.tensor([6.0, 1.0, 2.0])
        result = redistribute_rewards(orig, reenc, rewards)
        expected = torch.tensor([6.0, 6.0, 1.0, 2.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_diff_at_end(self):
        """Diff region at the end after last match."""
        orig = [1, 2, 10, 11]
        reenc = [1, 2, 99]
        rewards = torch.tensor([1.0, 2.0, 9.0])
        result = redistribute_rewards(orig, reenc, rewards)
        expected = torch.tensor([1.0, 2.0, 9.0, 9.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_orig_gap_no_reenc_gap(self):
        """Orig has extra tokens in a gap but reenc has none."""
        orig = [1, 10, 11, 2]
        reenc = [1, 2]
        rewards = torch.tensor([1.0, 2.0])
        result = redistribute_rewards(orig, reenc, rewards)
        expected = torch.tensor([1.0, 0.0, 0.0, 2.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_reenc_gap_no_orig_gap(self):
        """Reenc has extra tokens in a gap but orig has none."""
        orig = [1, 2]
        reenc = [1, 50, 51, 2]
        rewards = torch.tensor([1.0, 8.0, 4.0, 2.0])
        result = redistribute_rewards(orig, reenc, rewards)
        expected = torch.tensor([1.0, 2.0])
        self.assertTrue(torch.allclose(result, expected))

    # ------------------------------------------------------------------ #
    # Additional edge cases for redistribute_rewards                      #
    # ------------------------------------------------------------------ #

    def test_single_token_identical(self):
        result = redistribute_rewards([42], [42], torch.tensor([5.0]))
        self.assertTrue(torch.allclose(result, torch.tensor([5.0])))

    def test_single_token_different(self):
        result = redistribute_rewards([1], [2], torch.tensor([5.0]))
        self.assertTrue(torch.allclose(result, torch.tensor([5.0])))

    def test_all_diff_multiple_tokens(self):
        """All different → mean of all reenc rewards broadcast."""
        orig = [1, 2, 3, 4]
        reenc = [5, 6]
        rewards = torch.tensor([10.0, 20.0])
        result = redistribute_rewards(orig, reenc, rewards)
        expected = torch.tensor([15.0, 15.0, 15.0, 15.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_diff_at_both_ends(self):
        """Diff regions at both start and end, match in middle.
        orig  = [10, 1, 2, 20, 21]
        reenc = [90, 91, 1, 2, 30]
        LCS = [1, 2]
        """
        orig = [10, 1, 2, 20, 21]
        reenc = [90, 91, 1, 2, 30]
        rewards = torch.tensor([2.0, 4.0, 1.0, 2.0, 6.0])
        result = redistribute_rewards(orig, reenc, rewards)
        # orig[0] = mean(2.0, 4.0) = 3.0 (gap before LCS)
        # orig[1] = 1.0 (matched)
        # orig[2] = 2.0 (matched)
        # orig[3..4] = mean(6.0) = 6.0 (gap after LCS)
        expected = torch.tensor([3.0, 1.0, 2.0, 6.0, 6.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_many_small_gaps(self):
        """Many alternating match/gap regions.
        orig  = [1, 10, 2, 20, 3, 30, 4]
        reenc = [1, 11, 2, 21, 3, 31, 4]
        LCS = [1, 2, 3, 4]
        """
        orig = [1, 10, 2, 20, 3, 30, 4]
        reenc = [1, 11, 2, 21, 3, 31, 4]
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = redistribute_rewards(orig, reenc, rewards)
        # orig[0]=1.0, orig[1]=2.0(gap), orig[2]=3.0, orig[3]=4.0(gap),
        # orig[4]=5.0, orig[5]=6.0(gap), orig[6]=7.0
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertTrue(torch.allclose(result, expected))

    def test_reward_conservation_property(self):
        """For same-length sequences, total reward should be approximately
        conserved (modulo gap-broadcast effects). When lengths match and
        there are only 1-to-1 replacements, reward sum is exactly equal."""
        orig = [1, 99, 3, 98, 5]
        reenc = [1, 88, 3, 87, 5]
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = redistribute_rewards(orig, reenc, rewards)
        # Each gap is 1-to-1, so reward sum should be preserved
        self.assertAlmostEqual(result.sum().item(), rewards.sum().item(), places=5)

    # ------------------------------------------------------------------ #
    # Performance benchmark                                               #
    # ------------------------------------------------------------------ #

    def test_redistribute_128k_perf(self):
        """Benchmark redistribute_rewards on 128k sequences with ~1% diffs
        (realistic SPM re-encode scenario)."""
        seq_len = 128 * 1024
        diff_rate = 0.01
        rng = random.Random(77777)

        base = [rng.randint(0, 100000) for _ in range(seq_len)]
        reenc = list(base)
        num_diffs = 0
        for i in range(seq_len):
            if rng.random() < diff_rate:
                reenc[i] = 200000 + i
                num_diffs += 1

        rewards = torch.randn(len(reenc))

        # Warm-up lcs_match (not timed here, already benchmarked in TestLcsMatch)
        blocks = lcs_match(base, reenc)
        _assert_valid_blocks(self, base, reenc, blocks)

        # Time the full redistribute_rewards pipeline (lcs + redistribution)
        t0 = time.perf_counter()
        result = redistribute_rewards(base, reenc, rewards)
        elapsed = time.perf_counter() - t0

        print(
            f"\n[redist perf] seq_len={seq_len}, diff_rate={diff_rate}, "
            f"num_diffs={num_diffs}, elapsed={elapsed:.3f}s"
        )

        # Basic correctness checks
        self.assertEqual(result.shape[0], len(base))
        # Matched positions should have exact reward values
        for blk in blocks:
            if blk.match:
                self.assertTrue(
                    torch.allclose(
                        result[blk.x_start:blk.x_end],
                        rewards[blk.y_start:blk.y_end],
                    )
                )


class TestBatchAndAsync(unittest.TestCase):
    """Tests for redistribute_rewards_batch and redistribute_rewards_batch_async."""
    @classmethod
    def tearDownClass(cls):
        pass

    def _make_sample(self, rng, seq_len=100, diff_rate=0.05):
        """Generate a single (orig, reenc, rewards) sample."""
        base = [rng.randint(0, 100000) for _ in range(seq_len)]
        reenc = list(base)
        for i in range(seq_len):
            if rng.random() < diff_rate:
                reenc[i] = 200000 + i
        rewards = torch.randn(len(reenc), dtype=torch.float32)
        return base, reenc, rewards

    # ------------------------------------------------------------------ #
    # redistribute_rewards_batch (synchronous)                            #
    # ------------------------------------------------------------------ #

    def test_batch_empty(self):
        results = redistribute_rewards_batch([], [], [])
        self.assertEqual(results, [])

    def test_batch_single(self):
        orig = [1, 2, 3]
        reenc = [1, 2, 3]
        rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        results = redistribute_rewards_batch([orig], [reenc], [rewards])
        self.assertEqual(len(results), 1)
        self.assertTrue(torch.allclose(results[0], rewards))

    def test_batch_matches_sequential(self):
        """Batch result must match sequential calls exactly."""
        rng = random.Random(11111)
        n_samples = 10
        origs, reencs, rews = [], [], []
        for _ in range(n_samples):
            o, r, w = self._make_sample(rng)
            origs.append(o)
            reencs.append(r)
            rews.append(w)

        batch_results = redistribute_rewards_batch(origs, reencs, rews)
        for i in range(n_samples):
            expected = redistribute_rewards(origs[i], reencs[i], rews[i])
            self.assertTrue(
                torch.allclose(batch_results[i], expected),
                f"Sample {i}: batch result differs from sequential",
            )

    # ------------------------------------------------------------------ #
    # redistribute_rewards_batch_async                                    #
    # ------------------------------------------------------------------ #

    def test_async_empty(self):
        results = asyncio.run(redistribute_rewards_batch_mp([], [], []))
        self.assertEqual(results, [])

    def test_async_single(self):
        orig = [1, 2, 3]
        reenc = [1, 2, 3]
        rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        results = asyncio.run(redistribute_rewards_batch_mp([orig], [reenc], [rewards]))
        self.assertEqual(len(results), 1)
        self.assertTrue(torch.allclose(results[0], rewards))

    def test_async_matches_sequential(self):
        """Async batch result must match sequential calls exactly."""
        rng = random.Random(22222)
        n_samples = 10
        origs, reencs, rews = [], [], []
        for _ in range(n_samples):
            o, r, w = self._make_sample(rng)
            origs.append(o)
            reencs.append(r)
            rews.append(w)

        async_results = asyncio.run(
            redistribute_rewards_batch_mp(origs, reencs, rews, max_workers=2)
        )
        for i in range(n_samples):
            expected = redistribute_rewards(origs[i], reencs[i], rews[i])
            self.assertTrue(
                torch.allclose(async_results[i], expected),
                f"Sample {i}: async result differs from sequential",
            )

    def test_async_preserves_dtype(self):
        """Verify that output dtype matches input dtype."""
        orig = [1, 2, 3]
        reenc = [1, 2, 3]
        rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        results = asyncio.run(redistribute_rewards_batch_mp([orig], [reenc], [rewards]))
        self.assertEqual(results[0].dtype, torch.float32)

    def test_async_various_lengths(self):
        """Samples with different sequence lengths in the same batch."""
        rng = random.Random(33333)
        origs, reencs, rews = [], [], []
        for length in [1, 5, 50, 200, 500]:
            o, r, w = self._make_sample(rng, seq_len=length, diff_rate=0.03)
            origs.append(o)
            reencs.append(r)
            rews.append(w)

        async_results = asyncio.run(
            redistribute_rewards_batch_mp(origs, reencs, rews, max_workers=3)
        )
        for i in range(len(origs)):
            expected = redistribute_rewards(origs[i], reencs[i], rews[i])
            self.assertTrue(
                torch.allclose(async_results[i], expected),
                f"Sample {i} (len={len(origs[i])}): async result differs",
            )

    def test_async_thread_safety(self):
        """Launch async batch from multiple threads concurrently."""
        import concurrent.futures

        rng = random.Random(44444)
        n_threads = 4
        n_samples_per_thread = 5

        # Pre-generate data for each thread.
        thread_data = []
        for _ in range(n_threads):
            origs, reencs, rews = [], [], []
            for _ in range(n_samples_per_thread):
                o, r, w = self._make_sample(rng, seq_len=50)
                origs.append(o)
                reencs.append(r)
                rews.append(w)
            thread_data.append((origs, reencs, rews))

        errors = []

        def _run_in_thread(idx):
            origs, reencs, rews = thread_data[idx]
            try:
                async_results = asyncio.run(
                    redistribute_rewards_batch_mp(origs, reencs, rews, max_workers=2)
                )
                for i in range(n_samples_per_thread):
                    expected = redistribute_rewards(origs[i], reencs[i], rews[i])
                    if not torch.allclose(async_results[i], expected):
                        errors.append(f"Thread {idx}, sample {i}: mismatch")
            except Exception as e:
                errors.append(f"Thread {idx}: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as tex:
            futs = [tex.submit(_run_in_thread, i) for i in range(n_threads)]
            concurrent.futures.wait(futs)

        self.assertEqual(errors, [], f"Thread-safety errors: {errors}")

    # ------------------------------------------------------------------ #
    # Performance benchmark                                               #
    # ------------------------------------------------------------------ #

    def test_async_batch_perf(self):
        """Benchmark async batch on 8 x 16k sequences.
        Compare wall-clock time of async (multiprocess) vs sequential."""

        rng = random.Random(55555)
        n_samples = 64
        seq_len = 64 * 1024
        origs, reencs, rews = [], [], []
        for _ in range(n_samples):
            o, r, w = self._make_sample(rng, seq_len=seq_len, diff_rate=0.01)
            origs.append(o)
            reencs.append(r)
            rews.append(w)

        # Sequential timing
        t0 = time.perf_counter()
        seq_results = redistribute_rewards_batch(origs, reencs, rews)
        t_seq = time.perf_counter() - t0
        print(f'test_async_batch_perf trace1 {t_seq}')

        # Async timing
        t0 = time.perf_counter()
        async_results = async_results = asyncio.run(
            redistribute_rewards_batch_mp(origs, reencs, rews, max_workers=8)
        )
        t_async = time.perf_counter() - t0
        print(f'test_async_batch_perf trace2 {t_async}')

        print(
            f"\n[batch perf] {n_samples}x{seq_len}: "
            f"sequential={t_seq:.3f}s, async={t_async:.3f}s, "
            f"speedup={t_seq / t_async:.2f}x"
        )

        # Correctness: async must match sequential
        for i in range(n_samples):
            self.assertTrue(
                torch.allclose(async_results[i], seq_results[i]),
                f"Sample {i}: async/seq mismatch in perf test",
            )

    def test_real(self):
        path = '/mnt/ceph-hz1-csp/mm-base-plt2/user_astrachang/tmp/welm-refix.pt'
        if os.path.isfile(path):
            xs = torch.load(path, weights_only=False)
            for i, x in xs:
                orig_ids = x['infer_output_tokens']
                refix_ids = x['refix_infer_output_tokens']

                blocks = lcs_match(orig_ids, refix_ids)
                visualize_alignment(orig_ids, refix_ids, blocks)

                if i >= 10:
                    break


def visualize_alignment(
    orig_ids: list,
    refix_ids: list,
    blocks: list,
    cols: int = 30,
):
    """Visualize token alignment between refix and orig sequences.

    Walks through AlignedChunk blocks. For match blocks, both rows show
    green tokens at the same column. For diff blocks, each row shows its
    own red tokens; the shorter side is padded with blanks so subsequent
    matched tokens stay visually aligned.

    Output format (pairs of lines):
      odd  line = refix (y side)
      even line = orig  (x side)
    Green = matched, Red = unmatched.

    Args:
        orig_ids:  The orig (x) token id sequence.
        refix_ids: The refix (y) token id sequence.
        blocks:    List of AlignedChunk from lcs_match(orig_ids, refix_ids).
        cols:      Number of token columns per output line pair.
    """
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    WIDTH = 7
    BLANK = " " * WIDTH

    def _fmt(tid, color):
        return f"{color}{tid:>{WIDTH}}{RESET}"

    # Build two aligned cell lists: refix_cells (y) and orig_cells (x).
    refix_cells: list[str] = []
    orig_cells: list[str] = []

    for blk in blocks:
        if blk.match:
            # Matched region: x and y have equal length, show green.
            for k in range(blk.x_end - blk.x_start):
                orig_cells.append(_fmt(orig_ids[blk.x_start + k], GREEN))
                refix_cells.append(_fmt(refix_ids[blk.y_start + k], GREEN))
        else:
            # Diff region: show each side in red, pad shorter side with blanks.
            x_len = blk.x_end - blk.x_start
            y_len = blk.y_end - blk.y_start
            max_len = max(x_len, y_len)
            for k in range(max_len):
                orig_cells.append(_fmt(orig_ids[blk.x_start + k], RED) if k < x_len else BLANK)
                refix_cells.append(_fmt(refix_ids[blk.y_start + k], RED) if k < y_len else BLANK)

    total_cols = len(refix_cells)
    n_rows = (total_cols + cols - 1) // cols

    n_matched = sum(blk.x_end - blk.x_start for blk in blocks if blk.match)

    print(f"\n{'=' * 80}")
    print(f"  orig_ids  length: {len(orig_ids)}")
    print(f"  refix_ids length: {len(refix_ids)}")
    print(f"  matched tokens:   {n_matched}")
    print(f"{'=' * 80}")

    for row in range(n_rows):
        s = row * cols
        e = min(s + cols, total_cols)
        print(f" orig[{row:>4}]: {''.join(orig_cells[s:e])}")
        print(f"refix[{row:>4}]: {''.join(refix_cells[s:e])}")
        print()


class TestS1s(unittest.TestCase):
    """Tests for redistribute_rewards_batch and redistribute_rewards_batch_async."""
    @classmethod
    def tearDownClass(cls):
        pass

    def test_real(self):
        from pprint import pprint

        from transformers import AutoTokenizer

        path = '/mnt/ceph-hz1-csp/mm-base-plt2/user_xiaotaoliu/models/welm_v4_sft_v3_actor_orm_exp4_orm_adv_v2/800'
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        path = '/mnt/ceph-hz1-csp/mm-base-plt2/user_xiaotaoliu/data/nan-debug/ex_rollout_batches-0-0.pt'
        '''
        dict_keys(['tokens', 'sequence_lengths', 'sequence_lengths_original', 'prompt_lengths', 'gt_label',
        'sample_is_anchor', 'refix_token_mask', 'original_tokens',
        'rm_0_tokens', 'rm_0_prompt_lengths', 'rm_0_sequence_lengths', 'rm_0_output_mask', 'rm_0_fmt_status',
        ...
        'rm_9_tokens', 'rm_9_prompt_lengths', 'rm_9_sequence_lengths', 'rm_9_output_mask', 'rm_9_fmt_status',
        'routed_experts', 'rollout_log_probs', 'rm_rewards', 'rewards', 'per_token_rewards', 'diff_coef',
        'balance_coef', 'token_level_coef', 'offpolicy_mask', 'hdbq_prefer_well', 'hdbq_zy_fd_score', 'hdbq_cc_score',
        'user_prefer_well', 'dfsw_error_point', 'gshl_error_point', 'ljhl_error_point', 'point_ok', 'sscw_sentence_cnt',
        'xxry_sentence_cnt', 'think_lens_tensor', 'answer_lens_tensor', 'think_lens_scores', 'fmt_scores', 'pass_use',
        'select_idx', 'greedy_best', 'logprobs_original', 'logprobs', 'ref_logprobs', 'mask', 'mask_original',
        'returns', 'advantages'])
        '''

        n_mismatch = 0
        if os.path.isfile(path):
            xs = torch.load(path, weights_only=False)
            for i, x in enumerate(xs):
                orig_ids = x['original_tokens'].tolist()
                refix_ids = x['tokens'].tolist()

                blocks = lcs_match(orig_ids, refix_ids)
                if orig_ids != refix_ids:
                    n_mismatch += 1
                    # self.visualize_alignment(orig_ids, refix_ids, blocks)

                    # with open('orig.txt', 'w') as outf:
                    #     outf.write(tokenizer.decode(orig_ids))
                    # with open('refix.txt', 'w') as outf:
                    #     outf.write(tokenizer.decode(refix_ids))

                    prompt_len = x['prompt_lengths'].item()
                    seq_len = x['sequence_lengths_original'].item()
                    # print(x['logprobs'][: x['sequence_lengths'].item()])

                    # logprobs 看起来有一堆垃圾，这个和 rollout logps 有什么关联吗？
                    # 在 match 的情况下，看着还 ok...
                    logps = x['logprobs'][:seq_len].tolist()
                    rollout_log_probs = x['rollout_log_probs'][:seq_len].tolist()

                    RED = "\033[91m"
                    RESET = "\033[0m"
                    cols = 30
                    print()
                    n = max(len(logps), len(rollout_log_probs))
                    for row_start in range(0, n, cols):
                        row_end = min(row_start + cols, n)
                        oi_parts = []
                        ri_parts = []
                        lp_parts = []
                        rp_parts = []
                        for j in range(row_start, row_end):
                            c = RED if j >= prompt_len - 1 else ""
                            r = RESET if j >= prompt_len - 1 else ""

                            oi_parts.append(
                                f"{c}{orig_ids[j + 1]:>8d}{r}" if j +
                                1 < len(orig_ids) else "        "
                            )
                            ri_parts.append(
                                f"{c}{refix_ids[j + 1]:>8d}{r}" if j +
                                1 < len(refix_ids) else "        "
                            )

                            lp_parts.append(
                                f"{c}{logps[j]:8.3f}{r}" if j < len(logps) else "        "
                            )
                            rp_parts.append(
                                f"{c}{rollout_log_probs[j]:8.3f}{r}" if j <
                                len(rollout_log_probs) else "        "
                            )

                        print(f"orig_id[{row_start:>5}]: {' '.join(oi_parts)}")
                        print(f"rfix_id[{row_start:>5}]: {' '.join(ri_parts)}")
                        print(f"  logps[{row_start:>5}]: {' '.join(lp_parts)}")
                        print(f"rollout[{row_start:>5}]: {' '.join(rp_parts)}")
                        print()

                    break
        print(f'{n_mismatch=}')
