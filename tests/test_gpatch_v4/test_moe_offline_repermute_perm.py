# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com, yea@tencent.com
"""Unit tests for the MoE offline recursive-bisection permutation core."""
import unittest

import numpy as np

from tools.moe_offline_repermute.perm import (
    assert_valid_perm,
    compute_perm,
    imbalance_ratio,
    inverse_perm,
    multi_ep_load,
    recursive_bisect,
)


class PermBasics(unittest.TestCase):
    def test_recursive_bisect_is_a_permutation(self) -> None:
        rng = np.random.default_rng(0)
        for n in (4, 8, 16, 32, 64, 256):
            w = rng.exponential(scale=1.0, size=n)
            order = recursive_bisect(list(range(n)), w)
            self.assertEqual(sorted(order), list(range(n)))

    def test_inverse_round_trip(self) -> None:
        rng = np.random.default_rng(1)
        weights = rng.exponential(scale=1.0, size=(3, 64))
        perm = compute_perm(weights)
        inv = inverse_perm(perm)
        for L in range(3):
            self.assertTrue(np.array_equal(inv[L][perm[L]], np.arange(64)))
            self.assertTrue(np.array_equal(perm[L][inv[L]], np.arange(64)))

    def test_position_major_convention(self) -> None:
        """Gathering a tensor with ``perm`` reproduces the position-major rule."""
        rng = np.random.default_rng(2)
        n = 16
        weights = rng.exponential(scale=1.0, size=(1, n))
        perm = compute_perm(weights)[0]
        ids = np.arange(n) * 100  # toy "tensor" indexed by old expert id
        new_ids = ids[perm]
        for new_pos in range(n):
            self.assertEqual(int(new_ids[new_pos]), int(ids[perm[new_pos]]))


class BisectBalance(unittest.TestCase):
    """The plan's central claim: every EP=2^k contiguous segmentation is balanced."""
    def test_recursive_bisect_balances_root_split(self) -> None:
        rng = np.random.default_rng(3)
        for trial in range(5):
            weights = rng.exponential(scale=2.0, size=(1, 256))
            perm = compute_perm(weights)
            load = multi_ep_load(weights, perm, ep_size=2)[0]
            ratio = load.max() / load.min()
            self.assertLess(ratio, 1.05, f"root split too unbalanced: {ratio}")

    def test_recursive_bisect_no_worse_than_identity_at_all_dyadic_sizes(self) -> None:
        rng = np.random.default_rng(4)
        weights = rng.exponential(scale=2.0, size=(8, 256))
        identity = np.broadcast_to(np.arange(256, dtype=np.int64), weights.shape).copy()
        perm = compute_perm(weights)
        for ep in (2, 4, 8, 16, 32, 64, 128):
            load_b = multi_ep_load(weights, identity, ep)
            load_a = multi_ep_load(weights, perm, ep)
            ratio_b = imbalance_ratio(load_b).mean()
            ratio_a = imbalance_ratio(load_a).mean()
            self.assertLessEqual(
                ratio_a,
                ratio_b * 1.01,
                f"EP={ep}: ratio_after({ratio_a}) noticeably worse than "
                f"ratio_before({ratio_b})",
            )

    def test_recursive_bisect_strong_balance_at_small_ep(self) -> None:
        """At small EP sizes the segments are large enough to absorb tail
        weights -- expect tight max/min.
        """
        rng = np.random.default_rng(7)
        weights = rng.exponential(scale=2.0, size=(8, 256))
        perm = compute_perm(weights)
        for ep in (2, 4, 8):
            load = multi_ep_load(weights, perm, ep)
            ratio = imbalance_ratio(load).mean()
            self.assertLess(
                ratio,
                1.10,
                f"EP={ep}: small-EP balance unexpectedly loose: {ratio}",
            )

    def test_uniform_weights_gives_perfect_balance(self) -> None:
        weights = np.ones((1, 32))
        perm = compute_perm(weights)
        for ep in (2, 4, 8, 16, 32):
            load = multi_ep_load(weights, perm, ep)[0]
            self.assertEqual(load.max(), load.min())


class GatherSemantics(unittest.TestCase):
    """Exercise the same gather convention used by ``repermute_hf_ckpt.py``."""
    def test_gather_along_dim0_is_position_major(self) -> None:
        rng = np.random.default_rng(5)
        n = 16
        h = 4
        weights = rng.exponential(scale=1.0, size=(1, n))
        perm = compute_perm(weights)[0]
        tensor = rng.normal(size=(n, h))
        new_tensor = tensor[perm]
        for new_pos in range(n):
            self.assertTrue(np.array_equal(new_tensor[new_pos], tensor[perm[new_pos]]))


class Validation(unittest.TestCase):
    def test_assert_valid_perm_accepts_identity(self) -> None:
        n = 32
        perm = np.broadcast_to(np.arange(n, dtype=np.int64), (4, n)).copy()
        assert_valid_perm(perm, n)

    def test_assert_valid_perm_rejects_duplicate(self) -> None:
        n = 8
        bad = np.broadcast_to(np.arange(n, dtype=np.int64), (1, n)).copy()
        bad[0, 1] = bad[0, 0]
        with self.assertRaises(AssertionError):
            assert_valid_perm(bad, n)

    def test_imbalance_ratio_handles_zero_min(self) -> None:
        load = np.array([[10.0, 0.0, 5.0]])
        r = imbalance_ratio(load)
        self.assertTrue(np.isinf(r[0]))
