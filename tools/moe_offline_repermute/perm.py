# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com, yea@tencent.com
"""Recursive balanced bisection for EP-agnostic expert permutation.

For each layer independently we compute a permutation of the ``num_experts``
experts such that, **for any** ``EP_size = 2^k`` (``k = 1..log2(num_experts)``),
slicing the new expert order into ``EP_size`` contiguous segments yields
segments with approximately balanced total weight.

This is achieved by recursive 2-way splits driven by
``balanced_packing(num_packs=2)``: each recursion bisects the current item
list into two equal-size halves, and the final order is *left-subtree
concatenated with right-subtree*. Every dyadic interval in the new order is
exactly a node in the split tree and therefore enjoys the local balance the
greedy split provides.

Convention (position-major)
---------------------------
``perm[L, p] = old_id`` -- "at new position p of layer L sits the expert that
originally had id ``perm[L, p]``". Tensors are then rewritten by gathering
along dim 0 with ``perm[L]``.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .eplb_vendored import balanced_packing, inverse


def recursive_bisect(items: List[int], weights: np.ndarray) -> List[int]:
    """Return a permutation of ``items`` produced by recursive balanced bisection.

    Parameters
    ----------
    items : list[int]
        Subset of expert ids (initially ``list(range(num_experts))``).
    weights : np.ndarray, shape ``(num_experts,)``
        Per-expert weight of the *original* expert ids; indexed directly by id.

    Returns
    -------
    list[int]
        A permutation of ``items``. Length and contents match ``items``;
        only the order changes.
    """
    n = len(items)
    if n == 1:
        return list(items)
    assert n % 2 == 0, ("recursive_bisect requires power-of-two count, got "
                        f"len(items)={n}")

    sub_w = [[float(weights[i]) for i in items]]
    pack_index, _ = balanced_packing(sub_w, num_packs=2)
    pack_index_row = pack_index[0]

    left = [items[k] for k in range(n) if pack_index_row[k] == 0]
    right = [items[k] for k in range(n) if pack_index_row[k] == 1]
    assert len(left) == len(right) == n // 2

    return recursive_bisect(left, weights) + recursive_bisect(right, weights)


def compute_perm(weights: np.ndarray) -> np.ndarray:
    """Compute the per-layer position-major permutation.

    Parameters
    ----------
    weights : np.ndarray, shape ``(num_layers, num_experts)``
        Per-expert weight estimates.

    Returns
    -------
    perm : np.ndarray, shape ``(num_layers, num_experts)``, int64
        ``perm[L, p]`` is the original expert id at new position p of layer L.
    """
    num_layers, num_experts = weights.shape
    assert (num_experts &
            (num_experts - 1)) == 0, (f"num_experts must be a power of two, got {num_experts}")

    perm = np.empty_like(weights, dtype=np.int64)
    base = list(range(num_experts))
    for L in range(num_layers):
        order = recursive_bisect(base, weights[L])
        assert sorted(order) == base, f"perm row {L} is not a permutation"
        perm[L] = np.asarray(order, dtype=np.int64)
    return perm


def inverse_perm(perm: np.ndarray) -> np.ndarray:
    """Row-wise inverse permutation.

    ``inv[L, q] = new_position_of_old_expert_q``.
    """
    inv_lists = inverse(perm.tolist())
    return np.asarray(inv_lists, dtype=np.int64)


def multi_ep_load(weights: np.ndarray, perm: np.ndarray, ep_size: int) -> np.ndarray:
    """Per-EP-rank total load after applying a permutation.

    Parameters
    ----------
    weights : np.ndarray, shape ``(num_layers, num_experts)``
    perm : np.ndarray, shape ``(num_layers, num_experts)``
        Pass an identity permutation (``np.broadcast_to(np.arange(N), ...)``)
        to obtain the *before* baseline.
    ep_size : int
        Number of contiguous segments to split the new expert order into.
        Must divide ``num_experts``.

    Returns
    -------
    load : np.ndarray, shape ``(num_layers, ep_size)``
    """
    num_layers, num_experts = weights.shape
    assert num_experts % ep_size == 0, (
        f"num_experts={num_experts} not divisible by ep_size={ep_size}"
    )
    seg = num_experts // ep_size
    permuted = np.take_along_axis(weights, perm, axis=1)
    return permuted.reshape(num_layers, ep_size, seg).sum(axis=-1)


def imbalance_ratio(load: np.ndarray) -> np.ndarray:
    """Per-layer ``max / min`` ratio across EP segments. Shape ``(num_layers,)``.

    Layers with a zero-load segment are reported as ``inf`` so that summary
    statistics surface the pathological case loudly.
    """
    assert load.ndim == 2
    mx = load.max(axis=1).astype(np.float64)
    mn = load.min(axis=1).astype(np.float64)
    return np.where(mn > 0, mx / np.where(mn > 0, mn, 1.0), np.inf)


def assert_valid_perm(perm: np.ndarray, num_experts: int) -> None:
    """Validate that every row of ``perm`` is a permutation of ``range(N)``."""
    assert perm.ndim == 2, f"perm must be 2-D, got shape {perm.shape}"
    assert perm.shape[1] == num_experts, (f"perm cols={perm.shape[1]} != num_experts={num_experts}")
    expected = np.arange(num_experts, dtype=perm.dtype)
    for L in range(perm.shape[0]):
        row_sorted = np.sort(perm[L])
        assert np.array_equal(row_sorted, expected), (
            f"perm row {L} is not a valid permutation of [0, {num_experts}); "
            f"got unique={len(np.unique(perm[L]))}"
        )


__all__ = [
    "assert_valid_perm",
    "compute_perm",
    "inverse_perm",
    "multi_ep_load",
    "imbalance_ratio",
    "recursive_bisect",
]
