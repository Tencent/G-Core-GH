# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com, yea@tencent.com
"""Vendored helpers from EPLB_visualization/eplb_np.py.

Only `balanced_packing` and `inverse` are needed by this tool. Vendored to
avoid sys.path hacks against `EPLB_visualization` (which is a loose script
collection without a package layout).

Source: https://github.com/CalvinXKY/EPLB_visualization (DeepSeek 2025, MIT)
"""
from typing import List, Sequence, Tuple

import numpy as np


def balanced_packing(weight: Sequence[Sequence[float]],
                     num_packs: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Pack n weighted objects to m packs with balanced totals.

    Each pack is required to contain exactly ``n / m`` objects; within that
    constraint the per-pack weight sum is balanced greedily.

    Parameters
    ----------
    weight : nested list shaped ``[X, n]``
        Per-row item weights. ``X`` is treated as an outer batch (e.g. layers).
    num_packs : int
        Number of packs. Must divide ``n``.

    Returns
    -------
    pack_index : nested list ``[X, n]``
        Pack id of each item.
    rank_in_pack : nested list ``[X, n]``
        Position of the item inside its pack (0-based).
    """
    num_layers = len(weight)
    num_groups = len(weight[0])
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = [list(range(num_groups)) for _ in range(num_layers)]
        rank_in_pack = [[0] * num_groups for _ in range(num_layers)]
        return pack_index, rank_in_pack

    pack_index = [[-1] * num_groups for _ in range(num_layers)]
    rank_in_pack = [[-1] * num_groups for _ in range(num_layers)]

    for i in range(num_layers):
        sorted_indices = np.argsort([-x for x in weight[i]])
        pack_weights = [0.0] * num_packs
        pack_items = [0] * num_packs

        for group in sorted_indices:
            pack = min(
                (j for j in range(num_packs) if pack_items[j] < groups_per_pack),
                key=lambda x: pack_weights[x],
            )
            assert pack_items[pack] < groups_per_pack
            pack_index[i][group] = pack
            rank_in_pack[i][group] = pack_items[pack]
            pack_weights[pack] += weight[i][group]
            pack_items[pack] += 1

    return pack_index, rank_in_pack


def inverse(perm: Sequence[Sequence[int]]) -> List[List[int]]:
    """Compute the row-wise inverse permutation."""
    n_rows = len(perm)
    n_cols = len(perm[0])
    inv = [[-1] * n_cols for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(n_cols):
            inv[i][perm[i][j]] = j
    return inv


__all__ = ["balanced_packing", "inverse"]
