"""Reward redistribution via LCS alignment (Myers diff, O(N*D)).

Given:
  - orig_ids:      token IDs from rollout
  - reenc_ids:     token IDs after decode -> re-encode (may differ due to SPM)
  - rewards:       per-token rewards aligned to reenc_ids

We use Myers diff to find the shortest edit script (SES) between orig_ids
and reenc_ids, which implicitly gives us the LCS. LCS tokens are matched
and copy rewards directly. Non-LCS tokens form contiguous "diff regions";
rewards from the reencoded diff region are mean-pooled and broadcast to
the corresponding orig diff region.

Myers diff: O((N+M)*D) time, O(N+M) space, where D is the number of
differences. For SPM re-encode drift (~1% diffs), D << N — much faster
than O(N*M) DP.

Implementation notes (v2 optimised):
  - Primary path (_myers_lcs) stores V arrays as compact list-of-int
    snapshots, eliminating ~2 Python object allocations per snake step.
  - Back-tracking walks V snapshots in reverse and emits LCS match index
    pairs directly.
  - Legacy _myers_diff / _collect_edits path retained for debugging.

See:
  Myers (1986) "An O(ND) Difference Algorithm and Its Variations"
  http://www.xmailserver.org/diff2.pdf
"""

import asyncio
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Pool


@dataclass(slots=True)
class AlignedChunk:
    """A contiguous block in the alignment between two sequences.

    Attributes:
        match: True if x[x_start:x_end] == y[y_start:y_end] element-wise
               (i.e. this block is part of the LCS).  False for diff regions.
        x_start: Start index in x (inclusive).
        x_end:   End index in x (exclusive).
        y_start: Start index in y (inclusive).
        y_end:   End index in y (exclusive).

    Invariants:
      - For match=True:  (x_end - x_start) == (y_end - y_start) > 0.
      - For match=False: at least one of the spans is non-empty.
    """
    match: bool
    x_start: int
    x_end: int
    y_start: int
    y_end: int


def debug_match_indices_to_blocks(
    x_len: int,
    y_len: int,
    orig_indices: List[int],
    reenc_indices: List[int],
) -> List[AlignedChunk]:
    """Convert matched LCS index pairs into a list of AlignedBlock.

    Fills in mismatch blocks between consecutive matches and at the
    boundaries.  Pure conversion utility — no algorithmic logic.

    NOTE: ***For debugging purpose only***.
    """
    # Merge consecutive match indices into contiguous match blocks,
    # interleaved with mismatch gaps: gap, match-run, gap, ..., trailing gap.
    blocks_final: List[AlignedChunk] = []
    xi_prev, yi_prev = 0, 0
    k = 0
    n_matches = len(orig_indices)
    while k < n_matches:
        xi, yi = orig_indices[k], reenc_indices[k]
        # Mismatch gap before this match run
        if xi > xi_prev or yi > yi_prev:
            blocks_final.append(
                AlignedChunk(
                    match=False,
                    x_start=xi_prev,
                    x_end=xi,
                    y_start=yi_prev,
                    y_end=yi,
                )
            )
        # Find the end of the contiguous match run
        run_start_k = k
        while (
            k + 1 < n_matches and orig_indices[k + 1] == orig_indices[k] + 1 and
            reenc_indices[k + 1] == reenc_indices[k] + 1
        ):
            k += 1
        run_len = k - run_start_k + 1
        blocks_final.append(
            AlignedChunk(
                match=True,
                x_start=xi,
                x_end=xi + run_len,
                y_start=yi,
                y_end=yi + run_len,
            )
        )
        xi_prev = xi + run_len
        yi_prev = yi + run_len
        k += 1

    # Trailing mismatch
    if xi_prev < x_len or yi_prev < y_len:
        blocks_final.append(
            AlignedChunk(
                match=False,
                x_start=xi_prev,
                x_end=x_len,
                y_start=yi_prev,
                y_end=y_len,
            )
        )

    return blocks_final


def naive_dp_lcs_match(
    orig_ids: List[int],
    reenc_ids: List[int],
) -> List[AlignedChunk]:
    """Compute LCS via naive O(N*M) DP and return aligned blocks.

    Correctness reference; only the return value is converted to
    AlignedBlock list.

    NOTE: ***For debugging purpose only***.

    Args:
        orig_ids: Original token ID sequence.
        reenc_ids: Re-encoded token ID sequence.

    Returns:
        AlignedBlock list covering both sequences.
    """
    n, m = len(orig_ids), len(reenc_ids)

    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if orig_ids[i - 1] == reenc_ids[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to recover matched indices
    orig_indices: List[int] = []
    reencoded_indices: List[int] = []
    i, j = n, m
    while i > 0 and j > 0:
        if orig_ids[i - 1] == reenc_ids[j - 1]:
            orig_indices.append(i - 1)
            reencoded_indices.append(j - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    orig_indices.reverse()
    reencoded_indices.reverse()

    # Convert to AlignedBlock list
    return debug_match_indices_to_blocks(n, m, orig_indices, reencoded_indices)


@dataclass
class EditEntry:
    op: str
    x_i: int
    y_i: int


@dataclass
class VEntry:
    reached_x: int
    edit: EditEntry
    parent: 'VEntry | None' = None


def _collect_edits(node: VEntry) -> List[EditEntry]:
    """Walk the linked list back to the root and return edits in order."""
    edits = []
    cur = node
    while cur is not None:
        edits.append(cur.edit)
        cur = cur.parent
    edits.reverse()
    return edits


def naive_myers_diff(
    x: List[int],
    y: List[int],
) -> List[EditEntry]:
    """Compute the shortest edit script between two sequences using Myers algorithm.

    Returns a list of EditEntry where:
      op='equal': a[i] == b[j] (matched)
      op='delete': a[i] is deleted (only in a)
      tag='insert': b[j] is inserted (only in b)

    Ordered by position in the edit graph traversal.

    AI 写半天写不对，看了下基本上都在复读 google 的结果，然而互联网上的写法都是各种这里 hack 下那里短路下，
    AI 一直在幻觉融合，改了 3 遍还是错的...，算了手写吧。

    NOTE: ***For debugging purpose only***.
    """
    l_x, l_y = len(x), len(y)

    if l_x == 0 and l_y == 0:
        return []
    if l_x == 0:
        return [EditEntry(op="insert", x_i=-1, y_i=j) for j in range(l_y)]
    if l_y == 0:
        return [EditEntry(op="delete", x_i=i, y_i=-1) for i in range(l_x)]

    # Myers algorithm: find shortest edit path in the edit graph.
    # V[k] = furthest reaching x on diagonal k.
    # We index V with offset so that V[k + max_d] is V[k].
    max_d = l_x + l_y

    # Use a dict for sparse V storage (only ~2*d+1 entries per iteration)
    new_e = EditEntry(op='start', x_i=-1, y_i=-1)
    v = {
        0: VEntry(reached_x=-1, edit=new_e),
    }
    new_e_x_i = new_e.x_i + 1
    new_e_y_i = new_e.y_i + 1
    while new_e_x_i < l_x and new_e_y_i < l_y and x[new_e_x_i] == y[new_e_y_i]:
        new_e = EditEntry(op='equal', x_i=new_e_x_i, y_i=new_e_y_i)
        v[0] = VEntry(reached_x=new_e.x_i, edit=new_e, parent=v[0])
        new_e_x_i += 1
        new_e_y_i += 1

    found = False
    found_trace = None

    for d in range(1, max_d + 1):
        for k in range(-d, d + 1, 2):

            if k == -d:
                kp = k + 1
                last_e = v[kp].edit
                new_e = EditEntry(op='insert', x_i=last_e.x_i, y_i=last_e.y_i + 1)
                v[k] = VEntry(
                    reached_x=new_e.x_i,
                    edit=new_e,
                    parent=v[kp],
                )
            elif k == d:
                k_m = k - 1
                last_e = v[k_m].edit
                new_e = EditEntry(op='delete', x_i=last_e.x_i + 1, y_i=last_e.y_i)
                v[k] = VEntry(
                    reached_x=new_e.x_i,
                    edit=new_e,
                    parent=v[k_m],
                )
            else:
                if v[k - 1].reached_x >= v[k + 1].reached_x:
                    last_e = v[k - 1].edit
                    new_e = EditEntry(op='delete', x_i=last_e.x_i + 1, y_i=last_e.y_i)
                    v[k] = VEntry(
                        reached_x=new_e.x_i,
                        edit=new_e,
                        parent=v[k - 1],
                    )
                else:
                    last_e = v[k + 1].edit
                    new_e = EditEntry(op='insert', x_i=last_e.x_i, y_i=last_e.y_i + 1)
                    v[k] = VEntry(
                        reached_x=new_e.x_i,
                        edit=new_e,
                        parent=v[k + 1],
                    )

            new_e_x_i = new_e.x_i + 1
            new_e_y_i = new_e.y_i + 1
            while new_e_x_i < l_x and new_e_y_i < l_y and x[new_e_x_i] == y[new_e_y_i]:
                new_e = EditEntry(op='equal', x_i=new_e_x_i, y_i=new_e_y_i)
                v[k] = VEntry(reached_x=new_e.x_i, edit=new_e, parent=v[k])
                new_e_x_i += 1
                new_e_y_i += 1

            if new_e_x_i >= l_x and new_e_y_i >= l_y:
                # Reached the end
                found_trace = v[k]
                found = True
                break

        if found:
            break

    assert found_trace is not None, "Myers diff failed to find a path"
    edits = _collect_edits(found_trace)
    return edits


def myers_lcs(
    x: List[int],
    y: List[int],
) -> List[AlignedChunk]:
    """Compute LCS match indices via Myers diff using compact V-array snapshots (from x to y).

    Instead of building a linked list of VEntry/EditEntry dataclass objects
    (which creates ~2 Python objects per snake step), this implementation
    stores only the V array (diagonal k -> furthest-reaching x) as a plain
    dict snapshot per edit-distance round.  Back-tracking walks the snapshots
    in reverse and directly emits (x_idx, y_idx) pairs for equal (snake)
    positions, completely bypassing the intermediate EditEntry list.

    Time complexity: O((N+M)*D) where D = edit distance.
    Space complexity: O(D^2) for V snapshots (D rounds * O(D) entries each).
                      For D << N this is negligible.

    Args:
        x: First sequence (orig token IDs).
        y: Second sequence (reenc token IDs).

    Returns:
        List of AlignedBlock covering both sequences completely.
        Match blocks correspond to LCS segments; mismatch blocks fill gaps.
    """
    l_x, l_y = len(x), len(y)

    if l_x == 0 or l_y == 0:
        if l_x == 0 and l_y == 0:
            return []
        return [AlignedChunk(match=False, x_start=0, x_end=l_x, y_start=0, y_end=l_y)]

    # ------------------------------------------------------------------ #
    # Forward pass: compute V snapshots                                    #
    # ------------------------------------------------------------------ #
    # v[k] = furthest x reached on diagonal k  (y = x - k).
    # We snapshot v *before* each round so that back-tracking can replay
    # the decisions.

    v: dict[int, int] = {}

    # d=0: start at the virtual point (-1,-1) on diagonal 0 and extend snake.
    sx = 0
    while sx < l_x and sx < l_y and x[sx] == y[sx]:
        sx += 1
    v[0] = sx

    # Fast path: sequences are identical.
    if sx >= l_x and sx >= l_y:
        return [AlignedChunk(match=True, x_start=0, x_end=l_x, y_start=0, y_end=l_y)]

    # v_history[i] stores the V dict *after* round i (i=0 means after d=0).
    # We need the state "before round d" for back-tracking, which is
    # v_history[d-1].
    v_history: list[dict[int, int]] = [dict(v)]  # index 0 = after d=0

    max_d = l_x + l_y
    final_d = -1
    final_k = 0

    for d in range(1, max_d + 1):
        # Take a snapshot of v that this round's decisions read from.
        # (We already have it as v_history[d-1].)

        for k in range(-d, d + 1, 2):
            # Pick the better predecessor diagonal.
            if k == -d:
                # Must come from k+1 (insert: y moves, x stays)
                cur_x = v.get(k + 1, -1)
            elif k == d:
                # Must come from k-1 (delete: x moves)
                cur_x = v.get(k - 1, -1) + 1
            else:
                x_del = v.get(k - 1, -1) + 1  # delete
                x_ins = v.get(k + 1, -1)  # insert
                cur_x = x_del if x_del >= x_ins else x_ins

            # Extend snake along diagonal k.
            cur_y = cur_x - k
            while cur_x < l_x and cur_y < l_y and x[cur_x] == y[cur_y]:
                cur_x += 1
                cur_y += 1

            v[k] = cur_x

            if cur_x >= l_x and cur_y >= l_y:
                final_d = d
                final_k = k
                break

        # Snapshot v *after* this round (even if we found the end, we need
        # the snapshot for back-tracking the last round).
        v_history.append(dict(v))

        if final_d >= 0:
            break

    assert final_d >= 0, "Myers diff failed to find a path"

    # ------------------------------------------------------------------ #
    # Backward pass: trace V snapshots → collect snake (equal) segments    #
    # ------------------------------------------------------------------ #
    # Each snake segment is stored as (x_start, y_start, length).
    # We collect them in reverse order, then reverse at the end.
    segments_rev: list[tuple[int, int, int]] = []

    k = final_k
    for d in range(final_d, 0, -1):
        # v_after_prev = V state after round d-1 = state read by round d.
        v_after_prev = v_history[d - 1]

        # Determine which diagonal we came from.
        if k == -d:
            prev_k = k + 1  # insert
        elif k == d:
            prev_k = k - 1  # delete
        else:
            x_del = v_after_prev.get(k - 1, -1)
            x_ins = v_after_prev.get(k + 1, -1)
            prev_k = k - 1 if x_del >= x_ins else k + 1

        # Where the predecessor ended (before the edit step of round d).
        prev_end_x = v_after_prev[prev_k]

        # After the single edit step we land at the start of the snake:
        if prev_k == k - 1:
            # delete: x advances by 1, y stays
            snake_start_x = prev_end_x + 1
        else:
            # insert: y advances by 1, x stays
            snake_start_x = prev_end_x
        snake_start_y = snake_start_x - k

        # Where this round's snake ended.
        snake_end_x = v_history[d][k]

        snake_len = snake_end_x - snake_start_x
        if snake_len > 0:
            segments_rev.append((snake_start_x, snake_start_y, snake_len))

        k = prev_k

    # d=0 initial snake (on diagonal 0, starting from (0,0)).
    init_snake_end_x = v_history[0].get(0, 0)
    if init_snake_end_x > 0:
        segments_rev.append((0, 0, init_snake_end_x))

    # ------------------------------------------------------------------ #
    # Build AlignedBlock list from segments (reversed → ascending order)   #
    # ------------------------------------------------------------------ #
    # segments_rev is in reverse order; reverse to get ascending.
    # We interleave match blocks with mismatch gaps.
    segments = list(reversed(segments_rev))

    blocks: List[AlignedChunk] = []
    xi_prev, yi_prev = 0, 0  # next unprocessed position

    for sx, sy, slen in segments:
        # Mismatch gap before this match segment
        if sx > xi_prev or sy > yi_prev:
            blocks.append(
                AlignedChunk(
                    match=False,
                    x_start=xi_prev,
                    x_end=sx,
                    y_start=yi_prev,
                    y_end=sy,
                )
            )
        # Match segment
        blocks.append(
            AlignedChunk(
                match=True,
                x_start=sx,
                x_end=sx + slen,
                y_start=sy,
                y_end=sy + slen,
            )
        )
        xi_prev = sx + slen
        yi_prev = sy + slen

    # Trailing mismatch
    if xi_prev < l_x or yi_prev < l_y:
        blocks.append(
            AlignedChunk(
                match=False,
                x_start=xi_prev,
                x_end=l_x,
                y_start=yi_prev,
                y_end=l_y,
            )
        )

    return blocks


def lcs_match(
    x: List[int],
    y: List[int],
) -> List[AlignedChunk]:
    """Compute LCS-based alignment and return a list of aligned blocks.

    This is the primary public interface. It delegates to ``_myers_lcs``
    which uses compact V-array snapshots (no dataclass allocation on the
    hot path).

    Time complexity: O((N+M)*D) where D = edit distance (number of diffs).
    Space complexity: O(D^2) for V snapshots.

    Args:
        x: Original token ID sequence.
        y: Re-encoded token ID sequence.

    Returns:
        List of AlignedBlock covering both sequences end-to-end.
        Match blocks (match=True) are LCS segments; mismatch blocks
        (match=False) are diff regions.
    """
    return myers_lcs(x, y)


def redistribute_rewards(
    orig_ids: List[int],
    reenc_ids: List[int],
    rewards: torch.Tensor,
) -> torch.Tensor:
    """Redistribute per-token rewards from reencoded sequence to original sequence.

    Algorithm:
      1. Run LCS to find matched positions between orig_ids and reenc_ids.
      2. Walk through both sequences. Between consecutive LCS anchors, there are
         "gaps" (diff regions) in both sequences.
      3. For each gap:
         - Collect rewards from the reencoded gap.
         - Mean-pool them.
         - Broadcast the mean to all orig positions in the gap.
      4. For LCS-matched positions: copy the reward directly.

    Args:
        orig_ids: Original token IDs from rollout. Length N.
        reenc_ids: Re-encoded token IDs. Length M.
        rewards: Per-token rewards for reenc_ids. Shape (M,).

    Returns:
        Per-token rewards for orig_ids. Shape (N,).
    """
    assert len(reenc_ids) == rewards.shape[0], (
        f"rewards length {rewards.shape[0]} != reenc_ids length {len(reenc_ids)}"
    )

    n = len(orig_ids)
    if n == 0:
        return torch.zeros(0, dtype=rewards.dtype, device=rewards.device)

    blocks = lcs_match(orig_ids, reenc_ids)
    result = torch.zeros(n, dtype=rewards.dtype, device=rewards.device)

    for blk in blocks:
        if blk.match:
            # LCS match: copy rewards element-wise.
            result[blk.x_start:blk.x_end] = rewards[blk.y_start:blk.y_end]
        else:
            # Diff region: mean-pool reenc rewards, broadcast to orig.
            # 注：
            # 1. mismatch 正常不可能出现在尾部（EOS），出现的话一定是长度超长的情况。这种情况的 reward 可能不用太纠结，
            # 因为本来就不太正常。
            # 2. 正常情况下，如果是 token id encoding / decoding，不可能出现 mismatch 部分长度为 0 的情况，
            # 一定是两边长度都 > 1；

            if blk.x_start < blk.x_end:
                gap_rewards = rewards[blk.y_start:blk.y_end]
                if gap_rewards.numel() > 0:
                    mean_reward = gap_rewards.mean()
                else:
                    mean_reward = 0.0
                result[blk.x_start:blk.x_end] = mean_reward

    return result


# ====================================================================== #
# Multiprocess batch API (torch.multiprocessing + shared memory)          #
# ====================================================================== #


def redistribute_rewards_batch(
    orig_ids_list: List[List[int]],
    reenc_ids_list: List[List[int]],
    rewards_list: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Synchronous batch version of :func:`redistribute_rewards`.

    Semantically equivalent to calling ``redistribute_rewards`` in a loop.
    Provided as a convenience for callers that operate on batches.

    Args:
        orig_ids_list: List of original token ID sequences.
        reenc_ids_list: List of re-encoded token ID sequences.
        rewards_list: List of per-token reward tensors (one per sample).

    Returns:
        List of redistributed reward tensors (one per sample).
    """
    assert len(orig_ids_list) == len(reenc_ids_list) == len(rewards_list), (
        "All input lists must have the same length"
    )
    return [
        redistribute_rewards(orig, reenc, rew)
        for orig, reenc, rew in zip(orig_ids_list, reenc_ids_list, rewards_list)
    ]


def _apply_blocks_to_rewards(
    blocks: List[AlignedChunk],
    orig_len: int,
    rewards: torch.Tensor,
) -> torch.Tensor:
    """Apply pre-computed LCS alignment blocks to redistribute rewards.

    This is the reward-copy step separated from LCS computation so that
    it can run in the main process (avoiding torch tensor ops in forked
    children).

    Args:
        blocks: Alignment blocks from :func:`lcs_match`.
        orig_len: Length of the original token ID sequence.
        rewards: Per-token rewards for the re-encoded sequence.

    Returns:
        Per-token rewards redistributed to the original sequence.
    """
    result = torch.zeros(orig_len, dtype=rewards.dtype, device=rewards.device)
    for blk in blocks:
        if blk.match:
            result[blk.x_start:blk.x_end] = rewards[blk.y_start:blk.y_end]
        else:
            if blk.x_start < blk.x_end:
                gap_rewards = rewards[blk.y_start:blk.y_end]
                if gap_rewards.numel() > 0:
                    mean_reward = gap_rewards.mean()
                else:
                    mean_reward = 0.0
                result[blk.x_start:blk.x_end] = mean_reward
    return result


def _run_lcs_in_pool(
    orig_ids_list: List[List[int]],
    reenc_ids_list: List[List[int]],
    max_workers: int,
) -> List[List[AlignedChunk]]:
    """Run LCS alignment in a fork-based process pool (blocking).

    Separated so it can be offloaded to a thread via
    ``loop.run_in_executor`` without blocking the async event loop.
    """
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=max_workers) as p:
        return p.starmap(lcs_match, zip(orig_ids_list, reenc_ids_list))


async def redistribute_rewards_batch_mp(
    orig_ids_list: List[List[int]],
    reenc_ids_list: List[List[int]],
    rewards_list: List[torch.Tensor],
    max_workers: int = 4,
) -> List[torch.Tensor]:
    """Async batch reward redistribution with LCS computed in forked workers.

    The LCS alignment (CPU-bound, pure Python list operations) is
    offloaded to a fork-based multiprocessing pool for maximum
    throughput.  The blocking ``Pool.starmap`` call is wrapped with
    ``loop.run_in_executor`` so it does not block the async event loop.
    The reward tensor copy/redistribution stays in the main process to
    avoid torch tensor deadlocks in forked children
    (see https://github.com/pytorch/pytorch/issues/17199 and https://github.com/pytorch/pytorch/issues/2245).

    Data flow:
      1. Fork worker pool computes ``lcs_match(orig, reenc)`` for each
         sample in parallel → returns ``List[List[AlignedChunk]]``.
         The blocking pool call is awaited via ``run_in_executor``.
      2. Main process iterates over the alignment blocks and applies
         reward redistribution (tensor ops) sequentially.

    Args:
        orig_ids_list: List of original token ID sequences.
        reenc_ids_list: List of re-encoded token ID sequences.
        rewards_list: List of per-token reward tensors (one per sample).
        max_workers: Number of worker processes.

    Returns:
        List of redistributed reward tensors (one per sample), on the
        same device / dtype as the corresponding input rewards.
    """
    assert len(orig_ids_list) == len(reenc_ids_list) == len(rewards_list), (
        "All input lists must have the same length"
    )
    assert all(r.is_cpu for r in rewards_list), "All rewards must be on CPU"
    n = len(orig_ids_list)
    if n == 0:
        return []

    # Step 1: Run LCS alignment in forked workers (pure Python, no torch).
    # Use run_in_executor to avoid blocking the event loop.
    loop = asyncio.get_running_loop()
    all_blocks: List[List[AlignedChunk]] = await loop.run_in_executor(
        None, _run_lcs_in_pool, orig_ids_list, reenc_ids_list, max_workers
    )

    # Step 2: Apply reward redistribution in main process (torch tensor ops).
    results: List[torch.Tensor] = []
    for i in range(n):
        assert len(reenc_ids_list[i]) == rewards_list[i].shape[0], (
            f"rewards length {rewards_list[i].shape[0]} != "
            f"reenc_ids length {len(reenc_ids_list[i])} at index {i}"
        )
        results.append(
            _apply_blocks_to_rewards(all_blocks[i], len(orig_ids_list[i]), rewards_list[i])
        )

    return results
