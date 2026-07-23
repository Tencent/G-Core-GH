import heapq

import torch
from torch import distributed as dist


def karmarkar_karp(seqlen_list: list[int], k_partitions: int, equal_size: bool) -> list[list[int]]:
    """Partition items into k groups using the Karmarkar-Karp differencing method.

    LDM heuristic for balanced multi-way number partitioning: iteratively
    combine the sets with the largest difference.

    Args:
        seqlen_list: Values to partition (typically sequence lengths).
        k_partitions: Number of partitions.
        equal_size: If *True*, each partition has exactly
            ``len(seqlen_list) / k_partitions`` items.

    Returns:
        list[list[int]]: ``k`` partitions of indices into ``seqlen_list``.

    See Also:
        https://en.wikipedia.org/wiki/Largest_differencing_method

    Note:
        ``equal_size=True`` requires ``len(seqlen_list)`` divisible by ``k_partitions``.
    """

    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )
    return partitions


def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    """Balance sum of sequence lengths across k partitions of indices.

    Uses the Karmarkar-Karp differencing method.

    Args:
        seqlen_list (List[int]): Sequence lengths.
        k_partitions (int):
        equal_size (bool): If *True*, every partition has the same number
            of items (requires ``len(seqlen_list) % k_partitions == 0``).

    Returns:
        List[List[int]]: ``k_partitions`` lists of original indices,
        sorted within each partition.

    Raises:
        AssertionError: If ``len(seqlen_list) < k_partitions``,
            ``equal_size`` violation, or any partition is empty.
    """
    assert len(
        seqlen_list
    ) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    partitions = karmarkar_karp(
        seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size
    )
    return _check_and_sort_partitions(partitions)


def convert_mbs_for_pack_seq(
    samples: list[dict],
    max_token_len: int,
    pad_each_doc_to_multi_of: int = 128,
    dp_group=None,
    cp_size: int = 1,
) -> list[list[dict]]:
    """Group variable-length samples into micro-batches by token budget.

    Each sample's padded length (rounded up to ``pad_each_doc_to_multi_of``) is
    used to estimate the packed token count. Samples are partitioned with
    Karmarkar-Karp so micro-batches on the same rank stay load-balanced.
    ``num_mb`` starts at ``ceil(total_padded / max_token_len)`` and is raised
    until every KK group fits the hard ceiling after the same CP tail pad that
    ``pack_sequences`` applies (``T % (cp_size * pad) == 0``). ``num_mb`` is
    then aligned across DP ranks via ``all_reduce(MAX)`` and KK is re-run so
    EP / CP collectives stay in lockstep without dropping balance.

    Example::

        # 8 samples with seq_length [120, 500, 300, 80, 450, 200, 350, 90],
        # pad_each_doc_to_multi_of=128 → padded [128, 512, 384, 128, 512, 256, 384, 128]
        # total_padded = 2432, max_token_len = 1024
        # num_mb = ceil(2432 / 1024) = 3
        # KK partitions into 3 balanced groups:
        #   group 0: [512, 256, 128]  → 896 tokens
        #   group 1: [512, 128, 128]  → 768 tokens
        #   group 2: [384, 384]       → 768 tokens

        # Capacity-safe raise: [640, 640, 640] with budget 1024 needs num_mb=3
        # because KK with k=2 can still place [640, 640] in one group.

    Parameters
    ----------
    samples : list[dict]
        Each dict MUST contain ``"sequence_lengths"`` (int or 0-d tensor).
    max_token_len : int
        Max packed token count per micro-batch (typically ``seq_length``).
    pad_each_doc_to_multi_of : int
        Per-segment padding granularity (DSV4 HCA requires 128).
        表示特定模型的输入要 pad 到某个大小，但不是整体的 pad 大小。比如 cp 要求 pad 到 1024，但 dsv4 只要 pad 到 128，
        避免 pack seq 的时候估计不准确。
    dp_group : dist.ProcessGroup | None
        Data-parallel group for cross-rank alignment.
    cp_size : int
        Context-parallel world size. Matches ``pack_sequences`` alignment of
        ``T`` to ``cp_size * pad_each_doc_to_multi_of``.

    Returns
    -------
    list[list[dict]]
        ``num_mb`` groups; each group is a non-empty list of sample dicts.
    """
    n = len(samples)
    assert n > 0, "samples must be non-empty"
    assert cp_size >= 1, f"cp_size must be >= 1, got {cp_size}"
    assert pad_each_doc_to_multi_of > 0, (
        f"pad_each_doc_to_multi_of must be positive, got {pad_each_doc_to_multi_of}"
    )

    # Mirror pack_sequences: per-doc pad, then round the packed total up to
    # cp_size * pad_each_doc_to_multi_of (extra lands on the last segment).
    total_align = cp_size * pad_each_doc_to_multi_of

    def _padded_len(s):
        raw = s["sequence_lengths"]
        raw = int(raw) if isinstance(raw, int) else raw.item()
        return (
            (raw + pad_each_doc_to_multi_of - 1) // pad_each_doc_to_multi_of
        ) * pad_each_doc_to_multi_of

    def _aligned_total(load: int) -> int:
        rem = load % total_align
        return load if rem == 0 else load + (total_align - rem)

    def _kk_partitions_within_budget(k: int):
        partitions = get_seqlen_balanced_partitions(
            seqlen_list=padded_lens,
            k_partitions=k,
            equal_size=False,
        )
        for part in partitions:
            load = sum(padded_lens[i] for i in part)
            if _aligned_total(load) > max_token_len:
                return None
        return partitions

    padded_lens = [_padded_len(s) for s in samples]
    total_padded = sum(padded_lens)

    assert max_token_len >= max(_aligned_total(L) for L in padded_lens), (
        f"max_token_len ({max_token_len}) < longest CP-aligned sample "
        f"({max(_aligned_total(L) for L in padded_lens)}); samples should be "
        f"truncated to seq_length before grouping"
    )

    # Lower bound from average fill; raise k while KK still overflows the
    # hard ceiling (balance does not imply capacity, e.g. [640, 640, 640]).
    num_mb = max(1, (total_padded + max_token_len - 1) // max_token_len)
    partitions = None
    while num_mb <= n:
        partitions = _kk_partitions_within_budget(num_mb)
        if partitions is not None:
            break
        num_mb += 1
    assert partitions is not None, (
        f"unable to KK-partition {n} samples into capacity-safe micro-batches "
        f"(max_token_len={max_token_len}, cp_size={cp_size})"
    )

    if dp_group is not None and dist.is_initialized():
        mb_t = torch.tensor([num_mb], dtype=torch.long, device="cuda")
        dist.all_reduce(mb_t, op=dist.ReduceOp.MAX, group=dp_group)
        target_num_mb = int(mb_t.item())
        assert target_num_mb <= n, (
            f"DP pack-seq alignment needs {target_num_mb} micro-batches, but this "
            f"rank only has {n} samples; EP/CP would deadlock if ranks diverge"
        )
        if target_num_mb != num_mb:
            num_mb = target_num_mb
            partitions = _kk_partitions_within_budget(num_mb)
            assert partitions is not None, ("unable to KK-partition after DP micro-batch alignment")

    return [[samples[i] for i in part] for part in partitions]
