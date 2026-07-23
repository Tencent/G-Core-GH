from gpatch_v4.core.seqlen_balancing import convert_mbs_for_pack_seq


def _samples(lengths):
    return [
        {"id": sample_id, "sequence_lengths": length}
        for sample_id, length in enumerate(lengths)
    ]


def _group_loads(groups, pad_each_doc_to_multi_of=128):
    def rounded(length):
        return (
            (length + pad_each_doc_to_multi_of - 1) // pad_each_doc_to_multi_of
        ) * pad_each_doc_to_multi_of

    return [
        sum(rounded(sample["sequence_lengths"]) for sample in group)
        for group in groups
    ]


def test_pack_seq_token_budget_is_a_hard_ceiling():
    samples = _samples([640, 640, 640])

    groups = convert_mbs_for_pack_seq(
        samples,
        max_token_len=1024,
        pad_each_doc_to_multi_of=128,
    )

    assert len(groups) == 3
    assert all(load <= 1024 for load in _group_loads(groups))
    assert sorted(sample["id"] for group in groups for sample in group) == [0, 1, 2]


def test_pack_seq_groups_rounded_lengths_without_losing_order_identity():
    samples = _samples([120, 500, 300, 80, 450, 200, 350, 90])

    groups = convert_mbs_for_pack_seq(
        samples,
        max_token_len=1024,
        pad_each_doc_to_multi_of=128,
    )

    loads = _group_loads(groups)
    assert all(load <= 1024 for load in loads)
    assert sorted(sample["id"] for group in groups for sample in group) == list(
        range(len(samples))
    )
    # KK keeps same-rank micro-batches much closer than greedy fill.
    assert max(loads) - min(loads) <= 256


def test_pack_seq_budget_includes_cp_tail_alignment():
    """pack_sequences rounds T up to cp_size * pad; budgeting must match."""
    # pad=4 → each sample contributes 4. With cp=2, total_align=8.
    # Two samples → raw load 8 (aligned). Three samples → raw 12 → aligned 16.
    samples = _samples([4, 4, 4])

    groups = convert_mbs_for_pack_seq(
        samples,
        max_token_len=8,
        pad_each_doc_to_multi_of=4,
        cp_size=2,
    )

    assert len(groups) == 2
    assert sorted(len(group) for group in groups) == [1, 2]
    assert sorted(sample["id"] for group in groups for sample in group) == [0, 1, 2]

