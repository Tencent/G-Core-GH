import copy

import pytest
import torch

from gpatch_v4.utils.grpo_thd_alignment import (
    build_grpo_thd_dump_records,
    validate_grpo_thd_dump_file,
    validate_grpo_thd_dump_sample,
)


def _dumped_sample():
    return {
        "tokens": torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        "rollout_logprobs": torch.tensor([-1.0, -2.0, -3.0, 1.0]).to(torch.bfloat16),
        "pre_logprobs": torch.tensor([-1.1, -2.1, -3.1]).to(torch.bfloat16),
        "response_mask": torch.tensor([0.0, 1.0, 1.0]),
        "packed_segment_index": 0,
        "packed_segment_count": 1,
        "packed_offset": 0,
        "packed_actual_len": 3,
        "packed_padded_len": 4,
        "packed_token_ids": torch.tensor([10, 11, 12, 0]),
        "packed_target": torch.tensor([11, 12, 13, 0]),
        "packed_response_mask": torch.tensor([0.0, 1.0, 1.0, 0.0]),
        "packed_prev_logprobs": torch.tensor([-1.1, -2.1, -3.1, 0.0]),
        "packed_rollout_logprobs": torch.tensor([-1.0, -2.0, -3.0, 0.0]),
        "curr_logprobs": torch.tensor([-1.2, -2.2, -3.2, -9.0]),
    }


def test_build_dump_records_uses_padded_segment_offsets(tmp_path):
    batches = [{"tokens": torch.tensor([1, 2, 3, 4])}, {"tokens": torch.tensor([5, 6, 7])}]
    batch_data = {
        "packed_token_ids": torch.tensor([[1, 2, 3, 0, 5, 6, 0, 0]]),
        "target": torch.tensor([[2, 3, 4, 0, 6, 7, 0, 0]]),
        "mask": torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]]),
        "prev_log_probs": torch.tensor([[-1.0, -2.0, -3.0, 0.0, -4.0, -5.0, 0.0, 0.0]]),
        "rollout_log_probs": torch.tensor([[-1.1, -2.1, -3.1, 0.0, -4.1, -5.1, 0.0, 0.0]]),
        "cu_seqlens": torch.tensor([0, 3, 5]),
        "cu_seqlens_padded": torch.tensor([0, 4, 8]),
    }
    curr = torch.arange(8, dtype=torch.float32).unsqueeze(0)

    records = build_grpo_thd_dump_records(batches, batch_data, curr)

    assert len(records) == 2
    assert records[0]["packed_offset"] == 0
    assert records[1]["packed_offset"] == 4
    assert records[1]["packed_actual_len"] == 2
    assert torch.equal(records[1]["packed_token_ids"], torch.tensor([5, 6, 0, 0]))
    assert torch.equal(records[1]["packed_target"], torch.tensor([6, 7, 0, 0]))
    assert torch.equal(records[1]["curr_logprobs"], torch.tensor([4.0, 5.0, 6.0, 7.0]))

    sources = [
        {
            "tokens": torch.tensor([1, 2, 3, 4]),
            "response_mask": torch.tensor([1.0, 1.0, 1.0]),
            "pre_logprobs": torch.tensor([-1.0, -2.0, -3.0]),
            "rollout_logprobs": torch.tensor([-1.1, -2.1, -3.1, 99.0]),
        },
        {
            "tokens": torch.tensor([5, 6, 7]),
            "response_mask": torch.tensor([1.0, 1.0]),
            "pre_logprobs": torch.tensor([-4.0, -5.0]),
            "rollout_logprobs": torch.tensor([-4.1, -5.1, 99.0]),
        },
    ]
    for record, source in zip(records, sources, strict=True):
        record.update(source)
    path = tmp_path / "multi_segment.pt"
    torch.save(records, path)

    stats = validate_grpo_thd_dump_file(path)
    assert [item["segment_index"] for item in stats] == [0.0, 1.0]
    assert [item["segment_count"] for item in stats] == [2.0, 2.0]
    assert [item["packed_offset"] for item in stats] == [0.0, 4.0]


def test_dump_file_roundtrip_validates_exact_mapping(tmp_path):
    path = tmp_path / "thd_dump.pt"
    torch.save([_dumped_sample()], path)

    stats = validate_grpo_thd_dump_file(path)

    assert len(stats) == 1
    assert stats[0]["packed_offset"] == 0
    assert stats[0]["actual_len"] == 3


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("packed_token_ids", torch.tensor([11, 12, 13, 0])),
        ("packed_target", torch.tensor([10, 11, 12, 0])),
        ("packed_rollout_logprobs", torch.tensor([0.0, -1.0, -2.0, 0.0])),
        ("packed_prev_logprobs", torch.tensor([0.0, -1.1, -2.1, 0.0])),
    ],
)
def test_validator_rejects_shifted_or_wrong_segments(key, bad_value):
    sample = copy.deepcopy(_dumped_sample())
    sample[key] = bad_value

    with pytest.raises(AssertionError):
        validate_grpo_thd_dump_sample(sample)
