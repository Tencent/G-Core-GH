import torch

from gpatch_v4.utils.training_utils import pad_or_truncate_last_dim


def test_suffix_truncation_uses_valid_content_before_existing_padding():
    tensor = torch.tensor([0, 1, 2, 3, 99, 99])

    result = pad_or_truncate_last_dim(
        tensor,
        3,
        value=-1,
        truncate_left=True,
        valid_len=4,
    )

    assert result.tolist() == [1, 2, 3]


def test_prefix_truncation_uses_valid_content_before_existing_padding():
    tensor = torch.tensor([0, 1, 2, 3, 99, 99])

    result = pad_or_truncate_last_dim(
        tensor,
        3,
        value=-1,
        truncate_left=False,
        valid_len=4,
    )

    assert result.tolist() == [0, 1, 2]


def test_valid_content_is_right_padded_when_shorter_than_target():
    tensor = torch.tensor([0, 1, 2, 3, 99, 99])

    result = pad_or_truncate_last_dim(
        tensor,
        6,
        value=-1,
        truncate_left=True,
        valid_len=4,
    )

    assert result.tolist() == [0, 1, 2, 3, -1, -1]


def test_random_padding_mode_still_suffix_truncates_overlong_content():
    tensor = torch.tensor([0, 1, 2, 3, 4, 99])

    result = pad_or_truncate_last_dim(
        tensor,
        3,
        value=-1,
        pad_with_random_token=True,
        vocab_size=100,
        truncate_left=True,
        valid_len=5,
    )

    assert result.tolist() == [2, 3, 4]
