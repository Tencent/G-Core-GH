import torch

from gpatch_v4.utils.training_utils import pad_or_truncate_last_dim


def test_overlong_tensor_is_truncated_from_the_right():
    tensor = torch.tensor([0, 1, 2, 3, 4])

    result = pad_or_truncate_last_dim(tensor, 3, value=-1)

    assert result.tolist() == [0, 1, 2]


def test_short_tensor_is_right_padded():
    tensor = torch.tensor([0, 1, 2])

    result = pad_or_truncate_last_dim(tensor, 5, value=-1)

    assert result.tolist() == [0, 1, 2, -1, -1]


def test_random_padding_mode_also_truncates_from_the_right():
    tensor = torch.tensor([0, 1, 2, 3, 4])

    result = pad_or_truncate_last_dim(
        tensor,
        3,
        value=-1,
        pad_with_random_token=True,
        vocab_size=100,
    )

    assert result.tolist() == [0, 1, 2]


def test_random_padding_excludes_forbidden_token_ids():
    tensor = torch.tensor([0, 1])

    result = pad_or_truncate_last_dim(
        tensor,
        32,
        value=-1,
        pad_with_random_token=True,
        vocab_size=4,
        forbidden_token_ids=[2],
    )

    assert result.shape[-1] == 32
    assert 2 not in result[2:].tolist()
