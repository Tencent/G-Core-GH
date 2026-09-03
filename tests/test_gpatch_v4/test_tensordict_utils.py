import numpy as np
import pytest
import torch
from tensordict import TensorDict

from gpatch_v4.transfer.tensordict_utils import (
    dict_to_tensordict,
    tensordict_to_dict,
)


def test_dict_of_lists_tensordict_round_trip():
    data = {
        "dense": [torch.arange(3), torch.arange(3, 6)],
        "ragged": [torch.arange(4).reshape(2, 2), torch.arange(3)],
        "numpy": [np.arange(6).reshape(2, 3), np.arange(6)],
        "text": ["first", "second"],
        "metadata": [{"index": 0}, {"index": 1}],
        "optional": [None, None],
    }

    tensor_dict, fallback = dict_to_tensordict(data)
    restored = tensordict_to_dict(tensor_dict, fallback)

    assert tensor_dict.batch_size == torch.Size([1])
    assert tensor_dict["dense"].shape == (1, 6)
    assert not tensor_dict["ragged"].is_nested
    assert tensor_dict["ragged"].shape == (1, 7)
    assert tensor_dict["numpy"].shape == (1, 12)
    assert set(fallback) == set(data)
    assert fallback["dense"][0]["shape"] == (3,)
    assert fallback["ragged"][0]["shape"] == (2, 2)
    assert fallback["numpy"][0]["is_numpy"]
    assert fallback["metadata"] is data["metadata"]
    for field in ("dense", "ragged"):
        for expected, actual in zip(data[field], restored[field]):
            torch.testing.assert_close(actual, expected)
    for expected, actual in zip(data["numpy"], restored["numpy"]):
        assert isinstance(actual, np.ndarray)
        np.testing.assert_array_equal(actual, expected)
    for field in ("text", "metadata", "optional"):
        assert restored[field] == data[field]

    restored["dense"][0][0] = -1
    assert tensor_dict["dense"][0, 0].item() == -1


def test_dict_tensordict_round_trip_preserves_list_value():
    data = {
        "tensor": torch.arange(6).reshape(2, 3),
        "numpy": np.arange(4),
        "metadata": ["single", "dict", "value"],
    }

    tensor_dict, fallback = dict_to_tensordict(data)
    restored = tensordict_to_dict(
        tensor_dict,
        fallback,
    )

    torch.testing.assert_close(restored["tensor"], data["tensor"])
    np.testing.assert_array_equal(restored["numpy"], data["numpy"])
    assert restored["metadata"] == data["metadata"]


def test_dict_to_tensordict_keeps_non_array_fallback():
    data = {
        "tokens": [torch.arange(2), torch.arange(2)],
        "excluded": [torch.arange(3), torch.arange(3)],
        "metadata": ["first", "second"],
    }

    tensor_dict, fallback = dict_to_tensordict(
        data,
        fields=["tokens"],
    )
    restored = tensordict_to_dict(tensor_dict, fallback)

    assert list(tensor_dict.keys()) == ["tokens"]
    assert tensor_dict.batch_size == torch.Size([1])
    assert set(fallback) == set(data)
    assert fallback["tokens"][0]["shape"] == (2,)
    assert fallback["excluded"] is data["excluded"]
    assert fallback["metadata"] is data["metadata"]
    for expected, actual in zip(data["excluded"], restored["excluded"]):
        torch.testing.assert_close(actual, expected)


def test_dict_tensordict_round_trip_empty_values():
    data = {
        "tensor": [
            torch.arange(4).reshape(2, 2),
            torch.empty((0, 3), dtype=torch.int64),
        ],
        "numpy": [np.arange(3), np.empty((0, 2), dtype=np.int64)],
    }

    tensor_dict, fallback = dict_to_tensordict(data)
    restored = tensordict_to_dict(tensor_dict, fallback)

    assert tensor_dict.batch_size == torch.Size([1])
    assert tensor_dict["tensor"].shape == (1, 4)
    assert tensor_dict["numpy"].shape == (1, 3)
    for expected, actual in zip(data["tensor"], restored["tensor"]):
        torch.testing.assert_close(actual, expected)
        assert actual.shape == expected.shape
    for expected, actual in zip(data["numpy"], restored["numpy"]):
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == expected.shape


def test_tensordict_to_dict_accepts_nested_single_blob():
    data = {
        "vision_data": [
            torch.arange(4).reshape(2, 2),
            torch.arange(3),
        ],
    }
    tensor_dict, fallback = dict_to_tensordict(data)
    nested_blob = torch.nested.as_nested_tensor(
        [tensor_dict["vision_data"].reshape(-1)],
        layout=torch.jagged,
    )
    mooncake_tensor_dict = TensorDict(
        {"vision_data": nested_blob},
        batch_size=[1],
    )

    restored = tensordict_to_dict(
        mooncake_tensor_dict,
        fallback,
    )

    for expected, actual in zip(data["vision_data"], restored["vision_data"]):
        torch.testing.assert_close(actual, expected)


def test_dict_to_tensordict_rejects_inconsistent_lengths():
    data = {
        "tokens": [torch.arange(2), torch.arange(2)],
        "metadata": ["first"],
    }

    with pytest.raises(AssertionError, match="has length 1, expected 2"):
        dict_to_tensordict(data)


def test_dict_to_tensordict_mixed_none_vision():
    data = {
        "vision_data": [torch.arange(4).reshape(2, 2), None, torch.arange(3)],
        "tokens": [torch.arange(2), torch.arange(2), torch.arange(2)],
    }
    tensor_dict, fallback = dict_to_tensordict(data)
    restored = tensordict_to_dict(tensor_dict, fallback)
    assert restored["vision_data"][1] is None
    torch.testing.assert_close(restored["vision_data"][0], data["vision_data"][0])
    torch.testing.assert_close(restored["vision_data"][2], data["vision_data"][2])


def test_empty_dict_of_lists_batch_size():
    tensor_dict, fallback = dict_to_tensordict({"x": []})
    restored = tensordict_to_dict(tensor_dict, fallback)
    assert tensor_dict.batch_size == torch.Size([0])
    assert restored == {"x": []}
