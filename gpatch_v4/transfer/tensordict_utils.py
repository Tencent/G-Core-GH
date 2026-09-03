"""Conversions between dictionary payloads and TensorDict objects."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict

__all__ = [
    "TqPayloadType",
    "data_to_tensordict",
    "dict_to_tensordict",
    "tensordict_to_data",
    "tensordict_to_dict",
]


class TqPayloadType(str, Enum):
    """Payload shapes supported by the TransferQueue codec."""

    DICT = "dict"


def _numel_of_shape(shape: Tuple[int, ...]) -> int:
    numel = 1
    for size in shape:
        numel *= int(size)
    return numel


def _encode_array_value(value: Any, ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    # Text-only samples use None for vision fields; keep them in metadata and
    # skip their contribution to the cat blob (image / text mix).
    if value is None:
        return None, {
            "shape": None,
            "is_numpy": False,
            "is_none": True,
        }

    if isinstance(value, torch.Tensor):
        assert value.layout == torch.strided
        tensor = value
        is_numpy = False
    else:
        try:
            tensor = torch.from_numpy(value)
        except (TypeError, ValueError):
            try:
                tensor = torch.from_numpy(np.ascontiguousarray(value))
            except (TypeError, ValueError):
                assert False, f"unsupported NumPy array dtype: {value.dtype}"
        is_numpy = True

    return tensor.reshape(-1), {
        "shape": tuple(value.shape),
        "is_numpy": is_numpy,
        "is_none": False,
    }


def _encode_array_field(values: List[Any], ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    tensors = []
    metadata = []
    for value in values:
        flat, value_metadata = _encode_array_value(value)
        metadata.append(value_metadata)
        if flat is None:
            continue
        tensors.append(flat)

    if not tensors:
        # All-None field: empty float blob; decode restores Nones from metadata.
        encoded = torch.empty(0, dtype=torch.float32).unsqueeze(0)
        return encoded, metadata

    first = tensors[0]
    for tensor in tensors[1:]:
        assert tensor.dtype == first.dtype
        assert tensor.device == first.device

    # One contiguous blob for the whole field; TensorDict batch dim stays 1.
    encoded = torch.cat(tensors, dim=0).unsqueeze(0)
    return encoded, metadata


def _flatten_encoded_field(
    value: torch.Tensor,
    field: str,
) -> torch.Tensor:
    assert isinstance(
        value, torch.Tensor
    ), f"TensorDict field {field!r} must be a tensor, got {type(value)}"
    if value.is_nested:
        # Mooncake may keep one cat blob or per-sample nested rows for ragged
        # vision (image / text-only mix). Concatenate all rows either way.
        rows = [row.reshape(-1) for row in value.unbind()]
        if not rows:
            flat = value.new_empty(0)
        elif len(rows) == 1:
            flat = rows[0]
        else:
            flat = torch.cat(rows, dim=0)
    else:
        flat = value.reshape(-1)
    return flat


def _decode_array_value(
    flat: torch.Tensor,
    value_metadata: Dict[str, Any],
    field: str,
    offset: int,
) -> Tuple[Any, int]:
    if value_metadata.get("is_none", False):
        return None, offset

    shape = tuple(value_metadata["shape"])
    numel = _numel_of_shape(shape)
    piece = flat[offset:offset + numel].reshape(shape)
    offset += numel
    if value_metadata["is_numpy"]:
        assert piece.device.type == "cpu", (f"NumPy field {field!r} was restored on {piece.device}")
        piece = piece.numpy()
    return piece, offset


def _decode_array_field(
    value: torch.Tensor,
    field_metadata: List[Dict[str, Any]],
    field: str,
) -> List[Any]:
    flat = _flatten_encoded_field(value, field)
    restored = []
    offset = 0
    for value_metadata in field_metadata:
        piece, offset = _decode_array_value(
            flat,
            value_metadata,
            field,
            offset,
        )
        restored.append(piece)

    assert offset == flat.numel(), (
        f"TensorDict field {field!r} cat length {flat.numel()} "
        f"does not match metadata numels {offset}"
    )
    return restored


def _can_encode_array_values(values: List[Any]) -> bool:
    """Whether a value list can be cat-encoded into TensorDict."""
    if not values:
        return False
    if not all(value is None or isinstance(value, (torch.Tensor, np.ndarray)) for value in values):
        return False
    # Keep all-None fields in fallback.
    return any(isinstance(value, (torch.Tensor, np.ndarray)) for value in values)


def dict_to_tensordict(
    data: Dict[str, Any],
    fields: Optional[List[str]] = None,
) -> Tuple[TensorDict, Dict[str, Any]]:
    """Split a dict into TensorDict-compatible and fallback fields.

    Accepts either:
    - plain dict values (single tensor / ndarray), or
    - field-major dict of lists (list of tensor / ndarray / None).

    Encoding always goes through ``_encode_array_field``.

    Parameters
    ----------
    data : Dict[str, Any]
        Plain dict, or field-major dict of lists.
    fields : List[str], optional
        Fields allowed to move into the TensorDict.

    Returns
    -------
    Tuple[TensorDict, Dict[str, Any]]
        TensorDict-compatible fields and remaining / metadata fields.
    """
    assert isinstance(data, dict), f"data must be a dict, got {type(data)}"
    selected_fields = set(data) if fields is None else set(fields)

    # Dict-of-lists: every value is a list with the same length.
    list_mode = bool(data) and all(isinstance(value, list) for value in data.values())
    sample_count = 0
    if list_mode:
        sample_count = len(next(iter(data.values())))
        for field, values in data.items():
            assert len(values) == sample_count, (
                f"data[{field!r}] has length {len(values)}, "
                f"expected {sample_count}"
            )

    encoded: Dict[str, Any] = {}
    fallback: Dict[str, Any] = {}
    for field, value in data.items():
        if field not in selected_fields:
            fallback[field] = value
            continue

        if isinstance(value, list):
            values = value
            list_meta = True
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            values = [value]
            list_meta = False
        else:
            fallback[field] = value
            continue

        if not _can_encode_array_values(values):
            fallback[field] = value
            continue

        encoded[field], field_metadata = _encode_array_field(values)
        # List inputs keep list metadata; scalar inputs keep a single dict.
        fallback[field] = field_metadata if list_meta else field_metadata[0]

    if list_mode:
        tensor_batch_size = 0 if sample_count == 0 else 1
    else:
        tensor_batch_size = 0 if not data else 1
    return TensorDict(encoded, batch_size=[tensor_batch_size]), fallback


def tensordict_to_dict(
    tensor_dict: TensorDict,
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge a TensorDict and fallback into a dict.

    Metadata in ``fallback`` may be either:
    - a single dict (plain-dict encode), or
    - a list of dicts (dict-of-lists encode).
    """
    assert isinstance(
        tensor_dict, TensorDict
    ), f"tensor_dict must be a TensorDict, got {type(tensor_dict)}"
    assert tensor_dict.batch_dims == 1, (
        "tensor_dict must have exactly one batch dimension, "
        f"got batch_size={tensor_dict.batch_size}"
    )
    assert int(tensor_dict.batch_size[0]) in (0, 1), (
        "tensor_dict batch size must be 0 or 1, "
        f"got batch_size={tensor_dict.batch_size}"
    )

    data: Dict[str, Any] = {} if fallback is None else dict(fallback)
    if not data:
        assert int(tensor_dict.batch_size[0]) == 0, (
            "empty fallback requires empty TensorDict batch_size=[0], "
            f"got batch_size={tensor_dict.batch_size}"
        )
        assert len(tensor_dict.keys()) == 0
        return data

    # When fallback is dict-of-lists style, non-encoded fields are lists of equal
    # length; encoded fields store list metadata of that same length.
    list_lengths = [
        len(values) for values in data.values() if isinstance(values, list)
    ]
    sample_count = list_lengths[0] if list_lengths else None
    if sample_count is not None:
        for field, values in data.items():
            if isinstance(values, list):
                assert len(values) == sample_count, (
                    f"fallback[{field!r}] has length {len(values)}, "
                    f"expected {sample_count}"
                )

    for field, value in tensor_dict.items():
        field_metadata = data.pop(field, None)
        assert field_metadata is not None, (f"fallback is missing metadata for field {field!r}")
        if isinstance(field_metadata, list):
            if sample_count is not None:
                assert len(field_metadata) == sample_count, (
                    f"fallback metadata for {field!r} has length "
                    f"{len(field_metadata)}, expected {sample_count}"
                )
            data[field] = _decode_array_field(value, field_metadata, field)
            continue

        assert isinstance(field_metadata, dict), (
            f"fallback metadata for {field!r} must be a dict or list, "
            f"got {type(field_metadata)}"
        )
        flat = _flatten_encoded_field(value, field)
        restored, offset = _decode_array_value(
            flat,
            field_metadata,
            field,
            0,
        )
        assert offset == flat.numel(), (
            f"TensorDict field {field!r} cat length {flat.numel()} "
            f"does not match metadata numels {offset}"
        )
        data[field] = restored

    return data


def data_to_tensordict(
    data: Dict[str, Any],
    payload_type: TqPayloadType,
    fields: Optional[List[str]] = None,
) -> Tuple[TensorDict, Dict[str, Any]]:
    """Encode a supported payload into a TensorDict and inline fallback."""
    if payload_type is TqPayloadType.DICT:
        return dict_to_tensordict(data, fields)
    raise ValueError(f"Unsupported TQ payload type: {payload_type!r}")


def tensordict_to_data(
    tensor_dict: TensorDict,
    fallback: Optional[Dict[str, Any]],
    payload_type: TqPayloadType,
) -> Dict[str, Any]:
    """Decode a TensorDict using its recorded payload shape."""
    if payload_type is TqPayloadType.DICT:
        return tensordict_to_dict(tensor_dict, fallback)
    raise ValueError(f"Unsupported TQ payload type: {payload_type!r}")
