"""Reusable helpers for moving batched dictionaries through TransferQueue."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from gpatch_v4.transfer import get_tq_connector
from gpatch_v4.transfer.tensordict_utils import (
    TqPayloadType,
    data_to_tensordict,
    tensordict_to_data,
)
from gpatch_v4.transfer.tq_connector import TqConnector

_TQ_DATA_SLOT_FIELD = "_tq_data_slot"

__all__ = [
    "TqDataSlot",
    "TqPayloadType",
    "async_offload_to_tq",
    "async_restore_from_tq",
    "get_tq_data_slot",
    "restore_from_tq",
    "restore_tq_payloads_into_samples",
]


@dataclass(frozen=True)
class TqDataSlot:
    """Picklable reference and codec type for values stored in TransferQueue."""

    keys: Tuple[str, ...]
    payload_type: TqPayloadType


def _resolve_connector(connector: Optional[TqConnector]) -> TqConnector:
    return connector if connector is not None else get_tq_connector()


def get_tq_data_slot(batched_data: Dict[str, Any]) -> Optional[TqDataSlot]:
    """Return the typed TQ slot embedded in ``batched_data``."""
    slots = [value for value in batched_data.values() if isinstance(value, TqDataSlot)]
    assert len(slots) <= 1, f"expected at most one TqDataSlot, got {len(slots)}"
    return slots[0] if slots else None


async def async_offload_to_tq(
    data: Dict[str, Any],
    ppo_step: int,
    payload_type: TqPayloadType,
    fields: Optional[List[str]] = None,
    *,
    connector: Optional[TqConnector] = None,
) -> Dict[str, Any]:
    """Offload selected fields and embed a self-describing TQ slot."""
    connector = _resolve_connector(connector)
    tensor_dict, fallback = data_to_tensordict(
        data,
        payload_type,
        fields=fields,
    )
    if len(tensor_dict.keys()) == 0:
        return fallback

    assert _TQ_DATA_SLOT_FIELD not in fallback
    keys = tuple(connector.make_keys(
        int(tensor_dict.batch_size[0]),
        ppo_step=ppo_step,
    ))
    await connector.async_set(tensor_dict, keys, ppo_step)

    result: Dict[str, Any] = dict(fallback)
    result[_TQ_DATA_SLOT_FIELD] = TqDataSlot(
        keys=keys,
        payload_type=payload_type,
    )
    return result


async def async_restore_from_tq(
    data: Dict[str, Any],
    *,
    connector: Optional[TqConnector] = None,
) -> Dict[str, Any]:
    """Restore a payload according to the type recorded in its TQ slot."""
    connector = _resolve_connector(connector)
    slot = get_tq_data_slot(data)
    if slot is None:
        return data

    fallback = {field: value for field, value in data.items() if not isinstance(value, TqDataSlot)}
    tensor_dict = await connector.async_get(slot.keys)
    return tensordict_to_data(
        tensor_dict,
        fallback,
        slot.payload_type,
    )


def restore_from_tq(
    data: Dict[str, Any],
    *,
    connector: Optional[TqConnector] = None,
) -> Dict[str, Any]:
    """Synchronously restore a payload according to its recorded type."""
    connector = _resolve_connector(connector)
    slot = get_tq_data_slot(data)
    if slot is None:
        return data

    fallback = {field: value for field, value in data.items() if not isinstance(value, TqDataSlot)}
    tensor_dict = connector.get(slot.keys)
    return tensordict_to_data(
        tensor_dict,
        fallback,
        slot.payload_type,
    )


def restore_tq_payloads_into_samples(
    samples: List[Dict[str, Any]],
    payload_field: str,
    *,
    connector: Optional[TqConnector] = None,
) -> None:
    """Restore one complete TQ payload into each sample in-place."""
    connector = _resolve_connector(connector)
    for sample in samples:
        payload = sample[payload_field]
        assert isinstance(
            payload, dict
        ), f"sample[{payload_field!r}] must be a dict, got {type(payload)}"
        restored = restore_from_tq(payload, connector=connector)
        sample.update(restored)
