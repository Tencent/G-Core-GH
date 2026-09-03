import asyncio
import pickle

import numpy as np
import torch

from gpatch_v4.transfer.utils import (
    TqDataSlot,
    TqPayloadType,
    async_offload_to_tq,
    async_restore_from_tq,
    get_tq_data_slot,
    restore_tq_payloads_into_samples,
)


class _FakeConnector:
    def __init__(self):
        self.data = {}

    @staticmethod
    def make_keys(count, *, ppo_step):
        return [f"step_{ppo_step}:sample_{index}" for index in range(count)]

    async def async_set(self, tensor_dict, key, ppo_step):
        del ppo_step
        self.data[tuple(key)] = tensor_dict

    async def async_get(self, key):
        return self.data[tuple(key)]

    def get(self, key):
        return self.data[tuple(key)]


def test_tq_slot_batch_round_trip():
    async def run_test():
        connector = _FakeConnector()
        batched_data = {
            "tensor": [torch.arange(4).reshape(2, 2), torch.arange(3)],
            "numpy": [np.arange(2), np.arange(4).reshape(2, 2)],
            "metadata": ["first", "second"],
        }

        payload = await async_offload_to_tq(
            batched_data,
            ppo_step=3,
            payload_type=TqPayloadType.DICT,
            connector=connector,
        )
        slot = get_tq_data_slot(payload)

        assert slot is not None
        assert pickle.loads(pickle.dumps(slot)) == slot
        assert slot.keys == ("step_3:sample_0",)
        assert slot.payload_type is TqPayloadType.DICT
        assert payload["metadata"] is batched_data["metadata"]

        restored = await async_restore_from_tq(
            payload, connector=connector
        )
        assert restored["metadata"] == batched_data["metadata"]
        for expected, actual in zip(batched_data["tensor"], restored["tensor"]):
            torch.testing.assert_close(actual, expected)
        for expected, actual in zip(batched_data["numpy"], restored["numpy"]):
            np.testing.assert_array_equal(actual, expected)

    asyncio.run(run_test())


def test_tq_slot_respects_field_selection():
    async def run_test():
        connector = _FakeConnector()
        batched_data = {
            "offloaded": [torch.arange(2), torch.arange(2)],
            "inline": [torch.arange(3), torch.arange(3)],
        }

        payload = await async_offload_to_tq(
            batched_data,
            ppo_step=4,
            payload_type=TqPayloadType.DICT,
            fields=["offloaded"],
            connector=connector,
        )

        assert isinstance(get_tq_data_slot(payload), TqDataSlot)
        assert payload["inline"] is batched_data["inline"]
        restored = await async_restore_from_tq(
            payload, connector=connector
        )
        for field in batched_data:
            for expected, actual in zip(batched_data[field], restored[field]):
                torch.testing.assert_close(actual, expected)

    asyncio.run(run_test())


def test_restore_complete_hidden_state_into_sample():
    async def build_payload():
        connector = _FakeConnector()
        hidden_state = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
        payload = await async_offload_to_tq(
            {
                "teacher_hidden_states": hidden_state,
                "metadata": ["single", "dict", "value"],
            },
            ppo_step=5,
            payload_type=TqPayloadType.DICT,
            fields=["teacher_hidden_states"],
            connector=connector,
        )
        return connector, hidden_state, payload

    connector, hidden_state, payload = asyncio.run(build_payload())
    samples = [{"teacher_hidden_states": payload}]
    restore_tq_payloads_into_samples(
        samples,
        "teacher_hidden_states",
        connector=connector,
    )

    torch.testing.assert_close(samples[0]["teacher_hidden_states"], hidden_state)
    assert samples[0]["metadata"] == ["single", "dict", "value"]
