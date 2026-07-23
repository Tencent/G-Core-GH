import types
from unittest.mock import Mock, patch

import pytest
import torch

from gpatch_v4.core.smart_pad_helper import CatedSmartPadInferHelper
from gpatch_v4.training_backend.megatron_backend.mixin import ForwardStepMixin


_MIXIN_PATH = "gpatch_v4.training_backend.megatron_backend.mixin"
_SMART_PAD_HELPER_PATH = "gpatch_v4.core.smart_pad_helper"


def _make_smart_pad_mixin():
    return types.SimpleNamespace(
        forward_only_mbs=2,
        policy_config=types.SimpleNamespace(
            dynamic_mbs_target_seqlen_fwd_only=None,
            dynamic_mbs_limit_fwd_only=None,
        ),
        training_config=types.SimpleNamespace(pad_to_mulitiple_of=8),
        _smart_pad_forward_step=Mock(),
    )


def _make_entropy_results():
    return [
        {
            "logprobs": torch.tensor([1.0, 2.0], dtype=torch.float16),
            "prev_per_token_entropy": torch.tensor([0.1, 0.2], dtype=torch.bfloat16),
        },
        {
            "logprobs": torch.tensor([3.0, 4.0], dtype=torch.float16),
            "prev_per_token_entropy": torch.tensor([0.3, 0.4], dtype=torch.bfloat16),
        },
    ]


def test_smart_pad_forward_step_splits_entropy_by_sample():
    output_func = object()
    mixin = types.SimpleNamespace(
        get_logprob_output_only_func=Mock(return_value=output_func),
        _smart_pad_current_model=object(),
    )
    batched_result = {
        "logprobs": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "prev_per_token_entropy": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    }
    forward_backward = Mock(return_value=[batched_result])

    with (
        patch(f"{_MIXIN_PATH}.get_forward_backward_func", return_value=forward_backward),
        patch(f"{_MIXIN_PATH}.mpu.is_pipeline_last_stage", return_value=True),
        patch(f"{_MIXIN_PATH}.clear_memory"),
    ):
        result = ForwardStepMixin._smart_pad_forward_step(
            mixin,
            batch_iter=iter(()),
            num_microbatches=1,
            micro_batch_size=2,
            seq_length=3,
            return_per_token_entropy=True,
        )

    mixin.get_logprob_output_only_func.assert_called_once_with(
        3,
        inference_only=True,
        return_per_token_entropy=True,
    )
    assert len(result) == 1
    assert len(result[0]) == 2
    torch.testing.assert_close(result[0][0]["logprobs"], batched_result["logprobs"][0])
    torch.testing.assert_close(
        result[0][1]["prev_per_token_entropy"],
        batched_result["prev_per_token_entropy"][1],
    )


def test_smart_pad_forward_step_preserves_tensor_output():
    output_func = object()
    mixin = types.SimpleNamespace(
        get_logprob_output_only_func=Mock(return_value=output_func),
        _smart_pad_current_model=object(),
    )
    expected = [torch.tensor([[1.0, 2.0], [3.0, 4.0]])]

    with (
        patch(
            f"{_MIXIN_PATH}.get_forward_backward_func",
            return_value=Mock(return_value=expected),
        ),
        patch(f"{_MIXIN_PATH}.mpu.is_pipeline_last_stage", return_value=True),
        patch(f"{_MIXIN_PATH}.clear_memory"),
    ):
        result = ForwardStepMixin._smart_pad_forward_step(
            mixin,
            batch_iter=iter(()),
            num_microbatches=1,
            micro_batch_size=2,
            seq_length=3,
            return_per_token_entropy=False,
        )

    assert result is expected
    mixin.get_logprob_output_only_func.assert_called_once_with(
        3,
        inference_only=True,
        return_per_token_entropy=False,
    )


def test_smart_pad_helper_restores_entropy_dict_order():
    original_order = _make_entropy_results()
    helper = CatedSmartPadInferHelper(
        [{"tokens": torch.ones(3)}, {"tokens": torch.ones(5)}],
        forward_batch_size=2,
    )
    helper.batchid_fwd_rets = {0: [original_order[1], original_order[0]]}
    helper.extend_orders = [[1, 0]]

    with patch(f"{_SMART_PAD_HELPER_PATH}.mpu.is_pipeline_last_stage", return_value=True):
        restored = helper.get_rowed_based_forward_results(is_row_based_rets=True)

    torch.testing.assert_close(restored[0][0]["logprobs"], original_order[0]["logprobs"])
    torch.testing.assert_close(
        restored[0][1]["prev_per_token_entropy"],
        original_order[1]["prev_per_token_entropy"],
    )


def test_smart_pad_helper_restores_dynamic_mbs_with_remainder():
    sample_order = [4, 1, 5, 0, 3, 2]
    batches = [
        [{"sample_id": sample_order[i]}, {"sample_id": sample_order[i + 1]}]
        for i in range(0, len(sample_order), 2)
    ]
    helper = CatedSmartPadInferHelper([], forward_batch_size=2)
    helper.extend_batches = batches
    helper.extend_orders = [
        [sample["sample_id"] for sample in batch] for batch in batches
    ]
    helper.seqlen_batch_ids = {4: [0, 1, 2]}
    microbatch_sizes = []

    def forward_step(batch_iter, num_microbatches, micro_batch_size, seq_length):
        assert seq_length == 4
        microbatch_sizes.append(micro_batch_size)
        outputs = []
        for batch in batch_iter:
            outputs.append(
                [
                    {
                        "logprobs": torch.tensor([sample["sample_id"]], dtype=torch.float),
                        "prev_per_token_entropy": torch.tensor(
                            [sample["sample_id"] / 10], dtype=torch.float
                        ),
                    }
                    for sample in batch
                ]
            )
        assert len(outputs) == num_microbatches
        return outputs

    with (
        patch(f"{_SMART_PAD_HELPER_PATH}.mpu.is_pipeline_last_stage", return_value=True),
        patch(f"{_SMART_PAD_HELPER_PATH}.log"),
    ):
        helper.forward_per_seqlen_batches(
            forward_step_wrapped_func=forward_step,
            dynamic_mbs_target_seqlen=8,
            dynamic_mbs_limit=4,
        )
        restored = helper.get_rowed_based_forward_results(is_row_based_rets=True)

    assert microbatch_sizes == [4, 2]
    flattened = [sample for batch in restored for sample in batch]
    for sample_id, sample in enumerate(flattened):
        torch.testing.assert_close(
            sample["logprobs"], torch.tensor([sample_id], dtype=torch.float)
        )
        torch.testing.assert_close(
            sample["prev_per_token_entropy"],
            torch.tensor([sample_id / 10], dtype=torch.float),
        )


@pytest.mark.parametrize("return_per_token_entropy", [False, True])
def test_smart_pad_compute_logprobs_preserves_return_contract(return_per_token_entropy):
    mixin = _make_smart_pad_mixin()
    helper = Mock()
    if return_per_token_entropy:
        expected = _make_entropy_results()
    else:
        expected = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    helper.get_rowed_based_forward_results.return_value = [expected]

    with (
        patch(f"{_MIXIN_PATH}.CatedSmartPadInferHelper", return_value=helper),
        patch(f"{_MIXIN_PATH}.mpu.is_pipeline_last_stage", return_value=True),
        patch(
            f"{_MIXIN_PATH}.BroadcastUtils.broadcast_object_within_pp",
            side_effect=lambda value: value,
        ),
        patch(f"{_MIXIN_PATH}.clear_memory"),
    ):
        result = ForwardStepMixin.smart_pad_compute_logprobs(
            mixin,
            model=object(),
            batches_list=[{"tokens": torch.ones(3)}, {"tokens": torch.ones(5)}],
            batch_log_str="smart-pad",
            return_per_token_entropy=return_per_token_entropy,
        )

    assert len(result) == 2
    callback = helper.forward_per_seqlen_batches.call_args.kwargs["forward_step_wrapped_func"]
    assert callback.keywords["return_per_token_entropy"] is return_per_token_entropy
    if return_per_token_entropy:
        assert set(result[0]) == {"logprobs", "prev_per_token_entropy"}
        assert result[0]["logprobs"].device.type == "cpu"
        assert result[0]["logprobs"].dtype == torch.float32
        assert result[0]["prev_per_token_entropy"].device.type == "cpu"
        assert result[0]["prev_per_token_entropy"].dtype == torch.bfloat16
        torch.testing.assert_close(result[1]["logprobs"], expected[1]["logprobs"].float())
    else:
        assert isinstance(result[0], torch.Tensor)
        torch.testing.assert_close(result[1], expected[1])


def test_smart_pad_compute_logprobs_uses_pp_broadcast_on_non_last_stage():
    mixin = _make_smart_pad_mixin()
    helper = Mock()
    helper.get_rowed_based_forward_results.return_value = []
    expected = _make_entropy_results()

    with (
        patch(f"{_MIXIN_PATH}.CatedSmartPadInferHelper", return_value=helper),
        patch(f"{_MIXIN_PATH}.mpu.is_pipeline_last_stage", return_value=False),
        patch(
            f"{_MIXIN_PATH}.BroadcastUtils.broadcast_object_within_pp",
            return_value=expected,
        ) as broadcast,
        patch(f"{_MIXIN_PATH}.clear_memory"),
    ):
        result = ForwardStepMixin.smart_pad_compute_logprobs(
            mixin,
            model=object(),
            batches_list=[{"tokens": torch.ones(3)}, {"tokens": torch.ones(5)}],
            batch_log_str="smart-pad",
            return_per_token_entropy=True,
        )

    broadcast.assert_called_once_with([])
    assert result is expected
