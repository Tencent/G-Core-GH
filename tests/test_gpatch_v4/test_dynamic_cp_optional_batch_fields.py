"""Batch-level RL fields survive the dynamic-CP reroute."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from gpatch_v4.extended_model.llm import PrepareDataForwardLLM
from gpatch_v4.extended_model.qwen3_vl import Qwen3VLPrepareDataForward
from gpatch_v4.extended_model.welm_v4 import WelmV4PrepareDataForwardLLM
from gpatch_v4.extended_model.wemm_video import WemmVideoPrepareDataForward

ENTROPY_AUX_FIGURES = torch.tensor([1.25, -0.75, 0.125])

# Each prepare class reads a few fields of its own, at construction time and
# off the incoming batch. Only the two shared fields below are under test.
PREPARE_CLASSES = [
    (PrepareDataForwardLLM, "llm", {}, {}),
    (
        Qwen3VLPrepareDataForward,
        "qwen3_vl",
        {
            "model_arch": "qwen3_vl"
        },
        {
            "position_ids": torch.arange(12).reshape(1, 3, 4),
            "image_input_mask": torch.zeros(4, dtype=torch.bool),
            "vision_data": None,
            "vision_grid_thw": None,
        },
    ),
    (
        WelmV4PrepareDataForwardLLM,
        "welm_v4",
        {
            "hf_config": SimpleNamespace(oe_grams=[], vocab_size=32)
        },
        {},
    ),
    (
        WemmVideoPrepareDataForward,
        "wemm_video",
        {},
        {
            "position_ids": torch.arange(12).reshape(1, 3, 4),
            "image_input_mask": torch.zeros(4, dtype=torch.bool),
        },
    ),
]
CLASS_IDS = [name for _, name, _, _ in PREPARE_CLASSES]


def _batch(batch_extra, drop=()):
    batch = {
        "tokens": torch.arange(5),
        "mask": torch.ones(4),
        "advantages": torch.ones(4),
        "logprobs": torch.zeros(4),
        "prompt_lengths": torch.tensor([2]),
        "sequence_lengths": torch.tensor([5]),
        "entropy_aux_figures": ENTROPY_AUX_FIGURES,
        **batch_extra,
    }
    for key in drop:
        del batch[key]
    return batch


def _config(policy_extra):
    dist_config = SimpleNamespace(
        max_seqlen_per_dp_cp_rank=8192,
        dynamic_cp_scheduler_type="default",
    )
    return SimpleNamespace(
        policy=SimpleNamespace(dist_config=dist_config, **policy_extra),
        training=SimpleNamespace(moe_router_replay=False, online_mtp_sft=False),
    )


def _reroute(cls, module_name, policy_extra, batch_extra, drop=()):
    """Run one prepare class's reroute with the scheduler and mpu stubbed out."""
    group = MagicMock()
    group.size.return_value = 1

    def schedule(samples, *args, **kwargs):
        return samples, 1, 4.0, 16.0, {}

    prefix = f"gpatch_v4.extended_model.{module_name}"
    with patch(f"{prefix}.mpu.get_data_parallel_group", return_value=group), \
         patch(f"{prefix}.mpu.get_tensor_model_parallel_group", return_value=group), \
         patch(f"{prefix}.mpu.get_data_parallel_rank", return_value=0), \
         patch(f"{prefix}.torch.cuda.current_device", return_value="cpu"), \
         patch(f"{prefix}.torch.distributed.all_reduce"), \
         patch(f"{prefix}.dyn_cp_schedule_default", side_effect=schedule):
        samples, *_ = cls(
            _config(policy_extra)
        ).rl_reroute_data_for_dynamic_cp([_batch(batch_extra, drop)], pad_token_id=0)
    return samples


@pytest.mark.parametrize("cls,module,policy_extra,batch_extra", PREPARE_CLASSES, ids=CLASS_IDS)
def test_rerouted_samples_keep_entropy_aux_figures(cls, module, policy_extra, batch_extra):
    # Without the key the policy loss falls back to microbatch-local entropy
    # covariance instead of the global one the actor computed.
    for sample in _reroute(cls, module, policy_extra, batch_extra):
        assert torch.equal(sample["entropy_aux_figures"], ENTROPY_AUX_FIGURES)


@pytest.mark.parametrize("cls,module,policy_extra,batch_extra", PREPARE_CLASSES, ids=CLASS_IDS)
def test_a_batch_that_never_carried_the_fields_reroutes_without_them(
    cls, module, policy_extra, batch_extra
):
    """The field is optional, so forwarding it must not become a requirement."""
    for sample in _reroute(cls, module, policy_extra, batch_extra, drop=("entropy_aux_figures", )):
        assert "entropy_aux_figures" not in sample
