"""Prepare-data branches forward the batch-level RL fields to the loss."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gpatch_v4.extended_model.gemma4 import Gemma4PrepareDataForward
from gpatch_v4.extended_model.llm import PrepareDataForwardLLM

RETENTION_RATIO = torch.tensor([0.5])
ENTROPY_AUX_FIGURES = torch.tensor([1.25, -0.75, 0.125])
FIELDS = ("sample_mask", "entropy_aux_figures")


def _batch(**extra):
    return {
        "tokens": torch.tensor([1, 2, 3, 4]),
        "advantages": torch.tensor([0.1, 0.2, 0.3]),
        "mask": torch.tensor([0.0, 1.0, 1.0]),
        "logprobs": torch.tensor([-1.0, -2.0, -3.0]),
        "ref_logprobs": torch.tensor([-1.1, -2.1, -3.1]),
        **extra,
    }


class _Gemma4(Gemma4PrepareDataForward):
    """Concrete stand-in; grpo_train does not touch the loss-weight hook."""
    def prepare_loss_weights(self, *args, **kwargs):
        raise NotImplementedError


def _gemma4_grpo_train(batches):
    prepare = _Gemma4.__new__(_Gemma4)
    prepare.config = SimpleNamespace()
    batch, _ = prepare.grpo_train(batches, seqlen=4, pad_token_id=0, ppo_pack_seq=False)
    return batch


def _opd_batch(**extra):
    batch = _batch(sequence_lengths=torch.tensor(4), **extra)
    batch["teacher_logprobs_t"] = torch.tensor([-1.2, -2.2, -3.2])
    return batch


def _llm_opd_train(batches, pp_size=1):
    prepare = PrepareDataForwardLLM.__new__(PrepareDataForwardLLM)
    prepare.config = SimpleNamespace(
        teachers={"t": None},
        ppo=SimpleNamespace(g_opd_teacher_routing_field="teacher_type"),
        training=SimpleNamespace(online_mtp_sft=False),
    )
    module = "gpatch_v4.extended_model.llm"
    with patch(f"{module}.mpu.get_pipeline_model_parallel_world_size", return_value=pp_size), \
         patch(f"{module}.mpu.is_pipeline_first_stage", return_value=pp_size == 1), \
         patch(f"{module}.mpu.is_pipeline_last_stage", return_value=True), \
         patch(f"{module}.mpu.get_context_parallel_group", return_value=None), \
         patch(f"{module}.dist.get_world_size", return_value=1):
        batch, _ = prepare.opd_train(batches, seqlen=4, pad_token_id=0, ppo_pack_seq=False)
    return batch


# GRPO on Gemma4 and OPD on the text LLM are two prepare-data branches that
# build the forward batch by hand, so each has to carry both fields itself.
BRANCHES = [(_gemma4_grpo_train, _batch), (_llm_opd_train, _opd_batch)]
BRANCH_IDS = ["gemma4_grpo", "llm_opd"]


@pytest.mark.parametrize("prepare,make_batch", BRANCHES, ids=BRANCH_IDS)
@patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
def test_batch_level_fields_reach_the_loss(prepare, make_batch):
    # sample_mask is per sample; entropy_aux_figures is one value for the batch.
    batches = [
        make_batch(
            sample_mask=torch.tensor(1.0),
            entropy_aux_figures=ENTROPY_AUX_FIGURES,
        ),
        make_batch(sample_mask=torch.tensor(0.0)),
    ]

    batch = prepare(batches)

    assert batch["sample_mask"].tolist() == [1.0, 0.0]
    assert torch.equal(batch["entropy_aux_figures"], ENTROPY_AUX_FIGURES)


@pytest.mark.parametrize("prepare,make_batch", BRANCHES, ids=BRANCH_IDS)
@patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
def test_global_retention_ratio_is_not_forwarded(prepare, make_batch):
    # The new loss path does not read it, so these two branches stop carrying
    # it even when the input batch has it.
    batch = prepare([make_batch(global_retention_ratio=RETENTION_RATIO)])

    assert "global_retention_ratio" not in batch


@pytest.mark.parametrize("prepare,make_batch", BRANCHES, ids=BRANCH_IDS)
@patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
def test_absent_fields_are_not_synthesized(prepare, make_batch):
    batch = prepare([make_batch()])

    for field in FIELDS:
        assert field not in batch


@patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
def test_the_last_pipeline_stage_keeps_the_fields():
    # With one pipeline stage every key is required, so the explicit list only
    # runs when the pipeline is split.
    batches = [
        _opd_batch(
            sample_mask=torch.tensor(1.0),
            global_retention_ratio=RETENTION_RATIO,
            entropy_aux_figures=ENTROPY_AUX_FIGURES,
        ),
        _opd_batch(sample_mask=torch.tensor(0.0)),
    ]

    batch = _llm_opd_train(batches, pp_size=2)

    assert batch["sample_mask"].tolist() == [1.0, 0.0]
    assert torch.equal(batch["entropy_aux_figures"], ENTROPY_AUX_FIGURES)
    assert "global_retention_ratio" not in batch
