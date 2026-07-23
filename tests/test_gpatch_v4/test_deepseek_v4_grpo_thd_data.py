from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gpatch_v4.extended_model.deepseek_v4 import DeepseekV4PrepareDataForwardLLM
from gpatch_v4.training_backend.fsdp2_backend.mixin import (
    ForwardStepMixin,
    unpack_thd_log_probs,
)


@patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
@patch("gpatch_v4.extended_model.deepseek_v4.mpu")
@patch("gpatch_v4.extended_model.deepseek_v4.dist.get_world_size", return_value=1)
def test_grpo_thd_fields_share_segment_boundaries(_world_size, mock_mpu):
    mock_mpu.get_context_parallel_rank.return_value = 0
    prepare = DeepseekV4PrepareDataForwardLLM.__new__(DeepseekV4PrepareDataForwardLLM)
    prepare.config = SimpleNamespace(policy=SimpleNamespace(ppo_pack_seq=True), )
    prepare._model_config = SimpleNamespace(
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4
        },
        sliding_window=4,
    )
    prepare._pad_each_doc_to_multi_of = 4

    batches = [
        {
            "tokens": torch.tensor([1, 2, 3, 4]),
            "sequence_lengths": torch.tensor(4),
            "advantages": torch.tensor([0.1, 0.2, 0.3]),
            "mask": torch.tensor([0.0, 1.0, 1.0]),
            "logprobs": torch.tensor([-1.0, -2.0, -3.0]),
            "ref_logprobs": torch.tensor([-1.1, -2.1, -3.1]),
            "rollout_log_probs": torch.tensor([-1.2, -2.2, -3.2, 99.0]),
        },
        {
            "tokens": torch.tensor([5, 6, 7]),
            "sequence_lengths": torch.tensor(3),
            "advantages": torch.tensor([0.4, 0.5]),
            "mask": torch.tensor([1.0, 1.0]),
            "logprobs": torch.tensor([-4.0, -5.0]),
            "ref_logprobs": torch.tensor([-4.1, -5.1]),
            "rollout_log_probs": torch.tensor([-4.2, -5.2, 88.0]),
        },
    ]

    batch, fwd_kwargs = prepare._grpo_fsdp2_train_thd(
        batches,
        seq_len=8,
        pad_token_id=0,
        include_rl_fields=True,
    )

    assert fwd_kwargs["input_ids"].tolist() == [[1, 2, 3, 0, 5, 6, 0, 0]]
    assert fwd_kwargs["position_ids"].tolist() == [[0, 1, 2, 3, 0, 1, 2, 3]]
    assert batch["target"].tolist() == [[2, 3, 4, 0, 6, 7, 0, 0]]
    assert batch["mask"].tolist() == [[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]]
    assert batch["cu_seqlens"].tolist() == [0, 3, 5]
    assert batch["cu_seqlens_padded"].tolist() == [0, 4, 8]
    assert torch.allclose(
        batch["prev_log_probs"],
        torch.tensor([[-1.0, -2.0, -3.0, 0.0, -4.0, -5.0, 0.0, 0.0]]),
    )
    assert torch.allclose(
        batch["ref_log_probs"],
        torch.tensor([[-1.1, -2.1, -3.1, 0.0, -4.1, -5.1, 0.0, 0.0]]),
    )
    assert torch.allclose(
        batch["rollout_log_probs"],
        torch.tensor([[-1.2, -2.2, -3.2, 0.0, -4.2, -5.2, 0.0, 0.0]]),
    )


@patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
@patch("gpatch_v4.extended_model.deepseek_v4.mpu")
@patch("gpatch_v4.extended_model.deepseek_v4.dist.get_world_size", return_value=2)
def test_grpo_thd_cp_slice_can_cross_a_segment(_world_size, mock_mpu):
    mock_mpu.get_context_parallel_rank.return_value = 1
    prepare = DeepseekV4PrepareDataForwardLLM.__new__(DeepseekV4PrepareDataForwardLLM)
    prepare.config = SimpleNamespace(policy=SimpleNamespace(ppo_pack_seq=True))
    prepare._model_config = SimpleNamespace(
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4
        },
        sliding_window=4,
    )
    prepare._pad_each_doc_to_multi_of = 4

    batches = [
        {
            "tokens": torch.tensor([1, 2, 3, 4]),
            "sequence_lengths": torch.tensor(4),
            "advantages": torch.tensor([0.1, 0.2, 0.3]),
            "mask": torch.ones(3),
            "logprobs": torch.tensor([-1.0, -2.0, -3.0]),
        },
        {
            "tokens": torch.tensor([10, 11, 12, 13, 14, 15]),
            "sequence_lengths": torch.tensor(6),
            "advantages": torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4]),
            "mask": torch.ones(5),
            "logprobs": torch.tensor([-4.0, -5.0, -6.0, -7.0, -8.0]),
        },
    ]

    batch, fwd_kwargs = prepare._grpo_fsdp2_train_thd(
        batches,
        seq_len=8,
        pad_token_id=0,
        include_rl_fields=True,
    )

    # Padded segment lengths are [4, 12], so rank 1 owns global [8:16],
    # starting inside segment 1 rather than on a segment boundary.
    assert batch["cu_seqlens_padded"].tolist() == [0, 4, 16]
    assert fwd_kwargs["input_ids"].tolist() == [[14, 0, 0, 0, 0, 0, 0, 0]]
    assert fwd_kwargs["position_ids"].tolist() == [[4, 5, 6, 7, 8, 9, 10, 11]]
    assert fwd_kwargs["packed_seq_params"].layout.pad_token_mask.tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    # Policy-loss tensors remain global until CP logprobs/entropy are gathered.
    assert batch["target"].tolist() == [[2, 3, 4, 0, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0]]
    assert batch["mask"].tolist() == [
        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]


@patch(
    "gpatch_v4.training_backend.fsdp2_backend.mixin.mpu.get_context_parallel_world_size",
    return_value=1,
)
def test_gather_log_probs_distinguishes_bshd_and_pre_shifted_thd(_cp_size):
    class _IdentityPrepare:
        @staticmethod
        def rl_train_cp_chunk_single_data(tensor):
            return tensor

    mixin = ForwardStepMixin.__new__(ForwardStepMixin)
    mixin.prepare_data = _IdentityPrepare()
    logits = torch.tensor(
        [
            [
                [2.0, 1.0, 0.0, -1.0, -2.0],
                [0.0, 2.0, 1.0, -1.0, -2.0],
                [-1.0, 0.0, 2.0, 1.0, -2.0],
                [-2.0, -1.0, 0.0, 2.0, 1.0],
            ]
        ]
    )
    log_softmax = logits.log_softmax(dim=-1)

    input_ids = torch.tensor([[0, 1, 2, 3]])
    bshd = mixin.gather_log_probs_packed(
        logits,
        input_ids,
        allow_compile=False,
        pre_shifted=False,
    )
    expected_bshd = log_softmax[:, :3].gather(-1, torch.tensor([[[1], [2], [3]]])).squeeze(-1)
    assert torch.allclose(bshd, expected_bshd)
    assert bshd.shape == (1, 3)

    packed_targets = torch.tensor([[1, 2, 3, 0]])
    thd = mixin.gather_log_probs_packed(
        logits,
        packed_targets,
        allow_compile=False,
        pre_shifted=True,
    )
    expected_thd = log_softmax.gather(-1, packed_targets.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(thd, expected_thd)
    assert thd.shape == (1, 4)


@patch("gpatch_v4.training_backend.fsdp2_backend.mixin.mpu")
def test_entropy_alignment_matches_policy_axis(mock_mpu):
    mock_mpu.get_context_parallel_world_size.return_value = 1
    mixin = ForwardStepMixin.__new__(ForwardStepMixin)
    entropy = torch.tensor([[0.1, 0.2, 0.3, 9.9]])

    bshd_mask = torch.ones(1, 3)
    bshd = mixin._finalize_rl_policy_entropy(
        entropy,
        bshd_mask,
        pre_shifted=False,
    )
    assert torch.equal(bshd, entropy[:, :-1])

    thd_mask = torch.ones(1, 4)
    thd = mixin._finalize_rl_policy_entropy(
        entropy,
        thd_mask,
        pre_shifted=True,
    )
    assert torch.equal(thd, entropy)

    with pytest.raises(AssertionError, match="must match response mask"):
        mixin._finalize_rl_policy_entropy(
            entropy,
            bshd_mask,
            pre_shifted=True,
        )


def test_unpack_thd_log_probs_uses_padded_starts_and_effective_lengths():
    packed = torch.tensor([[10.0, 11.0, 12.0, -1.0, 20.0, 21.0, -1.0, -1.0]])
    restored = unpack_thd_log_probs(
        packed,
        cu_seqlens=torch.tensor([0, 3, 5]),
        cu_seqlens_padded=torch.tensor([0, 4, 8]),
    )

    assert len(restored) == 2
    assert torch.equal(restored[0], torch.tensor([10.0, 11.0, 12.0]))
    assert torch.equal(restored[1], torch.tensor([20.0, 21.0]))

    with pytest.raises(AssertionError, match="must match packed logprob"):
        unpack_thd_log_probs(
            packed,
            cu_seqlens=torch.tensor([0, 3, 5]),
            cu_seqlens_padded=torch.tensor([0, 4, 7]),
        )
