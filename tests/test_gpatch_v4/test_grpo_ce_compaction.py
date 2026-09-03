import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Callable, Iterator
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import gpatch_v4.utils.dynamic_cp_utils as dynamic_cp_utils
import gpatch_v4.utils.training_utils as training_utils
from gpatch_v4.configs.config import OnPolicyDistillConfig, RlConfig
from gpatch_v4.configs.policy_config import PolicyConfig
from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.configs.training_config import RLTrainingConfig, T2iRlTrainingConfig
from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.utils.dynamic_cp_utils import build_grpo_compact_ce_mask_dyn_cp
from gpatch_v4.utils.training_utils import build_grpo_compact_ce_mask


class _LinearCERecorder:
    def __init__(self) -> None:
        self.hidden: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []

    def __call__(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        *_: object,
        return_entropy: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_labels = labels.reshape(-1)
        self.hidden.append(flat_hidden.detach().clone())
        self.labels.append(flat_labels.detach().clone())
        logits = flat_hidden @ weight.t()
        loss = F.cross_entropy(logits, flat_labels, reduction="none").view_as(labels).float()
        if not return_entropy:
            return loss
        logprobs = F.log_softmax(logits, dim=-1)
        entropy = -(logprobs.exp() * logprobs).sum(dim=-1).view_as(labels).float()
        return loss, entropy


@contextmanager
def _mock_linear_ce(recorder: _LinearCERecorder) -> Iterator[None]:
    with (
        patch.object(training_utils, "set_linear_ce_backend", return_value=None),
        patch.object(training_utils, "linear_cross_entropy", side_effect=recorder),
        patch.object(training_utils.mpu, "get_context_parallel_rank", return_value=0),
        patch.object(
            training_utils.mpu,
            "get_context_parallel_world_size",
            return_value=1,
        ),
        patch.object(training_utils.dist, "get_world_size", return_value=1),
    ):
        yield


@contextmanager
def _stub_vocab_parallel_entropy(side_effect: Callable) -> Iterator[None]:
    # 不 import training_backend：本机 gcore_test 没有 TE，整包 import 会失败。
    names = (
        "gpatch_v4.training_backend",
        "gpatch_v4.training_backend.vocab_parallel_entropy",
    )
    saved = {name: sys.modules.get(name) for name in names}
    backend = ModuleType("gpatch_v4.training_backend")
    backend.__path__ = []
    entropy_mod = ModuleType("gpatch_v4.training_backend.vocab_parallel_entropy")
    entropy_mod.vocab_parallel_entropy = side_effect
    sys.modules["gpatch_v4.training_backend"] = backend
    sys.modules["gpatch_v4.training_backend.vocab_parallel_entropy"] = entropy_mod
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _linear_ce_input(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    sequence_parallel: bool = False,
    tp_group: object | None = None,
) -> dict[str, object]:
    return {
        "hidden_states":
            hidden,
        "weight":
            weight,
        "output_layer":
            SimpleNamespace(
                tp_group=tp_group,
                sequence_parallel=sequence_parallel,
                weight=weight,
                bias=None,
                gather_output=False,
            ),
        "runtime_gather_output":
            None,
    }


def _run_linear_ce(
    hidden_data: torch.Tensor,
    weight_data: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    compaction: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
]:
    hidden = hidden_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    recorder = _LinearCERecorder()
    with _mock_linear_ce(recorder):
        logprobs, scaled_entropy, entropy = training_utils.logprobs_from_linear_ce(
            linear_ce_backend="separate",
            linear_ce_output=_linear_ce_input(hidden, weight),
            target=target,
            mask=mask,
            return_entropy=True,
            token_compaction=compaction,
        )
    ((logprobs * mask).sum() + 0.2 * (entropy * mask).sum()).backward()
    return (
        logprobs.detach(),
        scaled_entropy.detach(),
        entropy.detach(),
        hidden.grad,
        weight.grad,
        recorder.hidden,
    )


@pytest.mark.parametrize(
    ("active_indices", "expected_rows"),
    [([0, 2, 5], 3), ([], 0), ([4], 2)],
)
def test_grpo_compaction_matches_dense_outputs_and_gradients(
    active_indices: list[int],
    expected_rows: int,
) -> None:
    torch.manual_seed(41)
    sequence, batch, hidden_size, vocab_size = 4, 2, 6, 11
    hidden = torch.randn(sequence, batch, hidden_size, dtype=torch.float64)
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.float64)
    target = torch.randint(0, vocab_size, (batch, sequence))
    mask = torch.zeros((batch, sequence - 1), dtype=torch.float32)
    mask.view(-1)[active_indices] = 1

    dense = _run_linear_ce(hidden, weight, target, mask, False)
    compact = _run_linear_ce(hidden, weight, target, mask, True)

    active = mask.bool()
    torch.testing.assert_close(compact[0][active], dense[0][active])
    torch.testing.assert_close(compact[1], dense[1])
    torch.testing.assert_close(compact[2][active], dense[2][active])
    torch.testing.assert_close(compact[0][~active], torch.zeros_like(compact[0][~active]))
    torch.testing.assert_close(compact[2][~active], torch.zeros_like(compact[2][~active]))
    torch.testing.assert_close(compact[3], dense[3])
    torch.testing.assert_close(compact[4], dense[4])
    assert dense[5][0].shape[0] == sequence * batch
    assert [value.shape[0]
            for value in compact[5]] == ([] if expected_rows == 0 else [expected_rows])


def test_grpo_compaction_supports_pre_shifted_dynamic_cp_layout() -> None:
    hidden = torch.randn(5, 1, 4, dtype=torch.float64)
    weight = torch.randn(9, 4, dtype=torch.float64)
    target = torch.randint(0, 9, (1, 5))
    mask = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]])
    recorder = _LinearCERecorder()

    with _mock_linear_ce(recorder):
        logprobs = training_utils.logprobs_from_linear_ce(
            linear_ce_backend="separate",
            linear_ce_output=_linear_ce_input(hidden, weight),
            target=target,
            ignore_cp=True,
            pre_shifted=True,
            mask=mask,
            token_compaction=True,
        )

    assert logprobs.shape == target.shape
    torch.testing.assert_close(recorder.labels[0], target[mask.bool()])


def test_grpo_compaction_runs_after_sequence_parallel_gather() -> None:
    local_hidden = torch.randn(2, 1, 4, dtype=torch.float64)
    gathered_hidden = torch.arange(16.0, dtype=torch.float64).reshape(4, 1, 4)
    weight = torch.randn(9, 4, dtype=torch.float64)
    recorder = _LinearCERecorder()

    with _mock_linear_ce(recorder), patch.object(
        training_utils.tensor_parallel,
        "gather_from_sequence_parallel_region",
        return_value=gathered_hidden,
    ):
        training_utils.logprobs_from_linear_ce(
            linear_ce_backend="separate",
            linear_ce_output=_linear_ce_input(
                local_hidden,
                weight,
                sequence_parallel=True,
            ),
            target=torch.tensor([[2, 3, 4, 5]]),
            mask=torch.tensor([[1.0, 0.0, 1.0]]),
            token_compaction=True,
        )

    torch.testing.assert_close(recorder.hidden[0], gathered_hidden[[0, 2], 0])


@pytest.mark.parametrize(
    ("active_indices", "expected_rows"),
    [([0, 2, 5], 3), ([], 0), ([4], 1)],
)
def test_ordinary_ce_compaction_matches_dense_outputs_and_gradients(
    active_indices: list[int],
    expected_rows: int,
) -> None:
    torch.manual_seed(41)
    sequence, batch, hidden_size, vocab_size = 4, 2, 6, 11
    hidden_data = torch.randn(sequence, batch, hidden_size, dtype=torch.float64)
    weight_data = torch.randn(vocab_size, hidden_size, dtype=torch.float64)
    target = torch.randint(0, vocab_size, (batch, sequence))
    mask = torch.zeros((batch, sequence - 1), dtype=torch.float32)
    mask.view(-1)[active_indices] = 1
    projected_rows: list[int] = []

    def vocab_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        projected_rows.append(logits.shape[0])
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)

    def vocab_entropy(
        logits: torch.Tensor,
        mask: torch.Tensor | None = None,
        ignore_cp: bool = False,
        pre_shifted: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logprobs = F.log_softmax(logits.float(), dim=-1)
        entropy = -(logprobs.exp() * logprobs).sum(dim=-1)
        return entropy.mean(), entropy

    hidden = hidden_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    with (
        patch.object(
            training_utils.tensor_parallel,
            "vocab_parallel_cross_entropy",
            side_effect=vocab_ce,
        ),
        _stub_vocab_parallel_entropy(vocab_entropy),
        patch.object(training_utils.mpu, "get_context_parallel_rank", return_value=0),
        patch.object(training_utils.mpu, "get_context_parallel_world_size", return_value=1),
        patch.object(training_utils.dist, "get_world_size", return_value=1),
    ):
        compact_logprobs, compact_scaled, compact_entropy = training_utils.logprobs_from_compact_ce(
            linear_ce_output=_linear_ce_input(hidden, weight),
            target=target,
            mask=mask,
            return_entropy=True,
        )
    ((compact_logprobs * mask).sum() + 0.2 * (compact_entropy * mask).sum()).backward()

    dense_hidden = hidden_data.clone().requires_grad_(True)
    dense_weight = weight_data.clone().requires_grad_(True)
    shifted = target.roll(shifts=-1, dims=-1)
    dense_logits = (dense_hidden @ dense_weight.t()).float()
    dense_nll = F.cross_entropy(
        dense_logits.reshape(-1, vocab_size),
        shifted.transpose(0, 1).reshape(-1),
        reduction="none",
    ).view(sequence, batch)
    dense_logprobs = (-dense_nll).transpose(0, 1)[:, :-1]
    dense_logp = F.log_softmax(dense_logits, dim=-1)
    dense_entropy = (-(dense_logp.exp() * dense_logp).sum(dim=-1)).transpose(0, 1)[:, :-1]
    ((dense_logprobs * mask).sum() + 0.2 * (dense_entropy * mask).sum()).backward()

    active = mask.bool()
    torch.testing.assert_close(compact_logprobs[active], dense_logprobs[active])
    torch.testing.assert_close(compact_entropy[active], dense_entropy[active])
    torch.testing.assert_close(
        compact_logprobs[~active],
        torch.zeros_like(compact_logprobs[~active]),
    )
    torch.testing.assert_close(
        compact_entropy[~active],
        torch.zeros_like(compact_entropy[~active]),
    )
    torch.testing.assert_close(hidden.grad, dense_hidden.grad)
    torch.testing.assert_close(weight.grad, dense_weight.grad)
    assert projected_rows == ([] if expected_rows == 0 else [expected_rows])
    if active.any():
        torch.testing.assert_close(compact_scaled, (dense_entropy * mask).sum() / mask.sum())


def test_ordinary_ce_compaction_preserves_sp_tp_gradient_mappings() -> None:
    local_hidden = torch.randn(2, 1, 4, dtype=torch.float64)
    gathered_hidden = torch.arange(16.0, dtype=torch.float64).reshape(4, 1, 4)
    weight = torch.randn(9, 4, dtype=torch.float64)
    tp_group = object()

    def vocab_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)

    with (
        patch.object(
            training_utils.tensor_parallel,
            "gather_from_sequence_parallel_region",
            return_value=gathered_hidden,
        ) as gather,
        patch.object(
            training_utils.tensor_parallel,
            "vocab_parallel_cross_entropy",
            side_effect=vocab_ce,
        ),
        patch.object(
            training_utils.dist,
            "get_world_size",
            side_effect=lambda group: 2 if group is tp_group else 1,
        ),
        patch.object(training_utils.mpu, "get_context_parallel_rank", return_value=0),
        patch.object(training_utils.mpu, "get_context_parallel_world_size", return_value=1),
        patch.object(
            training_utils.tensor_parallel,
            "copy_to_tensor_model_parallel_region",
            side_effect=lambda value, group: value,
        ) as copy_to_tp,
    ):
        training_utils.logprobs_from_compact_ce(
            linear_ce_output=_linear_ce_input(
                local_hidden,
                weight,
                sequence_parallel=True,
                tp_group=tp_group,
            ),
            target=torch.tensor([[2, 3, 4, 5]]),
            mask=torch.tensor([[1.0, 0.0, 1.0]]),
        )

    gather.assert_called_once_with(local_hidden, tensor_parallel_output_grad=False)
    projected_hidden = copy_to_tp.call_args.args[0]
    torch.testing.assert_close(projected_hidden, gathered_hidden[[0, 2], 0].unsqueeze(1))
    assert copy_to_tp.call_args.kwargs == {"group": tp_group}


def test_ordinary_ce_compaction_supports_pre_shifted_dynamic_cp_layout() -> None:
    hidden = torch.randn(5, 1, 4, dtype=torch.float64)
    weight = torch.randn(9, 4, dtype=torch.float64)
    target = torch.randint(0, 9, (1, 5))
    mask = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0]])
    captured_labels: list[torch.Tensor] = []

    def vocab_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        captured_labels.append(labels.reshape(-1).detach().clone())
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)

    with (
        patch.object(
            training_utils.tensor_parallel,
            "vocab_parallel_cross_entropy",
            side_effect=vocab_ce,
        ),
        patch.object(training_utils.mpu, "get_context_parallel_rank", return_value=0),
        patch.object(training_utils.mpu, "get_context_parallel_world_size", return_value=1),
    ):
        logprobs = training_utils.logprobs_from_compact_ce(
            linear_ce_output=_linear_ce_input(hidden, weight),
            target=target,
            ignore_cp=True,
            pre_shifted=True,
            mask=mask,
        )

    assert logprobs.shape == target.shape
    torch.testing.assert_close(captured_labels[0], target[mask.bool()])


def test_ordinary_ce_compaction_reorders_mask_with_static_cp() -> None:
    # 普通 CE 没有 fused 的 K=1 pad，rank0 只留下 1 个有效 label。
    target = torch.arange(8).reshape(1, 8)
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]])
    weight = torch.randn(10, 3, dtype=torch.float64)
    rank_labels: list[torch.Tensor] = []

    for cp_rank in range(2):
        captured_labels: list[torch.Tensor] = []

        def vocab_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            captured_labels.append(labels.reshape(-1).detach().clone())
            return F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            ).view_as(labels)

        with (
            patch.object(
                training_utils.tensor_parallel,
                "vocab_parallel_cross_entropy",
                side_effect=vocab_ce,
            ),
            patch.object(
                training_utils.mpu,
                "get_context_parallel_rank",
                return_value=cp_rank,
            ),
            patch.object(
                training_utils.mpu,
                "get_context_parallel_world_size",
                return_value=2,
            ),
            patch.object(
                training_utils,
                "all_gather_from_context_parallel_region",
                side_effect=lambda value: value,
            ),
        ):
            training_utils.logprobs_from_compact_ce(
                linear_ce_output=_linear_ce_input(
                    torch.randn(4, 1, 3, dtype=torch.float64),
                    weight,
                ),
                target=target,
                mask=mask,
            )
        rank_labels.append(captured_labels[0])

    torch.testing.assert_close(rank_labels[0], torch.tensor([1]))
    torch.testing.assert_close(rank_labels[1], torch.tensor([3, 6]))


def test_grpo_compaction_reorders_mask_with_static_cp() -> None:
    target = torch.arange(8).reshape(1, 8)
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]])
    weight = torch.randn(10, 3, dtype=torch.float64)
    rank_labels = []

    for cp_rank in range(2):
        recorder = _LinearCERecorder()
        with (
            patch.object(training_utils, "set_linear_ce_backend", return_value=None),
            patch.object(training_utils, "linear_cross_entropy", side_effect=recorder),
            patch.object(
                training_utils.mpu,
                "get_context_parallel_rank",
                return_value=cp_rank,
            ),
            patch.object(
                training_utils.mpu,
                "get_context_parallel_world_size",
                return_value=2,
            ),
            patch.object(
                training_utils,
                "all_gather_from_context_parallel_region",
                side_effect=lambda value: value,
            ),
        ):
            training_utils.logprobs_from_linear_ce(
                linear_ce_backend="separate",
                linear_ce_output=_linear_ce_input(
                    torch.randn(4, 1, 3, dtype=torch.float64),
                    weight,
                ),
                target=target,
                mask=mask,
                token_compaction=True,
            )
        rank_labels.append(recorder.labels[0])

    torch.testing.assert_close(rank_labels[0], torch.tensor([1, 1]))
    torch.testing.assert_close(rank_labels[1], torch.tensor([3, 6]))


def test_forward_only_mask_builder_uses_rollout_metadata() -> None:
    batches = [
        {
            "mask": torch.tensor([0.0, 1.0, 0.0, 1.0]),
            "sequence_lengths": torch.tensor(5),
        },
        {
            "prompt_lengths": torch.tensor(2),
            "sequence_lengths": torch.tensor(5)
        },
        {
            "prompt_lengths": torch.tensor(1),
            "sequence_lengths": torch.tensor(4),
            "sample_mask": torch.tensor(0.0),
        },
    ]

    mask = build_grpo_compact_ce_mask(
        batches,
        torch.zeros((3, 6), dtype=torch.long),
    )

    torch.testing.assert_close(
        mask,
        torch.tensor(
            [
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_forward_only_mask_builder_rejects_response_only_mask() -> None:
    # 已有 mask 必须覆盖未 pad 的整条 logprob 轴，不能只放 response 片段。
    with pytest.raises(AssertionError, match="unpadded logprob axis"):
        build_grpo_compact_ce_mask(
            [
                {
                    "mask": torch.tensor([1.0, 1.0]),
                    "prompt_lengths": torch.tensor(3),
                    "sequence_lengths": torch.tensor(5),
                }
            ],
            torch.zeros((1, 6), dtype=torch.long),
        )


def test_dynamic_cp_builds_and_selects_response_mask(monkeypatch: pytest.MonkeyPatch, ) -> None:
    group = SimpleNamespace(rank=lambda: 0)
    monkeypatch.setattr(
        dynamic_cp_utils.parallel_state,
        "get_dynamic_data_context_parallel_groups",
        lambda group_size: group,
    )
    monkeypatch.setattr(
        dynamic_cp_utils,
        "get_thd_partitioned_indices",
        lambda *_: torch.tensor([0, 3, 4, 7]),
    )
    batch = {
        "cu_seqlens_padded": torch.tensor([0, 4, 8], dtype=torch.int32),
        "dyn_cp_response_start": torch.tensor([1, 0], dtype=torch.int32),
        "dyn_cp_response_length": torch.tensor([2, 1], dtype=torch.int32),
        "local_cp_size": torch.tensor(2),
    }

    local_mask = build_grpo_compact_ce_mask_dyn_cp(
        batch,
        torch.zeros((1, 4), dtype=torch.long),
        cp_partition_mode="zigzag",
    )

    torch.testing.assert_close(local_mask, torch.tensor([[0.0, 0.0, 1.0, 0.0]]))


def _compaction_config(model_arch: str, use_linear_ce: bool = True) -> RlConfig:
    return RlConfig(
        training=RLTrainingConfig(
            use_linear_ce=use_linear_ce,
            ce_compaction=True,
        ),
        policy=PolicyConfig(model_arch=model_arch),
    )


@pytest.mark.parametrize(
    "model_arch",
    (
        MODEL_ARCH.QWEN3_VL,
        MODEL_ARCH.QWEN3_VL_MOE,
        MODEL_ARCH.QWEN3_5,
        MODEL_ARCH.QWEN3_5_MOE,
        MODEL_ARCH.WELMV4_MOE,
    ),
)
@pytest.mark.parametrize("use_linear_ce", (False, True))
def test_grpo_compaction_model_scope(model_arch: str, use_linear_ce: bool) -> None:
    config = _compaction_config(model_arch, use_linear_ce)
    assert config.training.ce_compaction
    assert config.training.return_hidden_states_for_ce


@pytest.mark.parametrize(
    "model_arch",
    (
        MODEL_ARCH.QWEN3,
        MODEL_ARCH.QWEN3_MOE,
        MODEL_ARCH.WELM_MOE,
        MODEL_ARCH.DEEPSEEK_V4,
        MODEL_ARCH.QWEN3_OMNI_MOE,
        MODEL_ARCH.WELM_OMNI_V4_5,
    ),
)
def test_grpo_compaction_rejects_unsupported_model(model_arch: str) -> None:
    with pytest.raises(AssertionError, match="Qwen3-VL/Qwen3.5/Qwen3.6 or WeLM v4.5"):
        _compaction_config(model_arch)


def test_grpo_compaction_rejects_unsupported_configuration() -> None:
    assert not RLTrainingConfig().ce_compaction
    training = RLTrainingConfig(ce_compaction=True)
    assert not training.use_linear_ce
    assert training.return_hidden_states_for_ce
    with pytest.raises(AssertionError, match="MCore GRPO only"):
        RLTrainingConfig(
            training_backend="fsdp2",
            ce_compaction=True,
        )
    with pytest.raises(AssertionError, match="has not been validated"):
        RLTrainingConfig(
            apply_deterministic_mode=True,
            use_linear_ce=True,
            ce_compaction=True,
        )
    with pytest.raises(AssertionError, match="GRPO loss only"):
        RlConfig(
            training=RLTrainingConfig(
                use_linear_ce=True,
                ce_compaction=True,
            ),
            policy=PolicyConfig(model_arch=MODEL_ARCH.QWEN3_VL),
            ppo=PpoConfig(loss_func="cispo"),
        )
    with pytest.raises(AssertionError, match="smart_pad_infer"):
        RlConfig(
            training=RLTrainingConfig(
                use_linear_ce=True,
                ce_compaction=True,
            ),
            policy=PolicyConfig(model_arch=MODEL_ARCH.QWEN3_VL, smart_pad_infer=True),
        )


def test_grpo_compaction_rejects_other_training_flows() -> None:
    training = RLTrainingConfig(
        use_linear_ce=True,
        ce_compaction=True,
    )
    with pytest.raises(AssertionError, match="not on-policy distillation"):
        OnPolicyDistillConfig(training=training)
    with pytest.raises(AssertionError, match="text GRPO only"):
        T2iRlTrainingConfig(
            use_linear_ce=True,
            ce_compaction=True,
        )
