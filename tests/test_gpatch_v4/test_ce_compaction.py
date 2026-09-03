from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F

import gpatch_v4.training_backend.loss_factory as loss_factory
import gpatch_v4.training_backend.megatron_backend.mixin as megatron_mixin
from gpatch_v4.configs.config import DpoConfig, FinetuneConfig
from gpatch_v4.configs.policy_config import BasePolicyConfig
from gpatch_v4.configs.training_config import (
    DpoTrainingConfig,
    FinetuneTrainingConfig,
    OffPolicyDistillTrainingConfig,
)
from gpatch_v4.core.constants import MODEL_ARCH


class _LinearCERecorder:
    def __init__(self) -> None:
        self.rows: list[int] = []

    def __call__(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        *_: object,
    ) -> torch.Tensor:
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_labels = labels.reshape(-1)
        self.rows.append(flat_hidden.shape[0])
        loss = F.cross_entropy(flat_hidden @ weight.t(), flat_labels, reduction="none")
        return loss.view_as(labels)


def _config(compaction: bool, use_linear_ce: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(
            use_linear_ce=use_linear_ce,
            linear_ce_backend="separate",
            ce_compaction=compaction,
            cross_entropy_loss_fusion=False,
            cross_entropy_fusion_impl="native",
        )
    )


def _linear_ce_input(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    sequence_parallel: bool = False,
    tp_group: object | None = None,
) -> dict[str, object]:
    output_layer = SimpleNamespace(
        tp_group=tp_group,
        sequence_parallel=sequence_parallel,
        weight=weight,
        bias=None,
        gather_output=False,
    )
    return {
        "hidden_states": hidden,
        "weight": weight,
        "output_layer": output_layer,
        "runtime_gather_output": None,
    }


def _run_linear_ce(
    hidden_data: torch.Tensor,
    weight_data: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_weights: torch.Tensor,
    compaction: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    hidden = hidden_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    recorder = _LinearCERecorder()
    with (
        patch.object(loss_factory, "set_linear_ce_backend", return_value=None),
        patch.object(loss_factory, "linear_cross_entropy", side_effect=recorder),
        patch.object(loss_factory.dist, "get_world_size", return_value=1),
        patch.object(loss_factory.mpu, "get_context_parallel_group", return_value=None),
    ):
        output = loss_factory.linear_ce_loss(
            _config(compaction),
            _linear_ce_input(hidden, weight),
            labels,
            loss_mask,
            loss_weights=loss_weights,
        )
    output[0].backward()
    return output.detach(), hidden.grad, weight.grad, recorder.rows


@pytest.mark.parametrize(
    ("active_indices", "expected_compact_rows"),
    [([0, 2, 5], 3), ([], 0), ([4], 2)],
)
def test_compaction_matches_dense_loss_and_gradients(
    active_indices: list[int],
    expected_compact_rows: int,
) -> None:
    torch.manual_seed(7)
    sequence, batch, hidden_size, vocab_size = 4, 2, 6, 11
    hidden = torch.randn(sequence, batch, hidden_size, dtype=torch.float64)
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.float64)
    labels = torch.randint(0, vocab_size, (batch, sequence))
    loss_mask = torch.zeros_like(labels, dtype=torch.float32)
    loss_mask.view(-1)[active_indices] = 1
    labels[loss_mask == 0] = -100
    loss_weights = torch.linspace(0.5, 1.5, labels.numel()).view_as(loss_mask)

    dense = _run_linear_ce(hidden, weight, labels, loss_mask, loss_weights, False)
    compact = _run_linear_ce(hidden, weight, labels, loss_mask, loss_weights, True)

    torch.testing.assert_close(compact[0], dense[0])
    torch.testing.assert_close(compact[1], dense[1])
    torch.testing.assert_close(compact[2], dense[2])
    assert dense[3] == [sequence * batch]
    assert compact[3] == ([] if expected_compact_rows == 0 else [expected_compact_rows])


def test_compaction_restores_per_token_loss_layout() -> None:
    torch.manual_seed(11)
    sequence, batch, hidden_size, vocab_size = 5, 2, 4, 13
    hidden = torch.randn(sequence, batch, hidden_size, dtype=torch.float64)
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.float64)
    labels = torch.randint(0, vocab_size, (batch, sequence))
    loss_mask = torch.tensor(
        [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]],
        dtype=torch.float32,
    )
    with (
        patch.object(loss_factory, "set_linear_ce_backend", return_value=None),
        patch.object(
            loss_factory,
            "linear_cross_entropy",
            side_effect=_LinearCERecorder(),
        ),
        patch.object(loss_factory.dist, "get_world_size", return_value=1),
    ):
        loss = loss_factory.linear_ce_loss(
            _config(True),
            _linear_ce_input(hidden, weight),
            labels,
            loss_mask,
            return_src_loss=True,
        )

    logits = hidden.transpose(0, 1) @ weight.t()
    expected = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    torch.testing.assert_close(loss[loss_mask.bool()], expected[loss_mask.bool()])
    assert torch.count_nonzero(loss[~loss_mask.bool()]) == 0


@pytest.mark.parametrize("active_indices", ([], [4], [0, 2, 5]))
def test_ordinary_ce_compaction_matches_dense_loss_and_gradients(
    active_indices: list[int],
) -> None:
    torch.manual_seed(19)
    sequence, batch, hidden_size, vocab_size = 4, 2, 6, 11
    hidden_data = torch.randn(sequence, batch, hidden_size, dtype=torch.float64)
    weight_data = torch.randn(vocab_size, hidden_size, dtype=torch.float64)
    labels = torch.randint(0, vocab_size, (batch, sequence))
    loss_mask = torch.zeros_like(labels, dtype=torch.float32)
    loss_mask.view(-1)[active_indices] = 1
    loss_weights = torch.linspace(0.5, 1.5, labels.numel()).view_as(loss_mask)

    hidden = hidden_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    projected_rows = []

    def record_vocab_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        projected_rows.append(logits.shape[0])
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            reduction="none",
        ).view_as(target)

    with (
        patch.object(
            loss_factory.tensor_parallel,
            "vocab_parallel_cross_entropy",
            side_effect=record_vocab_ce,
        ),
        patch.object(loss_factory.dist, "get_world_size", return_value=1),
        patch.object(loss_factory.mpu, "get_context_parallel_group", return_value=None),
    ):
        compact = loss_factory.compact_ce_loss(
            _config(True, use_linear_ce=False),
            _linear_ce_input(hidden, weight),
            labels,
            loss_mask,
            loss_weights=loss_weights,
        )
    compact[0].backward()

    dense_hidden = hidden_data.clone().requires_grad_(True)
    dense_weight = weight_data.clone().requires_grad_(True)
    dense_logits = dense_hidden.transpose(0, 1) @ dense_weight.t()
    dense_per_token = F.cross_entropy(
        dense_logits.reshape(-1, vocab_size).float(),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    dense_loss = (dense_per_token * loss_mask * loss_weights).sum()
    dense_loss.backward()

    torch.testing.assert_close(compact[0], dense_loss.detach())
    torch.testing.assert_close(compact[1], loss_mask.sum())
    torch.testing.assert_close(hidden.grad, dense_hidden.grad)
    torch.testing.assert_close(weight.grad, dense_weight.grad)
    assert projected_rows == ([] if not active_indices else [len(active_indices)])


def test_ordinary_ce_compaction_preserves_sp_tp_gradient_mappings() -> None:
    local_hidden = torch.randn(2, 1, 4, dtype=torch.float64)
    gathered_hidden = torch.randn(4, 1, 4, dtype=torch.float64)
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
            loss_factory.tensor_parallel,
            "gather_from_sequence_parallel_region",
            return_value=gathered_hidden,
        ) as gather,
        patch.object(
            loss_factory.tensor_parallel,
            "vocab_parallel_cross_entropy",
            side_effect=vocab_ce,
        ),
        patch.object(
            loss_factory.dist,
            "get_world_size",
            side_effect=lambda group: 2 if group is tp_group else 1,
        ),
        patch.object(loss_factory.mpu, "get_context_parallel_group", return_value=None),
        patch.object(
            loss_factory.tensor_parallel,
            "copy_to_tensor_model_parallel_region",
            side_effect=lambda value, group: value,
        ) as copy_to_tp,
    ):
        loss_factory.compact_ce_loss(
            _config(True, use_linear_ce=False),
            _linear_ce_input(
                local_hidden,
                weight,
                sequence_parallel=True,
                tp_group=tp_group,
            ),
            torch.tensor([[2, 3, 4, 5]]),
            torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        )

    gather.assert_called_once_with(local_hidden, tensor_parallel_output_grad=False)
    projected_hidden = copy_to_tp.call_args.args[0]
    torch.testing.assert_close(projected_hidden, gathered_hidden[[0, 2], 0].unsqueeze(1))
    assert copy_to_tp.call_args.kwargs == {"group": tp_group}


def test_compaction_injects_linear_fusion_impl() -> None:
    training = FinetuneTrainingConfig(ce_compaction=True)
    override_transformer_config = {}
    expected_bridge = object()
    harness = SimpleNamespace(
        training_config=training,
        checkpoint_config=SimpleNamespace(save_ckpt_path=None),
        config=SimpleNamespace(
            training=SimpleNamespace(
                apply_deterministic_mode=False,
                build_from_mbridge=True,
            )
        ),
        build_mbridge=Mock(return_value=expected_bridge),
    )

    with (
        patch.object(megatron_mixin, "cache_hf_metadata_files"),
        patch.object(megatron_mixin, "logging_rank0"),
    ):
        bridge = megatron_mixin.BridgeUtilsMixin.build_bridge(
            harness,
            "model-path",
            override_transformer_config,
        )

    assert bridge is expected_bridge
    assert override_transformer_config["cross_entropy_loss_fusion"]
    assert override_transformer_config["cross_entropy_fusion_impl"] == "linear"


def _run_real_linear_ce(
    hidden_data: torch.Tensor,
    weight_data: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    compaction: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = hidden_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    with (
        patch.object(loss_factory.dist, "get_world_size", return_value=1),
        patch.object(loss_factory.mpu, "get_context_parallel_group", return_value=None),
    ):
        output = loss_factory.linear_ce_loss(
            _config(compaction),
            _linear_ce_input(hidden, weight),
            labels,
            loss_mask,
        )
    output[0].backward()
    torch.cuda.synchronize()
    return output.detach(), hidden.grad, weight.grad


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.float(), expected.float()
    return (actual - expected).norm().item() / (expected.norm().item() + 1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("active_indices", ([], [3], [0, 5], [1, 2, 4, 6, 7]))
def test_real_kernel_compaction_matches_dense(active_indices: list[int]) -> None:
    # hidden_size 必须被 128 整除，是 efficient_entropy_forward 的硬约束
    sequence, batch, hidden_size, vocab_size = 4, 2, 128, 256
    generator = torch.Generator(device="cuda").manual_seed(20260811)
    hidden = torch.randn(
        sequence, batch, hidden_size, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(
        vocab_size, hidden_size, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    # label 全部落在词表内，dense 路径才有确定的参照值
    labels = torch.randint(vocab_size, (batch, sequence), generator=generator, device="cuda")
    loss_mask = torch.zeros(batch, sequence, device="cuda", dtype=torch.float32)
    loss_mask.view(-1)[active_indices] = 1

    dense = _run_real_linear_ce(hidden, weight, labels, loss_mask, False)
    compact = _run_real_linear_ce(hidden, weight, labels, loss_mask, True)

    torch.testing.assert_close(compact[0], dense[0], atol=5e-2, rtol=1e-2)
    assert _relative_error(compact[1], dense[1]) < 5e-3
    assert _relative_error(compact[2], dense[2]) < 5e-3


def _compaction_config(model_arch: str, use_linear_ce: bool = True) -> FinetuneConfig:
    return FinetuneConfig(
        training=FinetuneTrainingConfig(
            use_linear_ce=use_linear_ce,
            ce_compaction=True,
        ),
        policy=BasePolicyConfig(model_arch=model_arch),
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
def test_compaction_model_scope(model_arch: str, use_linear_ce: bool) -> None:
    config = _compaction_config(model_arch, use_linear_ce)
    assert config.training.ce_compaction
    assert config.training.return_hidden_states_for_ce


def test_compaction_rejects_unsupported_configuration() -> None:
    assert not FinetuneTrainingConfig().ce_compaction
    training = FinetuneTrainingConfig(ce_compaction=True)
    assert not training.use_linear_ce
    assert training.return_hidden_states_for_ce
    with pytest.raises(AssertionError, match="requires cross_entropy_fusion_impl"):
        FinetuneTrainingConfig(
            ce_compaction=True,
            cross_entropy_loss_fusion=True,
            cross_entropy_fusion_impl="linear",
        )
    with pytest.raises(AssertionError, match="MCore SFT only"):
        FinetuneTrainingConfig(
            training_backend="fsdp2",
            use_linear_ce=True,
            ce_compaction=True,
        )
    with pytest.raises(AssertionError, match="has not been validated"):
        FinetuneTrainingConfig(
            apply_deterministic_mode=True,
            use_linear_ce=True,
            ce_compaction=True,
        )


def test_compaction_rejects_off_policy_distillation() -> None:
    with pytest.raises(AssertionError, match="currently supports"):
        OffPolicyDistillTrainingConfig(
            use_linear_ce=True,
            ce_compaction=True,
        )


def test_compaction_rejects_dpo() -> None:
    with pytest.raises(AssertionError, match="not supported for DPO"):
        DpoConfig(training=DpoTrainingConfig(ce_compaction=True))
    with pytest.raises(AssertionError, match="not supported for DPO"):
        DpoConfig(training=DpoTrainingConfig(use_linear_ce=True))


@pytest.mark.parametrize(
    "model_arch",
    (
        MODEL_ARCH.QWEN3,
        MODEL_ARCH.QWEN3_MOE,
        MODEL_ARCH.WELM_MOE,
        MODEL_ARCH.QWEN3_5_WEMM,
        MODEL_ARCH.QWEN3_OMNI_MOE,
        MODEL_ARCH.WELM_OMNI_V4_5,
    ),
)
def test_compaction_rejects_unsupported_model(model_arch: str) -> None:
    with pytest.raises(AssertionError, match="Qwen3-VL/Qwen3.5/Qwen3.6 or WeLM v4.5"):
        _compaction_config(model_arch)


def test_compaction_requires_mbridge() -> None:
    with pytest.raises(AssertionError, match="build_from_mbridge=True"):
        FinetuneConfig(
            training=FinetuneTrainingConfig(
                build_from_mbridge=False,
                ce_compaction=True,
            ),
            policy=BasePolicyConfig(model_arch=MODEL_ARCH.QWEN3_VL),
        )
