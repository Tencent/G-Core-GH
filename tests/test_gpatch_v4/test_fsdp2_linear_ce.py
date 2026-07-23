import pytest
import torch
import torch.nn.functional as F
from torch import nn

from gpatch_v4.training_backend.fsdp2_backend.linear_ce import (
    install_linear_ce_head_bypass,
    linear_ce_head_context,
    linear_ce_forward_context,
)
from gpatch_v4.training_backend.fsdp2_backend.mtp_loss import calculate_mtp_loss
from gpatch_v4.utils.training_utils import selective_log_softmax_raw


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_BATCH = 2
_SEQUENCE = 16
_HIDDEN = 128
_VOCAB = 256
_ATOL = 5e-2
_RTOL = 1e-2


class _ModelWithLinearHead(nn.Module):
    def __init__(self, head: nn.Linear) -> None:
        super().__init__()
        self.lm_head = head

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(1234)
    hidden = torch.randn(
        _BATCH,
        _SEQUENCE,
        _HIDDEN,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        _VOCAB,
        _HIDDEN,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    labels = torch.randint(
        _VOCAB,
        (_BATCH, _SEQUENCE),
        generator=generator,
        device="cuda",
    )
    labels[:, -2:] = -100
    return hidden, weight, labels


def _reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = F.linear(hidden.float(), weight.float())
    safe_labels = labels.clamp_min(0)
    token_log_probs = selective_log_softmax_raw(logits, safe_labels)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    loss_mask = (labels != -100).to(token_log_probs.dtype)
    loss = -(token_log_probs * loss_mask).sum() / loss_mask.sum()
    return token_log_probs, entropy, loss


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).norm().item() / (expected.float().norm().item() + 1e-8)


def test_linear_ce_head_matches_dense_forward_and_backward() -> None:
    hidden, weight, labels = _make_inputs()
    fused_hidden = hidden.detach().clone().requires_grad_(True)
    fused_weight = weight.detach().clone().requires_grad_(True)
    head = nn.Linear(_HIDDEN, _VOCAB, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        head.weight.copy_(fused_weight)
    model = _ModelWithLinearHead(head)
    install_linear_ce_head_bypass(model)

    with linear_ce_forward_context(model, labels, "split_n"):
        fused_log_probs = head(fused_hidden)
    assert isinstance(fused_log_probs, torch.Tensor)
    loss_mask = (labels != -100).to(fused_log_probs.dtype)
    fused_loss = -(fused_log_probs * loss_mask).sum() / loss_mask.sum()
    fused_loss.backward()

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference_log_probs, reference_entropy, reference_loss = _reference(
        reference_hidden,
        reference_weight,
        labels,
    )
    reference_loss.backward()

    torch.cuda.synchronize()
    valid = labels != -100
    torch.testing.assert_close(
        fused_log_probs[valid].float(),
        reference_log_probs[valid],
        atol=_ATOL,
        rtol=_RTOL,
    )
    assert _relative_error(fused_hidden.grad, reference_hidden.grad) < 5e-3
    assert _relative_error(head.weight.grad, reference_weight.grad) < 5e-3

    with torch.no_grad():
        with linear_ce_forward_context(model, labels, "split_n", return_entropy=True):
            entropy_output = head(hidden)
    assert isinstance(entropy_output, tuple) and len(entropy_output) == 2
    entropy_log_probs, fused_entropy = entropy_output
    torch.testing.assert_close(
        entropy_log_probs[valid].float(),
        reference_log_probs[valid],
        atol=_ATOL,
        rtol=_RTOL,
    )
    torch.testing.assert_close(fused_entropy.float(), reference_entropy, atol=_ATOL, rtol=_RTOL)

    dense_output = head(hidden)
    assert isinstance(dense_output, torch.Tensor)


def test_linear_ce_mtp_matches_dense_loss_numerators() -> None:
    hidden, weight, labels = _make_inputs()
    loss_mask = (labels != -100).to(torch.float32)
    head = nn.Linear(_HIDDEN, _VOCAB, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        head.weight.copy_(weight)
    mtp_per_depth_h = [hidden, hidden * 0.5]
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    dense_numerators = calculate_mtp_loss(
        mtp_per_depth_h=mtp_per_depth_h,
        labels=labels,
        lm_head=head,
        loss_fct=loss_fct,
        loss_mask=loss_mask,
    )

    model = _ModelWithLinearHead(head)
    install_linear_ce_head_bypass(model)
    fused_numerators = calculate_mtp_loss(
        mtp_per_depth_h=mtp_per_depth_h,
        labels=labels,
        lm_head=head,
        loss_mask=loss_mask,
        use_linear_ce=True,
        linear_ce_backend="separate",
    )

    torch.cuda.synchronize()
    assert len(fused_numerators) == len(dense_numerators)
    for fused, dense in zip(fused_numerators, dense_numerators):
        torch.testing.assert_close(fused.float(), dense.float(), atol=_ATOL, rtol=_RTOL)
