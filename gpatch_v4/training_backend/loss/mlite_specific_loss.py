from dataclasses import dataclass

import torch

from gpatch_v4.configs import FinetuneConfig
from gpatch_v4.training_backend.loss.registry import register_loss


@dataclass(slots=True)
class MliteFinetuneLossInput:
    log_probs: torch.Tensor
    aligned_loss_mask: torch.Tensor
    global_valid_tokens: torch.Tensor
    dp_size: int


@dataclass(slots=True)
class MliteFinetuneLossResult:
    loss: torch.Tensor
    local_loss_sum: torch.Tensor
    local_valid_tokens: torch.Tensor


@register_loss(
    backends=("mlite", ),
    loss_name="cross_entropy",
    log_registration=True,
)
def mlite_cross_entropy_loss(
    config: FinetuneConfig,
    loss_input: MliteFinetuneLossInput,
) -> MliteFinetuneLossResult:
    del config
    log_probs = loss_input.log_probs
    if getattr(log_probs, "is_nested", False):
        log_probs = log_probs.values()
    log_probs = log_probs.reshape(-1)
    loss_mask = loss_input.aligned_loss_mask.reshape(-1).to(log_probs.device)
    if log_probs.numel() != loss_mask.numel():
        raise ValueError(
            f"unpacked log_probs and aligned loss mask differ: "
            f"{log_probs.numel()} != {loss_mask.numel()}"
        )
    if loss_input.global_valid_tokens.item() <= 0:
        raise ValueError("global_valid_tokens must be positive")

    local_valid_tokens = loss_mask.sum().to(dtype=torch.float32)
    local_loss_sum = -(log_probs.float() * loss_mask.float()).sum()
    loss = (
        local_loss_sum / loss_input.global_valid_tokens.to(local_loss_sum.device) *
        loss_input.dp_size
    )
    return MliteFinetuneLossResult(
        loss=loss,
        local_loss_sum=local_loss_sum.detach(),
        local_valid_tokens=local_valid_tokens.detach(),
    )
