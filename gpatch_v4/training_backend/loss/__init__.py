"""Loss package (``ppo.use_legacy_loss=False`` for PPO path)."""

from gpatch_v4.training_backend.loss import mcore_specific_loss as _mcore_specific_loss
from gpatch_v4.training_backend.loss.fsdp2_specific_loss import Fsdp2FinetuneLossInput
from gpatch_v4.training_backend.loss.ppo_loss import PolicyLossInput
from gpatch_v4.training_backend.loss.registry import (
    get_loss_fn,
    is_loss_registered,
    register_custom_loss_fn,
    register_loss,
)

__all__ = [
    "Fsdp2FinetuneLossInput",
    "PolicyLossInput",
    "get_loss_fn",
    "is_loss_registered",
    "register_custom_loss_fn",
    "register_loss",
]
