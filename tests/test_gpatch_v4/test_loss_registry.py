import pytest

from gpatch_v4.training_backend.loss import get_loss_fn, is_loss_registered
from gpatch_v4.training_backend.loss.fsdp2_specific_loss import (
    fsdp2_cross_entropy_loss,
    fsdp2_dpo_loss,
)
from gpatch_v4.training_backend.loss.ppo_loss import steer_loss
from gpatch_v4.training_backend.loss_factory import (
    cross_entroy_loss_func,
    dpo_loss_func,
    rm_bt_loss_func,
    value_loss_func,
)


def test_mcore_specific_losses_are_registered():
    assert get_loss_fn("mcore", "steer") is steer_loss
    assert get_loss_fn("mcore", "cross_entropy") is cross_entroy_loss_func
    assert get_loss_fn("mcore", "dpo") is dpo_loss_func
    assert get_loss_fn("mcore", "rm_bt") is rm_bt_loss_func
    assert get_loss_fn("mcore", "ppo_value_loss") is value_loss_func


def test_fsdp2_specific_losses_are_registered():
    assert get_loss_fn("fsdp2", "cross_entropy") is fsdp2_cross_entropy_loss
    assert get_loss_fn("fsdp2", "dpo") is fsdp2_dpo_loss


def test_loss_lookup_does_not_cross_backend_boundaries():
    assert is_loss_registered("mcore", "grpo")
    assert not is_loss_registered("fsdp2", "rm_bt")
    with pytest.raises(ValueError, match="Unknown fsdp2 loss type: 'rm_bt'"):
        get_loss_fn("fsdp2", "rm_bt")
