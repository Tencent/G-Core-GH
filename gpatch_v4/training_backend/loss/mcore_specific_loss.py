# #TODO: loss refact ing. Move MCore-specific implementations from loss_factory.py into this module.
# TODO: Remove these registration bindings after the legacy loss registry is retired.

from gpatch_v4.training_backend.loss.registry import register_loss
from gpatch_v4.training_backend.loss_factory import (
    cross_entroy_loss_func,
    dpo_loss_func,
    grad_cache_loss_replay_func,
    off_policy_loss_func,
    rm_bt_loss_func,
    square_averaging_cross_entroy_loss_func,
    value_loss_func,
)

register_loss(("mcore", ), "cross_entropy", log_registration=True)(cross_entroy_loss_func)
register_loss(("mcore", ), "ce_with_kl")(off_policy_loss_func)
register_loss(("mcore", ), "dpo")(dpo_loss_func)
register_loss(("mcore", ),
              "square_averaging_cross_entropy")(square_averaging_cross_entroy_loss_func)
register_loss(("mcore", ), "ppo_value_loss")(value_loss_func)
register_loss(("mcore", ), "rm_bt")(rm_bt_loss_func)
register_loss(("mcore", ), "grad_cache_loss")(grad_cache_loss_replay_func)
