import torch

from gpatch_v4.training_backend.fsdp2_backend.lr_scheduler import get_lr_scheduler


def setup_optimizer(config, model, latest_step=None):
    params_to_optimize = model.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer_config = config.optimizer
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=optimizer_config.lr,
        betas=(optimizer_config.adam_beta1, optimizer_config.adam_beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.adam_epsilon,
    )
    if latest_step is not None and (not config.checkpoint.no_load_optim):
        pass

    return optimizer


def setup_lr_scheduler(config, optimizer, latest_step=None):
    lr_scheduler = get_lr_scheduler(config, optimizer)

    if latest_step is not None and (not config.checkpoint.no_load_optim):
        pass

    return lr_scheduler
