import copy

import torch
from packaging import version

from megatron.core.optimizer import OptimizerConfig as McoreOptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer as mcore_get_megatron_optimizer

try:
    from megatron.core.optimizer import ParamKey
except:
    pass
from megatron.core import __version__
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler


def get_megatron_optimizer_config_overrides(model, config, optimizer_config):

    # Construct the appropriate config_overrides object.
    # TODO: add more logic here as needed down the road.
    config_overrides = {}
    vit_lr = optimizer_config.get('vit_lr', None)
    vit_min_lr = optimizer_config.get('vit_min_lr', None)
    vit_proj_lr = optimizer_config.get('vit_proj_lr', None)
    vit_proj_min_lr = optimizer_config.get('vit_proj_min_lr', None)

    if vit_lr is not None:
        decoupled_optimizer_config = copy.deepcopy(config)
        decoupled_optimizer_config.lr = vit_lr
        if vit_min_lr is not None:
            decoupled_optimizer_config.min_lr = vit_min_lr

        for name, param in model[0].named_parameters():
            if name.startswith(f'module.module.vision_model'):
                if vit_proj_lr is None or not name.startswith(f'module.module.vision_model.merger'):
                    decoupled_param_key = ParamKey(name=name)
                    config_overrides[decoupled_param_key] = decoupled_optimizer_config

    if vit_proj_lr is not None:
        decoupled_optimizer_config = copy.deepcopy(config)
        decoupled_optimizer_config.lr = vit_proj_lr
        if vit_proj_min_lr is not None:
            decoupled_optimizer_config.min_lr = vit_proj_min_lr

        for name, param in model[0].named_parameters():
            if name.startswith(f'module.module.vision_model.merger'):
                decoupled_param_key = ParamKey(name=name)
                config_overrides[decoupled_param_key] = decoupled_optimizer_config

    return config_overrides if config_overrides else None


def setup_megatron_optim_config(model, optimizer_config) -> McoreOptimizerConfig:
    optim_args = {
        "optimizer": optimizer_config.optimizer_type,
        "lr": optimizer_config.lr,
        "min_lr": optimizer_config.min_lr,
        "clip_grad": optimizer_config.max_grad_norm,
        "adam_beta1": optimizer_config.adam_beta1,
        "adam_beta2": optimizer_config.adam_beta2,
        "adam_eps": optimizer_config.adam_epsilon,
        "weight_decay": optimizer_config.weight_decay,
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "use_distributed_optimizer": True,
    }

    override_config = optimizer_config.override_optimizer_config
    if override_config:
        for k, v in override_config.items():
            optim_args[k] = v

    config = McoreOptimizerConfig(**optim_args)

    if version.parse(__version__) <= version.parse('0.15.0'):
        config_overrides = None
        assert optimizer_config.get("vit_lr", None) == None
        assert optimizer_config.get("vit_min_lr", None) == None
        assert optimizer_config.get("vit_proj_lr", None) == None
        assert optimizer_config.get("vit_proj_min_lr", None) == None
    else:
        config_overrides = get_megatron_optimizer_config_overrides(model, config, optimizer_config)
    return config, config_overrides


def get_megatron_optimizer(
    model,
    optimizer_config,
):
    mcore_optimizer_config, config_overrides = setup_megatron_optim_config(model, optimizer_config)
    # Base optimizer.
    extra_kwargs = {}
    if version.parse(__version__) > version.parse('0.15.0'):
        extra_kwargs['config_overrides'] = config_overrides
    return mcore_get_megatron_optimizer(
        config=mcore_optimizer_config,
        model_chunks=model,
        # param it
        use_gloo_process_groups=False,
        **extra_kwargs,
    )


def get_megatron_optimizer_param_scheduler(
    optimizer,
    config,
):
    """Megatron optimizer parameter scheduler."""
    lr_decay_steps = config.optimizer.lr_decay_steps
    lr_warmup_steps = config.optimizer.lr_warmup_steps
    update_lr_by_train_step = config.optimizer.get("update_lr_by_train_step", False)

    if update_lr_by_train_step:
        assert config.optimizer.get("lr_decay_steps", None) is None
        assert config.optimizer.get("lr_wsd_decay_steps", None) is None

    if config.training.get("total_ppo_step", None) is not None:
        wd_incr_steps = config.training.total_ppo_step
        if update_lr_by_train_step:
            train_iters_per_ppo_step = (
                config.training.rollout_gbs * config.training.sampling_keep_n //
                config.training.train_gbs * config.training.ppo_max_epochs_2
            )
            wd_incr_steps *= train_iters_per_ppo_step

    else:
        wd_incr_steps = config.training.total_training_step

    if config.optimizer.get("lr_decay_steps", None) is None:
        lr_decay_steps = wd_incr_steps

    wsd_decay_steps = None
    if config.optimizer.get("lr_wsd_decay_steps", None) is not None:
        wsd_decay_steps = config.lr_wsd_decay_steps
    if config.optimizer.get("lr_warmup_step_frac", None) is not None and (
        config.optimizer.get("lr_warmup_steps", None) is None or
        config.optimizer.lr_warmup_steps <= 0
    ):
        lr_warmup_steps = int(config.optimizer.lr_warmup_step_frac * lr_decay_steps)

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=config.optimizer.lr_warmup_init,
        max_lr=config.optimizer.lr,
        min_lr=config.optimizer.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=config.optimizer.lr_decay_style,
        start_wd=config.optimizer.weight_decay,
        end_wd=config.optimizer.weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=config.optimizer.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=config.optimizer.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=(not config.optimizer.use_checkpoint_opt_param_scheduler),
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=config.optimizer.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def get_optimizer_and_scheduler(config, model):
    optimizer = get_megatron_optimizer(model, config.optimizer)
    optimizer_scheduler = get_megatron_optimizer_param_scheduler(optimizer=optimizer, config=config)
    return optimizer, optimizer_scheduler


def get_megatron_last_lr(optimizer):
    """Last lr from the optimizer."""
    return optimizer.param_groups[0]["lr"]
