"""FSDP utilities shared by in-training validation runners."""

DEFAULT_FSDP_INFERENCE_METHODS = (
    "forward_cache_update_text",
    "forward_cache_update_vae",
    "forward_cache_update_vit",
    "generate_text",
    "generate_image",
)


def register_fsdp_forward_methods(model, method_names=DEFAULT_FSDP_INFERENCE_METHODS):
    """Register custom inference methods so FSDP pre/post hooks fire."""
    if getattr(model, "_val_fsdp_methods_registered", False):
        return

    try:
        from torch.distributed._composable.fsdp import register_fsdp_forward_method
    except ImportError:
        return

    for name in method_names:
        if hasattr(model, name):
            register_fsdp_forward_method(model, name)

    model._val_fsdp_methods_registered = True


def snapshot_module_modes(model):
    return [(module, module.training) for module in model.modules()]


def restore_module_modes(mode_snapshot):
    for module, was_training in mode_snapshot:
        module.train(was_training)


def reshard_fsdp_modules(model):
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        return

    # Validation may leave some FSDP modules unsharded after custom forward-method inference.
    for module in reversed(list(model.modules())):
        if not isinstance(module, FSDPModule):
            continue
        set_reshard_after_forward = getattr(module, "set_reshard_after_forward", None)
        if callable(set_reshard_after_forward):
            set_reshard_after_forward(True, recurse=False)
        module.reshard()


def restore_model_for_training(model, mode_snapshot):
    reshard_fsdp_modules(model)
    restore_module_modes(mode_snapshot)
