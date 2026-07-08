import inspect
import re
from typing import List, Optional

try:
    from mbridge import AutoBridge
    from mbridge.utils.post_creation_callbacks import make_value_model
except ImportError:
    print(
        "mbridge package not found. Please install mbridge with `pip install verl[mcore]` or `pip install mbridge`"
    )
    raise

from gpatch_v4.utils import log


### freeze moe router
def freeze_moe_router(
    model, pre_process, post_process, config, hf_config, freeze_moe_shared_experts=False
):
    for layer in model.decoder.layers:
        if hasattr(layer.mlp, "router"):
            log(f"moe warning: freeze moe router weight", rank=0)
            if hasattr(layer.mlp.router, "weight"):
                layer.mlp.router.weight.requires_grad = False
            if hasattr(layer.mlp.router, "bias") and layer.mlp.router.bias is not None:
                layer.mlp.router.bias.requires_grad = False
        if freeze_moe_shared_experts and hasattr(layer.mlp, "shared_experts"):
            log(f"moe warning: freeze moe shared_experts={freeze_moe_shared_experts}", rank=0)
            if hasattr(layer.mlp.shared_experts, "gate_weight") and \
                    layer.mlp.shared_experts.gate_weight is not None:
                layer.mlp.shared_experts.gate_weight.requires_grad = False
            if hasattr(layer.mlp.shared_experts, "gate_bias"):
                layer.mlp.shared_experts.gate_bias.requires_grad = False


def freeze_multimodal(
    model,
    pre_process,
    post_process,
    config,
    hf_config,
    freeze_language_model=False,
    freeze_vision_model=False,
    freeze_vision_projection=False,
    freeze_audio_model=False,
    freeze_audio_qformer=False,
):
    kwargs = {}
    fn_kwargs = inspect.signature(model.freeze).parameters
    if "freeze_language_model" in fn_kwargs:
        kwargs["freeze_language_model"] = freeze_language_model
    if "freeze_vision_model" in fn_kwargs:
        kwargs["freeze_vision_model"] = freeze_vision_model
    if "freeze_vision_projection" in fn_kwargs:
        kwargs["freeze_vision_projection"] = freeze_vision_projection
    if "freeze_audio_model" in fn_kwargs:
        kwargs["freeze_audio_model"] = freeze_audio_model
    if "freeze_audio_qformer" in fn_kwargs:
        kwargs["freeze_audio_qformer"] = freeze_audio_qformer
    model.freeze(**kwargs)


def apply_freeze_unfreeze_patterns(
    model,
    freeze_patterns: Optional[List[str]] = None,
    unfreeze_patterns: Optional[List[str]] = None,
):
    """Freeze/unfreeze parameters by wildcard-matching their names.

    Matching uses the same ``*`` wildcard syntax (``*`` matches any substring)
    as mbridge's LoRA ``target_modules``/``exclude_modules``, via
    ``mbridge.peft.utils.wildcard_match`` (see ``ModuleMatcher.match``).

    ``unfreeze_patterns`` takes priority over ``freeze_patterns``: a parameter
    matching both stays/becomes trainable.

    Parameters
    ----------
    model : torch.nn.Module
        A single model chunk (e.g. one element of the ``model`` list returned
        by ``bridge.get_model``).
    freeze_patterns : list of str, optional
        Parameter names matching any of these patterns get ``requires_grad =
        False``.
    unfreeze_patterns : list of str, optional
        Parameter names matching any of these patterns get ``requires_grad =
        True``, overriding ``freeze_patterns``.
    """
    def _wildcard_match(pattern: str, key: Optional[str]) -> Optional[bool]:
        if key is None:
            return None
        regex_pattern = re.compile("^" + pattern.replace("*", "(.*)") + "$")
        match = regex_pattern.match(key)
        return match is not None

    freeze_patterns = freeze_patterns or []
    unfreeze_patterns = unfreeze_patterns or []
    if not freeze_patterns and not unfreeze_patterns:
        return

    for name, param in model.named_parameters():
        if any(_wildcard_match(pattern, name) for pattern in freeze_patterns):
            param.requires_grad = False

    for name, param in model.named_parameters():
        if any(_wildcard_match(pattern, name) for pattern in unfreeze_patterns):
            param.requires_grad = True


__all__ = [
    "AutoBridge",
    "make_value_model",
    "freeze_moe_router",
    "freeze_multimodal",
    "apply_freeze_unfreeze_patterns",
]
