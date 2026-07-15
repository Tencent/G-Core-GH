import inspect

import torch

from megatron.core import tensor_parallel
from megatron.core.utils import get_model_config

try:
    from megatron.bridge import AutoBridge
except ImportError as e:
    print(
        "Megatron-Bridge package not found. Please install Megatron-Bridge with `pip install megatron-bridge`"
    )
    pass


def _ensure_model_list(model):
    return model if isinstance(model, list) else [model]


def freeze_multimodal(
    model,
    freeze_language_model=False,
    freeze_vision_model=False,
    freeze_vision_projection=False,
    freeze_audio_model=False,
    freeze_audio_qformer=False,
    freeze_audio_projection=False,
):
    for model_chunk in _ensure_model_list(model):
        kwargs = {}
        fn_kwargs = inspect.signature(model_chunk.freeze).parameters
        print(f"{fn_kwargs=}")
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
        if "freeze_audio_projection" in fn_kwargs:
            kwargs["freeze_audio_projection"] = freeze_audio_projection
        model_chunk.freeze(**kwargs)

    return model


def freeze_moe_router(model, freeze_moe_shared_experts=False):
    """Pre-wrap hook to freeze MoE router parameters.

    Args:
        model: list of MegatronModule or single module.

    Returns:
        Model with frozen router parameters.
    """
    for model_chunk in _ensure_model_list(model):
        if hasattr(model_chunk, "decoder") and hasattr(model_chunk.decoder, "layers"):
            for layer in model_chunk.decoder.layers:
                if hasattr(layer.mlp, "router"):
                    if hasattr(layer.mlp.router, "weight"):
                        layer.mlp.router.weight.requires_grad = False
                    if hasattr(layer.mlp.router, "bias") and layer.mlp.router.bias is not None:
                        layer.mlp.router.bias.requires_grad = False
                if freeze_moe_shared_experts and hasattr(layer.mlp, "shared_experts"):
                    if hasattr(layer.mlp.shared_experts, "gate_weight"):
                        layer.mlp.shared_experts.gate_weight.requires_grad = False
                    if hasattr(layer.mlp.shared_experts, "gate_bias"):
                        layer.mlp.shared_experts.gate_bias.requires_grad = False

    return model


class LinearForLastLayer(torch.nn.Linear):
    """Linear layer for the final transformer layer with sequence parallelism.

    Attributes:
        sequence_parallel: Whether sequence parallelism is enabled.
    """
    def __init__(
        self,
        input_size,
        output_size,
        *,
        config,
        bias=False,
    ):
        """Initialize.

        Args:
            input_size:
            output_size:
            config: Configuration with parallelism settings.
            bias: Whether to include a bias term.
        """
        super().__init__(in_features=input_size, out_features=output_size, bias=bias)
        self.sequence_parallel = config.sequence_parallel
        if self.sequence_parallel:
            self.weight.sequence_parallel = True

    def forward(
        self,
        input_,
        weight=None,
        runtime_gather_output=None,
    ):
        """Forward; gathers outputs across sequence-parallel regions if enabled.

        Args:
            input_:
            weight: optional weight to use instead of self.weight.
            runtime_gather_output: optional runtime gather param.

        Returns:
            tuple: ``(logits, None)``.
        """
        logits = super().forward(input_)
        logits = logits.float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(
                logits, tensor_parallel_output_grad=False
            )
        return logits, None


def make_value_model(model):
    for model_chunk in _ensure_model_list(model):
        model_chunk.share_embeddings_and_output_weights = False
        config = get_model_config(model_chunk)
        if hasattr(model_chunk, "output_layer"):
            model_chunk.output_layer = LinearForLastLayer(
                input_size=config.hidden_size,
                output_size=1,
                config=config,
            )


__all__ = [
    "AutoBridge",
    "make_value_model",
    "freeze_multimodal",
    "freeze_moe_router",
]
