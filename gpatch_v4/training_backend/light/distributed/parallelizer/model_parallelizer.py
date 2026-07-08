"""ModelParallelizer orchestration, registry, and ``parallelize_model`` entry."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Callable, Optional, Type

import torch.nn as nn

from ..core.config import ParallelConfig
from ..core.parallel_state import ParallelState, get_parallel_state
from ..core.utils import (
    is_activation_checkpoint_inner_layer,
    module_matches_cls_name,
    module_matches_layer_cls,
)
from .ep import ExpertsParallelPlan, build_experts_parallel_plan


def set_modules_to_forward_prefetch(layers, num_to_forward_prefetch: int) -> None:
    """Register the next ``num_to_forward_prefetch`` FSDP layers as forward-prefetch targets.

    ``layers`` must be ordered (execution order) and already wrapped by ``fully_shard``.
    """
    for i, layer in enumerate(layers):
        if i >= len(layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [layers[i + j] for j in range(1, num_to_forward_prefetch + 1)]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(layers, num_to_backward_prefetch: int) -> None:
    """Register the previous ``num_to_backward_prefetch`` FSDP layers as backward-prefetch targets.

    ``layers`` must be ordered (execution order) and already wrapped by ``fully_shard``.
    """
    for i, layer in enumerate(layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [layers[i - j] for j in range(1, num_to_backward_prefetch + 1)]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


@dataclass
class ParallelizeContext:
    config: ParallelConfig
    state: ParallelState
    transformer_layer_cls: str
    experts_cls: Optional[str] = None
    reshard_after_forward: bool = True
    mixed_precision: str = "bf16"
    enable_ac: bool = False
    use_reentrant_ac: bool = False
    num_forward_prefetch: int = 0
    num_backward_prefetch: int = 0
    reduce_scatter_ring_buffer: int = 0


class ModelParallelizer(ABC):
    """Per-model parallelization strategy (Automodel / VeOmni / Accelerate style)."""

    model_types: tuple[str, ...] = ()

    def experts_parallel_plan(self, model: nn.Module,
                              ctx: ParallelizeContext) -> Optional[ExpertsParallelPlan]:
        if not ctx.config.ep_enabled:
            return None
        for hook in ("get_experts_parallel_plan", "get_parallel_plan"):
            getter = getattr(model, hook, None)
            if getter is not None:
                plan = getter()
                if isinstance(plan, ExpertsParallelPlan):
                    return plan
        experts_cls = _require_cls_name("experts_cls", ctx.experts_cls or "")
        return build_experts_parallel_plan(model, experts_cls)

    def layer_wrap_policy(self, model: nn.Module,
                          ctx: ParallelizeContext) -> Callable[[nn.Module], bool]:
        """FSDP wrap predicate matched by ``ctx.transformer_layer_cls``."""
        cls_name = ctx.transformer_layer_cls
        return lambda m: module_matches_layer_cls(m, cls_name)

    def apply_cp(self, model: nn.Module, ctx: ParallelizeContext) -> nn.Module:
        return model

    def apply_ep(self, model: nn.Module, ctx: ParallelizeContext) -> nn.Module:
        if not ctx.config.ep_enabled:
            return model
        from ..moe import bind_ep_experts_model

        experts_cls = _require_cls_name("experts_cls", ctx.experts_cls or "")
        return bind_ep_experts_model(
            model, dispatch=ctx.config.ep_dispatch, experts_cls=experts_cls
        )

    def apply_fsdp(self, model: nn.Module, ctx: ParallelizeContext) -> nn.Module:
        from .fsdp import wrap_model_fsdp2

        experts_plan: Optional[ExpertsParallelPlan] = None
        if ctx.config.ep_enabled:
            experts_plan = self.experts_parallel_plan(model, ctx)
            if experts_plan is None:
                raise ValueError("EP enabled but no ExpertsParallelPlan available for this model.")
        return wrap_model_fsdp2(
            model,
            auto_wrap_policy=self.layer_wrap_policy(model, ctx),
            experts_parallel_plan=experts_plan,
            reshard_after_forward=ctx.reshard_after_forward,
            mixed_precision=ctx.mixed_precision,
            state=ctx.state,
            reduce_scatter_ring_buffer=ctx.reduce_scatter_ring_buffer,
        )

    def apply_ac(self, model: nn.Module, ctx: ParallelizeContext) -> nn.Module:
        """Activation checkpointing before FSDP (Automodel / VeOmni / Accelerate)."""
        if not ctx.enable_ac:
            return model

        _use_hf_native_grad_ckpt = False
        try:
            from transformers.modeling_layers import (
                GradientCheckpointingLayer as _HFGradLayer,
            )
            _use_hf_native_grad_ckpt = (
                bool(layers) and layers[0].__class__.__module__.startswith("transformers.") and
                isinstance(layers[0], _HFGradLayer) and
                getattr(model, "supports_gradient_checkpointing", False) and
                hasattr(model, "gradient_checkpointing_enable")
            )
        except:
            pass

        if _use_hf_native_grad_ckpt:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": ctx.use_reentrant_ac}
            )
            return model

        return self._apply_ac_checkpoint_wrapper(model, ctx)

    def _apply_ac_checkpoint_wrapper(self, model: nn.Module, ctx: ParallelizeContext) -> nn.Module:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper as ptd_checkpoint_wrapper,
        )

        cls_name = ctx.transformer_layer_cls
        for name, module in model.named_modules():
            if module_matches_cls_name(module, cls_name) and module is not model:
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                attr = name.rsplit(".", 1)[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                if not hasattr(parent, attr):
                    continue
                wrapped = ptd_checkpoint_wrapper(module, preserve_rng_state=True)
                setattr(parent, attr, wrapped)
        return model

    def _collect_transformer_layers(self, model: nn.Module,
                                    ctx: ParallelizeContext) -> list[nn.Module]:
        """Collect decoder layers (in execution order) matched by ``transformer_layer_cls``.

        Mirrors ``fsdp._collect_decoder_layers``: skips the root module and the
        inner layer hidden behind an activation-checkpoint wrapper, so the result
        contains exactly the ``fully_shard``-wrapped modules in definition order.
        """
        predicate = self.layer_wrap_policy(model, ctx)
        layers: list[nn.Module] = []
        for name, module in model.named_modules():
            if module is model:
                continue
            if is_activation_checkpoint_inner_layer(name):
                continue
            if predicate(module):
                layers.append(module)
        return layers

    def apply_prefetch(self, model: nn.Module, ctx: ParallelizeContext) -> nn.Module:
        """Set FSDP2 forward/backward prefetch on the wrapped transformer layers."""
        if ctx.num_forward_prefetch <= 0 and ctx.num_backward_prefetch <= 0:
            return model
        layers = self._collect_transformer_layers(model, ctx)
        if not layers:
            return model
        if ctx.num_forward_prefetch > 0:
            set_modules_to_forward_prefetch(layers, ctx.num_forward_prefetch)
        if ctx.num_backward_prefetch > 0:
            set_modules_to_backward_prefetch(layers, ctx.num_backward_prefetch)
        return model

    def parallelize(self, model: nn.Module, ctx: ParallelizeContext) -> nn.Module:
        """Apply parallel transforms: CP → EP → AC → FSDP2 → prefetch.

        Matches Automodel MoE ``parallelize_model`` (CP, EP, AC, FSDP) and
        VeOmni/Accelerate convention that AC runs before FSDP wrap. Forward/backward
        prefetch is configured after FSDP wrap.
        """
        model = self.apply_cp(model, ctx)
        if ctx.config.ep_enabled:
            model = self.apply_ep(model, ctx)
        model = self.apply_ac(model, ctx)
        if ctx.config.fsdp_enabled or ctx.config.ep_enabled:
            model = self.apply_fsdp(model, ctx)
            model = self.apply_prefetch(model, ctx)
        return model


class ParallelizerRegistry:
    _by_model_type: dict[str, Type[ModelParallelizer]] = {}
    _default: Type[ModelParallelizer] | None = None

    @classmethod
    def register(
        cls,
        *model_types: str,
        default: bool = False,
    ) -> Callable[[Type[ModelParallelizer]], Type[ModelParallelizer]]:
        def decorator(parallelizer_cls: Type[ModelParallelizer]) -> Type[ModelParallelizer]:
            for model_type in model_types:
                cls._by_model_type[model_type] = parallelizer_cls
            if default or cls._default is None:
                cls._default = parallelizer_cls
            parallelizer_cls.model_types = model_types
            return parallelizer_cls

        return decorator

    @classmethod
    def get(cls, model: nn.Module) -> ModelParallelizer:
        model_type = resolve_model_type(model)
        parallelizer_cls = cls._by_model_type.get(model_type, cls._default)
        if parallelizer_cls is None:
            raise KeyError(
                f"No parallelizer registered for model_type={model_type!r}. "
                "Register one with @ParallelizerRegistry.register(...)."
            )
        return parallelizer_cls()

    @classmethod
    def list_registered(cls) -> dict[str, Type[ModelParallelizer]]:
        return dict(cls._by_model_type)


def resolve_model_type(model: nn.Module) -> str:
    config = getattr(model, "config", None)
    if config is not None:
        model_type = getattr(config, "model_type", None)
        if model_type:
            return model_type
    return model.__class__.__name__


def _require_cls_name(name: str, value: str) -> str:
    if not value or not value.strip():
        raise ValueError(f"{name} is required and must be a non-empty class name string.")
    return value.strip()


def parallelize_model(
    model: nn.Module,
    *,
    transformer_layer_cls: str,
    experts_cls: str | None = None,
    config: Optional[ParallelConfig] = None,
    state: Optional[ParallelState] = None,
    parallelizer: Optional[ModelParallelizer] = None,
    enable_ac: bool = False,
    use_reentrant_ac: bool = False,
    reshard_after_forward: bool = True,
    mixed_precision: str = "bf16",
    num_forward_prefetch: int = 0,
    num_backward_prefetch: int = 0,
    reduce_scatter_ring_buffer: int = 0,
) -> nn.Module:
    state = state or get_parallel_state()
    config = config or state.config
    resolved_experts_cls = _require_cls_name(
        "experts_cls", experts_cls or ""
    ) if config.ep_enabled else (experts_cls.strip() if experts_cls else None)
    ctx = ParallelizeContext(
        config=config,
        state=state,
        transformer_layer_cls=_require_cls_name("transformer_layer_cls", transformer_layer_cls),
        experts_cls=resolved_experts_cls,
        enable_ac=enable_ac,
        use_reentrant_ac=use_reentrant_ac,
        reshard_after_forward=reshard_after_forward,
        mixed_precision=mixed_precision,
        num_forward_prefetch=num_forward_prefetch,
        num_backward_prefetch=num_backward_prefetch,
        reduce_scatter_ring_buffer=reduce_scatter_ring_buffer,
    )
    parallelizer = parallelizer or ParallelizerRegistry.get(model)
    return parallelizer.parallelize(model, ctx)


@ParallelizerRegistry.register(default=True)
class DefaultParallelizer(ModelParallelizer):
    """Default parallelizer for HF and custom models."""
