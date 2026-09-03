# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""EP + FSDP2 parallelization for Qwen3.8-Flash-Next.

Public API: :func:`apply_hp` — bind CP, swap the QSA / EP / Engram leaves, shard experts,
wrap with FSDP2, and bind ``clip_grad_norm_`` / checkpoint methods.
"""
from __future__ import annotations

import types
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard

from gpatch_v4.models.hp_module import HpModule

from .checkpoint import load_checkpoint_hp, save_checkpoint_hp
from .engram import (
    OwnerShardedNGramEmbedding,
    Qwen4ExpEngramEmbedding,
    materialize_engram_tables,
)
from .modeling_qwen4_exp import Qwen4ExpForCausalLM
from .moe import Qwen4ExpEPExperts
from .qsa import QSA_ATTN_BACKENDS, Qwen4ExpQSAAttention

__all__ = [
    "Qwen4ExpHpForCausalLM",
    "apply_hp",
    "set_activation_checkpointing",
    "swap_parallel_modules",
]


class Qwen4ExpHpForCausalLM(Qwen4ExpForCausalLM, HpModule):
    """Select GCore's hybrid-parallel construction path for Qwen4-Exp."""


def apply_hp(
    model: nn.Module,
    ep_2d_mesh: "torch.distributed.device_mesh.DeviceMesh",
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    cp_mesh: Optional["torch.distributed.device_mesh.DeviceMesh"] = None,
    attn_backend: str = "flex",
    ep_backend: str = "eager",
) -> nn.Module:
    """Shard experts for EP, apply FSDP2, and bind the HpModule methods.

    Parameters
    ----------
    model : nn.Module
        A meta-device :class:`Qwen4ExpForCausalLM`, built under
        ``torch.set_default_dtype(torch.float32)`` (fp32 master).
    ep_2d_mesh : DeviceMesh
        2-D mesh with ``mesh_dim_names=("ep_fsdp", "ep")``, spanning the whole world.
    mp_policy : MixedPrecisionPolicy, optional
        Defaults to ``param_dtype=bf16, reduce_dtype=fp32`` — fp32 master, bf16 forward,
        fp32 gradient reduction.
    cp_mesh : DeviceMesh, optional
        Contiguous context-parallel mesh used by QSA, GatedDeltaNet, and PLE.
    attn_backend : str, default "flex"
        QSA backend, see :data:`.qsa.QSA_ATTN_BACKENDS`.

    Returns
    -------
    nn.Module
        The same model, wrapped.
    """
    if attn_backend not in QSA_ATTN_BACKENDS:
        raise ValueError(f"attn_backend must be one of {QSA_ATTN_BACKENDS}, got {attn_backend!r}.")
    if ep_backend != "eager":
        raise NotImplementedError(
            f"ep_backend={ep_backend!r} is not implemented for qwen4_exp; DeepEP dispatch "
            "is a follow-up. Use ep_backend='eager'."
        )

    world_size = ep_2d_mesh.size()
    if dist.is_initialized() and world_size != dist.get_world_size():
        raise ValueError(
            f"ep_2d_mesh spans {world_size} ranks but the world has "
            f"{dist.get_world_size()}; the Engram table is owner-sharded over the whole "
            "world, so the mesh must cover it."
        )
    ep_mesh = ep_2d_mesh["ep"]
    ep_fsdp_mesh = ep_2d_mesh["ep_fsdp"]

    cp_size = cp_mesh.size() if cp_mesh is not None else 1
    if world_size % cp_size != 0:
        raise ValueError(
            f"world size {world_size} must be divisible by context parallel size {cp_size}"
        )
    model.config.attn_backend = attn_backend
    model.config.ep_backend = ep_backend

    swap_parallel_modules(
        model,
        attn_backend=attn_backend,
        ep_group=ep_mesh.get_group(),
        engram_group=dist.group.WORLD if dist.is_initialized() else None,
    )
    if cp_size > 1:
        _bind_cp(model, cp_mesh)
    layers = list(model.model.layers)
    _assert_fp32_master(model)

    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        )

    with torch.no_grad():
        for layer in layers:
            shard_experts_for_ep(layer.mlp.experts, ep_mesh)

    # The Engram table is excluded from FSDP2 below, so FSDP never applies the
    # mixed-precision cast to it. Every downstream consumer *is* cast to
    # ``mp_policy.param_dtype``, so the table has to cast its own output or the first
    # projection hits a dtype mismatch. Casting the looked-up values (not the table)
    # keeps the fp32 master for the optimizer.
    for module in model.modules():
        if isinstance(module, Qwen4ExpEngramEmbedding):
            module.ngram_embedding.output_dtype = mp_policy.param_dtype

    # ignored_params stay on meta through fully_shard; allocate the local shard
    # here so load_checkpoint_hp can copy_ and forward does not .to(meta).
    if ep_2d_mesh.device_type == "cpu":
        engram_device = torch.device("cpu")
    else:
        engram_device = torch.device(ep_2d_mesh.device_type, torch.cuda.current_device())
    materialize_engram_tables(model, engram_device, torch.float32)

    engram_params = _engram_parameters(model)

    # Effective number of independent samples. CP ranks see slices of the same sample, so
    # the FSDP mean must be over world/cp rather than world.
    grad_divide_factor = world_size // cp_size

    for layer in layers:
        # Experts reduce-scatter over ``ep_fsdp`` (each expert lives on one ``ep`` index),
        # everything else over the global mesh.
        fully_shard(layer.mlp.experts, mesh=ep_fsdp_mesh, mp_policy=mp_policy)
        layer.mlp.experts.set_gradient_divide_factor(grad_divide_factor)
        # The Engram shard is already partitioned by row ownership — FSDP must not shard
        # it a second time.
        fully_shard(layer, mp_policy=mp_policy, ignored_params=_engram_parameters(layer) or None)
        layer.set_gradient_divide_factor(grad_divide_factor)
        layer.set_modules_to_forward_prefetch([layer.mlp.experts])

    fully_shard(model, mp_policy=mp_policy, ignored_params=engram_params)
    model.set_gradient_divide_factor(grad_divide_factor)

    # After `fully_shard`, so FSDP receives plain tensors in `ignored_params` (its
    # well-tested path) and only the optimizer built later sees the DTensor.
    _wrap_engram_as_dtensor(model, world_size, grad_divide_factor)

    model._ep_size = ep_mesh.size()
    model._ep_rank = ep_mesh.get_local_rank()
    model._ep_group = ep_mesh.get_group()
    model._ep_fsdp_mesh = ep_fsdp_mesh
    model._ep_2d_mesh = ep_2d_mesh
    model._engram_param_ids = {id(p) for p in _engram_parameters(model)}
    model.clip_grad_norm_ = types.MethodType(_clip_grad_norm_multi_mesh, model)
    model.load_checkpoint_hp = types.MethodType(load_checkpoint_hp, model)
    model.save_checkpoint_hp = types.MethodType(save_checkpoint_hp, model)
    return model


def swap_parallel_modules(
    model: nn.Module,
    attn_backend: str = "flex",
    ep_group: Optional[dist.ProcessGroup] = None,
    engram_group: Optional[dist.ProcessGroup] = None,
) -> nn.Module:
    """Replace the three parallelism-sensitive leaves with gcore's versions.

    ``self_attn`` -> :class:`.qsa.Qwen4ExpQSAAttention` (batched selector + FlexAttention),
    ``mlp.experts`` -> :class:`.moe.Qwen4ExpEPExperts` (all-to-all EP),
    ``ple.ple_embedding`` -> :class:`.engram.Qwen4ExpEngramEmbedding` (owner-sharded table).

    Parameter names are unchanged in every case, so checkpoint FQNs still apply. Real
    (non-meta) weights are carried over, which is what makes this testable on CPU;
    the Engram table is row-sliced to this rank's ownership range as it is copied.

    Replacements are constructed on the **same device as the model**, which for the real
    flow is ``meta``. Building them eagerly would allocate a full 51.2 B n-gram table plus
    ~6.7 GB of fp32 experts per layer before immediately discarding most of it.

    Layer roles come from ``config.layer_types`` and ``config.ple_layer_ids`` rather than
    ``hasattr`` probing, so a layout change fails loudly instead of silently skipping.
    """
    config = model.config.get_text_config()
    layers = list(model.model.layers)
    if len(config.layer_types) != len(layers):
        raise ValueError(
            f"config.layer_types has {len(config.layer_types)} entries but the model has "
            f"{len(layers)} layers."
        )
    # `ple_layer_ids` is one-indexed upstream.
    ple_indices = {layer_id - 1 for layer_id in config.ple_layer_ids}
    device = _module_device(model)

    for layer_idx, (layer, layer_type) in enumerate(zip(layers, config.layer_types)):
        if layer_type not in ("qwen_sparse_attention", "linear_attention"):
            raise ValueError(f"unexpected layer_type {layer_type!r} at layer {layer_idx}")

        if layer_type == "qwen_sparse_attention":
            with torch.device(device):
                replacement = Qwen4ExpQSAAttention(config, layer_idx, attn_backend=attn_backend)
            layer.self_attn = _transplant(layer.self_attn, replacement)

        with torch.device(device):
            experts = Qwen4ExpEPExperts(config)
        layer.mlp.experts = _transplant(layer.mlp.experts, experts)
        layer.mlp.experts.configure_ep(ep_group)

        if layer_idx in ple_indices:
            old = layer.ple.ple_embedding
            layer.ple.ple_embedding = _replace_engram(config, old, engram_group, device)

    return model


def _bind_cp(
    model: nn.Module,
    cp_mesh: "torch.distributed.device_mesh.DeviceMesh",
) -> None:
    cp_group = cp_mesh.get_group()
    cp_size = cp_mesh.size()
    cp_rank = cp_mesh.get_local_rank()

    model._cp_group = cp_group
    model._cp_size = cp_size
    model._cp_rank = cp_rank
    model._cp_mesh = cp_mesh
    for layer in model.model.layers:
        layer.cp_size = cp_size


def _module_device(module: nn.Module) -> torch.device:
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    raise ValueError("cannot determine module device: no parameters or buffers")


def set_activation_checkpointing(model: nn.Module, enabled: bool = True) -> None:
    """Toggle per-layer activation checkpointing, keeping the Engram layer eager.

    The Engram-owning layer must run eagerly: recomputation would replay its
    owner-sharded all-to-all collectives, which is not supported (NVIDIA hit the same
    wall and exempts the same single block, checkpointing 47 of 48).

    Call **after** ``model.gradient_checkpointing_enable(...)``, which sets the flag on
    every :class:`GradientCheckpointingLayer`.
    """
    config = model.config.get_text_config()
    ple_indices = {layer_id - 1 for layer_id in config.ple_layer_ids}
    for layer_idx, layer in enumerate(model.model.layers):
        layer.gradient_checkpointing = enabled and layer_idx not in ple_indices


def _transplant(old: nn.Module, new: nn.Module) -> nn.Module:
    """Move ``old``'s tensors into ``new``, which must have an identical state dict.

    Meta tensors are left alone — the real values arrive later from
    ``load_checkpoint_hp``.
    """
    old_state = old.state_dict()
    new_state = new.state_dict()
    if set(old_state) != set(new_state):
        raise ValueError(
            "module replacement changed the state dict keys: "
            f"missing={sorted(set(old_state) - set(new_state))}, "
            f"unexpected={sorted(set(new_state) - set(old_state))}"
        )
    if not any(tensor.is_meta for tensor in old_state.values()):
        new.load_state_dict(old_state)
    return new


def _replace_engram(
    config,
    old: nn.Module,
    engram_group: Optional[dist.ProcessGroup],
    device: torch.device,
) -> Qwen4ExpEngramEmbedding:
    """Swap the dense n-gram table for the owner-sharded one, slicing this rank's rows."""
    with torch.device(device):
        new = Qwen4ExpEngramEmbedding(
            config,
            config.ple_embed_dim,
            old.layer_idx,
            old.ple_layer_index,
            process_group=engram_group,
        )
    old_weight = old.ngram_embedding.weight
    if not old_weight.is_meta:
        table = new.ngram_embedding
        start = table.global_row_start
        with torch.no_grad():
            table.weight.copy_(old_weight[start:start + table.local_rows])
    return new


def _wrap_engram_as_dtensor(
    model: nn.Module,
    world_size: int,
    gradient_divide_factor: int,
) -> None:
    """Re-wrap each Engram shard as a global ``DTensor(Shard(0))`` over all ranks.

    The table is excluded from FSDP2 (it is already partitioned by row ownership), which
    left it a plain ``nn.Parameter`` while every other parameter became a DTensor. AdamW
    defaults to ``foreach=True`` and batches same-device/dtype parameters into one
    ``torch._foreach_mul_`` call, which rejects a mix of DTensor and plain tensors:

        RuntimeError: aten._foreach_mul_.Scalar: got mixed torch.Tensor and DTensor

    Making it a DTensor also means ``clip_grad_norm_`` reduces its gradient norm through
    normal DTensor semantics instead of a hand-rolled all-reduce, and gives DCP a global
    FQN it can reshard later. Lookups go through
    :attr:`OwnerShardedNGramEmbedding.local_weight`.

    The lookup backward already routes and sums every requester's gradient on the row
    owner. Unlike FSDP parameters, this ignored parameter has no reduce-scatter that
    cancels the loss function's ``gradient_divide_factor`` multiplier, so its hook must
    apply that division explicitly.

    Skipped only when distributed is not initialized (single-process CPU tests). Note it
    still applies at ``world_size == 1``: ``fully_shard`` produces DTensors even on a
    one-rank mesh, so the type mismatch would occur there too.
    """
    if not dist.is_initialized():
        return

    tables = [
        module for module in model.modules()
        if isinstance(module, OwnerShardedNGramEmbedding)
    ]
    if not tables:
        return

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    engram_mesh = init_device_mesh(device_type, (world_size, ), mesh_dim_names=("engram", ))
    for table in tables:
        if isinstance(table.weight, DTensor):
            continue
        local = table.weight
        table.weight = nn.Parameter(
            DTensor.from_local(local.data, device_mesh=engram_mesh, placements=[Shard(0)]),
            requires_grad=local.requires_grad,
        )
        if table.weight.requires_grad and gradient_divide_factor != 1:
            def divide_owner_gradient(grad: torch.Tensor) -> torch.Tensor:
                return grad / gradient_divide_factor

            table.weight.register_hook(divide_owner_gradient)


def _engram_parameters(module: nn.Module) -> set:
    """The Engram table parameters under ``module``, which FSDP2 must leave alone."""
    return {
        submodule.ngram_embedding.weight
        for submodule in module.modules() if isinstance(submodule, Qwen4ExpEngramEmbedding)
    }


def _assert_fp32_master(model: nn.Module) -> None:
    """Fail fast unless every floating-point parameter is fp32.

    The default ``mp_policy(param_dtype=bf16, reduce_dtype=fp32)`` only behaves as
    intended with an fp32 master: the forward casts down to bf16 and gradients reduce in
    fp32. A bf16 master makes the forward cast a no-op. Build under
    ``torch.set_default_dtype(torch.float32)``.
    """
    for name, param in model.named_parameters():
        if param.dtype.is_floating_point and param.dtype != torch.float32:
            raise ValueError(
                f"apply_hp requires fp32 master weights, but {name}.dtype={param.dtype}. "
                "Build the model under `torch.set_default_dtype(torch.float32)`."
            )


def shard_experts_for_ep(
    experts: nn.Module,
    ep_mesh: "torch.distributed.device_mesh.DeviceMesh",
) -> None:
    """Slice stacked expert weights along dim 0 so each rank keeps its own experts.

    Handles meta tensors by allocating a meta placeholder of the local shape;
    ``load_checkpoint_hp`` fills in the real values.
    """
    ep_size = ep_mesh.size()
    if experts.num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts={experts.num_experts} not divisible by ep_size={ep_size}"
        )
    num_local = experts.num_experts // ep_size

    for name in ("gate_up_proj", "down_proj"):
        param = getattr(experts, name)
        if param.shape[0] == num_local:
            continue
        local_shape = (num_local, ) + tuple(param.shape[1:])
        if param.is_meta:
            local = torch.empty(local_shape, dtype=param.dtype, device="meta")
        else:
            replicated = DTensor.from_local(
                param.data, device_mesh=ep_mesh, placements=[Replicate()]
            )
            sharded = replicated.redistribute(device_mesh=ep_mesh, placements=[Shard(0)])
            # ``clone()`` cuts the view chain so the full tensor's storage is released.
            local = sharded.to_local().contiguous().clone()
        setattr(experts, name, nn.Parameter(local, requires_grad=param.requires_grad))

    experts.configure_ep(ep_mesh.get_group())


def _clip_grad_norm_multi_mesh(
    self: nn.Module,
    max_norm: float = 2.0,
) -> float:
    """``clip_grad_norm_`` across parameters living on different meshes.

    Parameters live on different meshes, and ``clip_grad_norm_`` cannot stack DTensors
    across meshes, so we take a sub-norm per mesh and combine:

    * expert params — sharded on the ``ep_fsdp`` sub-mesh. Each expert lives on a single
      ``ep`` index, so their squared norm additionally needs an all-reduce over
      ``ep_group`` to cover every expert.
    * everything else (including the Engram table, a ``Shard(0)`` DTensor over the world
      mesh) — DTensor semantics already reduce within the mesh.

    Adapted from DSV4's version. The Engram needed a hand-rolled all-reduce back when it
    was a plain local tensor; now that ``apply_hp`` wraps it as a DTensor it just falls
    into its own mesh group, and adding a manual reduce on top would **double-count** it.
    """
    ep_fsdp_key = str(self._ep_fsdp_mesh) if self._ep_fsdp_mesh is not None else None

    groups: dict[str, list] = {}
    for param in self.parameters():
        if param.grad is None:
            continue
        key = str(param.device_mesh) if isinstance(param, DTensor) else "default"
        groups.setdefault(key, []).append(param)

    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    non_expert_sq = torch.zeros((), device=device)
    expert_sq = torch.zeros((), device=device)

    for key, params in groups.items():
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))
        if isinstance(norm, DTensor):
            norm = norm.full_tensor()
        if key == ep_fsdp_key:
            expert_sq = expert_sq + norm**2
        else:
            non_expert_sq = non_expert_sq + norm**2

    if self._ep_group is not None:
        dist.all_reduce(expert_sq, group=self._ep_group)

    total_norm = (non_expert_sq + expert_sq).sqrt()
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for param in self.parameters():
            if param.grad is not None:
                param.grad.detach().mul_(clip_coef)
    return total_norm.item()
