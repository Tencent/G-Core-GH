# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
# nrwu@tencent.com
"""EP + FSDP2 parallelization helpers for DeepSeek-V4.

Public API: :func:`apply_hp` — slice experts for EP, wrap with FSDP2, bind
``clip_grad_norm_`` / ``load_checkpoint_hp`` / ``save_checkpoint_hp``.

Checkpoint I/O lives in :mod:`.checkpoint`.
"""
from __future__ import annotations

import types
from typing import Optional

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard

from .checkpoint import _load_checkpoint_hp, _save_checkpoint_hp
from .mtp import DeepseekV4MTPBlock

try:
    from .fp8 import MyGroupedLinearFp8
except ImportError:
    MyGroupedLinearFp8 = None

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOperatorIssue=false, reportGeneralTypeIssues=false

__all__ = [
    "apply_hp",
]

# ---------------------------------------------------------------------------
# fp32-keep configuration (see `amp_fp32` arg of apply_hp)
# ---------------------------------------------------------------------------
'''
python3 examples/nrwu/dseek/test_load_ckpt.py --layer 0~3 | grep float32

- [x] layers.0.attn.attn_sink                                                torch.float32          [64]
- [x] layers.0.hc_attn_base                                                  torch.float32          [24]
- [x] layers.0.hc_attn_fn                                                    torch.float32          [24, 16384]
- [x] layers.0.hc_attn_scale                                                 torch.float32          [3]
- [x] layers.0.hc_ffn_base                                                   torch.float32          [24]
- [x] layers.0.hc_ffn_fn                                                     torch.float32          [24, 16384]
- [x] layers.0.hc_ffn_scale                                                  torch.float32          [3]
- [x] layers.1.attn.attn_sink                                                torch.float32          [64]
- [x] layers.1.hc_attn_base                                                  torch.float32          [24]
- [x] layers.1.hc_attn_fn                                                    torch.float32          [24, 16384]
- [x] layers.1.hc_attn_scale                                                 torch.float32          [3]
- [x] layers.1.hc_ffn_base                                                   torch.float32          [24]
- [x] layers.1.hc_ffn_fn                                                     torch.float32          [24, 16384]
- [x] layers.1.hc_ffn_scale                                                  torch.float32          [3]
- [x] layers.2.attn.attn_sink                                                torch.float32          [64]
- [x] layers.2.attn.compressor.ape                                           torch.float32          [4, 1024]
- [x] layers.2.attn.indexer.compressor.ape                                   torch.float32          [4, 256]
- [x] layers.2.hc_attn_base                                                  torch.float32          [24]
- [x] layers.2.hc_attn_fn                                                    torch.float32          [24, 16384]
- [x] layers.2.hc_attn_scale                                                 torch.float32          [3]
- [x] layers.2.hc_ffn_base                                                   torch.float32          [24]
- [x] layers.2.hc_ffn_fn                                                     torch.float32          [24, 16384]
- [x] layers.2.hc_ffn_scale                                                  torch.float32          [3]
- [x] layers.3.attn.attn_sink                                                torch.float32          [64]
- [x] layers.3.attn.compressor.ape                                           torch.float32          [128, 512]
- [x] layers.3.ffn.gate.bias (register buffer in HF)                         torch.float32          [256]
- [x] layers.3.hc_attn_base                                                  torch.float32          [24]
- [x] layers.3.hc_attn_fn                                                    torch.float32          [24, 16384]
- [x] layers.3.hc_attn_scale                                                 torch.float32          [3]
- [x] layers.3.hc_ffn_base                                                   torch.float32          [24]
- [x] layers.3.hc_ffn_fn                                                     torch.float32          [24, 16384]
- [x] layers.3.hc_ffn_scale                                                  torch.float32          [3]
- [x] hc_head_base                                                           torch.float32          [4]
- [x] hc_head_fn                                                             torch.float32          [4, 16384]
- [x] hc_head_scale                                                          torch.float32          [1]
- [x] mtp.0.attn.attn_sink                                                   torch.float32          [64]
- [x] mtp.0.ffn.gate.bias                                                    torch.float32          [256]
- [x] mtp.0.hc_attn_base                                                     torch.float32          [24]
- [x] mtp.0.hc_attn_fn                                                       torch.float32          [24, 16384]
- [x] mtp.0.hc_attn_scale                                                    torch.float32          [3]
- [x] mtp.0.hc_ffn_base                                                      torch.float32          [24]
- [x] mtp.0.hc_ffn_fn                                                        torch.float32          [24, 16384]
- [x] mtp.0.hc_ffn_scale                                                     torch.float32          [3]
- [x] mtp.0.hc_head_base                                                     torch.float32          [4]
- [x] mtp.0.hc_head_fn                                                       torch.float32          [4, 16384]
- [x] mtp.0.hc_head_scale                                                    torch.float32          [1]
'''

# Disk-form FP32 leaves wrapped into single-Parameter holders in modeling
# (see `_Fp32ParamHolder`). Each path is resolved relative to either a
# decoder layer or the root model; `_resolve_dotted` returns None when an
# intermediate attribute is missing (sliding_attention layers have no
# compressor; HCA layers have no indexer), in which case we skip that path.

_FP32_KEEP_PER_LAYER_PATHS = (
    "attn_hc",
    "ffn_hc",
    "hc_head",
    "self_attn._sink_holder",
    "self_attn.compressor._position_bias_holder",
    "self_attn.compressor.indexer._position_bias_holder",
)

_FP32_KEEP_ROOT_PATHS = ("model.hc_head", )


def _resolve_dotted(root: nn.Module, dotted: str) -> Optional[nn.Module]:
    """Resolve ``root.a.b.c``; return None if any intermediate attr is None / missing."""
    m: Optional[nn.Module] = root
    for p in dotted.split("."):
        m = getattr(m, p, None)
        if m is None:
            return None
    return m


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_hp(
    model: nn.Module,
    ep_2d_mesh: "torch.distributed.device_mesh.DeviceMesh",
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    cp_mesh: Optional["torch.distributed.device_mesh.DeviceMesh"] = None,
    attn_backend: str = "eager",
    indexer_backend: str = "eager",
    ep_backend: str = "eager",
    deepep_num_sms: int = 24,
    *,
    amp_fp32: bool = True,
    fp8_qat: bool = False,
    fp4_qat: bool = False,
    fp8: bool = False,
    moe_router_force_load_balancing: bool = False,
) -> nn.Module:
    """Shard experts for EP, apply FSDP2, and bind ``clip_grad_norm_`` / ``load_checkpoint_hp`` / ``save_checkpoint_hp``.

    Parameters
    ----------
    model : nn.Module
        A :class:`DeepseekV4ForCausalLM` instance (CPU or meta).
    ep_2d_mesh : DeviceMesh
        2-D mesh with ``mesh_dim_names=("ep_fsdp", "ep")``.
    mp_policy : MixedPrecisionPolicy, optional
        Defaults to ``param_dtype=bf16, reduce_dtype=fp32,
        cast_forward_inputs=False`` — fp32 master, bf16 forward, fp32
        gradient reduction. The standard recipe with bit-identical
        results vs the transformers baseline (see
        ``test_deepseek_v4_ep.py`` and the EP status memory).
    cp_mesh : DeviceMesh, optional
        1-D context-parallel mesh with ``mesh_dim_names=("cp",)``. CP
        is sequence parallelism and does **not** participate in weight
        sharding, so it is orthogonal to ``ep_2d_mesh`` — pass any 1-D
        mesh whose ranks form the desired CP groups. The model's
        ``DeepseekV4Model`` and every ``DeepseekV4Attention`` get
        ``cp_group / cp_size / cp_rank`` set from this mesh. FSDP grad
        reduction stays on the original meshes; the only place CP
        enters FSDP is via ``set_gradient_divide_factor``, set to
        ``world_size / cp_size`` so the FSDP mean is over independent
        samples rather than total ranks. ``cp_mesh.size() == 1`` or
        ``None`` disables CP (identical to upstream).
    attn_backend : str, default "eager"
        Sets ``model.config.attn_backend``, read by every
        ``DeepseekV4Attention`` at forward time. ``"eager"`` routes to
        upstream :func:`eager_attention_forward` (bit-exact vs
        transformers baseline). ``"fused"`` is reserved for the future
        fused top-k SWA tilelang kernel and currently raises
        ``NotImplementedError``. Any other value raises ``ValueError``
        at forward time.
    ep_backend : str, default "eager"
        Expert dispatch backend. ``"eager"`` keeps the existing
        ``all_to_all_uneven`` path; ``"deepep"`` uses DeepEP dispatch and
        combine kernels.
    amp_fp32 : bool, keyword-only, default True
        When True (default), wrap each disk-FP32 leaf (mHC ``attn_hc /
        ffn_hc / hc_head``, ``attn._sink_holder``,
        ``compressor._position_bias_holder``,
        ``compressor.indexer._position_bias_holder``) in its own nested
        ``fully_shard`` with ``MixedPrecisionPolicy(param_dtype=None,
        reduce_dtype=fp32)``. The outer ``mp_policy`` then never casts
        these params on forward — they participate in compute as fp32,
        matching the dev-team ``inference/model.py`` semantics. Pass
        ``False`` to fall back to the legacy behaviour where everything
        gets bf16-cast in forward (the modeling ``_Fp32ParamHolder``
        wrappers exist either way; without this flag they are simply
        absorbed into the outer layer/model ``fully_shard``).
    fp8_qat : bool, keyword-only, default False
        Enable FP8 activation fake-quantization (QAT). Inserts
        block-wise E4M3 quantize→dequantize round-trips (STE backward)
        on KV nope dims, compressor compressed KV nope dims, and
        indexer query/key nope dims. Simulates inference-time FP8
        quantization noise so the model learns to be robust.
    fp4_qat : bool, keyword-only, default False
        Enable E2M1 1×32 fake-quantization of routed MoE expert weights.
        When both QAT flags are enabled, this takes priority over
        ``fp8_qat`` for these weights.
    fp8 : bool, keyword-only, default False
        When True, MoE expert ``gate_up`` / ``down`` grouped GEMMs use
        TE Float8BlockScaling via :class:`MyGroupedLinearFp8` (stacked
        ``[E,N,K]`` weights; auto-pads per-expert M to 16). Orthogonal
        to ``fp8_qat`` (fake-quant STE). Requires Transformer Engine.
    moe_router_force_load_balancing : bool, keyword-only, default False
        Sets ``model.config.moe_router_force_load_balancing`` (and MTP
        deepcopy configs). When True, TopKRouter picks random unique top-k
        indices for MoE load-balanced benchmarks. Mutually exclusive with
        router replay.

    Returns
    -------
    model : nn.Module
        Same model, FSDP-wrapped with EP. Adds attrs ``_ep_size``,
        ``_ep_rank``, ``_ep_group``, ``_ep_fsdp_mesh``, ``_ep_2d_mesh``;
        binds methods ``clip_grad_norm_``, ``load_checkpoint_hp``,
        ``save_checkpoint_hp``. With ``cp_mesh``, also sets
        ``model.model.cp_group / cp_size / cp_rank`` and the same on
        every ``DeepseekV4Attention``.

    Notes
    -----
    ``model.save_checkpoint_hp(save_path, orig_ckpt_dir=...)`` writes a
    HF-compatible bf16/fp32 checkpoint via a two-phase parallel pipeline:
    every rank gathers + D2H-stages its owned share of params in Phase 1,
    then independently splits / casts / writes to a rank-private shard
    file in Phase 2; rank 0 globally renames and writes index + config +
    tokenizer in Phase 3. See :func:`_save_checkpoint_hp`.
    """
    if ep_backend not in ("eager", "deepep"):
        raise ValueError(f"unknown ep_backend: {ep_backend}")
    if ep_backend == "deepep":
        from .deepep_a2a import set_deepep_num_sms
        set_deepep_num_sms(deepep_num_sms)

    model.config.attn_backend = attn_backend
    model.config.indexer_backend = indexer_backend
    model.config.ep_backend = ep_backend
    model.config.deepep_num_sms = deepep_num_sms
    model.config.amp_fp32 = amp_fp32
    model.config.fp8_qat = fp8_qat
    model.config.fp4_qat = fp4_qat
    model.config.fp8 = fp8
    model.config.moe_router_force_load_balancing = moe_router_force_load_balancing
    # Sync backend knobs to every DeepseekV4Attention / DeepseekV4Experts
    # (including MTP blocks), because MTP blocks are constructed before
    # apply_hp runs, and their self.self_attn.config is a deepcopy that
    # may not have these attrs.
    for layer in _get_layers(model):
        if isinstance(layer, DeepseekV4MTPBlock):
            layer.config.attn_backend = attn_backend
            layer.config.indexer_backend = indexer_backend
            layer.config.ep_backend = ep_backend
            layer.config.deepep_num_sms = deepep_num_sms
            layer.config.amp_fp32 = amp_fp32
            layer.config.fp8_qat = fp8_qat
            layer.config.fp4_qat = fp4_qat
            layer.config.fp8 = fp8
            layer.config.moe_router_force_load_balancing = moe_router_force_load_balancing
            assert layer.self_attn.config.attn_backend == attn_backend

    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        )

    fp32_keep_mp_policy = MixedPrecisionPolicy(
        param_dtype=None,  # explicit: no forward cast — leaf stays fp32
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )

    _assert_fp32_master(model)

    ep_mesh = ep_2d_mesh["ep"]
    ep_fsdp_mesh = ep_2d_mesh["ep_fsdp"]
    world_size = ep_2d_mesh.size()

    layers = _get_layers(model)

    with torch.no_grad():
        for layer in layers:
            _shard_experts_dtensor(layer.mlp.experts, ep_2d_mesh)

    if fp8:
        for layer in layers:
            experts = layer.mlp.experts
            experts.fp8_grouped_linear = MyGroupedLinearFp8(num_gemms=experts.num_local_experts, )

    # Effective number of independent samples = world_size / cp_size.
    # Inside a cp-pair, all ranks see the same input (sequence partitioned
    # across the cp axis), so for the purpose of FSDP gradient reduction
    # only ``world_size / cp_size`` distinct samples contribute. Setting
    # the divide factor to that value makes the FSDP mean match the
    # baseline's pure-FSDP mean over ``world_size_baseline = world_size /
    # cp_size`` ranks, removing the need for a ``loss * cp_size`` scaling
    # hack on the test side.
    cp_size = cp_mesh.size() if cp_mesh is not None else 1
    assert world_size % cp_size == 0, (
        f"world_size {world_size} not divisible by cp_size {cp_size}"
    )
    grad_divide_factor = world_size // cp_size

    for layer in layers:

        if amp_fp32:
            for path in _FP32_KEEP_PER_LAYER_PATHS:
                sub = _resolve_dotted(layer, path)
                if sub is None:
                    continue
                fully_shard(sub, mp_policy=fp32_keep_mp_policy)
                sub.set_gradient_divide_factor(grad_divide_factor)

        fully_shard(layer.mlp.experts, mesh=ep_fsdp_mesh, mp_policy=mp_policy)
        # Experts: ep_fsdp_mesh (size = world_size/ep_size) is used for the
        # FSDP reduce-scatter, but each expert's parameters live on a single
        # ``ep`` index; backward contributions from all 32 ranks ultimately
        # land on those ``ep_fsdp`` ranks, so the effective denominator is
        # ``grad_divide_factor`` (i.e. world_size/cp_size) — same as the
        # non-expert path.
        layer.mlp.experts.set_gradient_divide_factor(grad_divide_factor)
        fully_shard(layer, mp_policy=mp_policy)
        # ``fully_shard(layer)`` defaults to the global mesh
        # (cp_full_mesh, size=world_size). Override its divide factor to
        # match the independent-sample count.
        layer.set_gradient_divide_factor(grad_divide_factor)

    if amp_fp32:
        for path in _FP32_KEEP_ROOT_PATHS:
            sub = _resolve_dotted(model, path)
            if sub is None:
                continue
            fully_shard(sub, mp_policy=fp32_keep_mp_policy)
            sub.set_gradient_divide_factor(grad_divide_factor)

    fully_shard(model, mp_policy=mp_policy)
    model.set_gradient_divide_factor(grad_divide_factor)

    model._ep_size = ep_mesh.size()
    model._ep_rank = ep_mesh.get_local_rank()
    model._ep_group = ep_mesh.get_group()
    model._ep_fsdp_mesh = ep_fsdp_mesh
    # Kept around as a 2D ``(ep, ep_fsdp)`` view for diagnostics / future use.
    # NOTE: ``save_checkpoint_hp`` no longer gathers expert weights via this
    # mesh in a single ``DTensor.from_local`` call — the 2D ``[Shard(0),
    # Shard(0)]`` placement transposes the gathered layout in ways that don't
    # round-trip cleanly. The saver does a TWO-STEP gather instead:
    # ``ep_fsdp.full_tensor()`` first, then ``dist.all_gather`` over
    # ``_ep_group``. See ``_save_checkpoint_hp`` for details.
    model._ep_2d_mesh = ep_2d_mesh

    model.clip_grad_norm_ = types.MethodType(_clip_grad_norm_multi_mesh, model)
    model.load_checkpoint_hp = types.MethodType(_load_checkpoint_hp, model)
    model.save_checkpoint_hp = types.MethodType(_save_checkpoint_hp, model)

    if cp_mesh is not None and cp_mesh.size() > 1:
        _bind_cp(model, cp_mesh)

    return model


def _bind_cp(
    model: nn.Module,
    cp_mesh: "torch.distributed.device_mesh.DeviceMesh",
) -> None:
    """Recursively bind ``cp_group / cp_size / cp_rank`` on the model's
    inner :class:`DeepseekV4Model` and every :class:`DeepseekV4Attention`.

    CP is orthogonal to the EP/FSDP mesh — it only affects sequence-axis
    layout inside attention and the model's input slicing, not weight
    sharding. ``_clip_grad_norm_multi_mesh`` and ``_load_checkpoint_hp``
    don't need to know about CP.

    Stashes ``model._cp_group / _cp_size / _cp_rank / _cp_mesh`` on the
    outer ``DeepseekV4ForCausalLM`` for callers that want to access them
    (e.g. test code that needs the cp_group for cross-CP loss/logits
    aggregation).
    """
    cp_group = cp_mesh.get_group()
    cp_size = cp_mesh.size()
    cp_rank = cp_mesh.get_local_rank()

    # Outer wrapper attrs (for the test code & general introspection).
    model._cp_group = cp_group
    model._cp_size = cp_size
    model._cp_rank = cp_rank
    model._cp_mesh = cp_mesh

    # Inner DeepseekV4Model needs these for ``forward`` to know to slice
    # input_ids / position_ids and pass causal_mask=None down to attention.
    inner = model.model
    inner.cp_group = cp_group
    inner.cp_size = cp_size
    inner.cp_rank = cp_rank

    # Every attention block: CP comm primitives read these.
    layers = _get_layers(model)
    for layer in layers:
        layer.self_attn.cp_group = cp_group
        layer.self_attn.cp_size = cp_size
        layer.self_attn.cp_rank = cp_rank


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_layers(model: nn.Module):
    """Return all trainable decoder-like layers (backbone + optional MTP)."""
    layers = list(model.model.layers)
    mtp = getattr(model, "mtp", None)
    if mtp is not None and hasattr(mtp, "layers"):
        layers.extend(list(mtp.layers))
    return layers


def _assert_fp32_master(model: nn.Module) -> None:
    """Fail-fast: every floating-point param must be fp32 (recommended master).

    The default ``mp_policy(param_dtype=bf16, reduce_dtype=fp32)`` only gives
    bit-identical results vs the transformers baseline when master weights
    are fp32 (forward casts down to bf16, gradient reduction in fp32). A bf16
    master with this mp_policy makes the forward cast a no-op and exposes
    subtle non-determinism (see EP status memory § "bf16 master → fp32 master").
    Build the model under ``torch.set_default_dtype(torch.float32)`` to ensure
    fp32 master.
    """
    for name, p in model.named_parameters():
        if not p.dtype.is_floating_point:
            continue
        assert p.dtype == torch.float32, (
            f"apply_hp requires fp32 master weights, but {name}.dtype={p.dtype}. "
            f"Build the model under `torch.set_default_dtype(torch.float32)` and "
            f"use the default mp_policy (see docstring for canonical usage)."
        )


def _shard_experts_dtensor(
    experts: nn.Module,
    ep_2d_mesh: "torch.distributed.device_mesh.DeviceMesh",
) -> None:
    """Slice expert weights along dim-0 for EP using DTensor redistribute.

    Supports both real and meta tensors. For meta tensors, allocates a meta
    placeholder with the local EP shape (real data is filled later by
    :func:`_load_checkpoint_hp`).
    """
    ep_mesh = ep_2d_mesh["ep"]
    ep_size = ep_mesh.size()
    ep_rank = ep_mesh.get_local_rank()
    ep_group = ep_mesh.get_group()
    num_local = experts.num_experts // ep_size
    assert experts.num_experts % ep_size == 0, (
        f"num_experts={experts.num_experts} not divisible by ep_size={ep_size}"
    )

    for name in ("gate_up_proj", "down_proj"):
        p = getattr(experts, name)
        new_shape = (num_local, ) + tuple(p.shape[1:])

        if p.is_meta:
            local = torch.empty(new_shape, dtype=p.dtype, device="meta")
        else:
            dt_full = DTensor.from_local(p.data, device_mesh=ep_mesh, placements=[Replicate()])
            dt_local = dt_full.redistribute(device_mesh=ep_mesh, placements=[Shard(0)])
            # ``clone()`` cuts the view chain so we release the full tensor's storage.
            local = dt_local.to_local().contiguous().clone()

        setattr(experts, name, nn.Parameter(local, requires_grad=p.requires_grad))

    experts.ep_size = ep_size
    experts.ep_rank = ep_rank
    experts.ep_group = ep_group
    experts.num_local_experts = num_local


def _clip_grad_norm_multi_mesh(
    self: nn.Module,
    max_norm: float = 2.0,
    norm_type: float = 2.0,
) -> float:
    """``clip_grad_norm_`` that handles params on different DeviceMeshes.

    Expert params live on the ``ep_fsdp`` sub-mesh; non-expert params live
    on the global FSDP mesh. The standard ``clip_grad_norm_`` fails because
    it cannot stack DTensors from different meshes. We split into two
    sub-norms, all-reduce only the expert one across ``ep_group``, then combine.

    Bound to the model via :func:`apply_hp`; reads ``self._ep_group``
    and ``self._ep_fsdp_mesh``.
    """
    ep_group = self._ep_group
    ep_fsdp_mesh = self._ep_fsdp_mesh

    groups: dict[str, list] = {}
    for p in self.parameters():
        if p.grad is None:
            continue
        key = str(p.device_mesh) if isinstance(p, DTensor) else "default"
        groups.setdefault(key, []).append(p)

    ep_fsdp_key = str(ep_fsdp_mesh) if ep_fsdp_mesh is not None else None

    non_expert_norm_sq = torch.tensor(0.0, device="cuda")
    expert_norm_sq = torch.tensor(0.0, device="cuda")

    for key, params in groups.items():
        gn = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))
        if isinstance(gn, DTensor):
            gn = gn.full_tensor()
        if key == ep_fsdp_key:
            expert_norm_sq += gn**2
        else:
            non_expert_norm_sq += gn**2

    if ep_group is not None:
        dist.all_reduce(expert_norm_sq, group=ep_group)

    total_norm = (non_expert_norm_sq + expert_norm_sq).sqrt()
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in self.parameters():
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)

    return total_norm.item()
