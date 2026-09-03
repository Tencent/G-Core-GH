# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""CPU unit / light training tests for DeepSeek-V4 CSA indexer freeze.

Mirrors FSDP2 reality: CSA indexer only emits discrete top-k indices (no KL),
so task grads are absent. Params left in AdamW/Muon with ``weight_decay > 0``
still decay when grads are zero-filled (FSDP-like), not when ``grad is None``
(plain eager skip). Default training freezes via ``DeepseekV4PostInitModel``
before ``setup_optimizer``'s ``filter(requires_grad)``.

Covers:

1. Without freeze + zero-filled grads → AdamW WD shrinks indexer.
2. ``freeze_csa_indexer_params`` / PostInit ON → excluded from opt, unchanged.
3. Real ``DeepseekV4Indexer`` eager forward (topk-only) + one train step.
4. Freeze-after-optimizer still gets WD (ordering contract).

Usage::

    cd /work/wepsdl/gcore-dev
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=2400 tests/test_gfused/test_deepseek_v4_freeze_csa_indexer.py
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from gpatch_v4.extended_model.deepseek_v4 import DeepseekV4PostInitModel
from gpatch_v4.models.deepseek_v4.freeze_csa_indexer import freeze_csa_indexer_params
from gpatch_v4.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4CSACompressor,
    DeepseekV4Indexer,
)
from gpatch_v4.training_backend.fsdp2_backend.optimizer import setup_optimizer

HIDDEN = 64
Q_LORA = 16
INDEX_HEADS = 4
INDEX_HEAD_DIM = 16
INDEX_TOPK = 2
COMPRESS_RATE = 4
SEQ_LEN = 16  # must be divisible by COMPRESS_RATE
HEAD_DIM = 32


def _indexer_hf_config():
    """Tiny ``DeepseekV4Config`` sufficient to construct a real indexer."""
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config,
    )

    cfg = DeepseekV4Config(
        hidden_size=HIDDEN,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        q_lora_rank=Q_LORA,
        index_n_heads=INDEX_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=INDEX_TOPK,
        max_position_embeddings=64,
        num_hidden_layers=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        compress_rates={
            "compressed_sparse_attention": COMPRESS_RATE,
            "heavily_compressed_attention": 8,
        },
        partial_rotary_factor=0.5,
        rms_norm_eps=1e-6,
    )
    # apply_hp normally stamps these; indexer reads them off config.
    cfg.indexer_backend = "eager"
    cfg.attn_backend = "eager"
    cfg.fp8_qat = False
    cfg.fp4_qat_indexer = False
    return cfg


class _IndexerToyModel(nn.Module):
    """Real ``DeepseekV4Indexer`` + a scalar so opt is non-empty when frozen."""

    def __init__(self):
        super().__init__()
        self.indexer = DeepseekV4Indexer(_indexer_hf_config())
        # ``_Fp32ParamHolder`` uses ``torch.empty``; init so WD-shrink checks are finite.
        nn.init.normal_(self.indexer._position_bias_holder.weight, mean=0.0, std=0.02)
        self.scale = nn.Parameter(torch.ones(()))

    def run_topk(self) -> torch.Tensor:
        """One eager indexer forward; returns discrete top-k indices only."""
        b, s = 1, SEQ_LEN
        hidden = torch.randn(b, s, HIDDEN)
        q_residual = torch.randn(b, s, Q_LORA)
        position_ids = torch.arange(s).unsqueeze(0)
        return self.indexer(
            hidden,
            q_residual,
            position_ids,
            past_key_values=None,
            layer_idx=0,
        )


class _GradModeIndexer(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_enabled = None

    def forward(
        self,
        hidden_states,
        q_residual,
        position_ids,
        past_key_values,
        layer_idx,
        **kwargs,
    ):
        self.grad_enabled = torch.is_grad_enabled()
        return torch.zeros(
            hidden_states.shape[0],
            position_ids.shape[1],
            1,
            dtype=torch.long,
            device=hidden_states.device,
        )


def _trainable_params(model: nn.Module):
    """Mirror ``fsdp2_backend/optimizer.py::setup_optimizer`` filter."""
    return list(filter(lambda p: p.requires_grad, model.parameters()))


def _training_ns(*, freeze_csa_indexer: bool) -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(
            freeze_csa_indexer=freeze_csa_indexer,
            freeze_router_weight=False,
            freeze_router_correction_bias=True,
        )
    )


def _setup_optimizer_cfg(*, weight_decay: float = 0.1, lr: float = 0.1):
    return SimpleNamespace(
        optimizer=SimpleNamespace(
            optimizer_type="adamw",
            lr=lr,
            adam_beta1=0.9,
            adam_beta2=0.95,
            weight_decay=weight_decay,
            adam_epsilon=1e-8,
        ),
        checkpoint=SimpleNamespace(no_load_optim=True),
    )


def _fill_zero_grads(params, *, force: bool = False) -> None:
    """Simulate FSDP materializing zero grads.

    ``force=True`` also fills grads on ``requires_grad=False`` params that are
    still captured by an optimizer built before freeze (ordering-bug path).
    """
    for p in params:
        if force or p.requires_grad:
            p.grad = torch.zeros_like(p)


def _indexer_param_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: t.detach().clone() for n, t in model.indexer.named_parameters()}


def _assert_indexer_unchanged(model: nn.Module, before: dict[str, torch.Tensor]) -> None:
    for name, p in model.indexer.named_parameters():
        assert torch.equal(p, before[name]), f"{name} moved under freeze"


def _assert_indexer_shrunk(model: nn.Module, before: dict[str, torch.Tensor]) -> None:
    """AdamW decoupled WD with zero grads: ``p <- p * (1 - lr * wd)`` (approx)."""
    moved = False
    for name, p in model.indexer.named_parameters():
        if torch.equal(p, before[name]):
            continue
        moved = True
        before_abs = before[name].detach().float().abs().sum().item()
        after_abs = p.detach().float().abs().sum().item()
        assert after_abs < before_abs, (
            f"{name}: expected WD shrink, before_abs={before_abs}, after_abs={after_abs}"
        )
    assert moved, "indexer params unchanged; expected AdamW weight decay to shrink them"


# ---------------------------------------------------------------------------
# 1. Helper + AdamW WD semantics (zero-filled grads = FSDP-like)
# ---------------------------------------------------------------------------


def test_unfrozen_zero_grad_adamw_wd_shrinks_indexer():
    """No freeze + in AdamW + zero grads → decoupled WD shrinks weights.

    Plain ``grad is None`` would be skipped by AdamW; FSDP-style zero-filled
    grads still take the WD path — the failure mode freeze exists to prevent.
    """
    torch.manual_seed(0)
    model = _IndexerToyModel()
    assert all(p.requires_grad for p in model.indexer.parameters())

    before = _indexer_param_snapshot(model)
    opt = torch.optim.AdamW(_trainable_params(model), lr=0.1, weight_decay=0.1)
    loss = model.scale * 2.0
    loss.backward()
    for p in model.indexer.parameters():
        assert p.grad is None, "scale-only loss must leave indexer.grad as None"
    _fill_zero_grads(model.indexer.parameters())
    opt.step()

    _assert_indexer_shrunk(model, before)


def test_freeze_helper_excludes_from_optimizer_and_blocks_wd():
    """freeze helper → requires_grad False; AdamW+wd cannot touch indexer."""
    torch.manual_seed(0)
    model = _IndexerToyModel()
    freeze_csa_indexer_params(model)

    for p in model.indexer.parameters():
        assert p.requires_grad is False
    trainable = _trainable_params(model)
    indexer_ids = {id(q) for q in model.indexer.parameters()}
    assert all(id(p) not in indexer_ids for p in trainable)
    assert any(model.scale is p for p in trainable)

    before = _indexer_param_snapshot(model)
    opt = torch.optim.AdamW(trainable, lr=0.1, weight_decay=0.1)
    loss = model.scale * 2.0
    loss.backward()
    opt.step()

    for name, p in model.indexer.named_parameters():
        assert p.grad is None
        assert torch.equal(p, before[name])


# ---------------------------------------------------------------------------
# 2. DeepseekV4PostInitModel + real setup_optimizer wiring
# ---------------------------------------------------------------------------


def test_post_init_freeze_on_then_setup_optimizer_skips_indexer():
    """Default training path: PostInit freeze ON → setup_optimizer omits indexer."""
    torch.manual_seed(1)
    model = _IndexerToyModel()
    DeepseekV4PostInitModel(_training_ns(freeze_csa_indexer=True))(model)

    for p in model.indexer.parameters():
        assert p.requires_grad is False

    opt = setup_optimizer(_setup_optimizer_cfg(), model)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    for p in model.indexer.parameters():
        assert id(p) not in opt_ids

    before = _indexer_param_snapshot(model)
    loss = model.scale * 2.0
    loss.backward()
    # Even if something zero-fills frozen grads, they are not in the optimizer.
    _fill_zero_grads(model.parameters())
    opt.step()
    _assert_indexer_unchanged(model, before)


def test_post_init_freeze_off_setup_optimizer_wd_shrinks():
    """PostInit freeze OFF → indexer stays in AdamW and WD shrinks under zero grads."""
    torch.manual_seed(2)
    model = _IndexerToyModel()
    DeepseekV4PostInitModel(_training_ns(freeze_csa_indexer=False))(model)

    for p in model.indexer.parameters():
        assert p.requires_grad is True

    opt = setup_optimizer(_setup_optimizer_cfg(weight_decay=0.1, lr=0.1), model)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert all(id(p) in opt_ids for p in model.indexer.parameters())

    before = _indexer_param_snapshot(model)
    loss = model.scale * 2.0
    loss.backward()
    _fill_zero_grads(model.indexer.parameters())
    opt.step()
    _assert_indexer_shrunk(model, before)


def test_freeze_after_optimizer_still_allows_wd():
    """Ordering contract: freeze must run before optimizer construction."""
    torch.manual_seed(3)
    model = _IndexerToyModel()
    opt = setup_optimizer(_setup_optimizer_cfg(weight_decay=0.1, lr=0.1), model)
    # Too late: params already captured by AdamW.
    freeze_csa_indexer_params(model)
    for p in model.indexer.parameters():
        assert p.requires_grad is False

    before = _indexer_param_snapshot(model)
    loss = model.scale * 2.0
    loss.backward()
    # Params already in AdamW; force zero grads even after requires_grad=False.
    _fill_zero_grads(model.indexer.parameters(), force=True)
    opt.step()
    _assert_indexer_shrunk(model, before)


# ---------------------------------------------------------------------------
# 3. Real indexer forward (topk-only) + train step
# ---------------------------------------------------------------------------


def test_csa_compressor_disables_grad_only_for_indexer():
    torch.manual_seed(4)
    compressor = DeepseekV4CSACompressor(_indexer_hf_config())
    nn.init.zeros_(compressor._position_bias_holder.weight)
    indexer = _GradModeIndexer()
    compressor.indexer = indexer

    hidden = torch.randn(1, SEQ_LEN, HIDDEN, requires_grad=True)
    q_residual = torch.randn(1, SEQ_LEN, Q_LORA, requires_grad=True)
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
    compressed_kv, _, indices = compressor(
        hidden,
        q_residual,
        position_ids,
        past_key_values=None,
        layer_idx=0,
    )

    assert indexer.grad_enabled is False
    assert indices.requires_grad is False
    assert compressed_kv.requires_grad is True


def test_real_indexer_topk_forward_then_frozen_train_step():
    """Real eager indexer forward → discrete indices; freeze ON → weights stay.

    Loss does not depend on indexer scores (only ``scale``), matching CSA
    top-k selection with no indexer KL.
    """
    torch.manual_seed(4)
    model = _IndexerToyModel()
    model.train()
    DeepseekV4PostInitModel(_training_ns(freeze_csa_indexer=True))(model)
    opt = setup_optimizer(_setup_optimizer_cfg(weight_decay=0.1, lr=0.1), model)

    indices = model.run_topk()
    assert indices.dtype == torch.long
    assert indices.ndim == 3
    assert indices.shape[-1] == INDEX_TOPK

    before = _indexer_param_snapshot(model)
    # Indices are non-differentiable; train step only updates ``scale``.
    loss = model.scale * (1.0 + indices.float().mean().detach())
    loss.backward()
    for p in model.indexer.parameters():
        assert p.grad is None
    opt.step()
    _assert_indexer_unchanged(model, before)
    assert not torch.equal(model.scale.detach(), torch.ones(()))


def test_real_indexer_topk_forward_unfrozen_zero_grad_wd_shrinks():
    """Real forward + freeze OFF + FSDP-like zero grads → WD shrinks indexer."""
    torch.manual_seed(5)
    model = _IndexerToyModel()
    model.train()
    DeepseekV4PostInitModel(_training_ns(freeze_csa_indexer=False))(model)
    opt = setup_optimizer(_setup_optimizer_cfg(weight_decay=0.1, lr=0.1), model)

    indices = model.run_topk()
    assert indices.dtype == torch.long

    before = _indexer_param_snapshot(model)
    loss = model.scale * (1.0 + indices.float().mean().detach())
    loss.backward()
    for p in model.indexer.parameters():
        assert p.grad is None
    _fill_zero_grads(model.indexer.parameters())
    opt.step()
    _assert_indexer_shrunk(model, before)
