"""HF adapter load/export tests for GroupedExpert EP mapping (gcore checkpoint).

- Load: ``_load_adapter_from_hf_peft`` with ``ep_size=8``, ranks 1/3/6
- Export: ``gather_lora_state_dict`` mocked EP=8 all_gather, ranks 1/3/6
- Round-trip (EP=1): GroupedExpert + share_true

Lives in gcore-dev (depends on ``gpatch_v4``); mbridge must not reverse-depend.
"""

from __future__ import annotations

import os
import re
from typing import Tuple

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from safetensors.torch import save_file

from gpatch_v4.training_backend.megatron_backend import checkpoint as ckpt_mod
from gpatch_v4.training_backend.megatron_backend.checkpoint import (
    _load_adapter_from_hf_peft,
)
from mbridge.peft.grouped_expert_adapter import GroupedExpertLinearAdapter
from mbridge.peft.lora import LoRA, gather_lora_state_dict, mcore_adapter_name_to_hf
from mbridge.peft.lora_layers import LoRALinear
from mbridge.peft.utils import ParallelLinearAdapter, init_method_normal


@pytest.fixture(scope="module")
def te_mp():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if TEColumnParallelGroupedLinear is None:
        pytest.skip("needs Transformer Engine GroupedLinear")

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29581")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
    model_parallel_cuda_manual_seed(1234)
    yield
    try:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def _te_config(*, hidden: int = 16, ffn: int = 16, num_local_experts: int = 2) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        ffn_hidden_size=ffn,
        num_attention_heads=1,
        num_moe_experts=num_local_experts,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        perform_initialization=True,
        use_cpu_initialization=False,
    )


def _build_te_grouped_linear(
    *,
    num_gemms: int = 2,
    input_size: int = 16,
    output_size: int = 16,
) -> TEColumnParallelGroupedLinear:
    config = _te_config(hidden=input_size, ffn=output_size, num_local_experts=num_gemms)
    return TEColumnParallelGroupedLinear(
        num_gemms=num_gemms,
        input_size=input_size,
        output_size=output_size,
        config=config,
        init_method=init_method_normal(0.02),
        bias=False,
        skip_bias_add=True,
        is_expert=True,
        tp_comm_buffer_name="fc1",
    ).cuda()


class _GroupedExpertModel(nn.Module):
    def __init__(self, base: TEColumnParallelGroupedLinear) -> None:
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([nn.Module()])
        self.decoder.layers[0].mlp = nn.Module()
        self.decoder.layers[0].mlp.experts = nn.Module()
        self.decoder.layers[0].mlp.experts.linear_fc2 = base


def _wrap_grouped_expert(
    te_mp,
    *,
    size: int = 16,
    dim: int = 16,
    alpha: float = 16.0,
    num_gemms: int = 2,
) -> Tuple[LoRALinear, TEColumnParallelGroupedLinear, GroupedExpertLinearAdapter]:
    del te_mp
    base = _build_te_grouped_linear(num_gemms=num_gemms, input_size=size, output_size=size)
    adapter = GroupedExpertLinearAdapter(
        size,
        size,
        dim,
        num_local_experts=num_gemms,
        base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
        activation="identity",
        input_is_parallel=False,
        model_parallel_config=base.config,
        alpha=alpha,
        params_device=torch.device("cuda", 0),
        params_dtype=torch.bfloat16,
    )
    return LoRALinear(base, adapter), base, adapter


def _make_experts_fc2_root(wrapped: nn.Module) -> nn.Module:
    root = nn.Module()
    root.decoder = nn.Module()
    root.decoder.layers = nn.ModuleList([nn.Module()])
    root.decoder.layers[0].mlp = nn.Module()
    root.decoder.layers[0].mlp.experts = nn.Module()
    root.decoder.layers[0].mlp.experts.linear_fc2 = wrapped
    return root


def _fp_in(ep_r: int, i: int) -> float:
    # Keep values in 1..N so they are exactly representable in bf16.
    return float(ep_r * 2 + i + 1)


def _fp_out(ep_r: int, i: int) -> float:
    return float(ep_r * 2 + i + 1 + 32)


def _hf_expert_key(state: dict, expert_id: int, which: str) -> str:
    suffix = f".{expert_id}.{which}.weight"
    keys = [k for k in state if k.endswith(suffix)]
    assert len(keys) == 1, (suffix, sorted(state))
    return keys[0]


def _build_full_ep_hf_state(
    *,
    module_name: str,
    adapter: GroupedExpertLinearAdapter,
    ep_size: int,
    num_local: int,
) -> dict:
    """HF adapter dict as if exported under EP=ep_size (all global experts)."""
    state = {}
    for ep_r in range(ep_size):
        for i in range(num_local):
            gid = ep_r * num_local + i
            tin = torch.full(
                tuple(adapter.linear_in.weight[i].shape),
                _fp_in(ep_r, i),
                dtype=torch.float32,
            )
            tout = torch.full(
                tuple(adapter.linear_out.weight[i].shape),
                _fp_out(ep_r, i),
                dtype=torch.float32,
            )
            for part, tensor in (("linear_in", tin), ("linear_out", tout)):
                mcore_key = f"{module_name}.adapter.{gid}.{part}.weight"
                state[mcore_adapter_name_to_hf(mcore_key, bridge=None)] = tensor
    return state


@pytest.mark.parametrize("ep_rank", [1, 3, 6])
def test_load_adapter_grouped_expert_ep8_rank_mapping(
    te_mp, tmp_path, monkeypatch, ep_rank
):
    """ep_size=8: load only maps global_id=ep_rank*n+i into local weight[i]."""
    ep_size = 8
    num_local = 2
    wrapped, _, adapter = _wrap_grouped_expert(
        te_mp, size=16, dim=8, alpha=8.0, num_gemms=num_local
    )
    root = _make_experts_fc2_root(wrapped)
    module_name = "decoder.layers.0.mlp.experts.linear_fc2"

    state = _build_full_ep_hf_state(
        module_name=module_name,
        adapter=adapter,
        ep_size=ep_size,
        num_local=num_local,
    )
    save_file(state, str(tmp_path / "adapter_model.safetensors"))
    with torch.no_grad():
        adapter.linear_in.weight.zero_()
        adapter.linear_out.weight.zero_()

    monkeypatch.setattr(
        ckpt_mod.mpu, "get_expert_model_parallel_rank", lambda: ep_rank
    )
    _load_adapter_from_hf_peft([root], str(tmp_path), bridge=None)

    for i in range(num_local):
        expect_in = torch.full(
            tuple(adapter.linear_in.weight[i].shape),
            _fp_in(ep_rank, i),
            dtype=torch.float32,
        )
        expect_out = torch.full(
            tuple(adapter.linear_out.weight[i].shape),
            _fp_out(ep_rank, i),
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            adapter.linear_in.weight[i].detach().float().cpu(), expect_in
        )
        torch.testing.assert_close(
            adapter.linear_out.weight[i].detach().float().cpu(), expect_out
        )


@pytest.mark.parametrize("ep_rank", [1, 3, 6])
def test_export_grouped_expert_ep8_rank_indexing(te_mp, monkeypatch, ep_rank):
    """ep_size=8 gather: HF key for global_id=ep_rank*n+i equals that EP pack slice."""
    import mbridge.peft.lora as lora_mod

    ep_size = 8
    num_local = 2
    wrapped, _, adapter = _wrap_grouped_expert(
        te_mp, size=16, dim=8, alpha=8.0, num_gemms=num_local
    )
    # Distinct packed weights for every EP rank (all_gather simulation).
    packs_in = []
    packs_out = []
    for ep_r in range(ep_size):
        pin = torch.empty_like(adapter.linear_in.weight)
        pout = torch.empty_like(adapter.linear_out.weight)
        with torch.no_grad():
            for i in range(num_local):
                pin[i].fill_(_fp_in(ep_r, i))
                pout[i].fill_(_fp_out(ep_r, i))
        packs_in.append(pin)
        packs_out.append(pout)

    # Local adapter holds ep_rank's pack (what this rank would contribute).
    with torch.no_grad():
        adapter.linear_in.weight.copy_(packs_in[ep_rank])
        adapter.linear_out.weight.copy_(packs_out[ep_rank])

    def fake_all_gather(output_tensor_list, tensor, group=None, async_op=False):
        del group, async_op
        assert len(output_tensor_list) == ep_size
        if tensor.shape == adapter.linear_in.weight.shape:
            for r in range(ep_size):
                output_tensor_list[r].copy_(packs_in[r])
        elif tensor.shape == adapter.linear_out.weight.shape:
            for r in range(ep_size):
                output_tensor_list[r].copy_(packs_out[r])
        else:
            raise AssertionError(f"unexpected all_gather shape {tuple(tensor.shape)}")
        return None

    monkeypatch.setattr(
        lora_mod.parallel_state,
        "get_expert_model_parallel_world_size",
        lambda: ep_size,
    )
    monkeypatch.setattr(
        lora_mod.parallel_state,
        "get_expert_model_parallel_group",
        lambda: object(),
    )
    monkeypatch.setattr(lora_mod.dist, "all_gather", fake_all_gather)

    root = _make_experts_fc2_root(wrapped)
    state = gather_lora_state_dict([root], bridge=None)

    for i in range(num_local):
        gid = ep_rank * num_local + i
        torch.testing.assert_close(
            state[_hf_expert_key(state, gid, "lora_A")].cuda(),
            packs_in[ep_rank][i],
        )
        torch.testing.assert_close(
            state[_hf_expert_key(state, gid, "lora_B")].cuda(),
            packs_out[ep_rank][i],
        )


def test_hf_adapter_roundtrip_grouped_expert(te_mp, tmp_path):
    wrapped, _, adapter = _wrap_grouped_expert(
        te_mp, size=16, dim=16, alpha=16.0, num_gemms=2
    )
    device = adapter.linear_in.weight.device
    dtype = adapter.linear_in.weight.dtype
    gen = torch.Generator(device="cpu").manual_seed(21)
    with torch.no_grad():
        adapter.linear_in.weight.copy_(
            torch.randn(
                adapter.linear_in.weight.shape, generator=gen, dtype=torch.float32
            ).to(device=device, dtype=dtype)
        )
        adapter.linear_out.weight.copy_(
            torch.randn(
                adapter.linear_out.weight.shape, generator=gen, dtype=torch.float32
            ).to(device=device, dtype=dtype)
        )
    saved_in = adapter.linear_in.weight.detach().clone()
    saved_out = adapter.linear_out.weight.detach().clone()

    root = _make_experts_fc2_root(wrapped)
    state = gather_lora_state_dict([root], bridge=None)
    assert len(state) >= 4
    save_file(state, str(tmp_path / "adapter_model.safetensors"))

    with torch.no_grad():
        adapter.linear_in.weight.zero_()
        adapter.linear_out.weight.zero_()
    assert not torch.equal(adapter.linear_in.weight, saved_in)

    _load_adapter_from_hf_peft([root], str(tmp_path), bridge=None)
    torch.testing.assert_close(adapter.linear_in.weight, saved_in)
    torch.testing.assert_close(adapter.linear_out.weight, saved_out)


def test_hf_adapter_roundtrip_share_true_parallel_adapter(te_mp, tmp_path):
    model = _GroupedExpertModel(
        _build_te_grouped_linear(num_gemms=2, input_size=16, output_size=16)
    )
    lora = LoRA(
        target_modules=["linear_fc2"], dim=8, alpha=8, share_expert_adapters=True
    )
    transformed = lora(model, training=True)
    adapted = transformed.decoder.layers[0].mlp.experts.linear_fc2
    assert isinstance(adapted, LoRALinear)
    assert isinstance(adapted.adapter, ParallelLinearAdapter)

    with torch.no_grad():
        adapted.adapter.linear_in.weight.normal_(std=0.02)
        adapted.adapter.linear_out.weight.normal_(std=0.02)
    saved_in = adapted.adapter.linear_in.weight.detach().clone()
    saved_out = adapted.adapter.linear_out.weight.detach().clone()

    state = gather_lora_state_dict([transformed], bridge=None)
    assert any("lora_A" in k for k in state)
    assert any("lora_B" in k for k in state)
    assert not any(re.search(r"\.\d+\.lora_[AB]\.weight$", k) for k in state)

    save_file(state, str(tmp_path / "adapter_model.safetensors"))
    with torch.no_grad():
        adapted.adapter.linear_in.weight.zero_()
        adapted.adapter.linear_out.weight.zero_()

    _load_adapter_from_hf_peft([transformed], str(tmp_path), bridge=None)
    torch.testing.assert_close(adapted.adapter.linear_in.weight, saved_in)
    torch.testing.assert_close(adapted.adapter.linear_out.weight, saved_out)
