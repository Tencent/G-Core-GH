# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.

import os
import tempfile
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

import gpatch_v4.models.deepseek_v4.modeling_deepseek_v4 as dsv4_modeling
from gpatch_v4.models.deepseek_v4.fp8_tensor import Fp8TensorAg, Fp8TensorTrain
from gpatch_v4.models.deepseek_v4.hp import apply_hp
from gpatch_v4.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts
from gpatch_v4.models.deepseek_v4.weight_export import (
    _iter_deepseek_v4_gathered_state_dict,
)


def _make_experts(*, ep_backend: str = "eager") -> DeepseekV4Experts:
    config = SimpleNamespace(
        num_local_experts=2,
        hidden_size=8,
        intermediate_size=4,
        hidden_act="silu",
        swiglu_limit=7.0,
        ep_backend=ep_backend,
    )
    return DeepseekV4Experts(config)


def _wrap_fp8_expert_weights(experts: DeepseekV4Experts) -> None:
    for name in ("gate_up_proj", "down_proj"):
        weight = getattr(experts, name)
        data = torch.zeros(weight.shape, dtype=torch.uint8)
        scale = torch.ones((weight.shape[0], 1, 1), dtype=torch.float8_e8m0fnu)
        setattr(
            experts,
            name,
            torch.nn.Parameter(Fp8TensorTrain(data, scale, weight.dtype)),
        )


@pytest.mark.parametrize(
    ("fp8_qat", "fp4_qat"),
    ((True, False), (False, True), (True, True)),
)
def test_apply_hp_rejects_fp8_with_qat(fp8_qat: bool, fp4_qat: bool) -> None:
    with pytest.raises(ValueError, match="fp8 is incompatible"):
        apply_hp(
            SimpleNamespace(),
            SimpleNamespace(),
            fp8=True,
            fp8_qat=fp8_qat,
            fp4_qat=fp4_qat,
        )


def test_apply_hp_rejects_fsdp_fp8_gather_without_fp8() -> None:
    with pytest.raises(ValueError, match="fsdp_fp8_gather requires fp8=True"):
        apply_hp(
            SimpleNamespace(),
            SimpleNamespace(),
            fp8=False,
            fsdp_fp8_gather=True,
        )


def test_empty_expert_tokens_keep_fp8_weights_in_autograd_graph() -> None:
    experts = _make_experts()
    _wrap_fp8_expert_weights(experts)
    hidden_states = torch.empty(0, experts.hidden_dim, requires_grad=True)

    output = experts.fwd_gmm(
        hidden_states,
        torch.empty(0, dtype=torch.long),
        torch.empty(0),
    )
    output.sum().backward()

    assert hidden_states.grad is not None
    assert experts.gate_up_proj.grad is not None
    assert experts.down_proj.grad is not None
    assert torch.count_nonzero(experts.gate_up_proj.grad) == 0
    assert torch.count_nonzero(experts.down_proj.grad) == 0


def test_deepep_empty_dispatch_keeps_fp8_weights_in_autograd_graph(monkeypatch) -> None:
    experts = _make_experts(ep_backend="deepep")
    experts.ep_group = object()
    _wrap_fp8_expert_weights(experts)

    def fake_dispatch(hidden_states, *_args):
        recv_expert = torch.full((1, 1), -1, dtype=torch.long)
        recv_weight = hidden_states.new_zeros((1, 1))
        return hidden_states, recv_expert, recv_weight, None, object()

    def fake_combine(recv_out, *_args):
        return recv_out

    monkeypatch.setattr(dsv4_modeling, "fused_dispatch", fake_dispatch)
    monkeypatch.setattr(dsv4_modeling, "fused_combine", fake_combine)

    hidden_states = torch.randn(1, experts.hidden_dim, requires_grad=True)
    output = experts(
        hidden_states,
        torch.zeros((1, 1), dtype=torch.long),
        torch.ones((1, 1)),
    )
    output.sum().backward()

    assert hidden_states.grad is not None
    assert experts.gate_up_proj.grad is not None
    assert experts.down_proj.grad is not None
    assert torch.count_nonzero(experts.gate_up_proj.grad) == 0
    assert torch.count_nonzero(experts.down_proj.grad) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_weight_export_gathers_fp8_wrapped_master() -> None:
    if dist.is_initialized():
        pytest.skip("requires a private single-rank process group")

    fd, path = tempfile.mkstemp()
    os.close(fd)
    try:
        dist.init_process_group(
            "nccl",
            init_method=f"file://{path}",
            rank=0,
            world_size=1,
        )
        mesh = init_device_mesh("cuda", (1,))
        master = torch.randn(2, 128, 128, device="cuda", dtype=torch.float32)
        sharded = DTensor.from_local(
            Fp8TensorAg(master),
            device_mesh=mesh,
            placements=[Shard(0)],
            shape=master.shape,
            stride=master.stride(),
        )
        model = SimpleNamespace(
            _ep_size=1,
            _ep_group=dist.group.WORLD,
            _ep_fsdp_mesh=mesh,
            state_dict=lambda: OrderedDict(
                [("model.layers.0.mlp.experts.down_proj", sharded)]
            ),
        )

        exported = dict(_iter_deepseek_v4_gathered_state_dict(model))
        output = exported["layers.0.mlp.experts.down_proj"]
        assert not isinstance(output, Fp8TensorAg)
        torch.testing.assert_close(output, master, rtol=0.0, atol=0.0)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if os.path.exists(path):
            os.unlink(path)
