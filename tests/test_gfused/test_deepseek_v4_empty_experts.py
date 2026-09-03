from types import SimpleNamespace

import torch

import gpatch_v4.models.deepseek_v4.modeling_deepseek_v4 as dsv4_modeling
from gpatch_v4.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Experts


def test_empty_expert_tokens_keep_weights_in_autograd_graph() -> None:
    config = SimpleNamespace(
        num_local_experts=2,
        hidden_size=8,
        intermediate_size=4,
        hidden_act="silu",
        swiglu_limit=7.0,
    )
    experts = DeepseekV4Experts(config)
    torch.nn.init.zeros_(experts.gate_up_proj)
    torch.nn.init.zeros_(experts.down_proj)
    hidden_states = torch.empty(0, config.hidden_size, requires_grad=True)
    top_k_weights = torch.empty(0, requires_grad=True)

    output = experts.fwd_gmm(
        hidden_states,
        torch.empty(0, dtype=torch.long),
        top_k_weights,
    )
    output.sum().backward()

    assert output.shape == hidden_states.shape
    assert hidden_states.grad is not None
    assert top_k_weights.grad is not None
    assert experts.gate_up_proj.grad is not None
    assert experts.down_proj.grad is not None
    assert torch.count_nonzero(experts.gate_up_proj.grad) == 0
    assert torch.count_nonzero(experts.down_proj.grad) == 0
    assert torch.count_nonzero(top_k_weights.grad) == 0


def test_deepep_empty_valid_slots_keep_weights_in_autograd_graph(monkeypatch) -> None:
    # 准备：DeepEP 收到 token 行，但全部 slot 为 -1（本 rank 无有效 expert）
    config = SimpleNamespace(
        num_local_experts=2,
        hidden_size=8,
        intermediate_size=4,
        hidden_act="silu",
        swiglu_limit=7.0,
        ep_backend="deepep",
    )
    experts = DeepseekV4Experts(config)
    torch.nn.init.zeros_(experts.gate_up_proj)
    torch.nn.init.zeros_(experts.down_proj)
    experts.ep_group = object()

    def fake_dispatch(hidden_states, top_k_index, top_k_weights, *_args):
        recv_expert = torch.full(
            (hidden_states.shape[0], top_k_index.shape[-1]), -1, dtype=torch.long
        )
        return hidden_states, recv_expert, top_k_weights, None, object()

    def fake_combine(recv_out, *_args):
        return recv_out

    monkeypatch.setattr(dsv4_modeling, "fused_dispatch", fake_dispatch)
    monkeypatch.setattr(dsv4_modeling, "fused_combine", fake_combine)

    hidden_states = torch.randn(2, config.hidden_size, requires_grad=True)
    top_k_weights = torch.ones((2, 2), requires_grad=True)
    output = experts(
        hidden_states,
        torch.zeros((2, 2), dtype=torch.long),
        top_k_weights,
    )
    output.sum().backward()

    assert hidden_states.grad is not None
    assert top_k_weights.grad is not None
    assert experts.gate_up_proj.grad is not None
    assert experts.down_proj.grad is not None
    assert torch.count_nonzero(experts.gate_up_proj.grad) == 0
    assert torch.count_nonzero(experts.down_proj.grad) == 0
    assert torch.count_nonzero(top_k_weights.grad) == 0
