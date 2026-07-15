from types import SimpleNamespace

import pytest
import torch
from torch import nn


if not torch.cuda.is_available():
    pytest.skip("FP4 QAT requires CUDA", allow_module_level=True)

from gpatch_v4.models.deepseek_v4.qat import fp4_qat_linear, fp4_simulate_qat
from gpatch_v4.models.deepseek_v4 import modeling_deepseek_v4


def test_fp4_simulate_qat_round_trip_and_ste():
    values = torch.tensor(
        [
            -5.5, -4.75, -3.75, -3.25, -2.75, -2.25, -1.85, -1.65,
            -1.35, -1.15, -0.9, -0.6, -0.4, -0.2, -0.1, 0.0,
            0.1, 0.2, 0.4, 0.6, 0.9, 1.15, 1.35, 1.65,
            1.85, 2.25, 2.75, 3.25, 3.75, 4.75, 5.25, 6.0,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected_values = torch.tensor(
        [
            -6.0, -4.0, -4.0, -3.0, -3.0, -2.0, -2.0, -1.5,
            -1.5, -1.0, -1.0, -0.5, -0.5, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5,
            2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 6.0, 6.0,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = torch.stack(
        [
            torch.cat([values, values]),
            torch.cat([values * 2, values * 2]),
        ],
    ).unsqueeze(1).expand(-1, 3, -1).clone()
    expected = torch.stack(
        [
            torch.cat([expected_values, expected_values]),
            torch.cat([expected_values * 2, expected_values * 2]),
        ],
    ).unsqueeze(1).expand(-1, 3, -1)
    x.requires_grad_()
    grad_out = torch.randn_like(x)

    y = fp4_simulate_qat(x)
    y.backward(grad_out)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()
    assert torch.equal(y, expected)
    assert torch.equal(x.grad, grad_out)


def test_fp4_qat_linear():
    module = nn.Linear(64, 32, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    output = fp4_qat_linear(module, x)
    output.float().square().mean().backward()

    assert output.shape == (4, 32)
    assert torch.isfinite(output).all()
    assert module.weight.grad is not None
    assert torch.isfinite(module.weight.grad).all()


def test_fp4_qat_precedes_fp8_qat(monkeypatch):
    config = SimpleNamespace(
        num_local_experts=1,
        hidden_size=32,
        intermediate_size=32,
        hidden_act="silu",
        swiglu_limit=7.0,
        fp4_qat=True,
        fp8_qat=True,
        fp8=False,
    )
    experts = modeling_deepseek_v4.DeepseekV4Experts(config).cuda().bfloat16()
    fp4_calls = []

    def record_fp4(tensor):
        fp4_calls.append(tensor)
        return tensor

    def fail_fp8(*args):
        pytest.fail("fp8_qat must not quantize routed MoE weights when fp4_qat is enabled")

    monkeypatch.setattr(modeling_deepseek_v4, "fp4_simulate_qat", record_fp4)
    monkeypatch.setattr(modeling_deepseek_v4, "fp8_simulate_qat", fail_fp8)

    output = experts.fwd_gmm(
        torch.randn(2, 32, device="cuda", dtype=torch.bfloat16),
        torch.zeros(2, device="cuda", dtype=torch.long),
        torch.ones(2, device="cuda", dtype=torch.bfloat16),
    )

    assert len(fp4_calls) == 2
    assert fp4_calls[0] is experts.gate_up_proj
    assert fp4_calls[1] is experts.down_proj
    assert torch.isfinite(output).all()
