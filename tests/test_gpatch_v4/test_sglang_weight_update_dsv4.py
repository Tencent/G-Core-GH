"""Unit tests for FP4-QAT-aware SGLang DeepSeek-V4 weight updates."""

from unittest.mock import patch

import pytest
import torch

from gpatch_v4.generation_backend.sglang_model_specific import sglang_weight_update_dsv4 as update
from gpatch_v4.models.deepseek_v4.kernel.quantize_kernels import fp4_qat_to_sgl_fp8


_EXPERT_KEY = "layers.0.ffn.experts.0.w1.weight"
_DENSE_KEY = "layers.0.attn.wq_b.weight"


def _expert_weight() -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randn(128, 128, dtype=torch.bfloat16)


@pytest.mark.parametrize(
    ("m", "n", "seed"),
    [
        pytest.param(128, 128, 7, id="128x128"),
        pytest.param(128, 384, 11, id="128x384"),
        pytest.param(256, 256, 17, id="256x256"),
        pytest.param(384, 640, 23, id="384x640"),
    ],
)
def test_fp4_qat_expert_update_matches_torch_for_random_shapes(
    m: int,
    n: int,
    seed: int,
) -> None:
    """Exercise the real TileLang update path on multiple aligned shapes."""
    pytest.importorskip("tilelang")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required by the TileLang quantization kernel")

    torch.manual_seed(seed)
    weight = torch.randn(m, n, device="cuda", dtype=torch.float32)

    expected_weight, expected_scale = fp4_qat_to_sgl_fp8(weight)
    output = update.quantize_fp4_qat_expert(_EXPERT_KEY, weight)
    actual_weight, actual_scale = output[0][1], output[1][1]
    torch.cuda.synchronize()

    assert actual_weight.shape == (m, n)
    assert actual_scale.shape == (m // 128, n // 128)
    assert torch.equal(
        actual_weight.contiguous().view(torch.uint8),
        expected_weight.contiguous().view(torch.uint8),
    )
    assert torch.equal(actual_scale, expected_scale)


def test_fp4_qat_expert_update_uses_tilelang_kernel() -> None:
    weight = _expert_weight().float()
    qweight = torch.empty_like(weight)
    fp8_scale = torch.ones(1, 1)

    with (
        patch.object(
            update,
            "_tilelang_fp4_qat_to_sgl_fp8",
            return_value=(qweight, fp8_scale),
        ) as tilelang_quant,
        patch.object(update, "quantize_fp8") as regular_fp8,
    ):
        output = list(
            update.iter_fp8_quantized_weights(
                iter([(_EXPERT_KEY, weight)]),
                fp4_qat=True,
            )
        )

    tilelang_quant.assert_called_once_with(weight)
    regular_fp8.assert_not_called()
    assert [name for name, _ in output] == [
        _EXPERT_KEY,
        "layers.0.ffn.experts.0.w1.weight_scale_inv",
    ]
    assert output[0][1] is qweight
    assert torch.equal(output[1][1], fp8_scale)


def test_fp4_qat_expert_passes_fp32_master_weight_to_tilelang() -> None:
    master_weight = _expert_weight().float()
    kernel_output = (torch.empty_like(master_weight), torch.ones(1, 1))

    with patch.object(
        update,
        "_tilelang_fp4_qat_to_sgl_fp8",
        return_value=kernel_output,
    ) as tilelang_quant:
        update.quantize_fp4_qat_expert(_EXPERT_KEY, master_weight)

    tilelang_quant.assert_called_once_with(master_weight)
    assert tilelang_quant.call_args.args[0].dtype == torch.float32


def test_fp4_qat_does_not_change_dense_weight_quantizer() -> None:
    dense_weight = torch.ones(128, 128, dtype=torch.bfloat16)
    sentinel = [
        (_DENSE_KEY, torch.empty(0)),
        ("layers.0.attn.wq_b.weight_scale_inv", torch.empty(0)),
    ]

    with patch.object(update, "quantize_fp8", return_value=sentinel) as regular_fp8:
        output = list(
            update.iter_fp8_quantized_weights(
                iter([(_DENSE_KEY, dense_weight)]),
                fp4_qat=True,
            )
        )

    regular_fp8.assert_called_once_with(_DENSE_KEY, dense_weight, moe_deepgemm=False)
    assert [name for name, _ in output] == [name for name, _ in sentinel]
    assert output[0][1] is sentinel[0][1]
    assert output[1][1] is sentinel[1][1]


def test_fp4_qat_disabled_uses_regular_expert_fp8_quantizer() -> None:
    weight = _expert_weight()
    sentinel = [
        (_EXPERT_KEY, torch.empty(0)),
        ("layers.0.ffn.experts.0.w1.weight_scale_inv", torch.empty(0)),
    ]

    with patch.object(update, "quantize_fp8", return_value=sentinel) as regular_fp8:
        output = list(update.iter_fp8_quantized_weights(iter([(_EXPERT_KEY, weight)])))

    regular_fp8.assert_called_once_with(_EXPERT_KEY, weight, moe_deepgemm=False)
    assert [name for name, _ in output] == [name for name, _ in sentinel]
    assert output[0][1] is sentinel[0][1]
    assert output[1][1] is sentinel[1][1]


def test_fp4_qat_expert_uses_deepgemm_scale_layout_when_requested() -> None:
    transformed_scale = torch.empty(1, 1, dtype=torch.uint8)
    qweight = torch.empty(128, 128)
    fp8_scale = torch.ones(1, 1)

    with (
        patch.object(
            update,
            "_tilelang_fp4_qat_to_sgl_fp8",
            return_value=(qweight, fp8_scale),
        ),
        patch.object(update, "_init_fp8_quantizers") as init_quantizers,
        patch.object(update, "_use_ue8m0_scale", return_value=True) as use_ue8m0,
        patch.object(update, "_transform_scale_ue8m0_fn", return_value=transformed_scale) as transform,
    ):
        output = update.quantize_fp4_qat_expert(
            _EXPERT_KEY,
            _expert_weight().float(),
            moe_deepgemm=True,
        )

    init_quantizers.assert_called_once_with()
    use_ue8m0.assert_called_once_with(_EXPERT_KEY, moe_deepgemm=True)
    transform.assert_called_once_with(fp8_scale, mn=128)
    assert output[1][0] == "layers.0.ffn.experts.0.w1.weight_scale_inv"
    assert output[1][1] is transformed_scale
