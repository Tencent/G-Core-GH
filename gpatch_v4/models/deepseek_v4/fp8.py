# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""TE Float8BlockScaling grouped linear（无 Parameter，weight 外部传入）。"""

from __future__ import annotations

import os

import torch

# 这样写未必合理，小心有坑...
os.environ.setdefault("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")

import transformer_engine_torch as _tex
from transformer_engine.common.recipe import Float8BlockScaling, Format
from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm
from transformer_engine.pytorch.quantized_tensor import (
    QuantizedTensorStorage,
    prepare_for_saving,
    restore_from_saved,
)
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    Float8BlockQuantizer,
)

__all__ = ["MyGroupedLinearFp8", "_pad_m_splits"]


def _fp8_dtype_e4m3():
    return _tex.DType.kFloat8E4M3


def _make_recipe():
    return Float8BlockScaling(fp8_format=Format.E4M3)


# Megatron ``get_fp8_align_size``：非 MXFP8 为 16；覆盖 cuBLAS block-scaling 的 m%8。
_FP8_BLOCK_ALIGN = 16


def _pad_m_splits(m_splits: list, align_size: int = _FP8_BLOCK_ALIGN) -> list:
    return [(m + align_size - 1) // align_size * align_size for m in m_splits]


def _pad_rows(inp: torch.Tensor, m_splits: list, padded_m_splits: list) -> torch.Tensor:
    if m_splits == padded_m_splits:
        return inp
    in_features = inp.shape[-1]
    out = torch.empty(
        [sum(padded_m_splits), in_features],
        dtype=inp.dtype,
        device=inp.device,
    )
    _tex.fused_multi_row_padding(
        inp.view(-1, in_features),
        out,
        m_splits,
        padded_m_splits,
    )
    return out


def _unpad_rows(inp: torch.Tensor, m_splits: list, padded_m_splits: list) -> torch.Tensor:
    if m_splits == padded_m_splits:
        return inp
    in_features = inp.shape[-1]
    out = torch.empty(
        [sum(m_splits), in_features],
        dtype=inp.dtype,
        device=inp.device,
    )
    _tex.fused_multi_row_unpadding(
        inp.view(-1, in_features),
        out,
        padded_m_splits,
        m_splits,
    )
    return out


def _make_grouped_fp8_quantizers(num_gemms: int, recipe):
    """按 Float8BlockScaling recipe 为每个 expert 建 input/weight/grad_output quantizer。"""
    fp8_dtype = _fp8_dtype_e4m3()
    x_qparams = recipe.fp8_quant_fwd_inp
    w_qparams = recipe.fp8_quant_fwd_weight
    g_qparams = recipe.fp8_quant_bwd_grad
    input_qs = [
        Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=True,
            amax_epsilon=x_qparams.amax_epsilon,
            force_pow_2_scales=x_qparams.power_2_scale,
            block_scaling_dim=recipe.x_block_scaling_dim,
        ) for _ in range(num_gemms)
    ]
    weight_qs = [
        Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=True,
            amax_epsilon=w_qparams.amax_epsilon,
            force_pow_2_scales=w_qparams.power_2_scale,
            block_scaling_dim=recipe.w_block_scaling_dim,
        ) for _ in range(num_gemms)
    ]
    grad_out_qs = [
        Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=True,
            amax_epsilon=g_qparams.amax_epsilon,
            force_pow_2_scales=g_qparams.power_2_scale,
            block_scaling_dim=recipe.grad_block_scaling_dim,
        ) for _ in range(num_gemms)
    ]
    return input_qs, weight_qs, grad_out_qs


class _TeGroupedGemmFp8(torch.autograd.Function):
    """TE ``split_quantize`` + ``general_grouped_gemm`` 的精简 FP8 grouped linear.

    Weight 布局对齐 DSV4：``weight`` shape ``[E, N, K]``（``y = x @ W^T`` per expert）。
    quantizer / recipe 由外层 ``MyGroupedLinearFp8`` 持有并传入，避免每步重建。

    内部自动按 ``_FP8_BLOCK_ALIGN``(=16) pad/unpad 每个 expert 的 token 数，
    对齐 Megatron ``Fp8Padding`` / ``moe_router_padding_for_quantization``。
    """
    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        weight: torch.Tensor,
        m_splits,  # list[int]，非 Tensor
        input_qs,
        weight_qs,
        grad_out_qs,
        recipe,
    ):
        m_splits = [int(x) for x in m_splits]
        num_gemms = len(m_splits)
        if weight.dim() != 3 or weight.size(0) != num_gemms:
            raise ValueError(
                f"weight shape {tuple(weight.shape)} incompatible with m_splits={m_splits}"
            )
        in_features = weight.size(-1)
        out_features = weight.size(1)
        if inp.size(-1) != in_features:
            raise ValueError(f"inp last dim {inp.size(-1)} != weight K {in_features}")
        if sum(m_splits) != inp.numel() // in_features:
            raise ValueError(
                f"sum(m_splits)={sum(m_splits)} != inp rows={inp.numel() // in_features}"
            )
        if len(input_qs) != num_gemms or len(weight_qs) != num_gemms:
            raise ValueError(
                f"quantizer count mismatch: input={len(input_qs)} weight={len(weight_qs)} "
                f"num_gemms={num_gemms}"
            )

        padded_m_splits = _pad_m_splits(m_splits)

        weight_requires_grad = weight.requires_grad
        for q in input_qs:
            q.set_usage(
                rowwise=True,
                columnwise=weight_requires_grad,
            )
        columnwise_w = inp.requires_grad
        for q in weight_qs:
            q.set_usage(rowwise=True, columnwise=columnwise_w)

        inp_view = inp.reshape(-1, in_features)
        inp_padded = _pad_rows(inp_view, m_splits, padded_m_splits)
        inputmats = _tex.split_quantize(inp_padded, padded_m_splits, input_qs)
        weights_fp8 = [wq(w) for wq, w in zip(weight_qs, weight.unbind(0))]

        activation_dtype = inp.dtype
        out_padded = torch.empty(
            [sum(padded_m_splits), out_features],
            dtype=activation_dtype,
            device=inp.device,
        )
        use_split_acc = recipe.fp8_gemm_fprop.use_split_accumulator
        general_grouped_gemm(
            weights_fp8,
            inputmats,
            [out_padded],
            activation_dtype,
            single_output=True,
            m_splits=padded_m_splits,
            use_bias=False,
            use_split_accumulator=use_split_acc,
        )
        out = _unpad_rows(out_padded, m_splits, padded_m_splits)

        if weight_requires_grad:
            for im in inputmats:
                if isinstance(im, QuantizedTensorStorage):
                    im.update_usage(rowwise_usage=False, columnwise_usage=True)
        else:
            inputmats = [None] * num_gemms

        tensors_to_save, tensor_objects = prepare_for_saving(*inputmats, *weights_fp8)
        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects
        ctx.m_splits = m_splits
        ctx.padded_m_splits = padded_m_splits
        ctx.num_gemms = num_gemms
        ctx.activation_dtype = activation_dtype
        ctx.grad_out_qs = grad_out_qs
        ctx.recipe = recipe
        ctx.inp_shape = inp.shape
        ctx.weight_shape = weight.shape
        ctx.requires_dgrad = inp.requires_grad
        ctx.requires_wgrad = weight_requires_grad
        return out.view(*inp.shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)
        N = ctx.num_gemms
        inputmats = saved[:N]
        weights_fp8 = saved[N:2 * N]
        m_splits = ctx.m_splits
        padded_m_splits = ctx.padded_m_splits

        grad_view = grad_output.contiguous().view(-1, grad_output.shape[-1])
        grad_padded = _pad_rows(grad_view, m_splits, padded_m_splits)
        for q in ctx.grad_out_qs:
            q.set_usage(rowwise=True, columnwise=True)
        grad_output_fp8 = _tex.split_quantize(grad_padded, padded_m_splits, ctx.grad_out_qs)

        dgrad = None
        if ctx.requires_dgrad:
            for w in weights_fp8:
                if isinstance(w, QuantizedTensorStorage):
                    w.update_usage(columnwise_usage=True)
            dgrad_padded = torch.empty(
                (sum(padded_m_splits), ctx.weight_shape[-1]),
                dtype=ctx.activation_dtype,
                device=grad_output.device,
            )
            general_grouped_gemm(
                weights_fp8,
                grad_output_fp8,
                [dgrad_padded],
                ctx.activation_dtype,
                single_output=True,
                layout="NN",
                m_splits=padded_m_splits,
                grad=True,
                use_split_accumulator=ctx.recipe.fp8_gemm_dgrad.use_split_accumulator,
            )
            dgrad = _unpad_rows(dgrad_padded, m_splits, padded_m_splits).view(ctx.inp_shape)

        wgrad = None
        if ctx.requires_wgrad:
            wgrad_list = [
                torch.empty(
                    (ctx.weight_shape[1], ctx.weight_shape[2]),
                    dtype=ctx.activation_dtype,
                    device=grad_output.device,
                ) for _ in range(N)
            ]
            general_grouped_gemm(
                inputmats,
                grad_output_fp8,
                wgrad_list,
                ctx.activation_dtype,
                layout="NT",
                grad=True,
                m_splits=padded_m_splits,
                use_bias=False,
                use_split_accumulator=ctx.recipe.fp8_gemm_wgrad.use_split_accumulator,
            )
            wgrad = torch.stack(wgrad_list, dim=0)

        # inp, weight, m_splits, input_qs, weight_qs, grad_out_qs, recipe
        return dgrad, wgrad, None, None, None, None, None


class MyGroupedLinearFp8(torch.nn.Module):
    """无 Parameter 的 FP8 grouped linear：weight 外部传入，quantizer 常驻复用。

    对齐 DSV4 ``gate_up_proj`` / ``down_proj`` 的 stacked ``[E, N, K]`` 布局，
    避免 ``te.GroupedLinear`` 的 ``weight{i}`` 命名破坏 checkpoint。
    """
    def __init__(self, num_gemms: int, recipe=None):
        super().__init__()
        self.num_gemms = num_gemms
        self.recipe = _make_recipe() if recipe is None else recipe
        self.input_qs, self.weight_qs, self.grad_out_qs = _make_grouped_fp8_quantizers(
            num_gemms,
            self.recipe,
        )

    def forward(
        self,
        inp: torch.Tensor,
        weight: torch.Tensor,
        m_splits: list,
    ) -> torch.Tensor:
        if len(m_splits) != self.num_gemms:
            raise ValueError(f"len(m_splits)={len(m_splits)} != num_gemms={self.num_gemms}")
        return _TeGroupedGemmFp8.apply(
            inp,
            weight,
            m_splits,
            self.input_qs,
            self.weight_qs,
            self.grad_out_qs,
            self.recipe,
        )
