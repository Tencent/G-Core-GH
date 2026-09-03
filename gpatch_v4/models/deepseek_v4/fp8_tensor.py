# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""FSDP2 FP8 parameter all-gather + TE Float8BlockScaling grouped GEMM。

数据流：

1. ``_Float8AllGatherTensor`` 包住完整 FP32 Parameter（stacked ``[E, N, K]``）；
2. ``fully_shard`` 将它切成每 rank 常驻的 FP32 shard；
3. FSDP2 unshard 前调用 ``fsdp_pre_all_gather``，将本地 shard 量化成
   E4M3 + E8M0 per-block (128×128)；
4. FSDP2 all-gather ``(fp8_u8, scale_u8)``（block scale 各 rank 不同，不能像
   torchao tensorwise 那样只走 metadata）；
5. demo forward 把 gather 后的 FP8 weight 直接喂 TE ``general_grouped_gemm``
   （``MyTeGroupedLinearFp8`` 同款路径），不再 BF16 dequant + ``F.linear``；
6. FSDP2 reduce-scatter gradient，optimizer 更新常驻的 FP32 shard。

量化使用 ``quant_fp8_e4m3_scale_e8m0``（eager，128×128 E8M0）。

REFs:
1. https://github.com/pytorch/ao/blob/main/torchao/float8/fsdp_utils.py
2. https://github.com/pytorch/ao/blob/main/torchao/float8/float8_training_tensor.py#L242
3. TransformerEngine/transformer_engine/pytorch/tensor/float8_tensor.py
"""

from __future__ import annotations

from typing import Any

import torch
import torch.utils._pytree as pytree

from gpatch_v4.kernel.quantize.eager_quant_kernels import quant_fp8_e4m3_scale_e8m0

_FP8_BLOCK_SIZE = (128, 128)
# 对齐 torchao `_ops_to_preserve_subclass`，含训练侧 CPU offload/onload 需要的 copy op。
# https://github.com/pytorch/ao/blob/main/torchao/float8/fsdp_utils.py#L92-L103
_PRESERVE_WRAPPER_OPS = {
    torch.ops.aten.as_strided.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.copy_.default,
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.split.Tensor,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten.view.default,
}


def _new_hook_stats() -> dict[str, Any]:
    return dict(
        pre_calls=0,
        post_new_calls=0,
        post_reuse_calls=0,
        compute_dtype=None,
        payload_dtype=None,
        payload_bytes=0,
        fp8_bytes=0,
        scale_bytes=0,
        high_precision_bytes=0,
    )


class Fp8TensorAg(torch.Tensor):
    '''
    逻辑及 backing storage 都是 FP32、但 unshard 时用 FP8 通信的 Parameter wrapper。
    https://github.com/pytorch/ao/blob/main/torchao/float8/fsdp_utils.py#L139

    NOTE: fp8 gathering 目前是 beta 状态，veomni 目前看使用 bf16 gather。torch 目前对于 uneven
    的 param chunk 的 gather 支持还不够完善，scale 的 gathering 需要比较多额外的处理（或额外通信）。
    '''
    @staticmethod
    def __new__(cls, tensor: torch.Tensor) -> Fp8TensorAg:
        # Wrapper subclass 自己不分配 storage；shape/dtype 等 Tensor 元数据从
        # `_tensor` 镜像，真实 FP32 数据保存在 `_tensor`。
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    @classmethod
    def __torch_dispatch__(
        cls,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        # 对齐 torchao WeightWithDynamicFloat8CastTensor.__torch_dispatch__。
        # https://github.com/pytorch/ao/blob/main/torchao/float8/fsdp_utils.py#L177-L208
        if func == torch.ops.aten.detach.default:
            return cls(args[0]._tensor.detach())

        dtype: torch.dtype | None = None

        def unwrap(t: Fp8TensorAg) -> torch.Tensor:
            nonlocal dtype
            if dtype is None:
                dtype = t._tensor.dtype
            else:
                assert t._tensor.dtype == dtype
            return t._tensor

        unwrapped_args, unwrapped_kwargs = pytree.tree_map_only(
            cls,
            unwrap,
            (args, kwargs or {}),
        )
        output = func(*unwrapped_args, **unwrapped_kwargs)
        if func not in _PRESERVE_WRAPPER_OPS:
            return output
        return pytree.tree_map_only(torch.Tensor, cls, output)

    def __tensor_flatten__(self):
        # 告诉 PyTorch subclass 序列化/重建机制：唯一的 tensor backing 是 `_tensor`。
        return ["_tensor"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        del metadata, outer_size, outer_stride
        return Fp8TensorAg(inner_tensors["_tensor"])

    def fsdp_pre_all_gather(
        self,
        mesh,
        orig_size,
        contiguous_orig_stride,
        module,
        mp_policy,
    ):
        # torchao FSDP 路径是 tensorwise scale（metadata）；block scale 各 rank 不同，
        # 必须把 scale 也放进 all-gather inputs（同 TE MXFP8）。
        # https://github.com/pytorch/ao/blob/main/torchao/float8/fsdp_utils.py#L228-L251
        del contiguous_orig_stride, mp_policy
        mesh_size = mesh.size()
        chunk_outer = (orig_size[0] + mesh_size - 1) // mesh_size
        rank = mesh.get_local_rank()
        expected_local_outer = min(
            chunk_outer,
            max(orig_size[0] - rank * chunk_outer, 0),
        )
        local = self._tensor.detach()
        expected_local_shape = (expected_local_outer, ) + tuple(orig_size[1:])
        if local.shape != expected_local_shape:
            raise ValueError(
                f"FP8 all-gather local shape {tuple(local.shape)} "
                f"!= expected {expected_local_shape}"
            )

        # NOTE 这里 implicitly 依赖一个实现细节：fsdp2 是在 dim 0 chunk 的。
        # 所以和 128x128 能对上，而且有 shape 校验。
        bm, bn = _FP8_BLOCK_SIZE
        assert orig_size[-2] % bm == 0 and orig_size[-1] % bn == 0, (
            f"global shape {tuple(orig_size)} not divisible by block {_FP8_BLOCK_SIZE}"
        )

        # 当前实现有不足：如果 dp size 很大，会导致 expert 不够分，最终大量 allgather 无用 tensor。
        padded_local_shape = (chunk_outer, ) + tuple(orig_size[1:])
        if local.shape == padded_local_shape:
            padded_local = local
        else:
            padded_local = local.new_zeros(padded_local_shape)
            if local.numel() > 0:
                padded_local[:local.shape[0]].copy_(local)

        # TODO 这里最好是全部 quant 都改成用 fuse 版本，否则会有一致性问题
        fp8_data, scale = quant_fp8_e4m3_scale_e8m0(
            padded_local,
            block_size=_FP8_BLOCK_SIZE,
        )

        # NCCL 对 e8m0 支持不确定，按 uint8 通信；post 再 view 回去。
        payload = fp8_data.view(torch.uint8)
        scale_u8 = scale.view(torch.uint8).contiguous()
        is_uneven = orig_size[0] % mesh_size != 0
        if is_uneven:
            scale_transport = torch.zeros_like(payload)
            scale_transport.view(-1)[:scale_u8.numel()].copy_(scale_u8.view(-1))
        else:
            scale_transport = scale_u8

        stats = module._fp8_all_gather_stats
        stats["pre_calls"] += 1
        stats["payload_dtype"] = str(payload.dtype)
        stats["fp8_bytes"] = payload.numel() * payload.element_size()
        stats["scale_bytes"] = scale_transport.numel() * scale_transport.element_size()
        stats["payload_bytes"] = stats["fp8_bytes"] + stats["scale_bytes"]
        stats["high_precision_bytes"] = padded_local.numel() * padded_local.element_size()
        metadata = {
            "stats": stats,
            "is_uneven": is_uneven,
            "mesh_size": mesh_size,
            "orig_outer": orig_size[0],
            "chunk_outer": chunk_outer,
            "scale_tail_shape": tuple(scale_u8.shape[1:]),
            "scale_numel_per_rank": scale_u8.numel(),
        }
        return (payload, scale_transport), metadata

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
    ):
        data_transport, scale_transport = all_gather_outputs
        stats = metadata["stats"]
        if metadata["is_uneven"]:
            orig_outer = metadata["orig_outer"]
            data = data_transport[:orig_outer]
            scale_u8 = (
                scale_transport.reshape(metadata["mesh_size"],
                                        -1)[:, :metadata["scale_numel_per_rank"]].reshape(
                                            metadata["mesh_size"] * metadata["chunk_outer"],
                                            *metadata["scale_tail_shape"],
                                        )[:orig_outer]
            )
        else:
            data = data_transport
            scale_u8 = scale_transport
        scale = scale_u8.view(torch.float8_e8m0fnu)
        stats["compute_dtype"] = str(param_dtype)
        if out is not None:
            assert isinstance(out, Fp8TensorTrain)
            out._data = data
            out._scale = scale
            stats["post_reuse_calls"] += 1
        else:
            stats["post_new_calls"] += 1
            tensor = Fp8TensorTrain(data, scale, param_dtype)
            return tensor, (data_transport, scale_transport)


class Fp8TensorTrain(torch.Tensor):
    """FSDP2 unshard 后的完整 weight：逻辑 dtype BF16，backing 是 FP8+E8M0 bytes。"""
    # https://github.com/pytorch/ao/blob/main/torchao/float8/float8_training_tensor.py#L242

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype,
    ) -> Fp8TensorTrain:
        del scale
        # 对 autograd/FSDP 宣称 dtype=param_dtype；真正 storage 仍是 uint8 `data`。
        return torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=dtype,
            layout=data.layout,
            device=data.device,
            pin_memory=data.is_pinned(),
            requires_grad=False,
        )

    def __init__(
        self,
        data: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype,
    ) -> None:
        self._data = data
        self._scale = scale
        self._logical_dtype = dtype

    @classmethod
    def __torch_dispatch__(
        cls,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        kwargs = kwargs or {}
        tensor = args[0]
        # 这里只实现 FSDP2 生命周期需要的结构 op；grouped GEMM 显式消费
        # `_data` / `_scale`。
        if func == torch.ops.aten.detach.default:
            return cls(
                tensor._data.detach(),
                tensor._scale,
                tensor._logical_dtype,
            )
        if func in {
            torch.ops.aten.as_strided.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.slice.Tensor,
            torch.ops.aten.view.default,
        }:
            data = func(tensor._data, *args[1:], **kwargs)
            return cls(data, tensor._scale, tensor._logical_dtype)
        raise NotImplementedError(f"{cls.__name__} does not implement {func}")

    def __tensor_flatten__(self):
        return ["_data", "_scale"], {"dtype": self._logical_dtype}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        del outer_size, outer_stride
        return Fp8TensorTrain(
            inner_tensors["_data"],
            inner_tensors["_scale"],
            metadata["dtype"],
        )
