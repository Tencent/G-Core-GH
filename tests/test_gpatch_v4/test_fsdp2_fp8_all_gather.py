# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""不依赖 TE/torchao 的 FSDP2 FP8 parameter all-gather 最小示例。

数据流：

1. ``_Float8AllGatherTensor`` 包住完整 FP32 Parameter；
2. ``fully_shard`` 将它切成每 rank 常驻的 FP32 shard；
3. FSDP2 unshard 前调用 ``fsdp_pre_all_gather``，将本地 shard 量化成 FP8 bytes；
4. FSDP2 gather bytes，随后由 ``fsdp_post_all_gather`` 构造完整参数的逻辑视图；
5. demo forward 反量化成 BF16 后执行 linear，backward 返回 BF16 full weight gradient；
6. FSDP2 reduce-scatter gradient，SGD 更新常驻的 FP32 shard。

本测试只演示 FP8 参数通信扩展，不实现 FP8 Tensor Core GEMM。

REFs:
1. https://github.com/pytorch/ao/blob/main/torchao/float8/fsdp_utils.py
2. https://github.com/pytorch/ao/blob/main/torchao/float8/float8_training_tensor.py#L242
3. TransformerEngine/transformer_engine/pytorch/tensor/float8_tensor.py
"""

from __future__ import annotations

import os
import socket
from typing import Any

import pytest
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils._pytree as pytree
from ray.util.placement_group import remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from gpatch_v4.orches.placement_group import _create_placement_group


_WORLD_SIZE = 8
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max
# FSDP2 在切分、padding 和重建 Parameter 时会调用这些 ATen op；返回值必须继续
# 保持 wrapper subclass，否则 local shard 会退化成普通 Tensor 并丢失 gather hooks。
_PRESERVE_WRAPPER_OPS = {
    torch.ops.aten._pin_memory.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_empty.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.split.Tensor,
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
        high_precision_bytes=0,
    )


class _Float8AllGatherTensor(torch.Tensor):
    """逻辑及 backing storage 都是 FP32、但 unshard 时用 FP8 通信的 Parameter wrapper。"""

    @staticmethod
    def __new__(cls, tensor: torch.Tensor) -> _Float8AllGatherTensor:
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
        kwargs = kwargs or {}
        # nn.Parameter 构造时会 detach；必须保持 subclass 才能让 fully_shard 看到 hooks。
        if func == torch.ops.aten.detach.default:
            return cls(args[0]._tensor.detach())
        # copy_ 和 SGD 的 add_ 是原地更新，返回原 wrapper 以保持 Parameter identity。
        if func == torch.ops.aten.copy_.default:
            dst, src = args[:2]
            src = src._tensor if isinstance(src, cls) else src
            dst._tensor.copy_(src, *args[2:], **kwargs)
            return dst
        if func == torch.ops.aten.add_.Tensor:
            dst, src = args[:2]
            src = src._tensor if isinstance(src, cls) else src
            dst._tensor.add_(src, *args[2:], **kwargs)
            return dst

        # 其他 op 先作用于真实 FP32 tensor。FSDP2 的结构变换需要重新包回 subclass，
        # 普通数值 op 则直接返回普通 Tensor。
        unwrapped_args, unwrapped_kwargs = pytree.tree_map_only(
            cls,
            lambda tensor: tensor._tensor,
            (args, kwargs),
        )
        output = func(*unwrapped_args, **unwrapped_kwargs)
        if func not in _PRESERVE_WRAPPER_OPS:
            return output
        return pytree.tree_map_only(torch.Tensor, cls, output)

    def __tensor_flatten__(self):
        # 告诉 PyTorch subclass 序列化/重建机制：唯一的 tensor backing 是 `_tensor`。
        assert False
        return ["_tensor"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        assert False
        del metadata, outer_size, outer_stride
        return _Float8AllGatherTensor(inner_tensors["_tensor"])

    def fsdp_pre_all_gather(
        self,
        mesh,
        orig_size,
        contiguous_orig_stride,
        module,
        mp_policy,
    ):
        del orig_size, contiguous_orig_stride, mp_policy
        # Tensorwise scaling 必须使用完整参数的 amax。此处各 rank 先计算 local
        # shard amax，再以 MAX all-reduce 得到每个 rank 相同的全局 scale。
        amax = self._tensor.detach().abs().max().float()
        dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=mesh.get_group())
        scale = _FP8_MAX / amax.clamp_min(torch.finfo(torch.float32).tiny)
        fp8_data = (self._tensor.float() * scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
        # E4M3 与 uint8 都是一字节。用 uint8 view 明确让通信层只搬运原始 FP8 bytes。
        payload = fp8_data.view(torch.uint8)

        stats = module._fp8_all_gather_stats
        stats["pre_calls"] += 1
        stats["payload_dtype"] = str(payload.dtype)
        stats["payload_bytes"] = payload.numel() * payload.element_size()
        stats["high_precision_bytes"] = self._tensor.numel() * self._tensor.element_size()
        # 第一个 tuple 中的 Tensor 会被 FSDP2 all-gather；metadata 不通信，只在本
        # rank 传给 post-hook。scale 已经 all-reduce，所以各 rank 的值一致。
        return (payload,), (scale, stats)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = None,
    ):
        # `data` 已是 FSDP2 拼好的完整参数 bytes，不再是 local shard。
        (data,) = all_gather_outputs
        scale, stats = metadata
        stats["compute_dtype"] = str(param_dtype)
        if out is not None:
            # 首次 unshard 后 FSDP2 会保留 wrapper；后续只替换新 gather 的 backing，
            # 避免每轮重新创建 Parameter 对象。
            assert isinstance(out, _Float8ComputeTensor)
            out._data = data
            out._scale = scale
            stats["post_reuse_calls"] += 1
            return

        # 第一次 unshard 尚无目标对象，由 hook 创建完整参数的 compute wrapper。
        stats["post_new_calls"] += 1
        tensor = _Float8ComputeTensor(data, scale, param_dtype)
        return tensor, (data,)


class _Float8ComputeTensor(torch.Tensor):
    """FSDP2 unshard 后的完整 weight：逻辑 dtype BF16，backing 是 FP8 uint8 bytes。"""

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        scale: torch.Tensor,
        dtype: torch.dtype,
    ) -> _Float8ComputeTensor:
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
        # 这里只实现 FSDP2 生命周期需要的结构 op；真正 linear 前由
        # `_DequantizeFloat8` 显式消费 `_data` 和 `_scale`。
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
        return _Float8ComputeTensor(
            inner_tensors["_data"],
            inner_tensors["_scale"],
            metadata["dtype"],
        )


class _DequantizeFloat8(torch.autograd.Function):
    """将 gather 后的 FP8 weight 反量化；backward 用 STE 返回 param-dtype gradient。"""

    @staticmethod
    def forward(ctx, weight: _Float8ComputeTensor) -> torch.Tensor:
        del ctx
        # pre-hook 使用 q = weight * scale，所以反量化是 q / scale。
        fp8_data = weight._data.view(_FP8_DTYPE)
        return (fp8_data.float() / weight._scale).to(weight.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        del ctx
        return grad_output


class _ToyFp8Linear(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(_Float8AllGatherTensor(weight))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # FSDP2 forward pre-hook 已把 module.weight 从 sharded wrapper 临时替换为
        # post-hook 构造的完整 compute wrapper。
        assert isinstance(self.weight, _Float8ComputeTensor)
        assert self.weight.dtype == torch.bfloat16
        weight = _DequantizeFloat8.apply(self.weight)
        return F.linear(inputs, weight)


@ray.remote(num_gpus=1)
class _Fp8FsdpWorker:
    """一个 Ray actor 对应一个 FSDP rank 和一张可见 GPU。"""

    def master_addr_and_port(self) -> tuple[str, int]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return socket.gethostbyname(socket.gethostname()), port

    def run(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ) -> dict[str, Any]:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["NCCL_CUMEM_ENABLE"] = "0"
        # Ray 已通过 CUDA_VISIBLE_DEVICES 隔离物理卡，所以 actor 内统一使用 cuda:0。
        torch.cuda.set_device(0)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        try:
            return self._run_model(rank, world_size)
        finally:
            dist.destroy_process_group()

    def _run_model(self, rank: int, world_size: int) -> dict[str, Any]:
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        device = torch.device("cuda:0")
        # 各 rank 从同一完整 FP32 weight 开始，fully_shard 后只常驻本地 FP32 shard。
        weight = torch.randn(256, 128, device=device, dtype=torch.float32)
        inputs = torch.randn(16, 128, device=device, dtype=torch.bfloat16)
        model = _ToyFp8Linear(weight.clone())
        model._fp8_all_gather_stats = _new_hook_stats()
        # DeviceMesh 将 global rank 映射到 FSDP shard group；这里只有一维纯 FSDP。
        mesh = init_device_mesh("cuda", (world_size,))
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        )
        fully_shard(
            model,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
        )

        # 保存 optimizer step 前的高精度 shard，确认 FP8 只用于通信而非参数常驻。
        local_weight = model.weight.to_local()
        assert isinstance(local_weight, _Float8AllGatherTensor)
        assert local_weight._tensor.dtype == torch.float32
        local_weight_before = local_weight._tensor.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # 对拍完整 weight 的显式 FP8 QDQ，验证各 shard 量化后 gather 的重建结果。
        output = model(inputs)
        scale = _FP8_MAX / weight.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
        reference_weight = (
            (weight * scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE).float() / scale
        ).to(torch.bfloat16).requires_grad_(True)
        reference_output = F.linear(inputs, reference_weight)
        torch.testing.assert_close(output, reference_output, rtol=0.0, atol=0.0)

        # reference_weight 是 QDQ 后的 BF16 leaf；其梯度即 STE 的 full weight grad。
        loss = output.square().mean()
        reference_loss = reference_output.square().mean()
        loss.backward()
        reference_loss.backward()

        # FSDP2 将 full grad reduce-scatter 回 FP32 master shard；各 rank 输入相同，
        # 默认 DP 平均后的 local grad 应与 reference full grad 的对应 chunk 一致。
        local_grad = model.weight.grad.to_local()
        reference_local_grad = reference_weight.grad.chunk(world_size, dim=0)[rank].float()
        assert local_grad.dtype == torch.float32
        torch.testing.assert_close(
            local_grad,
            reference_local_grad,
            rtol=1e-6,
            atol=1e-7,
        )
        grad_max_diff = (local_grad - reference_local_grad).abs().max().item()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # 再跑一次 forward，验证更新后的 FP32 shard 可重新量化并完成 FP8 gather。
        second_output = model(inputs)
        assert torch.isfinite(second_output).all()

        local_weight_after = model.weight.to_local()
        assert isinstance(local_weight_after, _Float8AllGatherTensor)
        updated = not torch.equal(local_weight_before, local_weight_after._tensor)
        return {
            "rank": rank,
            "updated": updated,
            "loss_finite": bool(torch.isfinite(loss)),
            "master_dtype": str(local_weight_after._tensor.dtype),
            "grad_dtype": str(local_grad.dtype),
            "grad_max_diff": grad_max_diff,
            **model._fp8_all_gather_stats,
        }

    def shutdown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_fsdp2_fp8_parameter_all_gather() -> None:
    ray.init(address="auto")
    try:
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < _WORLD_SIZE:
            pytest.skip(f"requires {_WORLD_SIZE} GPUs, found {total_gpus}")

        # 固定并按 node IP/GPU ID 排序 actor placement，避免 Ray 自由调度导致
        # global rank 与物理 topology 顺序不稳定。
        placement_group, bundle_indices = _create_placement_group(_WORLD_SIZE)
        try:
            workers = [
                _Fp8FsdpWorker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=placement_group,
                        placement_group_bundle_index=bundle_indices[rank],
                    )
                ).remote()
                for rank in range(_WORLD_SIZE)
            ]
            master_addr, master_port = ray.get(workers[0].master_addr_and_port.remote())
            results = ray.get(
                [
                    worker.run.remote(
                        rank,
                        _WORLD_SIZE,
                        master_addr,
                        master_port,
                    )
                    for rank, worker in enumerate(workers)
                ],
                timeout=300,
            )
            for result in results:
                # FP32 一元素 4 bytes，FP8/uint8 一元素 1 byte，payload 应缩小到 1/4。
                assert result["updated"], result
                assert result["loss_finite"], result
                assert result["master_dtype"] == "torch.float32", result
                assert result["compute_dtype"] == "torch.bfloat16", result
                assert result["grad_dtype"] == "torch.float32", result
                assert result["grad_max_diff"] < 1e-6, result
                assert result["payload_dtype"] == "torch.uint8", result
                assert result["payload_bytes"] * 4 == result["high_precision_bytes"], result
                # forward、backward 和第二次 forward 至少触发两次 unshard；仅首次
                # 创建 compute wrapper，后续 all-gather 应走 `out` 复用路径。
                assert result["pre_calls"] >= 2, result
                assert result["post_new_calls"] == 1, result
                assert result["post_reuse_calls"] >= 1, result
        finally:
            print(f'', flush=True)
            print(f'', flush=True)
            print(f'', flush=True)
            remove_placement_group(placement_group)
    finally:
        ray.shutdown()
