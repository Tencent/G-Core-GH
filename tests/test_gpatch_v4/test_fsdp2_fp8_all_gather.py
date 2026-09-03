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
4. gpatch_v4/models/deepseek_v4/fp8.py (MyTeGroupedLinearFp8)
"""

from __future__ import annotations

import os
import socket
from typing import Any

import pytest
import ray
import torch
import torch.distributed as dist
from ray.util.placement_group import remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from gpatch_v4.kernel.quantize.eager_quant_kernels import (
    quant_fp8_e4m3_scale_e8m0,
)
from gpatch_v4.models.deepseek_v4.fp8 import MyTeGroupedLinearFp8
from gpatch_v4.models.deepseek_v4.fp8_tensor import (
    _FP8_BLOCK_SIZE,
    Fp8TensorAg,
    Fp8TensorTrain,
    _new_hook_stats,
)
from gpatch_v4.orches.placement_group import _create_placement_group
from gpatch_v4.training_backend.fsdp2_backend.swap import offload_model, onload_model


_WORLD_SIZE = 8
_NUM_GEMMS = 8
_MAX_UNEVEN_WORLD_SIZE = 4


class _ToyFp8GroupedLinear(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        m_splits: list[int],
    ) -> None:
        super().__init__()
        assert weight.ndim == 3
        self.num_gemms = weight.shape[0]
        assert len(m_splits) == self.num_gemms
        self.weight = nn.Parameter(Fp8TensorAg(weight))
        self.grouped = MyTeGroupedLinearFp8(num_gemms=self.num_gemms)
        self.m_splits = m_splits

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # FSDP2 forward pre-hook 已把 module.weight 从 sharded wrapper 临时替换为
        # post-hook 构造的完整 compute wrapper。
        assert isinstance(self.weight, Fp8TensorTrain)
        assert self.weight.dtype == torch.bfloat16
        return self.grouped(
            inputs,
            self.weight,
            self.m_splits,
        )


class _ToyUnevenParam(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        del inputs
        return self.weight.square().sum()


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

    def run_uneven(
        self,
        rank: int,
        world_size: int,
        num_experts: int,
        master_addr: str,
        master_port: int,
    ) -> dict[str, Any]:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        torch.cuda.set_device(0)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        try:
            device = torch.device("cuda:0")
            weight = torch.randn(num_experts, 128, 128, device=device)
            model = _ToyUnevenParam(weight)
            mesh = init_device_mesh("cuda", (world_size,))
            fully_shard(model, mesh=mesh, reshard_after_forward=True)

            native_local_shape = tuple(model.weight.to_local().shape)
            loss = model(torch.empty(0, device=device))
            loss.backward()
            native_grad_shape = tuple(model.weight.grad.to_local().shape)

            m_splits = [16] * num_experts
            fp8_weight = torch.randn(num_experts, 128, 128, device=device)
            inputs = torch.randn(
                sum(m_splits),
                128,
                device=device,
                dtype=torch.bfloat16,
            )
            fp8_model = _ToyFp8GroupedLinear(fp8_weight, m_splits)
            fp8_model._fp8_all_gather_stats = _new_hook_stats()
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=False,
            )
            fully_shard(
                fp8_model,
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=True,
            )
            fp8_local_weight = fp8_model.weight.to_local()
            fp8_local_shape = tuple(fp8_local_weight.shape)
            fp8_local_is_wrapper = isinstance(fp8_local_weight, Fp8TensorAg)
            fp8_local_master_dtype = str(fp8_local_weight._tensor.dtype)
            optimizer = torch.optim.AdamW(fp8_model.parameters(), lr=0.01, foreach=True)
            output = fp8_model(inputs)
            output.square().mean().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            second_output = fp8_model(inputs)
            second_output.square().mean().backward()
            fp8_local_grad = fp8_model.weight.grad.to_local()
            fp8_grad_shape = tuple(fp8_local_grad.shape)
            return {
                "rank": rank,
                "native_local_shape": native_local_shape,
                "native_grad_shape": native_grad_shape,
                "native_loss_finite": bool(torch.isfinite(loss)),
                "fp8_local_shape": fp8_local_shape,
                "fp8_grad_shape": fp8_grad_shape,
                "fp8_grad_dtype": str(fp8_local_grad.dtype),
                "fp8_local_is_wrapper": fp8_local_is_wrapper,
                "fp8_local_master_dtype": fp8_local_master_dtype,
                "fp8_output_finite": bool(
                    torch.isfinite(output).all()
                    and torch.isfinite(second_output).all()
                ),
                "fp8_bytes": fp8_model._fp8_all_gather_stats["fp8_bytes"],
                "scale_bytes": fp8_model._fp8_all_gather_stats["scale_bytes"],
                "post_reuse_calls": fp8_model._fp8_all_gather_stats["post_reuse_calls"],
            }
        finally:
            dist.destroy_process_group()

    def _run_model(self, rank: int, world_size: int) -> dict[str, Any]:
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        device = torch.device("cuda:0")
        assert world_size == _NUM_GEMMS
        bm, bn = _FP8_BLOCK_SIZE
        # stacked experts ``[E, N, K]``；FSDP 沿 dim0 切，每 rank 常驻一个 expert 的 FP32。
        weight = torch.randn(_NUM_GEMMS, bm, bn, device=device, dtype=torch.float32)
        # 每 expert 16 token；TE FP8 path 会 pad 到 16 对齐。
        m_splits = [16] * _NUM_GEMMS
        inputs = torch.randn(sum(m_splits), bn, device=device, dtype=torch.bfloat16)
        model = _ToyFp8GroupedLinear(weight.clone(), m_splits)
        model._fp8_all_gather_stats = _new_hook_stats()
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

        local_weight = model.weight.to_local()
        assert isinstance(local_weight, Fp8TensorAg)
        assert local_weight._tensor.dtype == torch.float32
        local_weight_before = local_weight._tensor.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=True)

        output = model(inputs)

        # 对拍：同一套 E8M0 quant + 同一条 TE grouped GEMM 路径（不经 FSDP）。
        q_ref, s_ref = quant_fp8_e4m3_scale_e8m0(weight, block_size=_FP8_BLOCK_SIZE)
        ref_compute = Fp8TensorTrain(
            q_ref.view(torch.uint8),
            s_ref,
            torch.bfloat16,
        )
        ref_compute.requires_grad_(True)
        ref_grouped = MyTeGroupedLinearFp8(num_gemms=_NUM_GEMMS)
        reference_output = ref_grouped(
            inputs,
            ref_compute,
            m_splits,
        )
        torch.testing.assert_close(output, reference_output, rtol=0.0, atol=0.0)

        loss = output.square().mean()
        reference_loss = reference_output.square().mean()
        loss.backward()
        reference_loss.backward()

        local_grad = model.weight.grad.to_local()
        # FSDP 沿 dim0 切 3D weight 后 local 仍带 size-1 的 leading dim。
        reference_local_grad = ref_compute.grad[rank].float().unsqueeze(0)
        assert local_grad.dtype == torch.float32
        assert local_grad.shape == reference_local_grad.shape, (
            local_grad.shape,
            reference_local_grad.shape,
        )
        torch.testing.assert_close(
            local_grad,
            reference_local_grad,
            rtol=0.0,
            atol=0.0,
        )
        grad_max_diff = (local_grad - reference_local_grad).abs().max().item()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        output = model(inputs)
        output.square().mean().backward()
        optimizer_no_foreach = torch.optim.AdamW(
            model.parameters(),
            lr=0.01,
            foreach=False,
        )
        optimizer_no_foreach.step()
        optimizer_no_foreach.zero_grad(set_to_none=True)

        offload_model(model, do_clear_memory=False)
        offloaded_weight = model.weight.to_local()
        assert isinstance(offloaded_weight, Fp8TensorAg)
        assert offloaded_weight._tensor.device.type == "cpu"
        onload_model(model, do_clear_memory=False)
        onloaded_weight = model.weight.to_local()
        assert isinstance(onloaded_weight, Fp8TensorAg)
        assert onloaded_weight._tensor.device.type == "cuda"

        second_output = model(inputs)
        assert torch.isfinite(second_output).all()

        local_weight_after = model.weight.to_local()
        assert isinstance(local_weight_after, Fp8TensorAg)
        updated = not torch.equal(local_weight_before, local_weight_after._tensor)
        return {
            "rank": rank,
            "updated": updated,
            "loss_finite": bool(torch.isfinite(loss)),
            "master_dtype": str(local_weight_after._tensor.dtype),
            "grad_dtype": str(local_grad.dtype),
            "grad_max_diff": grad_max_diff,
            "offload_onload_ok": True,
            "optimizer_modes_ok": True,
            **model._fp8_all_gather_stats,
        }

    def shutdown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_fsdp2_fp8_supports_uneven_and_empty_dim0_shards() -> None:
    ray.init(address="auto")
    try:
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < _MAX_UNEVEN_WORLD_SIZE:
            pytest.skip(f"requires {_MAX_UNEVEN_WORLD_SIZE} GPUs, found {total_gpus}")

        placement_group, bundle_indices = _create_placement_group(_MAX_UNEVEN_WORLD_SIZE)
        try:
            workers = [
                _Fp8FsdpWorker.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=placement_group,
                        placement_group_bundle_index=bundle_indices[rank],
                    )
                ).remote()
                for rank in range(_MAX_UNEVEN_WORLD_SIZE)
            ]
            for num_experts, world_size, expected_shapes in (
                (3, 2, [(2, 128, 128), (1, 128, 128)]),
                (1, 2, [(1, 128, 128), (0, 128, 128)]),
                (2, 4, [(1, 128, 128), (1, 128, 128), (0, 128, 128), (0, 128, 128)]),
            ):
                master_addr, master_port = ray.get(workers[0].master_addr_and_port.remote())
                results = ray.get(
                    [
                        worker.run_uneven.remote(
                            rank,
                            world_size,
                            num_experts,
                            master_addr,
                            master_port,
                        )
                        for rank, worker in enumerate(workers[:world_size])
                    ],
                    timeout=300,
                )
                assert [result["native_local_shape"] for result in results] == expected_shapes
                assert [result["native_grad_shape"] for result in results] == expected_shapes
                assert [result["fp8_local_shape"] for result in results] == expected_shapes
                assert [result["fp8_grad_shape"] for result in results] == expected_shapes
                assert all(result["native_loss_finite"] for result in results)
                assert all(result["fp8_output_finite"] for result in results)
                assert all(result["fp8_local_is_wrapper"] for result in results)
                assert all(
                    result["fp8_local_master_dtype"] == "torch.float32"
                    for result in results
                )
                assert all(
                    result["fp8_grad_dtype"] == "torch.float32"
                    for result in results
                )
                assert all(result["post_reuse_calls"] >= 1 for result in results)
                assert all(
                    result["scale_bytes"] == result["fp8_bytes"]
                    for result in results
                )
        finally:
            remove_placement_group(placement_group)
    finally:
        ray.shutdown()


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
                # FP8 bytes = master/4；scale 一并走 FSDP all-gather。
                assert result["updated"], result
                assert result["loss_finite"], result
                assert result["master_dtype"] == "torch.float32", result
                assert result["compute_dtype"] == "torch.bfloat16", result
                assert result["grad_dtype"] == "torch.float32", result
                assert result["grad_max_diff"] < 1e-5, result
                assert result["offload_onload_ok"], result
                assert result["optimizer_modes_ok"], result
                assert result["payload_dtype"] == "torch.uint8", result
                assert result["fp8_bytes"] * 4 == result["high_precision_bytes"], result
                assert result["scale_bytes"] > 0, result
                assert (
                    result["payload_bytes"]
                    == result["fp8_bytes"] + result["scale_bytes"]
                ), result
                assert result["pre_calls"] >= 2, result
                assert result["post_new_calls"] == 1, result
                assert result["post_reuse_calls"] >= 1, result
                print(
                    f"rank={result['rank']} "
                    f"fsdp_payload={result['payload_bytes']} "
                    f"fp8={result['fp8_bytes']} scale={result['scale_bytes']}",
                    flush=True,
                )
        finally:
            print(f'', flush=True)
            print(f'', flush=True)
            print(f'', flush=True)
            remove_placement_group(placement_group)
    finally:
        ray.shutdown()
