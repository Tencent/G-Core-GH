"""Context-driven transports for sampler weight updates."""

from __future__ import annotations

import asyncio
import os
import re
import socket
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import ray
import torch
import torch.distributed as dist

from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.core.parallel_state import cpu_barrier
from gpatch_v4.utils import clear_memory, log, perf_time
from gpatch_v4.utils.tensor_ipc import disable_expandable_segments

try:
    from gpatch_v4.orches.flashinfer_cudart_fix import patch_ctypes_for_cudart_stub

    with patch_ctypes_for_cudart_stub():
        from sglang.srt.utils import MultiprocessingSerializer as SglSerializer
        from sglang.srt.weight_sync.tensor_bucket import (
            FlattenedTensorBucket as SglFlatTensorBucket,
        )
except Exception:
    SglSerializer = None
    SglFlatTensorBucket = None

_ROUTED_EXPERT_WEIGHT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.weight$")
_TRACE_EXPERT_IDS = frozenset((0, 1, 2, 255))


def _dsv4_update_trace_enabled() -> bool:
    return os.getenv("GCORE_DSV4_UPDATE_TRACE", "0") == "1"


def _dsv4_trace_tensor_signature(tensor: torch.Tensor) -> tuple[int, ...]:
    """Small raw-byte fingerprint without copying an entire weight to CPU."""
    raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    if raw.numel() == 0:
        return ()
    indices = torch.tensor(
        (0, raw.numel() // 3, (2 * raw.numel()) // 3, raw.numel() - 1),
        device=raw.device,
        dtype=torch.long,
    )
    return tuple(int(x) for x in raw.index_select(0, indices).cpu().tolist())


def _dsv4_trace_bucket_probes(named_tensors: Any) -> tuple[tuple[Any, ...], ...]:
    """Return selected routed-expert entries for cross-sender IPC auditing."""
    probes = []
    for name, tensor in named_tensors:
        match = _ROUTED_EXPERT_WEIGHT_RE.match(name)
        if match is None or int(match.group(2)) not in _TRACE_EXPERT_IDS:
            continue
        probes.append(
            (
                name,
                tuple(tensor.shape),
                str(tensor.dtype),
                _dsv4_trace_tensor_signature(tensor),
            )
        )
    return tuple(probes)


@dataclass
class UpdateWeightContext:
    """Values and transport state required by a weight-update factory.

    Deliberately contains no ``SamplerClient``. Mutable transport fields
    (``ipc_*``, ``dist_weight_group*``) are owned by the client mixin and
    synced onto this context before factory calls.
    """

    infer_backend: str
    update_weight_max_size_bytes: int
    update_weight_use_bucketed_ipc: bool
    rpc_client_lst: list[Any]
    svr_cluster_num_per_sampler: list[int]
    ipc_gather_dst_rank: int | None = None
    ipc_gather_group: Any = None
    ipc_target: int | None = None
    dist_weight_group: Any = None
    dist_weight_group_name: str | None = None
    sampler_engine_gpu_counts: list[int] | None = None
    moe_deepgemm: bool = False
    sglang_export_fp4_qdq: bool = False
    placement_type: str = "disaggregated"
    wake_up: Any = None
    sleep: Any = None


class UpdateWeightFactory(ABC):
    """Select the IPC and distributed update implementation for one backend."""
    def __init__(self, context: UpdateWeightContext):
        self.context = context

    def get_target_endpoint(self, sampler_idx: int, ep_idx: int, sample_idx: int = None):
        return self.context.rpc_client_lst[sampler_idx].get_target_endpoint(
            sample_idx=sample_idx, ep_idx=ep_idx
        )

    def num_clusters(self, sampler_idx: int) -> int:
        return self.context.svr_cluster_num_per_sampler[sampler_idx]

    @staticmethod
    def _ensure_sglang_weight_update_ok(results: Any, *, context: str) -> None:
        if results is None:
            return
        if not isinstance(results, (list, tuple)):
            results = [results]
        for i, result in enumerate(results):
            ret = result.get("ret", result) if isinstance(result, dict) else result
            if isinstance(ret, (list, tuple)) and len(ret) >= 1:
                inner = ret[0] if len(ret) == 1 else ret
                if isinstance(inner,
                              (list, tuple
                              )) and len(inner) >= 1 and isinstance(inner[0], (bool, type(None))):
                    success, message = inner[0], (inner[1] if len(inner) > 1 else "")
                elif isinstance(inner, (bool, type(None))):
                    success, message = inner, ""
                elif hasattr(inner, "success"):
                    success, message = inner.success, getattr(inner, "message", "") or getattr(
                        inner, "error_message", ""
                    )
                else:
                    continue
            elif hasattr(ret, "success"):
                success = ret.success
                message = getattr(ret, "message", "") or getattr(ret, "error_message", "")
            elif isinstance(ret, bool):
                success, message = ret, ""
            else:
                continue
            if success is False:
                raise RuntimeError(
                    f"sglang weight update failed ({context}, endpoint={i}): {message}"
                )

    async def init_distributed_weight_group(self,
                                            group_name: str = "weight_update_group"
                                           ) -> tuple[Any, str]:
        """Create the backend weight-update group.

        Returns
        -------
        tuple
            ``(dist_weight_group, group_name)``. ``dist_weight_group`` is only
            set on trainer rank 0; other ranks get ``None``.
        """
        sampler_idx = 0
        backend = self.context.infer_backend
        if backend == "vllm" and self.context.placement_type != "disaggregated":
            assert self.context.wake_up is not None
            await self.context.wake_up(sampler_idx, tag_names=["weights"])
            cpu_barrier()
        num_clusters = self.num_clusters(sampler_idx)
        engine_gpu_count = self.context.sampler_engine_gpu_counts[sampler_idx]
        world_size = 1 + num_clusters * engine_gpu_count
        if dist.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
        else:
            master_address = None
            master_port = None
        meta = [master_address, master_port]
        dist.broadcast_object_list(meta, src=0)
        master_address, master_port = meta
        dist_weight_group = None
        if dist.get_rank() == 0:
            refs = []
            for ep_idx in range(num_clusters):
                req_data = {
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": 1 + ep_idx * engine_gpu_count,
                    "world_size": world_size,
                    "group_name": group_name,
                    "backend": "nccl",
                }
                refs.append(
                    self.get_target_endpoint(sampler_idx,
                                             ep_idx).init_weights_update_group.remote(req_data)
                )
            if backend == "vllm":
                from vllm.distributed.weight_transfer.nccl_engine import (
                    NCCLWeightTransferEngine,
                )
                dist_weight_group = NCCLWeightTransferEngine.trainer_init(
                    {
                        "master_address": master_address,
                        "master_port": master_port,
                        "world_size": world_size
                    }
                )
            else:
                from sglang.srt.utils import init_custom_process_group
                dist_weight_group = init_custom_process_group(
                    backend="nccl",
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )
            ray.get(refs)
        cpu_barrier()
        log(
            f"distributed weight group '{group_name}' initialized, "
            f"world_size={world_size} (1 train + {num_clusters}x{engine_gpu_count} sampler)",
            rank=0,
        )

        if backend == "vllm" and self.context.placement_type != "disaggregated":
            assert self.context.sleep is not None
            await self.context.sleep(sampler_idx)
            cpu_barrier()
        return dist_weight_group, group_name

    async def destroy_distributed_weight_group(self) -> tuple[Any, str | None]:
        """Destroy the backend weight-update group.

        Returns
        -------
        tuple
            Always ``(None, None)`` after a successful destroy (or no-op when
            no group name is present).
        """
        if self.context.dist_weight_group_name is None:
            return self.context.dist_weight_group, self.context.dist_weight_group_name
        assert self.context.infer_backend == "sglang", "only sglang backend is supported"
        sampler_idx = 0
        if dist.get_rank() == 0:
            refs = [
                self.get_target_endpoint(sampler_idx, ep_idx).destroy_weights_update_group.remote(
                    {"group_name": self.context.dist_weight_group_name}
                ) for ep_idx in range(self.num_clusters(sampler_idx))
            ]
            ray.get(refs)
            dist.destroy_process_group(self.context.dist_weight_group)
        cpu_barrier()
        return None, None

    @abstractmethod
    async def update_weights_by_ipc_handle(
        self, sampler_idx: int, model_engine: Any, replace_zeros: bool = False
    ) -> bool:
        """Update weights over the colocated IPC transport."""
        ...

    @abstractmethod
    def update_weights_by_distributed(
        self, sampler_idx: int, model_engine: Any, replace_zeros: bool = False
    ) -> bool:
        """Update weights over the distributed transport."""
        ...


class SglangUpdateWeightFactory(UpdateWeightFactory):
    """Standard SGLang weight-update transport."""
    async def update_weights_by_ipc_handle(
        self, sampler_idx: int, model_engine: Any, replace_zeros: bool = False
    ) -> bool:
        rank = dist.get_rank()
        participates_in_ipc = self.context.ipc_gather_group is not None

        update_weight_max_size_bytes = self.context.update_weight_max_size_bytes
        large_tensor_cleanup_threshold_bytes = 10 * update_weight_max_size_bytes
        weight_generator = model_engine.export_weights()

        async def async_update_weights():
            count_packed_bucket_num = 0
            with disable_expandable_segments(), perf_time(f"update weight total", rank=0):
                try:
                    converted_named_tensors_by_dtypes = {}
                    converted_buffer_size_by_dtypes = {}
                    for name, param in weight_generator:
                        is_gathered_tensor = bool(getattr(param, "is_gathered_tensor", False))
                        if participates_in_ipc:
                            if replace_zeros:
                                weight_tensor = torch.zeros_like(param)
                            elif is_gathered_tensor:
                                weight_tensor = param
                            else:
                                weight_tensor = param.detach().clone()
                            dtype = weight_tensor.dtype
                            tensor_bytes = weight_tensor.element_size() * weight_tensor.numel()
                        else:
                            weight_tensor = None
                            dtype = param.dtype
                            tensor_bytes = param.element_size() * param.numel()

                        if dtype not in converted_named_tensors_by_dtypes:
                            converted_named_tensors_by_dtypes[dtype] = []
                            converted_buffer_size_by_dtypes[dtype] = 0
                        if participates_in_ipc:
                            converted_named_tensors_by_dtypes[dtype].append((name, weight_tensor))
                        converted_buffer_size_by_dtypes[dtype] += tensor_bytes

                        if converted_buffer_size_by_dtypes[dtype] >= update_weight_max_size_bytes:
                            torch.cuda.synchronize()
                            named_tensors = converted_named_tensors_by_dtypes[dtype]
                            bucket_bytes = converted_buffer_size_by_dtypes[dtype]
                            if participates_in_ipc:
                                serialized_named_tensors = self.flattened_and_get_ipc_handle(
                                    named_tensors
                                )
                            else:
                                serialized_named_tensors = None

                            count_packed_bucket_num += 1
                            if participates_in_ipc and dist.get_rank(
                            ) == self.context.ipc_gather_dst_rank:
                                update_data = {
                                    "serialized_named_tensors": serialized_named_tensors,
                                    "load_format": "flattened_bucket",
                                }
                                resp = await self.update_co(sampler_idx, update_data)
                                log(f"update_weights response: {resp}", rank=0)
                                self._ensure_sglang_weight_update_ok(resp, context="ipc_handle")
                                if bucket_bytes >= large_tensor_cleanup_threshold_bytes:
                                    del update_data, resp

                            cpu_barrier()
                            converted_named_tensors_by_dtypes[dtype] = []
                            converted_buffer_size_by_dtypes[dtype] = 0
                            if bucket_bytes >= large_tensor_cleanup_threshold_bytes:
                                del named_tensors
                                del serialized_named_tensors
                                if weight_tensor is not None:
                                    del weight_tensor
                                del param
                                clear_memory()
                                torch.cuda.ipc_collect()

                    last_update_weight_co = []
                    for dtype, named_tensors in converted_named_tensors_by_dtypes.items():
                        if len(named_tensors) > 0:
                            torch.cuda.synchronize()
                            serialized_named_tensors = self.flattened_and_get_ipc_handle(
                                named_tensors
                            )

                            count_packed_bucket_num += 1
                            if dist.get_rank() == self.context.ipc_gather_dst_rank:
                                update_data = {
                                    "serialized_named_tensors": serialized_named_tensors,
                                    "load_format": "flattened_bucket",
                                }
                                last_update_weight_co.append(
                                    self.update_co(sampler_idx, update_data)
                                )

                    if dist.get_rank() == self.context.ipc_gather_dst_rank:
                        resps = await asyncio.gather(*last_update_weight_co)
                        log(f"update_weights response: {resps}", rank=0)
                        for resp in resps:
                            self._ensure_sglang_weight_update_ok(resp, context="ipc_handle")
                    cpu_barrier()
                    return count_packed_bucket_num
                except Exception as e:
                    log(f"update_weights error: {e}")
                    traceback.print_exc()
                    raise RuntimeError(f"update_weights error: {e}")

        count_packed_bucket_num = await async_update_weights()
        log(f"total packed bucket num: {count_packed_bucket_num}")
        clear_memory()
        return True

    def update_weights_by_distributed(
        self, sampler_idx: int, model_engine: Any, replace_zeros: bool = False
    ) -> bool:
        weight_generator = model_engine.export_weights()
        max_bucket_bytes = self.context.update_weight_max_size_bytes
        large_tensor_cleanup_threshold_bytes = 10 * max_bucket_bytes
        is_rank0 = dist.get_rank() == 0
        count_buckets = 0

        buffer = []
        buffer_size = 0

        for name, param in weight_generator:
            if is_rank0:
                if replace_zeros:
                    weight = torch.zeros_like(param, device="cuda")
                else:
                    weight = param.data
                    if not weight.is_cuda:
                        weight = weight.cuda()
                    weight = weight.clone()
                buffer.append((name, weight))

            buffer_size += param.element_size() * param.numel()

            if buffer_size >= max_bucket_bytes:
                bucket_bytes = buffer_size
                if is_rank0:
                    torch.cuda.synchronize()
                    self.broadcast_sglang_bucket(sampler_idx, buffer, flush_cache=False)
                    count_buckets += 1
                if bucket_bytes >= large_tensor_cleanup_threshold_bytes:
                    flush_refs = None
                    if is_rank0:
                        flush_refs = self.submit_cache_flush(sampler_idx, bucket_bytes)
                        del weight
                    del buffer
                    del param
                    clear_memory()
                    if is_rank0:
                        self.wait_cache_flush(flush_refs)
                buffer = []
                buffer_size = 0
                cpu_barrier()

        if buffer_size > 0:
            bucket_bytes = buffer_size
            if is_rank0:
                torch.cuda.synchronize()
                self.broadcast_sglang_bucket(sampler_idx, buffer, flush_cache=True)
                count_buckets += 1
            if bucket_bytes >= large_tensor_cleanup_threshold_bytes:
                flush_refs = None
                if is_rank0:
                    flush_refs = self.submit_cache_flush(sampler_idx, bucket_bytes)
                    del weight
                del buffer
                del param
                clear_memory()
                if is_rank0:
                    self.wait_cache_flush(flush_refs)
            cpu_barrier()

        log(f"distributed weight update done, {count_buckets} buckets broadcast", rank=0)
        return True

    async def update_co(self, sampler_idx, update_data):
        target_ep = self.get_target_endpoint(sampler_idx, self.context.ipc_target)
        return await asyncio.gather(
            self.context.rpc_client_lst[sampler_idx].call(target_ep, "update_weights", update_data)
        )

    def flattened_and_get_ipc_handle(self, named_tensors):
        assert SglFlatTensorBucket is not None and SglSerializer is not None, "sglang IPC is unavailable"
        bucket = SglFlatTensorBucket(named_tensors=named_tensors)
        data = {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata()
        }
        serialized = SglSerializer.serialize(
            data,
            output_str=True,
        )
        gathered = (
            [None] * dist.get_world_size(self.context.ipc_gather_group)
            if self.context.ipc_gather_dst_rank == dist.get_rank() else None
        )
        dist.gather_object(
            serialized,
            object_gather_list=gathered,
            dst=self.context.ipc_gather_dst_rank,
            group=self.context.ipc_gather_group
        )
        return gathered

    def submit_cache_flush(self, sampler_idx, bucket_bytes):
        log(
            f"flush sampler cache after large distributed weight bucket ({bucket_bytes / 1024**3:.3f} GiB)",
            rank=0
        )
        num_clusters = self.num_clusters(sampler_idx)
        return [
            self.get_target_endpoint(sampler_idx, ep_idx).flush_cache.remote({})
            for ep_idx in range(num_clusters)
        ]

    def wait_cache_flush(self, obj_refs):
        ray.get(obj_refs)
        log("sampler cache flushed after large distributed weight bucket", rank=0)

    def broadcast_sglang_bucket(self, sampler_idx, named_tensors, flush_cache=False):
        names = [name for name, _ in named_tensors]
        dtypes = [str(t.dtype).replace("torch.", "") for _, t in named_tensors]
        shapes = [list(t.shape) for _, t in named_tensors]

        payload = {
            "names": names,
            "dtypes": dtypes,
            "shapes": shapes,
            "group_name": self.context.dist_weight_group_name,
            "flush_cache": flush_cache,
            "load_format": "flattened_bucket",
        }
        obj_refs = [
            self.get_target_endpoint(sampler_idx,
                                     ep_idx).update_weights_from_distributed.remote(payload)
            for ep_idx in range(self.num_clusters(sampler_idx))
        ]
        flattened_tensor = SglFlatTensorBucket(named_tensors=named_tensors).get_flattened_tensor()
        handles = [
            dist.broadcast(
                flattened_tensor, src=0, group=self.context.dist_weight_group, async_op=True
            )
        ]
        for h in handles:
            h.wait()
        self._ensure_sglang_weight_update_ok(ray.get(obj_refs), context="distributed_broadcast")


class DeepSeekV4SglangUpdateWeightFactory(SglangUpdateWeightFactory):
    """DeepSeek-V4 SGLang transport with atomic FP8 bucket semantics."""
    def _moe_deepgemm(self) -> bool:
        try:
            return self.context.moe_deepgemm
        except AttributeError:
            return False

    def iter_dsv4_update_buckets(self, model_engine: Any, max_bucket_bytes: int):
        from gpatch_v4.generation_backend.sglang_model_specific.sglang_weight_update_dsv4 import (
            iter_sglang_dsv4_weight_buckets,
        )

        model = model_engine.model
        fp4_qat = bool(getattr(model.config, "fp4_qat", False))
        export_fp4_qdq = self.context.sglang_export_fp4_qdq or fp4_qat
        log(
            "iter_dsv4_update_buckets with "
            f"fp4_qat={fp4_qat} sglang_export_fp4_qdq="
            f"{self.context.sglang_export_fp4_qdq} export_fp4_qdq={export_fp4_qdq}",
            rank=0,
        )
        return iter_sglang_dsv4_weight_buckets(
            model_engine.export_weights(),
            max_bucket_bytes,
            moe_deepgemm=self._moe_deepgemm(),
            fp4_qat=export_fp4_qdq,
        )

    @staticmethod
    def prepare_bucket_tensors(named_params: Any, replace_zeros: bool, *, for_ipc: bool):
        out = []
        for name, param in named_params:
            if replace_zeros:
                weight = (
                    torch.zeros_like(param) if for_ipc else torch.zeros_like(param, device="cuda")
                )
            else:
                is_gathered = bool(getattr(param, "is_gathered_tensor", False))
                if for_ipc:
                    weight = param if is_gathered else param.detach().clone()
                else:
                    weight = param.data
                    if not weight.is_cuda:
                        weight = weight.cuda()
                    weight = weight.clone()
            out.append((name, weight))
        return out

    async def update_weights_by_ipc_handle(
        self, sampler_idx: int, model_engine: Any, replace_zeros: bool = False
    ) -> bool:
        max_bucket_bytes = self.context.update_weight_max_size_bytes
        cleanup_threshold = 10 * max_bucket_bytes
        buckets = self.iter_dsv4_update_buckets(model_engine, max_bucket_bytes)
        count_packed_bucket_num = 0

        with disable_expandable_segments(), perf_time("update weight total", rank=0):
            try:
                for raw_bucket in buckets:
                    named_tensors = self.prepare_bucket_tensors(
                        raw_bucket, replace_zeros, for_ipc=True
                    )
                    bucket_bytes = sum(t.element_size() * t.numel() for _, t in named_tensors)
                    torch.cuda.synchronize()
                    serialized_named_tensors = self.flattened_and_get_ipc_handle(named_tensors)
                    count_packed_bucket_num += 1
                    if dist.get_rank() == self.context.ipc_gather_dst_rank:
                        update_data = {
                            "serialized_named_tensors": serialized_named_tensors,
                            "load_format": "flattened_bucket",
                        }
                        resp = await self.update_co(sampler_idx, update_data)
                        log(f"update_weights response: {resp}", rank=0)
                        self._ensure_sglang_weight_update_ok(resp, context="ipc_handle_dsv4")
                    cpu_barrier()
                    if bucket_bytes >= cleanup_threshold:
                        del named_tensors
                        del serialized_named_tensors
                        clear_memory()
                        torch.cuda.ipc_collect()
            except Exception as error:
                log(f"update_weights error: {error}")
                traceback.print_exc()
                raise RuntimeError(f"update_weights error: {error}")

        log(f"total packed bucket num: {count_packed_bucket_num}")
        clear_memory()
        return True

    def update_weights_by_distributed(
        self, sampler_idx: int, model_engine: Any, replace_zeros: bool = False
    ) -> bool:
        max_bucket_bytes = self.context.update_weight_max_size_bytes
        cleanup_threshold = 10 * max_bucket_bytes
        count_buckets = 0
        pending = None
        for raw_bucket in self.iter_dsv4_update_buckets(model_engine, max_bucket_bytes):
            if pending is not None:
                count_buckets += self.flush_distributed_bucket(
                    sampler_idx, pending, replace_zeros, False, cleanup_threshold
                )
            pending = raw_bucket
        if pending is not None:
            count_buckets += self.flush_distributed_bucket(
                sampler_idx, pending, replace_zeros, True, cleanup_threshold
            )
        clear_memory()
        log(f"distributed weight update done, {count_buckets} buckets broadcast")
        return True

    def flush_distributed_bucket(
        self,
        sampler_idx: int,
        raw_bucket: Any,
        replace_zeros: bool,
        flush_cache: bool,
        cleanup_threshold: int,
    ) -> int:
        is_rank0 = dist.get_rank() == 0
        bucket_bytes = sum(param.element_size() * param.numel() for _, param in raw_bucket)
        if is_rank0:
            buffer = self.prepare_bucket_tensors(raw_bucket, replace_zeros, for_ipc=False)
            torch.cuda.synchronize()
            self.broadcast_sglang_bucket(sampler_idx, buffer, flush_cache=flush_cache)
        if bucket_bytes >= cleanup_threshold:
            flush_refs = None
            if is_rank0:
                flush_refs = self.submit_cache_flush(sampler_idx, bucket_bytes)
                del buffer
            del raw_bucket
            clear_memory()
            if is_rank0:
                self.wait_cache_flush(flush_refs)
        cpu_barrier()
        return 1 if is_rank0 else 0


class VllmUpdateWeightFactory(UpdateWeightFactory):
    """Standard vLLM weight-update transport."""
    async def update_weights_by_ipc_handle(
        self, sampler_idx: int, model_engine: Any, replace_zeros: bool = False
    ) -> bool:
        if self.context.update_weight_use_bucketed_ipc:
            return await self._update_bucketed_ipc(sampler_idx, model_engine, replace_zeros)
        return await self._update_native_ipc(sampler_idx, model_engine, replace_zeros)

    def update_weights_by_distributed(
        self, sampler_idx: int, model_engine: Any, replace_zeros: bool = False
    ) -> bool:
        is_rank0 = dist.get_rank() == 0
        if is_rank0:
            num_clusters = self.num_clusters(sampler_idx)
            start_refs = []
            for ep_i in range(num_clusters):
                target_ep = self.get_target_endpoint(sampler_idx, ep_i)
                start_refs.append(target_ep.start_weights_update.remote({}))
            ray.get(start_refs)
        cpu_barrier()

        weight_generator = model_engine.export_weights()
        max_bucket_bytes = self.context.update_weight_max_size_bytes
        count_buckets = 0

        buffer = []
        buffer_size = 0

        for name, param in weight_generator:
            if is_rank0:
                if replace_zeros:
                    weight = torch.zeros_like(param, device="cuda")
                else:
                    weight = param.data
                    if not weight.is_cuda:
                        weight = weight.cuda()
                    weight = weight.detach().contiguous().clone()
                buffer.append((name, weight))

            buffer_size += param.element_size() * param.numel()

            if buffer_size >= max_bucket_bytes:
                if is_rank0:
                    torch.cuda.synchronize()
                    self.broadcast_vllm_bucket(sampler_idx, buffer)
                    count_buckets += 1
                buffer = []
                buffer_size = 0
                cpu_barrier()
                torch.cuda.empty_cache()
        if buffer_size > 0:
            if is_rank0:
                torch.cuda.synchronize()
                self.broadcast_vllm_bucket(sampler_idx, buffer)
                count_buckets += 1
            cpu_barrier()
        del weight_generator
        clear_memory()

        if is_rank0:
            num_clusters = self.num_clusters(sampler_idx)
            fin_refs = []
            for ep_i in range(num_clusters):
                target_ep = self.get_target_endpoint(sampler_idx, ep_i)
                fin_refs.append(target_ep.finalize_weights_update.remote({}))
            ray.get(fin_refs)
        cpu_barrier()

        log(
            f"distributed weight update done (vllm), {count_buckets} buckets broadcast "
            f"(bucket={max_bucket_bytes} bytes)",
            rank=0,
        )
        return True

    def _start_vllm_update(self, sampler_idx):
        if dist.get_rank() == 0:
            obj_refs = [
                self.get_target_endpoint(sampler_idx, ep).start_weights_update.remote({})
                for ep in range(self.num_clusters(sampler_idx))
            ]
            ray.get(obj_refs)
        cpu_barrier()

    def finalize_vllm_update(self, sampler_idx):
        if dist.get_rank() == 0:
            obj_refs = [
                self.get_target_endpoint(sampler_idx, ep).finalize_weights_update.remote({})
                for ep in range(self.num_clusters(sampler_idx))
            ]
            ray.get(obj_refs)
        cpu_barrier()

    async def _update_native_ipc(self, sampler_idx, model_engine, replace_zeros):
        from torch.multiprocessing.reductions import reduce_tensor

        from gpatch_v4.generation_backend.bucketed_ipc_transfer import gcore_gpu_uuid
        world_size, rank = dist.get_world_size(), dist.get_rank()
        gpu_uuid = gcore_gpu_uuid(torch.device("cuda", torch.cuda.current_device()))
        self._start_vllm_update(sampler_idx)
        names, dtypes, shapes, handles, alive, size = [], [], [], [], [], 0

        async def flush():
            nonlocal names, dtypes, shapes, handles, alive, size
            if not names:
                cpu_barrier()
                return
            gathered = [None] * world_size
            dist.all_gather_object(gathered, handles)
            if rank == 0:
                merged = [{} for _ in handles]
                for rank_handles in gathered:
                    assert rank_handles is not None and len(rank_handles) == len(merged)
                    for idx, entry in enumerate(rank_handles):
                        merged[idx].update(entry)
                payload = {
                    "names": names,
                    "dtype_names": dtypes,
                    "shapes": shapes,
                    "ipc_handles": merged,
                    "is_checkpoint_format": True,
                    "is_last_bucket": False
                }
                ray.get(
                    [
                        self.get_target_endpoint(sampler_idx, ep).update_weights.remote(payload)
                        for ep in range(self.num_clusters(sampler_idx))
                    ]
                )
            names, dtypes, shapes, handles, alive, size = [], [], [], [], [], 0
            cpu_barrier()

        weight_generator = model_engine.export_weights()
        for name, param in weight_generator:
            source = param.data if param.data.is_cuda else param.data.cuda()
            weight = torch.zeros_like(source, device="cuda"
                                     ) if replace_zeros else source.detach().contiguous().clone()
            names.append(name)
            dtypes.append(str(weight.dtype).split(".")[-1])
            shapes.append(list(weight.shape))
            handles.append({gpu_uuid: reduce_tensor(weight)})
            alive.append(weight)
            size += weight.element_size() * weight.numel()
            if size >= self.context.update_weight_max_size_bytes:
                torch.cuda.synchronize()
                await flush()
        torch.cuda.synchronize()
        await flush()
        self.finalize_vllm_update(sampler_idx)
        return True

    async def _update_bucketed_ipc(self, sampler_idx, model_engine, replace_zeros):
        from gpatch_v4.generation_backend.bucketed_ipc_transfer import (
            FlatIpcBucketBuilder,
            gcore_gpu_uuid,
        )
        max_bytes = self.context.update_weight_max_size_bytes
        assert max_bytes > 0, f"update_weight_max_size_bytes must be > 0, got {max_bytes}"
        builder = FlatIpcBucketBuilder(max_bucket_bytes=max_bytes)
        rank, world_size = dist.get_rank(), dist.get_world_size()
        gpu_uuid = gcore_gpu_uuid(torch.device("cuda", torch.cuda.current_device()))
        self._start_vllm_update(sampler_idx)
        bucket_index = 0

        async def update_and_flush():
            nonlocal bucket_index
            if not builder.has_content():
                cpu_barrier()
                return
            torch.cuda.synchronize()
            local = builder.drain()
            gathered = [None] * world_size
            payloads = [
                {
                    "names": b.names,
                    "key_size": b.key_size,
                    "key_numel": b.key_numel,
                    "uuid_handle": {
                        gpu_uuid: b.local_ipc_handle
                    },
                    "flat_shape": b.flat_shape,
                    "flat_dtype": b.flat_dtype_name
                } for b in local
            ]
            dist.all_gather_object(gathered, payloads)
            if rank == 0:
                assert gathered[0] is not None
                num_buckets = len(gathered[0])
                for rank_i, rank_payload in enumerate(gathered):
                    assert rank_payload is not None and len(rank_payload) == num_buckets, (
                        f"flat-ipc all_gather length mismatch at rank {rank_i}"
                    )
                for index, base in enumerate(gathered[0]):
                    merged = {}
                    for rank_payload in gathered:
                        merged.update(rank_payload[index]["uuid_handle"])
                    bucket_index += 1
                    request = {
                        "names": base["names"],
                        "key_size": base["key_size"],
                        "key_numel": base["key_numel"],
                        "ipc_handles": merged,
                        "flat_shape": base["flat_shape"],
                        "flat_dtype": base["flat_dtype"],
                        "bucket_idx": bucket_index
                    }
                    obj_refs = [
                        self.get_target_endpoint(sampler_idx,
                                                 ep).update_weights_bucketed.remote(request)
                        for ep in range(self.num_clusters(sampler_idx))
                    ]
                    ray.get(obj_refs)
            del local
            cpu_barrier()

        weight_generator = model_engine.export_weights()
        for name, param in weight_generator:
            source = param.data if param.data.is_cuda else param.data.cuda()
            weight = torch.zeros_like(source, device="cuda"
                                     ) if replace_zeros else source.detach().contiguous()
            builder.add(name, weight)
            if builder.is_full():
                await update_and_flush()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        await update_and_flush()
        try:
            del name, param, source, weight
        except NameError:
            pass
        del weight_generator
        clear_memory()
        torch.cuda.ipc_collect()
        self.finalize_vllm_update(sampler_idx)
        return True

    def broadcast_vllm_bucket(self, sampler_idx, named_tensors):
        names, dtypes, shapes, sizes = [], [], [], []
        for n, t in named_tensors:
            names.append(n)
            dtypes.append(str(t.dtype).split(".")[-1])
            shapes.append(list(t.shape))
            sizes.append(t.element_size() * t.numel())
        total_bytes = sum(sizes)
        flat = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
        offset = 0
        for (_, tensor), size in zip(named_tensors, sizes):
            flat[offset:offset + size].copy_(
                tensor.detach().contiguous().view(-1).view(torch.uint8)
            )
            offset += size
        payload = {
            "names": names,
            "dtype_names": dtypes,
            "shapes": shapes,
            "total_bytes": total_bytes,
            "group_name": self.context.dist_weight_group_name
        }
        refs = [
            self.get_target_endpoint(sampler_idx,
                                     ep).update_weights_from_distributed.remote(payload)
            for ep in range(self.num_clusters(sampler_idx))
        ]
        self.context.dist_weight_group.broadcast(flat, src=0, stream=torch.cuda.current_stream())
        torch.cuda.current_stream().synchronize()
        ray.get(refs)


class DeepSeekV4VllmUpdateWeightFactory(VllmUpdateWeightFactory):
    """vLLM DeepSeek-V4 requires the existing bucketed IPC transport."""
    async def update_weights_by_ipc_handle(
        self, sampler_idx: int, model_engine: Any, replace_zeros: bool = False
    ) -> bool:
        assert self.context.update_weight_use_bucketed_ipc, (
            "DeepSeek-V4 vLLM update_weights requires bucketed IPC. "
            "Please set policy.sampler_client.update_weight_use_bucketed_ipc=True."
        )
        return await super().update_weights_by_ipc_handle(sampler_idx, model_engine, replace_zeros)


def get_update_weight_factory(
    *, context: UpdateWeightContext, model_arch: str | None
) -> UpdateWeightFactory:
    """Return the transport factory for a model architecture and backend."""
    if context.infer_backend == "sglang":
        if model_arch == MODEL_ARCH.DEEPSEEK_V4:
            return DeepSeekV4SglangUpdateWeightFactory(context)
        return SglangUpdateWeightFactory(context)
    if context.infer_backend == "vllm":
        if model_arch == MODEL_ARCH.DEEPSEEK_V4:
            return DeepSeekV4VllmUpdateWeightFactory(context)
        return VllmUpdateWeightFactory(context)
    raise AssertionError("only sglang and vllm backend are supported")
