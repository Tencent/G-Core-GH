import asyncio
import os
import traceback
from abc import ABC
import ray

import torch
import torch.distributed as dist

from gpatch_v4.orches.flashinfer_cudart_fix import patch_ctypes_for_cudart_stub

# sglang-native serialization (used only when backend == sglang)

try:
    with patch_ctypes_for_cudart_stub():
        from sglang.srt.model_executor.model_runner import (
            FlattenedTensorBucket as SglFlatTensorBucket,
        )
        from sglang.srt.utils import MultiprocessingSerializer as SglSerializer
    _has_sglang_ipc = True
except Exception as e:
    _has_sglang_ipc = False

from gpatch_v4.core.parallel_state import cpu_barrier
from gpatch_v4.utils import clear_memory, log, logging_rank0, perf_time

# Backend-agnostic serialization (used for vllm; no sglang dependency)
from gpatch_v4.utils.tensor_ipc import FlattenedTensorBucket as GpatchFlatTensorBucket
from gpatch_v4.utils.tensor_ipc import (
    serialize_pickle_base64 as _gpatch_serialize_pickle_base64,
)


class UpdateWeightIpcMixin:
    """Mixin for transferring model weights to sampler engines via IPC.

    Provides methods to flatten, serialize, and send weight tensors to
    remote inference engines. Covers both sglang (flattened-bucket + CUDA
    IPC via ``FlattenedTensorBucket``) and vLLM (per-tensor
    ``reduce_tensor`` handles in :meth:`_update_weights_by_ipc_handle_vllm`
    / flat per-dtype buckets in :meth:`_update_weights_by_bucketed_ipc_vllm`)
    backends.
    """
    async def update_co(self, sampler_idx, update_data):
        """Send an update-weights RPC call to the target sampler endpoint.

        Parameters
        ----------
        sampler_idx : int
        update_data : dict
            Serialized weight data.

        Returns
        -------
        list
            RPC response.
        """
        target_ep = self.rpc_client_lst[sampler_idx].get_target_endpoint(
            sample_idx=None, ep_idx=self._ipc_target
        )
        co = self.rpc_client_lst[sampler_idx].call(target_ep, "update_weights", update_data)
        return await asyncio.gather(co)

    def flatten_and_serialize(self, named_tensors):
        """Flatten + serialize named tensors using the right backend impl.

        * sglang: GPU tensors → CUDA IPC handles via sglang's serializer.
        * vllm:   CPU tensors → plain pickle (no CUDA IPC) because vllm
          workers have different CUDA_VISIBLE_DEVICES and cannot access
          training-side physical GPU devices via IPC handles.
        """

        backend = self.infer_backend
        if backend == "sglang" and _has_sglang_ipc:
            bucket = SglFlatTensorBucket(named_tensors=named_tensors)
            data = {
                "flattened_tensor": bucket.get_flattened_tensor(),
                "metadata": bucket.get_metadata(),
            }
            return SglSerializer.serialize(data, output_str=True)
        else:
            assert False, "never reach here in sglang backend"
            cpu_tensors = [(n, t.cpu()) for n, t in named_tensors]
            bucket = GpatchFlatTensorBucket(named_tensors=cpu_tensors)
            data = {
                "flattened_tensor": bucket.get_flattened_tensor(),
                "metadata": bucket.get_metadata(),
            }
            return _gpatch_serialize_pickle_base64(data)

    def flattened_and_get_ipc_handle(self, named_tensors):
        """Flatten named tensors, serialize, and gather IPC handles.

        Parameters
        ----------
        named_tensors : list of tuple[str, torch.Tensor]

        Returns
        -------
        list or None
            Serialized tensors gathered on the destination rank;
            *None* on other ranks.
        """
        serialized_tensors = self.flatten_and_serialize(named_tensors)

        serialized_named_tensors = (
            [None] * dist.get_world_size(self._ipc_gather_group)
            if self._ipc_gather_dst_rank == dist.get_rank() else None
        )
        dist.gather_object(
            serialized_tensors,
            object_gather_list=serialized_named_tensors,
            dst=self._ipc_gather_dst_rank,
            group=self._ipc_gather_group,
        )
        return serialized_named_tensors

    async def _dispatch_vllm_start_weights_update(self, sampler_idx: int) -> None:
        """RPC ``start_weights_update`` to every sampler cluster endpoint."""
        if dist.get_rank() != 0:
            cpu_barrier()
            return

        num_clusters = self.svr_cluster_num_per_sampler[sampler_idx]
        refs = []
        for ep_i in range(num_clusters):
            target_ep = self.rpc_client_lst[sampler_idx].get_target_endpoint(
                sample_idx=None,
                ep_idx=ep_i,
            )
            refs.append(target_ep.start_weights_update.remote({}))
        ray.get(refs)
        cpu_barrier()

    async def _dispatch_vllm_finalize_weights_update(self, sampler_idx: int) -> None:
        """RPC ``finalize_weights_update`` to every sampler cluster endpoint."""
        if dist.get_rank() != 0:
            cpu_barrier()
            return

        num_clusters = self.svr_cluster_num_per_sampler[sampler_idx]
        refs = []
        for ep_i in range(num_clusters):
            target_ep = self.rpc_client_lst[sampler_idx].get_target_endpoint(
                sample_idx=None,
                ep_idx=ep_i,
            )
            refs.append(target_ep.finalize_weights_update.remote({}))
        ray.get(refs)
        cpu_barrier()

    async def update_weights_by_ipc_handle(self, sampler_idx, model_engine, replace_zeros=False):
        """Stream model weights to the sampler via IPC with pipelining.

        Parameters
        ----------
        sampler_idx : int
        model_engine : object
            Exposes ``export_weights()`` generator.
        replace_zeros : bool, optional
            Send zero tensors instead of real weights (testing).

        Returns
        -------
        bool
        """
        backend = self.infer_backend
        if backend == "sglang":
            return await self._update_weights_by_ipc_handle_sglang(
                sampler_idx, model_engine, replace_zeros
            )
        else:
            assert backend == "vllm", "only sglang and vllm backend are supported"
            model_arch = self.config.policy.model_arch
            if model_arch == "deepseek_v4":
                assert self.update_weight_use_bucketed_ipc, (
                    "DeepSeek-V4 vLLM update_weights requires bucketed IPC. "
                    "Please set policy.sampler_client.update_weight_use_bucketed_ipc=True."
                )
            if self.update_weight_use_bucketed_ipc:
                return await self._update_weights_by_bucketed_ipc_vllm(
                    sampler_idx, model_engine, replace_zeros
                )
            return await self._update_weights_by_ipc_handle_vllm(
                sampler_idx, model_engine, replace_zeros
            )

    async def _update_weights_by_ipc_handle_sglang(
        self, sampler_idx, model_engine, replace_zeros=False
    ):
        rank = dist.get_rank()

        update_weight_max_size_bytes = self.update_weight_max_size_bytes
        large_tensor_cleanup_threshold_bytes = 10 * update_weight_max_size_bytes
        weight_generator = model_engine.export_weights()

        async def async_update_weights():
            count_packed_bucket_num = 0
            with perf_time(f"update weight total", rank=0):
                try:
                    converted_named_tensors_by_dtypes = {}
                    converted_buffer_size_by_dtypes = {}
                    for name, param in weight_generator:
                        is_gathered_tensor = bool(
                            getattr(param, "is_gathered_tensor", False)
                        )
                        if replace_zeros:
                            weight_tensor = torch.zeros_like(param)
                        elif is_gathered_tensor:
                            weight_tensor = param
                        else:
                            weight_tensor = param.detach().clone()

                        dtype = weight_tensor.dtype
                        if dtype not in converted_named_tensors_by_dtypes:
                            converted_named_tensors_by_dtypes[dtype] = []
                            converted_buffer_size_by_dtypes[dtype] = 0
                        converted_named_tensors_by_dtypes[dtype].append((name, weight_tensor))
                        converted_buffer_size_by_dtypes[dtype] += weight_tensor.element_size(
                        ) * weight_tensor.numel()

                        if converted_buffer_size_by_dtypes[dtype] >= update_weight_max_size_bytes:
                            torch.cuda.synchronize()
                            named_tensors = converted_named_tensors_by_dtypes[dtype]
                            bucket_bytes = converted_buffer_size_by_dtypes[dtype]
                            serialized_named_tensors = self.flattened_and_get_ipc_handle(
                                named_tensors
                            )

                            count_packed_bucket_num += 1
                            if dist.get_rank() == self._ipc_gather_dst_rank:
                                update_data = {
                                    "serialized_named_tensors": serialized_named_tensors,
                                    "load_format": "flattened_bucket",
                                }
                                resp = await self.update_co(sampler_idx, update_data)
                                log(f"update_weights response: {resp}", rank=0)
                                if bucket_bytes >= large_tensor_cleanup_threshold_bytes:
                                    del update_data, resp

                            cpu_barrier()
                            # Avoid prefetching the next bucket while the
                            # current flattened CUDA IPC bucket is still alive.
                            # Large MoE weights can otherwise require two
                            # bucket copies plus the flat tensor at once.
                            converted_named_tensors_by_dtypes[dtype] = []
                            converted_buffer_size_by_dtypes[dtype] = 0
                            if bucket_bytes >= large_tensor_cleanup_threshold_bytes:
                                del named_tensors
                                del serialized_named_tensors
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
                            if dist.get_rank() == self._ipc_gather_dst_rank:
                                update_data = {
                                    "serialized_named_tensors": serialized_named_tensors,
                                    "load_format": "flattened_bucket",
                                }
                                last_update_weight_co.append(
                                    self.update_co(sampler_idx, update_data)
                                )

                    if dist.get_rank() == self._ipc_gather_dst_rank:
                        resps = await asyncio.gather(*last_update_weight_co)
                        log(f"update_weights response: {resps}", rank=0)
                    cpu_barrier()
                    return count_packed_bucket_num
                except Exception as e:
                    log(f"update_weights error: {e}")
                    traceback.print_exc()
                    raise RuntimeError(f"update_weights error: {e}")

        count_packed_bucket_num = await async_update_weights()
        log(f"total packed bucket num: {count_packed_bucket_num}")
        return True

    async def _update_weights_by_ipc_handle_vllm(
        self, sampler_idx, model_engine, replace_zeros=False
    ):
        """Transfer weights via vLLM native IPCWeightTransferEngine.

        Uses ``torch.multiprocessing.reductions.reduce_tensor`` on the trainer
        rank 0 to create CUDA IPC handles for each weight tensor, then forwards
        the handles to every sampler cluster endpoint via Ray.  The sampler
        vLLM workers rebuild the tensor locally with ``rebuild_cuda_tensor``
        and copy into the model (see ``IPCWeightTransferEngine.receive_weights``
        in ``vllm/distributed/weight_transfer/ipc_engine.py``).

        Parameters
        ----------
        sampler_idx : int
        model_engine : object
            Exposes ``export_weights()`` generator.
        replace_zeros : bool, optional
            Send zero tensors instead of real weights (testing).

        Returns
        -------
        bool

        Notes
        -----
        * Requires trainer rank 0 and at least one vLLM worker to share the
          same physical GPU UUID (colocated placement).  Cross-node use is
          not supported; use ``_update_weights_by_distributed_vllm`` instead.
        * Weights are flushed in buckets sized by
          ``self.update_weight_max_size_bytes``.  After each bucket is
          acknowledged by Ray the local tensor references are released so
          ``reduce_tensor`` stops pinning GPU memory.
        """
        from torch.multiprocessing.reductions import reduce_tensor

        from gpatch_v4.generation_backend.bucketed_ipc_transfer import gcore_gpu_uuid

        weight_generator = model_engine.export_weights()
        world_size = dist.get_world_size()
        my_rank = dist.get_rank()
        is_rank0 = my_rank == 0

        # Each trainer rank computes an IPC handle for its *own* GPU. The
        # sampler side is a colocated vLLM cluster spanning the same set of
        # physical GPUs, so every worker needs to find its own UUID in the
        # broadcast handle dict. We therefore all-gather per-rank
        # ``{gpu_uuid: reduce_tensor(weight)}`` dicts at flush time and
        # rank 0 merges them into a single per-tensor dict covering every
        # physical GPU in the sampler cluster before issuing the RPC.
        gpu_uuid = gcore_gpu_uuid(torch.device("cuda", torch.cuda.current_device()))

        await self._dispatch_vllm_start_weights_update(sampler_idx)

        names: list = []
        dtype_names: list = []
        shapes: list = []
        local_handles: list = []  # [{gpu_uuid: reduced_tuple}] per tensor in bucket
        alive_tensors: list = []
        buffer_bytes = 0
        total_weights_sent = 0
        bucket_count = 0

        async def update_weight_fn(is_last: bool = False):
            nonlocal names, dtype_names, shapes, local_handles, alive_tensors
            nonlocal buffer_bytes, total_weights_sent, bucket_count
            if not names:
                cpu_barrier()
                return

            # Gather per-rank handle lists (same length on all ranks since
            # export_weights yields identical tensors in identical order for
            # DP-only training).
            gathered_lists: list = [None] * world_size
            dist.all_gather_object(gathered_lists, local_handles)

            if is_rank0:
                merged = [{} for _ in range(len(local_handles))]
                for rank_list in gathered_lists:
                    assert rank_list is not None and len(rank_list) == len(merged), (
                        f"native_ipc rank-handle length mismatch: {len(rank_list) if rank_list else 0}"
                        f" vs {len(merged)}"
                    )
                    for j, entry in enumerate(rank_list):
                        merged[j].update(entry)

                # Legacy key kept for backward compatibility; finalize is
                # handled by a separate RPC after all buckets are acked.
                update_info = {
                    "names": names,
                    "dtype_names": dtype_names,
                    "shapes": shapes,
                    "ipc_handles": merged,
                    "is_checkpoint_format": True,
                    "is_last_bucket": False,
                }
                num_clusters = self.svr_cluster_num_per_sampler[sampler_idx]
                obj_refs = []
                for ep_i in range(num_clusters):
                    target_ep = self.rpc_client_lst[sampler_idx].get_target_endpoint(
                        sample_idx=None,
                        ep_idx=ep_i,
                    )
                    # ``target_ep`` is a ``GrpoSamplerActor`` handle whose
                    # ``update_weights`` wrapper forwards into
                    # ``VllmEngine.update_weights``, which in turn fans out
                    # via ``collective_rpc`` to
                    # ``GCoreVllmWorkerExtension.gcore_update_weights_ipc``
                    # on every vLLM worker (see vllm_engine.py).
                    obj_refs.append(target_ep.update_weights.remote(update_info))
                ray.get(obj_refs)
                total_weights_sent += len(names)
                bucket_count += 1
                log(
                    f"native_ipc bucket {bucket_count} sent {len(names)} tensors "
                    f"({buffer_bytes / 1e9:.2f} GB, is_last={is_last})",
                    rank=0,
                )
            names = []
            dtype_names = []
            shapes = []
            local_handles = []
            alive_tensors = []
            buffer_bytes = 0
            cpu_barrier()

        for name, param in weight_generator:
            src = param.data
            if not src.is_cuda:
                src = src.cuda()
            if replace_zeros:
                weight = torch.zeros_like(src, device="cuda")
            else:
                weight = src.detach().contiguous().clone()

            names.append(name)
            dtype_names.append(str(weight.dtype).split(".")[-1])
            shapes.append(list(weight.shape))
            local_handles.append({gpu_uuid: reduce_tensor(weight)})
            alive_tensors.append(weight)
            buffer_bytes += weight.element_size() * weight.numel()

            if buffer_bytes >= self.update_weight_max_size_bytes:
                torch.cuda.synchronize()
                await update_weight_fn(is_last=False)

        torch.cuda.synchronize()
        await update_weight_fn(is_last=True)

        await self._dispatch_vllm_finalize_weights_update(sampler_idx)

        log(
            f"native_ipc weight update done, {total_weights_sent} tensors "
            f"across {bucket_count} buckets via vllm IPCWeightTransferEngine",
            rank=0,
        )
        return True

    async def _update_weights_by_bucketed_ipc_vllm(
        self, sampler_idx, model_engine, replace_zeros=False
    ):
        """Transfer weights via v3-style flat-IPC bucketing over Ray RPC.

        One CUDA IPC handle per (dtype, bucket) -- not per tensor -- which
        is the key speedup over :meth:`_update_weights_by_ipc_handle_vllm`
        for large MoE checkpoints (thousands of fused expert tensors).

        Flow
        ----
        1. Each trainer rank iterates its own copy of the weights and
           accumulates them into per-dtype buckets via
           :class:`FlatIpcBucketBuilder`.
        2. When the running byte count reaches
           ``self.update_weight_max_size_bytes`` we drain the builder,
           producing one :class:`FlatIpcBucket` per dtype. Each bucket
           carries a flat CUDA tensor + a single ``reduce_tensor`` IPC
           handle keyed by the rank's GPU UUID.
        3. ``all_gather_object`` collects per-rank handle dicts so rank 0
           can emit a single Ray RPC per (cluster, dtype, bucket) that
           reaches every colocated vLLM worker via ``collective_rpc`` ->
           :meth:`GCoreVllmWorkerExtension.gcore_update_weights_bucketed`.
        4. After draining the final bucket, rank 0 fires one
           :meth:`~GCoreVllmWorkerExtension.gcore_finalize_weights_update`
           RPC so :func:`gpatch_v4.generation_backend.vllm_weight_reload.finalize_weights_after_reload`
           runs exactly once.

        Notes
        -----
        * Requires colocated placement (trainer rank K and vLLM TP rank K
          share a physical GPU) so the per-rank UUID keys in
          ``ipc_handles`` cover every worker in the sampler cluster.
        * Flat tensors are kept alive in ``alive_buckets`` until
          ``ray.get`` acks the RPC, ensuring the CUDA IPC segment stays
          mapped for the entire receiver-side load.
        """
        from gpatch_v4.generation_backend.bucketed_ipc_transfer import (
            FlatIpcBucket,
            FlatIpcBucketBuilder,
            gcore_gpu_uuid,
        )

        max_bucket_bytes = self.update_weight_max_size_bytes
        assert max_bucket_bytes > 0, (
            f"update_weight_max_size_bytes must be > 0, got {max_bucket_bytes}"
        )

        world_size = dist.get_world_size()
        my_rank = dist.get_rank()
        is_rank0 = my_rank == 0

        gpu_uuid = gcore_gpu_uuid(torch.device("cuda", torch.cuda.current_device()))

        builder = FlatIpcBucketBuilder(max_bucket_bytes=max_bucket_bytes)
        weight_generator = model_engine.export_weights()

        await self._dispatch_vllm_start_weights_update(sampler_idx)

        total_sent = 0
        bucket_count = 0

        async def update_weight_fn() -> None:
            """Drain the builder and dispatch one RPC per dtype bucket.

            Every rank flattens + reduces locally, then all_gathers the
            ``{gpu_uuid: ipc_handle}`` entries so rank 0 can merge them
            into a full handle dict covering every GPU in the sampler
            cluster. The trainer-side flat tensors stay pinned in
            ``alive_buckets`` until the Ray round-trip completes so the
            receiver-side IPC views remain valid.
            """
            nonlocal total_sent, bucket_count
            if not builder.has_content():
                cpu_barrier()
                return

            torch.cuda.synchronize()
            local_buckets: list[FlatIpcBucket] = builder.drain()
            # Hold on to flat tensors so their CUDA backing memory stays
            # valid for the receiver through ``ray.get`` below.
            alive_buckets = local_buckets

            # DP-replicated weights -> every rank produces the same dtype
            # ordering and name ordering, so we can zip gathered dicts by
            # index. Each payload carries ``{gpu_uuid: handle}`` for this
            # rank; rank 0 merges across ranks per bucket index.
            local_payloads = [
                {
                    "names": b.names,
                    "key_size": b.key_size,
                    "key_numel": b.key_numel,
                    "uuid_handle": {
                        gpu_uuid: b.local_ipc_handle
                    },
                    "flat_shape": b.flat_shape,
                    "flat_dtype": b.flat_dtype_name,
                } for b in local_buckets
            ]
            gathered: list = [None] * world_size
            dist.all_gather_object(gathered, local_payloads)

            if is_rank0:
                assert gathered[0] is not None
                num_buckets = len(gathered[0])
                for rank_i, rank_payload in enumerate(gathered):
                    assert rank_payload is not None and len(rank_payload) == num_buckets, (
                        f"flat-ipc all_gather length mismatch at rank {rank_i}"
                    )
                num_clusters = self.svr_cluster_num_per_sampler[sampler_idx]
                for bucket_i in range(num_buckets):
                    base = gathered[0][bucket_i]
                    merged_handles: dict = {}
                    for rank_payload in gathered:
                        merged_handles.update(rank_payload[bucket_i]["uuid_handle"])
                    next_bucket_count = bucket_count + 1
                    req = {
                        "names": base["names"],
                        "key_size": base["key_size"],
                        "key_numel": base["key_numel"],
                        "ipc_handles": merged_handles,
                        "flat_shape": base["flat_shape"],
                        "flat_dtype": base["flat_dtype"],
                        "bucket_idx": next_bucket_count,
                    }
                    obj_refs = []
                    for ep_i in range(num_clusters):
                        target_ep = self.rpc_client_lst[sampler_idx].get_target_endpoint(
                            sample_idx=None,
                            ep_idx=ep_i,
                        )
                        obj_refs.append(target_ep.update_weights_bucketed.remote(req))
                    ray.get(obj_refs)
                    total_sent += len(base["names"])
                    bucket_count = next_bucket_count
                    log(
                        f"bucketed_ipc bucket {bucket_count} dtype={base['flat_dtype']} "
                        f"tensors={len(base['names'])} flat_numel={base['flat_shape'][0]}",
                        rank=0,
                    )

            del alive_buckets, local_buckets
            cpu_barrier()

        # Iterate weights and bucket by total bytes.
        for name, param in weight_generator:
            src = param.data
            if not src.is_cuda:
                src = src.cuda()
            if replace_zeros:
                weight = torch.zeros_like(src, device="cuda")
            else:
                weight = src.detach().contiguous()

            builder.add(name, weight)
            if builder.is_full():
                await update_weight_fn()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        await update_weight_fn()

        # ``for``-loop locals keep the last tensor alive until function exit.
        # Drop them explicitly before clear_memory so allocator segments can
        # be fully returned when possible.
        try:
            del name, param, src, weight
        except NameError:
            pass

        # Release GPU tensors held by the weight generator iteration.
        del weight_generator
        clear_memory()
        torch.cuda.ipc_collect()

        # Post-load finalize once (process_weights_after_loading on worker).
        await self._dispatch_vllm_finalize_weights_update(sampler_idx)

        log(
            f"bucketed_ipc weight update done, {total_sent} tensors across "
            f"{bucket_count} flat buckets (bucket={max_bucket_bytes} bytes)",
            rank=0,
        )
        return True


class UpdateWeightDistributedMixin:
    """Mixin for transferring model weights to sampler engines via NCCL broadcast.

    Uses sglang's ``init_weights_update_group`` / ``update_weights_from_distributed``
    API to create an NCCL group between training rank 0 and inference engine GPU
    workers, then broadcasts weights directly to avoid the serialisation and
    gather overhead of the IPC path.
    """

    _dist_weight_group = None
    _dist_weight_group_name = None

    async def init_distributed_weight_group(self, group_name="weight_update_group"):
        """Create NCCL group between training rank 0 and sampler engine workers."""
        import socket
        sampler_idx = 0

        backend = self.infer_backend
        if backend == "vllm" and self.config.placement_type != "disaggregated":
            await self.wake_up(sampler_idx, tag_names=["weights"])
            cpu_barrier()

        num_clusters = self.svr_cluster_num_per_sampler[sampler_idx]
        engine_gpu_count = self._get_sampler_engine_gpu_count(sampler_idx)
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

        if dist.get_rank() == 0:
            obj_refs = []
            for ep_i in range(num_clusters):
                rank_offset = 1 + ep_i * engine_gpu_count
                req_data = {
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": rank_offset,
                    "world_size": world_size,
                    "group_name": group_name,
                    "backend": "nccl",
                }
                target_ep = self.rpc_client_lst[sampler_idx].get_target_endpoint(
                    sample_idx=None,
                    ep_idx=ep_i,
                )
                obj_refs.append(target_ep.init_weights_update_group.remote(req_data))

            if backend == "vllm":
                from vllm.distributed.weight_transfer.nccl_engine import (
                    NCCLWeightTransferEngine,
                )
                self._dist_weight_group = NCCLWeightTransferEngine.trainer_init(
                    {
                        "master_address": master_address,
                        "master_port": master_port,
                        "world_size": world_size,
                    }
                )
            else:
                from sglang.srt.utils import init_custom_process_group
                self._dist_weight_group = init_custom_process_group(
                    backend="nccl",
                    init_method=f"tcp://{master_address}:{master_port}",
                    world_size=world_size,
                    rank=0,
                    group_name=group_name,
                )
            ray.get(obj_refs)

        cpu_barrier()
        self._dist_weight_group_name = group_name
        log(
            f"distributed weight group '{group_name}' initialized, "
            f"world_size={world_size} (1 train + {num_clusters}x{engine_gpu_count} sampler)",
            rank=0,
        )

        if backend == "vllm" and self.config.placement_type != "disaggregated":
            await self.sleep(sampler_idx)
            cpu_barrier()

    async def destroy_distributed_weight_group(self):
        """Destroy the NCCL group created by ``init_distributed_weight_group``."""
        if self._dist_weight_group_name is None:
            return

        backend = self.infer_backend
        assert backend == "sglang", "only sglang backend is supported"
        sampler_idx = 0
        if dist.get_rank() == 0:
            num_clusters = self.svr_cluster_num_per_sampler[sampler_idx]
            obj_refs = []
            for ep_i in range(num_clusters):
                target_ep = self.rpc_client_lst[sampler_idx].get_target_endpoint(
                    sample_idx=None,
                    ep_idx=ep_i,
                )
                obj_refs.append(
                    target_ep.destroy_weights_update_group.remote(
                        {"group_name": self._dist_weight_group_name}
                    )
                )
            ray.get(obj_refs)
            dist.destroy_process_group(self._dist_weight_group)
            self._dist_weight_group = None

        cpu_barrier()
        self._dist_weight_group_name = None
        log("distributed weight group destroyed", rank=0)

    def update_weights_by_distributed(self, sampler_idx, model_engine, replace_zeros=False):
        """Iterate ``export_weights()`` on all ranks and broadcast from rank 0."""
        backend = self.infer_backend
        if backend == "sglang":
            return self._update_weights_by_distributed_sglang(
                sampler_idx, model_engine, replace_zeros
            )
        else:
            assert backend == "vllm", "only sglang and vllm backend are supported"
            return self._update_weights_by_distributed_vllm(
                sampler_idx,
                model_engine,
                replace_zeros,
            )

    def _update_weights_by_distributed_sglang(self, sampler_idx, model_engine, replace_zeros=False):
        """Iterate ``export_weights()`` on all ranks and broadcast from rank 0."""
        weight_generator = model_engine.export_weights()
        max_bucket_bytes = self.update_weight_max_size_bytes
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
                    self._broadcast_weight_bucket(sampler_idx, buffer, flush_cache=False)
                    count_buckets += 1
                if bucket_bytes >= large_tensor_cleanup_threshold_bytes:
                    flush_refs = None
                    if is_rank0:
                        flush_refs = self._submit_sampler_cache_flush_after_distributed_bucket(
                            sampler_idx, bucket_bytes
                        )
                        del weight
                    del buffer
                    del param
                    clear_memory()
                    if is_rank0:
                        self._wait_sampler_cache_flush_after_distributed_bucket(flush_refs)
                buffer = []
                buffer_size = 0
                cpu_barrier()

        if buffer_size > 0:
            bucket_bytes = buffer_size
            if is_rank0:
                torch.cuda.synchronize()
                self._broadcast_weight_bucket(sampler_idx, buffer, flush_cache=True)
                count_buckets += 1
            if bucket_bytes >= large_tensor_cleanup_threshold_bytes:
                flush_refs = None
                if is_rank0:
                    flush_refs = self._submit_sampler_cache_flush_after_distributed_bucket(
                        sampler_idx, bucket_bytes
                    )
                    del weight
                del buffer
                del param
                clear_memory()
                if is_rank0:
                    self._wait_sampler_cache_flush_after_distributed_bucket(flush_refs)
            cpu_barrier()

        log(f"distributed weight update done, {count_buckets} buckets broadcast", rank=0)
        return True

    def _submit_sampler_cache_flush_after_distributed_bucket(self, sampler_idx, bucket_bytes):
        """Submit sampler-side cache flush after a large sglang distributed bucket."""
        log(
            f"flush sampler cache after large distributed weight bucket "
            f"({bucket_bytes / 1024**3:.3f} GiB)",
            rank=0,
        )
        num_clusters = self.svr_cluster_num_per_sampler[sampler_idx]
        obj_refs = []
        for ep_i in range(num_clusters):
            target_ep = self.rpc_client_lst[sampler_idx].get_target_endpoint(
                sample_idx=None,
                ep_idx=ep_i,
            )
            obj_refs.append(target_ep.flush_cache.remote({}))
        return obj_refs

    def _wait_sampler_cache_flush_after_distributed_bucket(self, obj_refs):
        """Wait for sampler-side cache flush submitted after a large bucket."""
        ray.get(obj_refs)
        log("sampler cache flushed after large distributed weight bucket", rank=0)

    def _update_weights_by_distributed_vllm(self, sampler_idx, model_engine, replace_zeros=False):
        """Iterate ``export_weights()`` on all ranks and broadcast from rank 0.


        * 所有 rank 同步迭代 ``model_engine.export_weights()``，以驱动
          mbridge 内部的 TP/EP all-gather（只 rank0 迭代会在第一次
          collective 就死锁）。
        * rank0 把 yielded tensor 攒进 ``buffer``，按
          ``self.update_weight_max_size_bytes`` 切桶；每满一桶通过
          :meth:`_broadcast_weight_bucket_vllm` flatten 成 uint8 扁平
          tensor，经 :class:`PyNcclCommunicator` 广播给 vLLM worker；
          worker 端由 ``gcore_update_weights_distributed`` 接收并切 view
          喂给 ``model.load_weights``。
        * 所有 bucket 发完后 rank0 单独下发一次
          :meth:`VllmEngine.finalize_weights_update` RPC，让 worker
          端统一做 ``process_weights_after_loading`` 一次（避免 MoE
          expert 跨桶被部分 finalize 搞坏）。

        Parameters
        ----------
        sampler_idx : int
        model_engine : object
            Exposes ``export_weights()`` generator.
        replace_zeros : bool, optional

        Returns
        -------
        bool
        """
        is_rank0 = dist.get_rank() == 0
        if is_rank0:
            num_clusters = self.svr_cluster_num_per_sampler[sampler_idx]
            start_refs = []
            for ep_i in range(num_clusters):
                target_ep = self.rpc_client_lst[sampler_idx].get_target_endpoint(
                    sample_idx=None,
                    ep_idx=ep_i,
                )
                start_refs.append(target_ep.start_weights_update.remote({}))
            ray.get(start_refs)
        cpu_barrier()

        weight_generator = model_engine.export_weights()
        max_bucket_bytes = self.update_weight_max_size_bytes
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
                    self._broadcast_weight_bucket_vllm(sampler_idx, buffer)
                    count_buckets += 1
                buffer = []
                buffer_size = 0
                cpu_barrier()
                torch.cuda.empty_cache()

        if buffer_size > 0:
            if is_rank0:
                torch.cuda.synchronize()
                self._broadcast_weight_bucket_vllm(sampler_idx, buffer)
                count_buckets += 1
            cpu_barrier()
        del weight_generator
        clear_memory()

        # 所有 bucket 发完后单发一次 finalize RPC，worker 端跑
        # ``process_weights_after_loading``（对 MoE 尤其重要，避免 expert
        # 跨桶被部分初始化）。与 _update_weights_by_bucketed_ipc_vllm 尾部
        # 是同一个 RPC 入口。
        if is_rank0:
            num_clusters = self.svr_cluster_num_per_sampler[sampler_idx]
            fin_refs = []
            for ep_i in range(num_clusters):
                target_ep = self.rpc_client_lst[sampler_idx].get_target_endpoint(
                    sample_idx=None,
                    ep_idx=ep_i,
                )
                fin_refs.append(target_ep.finalize_weights_update.remote({}))
            ray.get(fin_refs)
        cpu_barrier()

        log(
            f"distributed weight update done (vllm), {count_buckets} buckets broadcast "
            f"(bucket={max_bucket_bytes} bytes)",
            rank=0,
        )
        return True

    def _broadcast_weight_bucket(self, sampler_idx, named_tensors, flush_cache=False):
        """Send metadata via Ray, broadcast tensor data via NCCL."""
        backend = self.infer_backend
        assert backend == "sglang", "only sglang backend is supported"

        names = [n for n, _ in named_tensors]
        dtypes = [str(p.dtype).replace("torch.", "") for _, p in named_tensors]
        shapes = [list(p.shape) for _, p in named_tensors]

        num_clusters = self.svr_cluster_num_per_sampler[sampler_idx]
        obj_refs = []
        for ep_i in range(num_clusters):
            target_ep = self.rpc_client_lst[sampler_idx].get_target_endpoint(
                sample_idx=None,
                ep_idx=ep_i,
            )
            obj_refs.append(
                target_ep.update_weights_from_distributed.remote(
                    {
                        "names": names,
                        "dtypes": dtypes,
                        "shapes": shapes,
                        "group_name": self._dist_weight_group_name,
                        "flush_cache": flush_cache,
                        "load_format": "flattened_bucket",
                    }
                )
            )

        bucket = SglFlatTensorBucket(named_tensors=named_tensors)
        flattened_tensor = bucket.get_flattened_tensor()

        handles = [
            dist.broadcast(flattened_tensor, src=0, group=self._dist_weight_group, async_op=True)
        ]
        for h in handles:
            h.wait()

        ray.get(obj_refs)

    def _broadcast_weight_bucket_vllm(self, sampler_idx, named_tensors):
        """Flatten one bucket to uint8 and broadcast via vLLM NCCL group.

        对称 :meth:`_broadcast_weight_bucket`，但是：

        * 目标是 vLLM worker，``self._dist_weight_group`` 是
          :class:`PyNcclCommunicator`（由
          :meth:`NCCLWeightTransferEngine.trainer_init` 建立），其
          ``broadcast`` 签名是 ``(tensor, src, stream=...)``，无
          ``async_op``；调度到当前 CUDA stream，后跟 stream sync。
        * 打平策略：单桶混 dtype、按字节序 ``copy_`` 到一个 uint8 flat
          tensor；接收端按 ``names/dtype_names/shapes`` 切 uint8 view 再
          ``view(dtype).view(shape)`` 还原。
        * 接收端 RPC 落到
          :meth:`GCoreVllmWorkerExtension.gcore_update_weights_distributed`
          （经 :meth:`VllmEngine.update_weights_from_distributed` 派发）。

        Parameters
        ----------
        sampler_idx : int
        named_tensors : list[tuple[str, torch.Tensor]]
            rank0 侧本桶待广播的权重。必须是 contiguous、CUDA 上的 tensor。
        """
        backend = self.infer_backend
        assert backend == "vllm", "only vllm backend is supported"

        names: list = []
        dtype_names: list = []
        shapes: list = []
        byte_sizes: list = []
        for n, t in named_tensors:
            names.append(n)
            dtype_names.append(str(t.dtype).split(".")[-1])
            shapes.append(list(t.shape))
            byte_sizes.append(t.element_size() * t.numel())
        total_bytes = sum(byte_sizes)

        flat = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")
        offset = 0
        for (_, t), nbytes in zip(named_tensors, byte_sizes):
            flat[offset:offset + nbytes].copy_(t.detach().contiguous().view(-1).view(torch.uint8))
            offset += nbytes

        update_info = {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "total_bytes": total_bytes,
            "group_name": self._dist_weight_group_name,
        }

        num_clusters = self.svr_cluster_num_per_sampler[sampler_idx]
        obj_refs = []
        for ep_i in range(num_clusters):
            target_ep = self.rpc_client_lst[sampler_idx].get_target_endpoint(
                sample_idx=None,
                ep_idx=ep_i,
            )
            obj_refs.append(target_ep.update_weights_from_distributed.remote(update_info))

        self._dist_weight_group.broadcast(flat, src=0, stream=torch.cuda.current_stream())
        # PyNcclCommunicator.broadcast 只是把 NCCL op enqueue 到 stream，
        # ray.get 只同步 CPU；显式 sync 一次确保下一轮 buffer 重建前 flat
        # 已经发出去。
        torch.cuda.current_stream().synchronize()

        ray.get(obj_refs)

    def _get_sampler_engine_gpu_count(self, sampler_idx=0):
        """Return the number of GPU workers per sglang engine."""
        sampler_config = self.config.sampler
        infer_engine_config = sampler_config.infer_engine_configs[sampler_idx]
        dc = infer_engine_config.dist_config
        return min(
            dc.tensor_model_parallel_size * dc.pipeline_model_parallel_size,
            dc.num_gpus_per_node,
        )


class TestFuncMixin:
    """Mixin providing test / debug helper methods for sampler clients."""
    async def test_generate(self, sampler_idx):
        """Run a quick generation test against the sampler.

        Parameters
        ----------
        sampler_idx : int
        """
        # test data
        prompts = [
            "Hello, what is your name?",
            "Who is the president of the United States?",
            "What is the capital of France?",
            "What is the future of AI?",
        ]
        req_dict = {
            "prompts": prompts,
        }

        if dist.get_rank() == 0:
            resp = await self._batch_rpc_call(sampler_idx, 'test_generate', req_dict)
            logging_rank0(f"test_generate resp: {resp}")

    async def test_save_engine_ckpt(self, sampler_idx, save_path):
        """Ask the sampler to save an engine checkpoint.

        Parameters
        ----------
        sampler_idx : int
        save_path : str
        """
        req_dict = {"save_ckpt_dir": f"{save_path}"}

        if dist.get_rank() == 0:
            resp = await self._batch_rpc_call(sampler_idx, 'save_engine_ckpt', req_dict)
            logging_rank0(f"test_save_engine_ckpt resp: {resp}")
