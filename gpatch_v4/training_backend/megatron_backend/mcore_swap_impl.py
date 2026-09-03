import torch

from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer.optimizer import ChainedOptimizer

from gpatch_v4.core.device import sync_param_offload
from gpatch_v4.utils.common_utils import (
    clear_memory,
    log,
    logging_memory_usage,
    logging_memory_usage_details,
)


def _prepare_models_for_te_offload(models) -> int:
    # TE may pin/release extra state before host offload; skip when API is absent.
    try:
        from transformer_engine.pytorch import prepare_model_for_offload
    except ImportError:
        return 0
    return prepare_model_for_offload(models)


class McoreSwapImpl:
    def __init__(self, early_swap_model=False):
        self.early_swap_model = early_swap_model

    @torch.no_grad()
    def offload_model(self, models, tag=""):
        """Offload model params/grads to CPU.

        Megatron layout: bf16 param + fp32 grad chunked in MP group;
        fp32 main_param + optimizer state chunked in MP+DP groups.
        """
        clear_memory()
        logging_memory_usage_details(f"memory tracking before {tag}model offload", rank=0)
        if models is None:
            return

        sync_offload = sync_param_offload()
        offload_fn = (
            rebind_offload_tensor_to_cpu
            if sync_offload else offload_tensor_to_cpu
        )

        state_bytes_before = torch.cuda.memory_allocated()
        prepared_parameter_count = _prepare_models_for_te_offload(models)
        if prepared_parameter_count > 0:
            released_state_bytes = max(
                0, state_bytes_before - torch.cuda.memory_allocated()
            )
            log(
                "memory tracking Transformer Engine prepared "
                f"{prepared_parameter_count} parameters for offload, released "
                f"{released_state_bytes / (1024**3):.3f} GB allocated",
                rank=0,
            )

        staged = []
        for model_chunk in models:
            if isinstance(model_chunk, DDP):
                model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
                for buffers in model_chunk_all_buffers:
                    for buffer in buffers:
                        offload_tensor_to_cpu(buffer.param_data, non_blocking=not sync_offload)
                        staged.append(buffer.param_data)
                        if self.early_swap_model:
                            _attach_param_cpu_views(buffer)
                        release_tensor_mem(buffer.grad_data)
                for _, param in model_chunk.module.named_parameters():
                    if not param.requires_grad:
                        assert not param._is_view()
                        offload_tensor_to_cpu(param, non_blocking=not sync_offload)
                        staged.append(param)
            else:
                for _, param in model_chunk.named_parameters():
                    offload_fn(param)
                    if not sync_offload:
                        staged.append(param)
                    if param.grad is not None:
                        offload_fn(param.grad)
                        if not sync_offload:
                            staged.append(param.grad)
                for _, buf in model_chunk.named_buffers():
                    offload_fn(buf)
                    if not sync_offload:
                        staged.append(buf)

        # D2H is async; one sync then free all GPU storages (avoid per-tensor sync).
        sync_and_resize_offloaded_tensors(staged)

        cleared_bytes = clear_cached_gpu_tensors(models)
        log(
            f"memory tracking cleared {cleared_bytes / (1024**3):.3f} GB cached GPU tensors",
            rank=0,
        )
        clear_memory()
        logging_memory_usage_details(f"memory tracking after {tag}model offload", rank=0)

    @torch.no_grad()
    def release_grad(self, models):
        """Release FP32 grad_data to free GPU memory without offloading model params."""
        clear_memory()
        logging_memory_usage_details("memory tracking before release grad", rank=0)
        if models is None:
            return
        for model_chunk in models:
            if isinstance(model_chunk, DDP):
                model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
                for buffers in model_chunk_all_buffers:
                    for buffer in buffers:
                        release_tensor_mem(buffer.grad_data)
        clear_memory()
        logging_memory_usage_details("memory tracking after release grad", rank=0)

    @torch.no_grad()
    def onload_model(self, models, onload_grad=True, tag=""):
        """Onload model params/grads back to GPU.

        Megatron layout: bf16 param + fp32 grad chunked in MP group;
        fp32 main_param + optimizer state chunked in MP+DP groups.
        """
        clear_memory()
        logging_memory_usage_details(f"memory tracking before {tag}model onload", rank=0)
        if models is None:
            return
        sync_offload = sync_param_offload()
        onload_fn = (
            rebind_onload_tensor_to_gpu
            if sync_offload else onload_tensor_to_gpu
        )
        for model_chunk in models:
            if isinstance(model_chunk, DDP):
                model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
                for buffers in model_chunk_all_buffers:
                    for buffer in buffers:
                        # sometimes, we don't want to load grad for pure inference
                        if onload_grad:
                            recover_tensor_mem(buffer.grad_data)
                        onload_tensor_to_gpu(buffer.param_data)
                for _, param in model_chunk.module.named_parameters():
                    if not param.requires_grad:
                        assert not param._is_view()
                        onload_tensor_to_gpu(param)
            else:
                # we need this for ref module
                for _, param in model_chunk.named_parameters():
                    onload_fn(param)
                    if param.grad is not None:
                        onload_fn(param.grad)
                for _, buf in model_chunk.named_buffers():
                    onload_fn(buf)
        clear_memory()
        logging_memory_usage_details(f"memory tracking after {tag}model onload", rank=0)

    @torch.no_grad()
    def offload_optimizer(self, optimizers):
        clear_memory()
        logging_memory_usage_details("memory tracking before optimizer offload", rank=0)
        if optimizers is None:
            return
        optimizer_lst = []
        if isinstance(optimizers, ChainedOptimizer):
            chained_optimizers = optimizers.chained_optimizers
            optimizer_lst.extend(chained_optimizers)
        else:
            optimizer_lst.append(optimizers)

        sync_offload = sync_param_offload()
        staged = []
        for optimizer in optimizer_lst:
            staged.extend(offload_megatron_copy_params(optimizer, non_blocking=not sync_offload))
            opt_state_dict_values = optimizer.optimizer.state.values()

            for v in opt_state_dict_values:
                offload_tensor_to_cpu(v.get("exp_avg"), non_blocking=not sync_offload)
                staged.append(v.get("exp_avg"))
                offload_tensor_to_cpu(v.get("exp_avg_sq"), non_blocking=not sync_offload)
                staged.append(v.get("exp_avg_sq"))
        sync_and_resize_offloaded_tensors(staged)

        clear_memory()
        logging_memory_usage_details("memory tracking after optimizer offload", rank=0)

    @torch.no_grad()
    def onload_optimizer(self, optimizers):
        clear_memory()
        logging_memory_usage_details("memory tracking before optimizer onload", rank=0)
        if optimizers is None:
            return
        optimizer_lst = []
        if isinstance(optimizers, ChainedOptimizer):
            chained_optimizers = optimizers.chained_optimizers
            optimizer_lst.extend(chained_optimizers)
        else:
            optimizer_lst.append(optimizers)

        for optimizer in optimizer_lst:
            onload_megatron_copy_params(optimizer)

            opt_state_dict_values = optimizer.optimizer.state.values()
            for v in opt_state_dict_values:
                onload_tensor_to_gpu(v.get("exp_avg"))
                onload_tensor_to_gpu(v.get("exp_avg_sq"))

        clear_memory()
        logging_memory_usage_details("memory tracking after optimizer onload", rank=0)


# some helper functions


def _attach_param_cpu_views(buffer):
    """Expose DDP-buffer CPU slices on their original Parameters.

    DDP parameters are views into ``buffer.param_data``. Export needs a
    per-parameter CPU source so it can materialize only the current mbridge
    bucket instead of restoring the whole DDP buffer.
    """
    if buffer.param_data is None:
        return
    cpu_buffer = buffer.param_data.gcore_cpu_data
    for param, (start, end, _) in buffer.param_index_map.items():
        cpu_view = cpu_buffer[start:end]
        assert cpu_view.numel() == param.numel(
        ), ("early swap does not support packed DDP parameters")
        param.gcore_cpu_data = cpu_view.view(param.shape)
        param.gcore_untyped_storage_data_size = (param.numel() * param.element_size())
        if not hasattr(param, "gcore_ddp_param_data"):
            param.gcore_ddp_param_data = param.data


class _CheckTensorAttr:
    run_once_flag = False

    @staticmethod
    def check_attr():
        if not _CheckTensorAttr.run_once_flag:
            device_id = torch.cuda.current_device()
            tmp_tensor = torch.rand((10), device=device_id)
            assert not hasattr(tmp_tensor, "gcore_cpu_data")
            assert not hasattr(tmp_tensor, "gcore_untyped_storage_data_size")
            _CheckTensorAttr.run_once_flag = True


def offload_tensor_to_cpu(tensor, non_blocking=True):
    _CheckTensorAttr.check_attr()
    if tensor is None:
        return
    assert isinstance(tensor, torch.Tensor), f"{tensor=} type must be torch.Tensor"
    if not hasattr(tensor, "gcore_cpu_data"):
        setattr(tensor, "gcore_cpu_data", tensor.data.to("cpu", non_blocking=non_blocking))
    else:
        assert tensor.gcore_cpu_data.shape == tensor.data.shape
        assert tensor.gcore_cpu_data.dtype == tensor.data.dtype
        tensor.gcore_cpu_data.copy_(tensor.data, non_blocking=True)
    tensor.gcore_untyped_storage_data_size = tensor.untyped_storage().size()
    assert tensor.gcore_untyped_storage_data_size == tensor.gcore_cpu_data.untyped_storage().size()


def resize_offloaded_tensor_gpu(tensor):
    """Free GPU storage after D2H for ``tensor`` has completed."""
    if tensor is None:
        return
    if tensor.untyped_storage().size() > 0:
        tensor.untyped_storage().resize_(0)


def sync_and_resize_offloaded_tensors(tensors):
    """One stream sync, then ``resize_(0)`` for every staged offload tensor."""
    if any(tensor is not None and tensor.untyped_storage().size() > 0 for tensor in tensors):
        torch.cuda.current_stream().synchronize()
    for tensor in tensors:
        resize_offloaded_tensor_gpu(tensor)


def _can_safely_resize_storage(tensor: torch.Tensor) -> bool:
    return (
        tensor.untyped_storage().size() == tensor.numel() * tensor.element_size()
        and tensor.storage_offset() == 0
        and tensor.is_contiguous()
    )


def rebind_offload_tensor_to_cpu(tensor: torch.Tensor | None) -> None:
    if tensor is None:
        return
    assert isinstance(tensor, torch.Tensor), f"{tensor=} type must be torch.Tensor"
    if tensor.device.type == "cpu":
        return
    old_data = tensor.data
    tensor.data = old_data.to("cpu", non_blocking=False)
    if _can_safely_resize_storage(old_data):
        old_data.untyped_storage().resize_(0)


def rebind_onload_tensor_to_gpu(tensor: torch.Tensor | None) -> None:
    if tensor is None:
        return
    assert isinstance(tensor, torch.Tensor), f"{tensor=} type must be torch.Tensor"
    if tensor.device.type != "cpu":
        return
    tensor.data = tensor.data.to(torch.cuda.current_device(), non_blocking=False)


def onload_tensor_to_gpu(tensor):
    if tensor is None:
        return
    tensor.data.untyped_storage().resize_(tensor.gcore_untyped_storage_data_size)
    tensor.data.copy_(tensor.gcore_cpu_data, non_blocking=True)


def release_tensor_mem(tensor):
    _CheckTensorAttr.check_attr()
    if tensor is None:
        return
    current_size = tensor.untyped_storage().size()
    if current_size == 0:
        return
    tensor.gcore_untyped_storage_data_size = current_size
    tensor.untyped_storage().resize_(0)


def recover_tensor_mem(tensor):
    if tensor is None:
        return
    tensor.data.untyped_storage().resize_(tensor.gcore_untyped_storage_data_size)


def copy_tensor_to_cpu(tensor):
    _CheckTensorAttr.check_attr()
    if tensor is None:
        return
    if not hasattr(tensor, "gcore_cpu_data"):
        setattr(tensor, "gcore_cpu_data", tensor.data.to("cpu", non_blocking=True))
    else:
        assert tensor.gcore_cpu_data.shape == tensor.data.shape
        assert tensor.gcore_cpu_data.dtype == tensor.data.dtype
        tensor.gcore_cpu_data.copy_(tensor.data, non_blocking=True)


def copy_tensor_to_gpu(tensor):
    if tensor is None:
        return
    tensor.data.copy_(tensor.gcore_cpu_data, non_blocking=True)


_DISPATCHER_CACHED_ATTRS = (
    "probs",
    "routing_map",
    "reversed_local_input_permutation_mapping",
)

_FLEX_DISPATCH_MANAGER_CACHED_ATTRS = (
    "routing_map",
    "token_probs",
    "token_indices",
    "dispatched_probs",
    "dispatched_indices",
    "tokens_per_expert",
    "dispatched_routing_map",
    "reversed_mapping_for_combine",
    "pad_offsets",
    "num_permuted_tokens",
)


def _clear_cuda_tensor_attrs(owner, attr_names):
    """Clear direct CUDA tensor attributes and return their logical size."""
    cleared_bytes = 0
    for attr_name in attr_names:
        val = getattr(owner, attr_name, None)
        if isinstance(val, torch.Tensor) and val.is_cuda:
            cleared_bytes += val.nelement() * val.element_size()
            setattr(owner, attr_name, None)
    return cleared_bytes


def clear_cached_gpu_tensors(models):
    """Clear GPU tensors cached by MoE token dispatchers.

    During forward, each ``MoEAlltoAllTokenDispatcher`` stores ``probs``,
    ``routing_map`` and ``reversed_local_input_permutation_mapping`` as
    plain instance attrs. These survive offload and accumulate to several
    GB across MoE layers; we set them to ``None`` so ``empty_cache()`` can
    reclaim the GPU memory.

    ``MoEFlexTokenDispatcher`` stores equivalent routing and permutation
    metadata on its nested DeepEP/HybridEP communication manager. Clear
    those tensors as well after training has completed.

    Token dispatchers are NOT ``nn.Module`` instances; locate them via
    ``MoELayer.token_dispatcher`` since ``MoELayer`` is.

    Returns
    -------
    int
        Total bytes cleared.
    """
    from megatron.core.transformer.moe.moe_layer import MoELayer

    if models is None:
        return 0

    total_bytes = 0
    for model_chunk in models:
        root = model_chunk.module if isinstance(model_chunk, DDP) else model_chunk
        for module in root.modules():
            if not isinstance(module, MoELayer):
                continue
            dispatcher = getattr(module, "token_dispatcher", None)
            if dispatcher is None:
                continue
            total_bytes += _clear_cuda_tensor_attrs(dispatcher, _DISPATCHER_CACHED_ATTRS)

            comm_manager = getattr(dispatcher, "_comm_manager", None)
            if comm_manager is not None:
                total_bytes += _clear_cuda_tensor_attrs(
                    comm_manager, _FLEX_DISPATCH_MANAGER_CACHED_ATTRS
                )
    return total_bytes


@torch.no_grad()
def offload_megatron_copy_params(optimizers, non_blocking=True):
    """Offload optimizer parameters to CPU (D2H only; caller frees GPU storage).

    Args:
        optimizers:
        non_blocking: forwarded to ``offload_tensor_to_cpu``.

    Returns
    -------
    list
        Tensors that were staged and still need ``sync_and_resize_offloaded_tensors``.
    """
    staged = []

    def offload_group_to_cpu(group):
        if group is None:
            return

        if isinstance(group, list):
            for param_group in group:
                if isinstance(param_group, list):
                    for param in param_group:
                        offload_tensor_to_cpu(param, non_blocking=non_blocking)
                        staged.append(param)
                else:
                    offload_tensor_to_cpu(param_group, non_blocking=non_blocking)
                    staged.append(param_group)
        else:
            offload_tensor_to_cpu(group, non_blocking=non_blocking)
            staged.append(group)

    # Offload all parameter groups to CPU

    if hasattr(optimizers, 'shard_fp32_from_float16_groups'):
        offload_group_to_cpu(getattr(optimizers, 'shard_fp32_from_float16_groups'))
    return staged


@torch.no_grad()
def onload_megatron_copy_params(optimizers):
    """Load optimizer parameters back to GPU.

    Args:
        optimizers:
    """
    def load_group_to_gpu(group):
        if group is None:
            return

        if isinstance(group, list):
            for param_group in group:
                if isinstance(param_group, list):
                    for param in param_group:
                        onload_tensor_to_gpu(param)
                else:
                    onload_tensor_to_gpu(param_group)
        else:
            onload_tensor_to_gpu(group)

    # Load all parameter groups to GPU

    if hasattr(optimizers, 'shard_fp32_from_float16_groups'):
        load_group_to_gpu(getattr(optimizers, 'shard_fp32_from_float16_groups'))
