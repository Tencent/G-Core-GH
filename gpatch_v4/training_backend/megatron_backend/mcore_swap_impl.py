import torch

from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer.optimizer import ChainedOptimizer

from gpatch_v4.utils.common_utils import (
    clear_memory,
    log,
    logging_memory_usage,
    logging_memory_usage_details,
)


class McoreSwapImpl:
    @classmethod
    @torch.no_grad()
    def offload_model(cls, models):
        """Offload model params/grads to CPU.

        Megatron layout: bf16 param + fp32 grad chunked in MP group;
        fp32 main_param + optimizer state chunked in MP+DP groups.
        """
        clear_memory()
        logging_memory_usage_details("memory tracking before model offload", rank=0)
        if models is None:
            return

        for model_chunk in models:
            if isinstance(model_chunk, DDP):
                model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
                for buffers in model_chunk_all_buffers:
                    for buffer in buffers:
                        offload_tensor_to_cpu(buffer.param_data)
                        release_tensor_mem(buffer.grad_data)
                for _, param in model_chunk.module.named_parameters():
                    if not param.requires_grad:
                        assert not param._is_view()
                        offload_tensor_to_cpu(param)
            else:
                for _, param in model_chunk.named_parameters():
                    offload_tensor_to_cpu(param)
                    if param.grad is not None:
                        offload_tensor_to_cpu(param.grad)
                for _, buf in model_chunk.named_buffers():
                    offload_tensor_to_cpu(buf)

        cleared_bytes = clear_cached_gpu_tensors(models)
        log(
            f"memory tracking cleared {cleared_bytes / (1024**3):.3f} GB cached GPU tensors",
            rank=0,
        )
        clear_memory()
        logging_memory_usage_details("memory tracking after model offload", rank=0)

    @classmethod
    @torch.no_grad()
    def release_grad(cls, models):
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

    @classmethod
    @torch.no_grad()
    def onload_model(cls, models, onload_grad=True):
        """Onload model params/grads back to GPU.

        Megatron layout: bf16 param + fp32 grad chunked in MP group;
        fp32 main_param + optimizer state chunked in MP+DP groups.
        """
        clear_memory()
        logging_memory_usage_details("memory tracking before model onload", rank=0)
        if models is None:
            return
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
                    onload_tensor_to_gpu(param)
                    if param.grad is not None:
                        onload_tensor_to_gpu(param.grad)
                for _, buf in model_chunk.named_buffers():
                    onload_tensor_to_gpu(buf)
        clear_memory()
        logging_memory_usage_details("memory tracking after model onload", rank=0)

    @classmethod
    @torch.no_grad()
    def offload_optimizer(cls, optimizers):
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

        for optimizer in optimizer_lst:
            offload_megatron_copy_params(optimizer)
            opt_state_dict_values = optimizer.optimizer.state.values()

            for v in opt_state_dict_values:
                offload_tensor_to_cpu(v.get("exp_avg"))
                offload_tensor_to_cpu(v.get("exp_avg_sq"))

        clear_memory()
        logging_memory_usage_details("memory tracking after optimizer offload", rank=0)

    @classmethod
    @torch.no_grad()
    def onload_optimizer(cls, optimizers):
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


def offload_tensor_to_cpu(tensor):
    _CheckTensorAttr.check_attr()
    if tensor is None:
        return
    assert isinstance(tensor, torch.Tensor), f"{tensor=} type must be torch.Tensor"
    if not hasattr(tensor, "gcore_cpu_data"):
        #  non_blocking=True 都是异步的，在 cpu 访问这个 tensor 需要先同步再访问
        setattr(tensor, "gcore_cpu_data", tensor.data.to("cpu", non_blocking=True))
    else:
        assert tensor.gcore_cpu_data.shape == tensor.data.shape
        assert tensor.gcore_cpu_data.dtype == tensor.data.dtype
        tensor.gcore_cpu_data.copy_(tensor.data, non_blocking=True)
    tensor.gcore_untyped_storage_data_size = tensor.untyped_storage().size()
    assert tensor.gcore_untyped_storage_data_size == tensor.gcore_cpu_data.untyped_storage().size()
    tensor.untyped_storage().resize_(0)


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


def clear_cached_gpu_tensors(models):
    """Clear GPU tensors cached by MoEAlltoAllTokenDispatcher.

    During forward, each ``MoEAlltoAllTokenDispatcher`` stores ``probs``,
    ``routing_map`` and ``reversed_local_input_permutation_mapping`` as
    plain instance attrs. These survive offload and accumulate to several
    GB across MoE layers; we set them to ``None`` so ``empty_cache()`` can
    reclaim the GPU memory.

    ``MoEAlltoAllTokenDispatcher`` is NOT an ``nn.Module``; we locate it
    via ``MoELayer.token_dispatcher`` since ``MoELayer`` is.

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
            for attr_name in _DISPATCHER_CACHED_ATTRS:
                val = getattr(dispatcher, attr_name, None)
                if isinstance(val, torch.Tensor) and val.is_cuda:
                    total_bytes += val.nelement() * val.element_size()
                    setattr(dispatcher, attr_name, None)
    return total_bytes


@torch.no_grad()
def offload_megatron_copy_params(optimizers):
    """Offload optimizer parameters to CPU.

    Args:
        optimizers:
    """
    def offload_group_to_cpu(group):
        if group is None:
            return

        if isinstance(group, list):
            for param_group in group:
                if isinstance(param_group, list):
                    for param in param_group:
                        offload_tensor_to_cpu(param)
                else:
                    offload_tensor_to_cpu(param_group)
        else:
            offload_tensor_to_cpu(group)

    # Offload all parameter groups to CPU

    if hasattr(optimizers, 'shard_fp32_from_float16_groups'):
        offload_group_to_cpu(getattr(optimizers, 'shard_fp32_from_float16_groups'))


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
