import types
import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

from megatron.core import mpu
from megatron.core.parallel_state import (
    get_model_parallel_group,
    get_model_parallel_src_rank,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_last_rank,
    get_pipeline_model_parallel_world_size,
)

from gpatch_v4.core.parallel_state import (
    get_model_and_context_parallel_group,
    get_model_and_context_parallel_src_rank,
    is_mp_and_cp_head,
)
from gpatch_v4.utils import clear_memory, log


def recursive_clone_tensor(obj: Any) -> Any:
    """Clone view tensors to avoid redundant storage copies during broadcast.

    Parameters
    ----------
    obj : Any
        Supports nested dict / list.

    Returns
    -------
    Any
    """
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            obj[i] = recursive_clone_tensor(item)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = recursive_clone_tensor(value)
    elif isinstance(obj, torch.Tensor):
        if obj.numel() * obj.element_size() != obj.untyped_storage().size():
            obj = obj.detach().clone()
    else:
        type_name = type(obj).__name__
        if type_name not in ['NoneType', 'int', 'float', 'str', 'bool']:
            warnings.warn(
                f"run_inference: Unknown type {type_name} in rollout batches, " \
                "skip traversing tensor. Risk of abnormal memory growth caused" \
                " by copying tensor storage when broadcasting rollout batches."
            )
    return obj


def _is_cuda(tensor):
    """Check if a tensor is not none and is cuda."""
    assert tensor is not None
    assert tensor.is_cuda


def _is_cuda_contiguous(tensor):
    """Check if a tensor is not none, is cuda, and is contiguous."""
    _is_cuda(tensor)
    assert tensor.is_contiguous()


def _send_and_recv_from_last_to_first_pipeline_stage(tensor=None):
    is_last_stage = mpu.is_pipeline_last_stage()
    is_first_stage = mpu.is_pipeline_first_stage()

    if is_last_stage or is_first_stage:
        if is_first_stage:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                tensor,
                mpu.get_pipeline_model_parallel_last_rank(),
                group=get_pipeline_model_parallel_group()
            )
            reqs = torch.distributed.batch_isend_irecv([recv_prev_op])
        elif is_last_stage:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend,
                tensor,
                mpu.get_pipeline_model_parallel_first_rank(),
                group=get_pipeline_model_parallel_group()
            )
            reqs = torch.distributed.batch_isend_irecv([send_next_op])

        for req in reqs:
            req.wait()
        # To protect against race condition when using batch_isend_irecv().
        torch.cuda.synchronize()

        return tensor


def _send_and_recv_from_first_to_last_pipeline_stage(tensor=None):
    is_last_stage = mpu.is_pipeline_last_stage()
    is_first_stage = mpu.is_pipeline_first_stage()

    if is_last_stage or is_first_stage:
        if is_last_stage:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                tensor,
                mpu.get_pipeline_model_parallel_first_rank(),
                group=get_pipeline_model_parallel_group()
            )
            reqs = torch.distributed.batch_isend_irecv([recv_prev_op])
        elif is_first_stage:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend,
                tensor,
                mpu.get_pipeline_model_parallel_last_rank(),
                group=get_pipeline_model_parallel_group()
            )
            reqs = torch.distributed.batch_isend_irecv([send_next_op])

        for req in reqs:
            req.wait()
        # To protect against race condition when using batch_isend_irecv().
        torch.cuda.synchronize()

        return tensor


def average_losses_across_data_parallel_group(losses, use_gloo=False):
    """All-reduce and average a list of scalar losses across DP ranks.

    Parameters
    ----------
    losses : list of torch.Tensor
    use_gloo : bool, optional

    Returns
    -------
    torch.Tensor
    """
    group = mpu.get_data_parallel_group()
    if use_gloo:
        group = mpu.get_data_parallel_group_gloo()

    averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses, group=group)
    averaged_losses = averaged_losses / torch.distributed.get_world_size(group=group)
    return averaged_losses


def allreduce_loss_across_data_parallel_group(
    losses, use_gloo=False, op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM
):
    """All-reduce a list of scalar losses across DP ranks (no averaging).

    Parameters
    ----------
    losses : list of torch.Tensor
    use_gloo : bool, optional
    op : torch.distributed.ReduceOp, optional

    Returns
    -------
    torch.Tensor
    """
    group = mpu.get_data_parallel_group()
    if use_gloo:
        group = mpu.get_data_parallel_group_gloo()

    averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses, group=group, op=op)
    return averaged_losses


REDUCE_AVG = "avg"
REDUCE_MIN = "min"
REDUCE_MAX = "max"

DIST_REDUCE_OP = {
    REDUCE_MIN: dist.ReduceOp.MIN,
    REDUCE_MAX: dist.ReduceOp.MAX,
}


def infer_reduce_op_by_key_name(key: str) -> str:
    """Infer the DP reduce operation from metric key naming convention.

    The key is split into word-level tokens (on ``_`` and ``/``).
    If any token is exactly ``"min"`` the metric is reduced with
    ``ReduceOp.MIN``; ``"max"`` triggers ``ReduceOp.MAX``; everything
    else uses average (all-reduce sum / world-size).
    """
    tokens = set(key.replace("/", "_").split("_"))
    if "min" in tokens:
        return REDUCE_MIN
    if "max" in tokens:
        return REDUCE_MAX
    return REDUCE_AVG


def reduce_metrics_across_data_parallel_group(
    metrics: Dict[str, torch.Tensor],
    reduce_overrides: Optional[Dict[str, str]] = None,
) -> None:
    """Reduce all 0-dim tensor metrics in ``metrics`` across the DP group.

    Only 0-dim CUDA tensors are reduced; multi-dim tensors and Python
    scalars are left untouched. The reduce op is auto-inferred from
    ``infer_reduce_op_by_key_name`` and can be overridden per-key via
    ``reduce_overrides``.

    Parameters
    ----------
    metrics : dict
        Modified **in-place**.
    reduce_overrides : dict, optional
        ``{key: "avg"|"min"|"max"}``.
    """
    dp_group = mpu.get_data_parallel_group()
    dp_size = dp_group.size()

    buckets: Dict[str, list] = {REDUCE_AVG: [], REDUCE_MIN: [], REDUCE_MAX: []}
    for key, val in metrics.items():
        if not (isinstance(val, torch.Tensor) and val.dim() == 0):
            continue
        op = (reduce_overrides or {}).get(key, infer_reduce_op_by_key_name(key))
        buckets[op].append(key)

    for op, keys in buckets.items():
        if not keys:
            continue
        packed = torch.cat([metrics[k].clone().detach().float().view(1) for k in keys])
        if op == REDUCE_AVG:
            dist.all_reduce(packed, group=dp_group)
            packed /= dp_size
        else:
            dist.all_reduce(packed, group=dp_group, op=DIST_REDUCE_OP[op])
        for i, k in enumerate(keys):
            metrics[k] = packed[i]


class _AllReduce(torch.autograd.Function):
    """Implementation from old PyTorch `torch.distributed.nn.parallel`."""
    @staticmethod
    def forward(ctx, op, group, tensor):
        ctx.group, ctx.op = group, op
        tensor = tensor.clone()
        torch.distributed.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, _AllReduce.apply(ctx.op, ctx.group, grad_output))


def all_reduce_autograd(
    tensor, op=torch.distributed.ReduceOp.SUM, group=torch.distributed.group.WORLD
):
    """Differentiable all-reduce for gradients that differ per rank.

    Parameters
    ----------
    tensor : torch.Tensor
    op : torch.distributed.ReduceOp, optional
    group : ProcessGroup, optional

    Returns
    -------
    torch.Tensor
    """
    return _AllReduce.apply(op, group, tensor)


class BroadcastUtils:
    """Utility methods for broadcasting objects and tensors across parallel groups."""
    @staticmethod
    def broadcast_object_within_mp(obj: Any, make_recursive_clone_in_case_of_view=False):
        group = get_model_parallel_group()
        if torch.distributed.get_world_size(group) > 1:
            if make_recursive_clone_in_case_of_view:
                obj = recursive_clone_tensor(obj)
                clear_memory()
            obj_list = [obj]
            torch.distributed.broadcast_object_list(
                obj_list,
                src=get_model_parallel_src_rank(),
                group=group,
            )
            log(f"memory tracking after bcast obj_list", rank=0)
            return obj_list[0]
        else:
            return obj

    @staticmethod
    def broadcast_object_within_mp_and_cp(obj: Any, make_recursive_clone_in_case_of_view=False):
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            return BroadcastUtils.broadcast_object_within_mp(
                obj, make_recursive_clone_in_case_of_view
            )

        group = get_model_and_context_parallel_group()
        if torch.distributed.get_world_size(group) > 1:
            if make_recursive_clone_in_case_of_view:
                obj = recursive_clone_tensor(obj)
                clear_memory()
            obj_list = [obj]
            torch.distributed.broadcast_object_list(
                obj_list,
                src=get_model_and_context_parallel_src_rank(),
                group=group,
            )
            return obj_list[0]

        else:
            return obj

    @staticmethod
    def broadcast_rollout_batch(
        rbs, remove_before_broadcast_func=None, add_back_after_broadcast_func=None
    ):
        if is_mp_and_cp_head() and remove_before_broadcast_func is not None:
            assert isinstance(remove_before_broadcast_func, types.FunctionType)
            rbs = remove_before_broadcast_func(rbs)
        output = BroadcastUtils.broadcast_object_within_mp_and_cp(
            rbs, make_recursive_clone_in_case_of_view=True
        )
        if is_mp_and_cp_head() and add_back_after_broadcast_func is not None:
            assert isinstance(add_back_after_broadcast_func, types.FunctionType)
            output = add_back_after_broadcast_func(output)
        return output

    @staticmethod
    def broadcast_2d_tensor(tensor, src, group, dtype=torch.float32):
        """Broadcast any 2d tensor from the src rank to every other rank in the given group.
        All the ranks that send or receive data must call this function."""
        if torch.distributed.get_rank() == src:
            if tensor is None:
                input_info = [1, 0, 0]
            else:
                assert tensor.ndim == 2, f"tensor dims is not 2 but is {tensor.ndim} with shape {tensor.shape}"
                tensor = tensor.cuda().to(dtype)
                input_info = [0, tensor.size(0), tensor.size(1)]
            input_info_tensor = torch.tensor(
                input_info, dtype=torch.float32, device=torch.cuda.current_device()
            )

            torch.distributed.broadcast(input_info_tensor, src, group)
            if tensor is not None:
                torch.distributed.broadcast(tensor, src, group)
        else:
            input_info_tensor = torch.empty(
                3, dtype=torch.float32, device=torch.cuda.current_device()
            )
            torch.distributed.broadcast(input_info_tensor, src, group)

            is_none = bool(input_info_tensor[0].item())
            dim1 = int(input_info_tensor[1].item())
            dim2 = int(input_info_tensor[2].item())

            if not is_none:
                tensor = torch.empty(dim1, dim2, dtype=dtype, device=torch.cuda.current_device())
                torch.distributed.broadcast(tensor, src, group)
        return tensor

    @staticmethod
    def broadcast_2d_tensor_within_pp(tensor, dtype=torch.float32):
        if get_pipeline_model_parallel_world_size() > 1:
            return BroadcastUtils.broadcast_2d_tensor(
                tensor,
                get_pipeline_model_parallel_last_rank(),
                get_pipeline_model_parallel_group(),
                dtype=dtype,
            )
        else:
            return tensor.to(dtype) if tensor is not None else None

    @staticmethod
    def broadcast_from_last_to_first_pipeline_stage(size, dtype, tensor=None):
        # copying from megatron/inference/text_generation/communication.py
        """Broadcast tensor values from last stage into the first stage."""

        is_last_stage = mpu.is_pipeline_last_stage()
        is_first_stage = mpu.is_pipeline_first_stage()
        # If first stage and last state are the same, then there is no
        # pipeline parallelism and no need to communicate.
        if is_first_stage and is_last_stage:
            return tensor
        # Only first and last stage pipeline stages need to be involved.
        if is_last_stage or is_first_stage:
            if is_last_stage:
                _is_cuda_contiguous(tensor)
            else:
                tensor = torch.empty(size, dtype=dtype, device=torch.cuda.current_device())
            tensor = _send_and_recv_from_last_to_first_pipeline_stage(tensor)
        else:
            tensor = None

        return tensor

    @staticmethod
    def broadcast_from_first_to_last_pipeline_stage(size, dtype, tensor=None):
        is_last_stage = mpu.is_pipeline_last_stage()
        is_first_stage = mpu.is_pipeline_first_stage()
        # If first stage and last state are the same, then there is no
        # pipeline parallelism and no need to communicate.
        if is_first_stage and is_last_stage:
            return tensor
        # Only first and last stage pipeline stages need to be involved.
        if is_last_stage or is_first_stage:
            if is_first_stage:
                _is_cuda_contiguous(tensor)
            else:
                tensor = torch.empty(size, dtype=dtype, device=torch.cuda.current_device())
            tensor = _send_and_recv_from_first_to_last_pipeline_stage(tensor)
        else:
            tensor = None

    @staticmethod
    def broadcast_tensor_between_cp_group(shape_meta, dtype, tensor, group):
        curr_rank = dist.get_rank(group=group)
        source_rank = dist.get_process_group_ranks(group)[0]

        batch_size, seq_len, vocab_size = shape_meta

        if curr_rank == 0:
            assert tensor is not None
            tensor = tensor.contiguous()
        else:
            tensor = torch.empty(
                (batch_size, seq_len, vocab_size),
                dtype=dtype,
                device=torch.cuda.current_device(),
            )
        dist.broadcast(tensor, src=source_rank, group=group)
        return tensor

    @staticmethod
    def broadcast_object_within_pp(obj: Any, make_recursive_clone_in_case_of_view=False) -> Any:
        group = get_pipeline_model_parallel_group()

        if torch.distributed.get_world_size(group) > 1:
            if make_recursive_clone_in_case_of_view:
                obj = recursive_clone_tensor(obj)
                clear_memory()
            obj_list = [obj]
            torch.distributed.broadcast_object_list(
                obj_list,
                src=get_pipeline_model_parallel_last_rank(),
                group=group,
            )
            return obj_list[0]
        else:
            return obj
