import dataclasses
import hashlib
import itertools
import os
import random
import traceback
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed
import torch.distributed as dist
from einops import rearrange
from torch import Tensor

from megatron.core import mpu, tensor_parallel

from gpatch_v4.core.mappings import all_gather_from_context_parallel_region
from gpatch_v4.core.parallel_state import (
    get_model_and_context_parallel_group,
    is_mp_and_cp_head,
)
from gpatch_v4.utils import BroadcastUtils, all_reduce_autograd, log, logging_rank0

try:
    from megatron.core.gcore_utils import (
        get_gathered_routing_info,  # only branch wxdev support
    )
    from gpatch_v4.kernel import linear_cross_entropy, set_linear_ce_backend
except (ImportError, Exception) as e:
    get_gathered_routing_info = None
    linear_cross_entropy = None
    set_linear_ce_backend = None


def move_to_device_if_tensor(device, item):
    if torch.is_tensor(item):
        item = item.to(device)
    return item


def apply_func_to_dict(func, dictionary):
    return {k: func(v) for k, v in dictionary.items()}


cuda_dict = partial(apply_func_to_dict, partial(move_to_device_if_tensor, "cuda"))
cpu_dict = partial(apply_func_to_dict, partial(move_to_device_if_tensor, "cpu"))


def extend_value_to_dict(data: dict, new_data: dict, prefix: str = ""):
    """Append values from ``new_data`` to lists in ``data`` (in-place).

    Args:
        data (Dict):
        new_data (Dict):

    Returns:
        None:
    """
    for key, val in new_data.items():
        new_key = f"{prefix}{key}"
        if new_key not in data:
            data[new_key] = []
        if not isinstance(val, list):
            assert isinstance(val, int) or isinstance(val, float) or isinstance(
                val, torch.Tensor
            ), f"Expected int or torch.Tensor, got {type(val)}"
            val = [val]
        data[new_key].extend(val)


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    """Reduce each list in ``metrics`` by mean / max / min based on the key suffix.

    Keys ending in ``_max`` use ``np.max``, ``_min`` use ``np.min``; otherwise mean.

    Args:
        metrics:

    Returns:
        A dictionary with the same keys but each list replaced by its reduced value.

    Example:
        >>> reduce_metrics({"loss": [1.0, 2.0, 3.0], "max_reward": [5.0, 8.0]})
        {"loss": 2.0, "max_reward": 8.0}
    """
    for key, val in metrics.items():
        if key.endswith('_max'):
            metrics[key] = np.max(val)
        elif key.endswith('_min'):
            metrics[key] = np.min(val)
        else:
            metrics[key] = np.mean(val)
    return metrics


def check_rollout_batch(rb: Dict[str, List[Any]]) -> bool:
    if not isinstance(rb, dict):
        log(f"Error2 {type(rb)}")
        return False
    batch_size = None
    for k, v in rb.items():
        if not isinstance(k, str):
            log(f"Error3 {k} {type(k)}")
            return False
        if not isinstance(v, list):
            # suppot dict of multiple rewards
            if isinstance(v, dict) and all(
                isinstance(e, dict) and "rewards" in e for e in v.values()
            ):
                check_rollout_batch({kk: vv["rewards"] for (kk, vv) in v.items()})
                continue
            else:
                log(f"Error4 {k} {type(v)}")
                return False
        if batch_size is None:
            batch_size = len(v)
        if batch_size != len(v):
            log(f"Error5 {k} {batch_size} {len(v)}")

            return False
        for e in v[1:]:
            if type(e) != type(v[0]):
                log(f"Error6 {k} {type(e)} {type(v[0])}")
                return False
        if torch.is_tensor(v[0]):
            for e in v:
                if not e.is_cpu:
                    log(f"Error7 {k} {e.is_cpu=}")
                    return False
    return True


def check_rollout_batches(rbs: List[Dict[str, List[Any]]]) -> bool:
    if not isinstance(rbs, list):
        log(f"Error1 {type(rbs)}")
        return False
    for rb in rbs:
        if not check_rollout_batch(rb):
            return False
    return True


def list_of_tensor_to_list(
    data: List[torch.Tensor],
    flatten: bool,
    to_float32: bool = False,
) -> List[Any]:
    res = []
    for datum in data:
        if to_float32:
            datum = datum.float()
        if datum.ndim == 0:
            res.append(datum.item())
        else:
            if flatten:
                res.extend(datum.flatten().tolist())
            else:
                res.append(datum.tolist())
    return res


def unbind_tensor_to_list(data: torch.Tensor, ) -> List[torch.Tensor]:
    '''Unbind ``data`` along dim 0 into a list of tensors.'''
    tensor_list = list(torch.unbind(data, dim=0))
    return tensor_list


def expand_rollout_batch(rollout_batch: Dict[str, Union[int, List[Any]]], ) -> List[Dict[str, Any]]:
    """Expand a batched dict into a list of single-sample dicts.

    Parameters
    ----------
    rollout_batch : dict[str, list]
        A dict whose values are equal-length lists, e.g.

        .. code-block:: python

            {'a': [1, 2], 'b': [obj1, obj2]}

    Returns
    -------
    list[dict[str, Any]]
        Per-sample dicts, e.g. ``[{'a': 1, 'b': obj1}, {'a': 2, 'b': obj2}]``.
    """
    batch_list = []
    for k, vs in rollout_batch.items():
        assert isinstance(vs, list), f"{k} {type(vs)}"
        if len(batch_list) == 0:
            batch_list = [{} for _ in range(len(vs))]
        assert len(vs) == len(batch_list)
        for i in range(len(vs)):
            batch_list[i][k] = vs[i]
    return batch_list


def expand_rollout_batches(
    rollout_batches: List[Dict[str, Union[int, List[Any]]]],
) -> List[Dict[str, Any]]:
    """Expand a list of batched dicts into a flat list of single-sample dicts.

    Parameters
    ----------
    rollout_batches : list[dict[str, list]]
        E.g. ``[{'a': [1, 2], 'b': [obj1, obj2]}, {'a': [3], 'b': [obj3]}]``.

    Returns
    -------
    list[dict[str, Any]]
        Flat list, e.g. ``[{'a': 1, 'b': obj1}, {'a': 2, 'b': obj2}, {'a': 3, 'b': obj3}]``.
    """
    ex_rollout_batches = []
    for rollout_batch in rollout_batches:
        ex_rollout_batches.extend(expand_rollout_batch(rollout_batch))
    return ex_rollout_batches


def get_max_seqlen_within_ep(seqlen: int):
    t_seqlen = torch.tensor([seqlen], dtype=torch.int, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        t_seqlen, op=torch.distributed.ReduceOp.MAX, group=mpu.get_expert_model_parallel_group()
    )
    return t_seqlen.item()


def get_max_seqlen_within_dp(seqlen: int):
    t_seqlen = torch.tensor([seqlen], dtype=torch.int, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        t_seqlen, op=torch.distributed.ReduceOp.MAX, group=mpu.get_data_parallel_group()
    )
    return t_seqlen.item()


def get_batches_max_seqlen(batches: List[Dict[str, Any]], pad_to_multi_of: int) -> int:
    max_token_len = max([e['tokens'].shape[-1] for e in batches])
    max_token_len = ((max_token_len + pad_to_multi_of - 1) // pad_to_multi_of) * pad_to_multi_of
    return max_token_len


def update_square_averaging_token_len(batches: List[Dict[str, Any]], max_seq_length: int):
    # Compute on the same device as labels (may be CPU for lazy-loaded dynamic CP batches).
    labels_device = batches[0]['labels'].device if batches else torch.device('cpu')
    square_averaging_weight = torch.tensor(0, dtype=torch.float, device=labels_device)
    for batch in batches:
        labels = batch['labels']
        if batch['tokens'].shape[-1] > max_seq_length:
            labels = labels[-max_seq_length:]
        effective_token_len = torch.clamp_min((labels != -100).sum(), 1)
        square_averaging_weight += effective_token_len / effective_token_len.sqrt()
    square_averaging_weight /= len(batches)
    assert square_averaging_weight.item() > 0
    for batch in batches:
        assert 'square_averaging_weight' not in batch
        batch['square_averaging_weight'] = square_averaging_weight


def get_iterator_k_split_list(batches: List[Dict[str, Any]], num_microbatches: int) -> Iterator:
    if num_microbatches == 0:
        assert len(batches) == 0, f"len(batches) = {len(batches)}"
        return itertools.chain([])
    assert len(
        batches
    ) % num_microbatches == 0, f"len(batches) = {len(batches)} {num_microbatches=}"
    mbs = len(batches) // num_microbatches
    microbatches = []
    for i in range(num_microbatches):
        microbatches.append(batches[i * mbs:(i + 1) * mbs])
    # 这个 itertools.chain 是多余的，等价于返回 microbatches，但是先不修改了。
    return itertools.chain(microbatches)


def get_k_split_list(batches: List[Dict[str, Any]], num_microbatches: int) -> Iterator:
    assert len(batches) % num_microbatches == 0
    mbs = len(batches) // num_microbatches
    microbatches = []
    for i in range(num_microbatches):
        microbatches.append(batches[i * mbs:(i + 1) * mbs])
    return microbatches


def get_ltor_masks_and_position_ids(
    data,
    eod_token,
    reset_position_ids,
    reset_attention_mask,
    eod_mask_loss,
    compute_attention_mask=True
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1

    attention_mask = None
    if compute_attention_mask:
        # create it on CPU to avoid GPU memory fragmentation
        attention_mask = torch.tril(
            torch.ones((att_mask_batch, seq_length, seq_length), dtype=torch.bool, device="cpu"),
        ).view(att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).repeat(micro_batch_size, 1)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indicies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = False
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= i + 1 - prev_index
                    prev_index = i + 1

    if compute_attention_mask:
        # move from CPU to GPU
        non_blocking = True
        attention_mask = attention_mask.to(data.device, non_blocking=non_blocking)
        # Convert to Megatron convention: True = masked (cannot attend)
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def get_tensor_on_this_cp_rank(val, seq_dim, key_name=None):
    if key_name is not None:
        if key_name == "attention_mask":
            assert seq_dim == 2
        else:
            assert seq_dim == 1

    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    assert cp_size >= 1
    if cp_size == 1 or val is None:
        return val

    val = val.view(
        *val.shape[0:seq_dim],
        2 * cp_size,
        val.shape[seq_dim] // (2 * cp_size),
        *val.shape[(seq_dim + 1):],
    )
    index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu",
                         pin_memory=True).cuda(non_blocking=True)
    val = val.index_select(seq_dim, index)
    val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])
    return val


def reorder_target_for_cp(target, seq_dim=1):
    cp_size = mpu.get_context_parallel_world_size()
    assert cp_size >= 1
    if cp_size == 1:
        return target

    target = target.view(
        *target.shape[0:seq_dim],
        2 * cp_size,
        target.shape[seq_dim] // (2 * cp_size),
        *target.shape[(seq_dim + 1):],
    )
    reordered_indices = []
    for rank in range(cp_size):
        reordered_indices.append(rank)
        reordered_indices.append(2 * cp_size - rank - 1)
    target = target[:, reordered_indices, :]
    target = target.view(*target.shape[0:seq_dim], -1, *target.shape[(seq_dim + 2):])
    return target


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Compute the mean of ``values`` over positions where ``mask > 0``."""
    values = torch.where(mask > 0, values, 0.0)
    return values.sum() / torch.clamp_min(mask.sum(), 1)


def masked_mean_list(values: List[Tensor], mask: List[Tensor], dim=None) -> Tensor:
    """Per-sample masked mean over a list of (values, mask) pairs."""
    res = []
    for v, m in zip(values, mask):
        res.append((v * m).sum(dim=dim) / torch.clamp_min(m.sum(dim=dim), 1))
    return torch.stack(res)


def masked_sum(values: Tensor, mask: Tensor) -> Tensor:
    """Sum of ``values`` over positions where ``mask > 0``."""
    values = torch.where(mask > 0, values, 0.0)
    return values.sum()


def masked_var(values: Tensor, mask: Tensor, unbiased=True) -> Tensor:
    """Variance of ``values`` over positions where ``mask > 0``."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_statistic(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Mean / min / max of ``values`` over positions where ``mask > 0``."""
    # 这里不能直接 values = values * mask，因为 0 也可能是 max/min
    tmp = values[mask.bool()]
    if tmp.numel() == 0:
        zero_tmp = torch.tensor(0, dtype=values.dtype, device=values.device)
        return zero_tmp, zero_tmp, zero_tmp

    return tmp.mean(), tmp.min(), tmp.max()


def masked_statistic(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Mean / min / max of ``values`` over positions where ``mask > 0``."""
    # 这里不能直接 values = values * mask，因为 0 也可能是 max/min
    tmp = values[mask.bool()]
    if tmp.numel() == 0:
        zero_tmp = torch.tensor(0, dtype=values.dtype, device=values.device)
        return zero_tmp, zero_tmp, zero_tmp

    return tmp.mean(), tmp.min(), tmp.max()


def masked_global_statistics(values: Tensor,
                             mask: Tensor,
                             key_name: str = None,
                             group=None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Global mean / uncorrected var / min / max under a mask.

    ``mask`` and ``values`` must have the same shape; mask is {0, 1} with 1
    marking entries to keep.
    """
    assert values.shape == mask.shape, f"mismatch {key_name=} ({values.shape} != {mask.shape})"
    values = values.to(device=torch.cuda.current_device())
    mask = mask.to(device=torch.cuda.current_device())

    values = values * mask
    max_min = torch.tensor(
        [values.max(), -values.min()], dtype=torch.float32, device=torch.cuda.current_device()
    )
    torch.distributed.all_reduce(max_min, group=group, op=torch.distributed.ReduceOp.MAX)
    max_v, min_v = max_min
    min_v = -min_v

    # Get global sum and count and calculate the global mean and variance
    sum_and_count = torch.tensor(
        [values.sum(), mask.sum()], dtype=torch.float32, device=torch.cuda.current_device()
    )
    torch.distributed.all_reduce(sum_and_count, group=group)
    global_sum, global_count = sum_and_count
    if global_count == 0:  # avoid division by zero
        global_count = torch.ones_like(global_count)
    global_mean = global_sum / global_count
    variance_summed = (
        (((values - global_mean)**2) *
         mask).sum().to(device=torch.cuda.current_device(), dtype=torch.float32)
    )

    torch.distributed.all_reduce(variance_summed, group=group)

    return global_mean, variance_summed / global_count, min_v, max_v


def masked_global_statistics_list(
    values: List[Tensor],
    mask: List[Tensor],
    key_name: str = None,
    group=None
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """List variant of :func:`masked_global_statistics`.

    Concatenates the per-sample tensors then delegates.
    """
    assert len(values) == len(mask), f"mismatch ({len(values)} != {len(mask)})"
    values = torch.cat([v.view(-1) for v in values])
    mask = torch.cat([m.view(-1) for m in mask])
    return masked_global_statistics(values, mask, key_name, group)


def masked_global_topk_threshold(
    values: Tensor,
    mask: Tensor,
    k_global: int,
    group=None,
) -> Tensor:
    """Global top-k threshold of masked ``values`` across ranks in ``group``.

    Returns the scalar ``tau`` equal to the ``k_global``-th largest value among
    all masked entries across every rank in ``group``. Selecting entries with
    ``value >= tau`` therefore yields the exact global top-``k_global`` set.

    The threshold is computed exactly without gathering all values: the global
    top-k is contained in the union of each rank's local top-k (any global
    top-k value has fewer than ``k_global`` values above it globally, hence
    fewer than ``k_global`` above it on its own rank). So the ``k_global``-th
    largest of the gathered per-rank local top-k is the exact global threshold.

    Parameters
    ----------
    values : Tensor
        Per-token values on the local shard (any shape).
    mask : Tensor
        Same shape as ``values``; nonzero marks valid entries.
    k_global : int
        Global number of top entries to keep. Must be >= 1.
    group : optional
        Process group to reduce over.

    Returns
    -------
    Tensor
        Scalar threshold ``tau`` (float32, on the current cuda device).
    """
    assert k_global >= 1, f"k_global must be >= 1, got {k_global}"
    device = torch.cuda.current_device()
    values = values.to(device=device, dtype=torch.float32)
    mask = mask.to(device=device)
    valid = values[mask.bool()]

    local_k = min(k_global, valid.numel())
    if local_k > 0:
        local_topk = torch.topk(valid, local_k, largest=True).values
    else:
        local_topk = valid.new_empty(0)
    if local_topk.numel() < k_global:
        pad = local_topk.new_full((k_global - local_topk.numel(), ), float("-inf"))
        local_topk = torch.cat([local_topk, pad])

    world_size = torch.distributed.get_world_size(group=group)
    gathered = [torch.empty_like(local_topk) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local_topk.contiguous(), group=group)
    gathered = torch.cat(gathered)
    return torch.topk(gathered, k_global, largest=True).values[-1]


def pad_or_truncate_last_dim(
    t: torch.Tensor, len: int, value, pad_with_random_token: bool = False, vocab_size: int = 0
):
    if pad_with_random_token:
        if vocab_size == 0:
            t = pad_by_repeating_tokens(t, len)
        else:
            t = pad_by_random_tokens(t, len, vocab_size=vocab_size)
    else:
        if t.shape[-1] < len:
            padded_len = len - t.shape[-1]
            t = torch.nn.functional.pad(t, (0, padded_len), value=value)
        if t.shape[-1] > len:
            t = t[..., :len]
    assert t.shape[-1] == len, f"len mismatch {t.shape} {len=}"
    return t


def pad_by_repeating_tokens(t: torch.Tensor, seq_len: int):
    old_len = t.shape[-1]
    padded_len = seq_len - old_len
    if padded_len > 0:
        repeat_times = padded_len // old_len
        remainder = padded_len % old_len

        repeated_tokens = t.repeat(repeat_times)
        if remainder > 0:
            partial_tokens = t[:remainder]
            padding_tokens = torch.cat([repeated_tokens, partial_tokens], dim=-1)
        else:
            padding_tokens = repeated_tokens

        token = torch.cat([t, padding_tokens], dim=-1)
    else:
        token = t[..., :seq_len]
    return token


def pad_by_random_tokens(t: torch.Tensor, seq_len: int, vocab_size: int = 0):
    assert vocab_size > 0, f"vocab_size must be positive for random token padding, got {vocab_size}"
    old_len = t.shape[-1]
    padded_len = seq_len - old_len
    if padded_len > 0:
        gen = torch.Generator(device=t.device)
        gen.manual_seed(old_len)
        random_tokens = torch.randint(
            low=0,
            high=vocab_size,
            size=(*t.shape[:-1], padded_len),
            dtype=t.dtype,
            device=t.device,
            generator=gen
        )
        token = torch.cat([t, random_tokens], dim=-1)
    else:
        token = t[..., :seq_len]
    return token


def scatter_tensor(tensor, shape_meta, dtype, group):
    world_size = dist.get_world_size(group=group)
    # curr_rank means the rank of the current process in the group
    # source_rank means the global_rank in the group
    curr_rank = dist.get_rank(group=group)
    source_rank = dist.get_process_group_ranks(group)[0]

    batch_size, seq_len, vocab_size = shape_meta
    assert vocab_size % world_size == 0, f"vocab_size {vocab_size} must be divisible by world_size {world_size}"
    local_vocab_size = vocab_size // world_size

    if curr_rank == 0:
        scatter_list = []
        for i in range(world_size):
            start = i * local_vocab_size
            end = (i + 1) * local_vocab_size
            scatter_list.append(tensor[:, :, start:end].contiguous())
    else:
        scatter_list = None

    device = torch.cuda.current_device()
    local_tensor = torch.empty(
        (batch_size, seq_len, local_vocab_size),
        dtype=dtype,
        device=device,
    )
    try:
        dist.scatter(
            tensor=local_tensor,
            scatter_list=scatter_list,
            src=source_rank,
            group=group,
        )
    except Exception as e:
        log(f"EBUG error {e} {local_tensor.shape} {world_size=} {curr_rank=} {source_rank=}")
        if scatter_list is not None:
            log(f"EBUG error scatter_tensor {[lg_s.shape for lg_s in scatter_list]}")
        raise e
    return local_tensor


def repeat_interleave_tensor_or_list(t, repeat_n):
    if t is None:
        return None
    if torch.is_tensor(t):
        repeat_tensor = torch.repeat_interleave(t, repeat_n, dim=0)
        return repeat_tensor
    elif isinstance(t, list):
        new_l = [x for x in t for _ in range(repeat_n)]
        return new_l
    else:
        raise ValueError('unexpected data format, it must be a list of objects')


def save_images(images, rbi, save_images_dir):
    dp_rank = mpu.get_data_parallel_rank()
    assert save_images_dir is not None
    os.makedirs(save_images_dir, exist_ok=True)
    for ii, image in enumerate(images):
        image.save(f"{save_images_dir}/images_{dp_rank}_{rbi}_{ii}.png")


def split_dict_list_by_keys(data: List[Dict[str, Any]], selected_keys):
    selected_set = set(selected_keys)
    selected: List[Dict[str, Any]] = []
    unselected: List[Dict[str, Any]] = []
    for item in data:
        sel_item: Dict[str, Any] = {}
        unsel_item: Dict[str, Any] = {}
        for k, v in item.items():
            if k in selected_set:
                sel_item[k] = v
            else:
                unsel_item[k] = v
        selected.append(sel_item)
        unselected.append(unsel_item)
    return selected, unselected


def display_rollout_generation(tokenizer, disp_rng, rollout_batches: List[Dict[str, List[Any]]]):
    # 简化版 display，后面再调整
    bcast_list = [None]
    if torch.distributed.get_rank() == 0:
        assert len(rollout_batches) > 0, f"{len(rollout_batches)=}"
        pick_rank = disp_rng.randint(0, torch.distributed.get_world_size() - 1)
        pick_batch = disp_rng.randint(0, len(rollout_batches) - 1)
        bcast_list = [[pick_rank, pick_batch]]

    torch.distributed.broadcast_object_list(bcast_list, src=0)
    pick_rank, pick_batch = bcast_list[0]
    log(f"Displaying rollout generation from rank {pick_rank} batch {pick_batch}")

    rollout_batch = rollout_batches[pick_batch]
    tokens = list_of_tensor_to_list(rollout_batch['tokens'], False)
    prompt_lengths = list_of_tensor_to_list(rollout_batch["prompt_lengths"], True)
    seq_lengths = list_of_tensor_to_list(rollout_batch["sequence_lengths"], True)

    texts = tokenizer.batch_decode(tokens, skip_special_tokens=False)
    log_string = f'DISPLAY_ROLLOUT_GENERATION rollout_batches[{pick_batch}]\n'
    DISPLAY_ROLLOUT_GENERATION_SEP = '-' * 40 + '\n'
    num_to_pick = 2
    for ti, text in enumerate(texts[:num_to_pick]):
        log_string += DISPLAY_ROLLOUT_GENERATION_SEP
        log_string += f'micro_batch 0 idx {ti}\n'
        log_string += f'prompt_lengths {prompt_lengths[ti]}\n'
        log_string += f'length {seq_lengths[ti]}\n'

        log_string += DISPLAY_ROLLOUT_GENERATION_SEP
        log_string += f'^{text}$\n'
        log_string += DISPLAY_ROLLOUT_GENERATION_SEP
    if torch.distributed.get_rank() == pick_rank:
        log(log_string)


def display_offpolicy_generation(
    tokenizer, disp_rng, offpolicy_batches: List[Dict[str, List[Any]]]
):
    bcast_list = [None]
    if torch.distributed.get_rank() == 0:
        pick_rank = disp_rng.randint(0, torch.distributed.get_world_size() - 1)
        pick_batch = disp_rng.randint(0, len(offpolicy_batches) - 1)
        bcast_list = [[pick_rank, pick_batch]]

    torch.distributed.broadcast_object_list(bcast_list, src=0)
    pick_rank, pick_batch = bcast_list[0]
    log(f"Displaying generation from rank {pick_rank} batch {pick_batch}")

    rollout_batch = offpolicy_batches[pick_batch]
    tokens = list_of_tensor_to_list(rollout_batch['tokens'], False)
    prompt_lengths = list_of_tensor_to_list(rollout_batch["prompt_lengths"], True)
    seq_lengths = list_of_tensor_to_list(rollout_batch["sequence_lengths"], True)
    labels = list_of_tensor_to_list(rollout_batch["labels"], True)

    texts = tokenizer.batch_decode(tokens, skip_special_tokens=False)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=False)
    log_string = f'DISPLAY_TEACHER_GENERATION offpolicy_batches[{pick_batch}]\n'
    DISPLAY_ROLLOUT_GENERATION_SEP = '-' * 40 + '\n'
    num_to_pick = 2
    for ti, text in enumerate(texts[:num_to_pick]):
        label_text = label_texts[ti]
        log_string += DISPLAY_ROLLOUT_GENERATION_SEP
        log_string += f'micro_batch 0 idx {ti}\n'
        log_string += f'prompt_lengths {prompt_lengths[ti]}\n'
        log_string += f'length {seq_lengths[ti]}\n'

        log_string += DISPLAY_ROLLOUT_GENERATION_SEP
        log_string += f'^{text}$\n'
        log_string += f'^"label="{label_text}$\n'
        log_string += DISPLAY_ROLLOUT_GENERATION_SEP
    if torch.distributed.get_rank() == pick_rank:
        log(log_string)


def is_same_tokenizer(tokenizer1, tokenizer2, skip_eos_token_judge=False):
    if not skip_eos_token_judge:
        assert tokenizer1.eos_token_id == tokenizer2.eos_token_id
        assert tokenizer1.special_tokens_map == tokenizer2.special_tokens_map

    assert tokenizer1.vocab_size == tokenizer2.vocab_size
    assert tokenizer1.get_vocab() == tokenizer2.get_vocab()


def sort_list_by_sequence_lengths(data_list):
    return sorted(data_list, key=lambda x: x["sequence_lengths"])


def reorder_samples_by_alpha_and_seqlen(
    data_list: List[Dict[str, Any]], sort_batches: bool = False
) -> List[Dict[str, Any]]:
    """Reorder samples by ``offpd_loss_alpha`` and ``sequence_lengths``.

    Samples with ``offpd_loss_alpha > 0`` go first; otherwise after.
    When ``sort_batches`` is *True*, each side is also sorted by
    ``sequence_lengths``.

    Args:
        data_list:
        sort_batches:

    Returns:
        Reordered list.
    """
    if not data_list:
        return data_list

    positive_alpha = [d for d in data_list if d.get('offpd_loss_alpha', 0) > 0]
    non_positive_alpha = [d for d in data_list if d.get('offpd_loss_alpha', 0) <= 0]
    if sort_batches:
        positive_alpha_sorted = sorted(positive_alpha, key=lambda x: x.get("sequence_lengths", 0))
        non_positive_alpha_sorted = sorted(
            non_positive_alpha, key=lambda x: x.get("sequence_lengths", 0)
        )

        return positive_alpha_sorted + non_positive_alpha_sorted
    else:
        return positive_alpha + non_positive_alpha


# copied from Megatron-LM/megatron/training/utils.py
def logical_and_across_model_parallel_group(input: bool) -> bool:
    """All-reduce a bool across the model-parallel group via MIN (logical AND)."""
    if input is True:
        input = 1
    else:
        input = 0
    input = torch.tensor([input], dtype=torch.int, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        input, op=torch.distributed.ReduceOp.MIN, group=mpu.get_model_parallel_group()
    )
    return bool(input.item())


def reduce_max_stat_across_model_parallel_group(stat: float) -> float:
    """All-reduce ``stat`` (max) across the model-parallel group.

    Ranks without an optimizer have no grad_norm / num_zeros_in_grad stats; the
    logging / writer rank needs the value via MAX (values are already summed
    across optimizer ranks where applicable).
    """
    if stat is None:
        stat = -1.0
    stat = torch.tensor([stat], dtype=torch.float32, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        stat, op=torch.distributed.ReduceOp.MAX, group=mpu.get_model_parallel_group()
    )
    if stat.item() == -1.0:
        return None
    else:
        return stat.item()


def qwen2vl_pad_and_split(
    cp_size: int,
    hw_factor: int,
    pixel_values: list[torch.Tensor],
    image_grid_thws: list[torch.Tensor],
):
    assert len(pixel_values) == len(image_grid_thws)
    # split the pixel_values
    split_pixel_values = []
    split_image_grid_thws = []
    for pixel_value, image_grid_thw in zip(pixel_values, image_grid_thws):
        split_image_grid_thw = list(torch.split(image_grid_thw, 1, dim=0))
        split_image_grid_thws.extend(split_image_grid_thw)
        slice_begin = 0
        for ele in split_image_grid_thw:
            slice_end = slice_begin + ele.prod().item()
            split_pixel_values.append(pixel_value[slice_begin:slice_end].clone())
            slice_begin = slice_end

    pixel_values = split_pixel_values
    image_grid_thws = split_image_grid_thws
    img_num = len(image_grid_thws)

    img_num_per_rank = img_num // cp_size
    img_num_remain = img_num % cp_size
    cp_img_num = []
    for i in range(cp_size):
        cp_img_num.append(img_num_per_rank)
        if i < img_num_remain:
            cp_img_num[i] += 1

    img_idx = 0
    new_pixel_values = []
    new_image_grid_thws = []
    images_padded = []
    for i in range(cp_size):
        seq_len = 0
        img_begin_idx = img_idx
        img_end_idx = img_begin_idx + cp_img_num[i]
        img_idx += cp_img_num[i]

        for j in range(img_begin_idx, img_end_idx):
            seq_len += pixel_values[j].size(0)
            new_pixel_values.append(pixel_values[j])
            new_image_grid_thws.append(image_grid_thws[j])

        image_padded = 0 != seq_len % hw_factor
        if image_padded:
            padded_seqlen = (seq_len + hw_factor - 1) // hw_factor * hw_factor - seq_len
            assert padded_seqlen > 0 and padded_seqlen % 4 == 0
            new_pixel_values.append(
                torch.zeros(
                    [padded_seqlen, pixel_values[0].size(-1)],
                    dtype=pixel_values[0].dtype,
                    device=pixel_values[0].device,
                )
            )
            new_image_grid_thws.append(
                torch.tensor(
                    [[1, 2, padded_seqlen // 2]],
                    dtype=image_grid_thws[0].dtype,
                    device=image_grid_thws[0].device,
                )
            )
            cp_img_num[i] += 1
        images_padded.append(int(image_padded))

    return new_pixel_values, new_image_grid_thws, cp_img_num, images_padded


def to_device(data: Any, device: Union[str, torch.device, int], non_blocking: bool = False) -> Any:
    """Move inputs to a device"""
    if isinstance(data, Mapping):
        return type(data)({k: to_device(v, device, non_blocking) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(to_device(v, device, non_blocking) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=non_blocking)
    else:
        return data


def set_seed(seed: int):
    """Seed ``random`` / ``numpy`` / ``torch`` (CPU + CUDA) for reproducibility.

    Args:
        seed (`int`):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataclass_from_args(args, cls):
    kw_args = {}
    for f in dataclasses.fields(cls):
        if hasattr(args, f.name):
            # recursive convert
            if dataclasses.is_dataclass(f.type):
                kw_args[f.name] = dataclass_from_args(getattr(args, f.name), f.type)
            else:
                kw_args[f.name] = getattr(args, f.name)
    return cls(**kw_args)


def from_parallel_logits_to_logprobs(
    vocab_parallel_logits,
    target,
    inference_only=False,
    higher_stability=False,
    ignore_cp=False,
    pre_shifted=False,
):
    """Get log probs from a ``[B, S//CP, V//TP]`` tensor.

    ``ignore_cp``: skip CP gather (already gathered by CP).
    ``pre_shifted``: target is already next-token shifted (THD packed Dynamic
    CP), so skip the internal roll and the trailing truncation.

    Returns a ``[B, S-1]`` tensor, or ``[B, S]`` when ``pre_shifted``.
    """
    cp_rank = mpu.get_context_parallel_rank() if not ignore_cp else 0
    cp_size = mpu.get_context_parallel_world_size() if not ignore_cp else 1

    s = target.shape[1]
    assert s % cp_size == 0, f'{s=} {cp_size=}'
    local_s = s // cp_size

    if not pre_shifted:
        target = target.roll(shifts=-1, dims=-1)
    # NOTE(guanyouhe): Ulysess CP应该并不能这样分割
    if not ignore_cp:
        target = reorder_target_for_cp(target)

    local_target = target[:, cp_rank * local_s:(cp_rank + 1) * local_s]
    local_target = local_target.to(vocab_parallel_logits.device)

    local_target = rearrange(local_target, 'b s -> s b').contiguous()
    vocab_parallel_logits = rearrange(vocab_parallel_logits, 'b s h -> s b h').contiguous()
    curr_log_probs = -1 * tensor_parallel.vocab_parallel_cross_entropy(
        vocab_parallel_logits, local_target
    )
    curr_log_probs = rearrange(curr_log_probs, 's b -> b s').contiguous()

    if cp_size > 1 and not ignore_cp:
        curr_log_probs = all_gather_from_context_parallel_region(curr_log_probs)
    if pre_shifted:
        return curr_log_probs.contiguous()
    return curr_log_probs[:, :-1].contiguous()


def logprobs_from_linear_ce(
    linear_ce_backend,
    linear_ce_output: Dict[str, Any],
    target: torch.Tensor,
    ignore_cp=False,
    pre_shifted=False,
    mask: Optional[torch.Tensor] = None,
    return_entropy: bool = False,
) -> torch.Tensor:
    """Compute per-token logprobs from linear_ce model output.

    Parameters
    ----------
    linear_ce_output : dict
        Model output dict with keys ``hidden_states``, ``weight``,
        ``output_layer``.
    target : torch.Tensor
        Token ids ``[B, S]`` (unshifted).

    Returns
    -------
    torch.Tensor
        Log-probs ``[B, S-1]`` (float32), semantics consistent with
        ``from_parallel_logits_to_logprobs``.
    """
    set_linear_ce_backend(linear_ce_backend)

    cp_rank = mpu.get_context_parallel_rank() if not ignore_cp else 0
    cp_size = mpu.get_context_parallel_world_size() if not ignore_cp else 1

    s = target.shape[1]
    assert s % cp_size == 0, f'{s=} {cp_size=}'
    local_s = s // cp_size
    if not pre_shifted:
        target = target.roll(shifts=-1, dims=-1)
    # NOTE(guanyouhe): Ulysess CP应该并不能这样分割
    if not ignore_cp:
        target = reorder_target_for_cp(target)
    local_target = target[:, cp_rank * local_s:(cp_rank + 1) * local_s]

    hidden_states = linear_ce_output["hidden_states"]
    # [B, local_S] -> [local_S, B] to match hidden_states layout [local_S, B, H]
    local_target = local_target.transpose(0, 1).contiguous().to(hidden_states.device)

    output_layer = linear_ce_output["output_layer"]
    tp_group = output_layer.tp_group
    if output_layer.sequence_parallel:
        hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
            hidden_states,
            tensor_parallel_output_grad=True,
        )
    elif tp_group is not None and dist.get_world_size(tp_group) > 1:
        assert not hidden_states.requires_grad, (
            "linear_cross_entropy backward does not all-reduce d_hidden across TP ranks. "
            "With TP > 1, sequence_parallel=False, and hidden requiring grad, "
            "d_hidden would be incorrect. Enable sequence_parallel or use vocab_parallel_cross_entropy."
        )

    weight = linear_ce_output["weight"]
    if weight is None:
        weight = output_layer.weight

    linear_ce_output = linear_cross_entropy(
        hidden_states,
        weight,
        local_target,
        1.0,
        "none",
        tp_group,
        return_entropy=return_entropy,
    )

    if return_entropy:
        curr_log_probs, curr_entropy = linear_ce_output
    else:
        curr_log_probs = linear_ce_output
        curr_entropy = None

    curr_log_probs = -1 * curr_log_probs
    # [local_S, B] -> [B, local_S]
    curr_log_probs = curr_log_probs.transpose(0, 1).contiguous()
    if return_entropy:
        curr_entropy = curr_entropy.transpose(0, 1).contiguous()

    if cp_size > 1:
        curr_log_probs = all_gather_from_context_parallel_region(curr_log_probs)
        if return_entropy:
            curr_entropy = all_gather_from_context_parallel_region(curr_entropy)

    if pre_shifted:
        curr_log_probs = curr_log_probs.contiguous()
        if return_entropy:
            curr_entropy = curr_entropy.contiguous()
    else:
        curr_log_probs = curr_log_probs[:, :-1].contiguous()
        if return_entropy:
            curr_entropy = curr_entropy[:, :-1].contiguous()

    if return_entropy:
        if mask is not None:
            scaled_entropy = masked_mean(curr_entropy, mask)
        else:
            scaled_entropy = curr_entropy.mean()
        return curr_log_probs, scaled_entropy, curr_entropy

    return curr_log_probs


@torch.no_grad()
def from_parallel_logits_to_topk_logprobs(
    vocab_parallel_logits: torch.Tensor,
    topk: int,
    eps: float = 1e-10,
    ignore_cp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute global Top-K logprobs + token_ids under Vocab Parallel (TP).

    Args:
        vocab_parallel_logits: 单 TP 卡的 logits 切片，shape=[B, S//CP, V_p]
        topk: 全局 Top-K
        eps: 避免 log(0)
    Returns:
        global_topk_logprobs: shape=[B, S, k]
        global_topk_token_ids: shape=[B, S, k]
    """
    cp_size = mpu.get_context_parallel_world_size() if not ignore_cp else 1
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_world_size = mpu.get_tensor_model_parallel_world_size()
    partition_vocab_size = vocab_parallel_logits.size(-1)  # single GPU vocab slice size

    # vocabulary slice range (local index → global index offset)
    vocab_start_index, _ = tensor_parallel.utils.VocabUtility.vocab_range_from_per_partition_vocab_size(
        partition_vocab_size, tp_rank, tp_world_size
    )

    # per-GPU logits max (along v)
    logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True)[0]  # [B, S, 1]

    # global max within TP group
    torch.distributed.all_reduce(
        logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group
    )  # [B, S, 1]

    # local exp sum
    exp_logits = torch.exp(vocab_parallel_logits - logits_max)  # [B, S, V_p]
    sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)  # [B, S, 1]

    # global sum within TP group
    torch.distributed.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group)

    # local logprobs（log(softmax)）
    logprobs = torch.log(exp_logits / (sum_exp_logits + eps))  # [B, S, V_p]

    # local topk logprobs and token_ids
    local_topk_logprobs, local_topk_token_ids = logprobs.topk(
        topk, dim=-1, largest=True
    )  # 均为[B, S, k]

    # local token_ids -> global token_ids
    local_topk_global_indices = local_topk_token_ids + vocab_start_index  # [B, S, k]

    # gather topk logprobs, token_ids and logits along TP group
    gather_logprobs = torch.empty(
        (tp_world_size, *local_topk_logprobs.shape),
        dtype=local_topk_logprobs.dtype,
        device=local_topk_logprobs.device
    )  # [tp_world_size, B, S, k]
    gather_indices = torch.empty_like(
        gather_logprobs, dtype=local_topk_global_indices.dtype
    )  # [tp_world_size, B, S, k]
    torch.distributed.all_gather_into_tensor(gather_logprobs, local_topk_logprobs, group=tp_group)
    torch.distributed.all_gather_into_tensor(
        gather_indices, local_topk_global_indices, group=tp_group
    )

    # [tp_world_size, B, S, k] → [B, S, tp_world_size * k]
    gather_logprobs_flat = gather_logprobs.permute(1, 2, 0,
                                                   3).reshape(*local_topk_logprobs.shape[:-1], -1)
    gather_indices_flat = gather_indices.permute(1, 2, 0,
                                                 3).reshape(*local_topk_token_ids.shape[:-1], -1)

    # global topk logprobs and indices
    global_topk_logprobs, global_topk_token_ids_idx = gather_logprobs_flat.topk(
        topk, dim=-1, largest=True
    )
    global_topk_token_ids = torch.gather(
        gather_indices_flat, dim=-1, index=global_topk_token_ids_idx
    )

    if cp_size > 1 and not ignore_cp:
        global_topk_logprobs = all_gather_from_context_parallel_region(global_topk_logprobs)
        global_topk_token_ids = all_gather_from_context_parallel_region(global_topk_token_ids)

    return global_topk_logprobs, global_topk_token_ids


@torch.no_grad()
def from_parallel_logits_to_token_prob_and_rank(
    vocab_parallel_logits: torch.Tensor,
    token_id: int,
    eps: float = 1e-10,
    ignore_cp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one global token's probability and rank under vocab parallel logits."""
    cp_size = mpu.get_context_parallel_world_size() if not ignore_cp else 1
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_world_size = mpu.get_tensor_model_parallel_world_size()
    partition_vocab_size = vocab_parallel_logits.size(-1)
    padded_vocab_size = partition_vocab_size * tp_world_size
    if token_id < 0 or token_id >= padded_vocab_size:
        raise ValueError(
            f"token_id {token_id} is outside padded vocab range [0, {padded_vocab_size})"
        )
    vocab_parallel_logits = vocab_parallel_logits.float()

    vocab_start_index, vocab_end_index = tensor_parallel.utils.VocabUtility.vocab_range_from_per_partition_vocab_size(
        partition_vocab_size, tp_rank, tp_world_size
    )
    owns_token = vocab_start_index <= token_id < vocab_end_index

    logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True)[0]
    torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    shifted_logits = vocab_parallel_logits - logits_max
    sum_exp_logits = torch.exp(shifted_logits).sum(dim=-1, keepdim=True)
    torch.distributed.all_reduce(sum_exp_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group)

    if owns_token:
        local_token_idx = token_id - vocab_start_index
        local_token_logits = vocab_parallel_logits[..., local_token_idx]
        local_shifted_token_logits = shifted_logits[..., local_token_idx]
    else:
        local_token_logits = torch.zeros_like(vocab_parallel_logits[..., 0])
        local_shifted_token_logits = torch.zeros_like(vocab_parallel_logits[..., 0])

    token_logits = local_token_logits.clone()
    torch.distributed.all_reduce(token_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    shifted_token_logits = local_shifted_token_logits.clone()
    torch.distributed.all_reduce(
        shifted_token_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group
    )

    token_prob = torch.exp(shifted_token_logits) / (sum_exp_logits.squeeze(-1) + eps)
    local_rank = (vocab_parallel_logits > token_logits.unsqueeze(-1)).sum(dim=-1).to(torch.float32)
    torch.distributed.all_reduce(local_rank, op=torch.distributed.ReduceOp.SUM, group=tp_group)

    if cp_size > 1 and not ignore_cp:
        token_prob = all_gather_from_context_parallel_region(token_prob)
        local_rank = all_gather_from_context_parallel_region(local_rank)

    return token_prob, local_rank


def from_parallel_logits_to_opd_topk_logprobs(
    vocab_parallel_logits: torch.Tensor,
    target_ids: torch.Tensor,
    eps: float = 1e-10,
    ignore_cp: bool = False,
) -> torch.Tensor:
    """在指定的 token ids 上 gather log-probs（支持 TP + CP，保留梯度）。

    与 :func:`from_parallel_logits_to_topk_logprobs`（no_grad, 自选 top-K）不同，
    本函数保留 autograd graph，用于 top-K PPO loss 的 3D ratio 计算。

    Args:
        vocab_parallel_logits: 本 TP rank 的 logits ``[B, S // CP, V_p]``。
        target_ids: 要 gather 的全局 vocab ids ``[B, S, K]``，调用方需自行对齐
            predict-next 位移和 response 截取。
        eps: log 内的数值稳定常数。
        ignore_cp: 跳过 CP 切分（如 ``ppo_pack_seq`` 已拼接序列时设为 True）。

    Returns:
        ``[B, S, K]`` log-probs，所有 TP rank 一致，连接 autograd。
    """
    cp_size = 1 if ignore_cp else mpu.get_context_parallel_world_size()
    cp_rank = 0 if ignore_cp else mpu.get_context_parallel_rank()
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_world_size = mpu.get_tensor_model_parallel_world_size()
    partition_vocab_size = vocab_parallel_logits.size(-1)

    vocab_start_index, vocab_end_index = (
        tensor_parallel.utils.VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size, tp_rank, tp_world_size
        )
    )

    # 按 CP rank 切出本 shard 对应的 target_ids。
    if cp_size > 1 and not ignore_cp:
        target_ids = reorder_target_for_cp(target_ids, seq_dim=1)
    s = target_ids.shape[1]
    assert s % cp_size == 0, f'{s=} {cp_size=}'
    local_s = s // cp_size
    local_target_ids = target_ids[:, cp_rank * local_s:(cp_rank + 1) * local_s, :]
    local_target_ids = local_target_ids.to(vocab_parallel_logits.device, dtype=torch.long)

    # TP MAX-reduce 取全局 vocab max，detach 作为数值稳定偏移。
    logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True)[0]
    torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    logits_max = logits_max.detach()
    shifted_logits = vocab_parallel_logits - logits_max  # [B, local_s, V_p]

    # TP SUM-reduce 求全局 log-sum-exp（保留梯度）。
    local_sum_exp = torch.exp(shifted_logits).sum(dim=-1, keepdim=True)  # [B, local_s, 1]
    global_sum_exp = all_reduce_autograd(local_sum_exp, group=tp_group)
    log_norm = torch.log(global_sum_exp + eps)  # [B, local_s, 1], logprobs的分母

    # 只 gather 落在本 TP rank vocab 区间内的 id，其余置零，后续 TP SUM-reduce 自动汇合。
    # 判断每个 target id 是否落在本 TP rank 的 vocab 区间内
    in_range = (local_target_ids >= vocab_start_index) & (local_target_ids < vocab_end_index)
    # 计算 local index = target_id - vocab_start
    local_indices = (local_target_ids -
                     vocab_start_index).clamp(min=0, max=partition_vocab_size - 1)
    # torch.gather 从 shifted_logits 的 vocab 维度上按 local_indices 取值。
    local_target_logits = torch.gather(shifted_logits, dim=-1, index=local_indices)
    # 减去 log_norm 得到 log-prob
    local_target_logp = local_target_logits - log_norm  # [B, local_s, K]
    # torch.where，不在本 rank vocab 范围内的置零
    local_target_logp = torch.where( # [B, local_s, K]
        in_range, local_target_logp, torch.zeros_like(local_target_logp)
    )

    global_target_logp = all_reduce_autograd(local_target_logp, group=tp_group)  # [B, local_s, K]

    if cp_size > 1 and not ignore_cp:
        global_target_logp = all_gather_from_context_parallel_region(global_target_logp)

    return global_target_logp  # [B, S, K]


def get_dump_moe_metrics(is_full_recompute=False, num_samples=None):
    """Build per-sample MoE topk info from the gathered routing info.

    Data Structure:
    - gathered_routing_info: ``{"layer1": [sample1_data, sample2_data...], ...}``
    - sample_data: ``{"topk_scores": tensor[s, topk], "topk_indices": tensor[s, topk]}``

    Args:
        is_full_recompute:
        num_samples: passed to ``get_gathered_routing_info`` to trim recompute
            duplicates before gather.

    Returns:
        list: per-sample topk info, format
        ``[{"layer1": {topk_scores, topk_indices}, ...}, ...]``
    """
    gathered_routing_info = get_gathered_routing_info(
        is_full_recompute=is_full_recompute,
        num_samples=num_samples,
    )
    if mpu.is_pipeline_first_stage():
        if not gathered_routing_info:
            raise ValueError("gathered_routing_info is empty, cannot get MOE routing info")

        layer_keys = list(gathered_routing_info.keys())  # eg.["layer1", "layer2", ...]
        num_samples = len(gathered_routing_info[layer_keys[0]])
        all_sample_topk_info = [
            {
                layer_key: gathered_routing_info[layer_key][idx]
                for layer_key in layer_keys
            } for idx in range(num_samples)
        ]
        return all_sample_topk_info

    return None


def gcore_save_vllm_checkpoint(worker_wrap, checkpoint_dir):
    try:
        from vllm.model_executor.model_loader.loader import ShardedStateLoader
    except ImportError:
        from vllm.model_executor.model_loader.sharded_state_loader import (
            ShardedStateLoader,
        )

    try:
        model = worker_wrap.worker.model_runner.model
        # 这个好像没发检查 pp > 1 的情况。当 pp > 1 时，ShardedStateLoader 保存不完整
        ShardedStateLoader.save_model(
            model,
            checkpoint_dir,
            pattern=None,
            max_size=None,
        )
        logging_rank0(f"saved engine ckpt to {checkpoint_dir}")
    except Exception as e:
        logging_rank0(f"exception {e} when saving engine ckpt to {checkpoint_dir}")
    return True


@contextmanager
def catch_exception_ctx(scope=None):
    try:
        yield
    except Exception as e:
        # 打印完整异常堆栈
        exc_stack = traceback.format_exc()
        if torch.distributed.is_initialized():
            print(f"{torch.distributed.get_rank()} catch exception: {exc_stack}", flush=True)
        else:
            print(f"catch exception: {exc_stack}", flush=True)
        raise e
    finally:
        print(f"finish {scope}", flush=True)


@asynccontextmanager
async def catch_exception_ctx_async(scope=None):
    try:
        yield
    except Exception as e:
        # 打印完整异常堆栈
        exc_stack = traceback.format_exc()
        if torch.distributed.is_initialized():
            print(f"{torch.distributed.get_rank()} catch exception: {exc_stack}", flush=True)
        else:
            print(f"catch exception: {exc_stack}", flush=True)
        raise e
    finally:
        print(f"finish {scope}", flush=True)


def pad_3d_seq_dim(t: torch.Tensor, target_seq_len: int, value) -> torch.Tensor:
    """Pad / truncate a ``[S, K]`` tensor's sequence dim to ``target_seq_len``."""
    cur = t.shape[-2]
    if cur < target_seq_len:
        t = torch.nn.functional.pad(t, (0, 0, 0, target_seq_len - cur), value=value)
    elif cur > target_seq_len:
        t = t[..., :target_seq_len, :]
    assert t.shape[-2] == target_seq_len, f"shape mismatch {t.shape=} {target_seq_len=}"
    return t


def pad_topk_logprobs_to_target_len(
    logprobs_lst: List[torch.Tensor], target_logprobs: List[torch.Tensor]
):
    """
    对一个 list 中的每个 [S, K] tensor，逐个将 seq 维 pad 到和
    target list 对应元素一样长。只允许 pad（短→长），不允许 truncate。
    """
    for i in range(len(logprobs_lst)):
        tgt_s = target_logprobs[i].shape[-2]
        assert logprobs_lst[i].shape[-2] <= tgt_s, (
            f"topk logprobs seq dim exceeds target: {logprobs_lst[i].shape[-2]=} > {tgt_s=}"
        )
        logprobs_lst[i] = pad_3d_seq_dim(logprobs_lst[i], tgt_s, value=0)


def compute_topk_overlap_masks(
    stu_topk_ids: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute bidirectional membership masks between student and teacher top-K ids.

    Args:
        stu_topk_ids: [S-1, K] student top-K token ids.
        teacher_topk_ids: [S-1, K] teacher top-K token ids.

    Returns:
        stu_in_teacher_mask: [S-1, K] bool — True = in intersection, False = not.
        teacher_in_stu_mask: [S-1, K] bool — True = in intersection, False = not.
    """
    matches = (stu_topk_ids.unsqueeze(-1) == teacher_topk_ids.unsqueeze(-2))  # [S-1, K_s, K_t]
    stu_in_teacher_mask = matches.any(dim=-1)  # [S-1, K_s]
    # (rionawang)TODO union: teacher_in_stu_mask needed for union strategy
    # teacher_in_stu_mask = matches.any(dim=-2)   # [S-1, K_t]
    return stu_in_teacher_mask, None


def get_im_end_metrics_token_id(tokenizer) -> int:
    token = tokenizer.eos_token
    if token is None:
        raise ValueError("tokenizer.eos_token must be set when training.im_end_metrics_enable=True")
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"tokenizer.eos_token={token!r} must encode to exactly one token, got {token_ids}"
        )
    token_id = tokenizer.eos_token_id
    if token_id is None or token_id != token_ids[0]:
        raise ValueError(
            f"tokenizer.eos_token={token!r} is inconsistent with tokenizer.eos_token_id; "
            f"encode={token_ids}, eos_token_id={token_id}"
        )
    return int(token_id)


@torch.no_grad()
def whiten_advantages_cross_dp(
    advantages: List[Tensor],
    mask: List[Tensor],
    dp_group=None,
) -> Tuple[List[Tensor], Dict[str, float]]:
    """Whiten advantages globally across DP ranks.

    All DP ranks must call this in lockstep (same number of times per step),
    otherwise the all_reduce will deadlock.

    Parameters
    ----------
    advantages : list of Tensor
        Per-sample advantage tensors (CPU or CUDA).
    mask : list of Tensor
        Per-sample response masks, same shapes as ``advantages``.
    dp_group
        Process group for DP all-reduce.  Defaults to
        ``mpu.get_data_parallel_group()``.

    Returns
    -------
    tuple[list[Tensor], dict[str, float]]
        ``(whitened_advantages, sanity_metrics)`` where ``sanity_metrics``
        contains post-whiten mean / var / count for verification.
    """
    if dp_group is None:
        dp_group = mpu.get_data_parallel_group()

    flat_adv = torch.cat([a.flatten() for a in advantages])
    flat_mask = torch.cat([m.flatten().to(flat_adv.dtype) for m in mask])

    orig_device = flat_adv.device
    cuda_device = torch.device("cuda", torch.cuda.current_device())

    # Pass 1: global mean
    local_sum = (flat_adv * flat_mask).sum()
    local_cnt = flat_mask.sum()
    stats = torch.stack([local_sum, local_cnt]).to(cuda_device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=dp_group)
    stats = stats.to(orig_device)
    g_cnt = stats[1].clamp(min=1.0)
    mean = stats[0] / g_cnt

    # Pass 2: global variance (numerically stable, avoids catastrophic cancellation)
    local_sq_dev = ((flat_adv - mean)**2 * flat_mask).sum()
    sq_dev = local_sq_dev.to(cuda_device)
    dist.all_reduce(sq_dev, op=dist.ReduceOp.SUM, group=dp_group)
    sq_dev = sq_dev.to(orig_device)
    var = (sq_dev / g_cnt).clamp(min=0.0)
    inv_std = torch.rsqrt(var + 1e-8)

    advantages = [(a - mean) * inv_std for a in advantages]

    # Sanity check: post-whiten stats
    flat_adv_after = torch.cat([a.flatten() for a in advantages])
    local_sum_a = (flat_adv_after * flat_mask).sum()
    local_sq_dev_a = ((flat_adv_after)**2 * flat_mask).sum()
    stats_a = torch.stack([local_sum_a, local_sq_dev_a, local_cnt]).to(cuda_device)
    dist.all_reduce(stats_a, op=dist.ReduceOp.SUM, group=dp_group)
    stats_a = stats_a.to(orig_device)
    g_cnt_a = stats_a[2].clamp(min=1.0)
    g_mean_a = stats_a[0] / g_cnt_a
    g_var_a = (stats_a[1] / g_cnt_a - g_mean_a * g_mean_a).clamp(min=0.0)

    sanity_metrics = {
        "whiten_check/post_mean": g_mean_a.item(),
        "whiten_check/post_var": g_var_a.item(),
        "whiten_check/global_count": stats_a[2].item(),
    }
    return advantages, sanity_metrics


def build_token_loss_weights_from_spans(
    tokenizer,
    full_text: str,
    target_start_char: int,
    weight_spans: List[Dict[str, Any]],
    default_target_weight: float = 1.0,
    offset_mapping: Optional[List[Tuple[int, int]]] = None,
) -> List[float]:
    """Convert char-level ``weight_spans`` to per-token loss weights.

    Parameters
    ----------
    tokenizer
        HuggingFace tokenizer with ``return_offsets_mapping`` support.
    full_text : str
        Concatenated ``prompt + target`` string.
    target_start_char : int
        Character offset where target (label) begins in ``full_text``.
    weight_spans : list[dict]
        Each dict has ``start_char``, ``end_char`` (relative to target),
        and ``weight``.
    default_target_weight : float
        Weight for target tokens not covered by any span.
    offset_mapping : list[tuple[int, int]] | None
        Pre-computed ``(char_start, char_end)`` per token.  When provided
        the function skips re-tokenizing ``full_text``.

    Returns
    -------
    list[float]
        Per-token weights, same length as ``tokenizer(full_text)`` output.
    """
    if offset_mapping is None:
        encoded = tokenizer(
            full_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offset_mapping = encoded["offset_mapping"]

    offsets = offset_mapping
    loss_weights = [0.0] * len(offsets)

    abs_spans = []
    for span in weight_spans:
        abs_spans.append(
            {
                "start": target_start_char + span["start_char"],
                "end": target_start_char + span["end_char"],
                "weight": span["weight"],
            }
        )

    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= target_start_char:
            continue
        loss_weights[i] = default_target_weight
        for s in abs_spans:
            if tok_start < s["end"] and s["start"] < tok_end:
                loss_weights[i] = s["weight"]
                break

    return loss_weights


def format_token_weight_table(tokenizer, full_text, weights, target_start, spans=None):
    encoded = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    lines = [
        f"\n{'idx':>4} | {'token_text':20s} | {'char_span':12s} | {'weight':>6s} | region",
        "-" * 75,
    ]
    for i, (tid, (cs, ce)) in enumerate(zip(encoded["input_ids"], offsets)):
        tok_text = repr(full_text[cs:ce])
        span_str = f"[{cs:3d},{ce:3d})"
        region = "prompt" if ce <= target_start else "target"
        if spans and region == "target":
            for s in spans:
                abs_s = target_start + s["start_char"]
                abs_e = target_start + s["end_char"]
                if cs < abs_e and abs_s < ce:
                    region = f"span(w={s['weight']})"
                    break
        lines.append(f"{i:4d} | {tok_text:20s} | {span_str:12s} | {weights[i]:6.2f} | {region}")
    return "\n".join(lines)
