import itertools
import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Iterator, List

import torch
from typing_extensions import override

from megatron.core import parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams

from gpatch_v4.core.parallel_state import cpu_barrier
from gpatch_v4.utils import (
    log,
    logging_memory_usage,
    logging_memory_usage_details,
    logging_rank0,
)

_sample_idx_key = "sample_idx"
_seqlen_key = "sample_seqlen"


def get_dict_sum_expr(item):
    assert isinstance(item, dict), f"{type(item)=}"
    expr = ""
    for key in item.keys():
        if torch.is_tensor(item[key]):
            expr += f"{key=} sum={torch.sum(item[key])}"
            expr += '\n'
    return expr


def get_column_based_sum_expr(item):
    expr = ""
    for key in item.keys():
        assert isinstance(item[key], list) or torch.is_tensor(item[key]), f"{type(item[key])}"
        row_num = len(item[key])
        for row_id in range(row_num):
            if torch.is_tensor(item[key][row_id]):
                expr += f"{row_id=} {key=} sum={torch.sum(item[key][row_id])}"
                expr += '\n'
    return expr


def get_split_batchs(it: List, batch_size):
    new_list = iter(it)
    batch_list = []
    for batch in iter(lambda: list(itertools.islice(new_list, batch_size)), []):
        batch_list.append(batch)
    return batch_list


def get_split_batchs_iter(it: list, num_micro_batch_size: int):
    return itertools.chain(get_split_batchs(it, num_micro_batch_size))


def get_column_based_batches(row_based_batches, input_keys: List[str] = None):
    assert isinstance(
        row_based_batches, list
    ), f"batches must be a list, but got {type(row_based_batches)}"
    column_based_batches = {}
    if input_keys is None:
        input_keys = list(row_based_batches[0].keys())

    for input_key in input_keys:
        column_based_batches[input_key] = [
            batch.get(input_key, None) for batch in row_based_batches
        ]
    return column_based_batches


def get_row_based_batches(column_based_batches, input_keys=None, reserve_none=True):
    assert isinstance(column_based_batches, dict) or isinstance(
        column_based_batches, list
    ), f"batches must be a [dict[list]] or [list[list]], but got {type(column_based_batches)}"
    row_based_batches = []
    if input_keys is None:
        if isinstance(column_based_batches, dict):
            input_keys = column_based_batches.keys()
        else:
            input_keys = list(range(len(column_based_batches)))

    row_num = 0
    for input_key in input_keys:
        if column_based_batches[input_key] is not None:
            input_key_batch_size = len(column_based_batches[input_key])
            if row_num == 0:
                row_num = input_key_batch_size
            else:
                assert row_num == input_key_batch_size, f"[SMART_PAD] {input_key=} {input_key_batch_size=} {row_num=}"

    for i in range(row_num):
        row_based_batch = {}
        for input_key in input_keys:
            if column_based_batches[input_key] is not None:
                row_based_batch[input_key] = column_based_batches[input_key][i]
            elif reserve_none:
                row_based_batch[input_key] = None
        row_based_batches.append(row_based_batch)

    return row_based_batches


def get_sorted_split_batches(row_based_batches, batch_size, get_len_func: Callable):
    sort_func = lambda input: get_len_func(input)
    sorted_row_based_batches = sorted(row_based_batches, key=sort_func)
    sorted_row_based_split_batches = get_split_batchs(sorted_row_based_batches, batch_size)

    new_sorted_row_based_batches = []
    for sorted_row_based_split_batch in sorted_row_based_split_batches:
        new_sorted_row_based_batches.extend(sorted_row_based_split_batch)
    return sorted_row_based_split_batches, new_sorted_row_based_batches


def pad_batches_to_len(rollout_batches, key, pad_to_len, pad_value):
    # NOTE: row_based
    if isinstance(rollout_batches, list):
        for rollout_batch in rollout_batches:
            x = rollout_batch[key]
            if torch.is_tensor(x) and x.dim() > 0:
                # log(f"[PAD_TO_LEN] {key=} {x.shape=} {pad_to_len=} {pad_value=}", rank=0)
                rollout_batch[key] = torch.nn.functional.pad(
                    x, (0, pad_to_len - x.shape[-1]), value=pad_value
                )
    # NOTE: column_based
    elif isinstance(rollout_batches, dict):
        batchs = rollout_batches[key]
        assert isinstance(
            batchs, list
        ), f"rollout_batch[{key}] should be list, but get {type(batchs)}"
        for bid, x in enumerate(batchs):
            if torch.is_tensor(x) and x.dim() > 0:
                # log(f"[PAD_TO_LEN] {key=} {x.shape=} {pad_to_len=} {pad_value=}", rank=0)
                batchs[bid] = torch.nn.functional.pad(
                    x, (0, pad_to_len - x.shape[-1]), value=pad_value
                )
        rollout_batches[key] = batchs


# NOTE: move outside if need
def calc_pad_to_len(max_len, pad_to_multi_of):
    pad_to_len = (max_len + pad_to_multi_of - 1) // pad_to_multi_of * pad_to_multi_of
    return pad_to_len


'''
smart-pad EP / DP support help func
'''


def get_seqlen_dict_form_seqlens(seqlens: List) -> defaultdict:
    seqlen_dict = defaultdict(int)
    for seqlen in seqlens:
        seqlen_dict[seqlen] += 1
    return seqlen_dict


def gather_seqlen_dicts(my_seqlen_dict) -> List[Any]:
    all_seqlen_dicts = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(all_seqlen_dicts, my_seqlen_dict)
    return all_seqlen_dicts


def calc_global_seqlens(all_seqlen_dicts: List[defaultdict]) -> List:
    extend_seqlen_lists = []
    for rank_seqlen_dict in all_seqlen_dicts:
        rank_seqlens = []
        sorted_rank_seqlen_items = sorted(rank_seqlen_dict.items())
        for k, v in sorted_rank_seqlen_items:
            rank_seqlens.extend([k] * v)
        extend_seqlen_lists.append(rank_seqlens)

    global_seqlens = []
    tot = len(extend_seqlen_lists[0])
    for i in range(tot):
        global_seqlens.append(max(rank_seqlens[i] for rank_seqlens in extend_seqlen_lists))

    # log(f"[SMART_PAD] {global_seqlens=}")
    return global_seqlens


def recover_seqlens_from_splits_impl(
    splits: torch.Tensor, prefix_sums: List[int], origin_sorted_seqlen_items: list,
    seqlens: torch.Tensor, l: int, r: int
):
    split = int(splits[l][r].item())
    if split == -1:
        real_l = prefix_sums[l]
        real_r = prefix_sums[r + 1]
        for i in range(real_l, real_r):
            seqlens[i] = origin_sorted_seqlen_items[r][0]
    else:
        recover_seqlens_from_splits_impl(
            splits, prefix_sums, origin_sorted_seqlen_items, seqlens, l, split
        )
        recover_seqlens_from_splits_impl(
            splits, prefix_sums, origin_sorted_seqlen_items, seqlens, split + 1, r
        )


def recover_seqlens_from_splits(
    origin_sorted_seqlen_items: list, prefix_sums: List[int], splits: torch.Tensor
) -> List:
    num = len(origin_sorted_seqlen_items)
    tot = sum([item[1] for item in origin_sorted_seqlen_items])
    # log(f"[Split V2] {num=} {tot=} {prefix_sums=}")

    seqlens = torch.zeros(tot, dtype=torch.int32)
    recover_seqlens_from_splits_impl(
        splits, prefix_sums, origin_sorted_seqlen_items, seqlens, 0, num - 1
    )
    return seqlens.tolist()


def calc_score(seqlen: int, num_micro_batches: int):
    # world_size = torch.distributed.get_world_size()
    world_size = mpu.get_pipeline_model_parallel_world_size()
    score = seqlen**2 * (world_size + num_micro_batches - 1)
    return score


def calc_scores_from_seqlens(seqlens):
    task = defaultdict(int)
    if torch.is_tensor(seqlens):
        seqlens = seqlens.tolist()
    for seq in seqlens:
        task[seq] += 1
    tot_score = 0
    sorted_items = sorted(task.items())
    # log(f"[DEBUG] {sorted_items=}")
    for k, v in sorted_items:
        score = calc_score(k, v)
        # log(f"[CHECK] {k=} {v=} {score=}", rank=0)
        tot_score += score
    return tot_score


def convert_seqlens_to_dict(seqlens):
    seqlens_dict = defaultdict(int)
    if torch.is_tensor(seqlens):
        seqlens = seqlens.tolist()
    for seqlen in seqlens:
        seqlens_dict[seqlen] += 1
    return seqlens_dict


def calc_prefix_sums(origin_sorted_seqlen_items) -> list[int]:
    num = len(origin_sorted_seqlen_items)
    prefix_sums = [0]  # NOTE: [1, num] for convenience
    for i in range(1, num + 1):
        prefix_sum = origin_sorted_seqlen_items[i - 1][1]
        prefix_sum += prefix_sums[-1]
        prefix_sums.append(prefix_sum)
    return prefix_sums


def calc_optimized_seqlens(seqlens) -> List:
    seqlens_dict = defaultdict(int)
    assert isinstance(seqlens, list), f"seqlens must be list, but get type {type(seqlens)}"
    for seqlen in seqlens:
        seqlens_dict[seqlen] += 1
    sorted_seqlen_items = sorted(seqlens_dict.items())
    num = len(sorted_seqlen_items)
    scores = torch.full((num, num), -1, dtype=torch.int64)
    splits = torch.full((num, num), -1)

    prefix_sums = calc_prefix_sums(sorted_seqlen_items)
    # log(f"[Merge V2] {sorted_seqlen_items=} {num=} {prefix_sums=}")

    for w in range(1, num + 1):
        for l in range(0, num):
            r = l + w - 1
            if (r >= num):
                break
            # NOTE: [l, r]
            real_len = prefix_sums[r + 1] - prefix_sums[l]

            scores[l][r] = calc_score(sorted_seqlen_items[r][0], real_len)
            splits[l][r] = -1
            # log(f"[V2] {w=} {l=} {r=} {real_len=} {scores[l][r]=}", rank=0)
            for k in range(l, r):
                if scores[l][k] + scores[k + 1][r] < scores[l][r]:
                    scores[l][r] = scores[l][k] + scores[k + 1][r]
                    splits[l][r] = k

    best_score = scores[0][num - 1]
    optimized_seqlens = recover_seqlens_from_splits(sorted_seqlen_items, prefix_sums, splits)
    check_score = calc_scores_from_seqlens(optimized_seqlens)
    assert best_score == check_score, f"check score fail! {best_score=} {check_score=} {optimized_seqlens=}"
    return optimized_seqlens


"""
smart-pad-infer impl
"""


class SmartPadInferHelper():
    def __init__(self, batches: List[List[dict]], forward_batch_size):
        self.forward_batch_size = forward_batch_size
        self.origin_batches = batches
        self.row_based_batches = []  # List[dict] (rollout_batch_size * sampler_repeat_times) * dict

        # List[List[dict]] (rollout_batch_size * sampler_repeat_times // forward_batch_size, forward_batch_size) * dict
        self.extend_batches = []
        self.extend_orders = []

        self.seqlen_batch_ids = defaultdict(list)  # Dict[int:List[int]]
        self.batch_seqlens = [
        ]  # List: rollout_batch_size * sampler_repeat_times // forward_batch_size
        self.batchid_fwd_rets = {
        }  # Dict: rollout_batch_size * sampler_repeat_times // forward_batch_size

    def gen_row_based_batches(self):
        raise NotImplementedError("Base Class not support.")

    def gen_extend_batches(self, get_seqlen_func: Callable) -> None:
        extend_samples = []
        # log(f"[SMART_PAD_EXTEND] {seqlen_key=} {seq_related_keys=} {self.origin_batches=} {self.row_based_batches=}", rank=0)
        for sample_idx, batch in enumerate(self.row_based_batches):
            assert isinstance(batch, dict) or isinstance(
                batch, defaultdict
            ), f"expect type dict, but get type{type(batch)}"
            # if sample_idx == 0:
            #     log(f"[SMART_PAD_EXTEND] batch before extend: {batch}")
            extend_sample_info = {
                _sample_idx_key: torch.tensor(sample_idx),
                _seqlen_key: torch.tensor(get_seqlen_func(batch))
            }
            # if sample_idx == 0:
            #     log(f"[SMART_PAD_EXTEND] batch after extend: {batch}")

            batch.update(extend_sample_info)
            extend_samples.append(batch)
        self.extend_batches = extend_samples

    def gen_sorted_batches(self) -> None:
        assert self.extend_batches is not None
        get_len_func = lambda input: input[_seqlen_key]
        sorted_split_batches, _ = get_sorted_split_batches(
            self.extend_batches, batch_size=self.forward_batch_size, get_len_func=get_len_func
        )
        self.extend_batches = sorted_split_batches
        self.extend_orders = []
        for extend_batch in self.extend_batches:
            extend_order = [batch[_sample_idx_key] for batch in extend_batch]
            self.extend_orders.append(extend_order)
        assert len(self.extend_batches) == len(
            self.extend_orders
        ), f"extend_batches size {len(self.extend_batches)} extend_orders size {len(self.extend_orders)} mismatch!"
        # log(f"[SMART_PAD_DEBUG] {self.extend_batches=} {self.extend_orders=}")

    def gen_smart_pad_batches(self, pad_multi_of: int) -> None:
        batches = self.extend_batches
        assert len(batches) > 0

        # NOTE: for EP / DP, prepare seqlens
        my_seqlen_dict = defaultdict(int)
        for batch_id in range(len(batches)):
            batch = batches[batch_id]
            batch_seqlen = 0
            for input_id, input in enumerate(batch):
                # NOTE: input[_seqlen_key] is a tensor
                sample_seqlen = input[_seqlen_key].item()
                pad_to_len = calc_pad_to_len(sample_seqlen, pad_multi_of)
                seqlen = pad_to_len
                batch_seqlen = max(batch_seqlen, seqlen)
            my_seqlen_dict[batch_seqlen] += 1

        log(f"[SMART_PAD] {my_seqlen_dict=}", rank=0)
        all_seqlen_dicts = gather_seqlen_dicts(my_seqlen_dict)
        # log(f"{all_seqlen_dicts=}", rank=0)

        global_seqlens = calc_global_seqlens(all_seqlen_dicts)
        optimized_seqlens = calc_optimized_seqlens(seqlens=global_seqlens)
        log(f"[SMART_PAD] global_seqlen_dict: {convert_seqlens_to_dict(global_seqlens)}", rank=0)
        log(
            f"[SMART_PAD] calc_world_size: {mpu.get_pipeline_model_parallel_world_size()} optimized seqlen dict: {convert_seqlens_to_dict(optimized_seqlens)}",
            rank=0
        )

        self.seqlen_batch_ids = defaultdict(list)
        for batch_id in range(len(batches)):
            batch = batches[batch_id]
            seqlen = optimized_seqlens[batch_id]
            self.batch_seqlens.append(seqlen)
            self.seqlen_batch_ids[seqlen].append(batch_id)

        seqlen_batch_expr = {}
        for seqlen in self.seqlen_batch_ids.keys():
            seqlen_batch_expr[seqlen] = len(self.seqlen_batch_ids[seqlen])

        self.extend_batches = batches
        log(f"[SMART_PAD_SEQ] {seqlen_batch_expr=}", rank=0)

    @staticmethod
    def _calc_dynamic_mbs_for_seqlen(
        seqlen: int,
        num_batches: int,
        base_mbs: int,
        dynamic_mbs_target_seqlen: int,
        dynamic_mbs_limit: int,
    ) -> int:
        """Calculate dynamic micro-batch size for a given seqlen group.

        The returned dynamic_mbs may NOT evenly divide total_samples.
        Callers must handle the remainder (processed at base_mbs granularity).

        Parameters
        ----------
        seqlen : int
            Padded sequence length for this group.
        num_batches : int
            Total batches (each of size base_mbs) in this group.
        base_mbs : int
        dynamic_mbs_target_seqlen : int
        dynamic_mbs_limit : int

        Returns
        -------
        int
            Number of samples; always a multiple of base_mbs.
        """
        total_samples = num_batches * base_mbs
        mbs_factor = dynamic_mbs_target_seqlen // seqlen
        if mbs_factor < 1:
            mbs_factor = 1
        dynamic_mbs = mbs_factor * base_mbs
        if dynamic_mbs_limit is not None:
            dynamic_mbs = min(dynamic_mbs_limit, dynamic_mbs)
        # Clamp so that at least one full dynamic micro-batch can be formed
        dynamic_mbs = min(dynamic_mbs, total_samples)
        # Keep it a multiple of base_mbs
        dynamic_mbs = (dynamic_mbs // base_mbs) * base_mbs
        if dynamic_mbs < base_mbs:
            dynamic_mbs = base_mbs
        return dynamic_mbs

    def forward_per_seqlen_batches(
        self,
        forward_step_wrapped_func: Callable,
        dynamic_mbs_target_seqlen: int = None,
        dynamic_mbs_limit: int = None,
        update_total_iters_callback: Callable = None,
        skip_batch_id_merge: bool = False,
    ) -> None:
        self.batchid_fwd_rets = {}

        # Pre-calculate dynamic mbs for each seqlen group and cache it
        self.actual_total_forward_steps = 0
        seqlen_dynamic_mbs = {}  # cache: seqlen -> dynamic_mbs
        for seqlen, batch_ids in self.seqlen_batch_ids.items():
            num_batches = len(batch_ids)
            if dynamic_mbs_target_seqlen is not None and dynamic_mbs_target_seqlen > 0:
                dyn_mbs = self._calc_dynamic_mbs_for_seqlen(
                    seqlen=seqlen,
                    num_batches=num_batches,
                    base_mbs=self.forward_batch_size,
                    dynamic_mbs_target_seqlen=dynamic_mbs_target_seqlen,
                    dynamic_mbs_limit=dynamic_mbs_limit,
                )
            else:
                dyn_mbs = self.forward_batch_size
            seqlen_dynamic_mbs[seqlen] = dyn_mbs
            total_samples_in_group = num_batches * self.forward_batch_size
            num_dynamic_steps = total_samples_in_group // dyn_mbs
            remainder_samples = total_samples_in_group % dyn_mbs
            num_remainder_steps = remainder_samples // self.forward_batch_size
            self.actual_total_forward_steps += num_dynamic_steps + num_remainder_steps

        # Update external total_iters before starting forward
        if update_total_iters_callback is not None:
            update_total_iters_callback(self.actual_total_forward_steps)

        for seqlen, batch_ids in self.seqlen_batch_ids.items():
            num_batches = len(batch_ids)
            seqlen_batches = [self.extend_batches[batch_id] for batch_id in batch_ids]

            # Reuse pre-calculated dynamic mbs from cache
            dynamic_mbs = seqlen_dynamic_mbs[seqlen]

            if dynamic_mbs > self.forward_batch_size:
                flat_samples = [sample for batch in seqlen_batches for sample in batch]
                total_flat = len(flat_samples)
                num_microbatches = total_flat // dynamic_mbs
                remainder_samples = total_flat % dynamic_mbs
                num_remainder_microbatches = remainder_samples // self.forward_batch_size

                # -- main part: dynamic-mbs micro-batches --
                regrouped_batches = [
                    flat_samples[i * dynamic_mbs:(i + 1) * dynamic_mbs]
                    for i in range(num_microbatches)
                ]
                log(
                    f"[SMART_PAD_DYNAMIC_MBS] {seqlen=} {num_batches=} "
                    f"base_mbs={self.forward_batch_size} {dynamic_mbs=} "
                    f"{num_microbatches=} {num_remainder_microbatches=}",
                    rank=0
                )
                micro_fwd_step_rets = forward_step_wrapped_func(
                    itertools.chain(regrouped_batches), num_microbatches, dynamic_mbs, seqlen
                )

                # -- remainder part: base-mbs micro-batches --
                remainder_fwd_step_rets = []
                if num_remainder_microbatches > 0:
                    remainder_start = num_microbatches * dynamic_mbs
                    remainder_batches = [
                        flat_samples[remainder_start + i * self.forward_batch_size:remainder_start +
                                     (i + 1) * self.forward_batch_size]
                        for i in range(num_remainder_microbatches)
                    ]
                    remainder_fwd_step_rets = forward_step_wrapped_func(
                        itertools.chain(remainder_batches),
                        num_remainder_microbatches,
                        self.forward_batch_size,
                        seqlen,
                    )

                if mpu.is_pipeline_last_stage() and not skip_batch_id_merge:
                    sample_cursor = 0
                    for i, micro_ret in enumerate(micro_fwd_step_rets):
                        assert len(micro_ret) == dynamic_mbs, \
                            f"Expected {dynamic_mbs} results, got {len(micro_ret)}"
                        for j in range(0, dynamic_mbs, self.forward_batch_size):
                            orig_batch_idx = sample_cursor // self.forward_batch_size
                            batch_id = batch_ids[orig_batch_idx]
                            self.batchid_fwd_rets[batch_id] = \
                                micro_ret[j:j + self.forward_batch_size]
                            sample_cursor += self.forward_batch_size
                    for i, micro_ret in enumerate(remainder_fwd_step_rets):
                        orig_batch_idx = sample_cursor // self.forward_batch_size
                        batch_id = batch_ids[orig_batch_idx]
                        self.batchid_fwd_rets[batch_id] = micro_ret
                        sample_cursor += self.forward_batch_size
            else:
                log(
                    f"[SMART_PAD] forward {seqlen=} {num_batches=} "
                    f"mbs={self.forward_batch_size}",
                    rank=0
                )
                micro_fwd_step_rets = forward_step_wrapped_func(
                    itertools.chain(seqlen_batches), num_batches, self.forward_batch_size, seqlen
                )
                # NOTE: 当 off-policy-distill 用 skip_batch_id_merge
                # 避免收集所有 logits 到成员变量，无法释放 micro_fwd_step_rets，导致显存爆炸
                if mpu.is_pipeline_last_stage() and not skip_batch_id_merge:
                    for i in range(len(micro_fwd_step_rets)):
                        batch_id = batch_ids[i]
                        self.batchid_fwd_rets[batch_id] = micro_fwd_step_rets[i]

    def forward_pipeline(
        self,
        pad_to_multi_of,
        get_seqlen_func: Callable,
        forward_step_wrapped_func: Callable,
        dynamic_mbs_target_seqlen: int = None,
        dynamic_mbs_limit: int = None,
        update_total_iters_callback: Callable = None,
    ):
        self.gen_row_based_batches()
        self.gen_extend_batches(get_seqlen_func)
        self.gen_sorted_batches()
        self.gen_smart_pad_batches(pad_to_multi_of)
        self.forward_per_seqlen_batches(
            forward_step_wrapped_func=forward_step_wrapped_func,
            dynamic_mbs_target_seqlen=dynamic_mbs_target_seqlen,
            dynamic_mbs_limit=dynamic_mbs_limit,
            update_total_iters_callback=update_total_iters_callback,
        )

    def get_rowed_based_forward_results(self, is_row_based_rets=False) -> List[List[Any]]:
        row_based_fwd_rets = [list() for _ in range(len(self.batchid_fwd_rets))]
        for i in range(len(row_based_fwd_rets)):
            row_based_fwd_rets[i] = [None for _ in range(self.forward_batch_size)]

        if mpu.is_pipeline_last_stage():
            # log(f"[SMART_PAD_MERGE] {self.batchid_fwd_rets=}")
            for batch_id in range(len(self.batchid_fwd_rets)):
                batch_step_rets = self.batchid_fwd_rets[batch_id]
                if not is_row_based_rets:
                    row_based_batch_step_rets = get_row_based_batches(batch_step_rets)
                else:
                    row_based_batch_step_rets = batch_step_rets
                # log(f"[SMART_PAD_MERGE] {batch_step_rets=} {row_based_batch_step_rets=}")
                for idx in range(len(row_based_batch_step_rets)):
                    sample_idx = self.extend_orders[batch_id][idx]
                    real_batch_id = sample_idx // self.forward_batch_size
                    real_sub_id = sample_idx % self.forward_batch_size
                    # log(f"{sample_idx=} {real_batch_id=} {real_sub_id=}")
                    row_based_fwd_rets[real_batch_id][real_sub_id] = row_based_batch_step_rets[idx]

        return row_based_fwd_rets


class GroupSmartPadInferHelper(SmartPadInferHelper):
    @override
    def gen_row_based_batches(self):
        # log(f"[SMART_PAD_GEN_ROW] {self.__class__.__name__} Gen Row {self.origin_batches=}", rank=0)
        self.row_based_batches = []
        for batch in self.origin_batches:
            assert isinstance(batch, dict), f"expect type dict, but get type {type(batch)}"
            self.row_based_batches.extend(get_row_based_batches(batch))


class CatedSmartPadInferHelper(SmartPadInferHelper):
    @override
    def gen_row_based_batches(self):
        assert isinstance(
            self.origin_batches[0], dict
        ), f"expect type dict, but get type{type(self.origin_batches)}"
        self.row_based_batches = self.origin_batches


"""
smart-pad-train impl
"""


def smart_pad_train_get_reorder_rollout_batches(
    ex_rollout_batches, num_global_batch, train_global_batch_size, pad_to_multi_of, reorder_seed
):
    get_len_func = lambda input: input['sequence_lengths'].item()

    # log(f"[SMART_PAD_TRAIN] get_len first data {get_len_func(ex_rollout_batches[0])}")
    row_based_split_batches, _ = get_sorted_split_batches(
        ex_rollout_batches, train_global_batch_size, get_len_func=get_len_func
    )

    # NOTE: for EP support
    sorted_seqlens = []
    my_seqlen_dict = defaultdict(int)
    for row_based_split_batch in row_based_split_batches:
        for input in row_based_split_batch:
            sorted_seqlens.append(get_len_func(input))
    for seqlen in sorted_seqlens:
        pad_to_len = calc_pad_to_len(seqlen, pad_to_multi_of)
        my_seqlen_dict[pad_to_len] += 1
    # log(f"[SMART_PAD_TRAIN] {my_seqlen_dict=}")

    all_seqlen_dicts = gather_seqlen_dicts(my_seqlen_dict)
    global_seqlens = calc_global_seqlens(all_seqlen_dicts)
    global_seqlens_split_by_gbs = get_split_batchs(global_seqlens, train_global_batch_size)
    pad_gloabl_seqlens_per_global_batch = [
        max(global_seqlens_split) for global_seqlens_split in global_seqlens_split_by_gbs
    ]
    log(
        f"[SMART_PAD_TRAIN] global_seqlen_dict: {get_seqlen_dict_form_seqlens(global_seqlens)}",
        rank=0
    )

    # diversity
    idxs = list(range(num_global_batch))
    rng = random.Random(reorder_seed)
    rng.shuffle(idxs)

    reorder_row_based_batches = []
    reorder_global_seqlens_per_gbs = []
    for i in range(num_global_batch):
        idx = idxs[i]
        reorder_row_based_batches.extend(row_based_split_batches[idx])
        reorder_global_seqlens_per_gbs.append(pad_gloabl_seqlens_per_global_batch[idx])

    final_global_seqlens = []
    for pad_global_seqlen in reorder_global_seqlens_per_gbs:
        final_global_seqlens.extend([pad_global_seqlen] * train_global_batch_size)
    log(
        f"[SMART_PAD_TRAIN] final_global_seqlen_dict: {get_seqlen_dict_form_seqlens(final_global_seqlens)=}",
        rank=0
    )

    return reorder_row_based_batches


# pack seq impl (adapted from verl)


def preprocess_packed_seqs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pre_process: bool = True
) -> tuple[torch.Tensor, PackedSeqParams]:
    """
    Preprocess packed sequences
    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0 gets first and last chunks, GPU1
    gets second and second last chunks, and so on), this is for load balancing with causal masking.
    See https://github.com/NVIDIA/TransformerEngine/issues/1368
    """
    batch_size = input_ids.shape[0]

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size

    pad_size = (align_size - seqlens_in_batch % align_size) % align_size
    seqlens_in_batch_padded = seqlens_in_batch + pad_size

    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)

    # ----------------------------------------------------------------------------
    # Move the index information needed in the subsequent loop to the CPU at once,
    # to avoid frequent .item() calls in the loop that cause D2H synchronization
    # ----------------------------------------------------------------------------
    seqlens_in_batch_cpu: list[int] = seqlens_in_batch.tolist()  # original valid lengths
    seqlens_in_batch_padded_cpu: list[int] = seqlens_in_batch_padded.tolist(
    )  # lengths after padding
    cu_seqlens_padded_cpu: list[int] = cu_seqlens_padded.tolist()  # start positions (after padding)

    # Pure Python int calculation to avoid further synchronization
    max_seqlen_in_batch = max(seqlens_in_batch_padded_cpu)

    shape = list(input_ids.shape[1:])
    shape[0] = sum(seqlens_in_batch_padded_cpu) // cp_size
    if pre_process:
        input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
        for i in range(batch_size):
            # Use Python int, so no GPU→CPU sync in the loop
            if cp_size <= 1:
                seqlen = seqlens_in_batch_cpu[i]
                start_idx = cu_seqlens_padded_cpu[i]
                input_ids_rmpad[start_idx:start_idx + seqlen] = input_ids[i, attention_mask[i]]
                continue

            seqlen_padded_i = seqlens_in_batch_padded_cpu[i]
            seqlen = seqlen_padded_i // cp_size
            half_seqlen = seqlen // 2
            start_idx = cu_seqlens_padded_cpu[i] // cp_size
            # split to 2 chunks
            d = input_ids[i, attention_mask[i]]
            input_ids_rmpad[start_idx:start_idx +
                            half_seqlen] = d[half_seqlen * cp_rank:half_seqlen * (cp_rank + 1)]

            remain_start = seqlen_padded_i - half_seqlen * (cp_rank + 1)
            remain_end = seqlen_padded_i - half_seqlen * cp_rank
            remain_end = min(remain_end, d.shape[0])
            remain_len = remain_end - remain_start
            if remain_len > 0:
                input_ids_rmpad[start_idx + half_seqlen:start_idx + half_seqlen +
                                remain_len] = d[remain_start:remain_end]

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )
    if pre_process:
        return input_ids_rmpad.unsqueeze(0), packed_seq_params
    else:
        return input_ids, packed_seq_params


def postprocess_packed_seqs(
    output: torch.Tensor,
    packed_seq_params: PackedSeqParams,
    attention_mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
    post_process: bool = True,
) -> torch.Tensor:
    """
    Postprocess packed sequences
    """
    if not post_process:
        return output

    # -------------------------------------------------------------------------
    # Move the lengths and offsets needed for subsequent Python-level indexing to the CPU in advance,
    # to avoid a large number of .item() calls in the loop
    # -------------------------------------------------------------------------
    cu_padded_cpu: list[int] = packed_seq_params.cu_seqlens_q_padded.tolist()
    seq_lens_cpu: list[int] = attention_mask.sum(dim=1, dtype=torch.int32).cpu().tolist()

    shape = [batch_size, seq_len
            ] + list(output.shape[2:])  # 1,packed, dim -> batch_size, seq_len, dim
    output_new = torch.zeros(shape, dtype=output.dtype, device=output.device)

    cp_size = mpu.get_context_parallel_world_size()
    # all gather output across context parallel group
    if cp_size > 1:
        # output shape: [1, packed_len, hidden_dim]
        # need to gather across cp group and concatenate in sequence dimension
        output_list = [torch.empty_like(output) for _ in range(cp_size)]
        torch.distributed.all_gather(
            output_list, output.detach(), group=mpu.get_context_parallel_group()
        )
        output_list[mpu.get_context_parallel_rank()] = output
    else:
        output_list = [output]
    for i in range(batch_size):
        if cp_size <= 1:
            s = seq_lens_cpu[i]
            start_idx = cu_padded_cpu[i]
            output_new[i, attention_mask[i]] = output[0][start_idx:start_idx + s]
            continue
        s_len_padded_chunk = (cu_padded_cpu[i + 1] - cu_padded_cpu[i]) // cp_size
        half_seqlen = s_len_padded_chunk // 2
        s_len = seq_lens_cpu[i]
        s_len_padded = s_len_padded_chunk * cp_size
        tmp = torch.empty(s_len_padded, *output.shape[2:], device=output.device)
        for j in range(cp_size):
            o = output_list[j][0]
            # split to 2 chunks
            packed_start_idx = cu_padded_cpu[i] // cp_size
            o0, o1 = (
                o[packed_start_idx:packed_start_idx + half_seqlen],
                o[packed_start_idx + half_seqlen:packed_start_idx + s_len_padded_chunk],
            )
            tmp[j * half_seqlen:(j + 1) * half_seqlen] = o0
            tmp[s_len_padded - (j + 1) * half_seqlen:s_len_padded - j * half_seqlen] = o1
        output_new[i, attention_mask[i]] = tmp[:s_len]

    return output_new


"""
DP-balance helper: static methods wrapping dp_balancing rebalance/restore so
they can be called once externally before passing data to multiple engine calls.
"""


class DPBalanceHelper:
    """Thin static wrapper around dp_balancing utilities.

    Usage in the actor:

        rebalanced_batches, restore_info = DPBalanceHelper.rebalance(rollout_batches, samples_per_batch)
        ref_logps, prev_logps = engine.compute_log_probs(rebalanced_batches)
        # attach logps to rebalanced_batches …
        rollout_batches = DPBalanceHelper.restore(rebalanced_batches, restore_info)
    """
    @staticmethod
    def rebalance(
        rollout_batches: List[Dict[str, Any]],
        samples_per_batch: int,
        require_keys: List[str] = None,
    ):
        """Rebalance samples across DP ranks to minimize padding waste.

        Parameters
        ----------
        rollout_batches : list of dict
            Column-based, original order.
        samples_per_batch : int
        require_keys : list of str, optional
            If given, only exchange these keys (saves bandwidth). After
            restore, only the exchanged keys are present.

        Returns
        -------
        tuple
            ``(rebalanced_rollout_batches, restore_info)``.
        """
        from gpatch_v4.core.dp_balancing import rebalance_across_dp_ranks
        return rebalance_across_dp_ranks(
            rollout_batches, samples_per_batch, require_keys=require_keys
        )

    @staticmethod
    def restore(
        rebalanced_rollout_batches: List[Dict[str, Any]],
        restore_info: dict,
    ):
        """Restore samples to original DP rank distribution and order.

        Parameters
        ----------
        rebalanced_rollout_batches : list of dict
            Column-based, rebalanced order.
        restore_info : dict
            Returned by :meth:`rebalance`.

        Returns
        -------
        list of dict
            Column-based, original order.
        """
        from gpatch_v4.core.dp_balancing import restore_original_order
        return restore_original_order(rebalanced_rollout_batches, restore_info)

    @staticmethod
    def rebalance_for_compute_log_probs(
        rollout_batches: List[Dict[str, Any]],
        samples_per_batch: int,
        add_custom_keys: List[str] = None,
    ):
        """Rebalance rollout batches before compute_log_probs."""

        infer_require_keys = list(rollout_batches[0].keys())
        infer_require_keys = DPBalanceHelper.filter_keys(
            infer_require_keys,
            add_custom_keys=add_custom_keys,
        )
        logging_rank0(f"DPBalanceHelper require_keys before compute_logps {infer_require_keys=}")

        cpu_barrier()
        logging_memory_usage_details("memory tracking before dp rebalance", rank=0)
        rebalanced_batches, restore_info = DPBalanceHelper.rebalance(
            rollout_batches,
            samples_per_batch,
            require_keys=infer_require_keys,
        )
        cpu_barrier()
        logging_memory_usage_details("memory tracking after dp rebalance", rank=0)
        return rebalanced_batches, restore_info

    @staticmethod
    def restore_log_probs_to_original_batches(
        origin_rollout_batches: List[Dict[str, Any]],
        rebalanced_rollout_batches: List[Dict[str, Any]],
        restore_info: dict,
        without_ref: bool,
        extra_keys: List[str] = None,
    ):
        """Restore logprob fields from rebalanced batches back to original order.

        ``extra_keys``: rebalance 之后新增的字段（如 top-K / teacher 相关），
        不存在的 key 会被静默跳过。
        """

        restore_keys = list(restore_info.get('require_keys') or [])
        restore_keys.append("logprobs")
        if not without_ref:
            restore_keys.append("ref_logprobs")
        if extra_keys:
            for k in extra_keys:
                if k not in restore_keys:
                    restore_keys.append(k)
        restore_info = {**restore_info, "require_keys": restore_keys}

        cpu_barrier()
        logging_memory_usage_details("memory tracking before dp restore", rank=0)
        restored_batches = DPBalanceHelper.restore(rebalanced_rollout_batches, restore_info)
        cpu_barrier()
        logging_memory_usage_details("memory tracking after dp restore", rank=0)

        for orig_rb, restored_rb in zip(origin_rollout_batches, restored_batches, strict=True):
            assert "src_dp" in restored_rb and "tokens" in restored_rb
            for i in range(len(orig_rb["src_dp"])):
                assert orig_rb["src_dp"][i] == restored_rb["src_dp"][i]
                assert orig_rb["tokens"][i].sum() == restored_rb["tokens"][i].sum(
                ) and orig_rb["tokens"][i].shape == restored_rb["tokens"][i].shape

            if "ref_logprobs" in restored_rb:
                orig_rb["ref_logprobs"] = restored_rb["ref_logprobs"]
            if "logprobs" in restored_rb:
                orig_rb["logprobs"] = restored_rb["logprobs"]
            if extra_keys:
                for k in extra_keys:
                    if k in restored_rb:
                        orig_rb[k] = restored_rb[k]

        return origin_rollout_batches

    @staticmethod
    def rebalance_row_batches_for_train(
        expanded_rbs: List[Dict[str, Any]],
        add_custom_keys: List[str] = None,
    ):
        """Rebalance row-based samples across DP ranks before training."""
        cpu_barrier()
        logging_memory_usage_details("memory tracking before dp rebalance (train)", rank=0)
        train_col_batch = [get_column_based_batches(expanded_rbs)]
        require_keys = DPBalanceHelper.filter_keys(
            list(expanded_rbs[0].keys()),
            add_custom_keys=add_custom_keys,
        )
        logging_rank0(f"DPBalanceHelper require_keys before train {require_keys=}")

        train_col_batch, _ = DPBalanceHelper.rebalance(
            train_col_batch,
            len(expanded_rbs),
            require_keys=require_keys,
        )

        rebalanced_rbs = []
        for col_batch in train_col_batch:
            rebalanced_rbs.extend(get_row_based_batches(col_batch))

        cpu_barrier()
        logging_memory_usage_details("memory tracking after dp rebalance (train)", rank=0)
        return rebalanced_rbs

    @staticmethod
    def filter_keys(all_keys: List[str], add_custom_keys: List[str] = None) -> List[str]:
        """Filter out reward-related and rm-auxiliary keys from rollout batch keys.

        Excluded:
        - any key containing ``reward`` (``rewards``, ``per_token_rewards``,
          ``reward_bt_rm_0``, …);
        - keys matching ``rm_*_tokens`` / ``rm_*_prompt_lengths`` /
          ``rm_*_sequence_lengths`` / ``rm_*_output_mask``.

        If *add_custom_keys* is given, they are appended (dedup) to the result.

        Parameters
        ----------
        all_keys : list of str
        add_custom_keys : list of str, optional

        Returns
        -------
        list of str
        """
        import re

        # Pattern for rm auxiliary fields: rm_<anything>_tokens / _prompt_lengths / _sequence_lengths / _output_mask
        _rm_aux_pattern = re.compile(
            r'^rm_.+_(tokens|prompt_lengths|sequence_lengths|output_mask)$'
        )
        #TODO(guanyouhe): 看看这里多模态有没有什么是 compute_logps / train 本身不需要用到的key，提前过滤掉

        filtered = []
        for key in all_keys:
            # Skip keys containing "reward"
            if 'reward' in key:
                continue
            # Skip rm auxiliary keys
            if _rm_aux_pattern.match(key):
                continue
            filtered.append(key)

        # Append custom keys (deduplicated)
        if add_custom_keys is not None:
            assert isinstance(
                add_custom_keys, list
            ), (f"add_custom_keys must be a list, got {type(add_custom_keys)} {add_custom_keys=}")
            existing = set(filtered)
            for key in add_custom_keys:
                if key not in existing:
                    filtered.append(key)
                    existing.add(key)

        return filtered
