import asyncio
import math
import os
from collections import Counter, deque
from collections.abc import Callable, Iterator
from typing import Any, Deque, Dict, List, Tuple

import torch
import torch.distributed
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.core.parallel_state import (
    cpu_barrier,
    cpu_group,
    get_model_and_context_parallel_group,
    get_model_and_context_parallel_group_gloo,
    get_model_and_context_parallel_src_rank,
)
from gpatch_v4.rollout_generator.base_generator import BaseRolloutGenerator
from gpatch_v4.rollout_generator.mixin import build_stream_ready_view
from gpatch_v4.training_backend.megatron_backend.checkpoint import (
    get_dynamic_sampling_save_path,
)
from gpatch_v4.utils import (
    TimerSingleton,
    destroy_process_groups,
    reload_process_groups,
)
from gpatch_v4.utils.common_utils import (
    clear_memory,
    import_fn_from_path,
    logging_memory_usage_details,
    logging_rank0,
)
from gpatch_v4.utils.training_utils import check_rollout_batches


def _concat_values(values: List[Any]):
    if isinstance(values[0], dict):
        return {key: _concat_values([value[key] for value in values]) for key in values[0]}
    if not isinstance(values[0], list):
        return values[0]
    merged = []
    for value in values:
        merged.extend(value)
    return merged


def _prompts_to_batch(prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
    stripped = [dict(prompt) for prompt in prompts]
    cache_keys = stripped[0].pop("cache_keys", None)
    for prompt in stripped[1:]:
        prompt.pop("cache_keys", None)
    batch = {key: _concat_values([prompt[key] for prompt in stripped]) for key in stripped[0]}
    if cache_keys is not None:
        batch["cache_keys"] = cache_keys
    return batch


class DynamicSamplingDataSource:
    """Resettable prompt iterator used by dynamic sampling."""
    def __init__(
        self,
        reset_iter: Callable[..., Iterator],
        dataloader,
        current_epoch: int,
        max_epochs: int,
        batch_size: int,
    ):
        self.reset_iter = reset_iter
        self.current_epoch = current_epoch
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.batches_consumed = 0
        self.reached_epoch_limit = False
        self.prompt_buffer: Deque[Dict[str, List[Any]]] = deque()
        self.data_iter = reset_iter(current_epoch, skip_batches=0)
        self.batches_per_epoch = len(dataloader)

    def next_batch(self) -> Dict[str, List[Any]]:
        if self.batches_consumed == self.batches_per_epoch:
            self._advance_epoch()
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self._advance_epoch()
            batch = next(self.data_iter)
        self.batches_consumed += 1
        if (
            self.batches_consumed == self.batches_per_epoch and
            self.current_epoch + 1 >= self.max_epochs
        ):
            self.reached_epoch_limit = True
        return batch

    def take_wave(self, num_mbs: int) -> List[Dict[str, List[Any]]]:
        """Return ``num_mbs`` prompt batches from the buffer or the dataloader.

        Uses the buffer only when it already holds a full wave; otherwise
        every batch comes from the dataloader and the buffer is left intact.
        """
        num_prompts = num_mbs * self.batch_size
        if len(self.prompt_buffer) >= num_prompts:
            batches = []
            for _ in range(num_mbs):
                prompts = [self.prompt_buffer.popleft() for _ in range(self.batch_size)]
                batches.append(_prompts_to_batch(prompts))
            return batches
        return [self.next_batch() for _ in range(num_mbs)]

    def put_back(self, prompt: Dict[str, List[Any]]) -> None:
        """Append one unstripped prompt to the leftover buffer."""
        self.prompt_buffer.append(prompt)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_epoch": self.current_epoch,
            "batches_consumed": self.batches_consumed,
            "reached_epoch_limit": self.reached_epoch_limit,
            "prompt_buffer": list(self.prompt_buffer),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.current_epoch = state["current_epoch"]
        self.batches_consumed = state["batches_consumed"]
        self.reached_epoch_limit = state["reached_epoch_limit"]
        self.prompt_buffer = deque(state["prompt_buffer"])
        self.data_iter = self.reset_iter(self.current_epoch, skip_batches=self.batches_consumed)

    def _advance_epoch(self):
        self.current_epoch += 1
        self.data_iter = self.reset_iter(self.current_epoch, skip_batches=0)
        self.batches_consumed = 0


class DynamicSamplingRolloutGenerator(BaseRolloutGenerator):
    """Rollout generator that refills until enough valid prompt groups exist.
    Each wave is sampler → gen-RM → BT-RM → external reward → filter, matching
    ``BaseRolloutGenerator`` phases so colocate can time-share GPUs.
    NOTE: We do not use the early stop + abort to saving time yet, since:
    1. filter needs reward &
    2. in colocate mode, it is tricky to process generation+reward in prmopt
    level (we have to onload/offload models, so the schedule level is batch), you
    have to wait for all ranks to finish oversampling generation before you can
    compute reward and abort the sampler.
    """
    def __init__(
        self,
        config: RlConfig,
        sampler_client,
        gen_rm_client,
        bt_rm_client,
        run_eval=False,
        external_reward=None,
    ):
        super().__init__(config, sampler_client, gen_rm_client, bt_rm_client, run_eval)
        self.handle_external_reward_in_generator = True
        self.external_reward = external_reward
        self.dynamic_config = config.training.dynamic_sampling
        self.ema_expansion_ratio = self.dynamic_config.init_expansion_ratio
        self.prompt_issue_idx = 0
        self.request_idx = 0
        self.filter_reasons = Counter()
        self._step_metrics: Dict[str, float] = {}
        self.data_source = None
        self.dynamic_filter = None
        if self.dynamic_config.filter_py_path is not None:
            self.dynamic_filter = import_fn_from_path(
                self.dynamic_config.filter_py_path, self.dynamic_config.filter_fn_name
            )

    @override
    def set_external_reward(self, external_reward) -> None:
        self.external_reward = external_reward

    @override
    def setup_data_source(
        self,
        dataloader,
        reset_iter: Callable[..., Iterator],
        resume_step: int = 0,
    ) -> Tuple[DynamicSamplingDataSource, bool, bool]:
        self.data_source = DynamicSamplingDataSource(
            reset_iter,
            dataloader,
            current_epoch=0,
            max_epochs=self.config.training.num_train_epoches,
            batch_size=self.config.training.rollout_mbs,
        )
        if resume_step > 0:
            self.load_resume_state(resume_step)
        return self.data_source, self.should_stop_for_consumed_data_epochs(), True

    @override
    def should_stop_for_consumed_data_epochs(self) -> bool:
        if self.dynamic_config.epoch_mode != "consumed_data_epochs":
            return False
        reached = bool(self.data_source.reached_epoch_limit)
        tensor = torch.tensor([int(reached)], dtype=torch.int64, device="cuda")
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
        return bool(tensor.item())

    def _put_back_prompt(self, orig_prompt: Dict[str, List[Any]]) -> None:
        unique_id = orig_prompt["unique_id"][0]
        cache = self.apply_sampling_rollout_attr.cached_rollout_attrs()
        cached = cache[unique_id]
        for key, value in cached.items():
            orig_prompt[key] = [value]
        if cached:
            orig_prompt["cache_keys"] = list(cached.keys())
        del cache[unique_id]
        self.data_source.put_back(orig_prompt)

    @override
    def save_resume_state(self, step: int) -> None:
        if not self.is_mp_and_cp_head:
            return
        save_ckpt_path = self.config.checkpoint.save_ckpt_path
        assert save_ckpt_path, "dynamic sampling checkpoint requires checkpoint.save_ckpt_path"
        dp_rank = mpu.get_data_parallel_rank()
        dp_size = mpu.get_data_parallel_world_size()
        rollout_mbs = self.config.training.rollout_mbs
        out_dir, state_path = get_dynamic_sampling_save_path(save_ckpt_path, step, dp_rank)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(
            {
                "datasource": self.data_source.state_dict(),
                "ema_expansion_ratio": self.ema_expansion_ratio,
                "dp_size": dp_size,
                "rollout_mbs": rollout_mbs,
            },
            state_path,
        )
        logging_rank0(f"saved dynamic sampling state to {state_path}")

    def load_resume_state(self, step: int) -> None:
        if not self.is_mp_and_cp_head:
            return
        load_ckpt_path = self.config.checkpoint.load_ckpt_path
        assert load_ckpt_path, "dynamic sampling resume requires checkpoint.load_ckpt_path"
        dp_rank = mpu.get_data_parallel_rank()
        dp_size = mpu.get_data_parallel_world_size()
        rollout_mbs = self.config.training.rollout_mbs
        _, state_path = get_dynamic_sampling_save_path(load_ckpt_path, step, dp_rank)
        assert os.path.exists(state_path
                             ), (f"dynamic sampling resume missing state file {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        assert state["dp_size"] == dp_size, (
            f"dynamic sampling resume dp_size mismatch: saved {state['dp_size']} vs current {dp_size}"
        )
        assert state["rollout_mbs"] == rollout_mbs, (
            f"dynamic sampling resume rollout_mbs mismatch: "
            f"saved {state['rollout_mbs']} vs current {rollout_mbs}"
        )
        self.data_source.load_state_dict(state["datasource"])
        self.ema_expansion_ratio = state["ema_expansion_ratio"]
        logging_rank0(f"loaded dynamic sampling state from {state_path}")

    @override
    async def rollout_samples(
        self, data_iter, num_microbatches, curr_ppo_step, dp_rank=None, on_ready=None
    ):
        if self.run_eval:
            return await super().rollout_samples(
                data_iter,
                num_microbatches,
                curr_ppo_step,
                dp_rank=dp_rank,
                on_ready=on_ready,
            )
        raise RuntimeError("Dynamic sampling is driven by __call__, not rollout_samples")

    @override
    async def __call__(self, data_iter, num_microbatches, curr_ppo_step, on_ready=None):
        if self.run_eval:
            return await super().__call__(
                data_iter, num_microbatches, curr_ppo_step, on_ready=on_ready
            )

        timers = TimerSingleton.get_timer()
        dp_rank = mpu.get_data_parallel_rank()
        offload_process_group = self.config.training.offload_process_group
        if offload_process_group:
            destroy_process_groups()
            clear_memory()

        # TODO: pipeline sampler / gen-rm / bt-rm across microbatches for
        # disaggregated placement so RM can overlap with generation. v1 keeps
        # wave-level phases so colocate can time-share GPUs. External reward
        # starts per microbatch as generation finishes.
        rollout_mbs = self.config.training.rollout_mbs
        target_num_groups = num_microbatches * rollout_mbs
        selected_groups: List[Tuple[int, Dict[str, List[Any]]]] = []
        invalid_groups: List[Tuple[int, Dict[str, List[Any]], Any]] = []
        num_valid_groups = 0
        num_invalid_groups = 0
        num_issued_groups = 0
        num_padded_invalid_groups = 0
        num_waves = 0
        need_sample_mask = False

        for refill_idx in range(self.dynamic_config.max_refill_times + 1):
            gap = 0
            if self.is_mp_and_cp_head:
                gap = target_num_groups - len(selected_groups)
            if not self._sync_flag(gap > 0):
                break

            num_waves += 1
            timers("sampler_generate", log_level=0).start(barrier=True)
            issue_ids, orig_batches = self._get_wave_batches(data_iter, gap, dp_rank)
            num_issued_groups += sum(len(ids) for ids in issue_ids)
            await self.sampler_client.mark_ppo_step_begin(0, curr_ppo_step)
            cpu_barrier()
            if self.config.placement_type != "disaggregated":
                logging_memory_usage_details("memory tracking after sampler 0 wake_up", rank=0)
            rbs = orig_batches
            external_futs = {}
            if self.is_mp_and_cp_head and orig_batches:
                on_ready = None
                if self.config.training.use_external_reward:
                    assert self.external_reward is not None
                    cached = self.apply_sampling_rollout_attr.cached_rollout_attrs()

                    def on_ready(rbi, rb, _futs=external_futs, _cached=cached):
                        view = build_stream_ready_view(rb, _cached)
                        _futs[rbi] = asyncio.ensure_future(
                            self.external_reward.calc_external_reward(
                                [view], curr_ppo_step, is_eval=False
                            )
                        )

                rbs = await self.sampler_gen_out(
                    orig_batches,
                    0,
                    curr_ppo_step,
                    self.sample_idx,
                    self.sampling_repeat,
                    on_ready=on_ready,
                )
            cpu_barrier()
            await self.sampler_client.infer_engine_flush_cache(0)
            await self.sampler_client.mark_ppo_step_end(0, curr_ppo_step)
            cpu_barrier()
            if self.config.placement_type != "disaggregated":
                logging_memory_usage_details("memory tracking after sampler 0 sleep", rank=0)
            timers("sampler_generate").stop()

            timers("external_reward", log_level=0).start(barrier=True)
            if self.is_mp_and_cp_head and external_futs:
                ordered = [external_futs[i] for i in range(len(rbs))]
                nested = await asyncio.gather(*ordered)
                for rb, updates in zip(rbs, nested, strict=True):
                    assert len(updates) == 1
                    rb.update(updates[0])
            timers("external_reward").stop()

            wave_mb = len(rbs)
            timers("gen_rm_generate", log_level=0).start(barrier=True)
            if self.config.training.use_gen_rm_reward:
                rbs = await self.generate_gen_rm_reward(rbs, wave_mb, curr_ppo_step)
            cpu_barrier()
            timers("gen_rm_generate").stop()

            timers("bt_rm_generate", log_level=0).start(barrier=True)
            if self.config.training.use_bt_rm_reward:
                rbs = await self.calc_bt_rm_reward(rbs, wave_mb, curr_ppo_step)
            cpu_barrier()
            timers("bt_rm_generate").stop()

            if self.is_mp_and_cp_head:
                cached = self.apply_sampling_rollout_attr.cached_rollout_attrs()
                for ids, orig_batch, rollout_batch in zip(
                    issue_ids, orig_batches, rbs, strict=True
                ):
                    num_prompts = len(ids)
                    orig_prompts = self._split_rollout_batch(orig_batch, num_prompts, 1)
                    raw_groups = self._split_rollout_batch(
                        rollout_batch, num_prompts, self.sampling_repeat
                    )
                    filter_view = build_stream_ready_view(rollout_batch, cached)
                    filter_groups = self._split_rollout_batch(
                        filter_view, num_prompts, self.sampling_repeat
                    )
                    for issue_idx, orig_prompt, raw_group, filter_group in zip(
                        ids, orig_prompts, raw_groups, filter_groups, strict=True
                    ):
                        keep, reason = self._filter_group(filter_group)
                        self.filter_reasons[reason] += 1
                        if not keep:
                            num_invalid_groups += 1
                            invalid_groups.append(
                                (issue_idx, raw_group, orig_prompt["unique_id"][0])
                            )
                            continue
                        num_valid_groups += 1
                        if len(selected_groups) < target_num_groups:
                            selected_groups.append((issue_idx, raw_group))
                        else:
                            self._put_back_prompt(orig_prompt)
            self.sample_idx += wave_mb

        if self.is_mp_and_cp_head:
            need = target_num_groups - len(selected_groups)
            for i, (issue_idx, raw_group, unique_id) in enumerate(invalid_groups):
                if i < need:
                    raw_group["sample_mask"] = [
                        torch.tensor(False) for _ in range(self.sampling_repeat)
                    ]
                    selected_groups.append((issue_idx, raw_group))
                    num_padded_invalid_groups += 1
                    need_sample_mask = True
                else:
                    # filter 掉的 prompt 不会 add_back，cache 不会被 clear_data_cache 清掉
                    cache = self.apply_sampling_rollout_attr.cached_rollout_attrs()
                    assert unique_id in cache
                    del cache[unique_id]

        if offload_process_group:
            reload_process_groups()
        need_sample_mask = self._sync_flag(need_sample_mask)
        if self.is_mp_and_cp_head and need_sample_mask:
            for _, group in selected_groups:
                if "sample_mask" not in group:
                    group["sample_mask"] = [torch.tensor(True) for _ in range(self.sampling_repeat)]
        self._update_ema(num_valid_groups, num_invalid_groups)
        self._step_metrics = self._collect_step_metrics(
            num_waves,
            num_issued_groups,
            len(selected_groups),
            num_valid_groups,
            num_invalid_groups,
            num_padded_invalid_groups,
        )

        if self.is_mp_and_cp_head:
            selected_groups.sort(key=lambda item: item[0])
            groups = [group for _, group in selected_groups]
            rbs = []
            for offset in range(0, len(groups), rollout_mbs):
                microbatch_groups = groups[offset:offset + rollout_mbs]
                keys = microbatch_groups[0].keys()
                rbs.append(
                    {
                        key: _concat_values([group[key] for group in microbatch_groups])
                        for key in keys
                    }
                )
            rbs = self._post_process_rm_rollout_batch(rbs)
            assert check_rollout_batches(rbs), f"dynamic sampling output format error: {rbs=}"
        else:
            rbs = [None for _ in range(num_microbatches)]

        return rbs

    def _get_wave_batches(self, data_iter, gap: int,
                          dp_rank: int) -> Tuple[List[List[int]], List[Dict[str, List[Any]]]]:
        rollout_mbs = self.config.training.rollout_mbs
        issue_ids = []
        rollout_batches = []
        if self.is_mp_and_cp_head and gap > 0:
            num_prompts = math.ceil(
                gap * self.ema_expansion_ratio * self.dynamic_config.oversampling_ratio
            )
            num_prompts = math.ceil(num_prompts / rollout_mbs) * rollout_mbs
            num_mbs = num_prompts // rollout_mbs
            orig_batches = data_iter.take_wave(num_mbs)
            for batched_data in orig_batches:
                batched_data = self.assign_unique_id_to_batches(
                    batched_data, dp_rank, self.request_idx
                )
                self.request_idx += 1
                stripped_batch = self.remove_rollout_attr_before_sampling(batched_data)
                batch_size = len(stripped_batch["unique_id"])
                issue_ids.append(
                    list(range(self.prompt_issue_idx, self.prompt_issue_idx + batch_size))
                )
                self.prompt_issue_idx += batch_size
                rollout_batches.append(stripped_batch)

        if torch.distributed.is_initialized():
            wave_mb_t = torch.tensor([len(rollout_batches)], dtype=torch.int64)
            torch.distributed.broadcast(
                wave_mb_t,
                src=get_model_and_context_parallel_src_rank(),
                group=get_model_and_context_parallel_group_gloo(),
            )
            wave_mb = int(wave_mb_t.item())
        else:
            wave_mb = len(rollout_batches)
        if not self.is_mp_and_cp_head:
            rollout_batches = [None for _ in range(wave_mb)]
        return issue_ids, rollout_batches

    def _filter_group(self, group: Dict[str, List[Any]]) -> Tuple[bool, str]:
        if self.dynamic_filter is not None:
            keep, reason = self.dynamic_filter(self.config, group)
            return keep, reason
        if self.config.ppo.advantage_type not in [
            "grpo",
            "gdpo",
            "gdpo_sample_bn",
            "group_gdpo",
            "group_gdpo_sample_bn",
        ]:
            return True, "no_filter"
        rewards = group["rewards"]
        scalar_rewards = [reward.item() for reward in rewards]
        keep = any(reward != scalar_rewards[0] for reward in scalar_rewards[1:])
        return keep, "valid" if keep else "equal_rewards"

    def _update_ema(self, num_valid_groups: int, num_invalid_groups: int) -> None:
        if self.is_mp_and_cp_head:
            counts = torch.tensor(
                [num_valid_groups, num_invalid_groups],
                dtype=torch.float32,
                device="cuda",
            )
            torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
            num_valid_groups, num_invalid_groups = counts.tolist()
            total_groups = num_valid_groups + num_invalid_groups
            if num_valid_groups == 0:
                current_ratio = self.dynamic_config.max_expansion_ratio
            else:
                current_ratio = min(
                    total_groups / num_valid_groups,
                    self.dynamic_config.max_expansion_ratio,
                )
            next_ratio = (
                self.dynamic_config.ema_decay * self.ema_expansion_ratio +
                (1.0 - self.dynamic_config.ema_decay) * current_ratio
            )
            ratio = torch.tensor([next_ratio], dtype=torch.float32, device="cuda")
        else:
            ratio = torch.zeros(1, dtype=torch.float32, device="cuda")
        torch.distributed.broadcast(
            ratio,
            src=get_model_and_context_parallel_src_rank(),
            group=get_model_and_context_parallel_group(),
        )
        self.ema_expansion_ratio = ratio.item()

    @override
    def pop_step_metrics(self) -> Dict[str, float]:
        assert self._step_metrics, "dynamic sampling step metrics missing"
        metrics = self._step_metrics
        self._step_metrics = {}
        return metrics

    def _collect_step_metrics(
        self,
        num_waves: int,
        num_issued_groups: int,
        num_selected_groups: int,
        num_valid_groups: int,
        num_invalid_groups: int,
        num_padded_invalid_groups: int,
    ) -> Dict[str, float]:
        counts = torch.tensor(
            [
                num_issued_groups,
                num_selected_groups,
                num_valid_groups,
                num_invalid_groups,
                num_padded_invalid_groups,
                len(self.data_source.prompt_buffer) if self.is_mp_and_cp_head else 0,
            ],
            dtype=torch.float32,
            device="cuda",
        )
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts)
            gathered_reasons = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered_reasons, dict(self.filter_reasons))
            merged_reasons = Counter()
            for reasons in gathered_reasons:
                merged_reasons.update(reasons)
        else:
            merged_reasons = Counter(self.filter_reasons)
        self.filter_reasons.clear()
        issued, selected, valid, invalid, padded, buffer_size = counts.tolist()
        metrics = {
            "dynamic_sampling/ema_expansion_ratio": self.ema_expansion_ratio,
            "dynamic_sampling/num_waves": float(num_waves),
            "dynamic_sampling/issued_groups": issued,
            "dynamic_sampling/selected_groups": selected,
            "dynamic_sampling/num_valid_groups": valid,
            "dynamic_sampling/num_invalid_groups": invalid,
            "dynamic_sampling/padded_invalid_groups": padded,
            "dynamic_sampling/buffer_size": buffer_size,
        }
        for reason, count in merged_reasons.items():
            metrics[f"dynamic_sampling/filter_reason/{reason}"] = float(count)
        return metrics

    @staticmethod
    def _split_rollout_batch(
        rollout_batch: Dict[str, Any],
        num_prompts: int,
        repeat_n: int,
    ) -> List[Dict[str, Any]]:
        def _slice_value(value, start: int, end: int):
            if isinstance(value, dict):
                return {key: _slice_value(item, start, end) for key, item in value.items()}
            if isinstance(value, list):
                return value[start:end]
            return value

        groups = []
        for prompt_idx in range(num_prompts):
            start = prompt_idx * repeat_n
            end = start + repeat_n
            groups.append(
                {
                    key: _slice_value(value, start, end)
                    for key, value in rollout_batch.items()
                }
            )
        return groups

    def _sync_flag(self, flag: bool) -> bool:
        if not torch.distributed.is_initialized():
            return flag
        tensor = torch.tensor([int(flag)], dtype=torch.int64)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX, group=cpu_group())
        return bool(tensor.item())
