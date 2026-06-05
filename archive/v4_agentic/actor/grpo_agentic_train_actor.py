import asyncio
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.actor.mixin import (
    CheckpointConverterMixin,
    MetricsMixin,
    RetryActorMixin,
    RlTrainerMixin,
    TokenizerMixin,
)
from gpatch_v4.agentic.proto import DataProto
from gpatch_v4.agentic.utils import (
    LoggerAdaptor,
    compute_discounted_returns,
    compute_response_level_rewards,
)
from gpatch_v4.core.parallel_state import (
    cpu_barrier,
    init_pg,
    initlize_parallel_state,
    is_last_rank,
    is_mp_and_cp_head,
)
from gpatch_v4.orches.train_actor import BaseActor
from gpatch_v4.utils import (
    BroadcastUtils,
    TimerSingleton,
    TrainReporterSingleton,
    check_rollout_batches,
    clear_memory,
    cpu_dict,
    display_rollout_generation,
    expand_rollout_batch,
    expand_rollout_batches,
    extend_value_to_dict,
    get_iterator_k_split_list,
    import_fn_from_path,
    log,
    logging_meminfo_str,
    logging_memory_usage,
    logging_memory_usage_details,
    logging_rank0,
    masked_global_statistics_list,
    masked_mean,
    masked_mean_list,
    masked_var,
    record_time_to_metrics,
    reduce_metrics,
    sync_cuda_and_get_time,
    unbind_tensor_to_list,
)
from gpatch_v4.utils.ppo_utils import calculate_kl_penalty, create_response_mask

from .grpo_train_actor import GrpoTrainActor


def dict_of_list_to_list_of_dict(dict_of_list):
    return [dict(zip(dict_of_list.keys(), values)) for values in zip(*dict_of_list.values())]


def split_list_to_group(input_list, group_size):
    """
    将列表按照指定的组大小拆分成多个子列表

    Args:
        input_list: 要拆分的列表
        group_size: 每个子列表的大小

    Returns:
        包含子列表的列表
    """
    if not input_list:
        return []

    if group_size <= 0:
        raise ValueError("group_size must be positive")

    assert len(input_list) % group_size == 0

    return [input_list[i:i + group_size] for i in range(0, len(input_list), group_size)]


def strip_traj_sample_padding(batch: Dict[str, Any]) -> None:
    """按真实序列长度去掉 TrajEnvManager.formulate_rollouts 中 pad_to_length 的右侧 padding。

    对每条样本，在所有值为 1D tensor 的字段里取最大长度 ``max_len``：长度等于 ``max_len`` 的视为
    与整段 token 对齐（截到 ``L``）；长度等于 ``max_len - 1`` 的视为与 ``L-1`` 对齐（截到
    ``max(L-1, 0)``）。这样无需维护固定的 full_len / minus_one 键名列表。
    """
    n = len(batch["sequence_lengths"])
    for i in range(n):
        L = int(batch["sequence_lengths"][i].item())
        if L <= 0:
            continue
        tgt_m1 = max(L - 1, 0)

        key_lengths: List[Tuple[str, int]] = []
        for key, row in batch.items():
            if key == "sequence_lengths":
                continue
            if not isinstance(row, list) or i >= len(row):
                continue
            t = row[i]
            if isinstance(t, torch.Tensor) and t.dim() == 1:
                key_lengths.append((key, int(t.shape[0])))

        if not key_lengths:
            continue

        max_len = max(ln for _, ln in key_lengths)

        for key, len_t in key_lengths:
            t = batch[key][i]
            if len_t == max_len and len_t > L:
                batch[key][i] = t[:L]
            elif len_t == max_len - 1 and len_t > tgt_m1:
                batch[key][i] = t[:tgt_m1]


def strip_rollout_logprobs_to_sequence_lengths(rollout_batch: Dict[str, Any]) -> None:
    """把 ``compute_log_probs`` 得到的 logprob 裁到每条样本的真实 ``L-1`` 长度。

    forward 计算 logprob 会把所有 logprob pad 到 batch 中最长的序列长度。

    """
    seq_lens = rollout_batch.get("sequence_lengths")
    for key in ("logprobs", "ref_logprobs"):
        row = rollout_batch.get(key, None)  # 可能没有 ref_logprobs 这个 key
        for i, logp in enumerate(row):
            sl = seq_lens[i]
            L = int(sl.item()) if isinstance(sl, torch.Tensor) else int(sl)
            tgt = L - 1
            row[i] = logp[:tgt]


def compute_reinforce_return(
    token_level_rewards: torch.Tensor, gamma: torch.Tensor, lambd: torch.Tensor
):
    with torch.no_grad():
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]
        cumulative_reward = 0
        for t in reversed(range(gen_len)):
            local_reward = token_level_rewards[:, t] if t < gen_len else 0.0
            cumulative_reward = local_reward + gamma * cumulative_reward
            advantages_reversed.append(cumulative_reward)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages
    return advantages, returns


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True):
    """Whiten values with masked values."""
    #TODO: global whiten across all dp ranks
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


class GrpoAgenticTrainActor(GrpoTrainActor):
    @override
    def build_dataset_and_dataloader(self):
        self.train_dataloader = []
        self.train_sampler = None

    @override
    def maybe_set_epoch(self, epoch, reset_start_index=True):
        """Set epoch on the distributed sampler for proper shuffling.

        Parameters
        ----------
        epoch : int
            Current epoch.
        reset_start_index : bool
            Whether to reset the start_index of ResumableDistributedSampler.
            Set to False when resuming mid-epoch to preserve the skip offset.
        """
        if self.train_sampler:
            assert isinstance(self.train_sampler, DistributedSampler)
            self.train_sampler.set_epoch(epoch)
        elif getattr(self, "train_dataset",
                     None) is not None and hasattr(self.train_dataset, "set_epoch"):
            self.train_dataset.set_epoch(epoch)

    @override
    async def setup_rollout_generator(self):
        """
        Initialize the rollout generator for agentic training.
        """
        self.logger = LoggerAdaptor()
        await super().setup_rollout_generator()
        await self.train_rollout_generator.init_env_worker()

    def cal_before_filter(self, batch: DataProto) -> None:
        metrics = batch.meta_info.get("metrics", {})
        suc_traj_cnt = 0
        batch_group_by_traj: Dict[str, DataProto] = batch.group_by(keys="traj_id")
        scores = []
        for traj_id, traj_batch in batch_group_by_traj.items():
            if traj_batch.non_tensor_batch["success"][0] == False:
                continue
            episode_scores = traj_batch.non_tensor_batch["episode_scores"][0]
            suc_traj_cnt += 1
            scores.append(episode_scores)
        metrics["rollout-metrics/score/ori_mean"] = np.mean(scores).item()
        metrics["rollout-metrics/success_traj_rate"] = suc_traj_cnt / len(batch_group_by_traj)
        batch.meta_info["metrics"] = metrics
        return batch

    def filter_invalid_sample(self, data: DataProto):
        if "valid" in data.non_tensor_batch:
            print("filter invalid sample", flush=True)
            valid_index = np.where(data.non_tensor_batch["valid"])
            assert len(valid_index) == 1
            valid_index = valid_index[0]
            data = data.select_idxs(valid_index)
        return data

    def adjust_batch(self, data: DataProto, mode="copy") -> DataProto:
        """
        ref: https://github.com/langfengQ/verl-agent/blob/e03bd502667c45172e8c093cc506db8438ae8ab5/agent_system/multi_turn_rollout/utils.py#L86
        """
        global_sample_num = self.config.training.rollout_gbs * self.config.training.sampling_keep_n
        local_sample_num = global_sample_num // mpu.get_data_parallel_world_size()

        batch_size = data.batch.batch_size[0]

        if batch_size == local_sample_num:
            return data

        if mode != "random_sample":
            if batch_size > local_sample_num:
                mode = "delete"
                threshold = batch_size - local_sample_num
            else:
                mode = "copy"
                threshold = batch_size

        metrics = data.meta_info.get("metrics", {})
        metrics["rollout-metrics/batch_add_count"] = 0
        metrics["rollout-metrics/batch_remove_count"] = 0
        if mode == "delete":
            remove_indices = np.random.choice(batch_size, threshold, replace=False)
            remove_indices = np.sort(remove_indices)
            keep_mask = np.ones(batch_size, dtype=bool)
            keep_mask[remove_indices] = False
            keep_mask_tensor = torch.tensor(
                keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device
            )
            tensor_data = data.batch[keep_mask_tensor]
            non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
            adjusted_batch = DataProto(
                batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info
            )
            metrics["rollout-metrics/batch_remove_count"] = len(remove_indices)
        elif mode == "copy":
            to_add = local_sample_num - threshold
            dup_indices = np.random.choice(
                batch_size, to_add, replace=True
            ) if to_add > batch_size else np.random.choice(batch_size, to_add, replace=False)
            dup_proto = data.select_idxs(dup_indices)
            # TODO: set dup_proto response_mask to 0
            adjusted_batch = DataProto.concat([data, dup_proto])
            metrics["rollout-metrics/batch_add_count"] = to_add
        elif mode == "random_sample":
            select_indices = np.random.choice(batch_size, local_sample_num, replace=False)
            select_indices = np.sort(select_indices)
            adjusted_batch = data.select_idxs(select_indices)
            metrics["rollout-metrics/batch_remove_count"] = batch_size - local_sample_num
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        adjusted_batch.meta_info["metrics"] = metrics

        return adjusted_batch

    def traj_process(self, rollout_batches, ppo_step_i):
        """
        Process the rollout batches for agentic training.
        the data processing logic specific to agentic training is implemented here.
        """
        rollout_batches = DataProto.concat(rollout_batches)
        rollout_batches.meta_info["global_step"] = ppo_step_i
        rollout_batches = self.cal_before_filter(rollout_batches)
        rollout_batches = compute_discounted_returns(
            rollout_batches, self.config.training.agentic.adv_estimator,
            self.config.training.agentic.step_reward_gamma
        )
        rollout_batches = compute_response_level_rewards(
            batch=rollout_batches, agentic_config=self.config.training.agentic
        )

        rollout_batches = self.filter_invalid_sample(rollout_batches)

        rollout_batches = self.adjust_batch(
            rollout_batches, mode=self.config.training.agentic.batch_adjust_mode
        )
        meta_info = rollout_batches.meta_info
        batch = dict(rollout_batches.batch)

        # report score as rewards
        report_rewards = batch["response_level_rewards"]
        assert report_rewards is not None
        response_mask = batch["response_mask"]
        batch["rewards"] = report_rewards

        # bridge the batch keys
        # tokens
        batch["tokens"] = batch["input_ids"]
        batch.pop("input_ids")
        # prompt_lengths, sequence_lengths
        inputs_lengths = batch["prompt_mask"].sum(dim=-1)
        outputs_lengths = batch["response_mask"].sum(dim=-1)
        batch["prompt_lengths"] = inputs_lengths
        # 对于 traj 来讲不能用 prompt_mask + response_mask 来算 sequence_lengths
        # 因为还有中间的 tool 调用的 token
        batch["sequence_lengths"] = batch["attention_mask"].sum(dim=-1)

        metrics = rollout_batches.meta_info.get("metrics", {})

        non_tensor_batch = rollout_batches.non_tensor_batch
        # assert False, f"meta_info {meta_info.keys()} non_tensor_batch {non_tensor_batch.keys()} batch {batch.keys()}"

        rollout_mbs = self.config.training.rollout_mbs
        repeat_n = self.config.training.sampling_repeat_n
        expected_len = rollout_mbs * repeat_n

        # compute advantage here, since kl is not mixed in rewards
        advantages, returns = self.compute_advantages(batch)
        batch["advantages"] = advantages
        batch["returns"] = returns
        # TODO: add per_token_rewards
        batch["per_token_rewards"] = returns

        for k in batch.keys():
            v = batch[k]
            v = unbind_tensor_to_list(v)
            batch[k] = v
        strip_traj_sample_padding(batch)
        for k in batch.keys():
            v = batch[k]
            batch[k] = split_list_to_group(v, expected_len)

        # multi_modal data
        multi_modal_data = non_tensor_batch.get("multi_modal_data", None)
        if multi_modal_data is not None:
            multi_modal_data = multi_modal_data.tolist()
            pixel_values = [e["pixel_values"] for e in multi_modal_data]
            image_grid_thw = [e["image_grid_thw"] for e in multi_modal_data]
            batch["pixel_values"] = split_list_to_group(pixel_values, expected_len)
            batch["image_grid_thw"] = split_list_to_group(image_grid_thw, expected_len)

        batch = dict_of_list_to_list_of_dict(batch)
        return batch, metrics

    # TODO (@yeazhao) rename?
    def compute_advantages(self, rollout_batch):
        response_masks = rollout_batch["response_mask"]
        tokens = rollout_batch["tokens"]
        response_level_rewards = rollout_batch["response_level_rewards"]
        sequence_lengths = rollout_batch["sequence_lengths"]

        # compute token level rewards
        batch_size = response_masks.size(0)
        token_level_rewards = torch.zeros_like(tokens, dtype=torch.float)
        # TODO(hessianliu): double check this
        # NOTE yeazhao 这里只是把最后一个位置的token的reward设置为response_level_rewards
        token_level_rewards[torch.arange(batch_size),
                            sequence_lengths.view(-1) - 1] = response_level_rewards
        # NOTE yeazhao 这里会设置所有的token，对于grpo，这里把所有token的adv设置为response_level_rewards
        advantages, returns = compute_reinforce_return(
            token_level_rewards, self.config.training.agentic.gamma,
            self.config.training.agentic.lambd
        )
        # remove the last token
        advantages = advantages[:, :-1]
        returns = returns[:, :-1]
        # [prompt_len -1, response_len)
        response_masks = response_masks[:, 1:]
        # NOTICE: There is no group dimension for grouping (grpo),
        # and the relative advantage relies on performing whitening within a sufficiently large batch
        if self.config.training.agentic.adv_estimator == "grpo":
            # grpo
            pass
        else:
            if self.config.training.agentic.whiten_advantages:
                advantages = masked_whiten(advantages, response_masks)

        if self.config.training.agentic.mask_negative_samples:
            advantage_mask = (advantages > 0).float()
            advantages = advantages * advantage_mask

        advantages = advantages * response_masks
        # clip advantages
        if self.config.training.agentic.advantage_clip is not None:
            assert self.config.training.agentic.advantage_clip > 0
            advantages = torch.clamp(
                advantages, -self.config.training.agentic.advantage_clip,
                self.config.training.agentic.advantage_clip
            )
        return advantages, returns

    @override
    def generate_ppo_data(self, rollout_batches):
        # TODO: implement PPO data generation for agentic training
        """
        calculate advantages and masks
        """
        ppo_rollout_metrics = defaultdict(lambda: 0)
        num_samples = 0
        log('rollout generate_ppo_data', rank=0)

        mask_list = []

        for rollout_batch in rollout_batches:
            # NOTE: all items in rollout batch or out of this computation must have a leading Batch dimension
            prompt_lengths = rollout_batch["prompt_lengths"]
            sequence_lengths = rollout_batch["sequence_lengths"]
            values = rollout_batch.get("values", [None])
            if values[0] is None:
                values = None

            logprobs = rollout_batch["logprobs"]
            mask = rollout_batch.get("mask", None)
            rewards = rollout_batch.get("rewards", None)
            num_samples += len(prompt_lengths)
            if rollout_batch.get("response_mask", None) is not None:
                # mask = [torch.roll(m, shifts=-1, dims=-1) for m in rollout_batch.get("response_mask", None)]
                mask = [m[..., 1:] for m in rollout_batch.get("response_mask", None)]
            elif mask is None:
                if rewards is None:
                    mask_dtype = torch.float32
                else:
                    mask_dtype = rewards[0].dtype
                mask = create_response_mask(
                    values=logprobs,
                    prompt_lengths=prompt_lengths,
                    sequence_lengths=sequence_lengths,
                    dtype=mask_dtype,
                )
            else:
                # pad mask to logprobs length
                for i, (m, logps) in enumerate(zip(mask, logprobs, strict=True)):
                    mask[i] = torch.nn.functional.pad(
                        m,
                        (0, logps.size(-1) - m.size(-1)),
                        value=False,
                    )
            rollout_batch["mask"] = mask
            mask_list.extend(rollout_batch["mask"])

            if self.config.ppo.ppo_initial_policy_kl_penalty > 0:
                ref_logprobs = rollout_batch["ref_logprobs"]
                init_policy_kl = calculate_kl_penalty(
                    log_probs_a=logprobs,
                    log_probs_b=ref_logprobs,
                    use_absolute_kl=self.config.ppo.ppo_use_absolute_kl,
                )
            else:
                init_policy_kl = [
                    torch.tensor(0, dtype=logprobs[0].dtype, device=logprobs[0].device)
                    for _ in range(len(logprobs))
                ]

            # calculate advantage for ppo
            if self.config.ppo.advantage_type == "ppo":
                advantages, returns = self.calculate_ppo_advantages_and_returns(
                    values, rewards, None, mask, init_policy_kl, sequence_lengths
                )
                assert returns[0].dtype == torch.float32
                rollout_batch["returns"] = returns
                rollout_batch["advantages"] = advantages

            # compute metrics
            # NOTE: this metric is not accumulated globally so it will differ between DP ranks
            if self.config.ppo.ppo_initial_policy_kl_penalty > 0:
                ppo_rollout_metrics["ppo-metrics/init_policy_kl"] += masked_mean_list(
                    init_policy_kl, mask, dim=-1
                ).sum().item()
        assert check_rollout_batches(rollout_batches), f"check rbs fmt error {rollout_batches=}"
        # average across the samples for the non global metrics
        ppo_rollout_metrics = {k: v / num_samples for k, v in ppo_rollout_metrics.items()}

        for key in ["advantages", "returns", "values", 'per_token_rewards']:
            if key not in rollout_batches[0] or rollout_batches[0][key][0] is None:
                continue
            tensor_list = []
            for rollout_batch in rollout_batches:
                tensor_list.extend(rollout_batch[key])
            global_mean, global_var, min_var, max_var = masked_global_statistics_list(
                tensor_list,
                mask_list,
                key_name=key,
                group=mpu.get_data_parallel_group(),
            )
            ppo_rollout_metrics[f"ppo-metrics/global_{key}_mean"] = global_mean.item()
            ppo_rollout_metrics[f"ppo-metrics/global_{key}_std"] = global_var.sqrt().item()
            ppo_rollout_metrics[f"ppo-metrics/global_{key}_min"] = min_var.item()
            ppo_rollout_metrics[f"ppo-metrics/global_{key}_max"] = max_var.item()

        return rollout_batches, cpu_dict(ppo_rollout_metrics)

    @override
    async def rollout(
        self, epoch_i, ppo_step_i, num_rollout_micro_batches, debug_disable_advantage=False
    ):
        timers = TimerSingleton.get_timer()
        rollout_batches = []

        timers("rollout", log_level=0).start(barrier=True)
        rollout_batches = await self.train_rollout_generator(
            self.train_iter,
            num_rollout_micro_batches,
            ppo_step_i,
        )
        cpu_barrier()
        # the data processing logic specific to agentic training is implemented here.
        # other
        metrics = {}
        if is_mp_and_cp_head():
            rollout_batches, metrics = self.traj_process(rollout_batches, ppo_step_i)
        print(f"after rollout_batches metrics: {metrics}", flush=True)
        cpu_barrier()
        timers("rollout").stop()
        logging_memory_usage_details("memory tracking before bcast data", rank=0)
        rollout_batches = BroadcastUtils.broadcast_rollout_batch(rollout_batches)
        metrics = BroadcastUtils.broadcast_object_within_mp_and_cp(metrics)
        clear_memory()
        logging_memory_usage_details("memory tracking after bcast data", rank=0)
        assert check_rollout_batches(
            rollout_batches
        ), f"rollout_batches fmt error: {rollout_batches=}"
        logging_meminfo_str("CPU Memory: after get_rollout_batches: ")

        timers("compute_logps", log_level=0).start(barrier=True)
        ref_logprobs, prev_logprobs = self.policy_engine.compute_log_probs(rollout_batches)
        cpu_barrier()
        timers("compute_logps").stop()

        if not self.config.policy.without_ref:
            for rb, ref_logps in zip(rollout_batches, ref_logprobs):
                rb["ref_logprobs"] = ref_logps
        for rb, prev_logps in zip(rollout_batches, prev_logprobs, strict=True):
            rb["logprobs"] = prev_logps
        for rb in rollout_batches:
            strip_rollout_logprobs_to_sequence_lengths(rb)
        clear_memory()
        logging_memory_usage_details("memory tracking after compute_log_probs", rank=0)
        logging_meminfo_str("CPU Memory: after get_rollout_batches: ")

        expected_mbs = self.config.training.rollout_mbs * self.config.training.sampling_keep_n
        for rb in rollout_batches:
            assert len(
                rb['tokens']
            ) == expected_mbs, f"len(rb['tokens']) {len(rb['tokens'])} != {expected_mbs}"
        assert check_rollout_batches(rollout_batches), f"rbs fmt error: {rollout_batches=}"
        rollout_metrics = self.compute_rollout_metrics(rollout_batches)
        cpu_barrier()
        self.maybe_calculate_values(rollout_batches)

        timers("generate_ppo_data", log_level=0).start(barrier=True)
        rollout_batches, ppo_metrics = self.generate_ppo_data(rollout_batches)
        cpu_barrier()
        timers("generate_ppo_data").stop()

        metrics = metrics | rollout_metrics | ppo_metrics
        display_rollout_generation(self.tokenizer, self.disp_rng, rollout_batches)
        self.train_rollout_generator.clear_data_cache()
        return rollout_batches, metrics
