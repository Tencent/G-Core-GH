import gzip
import os
import shutil
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
from torch.profiler import ProfilerActivity, profile
from transformers import AutoProcessor, AutoTokenizer

from megatron.core import mpu

from gpatch_v4.configs.config import (
    FinetuneConfig,
    OffPolicyDistillConfig,
    OnPolicyDistillConfig,
)
from gpatch_v4.core.advantage_helper import (
    AdvantageContext,
    AdvantageResult,
    PostAdvantageContext,
    get_advantage_fn,
    get_post_advantage_fn,
)
from gpatch_v4.core.advantage_impl import mask_single_valid_sample_groups
from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.core.parallel_state import cpu_barrier, is_mp_and_cp_head
from gpatch_v4.core.ppo_feature_store.keys import feature_history_key
from gpatch_v4.orches import get_actor
from gpatch_v4.utils import (
    BroadcastUtils,
    allreduce_loss_across_data_parallel_group,
    average_losses_across_data_parallel_group,
    check_rollout_batches,
    cpu_dict,
    get_tokenizer_template,
    log,
    log_debug,
    logging_memory_usage,
    logging_memory_usage_details,
    logging_rank0,
    masked_global_statistics_list,
    masked_global_topk_threshold,
    masked_mean_list,
    n_times_clear_memory,
    pad_or_truncate_last_dim,
    save_data,
    sync_cuda_and_get_time,
)
from gpatch_v4.utils.flops_counter import FlopsCounter
from gpatch_v4.utils.ppo_utils import (
    align_token_level_tensors_to_logprobs,
    calculate_kl_penalty,
    count_advantage_clip_samples,
    create_response_mask,
    get_advantage_clip_bounds,
)
from gpatch_v4.utils.training_utils import whiten_advantages_cross_dp


class TokenizerMixin:
    """Mixin that builds tokenizers for various roles (actor, sampler, RM, teacher)."""
    def _build_one_tokenizer(self, model_path, use_fast):
        """Build a single HuggingFace tokenizer.

        Parameters
        ----------
        model_path : str
        use_fast : bool

        Returns
        -------
        AutoTokenizer
        """
        return AutoTokenizer.from_pretrained(model_path, use_fast=use_fast, trust_remote_code=True)

    def build_tokenizer(self):
        """Build all required tokenizers based on the current configuration.

        Populates ``self.actor_tokenizer``, ``self.rm_tokenizers``,
        ``self.gen_rm_tokenizers``, ``self.teacher_tokenizer``, and
        ``self.sampler_tokenizers``.
        """
        infer_only_mode = not hasattr(self.config,
                                      'policy') and not hasattr(self.config, 'training')
        train_config = self.config.training if not infer_only_mode else None
        self.actor_tokenizer = self._build_one_tokenizer(
            self.config.policy.hf_tokenizer_path,
            use_fast=train_config.use_fast_tokenizer,
        ) if not infer_only_mode else self._build_one_tokenizer(
            self.config.sampler.model_info[0].hf_model_path,
            use_fast=self.config.sampler.infer_engine_configs[0].use_fast_tokenizer,
        )
        self.rm_tokenizers = None
        self.gen_rm_tokenizers = None
        self.teacher_tokenizer = None
        self.sampler_tokenizers = None

        if hasattr(self.config,
                   "sampler") and getattr(self.config.sampler, "model_info", None) is not None:
            sampler_tokenizers = []
            for idx, model_info in enumerate(self.config.sampler.model_info):
                sampler_tokenizer = self._build_one_tokenizer(
                    model_info.hf_model_path,
                    use_fast=(
                        train_config.use_fast_tokenizer if not infer_only_mode else
                        self.config.sampler.infer_engine_configs[idx].use_fast_tokenizer
                    ),
                )
                sampler_tokenizers.append(sampler_tokenizer)
            self.sampler_tokenizers = sampler_tokenizers

            if infer_only_mode:
                return

        if isinstance(self.config, OffPolicyDistillConfig):
            self.teacher_tokenizer = self._build_one_tokenizer(
                self.config.teacher.hf_tokenizer_path,
                use_fast=train_config.use_fast_tokenizer,
            )
        elif isinstance(self.config, OnPolicyDistillConfig):
            first_teacher_cfg = next(iter(self.config.teachers.values()))
            tok_path = (
                first_teacher_cfg.hf_tokenizer_path
                if hasattr(first_teacher_cfg, 'hf_tokenizer_path') else
                first_teacher_cfg.get('hf_tokenizer_path')
            )
            self.teacher_tokenizer = self._build_one_tokenizer(
                tok_path,
                use_fast=train_config.use_fast_tokenizer,
            )

        if getattr(train_config, 'use_bt_rm_reward', False):
            rm_tokenizers = []
            for rm_idx, rm_model_info in enumerate(self.config.bt_rm.reward_model_info):
                rm_tokenizer = self._build_one_tokenizer(
                    rm_model_info.hf_model_path,
                    use_fast=train_config.use_fast_tokenizer,
                )
                rm_tokenizers.append(rm_tokenizer)
            self.rm_tokenizers = rm_tokenizers

        if getattr(train_config, 'use_gen_rm_reward', False):
            gen_rm_tokenizers = []
            for rm_idx, rm_model_info in enumerate(self.config.gen_rm.reward_model_info):
                gen_rm_tokenizer = self._build_one_tokenizer(
                    rm_model_info.hf_model_path,
                    use_fast=train_config.use_fast_tokenizer,
                )
                gen_rm_tokenizers.append(gen_rm_tokenizer)
            self.gen_rm_tokenizers = gen_rm_tokenizers

    def post_process_tokenizer_template(self, tokenizer, model_arch):
        if tokenizer.chat_template is None:
            tokenizer.chat_template = get_tokenizer_template(model_arch)


class T2iTokenizerMixin:
    """Mixin providing tokenizer / processor setup for T2I actors."""
    def setup_tokenizer(self, tokenizer_path):
        """Load a tokenizer for T2I tasks.

        Parameters
        ----------
        tokenizer_path : str
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    def setup_processor(self, processor_path):
        """Load an image processor for T2I tasks.

        Parameters
        ----------
        processor_path : str
        """
        self.processor = AutoProcessor.from_pretrained(processor_path, use_fast=True)


class MetricsMixin:
    """Mixin providing reward and advantage summary / cleanup utilities."""
    def _summary_reward_and_advantage_metrics(self, rollout_batches: List[Dict[str, List[Any]]]):
        """Summarize reward and advantage statistics across DP ranks.

        Parameters
        ----------
        rollout_batches : list of dict

        Returns
        -------
        dict
        """
        gen_rm_info = self.config.gen_rm.reward_model_info
        bt_rm_info = self.config.bt_rm.reward_model_info
        num_gen_rm = len(gen_rm_info) if self.config.training.use_gen_rm_reward else 0
        num_bt_rm = len(bt_rm_info) if self.config.training.use_bt_rm_reward else 0
        total_num_rm = num_gen_rm + num_bt_rm

        metrics = {}
        mean_reward_lst = []
        weighted_reward_lst = []
        advantage_mean_lst = []
        advantage_max_lst = []
        advantage_min_lst = []

        def metric_stat(val):
            val_mean = average_losses_across_data_parallel_group(val, use_gloo=True).mean()
            val_max = allreduce_loss_across_data_parallel_group(
                val, use_gloo=True, op=ReduceOp.MAX
            ).max()
            val_min = allreduce_loss_across_data_parallel_group(
                val, use_gloo=True, op=ReduceOp.MIN
            ).min()
            return val_min, val_max, val_mean

        def add_metric(key, val):
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(val)

        for rb in rollout_batches:
            mean_reward = 0.0
            weighted_reward = 0.0

            def process_one_reward(key, val, weight):
                # support multiple reward from one reward actor
                if isinstance(val, dict):
                    for (k, v) in val.items():
                        process_one_reward(f"{key}_{k}", v["rewards"], v["weight"] * weight)
                else:
                    reward_min, reward_max, reward_mean = metric_stat(val)
                    nonlocal mean_reward, weighted_reward
                    mean_reward += reward_mean
                    weighted_reward += (reward_mean * weight)
                    metric_key = f'{key}_mean'
                    add_metric(metric_key, reward_mean)
                    metric_key = f'{key}_max'
                    add_metric(metric_key, reward_max)
                    metric_key = f'{key}_min'
                    add_metric(metric_key, reward_min)

            for gen_rmi in range(num_gen_rm):

                rm_key = f"reward_gen_rm_{gen_rmi}"
                assert rm_key in rb.keys(), f'{rm_key} not in {rb.keys()}'

                process_one_reward(
                    f'reward/gen_rm_{gen_rmi}_{gen_rm_info[gen_rmi].model_arch}',
                    rb[rm_key],
                    gen_rm_info[gen_rmi].reward_weight,
                )

            for bt_rmi in range(num_bt_rm):
                rm_key = f"reward_bt_rm_{bt_rmi}"
                assert rm_key in rb.keys(), f'{rm_key} not in {rb.keys()}'
                process_one_reward(
                    f'reward/bt_rm_{bt_rmi}_{bt_rm_info[bt_rmi].model_arch}',
                    rb[rm_key],
                    bt_rm_info[bt_rmi].reward_weight,
                )

            mean_reward /= total_num_rm
            weighted_reward /= total_num_rm

            adv_for_metric = rb['advantages']
            if adv_for_metric and adv_for_metric[0].dim() == 2:
                adv_for_metric = [a.sum(dim=-1) for a in adv_for_metric]
            avg_adv = average_losses_across_data_parallel_group(adv_for_metric, use_gloo=True)
            max_adv = allreduce_loss_across_data_parallel_group(
                max(adv_for_metric), use_gloo=True, op=ReduceOp.MAX
            )
            min_adv = allreduce_loss_across_data_parallel_group(
                min(adv_for_metric), use_gloo=True, op=ReduceOp.MIN
            )
            advantage_mean_lst.append(avg_adv.mean())
            advantage_max_lst.append(max_adv)
            advantage_min_lst.append(min_adv)
            mean_reward_lst.append(mean_reward)
            weighted_reward_lst.append(weighted_reward)

        # TODO also show std
        metrics['reward/reward_mean'] = mean_reward_lst
        metrics['reward/reward_weighted'] = weighted_reward_lst
        metrics['advantages/advantages_mean'] = advantage_mean_lst
        metrics['advantages/advantages_max'] = advantage_max_lst
        metrics['advantages/advantages_min'] = advantage_min_lst
        return metrics

    def remove_raw_rewards(self, rollout_batches):
        """Remove raw reward keys from rollout batches after metric reporting.

        Parameters
        ----------
        rollout_batches : list of dict

        Returns
        -------
        list of dict
        """
        gen_rm_info = self.config.gen_rm.reward_model_info
        bt_rm_info = self.config.bt_rm.reward_model_info
        num_gen_rm = len(gen_rm_info) if self.config.training.use_gen_rm_reward else 0
        num_bt_rm = len(bt_rm_info) if self.config.training.use_bt_rm_reward else 0

        for rb in rollout_batches:
            for gen_rmi in range(num_gen_rm):
                rm_key = f"reward_gen_rm_{gen_rmi}"
                rb.pop(rm_key)

            for bt_rmi in range(num_bt_rm):
                rm_key = f"reward_bt_rm_{bt_rmi}"
                rb.pop(rm_key)

        return rollout_batches

    def report_extra_metrics(
        self,
        output_metrics: Dict[str, Any],
        final_values: Dict[str, Optional[float]],
    ) -> None:
        if not self.config.ppo.feature_store_enable:
            return
        for name, final_value in final_values.items():
            if final_value is not None:
                output_metrics[f"extra/{name}"] = final_value
            hist = self.feature_store.get(feature_history_key(name), [])
            output_metrics[f"extra/{name}_history_len"] = float(
                len(hist) if isinstance(hist, list) else 0
            )


class CheckpointConverterMixin:
    """Mixin that converts a Megatron-Core engine checkpoint to HuggingFace format."""
    async def convert_to_hf_checkpoint(self):
        """Convert the current model engine's checkpoint to HuggingFace format.

        Raises
        ------
        ValueError
            If no model engine is found or engine is not ``McoreEngine``.
        """
        model_engine = None
        if hasattr(self, "policy_engine"):
            model_engine = self.policy_engine
        elif hasattr(self, "model_engine"):
            model_engine = self.model_engine
        else:
            raise ValueError("No model engine found")
        from gpatch_v4.training_backend.megatron_backend import McoreEngine
        if not isinstance(model_engine, McoreEngine):
            raise ValueError("Model engine is not McoreEngine")

        model_engine.convert_to_hf_checkpoint()


class OffloadManager:
    """Context manager that offloads optimizer and optionally model to CPU.

    Parameters
    ----------
    model_engine : object
        Exposes ``offload_*`` / ``onload_*`` methods.
    early_swap : bool
        If *True*, also offload the model on entry.
    """
    def __init__(self, model_engine, early_swap):
        self.model_engine = model_engine
        self.early_swap = early_swap

    def __enter__(self):
        self.model_engine.offload_optimizer()
        self.model_engine.release_grad()
        if self.early_swap:
            self.model_engine.offload_model()
            logging_memory_usage_details(
                f"memory tracking before clear memory (early_swap enter)", rank=0
            )
            n_times_clear_memory(3)
            logging_memory_usage_details(
                f"memory tracking after clear memory (early_swap enter)", rank=0
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.early_swap:
            self.model_engine.offload_model()
        logging_memory_usage_details(f"memory tracking before clear memory", rank=0)
        n_times_clear_memory(3)
        logging_memory_usage_details(f"memory tracking after clear memory", rank=0)


class OnloadManager:
    """Context manager that on-loads optimizer (and optionally model) for a scope.

    Parameters
    ----------
    model_engine : object
        Exposes ``offload_*`` / ``onload_*`` methods.
    model_onloaded : bool
        If *False*, offload the model on entry before on-loading optimizer.
    """
    def __init__(self, model_engine, model_onloaded):
        self.model_engine = model_engine
        self.model_onloaded = model_onloaded

    def __enter__(self):
        if not self.model_onloaded:
            self.model_engine.offload_model()
        self.model_engine.onload_optimizer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model_engine.offload_optimizer()
        self.model_engine.offload_model()


class RlTrainerMixin:
    """Mixin providing RL-specific training utilities.

    Includes weight synchronization, sampling filtering, rollout
    metric computation, PPO data generation, and debug helpers.
    """
    async def update_weights(self, replace_zeros=False, offload=True, flush_cache=False):
        """Push updated policy weights to the sampler inference engine.

        Parameters
        ----------
        replace_zeros : bool, optional
            Debug switch.
        offload : bool, optional
            *True* (default) → wrap in ``OffloadManager``. *False* keeps
            model and optimizer on GPU (used by async rollout trainer).
        """
        ctx = OffloadManager(self.policy_engine,
                             self.config.training.early_swap_model) if offload else nullcontext()
        with ctx:
            begin_time = sync_cuda_and_get_time()
            # barrier to make sure all rank has finished offloading before wakeup inference engine
            cpu_barrier()
            update_ret = await self.sampler_client.update_weights(
                self.policy_engine, replace_zeros=replace_zeros
            )
            cpu_barrier()
            if flush_cache:
                for sampler_idx in range(self.sampler_client.num_samplers):
                    await self.sampler_client.infer_engine_flush_cache(sampler_idx)
                    cpu_barrier()
            end_time = sync_cuda_and_get_time()
            log(
                f"finished update weights to generate_engine {update_ret} using time {end_time - begin_time:.3f}s",
                rank=0
            )

    def compute_rollout_metrics(self, rollout_batches: List[Dict[str,
                                                                 List[Any]]]) -> Dict[str, float]:
        """Compute global rollout statistics across data-parallel ranks.

        Parameters
        ----------
        rollout_batches : list of dict
            Must have ``prompt_lengths``, ``sequence_lengths``, and
            optionally ``rewards``.

        Returns
        -------
        dict[str, float]
        """
        metrics = defaultdict(lambda: 0)

        num_samples = 0
        response_lengths_max = float(0)
        response_lengths_min = float(2**48)
        prompt_lengths_max = float(0)
        prompt_lengths_min = float(2**48)

        response_lengths_list = []

        _metrics_info = []
        for metric_name in self.metrics_report:
            if metric_name in rollout_batches[0].keys():
                _metrics_info.append(metric_name)

        for _, rb in enumerate(rollout_batches):
            prompt_lengths = torch.stack(rb["prompt_lengths"]).view(-1)
            sequence_lengths = torch.stack(rb["sequence_lengths"]).view(-1)
            response_lengths = sequence_lengths - prompt_lengths
            response_lengths_list.extend(response_lengths.tolist())
            metrics["sequence_lengths"] += response_lengths.sum()
            metrics["prompt_lengths"] += prompt_lengths.sum()
            if "rewards" in rb.keys():
                rewards = torch.stack(rb["rewards"]).view(-1)
                metrics["rewards"] += rewards.sum()
            else:
                metrics["rewards"] += 0.0

            response_lengths_max = max(response_lengths_max, response_lengths.max().item())
            response_lengths_min = min(response_lengths_min, response_lengths.min().item())
            prompt_lengths_max = max(prompt_lengths_max, prompt_lengths.max().item())
            prompt_lengths_min = min(prompt_lengths_min, prompt_lengths.min().item())

            for metric_name in _metrics_info:
                assert metric_name in rb.keys()
                em_v = torch.stack(rb[metric_name]).view(-1)
                metrics[metric_name] += em_v.sum()
            num_samples += prompt_lengths.size(0)

        tensor_to_accumulate = [
            metrics["sequence_lengths"],
            metrics["prompt_lengths"],
            metrics["rewards"],
            num_samples,
        ]
        for key_name in _metrics_info:
            tensor_to_accumulate.append(metrics[key_name])
        tensor_to_accumulate = torch.tensor(
            tensor_to_accumulate,
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        torch.distributed.all_reduce(tensor_to_accumulate, group=mpu.get_data_parallel_group())

        tensor_to_max = torch.tensor(
            [
                response_lengths_max,
                -response_lengths_min,
                prompt_lengths_max,
                -prompt_lengths_min,
            ],
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        torch.distributed.all_reduce(
            tensor_to_max,
            group=mpu.get_data_parallel_group(),
            op=torch.distributed.ReduceOp.MAX,
        )
        (
            response_lengths_max,
            response_lengths_min,
            prompt_lengths_max,
            prompt_lengths_min,
        ) = tensor_to_max.tolist()
        response_lengths_min = -response_lengths_min
        prompt_lengths_min = -prompt_lengths_min

        tensor_to_accumulate_as_list = tensor_to_accumulate.tolist()
        (
            global_response_lengths,
            global_prompt_lengths,
            global_rewards,
            global_num_samples,
        ) = tensor_to_accumulate_as_list[:4]
        metrics = {
            "rollout-metrics/global_response_lengths_mean":
                global_response_lengths / global_num_samples,
            "rollout-metrics/global_prompt_lengths":
                global_prompt_lengths / global_num_samples,
            "rollout-metrics/global_response_lengths_max":
                response_lengths_max,
            "rollout-metrics/global_response_lengths_min":
                response_lengths_min,
            "rollout-metrics/global_prompt_lengths_max":
                prompt_lengths_max,
            "rollout-metrics/global_prompt_lengths_min":
                prompt_lengths_min,
        }
        metrics["rollout-rewards/global_rewards"] = global_rewards / global_num_samples
        for em_i, metric_name in enumerate(_metrics_info):
            prefix = "rollout-metrics"
            if "reward" in metric_name:
                prefix = "rollout-rewards"
            metrics[f"{prefix}/global_{metric_name}"] = tensor_to_accumulate_as_list[
                4 + em_i] / global_num_samples

        dp_size = mpu.get_data_parallel_world_size()
        gathered_lengths = [None] * dp_size
        torch.distributed.all_gather_object(
            gathered_lengths, response_lengths_list, group=mpu.get_data_parallel_group()
        )
        all_response_lengths = sorted(
            length for rank_list in gathered_lengths for length in rank_list
        )
        if all_response_lengths:
            n = len(all_response_lengths)

            def _percentile(p):
                return all_response_lengths[max(0, min(n - 1, int(p * n)))]

            metrics["rollout-metrics/global_response_lengths_p10"] = _percentile(0.1)
            metrics["rollout-metrics/global_response_lengths_p50"] = _percentile(0.5)
            metrics["rollout-metrics/global_response_lengths_p90"] = _percentile(0.9)
            metrics["rollout-metrics/global_response_lengths_max_cnt"] = sum(
                1 for x in all_response_lengths if x == all_response_lengths[-1]
            )
            metrics["rollout-metrics/global_response_lengths_min_cnt"] = sum(
                1 for x in all_response_lengths if x == all_response_lengths[0]
            )

        return cpu_dict(metrics)

    def generate_ppo_data(
        self, rollout_batches: List[Dict[str, List[Any]]]
    ) -> Tuple[List[Dict[str, List[Any]]], Dict[str, float]]:
        """Calculate advantages, returns, and masks for PPO training.

        Parameters
        ----------
        rollout_batches : list of dict
            Must contain logprobs, rewards, etc.

        Returns
        -------
        tuple[list[dict], dict[str, float]]
            ``(processed_rollout_batches, ppo_rollout_metrics)``.
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

            rewards = rollout_batch.get("rewards", [None])
            per_token_rewards = rollout_batch.get("per_token_rewards", [None])
            if rewards[0] is None:
                rewards = None
            if per_token_rewards[0] is None:
                per_token_rewards = None

            # rewards = rollout_batch["rewards"]
            # per_token_rewards = rollout_batch["per_token_rewards"]

            logprobs = rollout_batch["logprobs"]
            mask = rollout_batch.get("mask", None)
            num_samples += len(prompt_lengths)

            # Internal critic values are already on the S-1 logprob axis.
            # External full-token values are aligned here instead of in an
            # RPC client so every rollout path follows the same contract.
            if values is not None and not self.require_critic_model():
                values = align_token_level_tensors_to_logprobs(
                    values,
                    logprobs,
                    sequence_lengths,
                    self.config.ppo.ppo_value_truncate_head,
                )
                rollout_batch["values"] = values

            gdpo_rewards = None
            if self.config.ppo.advantage_type in [
                "gdpo",
                "gdpo_sample_bn",
                "group_gdpo",
                "group_gdpo_sample_bn",
            ]:
                gdpo_rewards = {}
                for k, v in self.config.ppo.gdpo_reward_weights.items():
                    assert k in rollout_batch.keys(
                    ), f"{k} not in rollout batch {rollout_batch.keys()}"
                    gdpo_rewards[k] = rollout_batch[k]

            if mask is None:
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
            # 把sample mask为0的样本的mask置为0
            sample_mask = rollout_batch.get("sample_mask", None)
            # post check for sample_mask
            if sample_mask is not None:
                assert self.config.training.train_mbs == 1, "sample_mask only support train_mbs == 1"
                assert self.config.policy.dynamic_mbs_target_seqlen is None, "sample_mask only support dynamic_mbs_target_seqlen is None"
                assert self.config.policy.dynamic_mbs_limit is None, "sample_mask only support dynamic_mbs_limit is None"

            valid_indices = []
            if sample_mask is not None:
                for i, (sm, m) in enumerate(zip(sample_mask, mask, strict=True)):
                    if 0 == sm.item():  # sm is tensor(0.) or tensor(1.)
                        m.zero_()
                    else:
                        valid_indices.append(i)
            rollout_batch["mask"] = mask
            mask_list.extend(rollout_batch["mask"])

            if per_token_rewards is not None:
                per_token_rewards = align_token_level_tensors_to_logprobs(
                    per_token_rewards,
                    logprobs,
                    sequence_lengths,
                    self.config.ppo.ppo_value_truncate_head,
                )
                rollout_batch["per_token_rewards"] = per_token_rewards

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

            if self.require_critic_model():
                assert values[0].shape == logprobs[
                    0
                ].shape, f"values {values[0].shape} and logprobs {logprobs[0].shape} should have the same shape"

            if self.config.ppo.advantage_type in [
                "grpo",
                "gdpo",
                "gdpo_sample_bn",
                "group_gdpo",
                "group_gdpo_sample_bn",
            ]:
                # boundary protection — if only 1 valid sample remains in the group,
                # std() would return NaN, so we discard the entire group.
                if sample_mask is not None:
                    mask_single_valid_sample_groups(
                        mask,
                        sample_mask,
                        self.config.training.sampling_keep_n,
                    )
                    rollout_batch["mask"] = mask
                    rollout_batch["sample_mask"] = sample_mask
            advantage_fn = get_advantage_fn(self.config.ppo.advantage_type)
            ctx = AdvantageContext(
                rollout_batch=rollout_batch,
                config=self.config,
                mask=mask,
                logprobs=logprobs,
                rewards=rewards,
                sample_mask=sample_mask,
                values=values,
                per_token_rewards=per_token_rewards,
                init_policy_kl=init_policy_kl,
                sequence_lengths=sequence_lengths,
                prompt_lengths=prompt_lengths,
                gdpo_rewards=gdpo_rewards,
            )
            result: AdvantageResult = advantage_fn(ctx)

            if result.pre_bn_advantages is not None:
                # Deferred: store pre-BN scalars for post-loop processing
                rollout_batch["pre_bn_advantages"] = result.pre_bn_advantages
            else:
                # Final advantages ready — store directly
                advantages = result.advantages
                if result.returns is not None:
                    rollout_batch["returns"] = result.returns
                if result.metrics and result.metrics_prefix:
                    for k, v in result.metrics.items():
                        ppo_rollout_metrics[f"{result.metrics_prefix}/{k}"] += v

                assert advantages[0].dtype == torch.float32

                # 打印advantages的形状
                log_debug(f"[GrpoTrainActor] advantages shape: {advantages[0].shape}", rank=0)

                # ---- whiten advantages globally across DP ranks ----
                # 对所有 DP rank 上 batch 内的有效 response token 做统一的
                # mean/std 归一化；whiten 必须在 advantage_clip 之前。
                # 注意：此处假设所有 DP rank 在 generate_ppo_data 中
                # 迭代的 rollout_batch 数量一致（否则 all_reduce 会卡住）。
                if self.config.ppo.whiten_advantages and mask is not None:
                    assert self.config.ppo.advantage_type in [
                        "identity", "reinforce", "ppo", "grpo"
                    ], "whiten_advantages only support identity, reinforce and ppo"
                    advantages, whiten_metrics = whiten_advantages_cross_dp(advantages, mask)
                    for k, v in whiten_metrics.items():
                        ppo_rollout_metrics[k] += v

                advantage_clip_bounds = get_advantage_clip_bounds(
                    self.config.ppo.advantage_clip,
                    self.config.ppo.advantage_clip_lower_bound,
                    self.config.ppo.advantage_clip_upper_bound,
                )
                if advantage_clip_bounds is not None:
                    advantage_clip_lower_bound, advantage_clip_upper_bound = advantage_clip_bounds
                    rollout_batch["original_advantages"] = advantages
                    advantages = [
                        a.clamp(min=advantage_clip_lower_bound, max=advantage_clip_upper_bound)
                        for a in advantages
                    ]
                rollout_batch["advantages"] = advantages

            # NOTE: this metric is not accumulated globally so it will differ between DP ranks
            if self.config.ppo.ppo_initial_policy_kl_penalty > 0:
                ppo_rollout_metrics["ppo-metrics/init_policy_kl"] += masked_mean_list(
                    init_policy_kl, mask, dim=-1
                ).sum().item()

        # ---- Post-advantage processing (e.g. GDPO global BN) ----
        post_advantage_fn = get_post_advantage_fn(self.config.ppo.advantage_type)
        if post_advantage_fn is not None:
            post_ctx = PostAdvantageContext(
                rollout_batches=rollout_batches,
                config=self.config,
                dp_group=mpu.get_data_parallel_group(),
                num_samples=num_samples,
            )
            post_result = post_advantage_fn(post_ctx)
            rollout_batches = post_result.rollout_batches
            ppo_rollout_metrics.update(post_result.metrics)

        assert check_rollout_batches(rollout_batches), f"check rbs fmt error {rollout_batches=}"
        # average across the samples for the non global metrics
        ppo_rollout_metrics = {k: v / num_samples for k, v in ppo_rollout_metrics.items()}

        global_stats = self.compute_ppo_global_statistics(rollout_batches)
        ppo_rollout_metrics.update(global_stats)

        return rollout_batches, cpu_dict(ppo_rollout_metrics)

    def compute_ppo_global_statistics(
        self, rollout_batches: List[Dict[str, List[Any]]]
    ) -> Dict[str, float]:
        """Compute global statistics over PPO data fields across DP ranks.

        Gathers ``mask``-weighted mean/std/min/max for ``advantages``,
        ``returns``, ``values``, ``per_token_rewards``, and the global
        ``sample_mask`` mean. When ``original_advantages`` is present
        (advantage clip enabled), also reports
        ``advantage_clip_{lower,upper}_sample_frac``. Does NOT recompute
        advantages — only reads what is already stored in *rollout_batches*.

        Parameters
        ----------
        rollout_batches : list of dict
            Must already contain ``mask``, ``advantages``, and optionally
            ``returns``, ``values``, ``per_token_rewards``, ``sample_mask``,
            ``original_advantages``.

        Returns
        -------
        dict[str, float]
        """
        metrics: Dict[str, float] = {}
        mask_list = []
        for rb in rollout_batches:
            mask_list.extend(rb["mask"])

        for key in ["advantages", "original_advantages", "returns", "values", "per_token_rewards"]:
            if key not in rollout_batches[0] or rollout_batches[0][key][0] is None:
                continue
            tensor_list = []
            for rb in rollout_batches:
                tensor_list.extend(rb[key])
            if tensor_list and tensor_list[0].dim() == 2:
                tensor_list = [t.sum(dim=-1) for t in tensor_list]
            global_mean, global_var, min_var, max_var = masked_global_statistics_list(
                tensor_list,
                mask_list,
                key_name=key,
                group=mpu.get_data_parallel_group(),
            )
            metrics[f"ppo-metrics/global_{key}_mean"] = global_mean.item()
            metrics[f"ppo-metrics/global_{key}_std"] = global_var.sqrt().item()
            metrics[f"ppo-metrics/global_{key}_min"] = min_var.item()
            metrics[f"ppo-metrics/global_{key}_max"] = max_var.item()

        if "sample_mask" in rollout_batches[0]:
            sample_mask_list = []
            for rb in rollout_batches:
                sample_mask_list.extend(rb["sample_mask"])
            sample_mask_tensor = torch.stack(sample_mask_list).view(-1)
            sum_and_count = torch.tensor(
                [sample_mask_tensor.sum(), sample_mask_tensor.numel()],
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
            torch.distributed.all_reduce(sum_and_count, group=mpu.get_data_parallel_group())
            global_mean = sum_and_count[0] / sum_and_count[1]
            metrics["ppo-metrics/global_sample_mask_mean"] = global_mean.item()

            retention_value = global_mean.item()
            if retention_value <= 0.0:
                retention_value = 1.0
            ref_dtype = sample_mask_tensor.dtype
            for rb in rollout_batches:
                n = len(rb["sample_mask"])
                rb["global_retention_ratio"] = [
                    torch.tensor(retention_value, dtype=ref_dtype) for _ in range(n)
                ]

        if "original_advantages" in rollout_batches[0]:
            orig_list = []
            adv_list = []
            for rb in rollout_batches:
                orig_list.extend(rb["original_advantages"])
                adv_list.extend(rb["advantages"])
            n_lower, n_upper, n_samples = count_advantage_clip_samples(
                orig_list, adv_list, mask_list
            )
            clip_counts = torch.tensor(
                [n_lower, n_upper, n_samples],
                dtype=torch.float64,
                device=torch.cuda.current_device(),
            )
            torch.distributed.all_reduce(clip_counts, group=mpu.get_data_parallel_group())
            global_n = clip_counts[2].item()
            if global_n > 0:
                lower_frac = clip_counts[0].item() / global_n
                upper_frac = clip_counts[1].item() / global_n
            else:
                lower_frac = 0.0
                upper_frac = 0.0
            metrics["ppo-metrics/advantage_clip_lower_sample_frac"] = lower_frac
            metrics["ppo-metrics/advantage_clip_upper_sample_frac"] = upper_frac
            metrics["ppo-metrics/advantage_clip_sample_frac"] = lower_frac + upper_frac

        if (
            self.config.ppo.ppo_entropy_global_cov and
            self.config.ppo.ppo_entropy_regularization_type in ("clip-cov", "kl-cov")
        ):
            self._attach_entropy_aux_figures(rollout_batches, mask_list)

        return metrics

    def _attach_entropy_aux_figures(
        self, rollout_batches: List[Dict[str, List[Any]]], mask_list: List[torch.Tensor]
    ) -> None:
        """Compute global covariance figures and attach to each rollout batch.

        Stores a per-sample ``entropy_aux_figures`` tensor of shape ``(3,)`` =
        ``[global_mean_adv, global_mean_logp, kl_cov_threshold]`` so the loss can
        center the advantage-logprob covariance on the global (train_gbs,
        cross-DP) mean and, for kl-cov, select the true global top-rho% via the
        threshold. The covariance log-prob is ``prev_logp`` (the ``logprobs``
        field), or ``rollout_log_probs`` when ``skip_prev_logps`` is set.

        Stored per-sample (a list of identical ``(3,)`` tensors) because the
        downstream batch is a list of per-sample dicts (see
        :func:`expand_rollout_batch`); this mirrors ``global_retention_ratio``.
        """
        ppo = self.config.ppo
        dp_group = mpu.get_data_parallel_group()
        logp_key = "rollout_log_probs" if ppo.skip_prev_logps else "logprobs"

        adv_list: List[torch.Tensor] = []
        logp_list: List[torch.Tensor] = []
        for rb in rollout_batches:
            adv_list.extend(rb["advantages"])
            # Align logp to the response-mask length: rollout_log_probs is one
            # token longer than the seqlen-1 response convention.
            for lp, m in zip(rb[logp_key], rb["mask"], strict=True):
                logp_list.append(pad_or_truncate_last_dim(lp, m.shape[-1], 0))

        mean_adv, _, _, _ = masked_global_statistics_list(
            adv_list, mask_list, key_name="entropy_cov_adv", group=dp_group
        )
        mean_logp, _, _, _ = masked_global_statistics_list(
            logp_list, mask_list, key_name="entropy_cov_logp", group=dp_group
        )

        tau = float("inf")
        if ppo.ppo_entropy_regularization_type == "kl-cov" and ppo.ppo_kl_cov_ratio > 0:
            device = torch.cuda.current_device()
            adv_cat = torch.cat([a.view(-1) for a in adv_list]).to(device, torch.float32)
            logp_cat = torch.cat([l.view(-1) for l in logp_list]).to(device, torch.float32)
            mask_cat = torch.cat([m.view(-1) for m in mask_list]).to(device)
            count_t = torch.tensor([mask_cat.sum()], dtype=torch.float32, device=device)
            torch.distributed.all_reduce(count_t, group=dp_group)
            n_global = int(count_t.item())
            if n_global > 0:
                cov = (adv_cat - mean_adv) * (logp_cat - mean_logp)
                k_global = max(1, int(ppo.ppo_kl_cov_ratio * n_global))
                tau = masked_global_topk_threshold(cov, mask_cat, k_global, group=dp_group).item()

        figures = torch.tensor([mean_adv.item(), mean_logp.item(), tau], dtype=torch.float32)
        for rb in rollout_batches:
            n = len(rb["mask"])
            rb["entropy_aux_figures"] = [figures.clone() for _ in range(n)]
        logging_rank0(f"entropy_aux_figures: {figures}")

    async def _debug_update_weight_stage1(self, sampler_idx):
        # debug funcion
        assert self.config.debug.debug_engine_update_weight is True
        save_path = self.config.debug.debug_engine_save_path
        os.makedirs(save_path, exist_ok=True)

        # save the src weight
        await self.sampler_client.test_save_engine_ckpt(sampler_idx, f"{save_path}/src_weights")
        cpu_barrier()
        await self.sampler_client.test_generate(sampler_idx)
        cpu_barrier()
        # disaggregated 下 sampler 常驻，sleep/wake_up 不可用；跳过即可。
        if self.config.placement_type != "disaggregated":
            await self.sampler_client.sleep(sampler_idx)
            cpu_barrier()
        logging_memory_usage_details(f"memory tracking after stage1", rank=0)

    async def _debug_update_weight_stage2(self, sampler_idx):
        assert self.config.debug.debug_engine_update_weight is True
        save_path = self.config.debug.debug_engine_save_path
        os.makedirs(save_path, exist_ok=True)

        # replace zero
        logging_rank0("debug_update replace zeros weights")
        await self.update_weights(replace_zeros=True)
        cpu_barrier()

        await self.sampler_client.mark_ppo_step_begin(sampler_idx, 0)
        cpu_barrier()

        await self.sampler_client.test_generate(sampler_idx)
        cpu_barrier()
        await self.sampler_client.test_save_engine_ckpt(sampler_idx, f"{save_path}/zero_weights")
        cpu_barrier()
        if self.config.placement_type != "disaggregated":
            await self.sampler_client.sleep(sampler_idx)
            cpu_barrier()
        logging_memory_usage_details(f"memory tracking after zero update", rank=0)

        self.policy_engine.onload_optimizer()
        self.policy_engine.onload_model()

        logging_rank0("debug_update replace real weights")
        await self.update_weights()
        cpu_barrier()

        await self.sampler_client.mark_ppo_step_begin(sampler_idx, 0)
        cpu_barrier()
        await self.sampler_client.test_generate(sampler_idx)
        cpu_barrier()
        await self.sampler_client.test_save_engine_ckpt(sampler_idx, f"{save_path}/real_weights")
        cpu_barrier()
        if self.config.placement_type != "disaggregated":
            await self.sampler_client.sleep(sampler_idx)
            cpu_barrier()
        logging_memory_usage_details(f"memory tracking after real update", rank=0)

    async def debug_update_weight(self, stage):
        """Debug helper to exercise the weight-update path.

        Parameters
        ----------
        stage : int
            ``1`` → save + generate; ``2`` → zero-replace + real-replace.
        """
        sampler_idx = 0
        if stage == 1:
            # 这里不能 将 sampler offload 下去
            await self._debug_update_weight_stage1(sampler_idx)
        elif stage == 2:
            await self._debug_update_weight_stage2(sampler_idx)
        else:
            assert False, "no support this stage"

    def require_critic_model(self):
        """``True`` when the advantage type requires a critic model.

        Returns
        -------
        bool
        """
        return self.config.ppo.advantage_type in ["ppo"]


class TestActorMixin:
    """Mixin providing debug / test utilities for distributed communication."""
    async def debug_scatter_and_gather(self):
        """Debug helper that exercises scatter and gather across MP/CP group."""
        from gpatch_v4.core.parallel_state import (
            get_model_and_context_parallel_group,
            get_model_and_context_parallel_group_gloo,
        )

        # group = get_model_and_context_parallel_group_gloo()
        group = get_model_and_context_parallel_group()
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank()

        group_ranks = dist.get_process_group_ranks(group)
        source_rank = group_ranks[0]

        begin_time = sync_cuda_and_get_time()
        for ti in range(16):
            t1 = sync_cuda_and_get_time()
            if is_mp_and_cp_head():
                seq_len = 32 * 1024
                vocab_size = 151936
                logits = torch.randn((seq_len, vocab_size), dtype=torch.float32, device="cuda")
                meta_info = [[seq_len, vocab_size, logits.dtype]]
            else:
                logits = None
                meta_info = [None]
            meta_info = BroadcastUtils.broadcast_rollout_batch(meta_info)
            seq_len, vocab_size, dtype = meta_info[0]

            base_shard_size = vocab_size // world_size
            if rank < world_size - 1:
                local_vocab_size = base_shard_size
            else:
                local_vocab_size = vocab_size - base_shard_size * (world_size - 1)
            t2 = sync_cuda_and_get_time()
            if rank == source_rank:
                scatter_list = []
                for i in range(world_size):
                    start = i * base_shard_size
                    end = (i + 1) * base_shard_size if i < world_size - 1 else vocab_size
                    scatter_list.append(logits[:, start:end].contiguous())
            else:
                scatter_list = None

            local_logits_cpu = torch.empty(
                (seq_len, local_vocab_size),
                dtype=dtype,
                device="cuda",
            )
            dist.scatter(
                tensor=local_logits_cpu,
                scatter_list=scatter_list,
                src=source_rank,
                group=group,
            )
            t3 = sync_cuda_and_get_time()
            log(f"scatter {ti} time: {t2 - t1} {t3 - t2} sum {t3 - t1}")
        end_time = sync_cuda_and_get_time()
        log(f"scatter time: {end_time - begin_time}")
        save_data(local_logits_cpu, "debug-tmp", f"local_logits_cpu_{rank}.pt")
        if rank == source_rank:
            save_data(logits, "debug-tmp", f"logits_{rank}.pt")

        dst_rank = group_ranks[-1]
        gather_list = None
        if rank == dst_rank:
            gather_list = [None for _ in range(world_size)]

        dist.gather_object(
            obj=local_logits_cpu,
            object_gather_list=gather_list,
            dst=dst_rank,
            group=group,
        )
        if rank == dst_rank:
            full_logits = torch.cat(gather_list, dim=1)
            save_data(full_logits, "debug-tmp", f"gather_logits_{rank}.pt")

        return True

    async def test_ray_rpc(self):
        """Debug helper that tests Ray RPC data transfer to a teacher actor."""
        import ray

        test_dtype = "tensor"
        teacher_node = "teacher_1_7"
        dtype = torch.bfloat16
        b = 32
        s = 16 * 1024
        v = 151936
        shape_meta = [b, s, v]
        ray_actor = ray.get_actor(teacher_node)
        log(f"test_ray_rpc_data {test_dtype} {dtype} {teacher_node}")
        if torch.distributed.get_rank() == 0:
            t1 = sync_cuda_and_get_time()
            ret = await ray_actor.test_ray_rpc_data.remote(test_dtype, dtype, shape_meta)
            t2 = sync_cuda_and_get_time()
            log(f"ray get time: {t2 - t1}")
            log(f"ret: {ret.sum()}")
        cpu_barrier()


class RetryActorMixin:
    """Mixin tracking training progress for deadlock detection."""
    def retry_actor_init(self):
        """Initialize retry-related state variables."""
        self.last_progress_time = None
        self.train_step_finished = False

    def get_node_ip(self) -> str:
        """IP address of the node this actor is running on.

        Used for failure attribution — when an actor loses liveness, the
        trainer uses this to identify the faulty node.

        Returns
        -------
        str
        """
        from gpatch_v4 import orches
        return orches.get_node_ip()

    def is_train_step_finished(self):
        """``True`` once training is done.

        Returns
        -------
        bool
        """
        return self.train_step_finished

    def check_liveness(self):
        """``True`` if no progress for longer than ``max_train_step_waiting_time``.

        Returns
        -------
        bool
        """
        if self.last_progress_time is None or self.train_step_finished or self.config.training.max_train_step_waiting_time is None:
            return False

        if (
            time.time() - self.last_progress_time
        ) > self.config.training.max_train_step_waiting_time:
            return True

        return False


class ProfileMixin:
    """Mixin providing PyTorch profiler integration for training actors."""
    def setup_profile(self):
        """Initialize the profiler based on the report configuration."""
        self.profile_config = self.config.report.profile
        if self.profile_config.enable_profile:
            self.prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=bool(self.profile_config.record_shapes),
                with_stack=bool(self.profile_config.with_stack),
                profile_memory=bool(self.profile_config.profile_memory),
                with_modules=bool(self.profile_config.with_modules),
            )
            logging_rank0(f"profile config: {self.profile_config}")

        else:
            self.prof = None
        self.init_profile_flag = True

    def _should_export_trace(self) -> bool:
        export_ranks = list(self.profile_config.export_ranks)
        if len(export_ranks) == 0:
            return True
        return torch.distributed.get_rank() in export_ranks

    def _nsys_enabled(self) -> bool:
        return self.profile_config.use_nsys

    def profile_start(self, train_step):
        """Start profiling if ``train_step`` matches the configured start step.

        Parameters
        ----------
        train_step : int
        """
        assert getattr(self, "init_profile_flag", False), "profile not initialized"
        if self.profile_config.enable_profile and train_step == self.profile_config.profile_start_step:
            self.prof.start()
        if self._nsys_enabled() and train_step == self.profile_config.profile_start_step:
            torch.cuda.profiler.start()

    def profile_end(self, train_step, save_name="timeline"):
        """Stop profiling and export a Chrome trace if ``train_step`` matches.

        Parameters
        ----------
        train_step : int
        save_name : str, optional
        """
        assert getattr(self, "init_profile_flag", False), "profile not initialized"
        if self._nsys_enabled() and train_step == self.profile_config.profile_end_step:
            torch.cuda.profiler.stop()
        if not (
            self.profile_config.enable_profile and
            train_step == self.profile_config.profile_end_step
        ):
            return

        self.prof.stop()
        self._log_profile_summary(train_step)

        if not self._should_export_trace():
            return

        profile_dir = self.profile_config.profile_save_dir
        os.makedirs(profile_dir, exist_ok=True)
        rank = torch.distributed.get_rank()
        json_path = (f"{profile_dir}/{save_name}_rank_{rank}_step_{train_step}.json")
        self.prof.export_chrome_trace(json_path)
        out_path = json_path
        if self.profile_config.gzip_trace:
            gz_path = json_path + ".gz"
            with open(json_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(json_path)
            out_path = gz_path
        try:
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
        except OSError:
            size_mb = -1.0
        log(f"[profile] exported {out_path} ({size_mb:.1f} MiB)")

    def _log_profile_summary(self, train_step):
        """Print a kernel-level CUDA-time summary and a comm-vs-compute split.

        Device-side kernel durations are unaffected by profiler CPU overhead, so
        the totals here are reliable for A/B comparison across images even when
        wall-clock is inflated by profiling. Rank 0 only; never raises.
        """
        if torch.distributed.get_rank() != 0:
            return
        try:
            events = self.prof.key_averages()
            comm_markers = (
                "nccl",
                "allgather",
                "all_gather",
                "reducescatter",
                "reduce_scatter",
                "alltoall",
                "all_to_all",
                "allreduce",
                "all_reduce",
                "broadcast",
                "c10d",
                "reduce_kernel",
            )

            def _dev_us(evt):
                # torch>=2.1 renamed self_cuda_time_total -> self_device_time_total
                for attr in ("self_device_time_total", "self_cuda_time_total"):
                    v = getattr(evt, attr, 0.0)
                    if v:
                        return float(v)
                return 0.0

            comm_us = 0.0
            compute_us = 0.0
            for evt in events:
                cuda_us = _dev_us(evt)
                if cuda_us <= 0:
                    continue
                name = evt.key.lower()
                if any(m in name for m in comm_markers):
                    comm_us += cuda_us
                else:
                    compute_us += cuda_us
            total_us = comm_us + compute_us
            try:
                table = events.table(sort_by="self_device_time_total", row_limit=30)
            except Exception:
                table = events.table(sort_by="self_cuda_time_total", row_limit=30)
            pct = (comm_us / total_us * 100.0) if total_us > 0 else 0.0
            print(
                f"[profile_summary] step={train_step} "
                f"total_cuda={total_us / 1e3:.2f}ms "
                f"compute={compute_us / 1e3:.2f}ms "
                f"comm={comm_us / 1e3:.2f}ms ({pct:.1f}% comm)\n{table}",
                flush=True,
            )
        except Exception as e:  # never break training because of profiling
            print(f"[profile_summary] failed: {e}", flush=True)


class TrainingPltMixin:
    class TrainType:
        SFT = "sft"
        GRPO = "grpo"
        DPO = "dpo"

    class TrainState:
        TRAIN_START = "train_start"
        TRAIN_END = "train_end"
        TRAIN_STEP = "train_step"
        EVAL_START = "eval_start"
        EVAL_END = "eval_end"

    def training_plt_init(self):
        self.training_plt_actor = None
        self.training_plt_once = True

    def _get_actor(self):
        self.training_plt_once = False
        try:
            self.training_plt_actor = get_actor("training_plt")
        except ValueError:
            self.training_plt_actor = None

    def training_plt_report(
        self,
        actor: str,
        model_arch: str,
        train_type: TrainType,
        train_state: TrainState,
        data: dict,
    ):
        if self.training_plt_once:
            self._get_actor()

        if self.training_plt_actor is not None:
            data = {
                "actor": actor,
                "model_arch": model_arch,
                "train_type": train_type,
                "train_state": train_state,
                "tp_rank": mpu.get_tensor_model_parallel_rank(),
                "pp_rank": mpu.get_pipeline_model_parallel_rank(),
                "cp_rank": mpu.get_context_parallel_rank(),
                "dp_rank": mpu.get_data_parallel_rank(),
                "time": time.time(),
                "data": data,
            }
            self.training_plt_actor.add_data.remote(data)


class FlopsCounterMixin:
    """Mixin providing FLOPS counter for training actors."""
    def flops_counter_init(self, hf_config):
        """Initialize the FLOPS counter.

        Parameters
        ----------
        hf_config : transformers.PretrainedConfig
        """
        self.flops_counter = None
        if self.config.training.calc_mfu_freq:
            assert self.config.training.calc_mfu_freq > 0
            self.calc_mfu_freq = self.config.training.calc_mfu_freq
            self.image_grid_thw_name = self.config.training.image_grid_thw_name
            self.mfu_sum = 0
            self.mfu_count = 0
            self.flops_counter = FlopsCounter(hf_config, self.config.policy.model_arch)

    def flops_counter_calc(
        self,
        train_step,
        expanded_rbs,
        delta_time,
        seq_length,
        seqlen_sum=None,
        seqlen_sq_sum=None,
    ):
        """Calculate the MFU based on the FLOPS counter.

        If both ``seqlen_sum`` and ``seqlen_sq_sum`` are supplied, FLOPs are
        estimated from those global-batch aggregates; otherwise every sample
        is assumed to be ``seq_length`` tokens long.

        Parameters
        ----------
        train_step : int
        expanded_rbs : list
        delta_time : float
            Seconds to process the batch.
        seq_length : int
            Uniform sequence length assumed when exact sums are not supplied.
        seqlen_sum : int | float, optional
            ``Σ S`` over the global batch.
        seqlen_sq_sum : int | float, optional
            ``Σ S²`` over the global batch.

        Returns
        -------
        tuple
            ``(mfu, avg_mfu)``.
        """

        mfu = None
        avg_mfu = None

        if self.flops_counter and (train_step + 1) % self.calc_mfu_freq == 0:
            uses_global_seqlen_sums = (seqlen_sum is not None and seqlen_sq_sum is not None)
            images_seqlens = []
            audio_seqlens = []
            for rbs in expanded_rbs:
                if self.image_grid_thw_name in rbs and rbs[self.image_grid_thw_name] is not None:
                    image_grid_thw = rbs[self.image_grid_thw_name]
                    if image_grid_thw.ndim >= 2 and image_grid_thw.shape[0] > 0:
                        image_seqlen = torch.repeat_interleave(
                            image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]
                        )
                        images_seqlens.extend(image_seqlen.tolist())
                # for welm_omni_v4_5
                if "audio_feature_lengths" in rbs and rbs["audio_feature_lengths"] is not None:
                    audio_seqlens.extend(rbs["audio_feature_lengths"].tolist())
            if (
                uses_global_seqlen_sums and
                self.config.policy.model_arch == MODEL_ARCH.WELM_OMNI_V4_5
            ):
                # Dyn-CP seqlen sums are global-batch aggregates, while
                # ``expanded_rbs`` only holds this DP rank's share. Gather the
                # audio lengths over DP so the audio-tower FLOPs are global
                # too. In the rank-local fallback below, the DP all-reduce of
                # estimated_flops already scales them up.
                gathered = [None] * mpu.get_data_parallel_world_size()
                dist.all_gather_object(gathered, audio_seqlens, group=mpu.get_data_parallel_group())
                audio_seqlens = [length for rank_lens in gathered for length in rank_lens]
            kwargs = {}
            if len(images_seqlens) > 0:
                kwargs["images_seqlens"] = images_seqlens
            if len(audio_seqlens) > 0:
                kwargs["audio_seqlens"] = audio_seqlens
            delta_time = torch.tensor(delta_time, device=torch.cuda.current_device())
            dist.all_reduce(delta_time, op=dist.ReduceOp.MAX)
            if uses_global_seqlen_sums:
                estimated_flops, promised_flops = self.flops_counter.estimate_flops_from_sums(
                    seqlen_sum, seqlen_sq_sum, delta_time.item(), **kwargs
                )
            else:
                batch_seqlens = [seq_length for _ in expanded_rbs]
                estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                    batch_seqlens, delta_time.item(), **kwargs
                )
            estimated_flops = torch.tensor(estimated_flops, device=torch.cuda.current_device())
            if not uses_global_seqlen_sums:
                # The fallback estimates rank-local batches, so sum them over
                # DP. Dynamic-CP sums are already global-batch aggregates and
                # must not be multiplied by the DP size again.
                dist.all_reduce(
                    estimated_flops, op=dist.ReduceOp.SUM, group=mpu.get_data_parallel_group()
                )
            mfu = estimated_flops.item() / promised_flops / torch.distributed.get_world_size() * 100
            self.mfu_sum += mfu
            self.mfu_count += 1
            avg_mfu = self.mfu_sum / self.mfu_count

        return mfu, avg_mfu
