# copyright (c) 2025 tencent inc. all rights reserved.
# nrwu@tencent.com, xiaotaoliu@tencent.com, guanyouhe@tencent.com

from collections import defaultdict
from typing_extensions import override
from types import SimpleNamespace
from typing import List, Any, Dict, Union

import torch

from megatron.core import mpu
from megatron.training.global_vars import get_args, get_tokenizer

from gpatch.training.utils import print_with_rank_and_datetime
from gpatch.training.v3.ppo_actor import PPOActorTrainerV3
from gpatch.core.parallel_state import is_mp_and_cp_head
from gpatch.core.utils import list_for_tensor_tolist, print_with_rank_and_datetime

from tasks.retool.retool_quality_monitor import log_rollout_quality_samples


class MathRLActorTrainer(PPOActorTrainerV3):
    @override
    def compute_global_rollout_metrics(self, rollout_batches: List[Dict[str, List[Any]]]):
        """
        Override to add num_turns metrics (similar to verl's implementation).

        Extracts retool_num_turns from gt_label and computes mean/max/min statistics.
        """
        # Call parent implementation to get base metrics
        metrics = super().compute_global_rollout_metrics(rollout_batches)

        # Collect num_turns from all rollout batches
        num_turns_list = []
        for rb in rollout_batches:
            gt_label_list = rb.get("gt_label", [])
            for gt_label in gt_label_list:
                if isinstance(gt_label, dict) and "retool_num_turns" in gt_label:
                    num_turns = gt_label["retool_num_turns"]
                    if num_turns > 0:  # Only include valid turns
                        num_turns_list.append(num_turns)

        # Compute statistics if we have data
        if num_turns_list:
            import numpy as np
            num_turns_array = np.array(num_turns_list, dtype=np.float32)

            # Gather statistics across all DP ranks
            local_sum = float(num_turns_array.sum())
            local_count = len(num_turns_list)
            local_max = float(num_turns_array.max())
            local_min = float(num_turns_array.min())

            # All-reduce across data parallel group
            tensor_to_accumulate = torch.tensor(
                [local_sum, local_count],
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
            torch.distributed.all_reduce(tensor_to_accumulate, group=mpu.get_data_parallel_group())
            global_sum, global_count = tensor_to_accumulate.tolist()

            # All-reduce max/min
            tensor_to_max = torch.tensor(
                [local_max, -local_min],
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
            torch.distributed.all_reduce(
                tensor_to_max,
                group=mpu.get_data_parallel_group(),
                op=torch.distributed.ReduceOp.MAX,
            )
            global_max, global_min = tensor_to_max.tolist()
            global_min = -global_min

            # Add metrics (matching verl's naming convention)
            if global_count > 0:
                metrics["num_turns/mean"] = global_sum / global_count
                metrics["num_turns/max"] = global_max
                metrics["num_turns/min"] = global_min

        return metrics

    @override
    def is_rollout_batch_accepted(self, rb):
        """
        check if all batches are accepted
        
        rb: a list of rollout batches

        returns: no return
        """
        return rb['sample_useful'][0].item()

    @override
    def update_replay_samples_dict(self, rb, sample_idx):
        """
        update replay samples dict
        
        rb: a list of rollout batches, assert 
        sample_idx: index of the sample

        returns: no return
        """
        args = get_args()
        max_replay_times = args.ppo_dynamic_sampling_max_replay

        if sample_idx not in self.replay_samples_dict:
            prompt_lengths = rb['prompt_lengths']
            assert prompt_lengths.ndim == 1
            lpad_lens_list = prompt_lengths.tolist()
            response_tokens = rb["response_tokens"]
            prompt_tokens = response_tokens[0][:lpad_lens_list[0]]
            # TODO: gt_label 通过 extra_attr_info 传递
            gt_label = rb['gt_label']
            train_data_consuming_progress = rb.get('train_data_consuming_progress', None)
            prompt_data = {
                "prompt_token_ids": [dict(prompt_token_ids=prompt_tokens.tolist())],
                "lpad_lens": prompt_lengths[0].view(1),
                "gt_label": gt_label[0].view(1),
                "train_data_consuming_progress": train_data_consuming_progress,
            }
            self.replay_samples_dict.update(
                {sample_idx: SimpleNamespace(prompt_data=prompt_data, replay_times=1)}
            )
            # 首次加入队列
            self.replay_queue.append(sample_idx)
        else:
            if self.replay_samples_dict[sample_idx].replay_times > max_replay_times:
                # 超过最大重试限制，弹出数据
                removed_value = self.replay_samples_dict.pop(sample_idx)
                print_with_rank_and_datetime(
                    f"failure. give up replay sample {sample_idx=} replay_times {removed_value.replay_times}"
                )
            else:
                self.replay_samples_dict[sample_idx].replay_times += 1
                # 重新加入队列
                self.replay_queue.append(sample_idx)

    @override
    def is_time_to_replay_samples(self, replay_queue, replay_samples_dict):
        """
        check if need to replay samples
        replay_queue is a list that contains sample_idx
        replay_samples_dict is a dict that {sample_idx: sample_data}

        returns: a list of bool
        """
        # # for test：模拟一种情况不完全消费完的情况
        # if self.test_flag:
        #     self.test_flag = False
        #     res = [True] * len(replay_queue)
        #     res[-1] = False
        #     return res

        # 暂时规则，有就直接重放，你需要自定义规则
        return [True] * len(replay_queue)

    @override
    def hook_after_sampling(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
        ppo_step: int,
        is_eval: bool = False,
    ) -> List[Dict[str, List[Any]]]:
        """
        hook after sampling

        Parameters
        ----------
        rollout_batches : List[Dict[str, List[Any]]]
            rollout batches containing sampling results.
        ppo_step : int
            ppo step of train or eval.
        is_eval : bool
            whether is eval.

        Returns
        -------
        List[Dict[str, List[Any]]]
            processed rollout batches.
        """
        assert is_mp_and_cp_head()

        args = get_args()
        if args.no_hook_webapi:
            return rollout_batches

    @override
    def hook_before_computing_metrics(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
        ppo_step: int,
        is_eval: bool = False,
    ) -> List[Dict[str, List[Any]]]:
        """
        hook before computing metrics

        Parameters
        ----------
        rollout_batches : List[Dict[str, List[Any]]]
            rollout batches before computing metrics, maybe containing sampling results, logprobs, something add in hook_after_sampling, etc.
        ppo_step : int
            ppo step of train or eval.
        is_eval : bool
            whether is eval.
    
        Returns
        -------
        List[Dict[str, List[Any]]]
            processed rollout batches.
        """

        args = get_args()
        if args.no_hook_webapi:
            # =====================================================================
            # ReTool Quality Monitor (no_hook_webapi mode):
            # 在每个 PPO Step 后记录样本质量日志
            # 此模式下 reward 已在 sampler 端预计算并存储在 gt_label 中
            # =====================================================================
            try:
                log_rollout_quality_samples(
                    rollout_batches=rollout_batches,
                    ppo_step=ppo_step,
                    is_eval=is_eval,
                )
            except Exception as e:
                print_with_rank_and_datetime(
                    f"[ReTool Monitor] Warning: Failed to log quality samples: {e}"
                )
            return rollout_batches
