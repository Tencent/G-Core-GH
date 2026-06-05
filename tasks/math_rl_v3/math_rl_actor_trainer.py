# copyright (c) 2025 tencent inc. all rights reserved.
# nrwu@tencent.com, xiaotaoliu@tencent.com, guanyouhe@tencent.com
import torch

from collections import defaultdict
from typing_extensions import override
from types import SimpleNamespace

from megatron.training.global_vars import get_args

from gpatch.training.utils import print_with_rank_and_datetime
from gpatch.training.v3.ppo_actor import PPOActorTrainerV3


class MathRLActorTrainer(PPOActorTrainerV3):
    @override
    def is_rollout_batch_accepted(self, rb):
        """
        check if all batches are accepted
        
        rb: a list of rollout batches

        returns: no return
        """
        # only for dynamic sampling test, seed varies from rank and time
        # import random
        # import time
        # random.seed(int(time.time() * 1e6) % 10000 + torch.distributed.get_rank())
        # if random.random() < 0.3:
        #     return False
        # return True
        if 'sample_useful' in rb:
            return rb['sample_useful'][0].item()
        # a simple dynamic sample rule
        if 'rewards' in rb:
            eps = 1e-4
            if torch.stack(rb['rewards']).std() < eps:
                return False
            return True

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
            #TODO: gt_label 通过 extra_attr_info 传递
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


class MathRLHeteroGenRMActorTrainer(MathRLActorTrainer):
    def __init__(self, extra_metric_info=None):
        super().__init__(extra_metric_info)
        self.rm_prompt_index_mapping = [[0], [0]]