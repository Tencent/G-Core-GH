from typing import List, Dict, Any

import torch


def test(args, rollout_batches: List[Dict[str, List[Any]]]):
    for rb in rollout_batches:
        # assert len(rb) == 8
        for k in rb.keys():
            rb[k] = rb[k][:args.ppo_sampling_keep]


# NOTE: 这里没有经过严格的测试
def best_and_worst(args, rollout_batches: List[Dict[str, List[Any]]]):
    for rb in rollout_batches:
        # assert len(rb) == 8
        rewards = torch.cat(rb["rewards"]).view(-1)
        max_ind = torch.argmax(rewards).item()
        min_ind = torch.argmin(rewards).item()
        for k in rb.keys():
            rb[k] = [rb[k][max_ind], rb[k][min_ind]]


def filter_samplings(args, rollout_batches):
    if args.ppo_sampling_keeping_strategy == 'test':
        test(args, rollout_batches)
    elif args.ppo_sampling_keeping_strategy == 'best-and-worst':
        best_and_worst(args, rollout_batches)
    elif args.ppo_sampling_keeping_strategy == 'all':
        return rollout_batches
    else:
        raise NotImplementedError(f'wtf')
