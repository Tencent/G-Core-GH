# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# guanyouhe@tencent.com, xiaotaoliu@tencent.com, nrwu@tencent.com

import os
import sys
import time
import copy
import uuid
import asyncio
import traceback

import torch

from megatron.core import mpu
from megatron.training.global_vars import get_tokenizer
from megatron.training.utils import print_rank_0
from megatron.training.global_vars import get_args

from gpatch.training.v3.grpo_sampler import run_grpo_sampler_v3, GrpoSamplerV3
from gpatch.core.parallel_state import is_mp_and_cp_head, get_mp_and_cp_size
from gpatch.core.sampler_v3.infer_engine import InferEngine
from gpatch.core.sampler_v3.routed_experts_utils import process_routed_experts
from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.training.v3.default_model_provider import default_sampler_model_provider

from tasks.math_rl_v3.args import get_tasks_args

model_provider = default_sampler_model_provider


def get_sampling_params(engine):
    tokenizer = get_tokenizer()
    args = get_args()
    stop_at_token_id = tokenizer._tokenizer.eos_token_id

    # Create a sampling params object.
    return engine.get_sampling_params(
        n=1,  # by passing vllm async llm issues
        temperature=args.ppo_rollout_temperature,
        top_k=args.ppo_rollout_top_k if args.ppo_rollout_top_k > 0 else -1,
        top_p=args.ppo_rollout_top_p,
        max_tokens=args.ppo_resp_seq_len,
        stop_token_ids=[stop_at_token_id],
        seed=args.seed,
    )


# tokens 包括 prompt 与 output，sequence_lengths 包括 prompt 与 output。
# 如果你的 RM 和 actor 不是一个系列的 model（甚至于一个系列，但 tokenizer 略有不同），
# 很明显你需要修改你的 prompt，以及 prompt 修改后，用另一个 tokenizer tokenize 出来的结果。
# 如果你有多个 RM，修改中间的数字即可。
# 我这里写了个简化的示意（用 cpu 处理的 if/else 特定业务逻辑这里实在是不方便写）。
#
# 千万要注意的是：transformers tokenizer 有很多需要留意的 pitfalls！
# 例如：https://git.xxx.com/wepsdl/docker-images/blob/master/test-cases/case-1-transformers/train.py
# 不要相信你的直觉，一定要把结果 print 出来人工检查，增加测试！
# ```python
# rollout_batch['rm_0_tokens'] = tokens.flip(1)
# rollout_batch['rm_0_prompt_lengths'] = lpad_lens.clone()
# rollout_batch['rm_0_sequence_lengths'] = generated_sequence_lengths.clone()
# rollout_batch['rm_0_output_mask'] = None
# ```


@torch.no_grad()
async def gen_rollouts(engine, batch, sampling_repeat):
    # batch 就是 actor 的 rollout_get_batch 的返回值，prompt_token_ids 是 list of dict，lpad_lens 和
    # gt_label 是 [b,] 的 tensor。
    args = get_args()
    prompt_token_ids, lpad_lens, gt_label = batch["prompt_token_ids"], batch["lpad_lens"], batch[
        "gt_label"]

    # 不同的序列使用不同的seed，这样保证了相同输入的生成结果随机性
    # 另外，如果有拒绝采样（不符合要求需多次采样），的情况，还得再吃修改 seed，可考虑加一个
    # time.time()。
    sampling_params = get_sampling_params(engine)
    gens = []
    for i in range(len(prompt_token_ids)):
        for j in range(sampling_repeat):
            tmp_sampling_params = copy.deepcopy(sampling_params)
            tmp_sampling_params.seed += i * sampling_repeat + j
            gen = engine.async_generate(
                prompt_token_ids[i],
                tmp_sampling_params,
                str(uuid.uuid4().hex),
                return_routed_experts=args.moe_router_replay
            )
            gens.append(gen)

    tokens = []
    sequence_lengths = []
    max_response_len = 0
    prompt_lens = []
    routed_experts_list = []
    gt_label_list = []
    gen_outputs = await engine.wait_and_get_async_generate_output(gens)

    for gi, gen in enumerate(gens):
        i = gi // sampling_repeat
        j = gi % sampling_repeat
        output = gen_outputs[gi]
        assert len(output.outputs) == 1
        output_tokens = list(output.outputs[0].token_ids)
        routed_experts = process_routed_experts(
            output.outputs[0], args.num_layers, args.moe_router_topk
        )
        token = prompt_token_ids[i]['prompt_token_ids'] + output_tokens
        assert len(token) <= args.seq_length
        tokens.append(torch.tensor(token, dtype=torch.long))
        sequence_lengths.append(torch.tensor(len(token), dtype=torch.long))
        max_response_len = max(max_response_len, len(token))
        prompt_lens.append(lpad_lens[i])
        gt_label_list.append(gt_label[i])
        routed_experts_list.append(routed_experts)

    rollout_batch = {
        'tokens': tokens,
        'sequence_lengths': sequence_lengths,
        'prompt_lengths': prompt_lens,
        'gt_label': gt_label_list,
    }
    if args.moe_router_replay:
        rollout_batch["routed_experts"] = routed_experts_list
    return rollout_batch


if __name__ == "__main__":
    init_gpatch_for_mcore()
    grpo_sampler = GrpoSamplerV3()
    run_grpo_sampler_v3(
        grpo_sampler,
        model_provider,
        gen_rollouts,
        extra_args_provider=get_tasks_args,
    )
