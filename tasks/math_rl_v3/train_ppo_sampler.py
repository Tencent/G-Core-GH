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
from typing import Optional

import torch

from gpatch.training.global_vars import get_rm_tokenizer
from gpatch.training.utils import get_hetero_gen_rm_idx_by_rm_idx
from megatron.core import mpu
from megatron.training.global_vars import get_tokenizer
from megatron.training.utils import print_rank_0
from megatron.training.global_vars import get_args

from gpatch.training.v3.grpo_sampler import run_grpo_sampler_v3, GrpoSamplerV3
from gpatch.core.parallel_state import is_mp_and_cp_head, get_mp_and_cp_size
from gpatch.core.sampler_v3.infer_engine import InferEngine
from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.training.v3.default_model_provider import default_sampler_model_provider, extra_sampler_model_provider
from tasks.math_rl_v3.math_rule_rm import get_rm_verification, get_question_and_answer
from tasks.math_rl_v3.sp import get_gen_rm_prompt_format

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
    if args.enable_off_policy_correction:
        assert (
            args.ppo_partial_rollout_global_batch_size == args.ppo_rollout_global_batch_size
        ), "TIS currently do not support partial rollout."

    tokenizer = get_tokenizer()

    stop_at_token_id = tokenizer._tokenizer.eos_token_id

    # 不同的序列使用不同的seed，这样保证了相同输入的生成结果随机性
    # 另外，如果有拒绝采样（不符合要求需多次采样），的情况，还得再次修改 seed，可考虑加一个
    # time.time()。
    prompt_token_ids, lpad_lens, gt_label = batch["prompt_token_ids"], batch["lpad_lens"], batch[
        "gt_label"]

    ret_len = len(prompt_token_ids) * sampling_repeat
    tokens = [None for _ in range(ret_len)]
    sequence_lengths = [None for _ in range(ret_len)]
    max_response_len = 0
    prompt_lens = [None for _ in range(ret_len)]
    gt_label_list = [None for _ in range(ret_len)]
    position_ids_list = [None for _ in range(ret_len)]
    partial_tokens = None
    is_partial_rollout = False
    position_ids = batch.get('position_ids', None)

    # 需要处理 partial rollout 中未处理完成的部份
    if hasattr(batch, "partial_tokens"):
        partial_tokens = batch["partial_tokens"]
        assert len(partial_tokens) == sampling_repeat
        is_partial_rollout = True
        for i in range(len(partial_tokens)):
            mbs_idx = i // sampling_repeat
            # stopped or no enough token budget
            if partial_tokens[i][-1] == stop_at_token_id or \
               sampling_params.max_new_tokens - len(partial_tokens[i]) + lpad_lens[mbs_idx] <= 0:
                tokens[i] = torch.tensor(partial_tokens[i], dtype=torch.long)
                sequence_lengths[i] = torch.tensor(len(partial_tokens[i]), dtype=torch.long)
                max_response_len = max(max_response_len, len(partial_tokens[i]))
                prompt_lens[i] = lpad_lens[mbs_idx]
                gt_label_list[i] = gt_label[mbs_idx]
                continue

    sampling_params = get_sampling_params(engine)
    gens = []
    repeat_idxs = []

    for i in range(len(prompt_token_ids)):
        for j in range(sampling_repeat):
            tmp_sampling_params = copy.deepcopy(sampling_params)
            repeat_idx = i * sampling_repeat + j
            if tokens[repeat_idx] is not None:
                continue

            tmp_sampling_params.seed += repeat_idx

            token_ids_for_gen = prompt_token_ids[i]
            if is_partial_rollout:
                # for partial rollout
                tmp_sampling_params.max_new_tokens = max(0, tmp_sampling_params.max_new_tokens - \
                                        len(token_ids_for_gen) + lpad_lens[i])
                token_ids_for_gen = {"prompt_token_ids": partial_tokens[repeat_idx]}

            gen = engine.async_generate(
                token_ids_for_gen, tmp_sampling_params, str(uuid.uuid4().hex)
            )
            gens.append(gen)
            repeat_idxs.append(repeat_idx)

    rollout_log_probs = [None for _ in range(ret_len)]

    gen_outputs = await engine.wait_and_get_async_generate_output(gens)

    is_aborted = False

    for gi, gen in enumerate(gens):
        gen_repeat_idx = repeat_idxs[gi]
        mbs_idx = gen_repeat_idx // sampling_repeat
        output = gen_outputs[gi]
        assert len(output.outputs) == 1
        output_tokens = list(output.outputs[0].token_ids)
        is_aborted = is_aborted or output.outputs[0].is_aborted
        token = prompt_token_ids[mbs_idx]['prompt_token_ids'] + output_tokens
        assert len(token) <= args.seq_length

        tokens[gen_repeat_idx] = torch.tensor(token, dtype=torch.long)
        sequence_lengths[gen_repeat_idx] = torch.tensor(len(token), dtype=torch.long)
        max_response_len = max(max_response_len, len(token))
        prompt_lens[gen_repeat_idx] = lpad_lens[mbs_idx]
        gt_label_list[gen_repeat_idx] = gt_label[mbs_idx]
        if position_ids is not None:
            position_ids_list[gen_repeat_idx] = position_ids
        # 组装 rollout logps
        gen_lp = torch.tensor(output.outputs[0].output_logprobs, dtype=torch.float32)
        prompt_len = prompt_lens[gen_repeat_idx]
        total_len = len(tokens[gen_repeat_idx])
        full_lp = torch.ones(total_len, dtype=torch.float32)
        gen_len = gen_lp.size(0)
        assert total_len == prompt_len + gen_len
        full_lp[prompt_len - 1:prompt_len + gen_len - 1] = gen_lp

        rollout_log_probs[gen_repeat_idx] = full_lp
        resp_len = len(output.outputs[0].token_ids)
        assert resp_len == gen_lp.size(0), \
            f"resp_len={resp_len} != len(output_logprobs)={gen_lp.size(0)}"

    for i in range(len(tokens)):
        assert tokens[i] is not None

    # is_aborted is used to indicate whether the generation is aborted, and is dropped right after sampler finishes.
    rollout_batch = {
        'tokens': tokens,
        'sequence_lengths': sequence_lengths,
        'prompt_lengths': prompt_lens,
        'gt_label': gt_label_list,
        'rollout_log_probs': rollout_log_probs,
        'is_aborted': is_aborted,
    }
    if position_ids is not None:
        rollout_batch.update({"position_ids": position_ids_list})

    if args.use_gen_rm:
        gen_rm_prompt = get_gen_rm_prompt_format(args)
        resp_toks_as_l = [token[:length] for token, length in zip(tokens, sequence_lengths)]
        resp_strs = tokenizer._tokenizer.batch_decode(resp_toks_as_l, skip_special_tokens=False)
        rm_prompts = []
        for resp in resp_strs:
            question, solution = get_question_and_answer(resp, tokenizer._tokenizer.eos_token)
            content = f"Question: {question}\nSolution: {solution}"
            rm_prompt = gen_rm_prompt.format(content)
            rm_prompts.append(rm_prompt)
        for i in range(args.ppo_num_rm):
            hidx = get_hetero_gen_rm_idx_by_rm_idx(args, i)
            rm_tokenizer = get_rm_tokenizer(hidx)
            rm_tokens_per_rm = []
            for p in rm_prompts:
                prompt_token_ids = rm_tokenizer._tokenizer(p, add_special_tokens=True).input_ids
                rm_tokens_per_rm.append(torch.tensor(prompt_token_ids, dtype=torch.long))
            rollout_batch[f"rm_{i}_tokens"] = rm_tokens_per_rm

    original_batch_with_partial = copy.deepcopy(batch)
    original_batch_with_partial["partial_tokens"] = tokens
    return rollout_batch, original_batch_with_partial


if __name__ == "__main__":
    init_gpatch_for_mcore()
    grpo_sampler = GrpoSamplerV3()
    run_grpo_sampler_v3(
        grpo_sampler,
        model_provider,
        gen_rollouts,
        extra_args_provider=get_tasks_args,
    )
