# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# guanyouhe@tencent.com, xiaotaoliu@tencent.com, nrwu@tencent.com
import copy
import uuid
from typing import Dict, List, Union, Any, Optional

import torch
import numpy as np

from megatron.core import mpu
from megatron.training.global_vars import get_tokenizer
from megatron.training.utils import print_rank_0
from megatron.training.global_vars import get_args

from gpatch.training.v3.grpo_gen_rm import GrpoGenRmTrainerV3, run_grpo_gen_rm_v3
from gpatch.core.parallel_state import is_mp_and_cp_head, get_mp_and_cp_size
from gpatch.core.sampler_v3.infer_engine import InferEngine
from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.training.global_vars import get_actor_tokenizer, get_rm_tokenizer
from gpatch.training.v3.default_model_provider import default_gen_rm_model_provider

from tasks.math_rl_v3.args import get_tasks_args
from tasks.math_rl_v3.math_rule_rm import get_rm_verification, get_question_and_answer
from tasks.math_rl_v3.sp import get_gen_rm_prompt_format

model_provider = default_gen_rm_model_provider


def get_sampling_params(engine):
    tokenizer = get_tokenizer()
    args = get_args()
    stop_at_token_id = tokenizer._tokenizer.eos_token_id

    # Create a sampling params object.
    return engine.get_sampling_params(
        n=1,  # by passing vllm async llm issues
        temperature=args.ppo_gen_rm_temperature,
        top_k=args.ppo_gen_rm_top_k if args.ppo_gen_rm_top_k > 0 else -1,
        top_p=args.ppo_gen_rm_top_p,
        max_tokens=args.ppo_gen_rm_resp_seq_len,
        stop_token_ids=[stop_at_token_id],
        seed=args.seed,
    )


def preprocess_data(batch, actor_tokenizer):
    #NOTE: 输入的 batch 是一个 Dict[str, Union[int, List[Any]]],
    # 其中 tokens / sequence_lengths/ prompt_lengths 字段均是 List of tensor, 如果有需要的话
    # ，可以在此函数里临时转为 tensor
    # 比如
    input_tokens = batch['tokens']
    sequence_lens = batch['sequence_lengths']
    prompt_lens = batch['prompt_lengths']
    # padded_tokens_tensor = torch.nn.utils.rnn.pad_sequence(
    #     input_tokens,
    #     batch_first=True,
    #     padding_value=actor_tokenizer._tokenizer.pad_token_id)
    # sequence_lens_tensor = torch.stack(sequence_lens)
    # prompt_lens_tensor = torch.stack(prompt_lens)
    # padd_seq_len = padded_tokens_tensor.shape[-1]
    # return padded_tokens_tensor, sequence_lens, prompt_lens, padd_seq_len
    return input_tokens, sequence_lens, prompt_lens, None


def postprocess_resp_dict(resp_dict, sequence_lens, padd_seq_len=None):
    #NOTE: resp_dict 是一个 Dict[str, Union[int, List[Any]]],
    # 如果算出来的 reward 是一个 tensor，那么需要转化为 list of tensor，也就是把 batch size 纬度拆到 List 中
    for key in resp_dict.keys():
        if torch.is_tensor(resp_dict[key]):
            assert resp_dict[key].ndim == 2 and resp_dict[key].shape[0] == len(sequence_lens)
            if resp_dict[key].shape[-1] == padd_seq_len:
                resp_dict[key] = [
                    resp_dict[key][i, :sequence_lens[i]] for i in range(len(sequence_lens))
                ]
            else:
                assert resp_dict[key].shape[-1] == 1, f"resp_dict[key].shape {resp_dict[key].shape}"
                resp_dict[key] = list(torch.unbind(resp_dict[key], dim=0))
        elif isinstance(resp_dict[key], dict):
            resp_dict[key] = postprocess_resp_dict(resp_dict[key], sequence_lens, padd_seq_len)
        elif isinstance(resp_dict[key], list):
            assert len(resp_dict[key]) == len(sequence_lens)
            pass
        else:
            raise ValueError(f"Unknown key {key} {type(resp_dict)}")
    return resp_dict


# GEN RM：业务、算法侧的 RM generation 的 parse，这里为了简化就写了一个简单的手写规则作为 demo。
@torch.no_grad()
async def gen_rm_func(engine, batch: Optional[Dict[str, Union[int, List[Any]]]]):
    args = get_args()
    actor_tokenizer = get_actor_tokenizer()
    rm_tokenizer = get_rm_tokenizer()
    gen_rm_prompt = get_gen_rm_prompt_format(args)
    sampling_params = get_sampling_params(engine)

    tokens, sequence_lengths, prompt_lengths, padd_seq_len = preprocess_data(batch, actor_tokenizer)

    resp_toks_as_l = [token[:length] for token, length in zip(tokens, sequence_lengths)]
    resp_strs = actor_tokenizer._tokenizer.batch_decode(resp_toks_as_l, skip_special_tokens=False)
    rm_prompts = []
    for resp in resp_strs:
        question, solution = get_question_and_answer(resp, actor_tokenizer._tokenizer.eos_token)
        content = f"Question: {question}\nSolution: {solution}"
        rm_prompt = gen_rm_prompt.format(content)
        rm_prompts.append(rm_prompt)
    prompt_token_ids = rm_tokenizer._tokenizer(rm_prompts, add_special_tokens=True).input_ids

    gens = []
    for i in range(len(prompt_token_ids)):
        for j in range(args.ppo_gen_rm_repeat):
            tmp_sampling_params = copy.deepcopy(sampling_params)
            tmp_sampling_params.seed += i * args.ppo_gen_rm_repeat + j
            gen = engine.async_generate(
                {'prompt_token_ids': prompt_token_ids[i]}, tmp_sampling_params,
                str(uuid.uuid4().hex)
            )
            gens.append(gen)
    gen_outputs = await engine.wait_and_get_async_generate_output(gens)

    x = []
    gen_rewards = [0. for _ in range(len(prompt_token_ids))]
    for gi, gen in enumerate(gens):
        i = gi // args.ppo_gen_rm_repeat
        j = gi % args.ppo_gen_rm_repeat
        assert len(gen_outputs[gi].outputs) == 1
        output_tokens = gen_outputs[gi].outputs[0].token_ids
        resp_i_j = rm_tokenizer._tokenizer.batch_decode([output_tokens],
                                                        skip_special_tokens=False)[0]
        reward_i_j = get_rm_verification(resp_i_j)
        gen_rewards[i] += torch.tensor(reward_i_j / args.ppo_gen_rm_repeat, dtype=torch.float32)

    resp_dict = {
        "rm_rewards": gen_rewards,
        "rewards": gen_rewards,
        "per_token_rewards": [None] * len(gen_rewards)
    }
    resp_dict = postprocess_resp_dict(resp_dict, batch["sequence_lengths"], padd_seq_len)

    ret_rb = copy.deepcopy(batch)
    for k, v in resp_dict.items():
        ret_rb[k] = v

    return ret_rb


if __name__ == "__main__":
    init_gpatch_for_mcore()
    trainer = GrpoGenRmTrainerV3()
    run_grpo_gen_rm_v3(
        trainer,
        model_provider,
        gen_rm_func,
        extra_args_provider=get_tasks_args,
    )
