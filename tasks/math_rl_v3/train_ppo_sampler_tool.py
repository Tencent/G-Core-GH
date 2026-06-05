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
import json
from functools import partial
from typing_extensions import Optional, List, Dict, Tuple

import torch

from megatron.core import mpu
from megatron.training.global_vars import get_tokenizer
from megatron.training.utils import print_rank_0
from megatron.training.global_vars import get_args

from gpatch.training.v3.grpo_sampler import run_grpo_sampler_v3, GrpoSamplerV3
from gpatch.core.parallel_state import is_mp_and_cp_head, get_mp_and_cp_size
from gpatch.core.sampler_v3.infer_engine import InferEngine
from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.training.v3.default_model_provider import default_sampler_model_provider
from gpatch.training.utils import print_with_rank_and_datetime

from tasks.math_rl_v3.args import get_tasks_args
from megatron_datasets.tasks.math_rl_v3.ppo_actor_dataset import tokenize_text

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


def verify_answer(answer, gt):
    return answer == gt


def get_tools():
    tools = [
        {
            "type": "function",
            "function":
                {
                    "name": "verify_answer",
                    "description": "Verify if the final answer is correct",
                    "parameters":
                        {
                            "type": "object",
                            "properties":
                                {
                                    "answer":
                                        {
                                            "type": "int",
                                            "description": "The final answer to the question",
                                        },
                                },
                            "required": ["answer"],
                        },
                },
        },
    ]

    name_to_tool = {
        'verify_answer': verify_answer,
    }

    return tools, name_to_tool


def get_tool_parser(tools):
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.managers.io_struct import Tool, Function

    def convert_dict_to_tool(tool_dict: dict) -> Tool:
        function_dict = tool_dict.get("function", {})
        return Tool(
            type=tool_dict.get("type", "function"),
            function=Function(
                name=function_dict.get("name"),
                description=function_dict.get("description"),
                parameters=function_dict.get("parameters"),
            ),
        )

    tools = [convert_dict_to_tool(raw_tool) for raw_tool in tools]

    args = get_args()
    if args.model_arch == "qwen2.5-3b-instruct":
        tool_call_parser = "qwen25"
    else:
        raise ValueError(
            f'Please set tool_call_parser, available choices are {list(FunctionCallParser.ToolCallParserEnum.keys())}'
        )

    parser = FunctionCallParser(tools=tools, tool_call_parser=tool_call_parser)
    return parser


# tool calling，调用了一个简单的工具用来 verify 答案（简单和数据 ground truth 做了比较），真实 case
# 会更加复杂。
def find_invalid_resp(
    pending_idx_list,
    resp_strs_list,
    gt_label_list,
    messages_list,
    tool_parser,
    name_to_tool,
):
    assert (
        len(pending_idx_list) == len(resp_strs_list) and
        len(pending_idx_list) == len(gt_label_list) and len(pending_idx_list) == len(messages_list)
    )

    should_retry_idx_list = []
    for pending_idx, resp_str, gt_label, messages in zip(
        pending_idx_list, resp_strs_list, gt_label_list, messages_list, strict=True
    ):
        should_retry = False

        # try to parse tool calling
        try:
            normal_text, calls = tool_parser.parse_non_stream(resp_str)
            valid_calls = [call for call in calls if call.name in name_to_tool]
            if len(valid_calls) == 0:
                # no tool calling is made
                should_retry = True
                msg = {
                    "role": "tool",
                    "content": "No tool calling is made. Please try again.",
                }
        except Exception as e:
            # exception when parsing tool calling in sglang
            # could be anything due to wrong answer format
            should_retry = True
            msg = {
                "role": "tool",
                "content": f"An error occured when parsing tool calling: {e}. Please try again.",
            }

        if not should_retry:
            # call tools
            try:
                for call in valid_calls:
                    tool_to_call = name_to_tool[call.name]
                    if call.name == 'verify_answer':
                        tool_to_call = partial(tool_to_call, gt=int(gt_label))
                        kwargs = json.loads(call.parameters)
                        kwargs['answer'] = int(kwargs['answer'])
                        is_correct = tool_to_call(**kwargs)
                        if not is_correct:
                            # the answer is incorrect
                            should_retry = True
                            msg = {
                                "role": "tool",
                                "content": "Your answer is incorrect. Please try again.",
                                "name": "verify_answer",
                            }
            except (json.decoder.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                # exception when calling tools
                should_retry = True
                msg = {
                    "role": "tool",
                    "content": f"An error occured when calling tools: {e}. Please try again.",
                }

        if should_retry:
            # retry in the next round
            should_retry_idx_list.append(pending_idx)
            messages.append(msg)

    return should_retry_idx_list


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
    tokenizer = get_tokenizer()

    if args.use_tool_calling:
        assert args.infer_engine_impl == 'sglang', 'use tool calling with sglang'
        assert args.px_apply_chat_template, 'use tool calling with chat template'
        tools, name_to_tool = get_tools()
        tool_parser = get_tool_parser(tools)

    prompt_token_ids, lpad_lens, gt_label, messages = (
        batch["prompt_token_ids"],
        batch["lpad_lens"],
        batch["gt_label"],
        batch["messages"],
    )

    num_results = len(prompt_token_ids) * sampling_repeat

    # 未完成的序列 idx
    # 初始是所有 idx
    # 在后续每一轮 tool calling 中仅保留需要 retry 的 idx
    pending_idx_list = list(range(num_results))
    tokens = [None] * num_results
    sequence_lengths = [None] * num_results
    prompt_lens = [None] * num_results
    gt_label_list = [None] * num_results
    # 每个序列的 resp spans, 记录每一轮 resp 在完整 token 中的 [start, end)，用于生成 mask
    resp_spans_list: List[List[Tuple[int, int]]] = [[] for _ in range(num_results)]
    mask_list = [None] * num_results

    # 多轮对话的 messages
    # 无 px_apply_chat_template 时是 list of None
    # 有 px_apply_chat_template 时是 list of messages
    # 每个 messages: List[Dict[str, str]] 格式类似于
    # [
    #     {"role": "system", "content": "system prompt"},
    #     {"role": "user", "content": "user prompt"},
    #     {"role": "assistant", "content": "assistant response"},
    #     {"role": "tool", "content": "tool result", "name": "tool name"},
    #     {"role": "assistant", "content": "new assistant response"},
    #     ...
    # ]
    messages_list: List[Optional[List[Dict[str, str]]]] = [
        copy.deepcopy(messages[pending_idx // sampling_repeat]) for pending_idx in pending_idx_list
    ]

    max_tool_calling_rounds = max(args.max_tool_calling_rounds, 1) if args.use_tool_calling else 1
    assert max_tool_calling_rounds > 0
    for tool_calling_round in range(max_tool_calling_rounds):
        # 当前 round 的 input 和 resp
        input_tokens_list = []
        resp_tokens_list = []

        # 不同的序列使用不同的seed，这样保证了相同输入的生成结果随机性
        # 另外，如果有拒绝采样（不符合要求需多次采样），的情况，还得再次修改 seed，可考虑加一个
        # time.time()。
        seed_offset = tool_calling_round * num_results
        sampling_params = get_sampling_params(engine)
        gens = []
        for pending_idx in pending_idx_list:
            i = pending_idx // sampling_repeat
            j = pending_idx % sampling_repeat
            tmp_sampling_params = copy.deepcopy(sampling_params)
            tmp_sampling_params.seed += seed_offset + i * sampling_repeat + j
            if args.use_tool_calling:
                prompt = tokenizer._tokenizer.apply_chat_template(
                    messages_list[pending_idx],
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=tools,
                )
                # 这里需要解释下，在 tool calling 的情况，由于多轮调用，这里长度可能已经超出了限制，
                # 由于只是 demo，所以我们简单做了 cut，真实 case 可能不能如此粗暴处理。
                input_ids, unpadded_lens = tokenize_text(
                    tokenizer._tokenizer,
                    args.seq_length - args.ppo_resp_seq_len,
                    prompt,
                )
                inp = {'prompt_token_ids': input_ids}
            else:
                inp = prompt_token_ids[i]
            input_tokens_list.append(inp)
            gen = engine.async_generate(inp, tmp_sampling_params, str(uuid.uuid4().hex))
            gens.append(gen)

        gen_outputs = await engine.wait_and_get_async_generate_output(gens)

        for pending_idx, input_tokens, output in zip(
            pending_idx_list, input_tokens_list, gen_outputs, strict=True
        ):
            i = pending_idx // sampling_repeat
            j = pending_idx % sampling_repeat
            assert len(output.outputs) == 1
            output_tokens = list(output.outputs[0].token_ids)
            token = input_tokens['prompt_token_ids'] + output_tokens
            # resp 在 token 中的 [start, end)
            resp_spans_list[pending_idx].append((len(input_tokens['prompt_token_ids']), len(token)))
            assert len(token) <= args.seq_length
            tokens[pending_idx] = torch.tensor(token, dtype=torch.long)
            sequence_lengths[pending_idx] = torch.tensor(len(token), dtype=torch.long)
            # NOTE: prompt_lens 保留为最初始的 prompt 长度，不在每一轮更新，否则可能影响后续多处计算，需留意观察是否正确
            prompt_lens[pending_idx] = lpad_lens[i]
            gt_label_list[pending_idx] = gt_label[i]

            # 去掉最后的 <|im_end|>，否则下一轮 apply_chat_template 时会再加一个导致重复
            # NOTE 这里偶发可能导致 truncate 掉 non EOS 的 token，不过这已经是 bad case 了，只能是
            # case by case 的业务处理了，这里只是个 demo。
            resp_tokens_list.append(output_tokens[:-1])

        # an example of tool calling
        # simply retry if the answer is incorrect or no tool calling is made
        if args.use_tool_calling:
            resp_strs_list = tokenizer._tokenizer.batch_decode(
                resp_tokens_list, skip_special_tokens=False
            )

            for pending_idx, resp_str in zip(pending_idx_list, resp_strs_list, strict=True):
                messages_list[pending_idx].append({"role": "assistant", "content": resp_str})

            # find invalid responses
            # update pending_idx_list and messages_list
            # and retry in the next round
            pending_idx_list = find_invalid_resp(
                pending_idx_list=pending_idx_list,
                resp_strs_list=resp_strs_list,
                gt_label_list=[gt_label_list[pending_idx] for pending_idx in pending_idx_list],
                messages_list=[messages_list[pending_idx] for pending_idx in pending_idx_list],
                tool_parser=tool_parser,
                name_to_tool=name_to_tool,
            )
            print_with_rank_and_datetime(
                (
                    f"End of tool calling round {tool_calling_round + 1} / {max_tool_calling_rounds}: "
                    f"should retry in the next round {len(pending_idx_list)} / {len(resp_strs_list)}"
                )
            )
            # break if no invalid responses
            if not pending_idx_list:
                break

    # create loss mask
    # 单次对话不需要，多次 tool calling 需要分段屏蔽 gen 和 tool 结果。rollout batch
    # 里直接不传递即可。
    for i, (resp_spans, token) in enumerate(zip(resp_spans_list, tokens, strict=True)):
        mask = torch.zeros(len(token) - 1, dtype=torch.bool)
        for span in resp_spans:
            # NOTE: 注意这里有个 shift，上面 len(token) - 1 同理
            mask[span[0] - 1:span[1] - 1] = True
        mask_list[i] = mask

    assert (
        None not in tokens and None not in sequence_lengths and None not in prompt_lens and
        None not in gt_label_list and None not in mask_list
    )

    rollout_batch = {
        'tokens': tokens,
        'sequence_lengths': sequence_lengths,
        'prompt_lengths': prompt_lens,
        'gt_label': gt_label_list,
        'mask': mask_list,
    }
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
