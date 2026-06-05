import re
import os
import traceback
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Union

import torch
from transformers import AutoTokenizer

from gpatch_v4.utils import list_of_tensor_to_list, log
from gpatch_v4.utils.ppo_utils import calculate_repetition_penalty_reward


def is_number(text):
    try:
        Decimal(text)
        return True
    except InvalidOperation:
        return False


def extract_last_boxed_content(text):
    boxed_pattern = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
    matches = boxed_pattern.findall(text)

    if matches:
        return matches[-1]
    else:
        return None


def parse_boxed_content(content):
    try:
        content = content.strip()
        t1 = int(content)
        return t1
    except:
        return None


def parse_content(text):
    boxed_content = extract_last_boxed_content(text)
    boxed_value = parse_boxed_content(boxed_content)
    return boxed_value, boxed_content


def cal_accuracy_reward(resp_strs, gt_answer):
    acc_rewards = []
    boxed_content_tmp = []
    boxed_value_tmp = []
    for index, resp in enumerate(resp_strs):
        boxed_value, boxed_content = parse_content(resp)
        try:
            reward = float(boxed_value == gt_answer[index])
        except OverflowError as e:
            print(f"catch OverflowError {e} {index=} {boxed_value=}")
            reward = 0.0

        acc_rewards.append(reward)
        boxed_content_tmp.append(boxed_content)
        boxed_value_tmp.append(boxed_value)

    return acc_rewards, boxed_content_tmp, boxed_value_tmp


def cal_format_reward(resp_strs, **kwargs):
    """Reward function that checks if the completion has the format:
    'xxx final answer to this question is \boxed{...}' and extracts the content inside \boxed{}.
    """
    # pattern = r"final answer to this question is \\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    pattern = r'\\boxed\{[+-]?(\d+(\.\d*)?|\.\d+)\}'

    completion_contents = [resp for resp in resp_strs]
    matches = [
        re.search(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents
    ]

    fmt_reward = [1.0 if match else 0.0 for match in matches]

    return fmt_reward


def math_rl_rule_reward(
    batched_data: List[Dict[str, Union[int, List[Any]]]],
    tokenizer: AutoTokenizer = None,
    actor_tokenizer: AutoTokenizer = None
):
    """
    Custom rule function for reward calculation

    Args:
        batched_data: List of batch data
    
    Returns:
        rule_reward: Tensor
        per_token_reward: Tensor
        metrics: Dict[str, Tensor]
    """
    assert actor_tokenizer is not None

    inputs_list: List[torch.Tensor] = []
    sequence_lengths_list = []
    prompt_lengths_list = []
    # gt_label's type may be change to str
    gt_label_list: List[torch.Tensor] = []

    for batch in batched_data:
        inputs_list.extend(batch["tokens"])
        sequence_lengths_list.extend(batch["sequence_lengths"])
        prompt_lengths_list.extend(batch["prompt_lengths"])
        # batch["gt_label"] shape is [1]
        gt_label_list.extend(batch["gt_label"])

    tokens_cpu: List[List[int]] = list_of_tensor_to_list(inputs_list, False)
    seq_len_cpu: List[int] = torch.stack(sequence_lengths_list).view(-1).tolist()
    # type may be change to List[str]
    gt_label: List[int] = list_of_tensor_to_list(gt_label_list, True)
    assert len(tokens_cpu) == len(seq_len_cpu)
    assert len(tokens_cpu) == len(gt_label)

    try:
        for i in range(len(tokens_cpu)):
            tokens_cpu[i] = tokens_cpu[i][:seq_len_cpu[i]]

        resp_strs = actor_tokenizer.batch_decode(tokens_cpu, skip_special_tokens=False)

        acc_reward, boxed_content_tmp, boxed_value_tmp = cal_accuracy_reward(resp_strs, gt_label)
        fmt_reward = cal_format_reward(resp_strs)
        acc_reward_tensor = torch.tensor(acc_reward, dtype=torch.float32).view(-1, 1)
        fmt_reward_tensor = torch.tensor(fmt_reward, dtype=torch.float32).view(-1, 1)

        rule_reward = acc_reward_tensor + fmt_reward_tensor
        metrics = {
            "acc_reward": acc_reward_tensor,
            "fmt_reward": fmt_reward_tensor,
        }

    except Exception as e:
        os.makedirs("test_llm", exist_ok=True)
        torch.save(tokens_cpu, f"test_llm/debug_save_{torch.distributed.get_rank()}.pt")
        log(f"rule_reward_func error: {e} {tokens_cpu}")
        traceback.print_exc()
        tmp_tokenizer = AutoTokenizer.from_pretrained("/root/hf-hub/Qwen/Qwen2.5-Math-1.5B/")
        resp_strs = tmp_tokenizer.batch_decode(tokens_cpu, skip_special_tokens=False)
        log(f"test temp tokenizer {resp_strs}")
        import sys
        sys.exit()

    return rule_reward, None, metrics


def repetition_penalty_reward(
    batched_data: List[Dict[str, Union[int, List[Any]]]],
    tokenizer: AutoTokenizer = None,
    actor_tokenizer: AutoTokenizer = None
):

    inputs_list: List[torch.Tensor] = []
    sequence_lengths_list = []
    prompt_lengths_list = []

    for batch in batched_data:
        inputs_list.extend(batch["tokens"])
        sequence_lengths_list.extend(batch["sequence_lengths"])
        prompt_lengths_list.extend(batch["prompt_lengths"])

    tokens_cpu: List[List[int]] = list_of_tensor_to_list(inputs_list, False)
    seq_len_cpu: List[int] = torch.stack(sequence_lengths_list).view(-1).tolist()
    pmt_len_cpu: List[int] = torch.stack(prompt_lengths_list).view(-1).tolist()

    assert len(tokens_cpu) == len(seq_len_cpu)
    assert len(tokens_cpu) == len(pmt_len_cpu)
    repetition_penalty_rewards = []

    for i in range(len(tokens_cpu)):
        tokens_cpu[i] = tokens_cpu[i][pmt_len_cpu[i]:seq_len_cpu[i]]
        repetition_penalty_rewards.append(
            calculate_repetition_penalty_reward(tokens_cpu[i], ngram_size=3)
        )
    repetition_penalty_rewards = torch.tensor(repetition_penalty_rewards,
                                              dtype=torch.float32).view(-1, 1)

    metrics = {
        "repetition_penalty_rewards": repetition_penalty_rewards,
    }
    return repetition_penalty_rewards, None, metrics
