import re
import json
from typing import Any, Dict, List, Union, Optional

import torch
from transformers import AutoTokenizer
from mathruler.grader import extract_boxed_content, grade_answer

from gpatch_v4.utils import list_of_tensor_to_list, log


def geo3k_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<reason>.*</reason>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def geo3k_acc_reward(predict_str: str, ground_truth: str) -> float:
    ground_truth = json.loads(ground_truth)
    label_answer = ground_truth["answer"]
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, label_answer) else 0.0


def match_results_geo3k(outputs: List[str], labels: List[str]):
    fmt_val = []
    score_val = []
    for output, label in zip(outputs, labels):
        fmt_val.append(geo3k_format_reward(output))
        score_val.append(geo3k_acc_reward(output, label))

    fmt_val = torch.tensor(fmt_val, dtype=torch.float32, device="cpu").view(-1, 1)
    score_val = torch.tensor(score_val, dtype=torch.float32, device="cpu").view(-1, 1)
    return score_val, fmt_val


def geo3k_rl_rule_reward(
    batched_data: List[Dict[str, Union[int, List[Any]]]],
    tokenizer: AutoTokenizer = None,
    actor_tokenizer: AutoTokenizer = None
) -> tuple[torch.Tensor, Optional[torch.Tensor], dict[str, torch.Tensor]]:
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
    labels = []
    for batch in batched_data:
        inputs_list.extend(batch["tokens"])
        sequence_lengths_list.extend(batch["sequence_lengths"])
        prompt_lengths_list.extend(batch["prompt_lengths"])
        labels.extend(batch["labels"])

    tokens_cpu: List[List[int]] = list_of_tensor_to_list(inputs_list, False)
    seq_len_cpu: List[int] = torch.stack(sequence_lengths_list).view(-1).tolist()
    prompt_len_cpu = torch.stack(prompt_lengths_list).view(-1).tolist()

    assert len(tokens_cpu) == len(seq_len_cpu)
    assert len(tokens_cpu) == len(prompt_len_cpu)
    assert len(tokens_cpu) == len(labels)

    resp_token_ids = []
    for i in range(len(tokens_cpu)):
        resp_token_ids.append(tokens_cpu[i][prompt_len_cpu[i]:seq_len_cpu[i]])
    resp_strs = actor_tokenizer.batch_decode(resp_token_ids, skip_special_tokens=True)
    log(f"rule_reward_func debug {resp_strs=}")

    acc_reward_tensor, fmt_reward_tensor = match_results_geo3k(resp_strs, labels)
    rule_reward = acc_reward_tensor + fmt_reward_tensor
    metrics = {
        "acc_reward": acc_reward_tensor,
        "fmt_reward": fmt_reward_tensor,
    }

    return rule_reward, None, metrics
