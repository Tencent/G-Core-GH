# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com, guanyouhe@tencent.com

import re
import json
import string
import importlib.util
from typing import List, Dict, Union, Any

import torch
import numpy as np
from mathruler.grader import extract_boxed_content, grade_answer

from megatron.core import mpu
from megatron.training import get_tokenizer, get_args

from gpatch.core.models.gpt import GptPpoCriticModel
from gpatch.core.utils import list_for_tensor_tolist, load_and_call


# return all_match lower_case_match
def match_result_captcha(output, label):
    pattern = r"\\boxed\{(.+?)\}"
    matches = re.findall(pattern, output)
    if len(matches) != 1:
        return 0, 0, 0

    output_label = matches[0]
    all_match = (output_label == label)
    lower_case_match = (output_label.lower() == label.lower())
    return 1, int(all_match), int(lower_case_match)


def match_results_captcha(outputs, labels):
    fmt_val = []
    match_val = []
    for output, label in zip(outputs, labels):
        fmt_score, all_score, lower_score = match_result_captcha(output, label)
        fmt_val.append(float(fmt_score))
        # 满分为1
        match_val.append(float(all_score + lower_score / 2) / 1.5)

    fmt_val = torch.tensor(fmt_val, dtype=torch.float32, device="cpu").view(-1, 1)
    match_val = torch.tensor(match_val, dtype=torch.float32, device="cpu").view(-1, 1)
    return match_val, fmt_val


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str, use_strict=False):
    """Extract the equation from the solution string."""

    if not use_strict:
        answer_pattern = r'<answer>(.*?)</answer>'
        group_idx = 1
    else:
        # 更严格的匹配方式
        answer_pattern = r'<think>(.*?)</think>\s*<search>(.*?)</search>\s*<information>(.*?)</information>\s*<answer>(.*?)</answer>'
        group_idx = 4

    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0 matches, return None
    if len(matches) < 1:
        return None

    # If there are 1 or more matches, return the last one
    return matches[-1].group(group_idx).strip()


def compute_score_em(solution_str, lables_str, format_score=1.0, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        lables_str: include the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str, False)

    if answer is None:
        return 0, 0
    else:
        label_dict = json.loads(lables_str)
        if "ground_truth" not in label_dict:
            return format_score, 0
        label_answers = label_dict["ground_truth"]["target"]
        if em_check(answer, label_answers):
            return format_score, score
        else:
            return format_score, 0


def _cal_reward(content, sol):
    sol = json.loads(sol)
    content = content.replace("\n", "")
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern_click = r'click\s+(\d+)\s+(\d+)'
    bbox_pattern_input = r'input\s+(\d+)\s+(\d+)\s+([a-zA-Z0-9\u4e00-\u9fa5]+)'
    reward = 0.0
    try:
        content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            if content_answer.startswith(sol[0]):
                reward = reward + 0.2
                if sol[0] == 'click':
                    bbox_match = re.search(bbox_pattern_click, content_answer)
                    bbox_0 = int(bbox_match.group(1))
                    bbox_1 = int(bbox_match.group(2))
                    abs_0 = abs(bbox_0 - sol[1])
                    abs_1 = abs(bbox_1 - sol[2])
                    if abs_0 <= 5:
                        reward = reward + 0.4
                    elif abs_0 <= 100:
                        reward = reward + 0.001 * (100.0 / (abs_0 - 4))
                    else:
                        pass
                    if abs_1 <= 5:
                        reward = reward + 0.4
                    elif abs_1 <= 100:
                        reward = reward + 0.001 * (100.0 / (abs_1 - 4))
                    else:
                        pass
                if sol[0] == 'input':
                    bbox_match = re.search(bbox_pattern_input, content_answer)
                    bbox_0 = int(bbox_match.group(1))
                    bbox_1 = int(bbox_match.group(2))
                    input_content = bbox_match.group(3)
                    abs_0 = abs(bbox_0 - sol[1])
                    abs_1 = abs(bbox_1 - sol[2])
                    if abs_0 <= 5:
                        reward = reward + 0.25
                    elif abs_0 <= 100:
                        reward = reward + 0.001 * (100.0 / (abs_0 - 4))
                    else:
                        pass
                    if abs_1 <= 5:
                        reward = reward + 0.25
                    elif abs_1 <= 100:
                        reward = reward + 0.001 * (100.0 / (abs_1 - 4))
                    else:
                        pass
                    if sol[3].lower() == input_content.lower():
                        reward = reward + 0.3
                if sol[0] == 'finish':
                    reward = reward + 0.8
                if sol[0] == 'stop':
                    pass
    except Exception:
        pass

    pattern = r"<think>.*?</think>\s*<answer>(click\s+\d+\s+\d+|input\s+\d+\s+\d+\s+[a-zA-Z0-9\u4e00-\u9fa5]+|finish)</answer><\|im_end\|>"
    match = re.fullmatch(pattern, content, re.DOTALL)
    if match:
        fmt_reward = 1.0
    else:
        fmt_reward = 0.0

    if torch.distributed.get_rank() == 0:
        print("\n" + "-" * 80)
        print(f" Final Score ".center(80, '-'))
        print(f"  ACC Score: {reward}")
        print(f"  Fmt Score: {fmt_reward}")
        print(f"  Answer: {content}")
        print(f"  Gt: {sol}")
        print("=" * 80 + "\n")
    return fmt_reward, reward


def match_results_nq_hotpotq(outputs, labels):
    fmt_val = []
    score_val = []
    for output, label in zip(outputs, labels):
        fmt_score, score = compute_score_em(output, label)
        fmt_val.append(fmt_score)
        score_val.append(score)

    fmt_val = torch.tensor(fmt_val, dtype=torch.float32, device="cpu").view(-1, 1)
    score_val = torch.tensor(score_val, dtype=torch.float32, device="cpu").view(-1, 1)
    return score_val, fmt_val


def geo3k_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def geo3k_format_reward_v2(predict_str: str) -> float:
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


def match_results_geo3k_v2(outputs: List[str], labels: List[str]):
    fmt_val = []
    score_val = []
    for output, label in zip(outputs, labels):
        fmt_val.append(geo3k_format_reward_v2(output))
        score_val.append(geo3k_acc_reward(output, label))

    fmt_val = torch.tensor(fmt_val, dtype=torch.float32, device="cpu").view(-1, 1)
    score_val = torch.tensor(score_val, dtype=torch.float32, device="cpu").view(-1, 1)
    return score_val, fmt_val


def imagepair_format_reward(predict_str: str) -> float:
    pattern = r"\\boxed\{(.+?)\}"
    matches = re.findall(pattern, predict_str)
    #print(f"matches {matches}", flush=True)
    if len(matches) != 1:
        return 0.0
    else:
        return 1.0


def imagepair_acc_reward(predict_str: str, ground_truth: str) -> float:
    def match_result(output):
        pattern = r"\\boxed\{(.+?)\}"
        matches = re.findall(pattern, output)
        if len(matches) != 1:
            return ""
        else:
            return matches[0]

    res = match_result(predict_str)
    print(f"res {res}; label {ground_truth}", flush=True)
    return 1.0 if res == ground_truth else 0.0


def match_results_imagepair(outputs: List[str], labels: List[str]):
    fmt_val = []
    score_val = []
    for output, label in zip(outputs, labels):
        fmt_val.append(imagepair_format_reward(output))
        score_val.append(imagepair_acc_reward(output, label))

    fmt_val = torch.tensor(fmt_val, dtype=torch.float32, device="cpu").view(-1, 1)
    score_val = torch.tensor(score_val, dtype=torch.float32, device="cpu").view(-1, 1)
    return score_val, fmt_val


def pmc_vqa_format_reward(predict_str: str) -> float:
    pattern = re.compile(r'.*<answer>(.*?)</answer>.*', re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def pmc_vqa_acc_reward(predict_str: str, ground_truth: str) -> float:
    pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    results = re.findall(pattern, predict_str)
    if len(results) == 1:
        ground_truth = json.loads(ground_truth)
        label_answer = ground_truth["answer"]
        answer = results[0]
        if answer == label_answer:
            return 1.0
    return 0.0


def match_results_pmc_vqa(outputs: List[str], labels: List[str]):
    fmt_val = []
    score_val = []
    for output, label in zip(outputs, labels):
        fmt_val.append(pmc_vqa_format_reward(output))
        score_val.append(pmc_vqa_acc_reward(output, label))

    fmt_val = torch.tensor(fmt_val, dtype=torch.float32, device="cpu").view(-1, 1)
    score_val = torch.tensor(score_val, dtype=torch.float32, device="cpu").view(-1, 1)
    return score_val, fmt_val


def cal_rewards(outputs, labels):
    fmt_val = []
    score_val = []
    for output, label in zip(outputs, labels):
        fmt_score, score = _cal_reward(output, label)
        fmt_val.append(fmt_score)
        score_val.append(score)

    fmt_val = torch.tensor(fmt_val, dtype=torch.float32, device="cpu").view(-1, 1)
    score_val = torch.tensor(score_val, dtype=torch.float32, device="cpu").view(-1, 1)
    return score_val, fmt_val


def rule_based_rm(
    seq_len_cpu: List[int],
    prompt_len_cpu: List[int],
    tokens_cpu: List[List[int]],
    labels: List[str],
    rule_type: str,
    rule_file: str,
    fmt_factor: float,
    skip_special_tokens: bool,
    tokenizer,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    '''
    多模态计算 rule base 分数的函数

    Parameters
    ----------
    seq_len_cpu: List[int]
        序列总长度
    prompt_len_cpu: List[int]
        序列中的 prompt 长度
    tokens_cpu: List[List[int]]
        token_ids
    labels: List[str]
        用户数据集中的 label
    rule_type: str
        用户使用的计算规则
    rule_file: str
        用户自定义计算规则的脚本文件
    fmt_factor: float
        fmt 的占 reward 分数的权重，(1 - fmt_factor) 为 acc 的权重

    Returns
    -------
    : tuple[torch.Tensor, dict[str, torch.Tensor]]
        rewards tensor and reward_extra_info dict

    Examples
    --------
    A example of rule_type == 'import_file', the function_name must be `compute_score`, here is the rule_file:
    
    .. highlight:: python
    .. code-block:: python

    def compute_score(outputs, labels, **kwargs):
        fmts = []
        accs = []
        rewards = []

        fmts = torch.tensor(fmts, dtype=torch.float32).view(-1, 1)
        accs = torch.tensor(accs, dtype=torch.float32).view(-1, 1)

        rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)

        return rewards, {
            'acc_rewards': fmts,
            'fmt_rewards': accs,
        }
    '''

    assert mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0

    assert len(tokens_cpu) == len(seq_len_cpu)
    assert len(tokens_cpu) == len(prompt_len_cpu)
    resp_token_ids = []
    prompt_token_ids = []
    for i in range(len(tokens_cpu)):
        resp_token_ids.append(tokens_cpu[i][prompt_len_cpu[i]:seq_len_cpu[i]])
        prompt_token_ids.append(tokens_cpu[i][:prompt_len_cpu[i]])
    resp_strs = tokenizer._tokenizer.batch_decode(
        resp_token_ids, skip_special_tokens=skip_special_tokens
    )
    prompt_strs = tokenizer._tokenizer.batch_decode(prompt_token_ids, skip_special_tokens=False)

    assert len(resp_strs) == len(labels)
    if rule_type in ['captcha']:
        acc_reward_tensor, fmt_reward_tensor = match_results_captcha(resp_strs, labels)
    elif rule_type in ['nq_hotpotq']:
        acc_reward_tensor, fmt_reward_tensor = match_results_nq_hotpotq(resp_strs, labels)
    elif rule_type in ['geometry3k']:
        acc_reward_tensor, fmt_reward_tensor = match_results_geo3k(resp_strs, labels)
    elif rule_type in ['geometry3k-v2']:
        acc_reward_tensor, fmt_reward_tensor = match_results_geo3k_v2(resp_strs, labels)
    elif rule_type in ['pmc_vqa']:
        acc_reward_tensor, fmt_reward_tensor = match_results_pmc_vqa(resp_strs, labels)
    elif rule_type in ['agent']:
        acc_reward_tensor, fmt_reward_tensor = cal_rewards(resp_strs, labels)
    elif rule_type in ['image_pair']:
        acc_reward_tensor, fmt_reward_tensor = match_results_imagepair(resp_strs, labels)
    elif rule_type == 'import_file':
        return load_and_call(
            rule_file,
            "compute_score",
            resp_strs,
            labels,
            seq_len_cpu=seq_len_cpu,
            prompt_len_cpu=prompt_len_cpu,
            tokens_cpu=tokens_cpu,
            fmt_factor=fmt_factor,
            prompt_strs=prompt_strs,
        )
    else:
        print(f"not support to this type: {rule_type}")
        raise NotImplemented

    rewards = fmt_factor * fmt_reward_tensor + (1 - fmt_factor) * acc_reward_tensor

    return rewards, {
        'acc_rewards': acc_reward_tensor,
        'fmt_rewards': fmt_reward_tensor,
    }


class MultiModalRuleOnlyCriticModel(GptPpoCriticModel):
    def infer_rule_based_rm(
        self,
        rewards,
        per_token_rewards=None,
        sequence_lengths: torch.Tensor = None,
        prompt_lengths: torch.Tensor = None,
        batches: List[Dict[str, Union[int, List[Any]]]] = None,
    ):
        # TODO 支持 cp 的话，这里要改
        is_mp_head = mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0
        if not is_mp_head:
            assert rewards is None
            assert per_token_rewards is None
            acc_reward_tensor = None
            fmt_reward_tensor = None
            reward_extra_info = {
                'acc_rewards': acc_reward_tensor,
                'fmt_rewards': fmt_reward_tensor,
            }
        else:
            args = get_args()
            inputs_list = []
            labels = []
            for batch in batches:
                inputs_list.extend(batch["tokens"])
                labels.extend(batch["labels"])

            tokens_cpu = list_for_tensor_tolist(inputs_list, False)
            seq_len_cpu = sequence_lengths.tolist()
            prompt_len_cpu = prompt_lengths.tolist()
            rewards, reward_extra_info = rule_based_rm(
                seq_len_cpu,
                prompt_len_cpu,
                tokens_cpu,
                labels,
                args.ppo_mm_rule_type,
                args.ppo_custom_rule_file,
                args.ppo_fmt_factor,
                args.ppo_skip_special_tokens,
                get_tokenizer(),
            )

        return rewards, per_token_rewards, reward_extra_info
