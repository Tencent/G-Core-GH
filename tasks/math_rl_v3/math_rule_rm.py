import json
import re
from decimal import Decimal, InvalidOperation

import torch


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


def validate_samples_useful(rewards, repeat_times):
    # TODO yeazhao: verl dapo 里存在判断 repeat_times==1 （对该prompt进行1次采样）时总是 useful
    # 推测verl中不同prompt可能repeat_time可能不同，有些可能等于1？gcore是否需要添加该逻辑？
    grouped_rewards = rewards.view(-1, repeat_times)
    variances = grouped_rewards.var(dim=1, unbiased=False)
    means = grouped_rewards.mean(dim=1)

    # reward 在方差不为0才是有效的
    #sample_useful_mask = variances != 0

    # 为了测试, 暂时加上 means > 1
    sample_useful_mask = (variances != 0) | (means >= 1)
    sample_useful_mask = torch.repeat_interleave(sample_useful_mask, repeat_times, dim=0)
    return sample_useful_mask


# gen rm
def extract_by_split(tag_name, text, eos_token):
    start_tag = f"<|im_start|>{tag_name}\n"
    end_tag = eos_token

    try:
        content = text.split(start_tag)[1].split(end_tag)[0]
        return content.strip()
    except IndexError:
        return None


# gen rm
def get_question_and_answer(text, actor_eos_token):
    user_content = extract_by_split('user', text, '<|im_end|>')
    assistant_content = extract_by_split('assistant', text, actor_eos_token)
    assert user_content is not None
    assert assistant_content is not None
    return (user_content, assistant_content)


# gen rm
def get_rm_verification(text):
    # 粗暴的方式，佛了。因为 demo 使用的 model 指令跟随能力的问题，无法完全跟随指令要求,
    # 所以拿到的结果不是很准，先用这个简单粗暴的方式，具体业务需要自己训练好 gen-rm model
    answer_lst = text.split("Is the answer correct (Yes/No)")
    if len(answer_lst) < 2:
        return 0
    answer = answer_lst[1].lower()
    yes_pos = answer.find("yes")
    no_pos = answer.find("no")

    # 本来应该使用正则表达式匹配，受限于 model 指令跟随能力。有时候可能抽取不出来 yes / no
    # pattern = r"Is the answer correct \(Yes/No\)\?\s*(Yes|No)"
    # match = re.search(pattern, text, re.IGNORECASE)

    # if match:
    #     answer = match.group(1).lower()
    #     return 1 if answer == "yes" else -1
    # else:
    #     return 0
    if yes_pos == -1 and no_pos == -1:
        return 0
    elif yes_pos == -1:
        return -1
    elif no_pos == -1:
        return 1
    elif yes_pos < no_pos:
        return 1
    else:
        return -1
