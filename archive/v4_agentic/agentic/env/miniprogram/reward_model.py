import json
import re

from .utils_qwen import qwen72b_inference


def extract_reward(text):
    pattern = r'<reward>(.*?)</reward>'
    matches = re.findall(pattern, text)
    return matches


def extract_think(text):
    pattern = r'<reason>(.*?)</reason>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def format_reward_score(content):
    think_tag_pattern = r'<think>(.*?)</think>'
    sub_goal_tag_pattern = r'<sub_goal>(.*?)</sub_goal>'

    content_think_match = re.search(think_tag_pattern, content, re.DOTALL)
    if content_think_match:
        content_think = content_think_match.group(1).strip()
    else:
        return 0.0

    content_sub_goal_match = re.search(sub_goal_tag_pattern, content,
                                       re.DOTALL)
    if content_sub_goal_match:
        content_sub_goal = content_sub_goal_match.group(1).strip()
    else:
        return 0.0

    if len(content_think) > 0 and len(content_sub_goal) > 0:
        return 1.0
    else:
        return 0.0


def format_reward_score_v1(think, sub_goal):
    if len(think) > 0 and len(sub_goal) > 0:
        return 1.0
    else:
        return 0.0


def format_reward_score_v2(content):
    # pattern = r"^<think>((?:(?!</think>).)*)</think><sub_goal>((?:(?!</sub_goal>).)*)</sub_goal>$"
    pattern = r"^<think>((?:(?!</think>).)*)</think><sub_goal>((?:(?!</sub_goal>).)*)</sub_goal><answer>((?:(?!</answer>).)*)</answer>$"
    match = re.match(pattern, content, re.DOTALL)
    if match:
        return 1.0
    else:
        print(f"content format err: {content}")
        return 0.0


def get_reward_score_from_GLM_Qwen72B(inputs, model_name):

    input_img_paths = []
    inst = ""
    subgoals = ""
    thinks = ""
    label = 0
    cur_data_list_len = len(inputs)
    format_score = 0.0
    final_answer = ""
    for index, res in enumerate(inputs):
        inst = res.instruction
        input_img_paths.append(res.before_standard_output.screenshot_dto.url)

        sub_goal_tmp = ""
        if len(res.intent_recognition_result.sub_goal) == 0:
            sub_goal_tmp = res.intent_recognition_result.summary
        else:
            sub_goal_tmp = res.intent_recognition_result.sub_goal

        thinks = res.intent_recognition_result.think

        if sub_goal_tmp.endswith("。"):
            subgoals = subgoals + str(index) + "." + sub_goal_tmp
        else:
            subgoals = subgoals + str(index) + "." + sub_goal_tmp + "。"
        if (index + 1 == cur_data_list_len) and (
            (not res.intent_recognition_result.answer.startswith("finish")) and
            (not res.intent_recognition_result.answer.startswith("calluser"))):
            input_img_paths.append(
                res.after_standard_output.screenshot_dto.url)
        # full_content = res.intent_recognition_result.full_content
        raw_content = res.intent_recognition_result.raw_content
        # format_score = format_score + format_reward_score(full_content)
        # format_score = format_score + format_reward_score_v1(res.intent_recognition_result.think, res.intent_recognition_result.sub_goal)
        format_score = format_score + format_reward_score_v2(raw_content)
        final_answer = res.intent_recognition_result.answer

    # print('*'*100)
    # print("inst:", inst)
    # print("subgoals:", subgoals)
    # print("thinks:", thinks)
    max_retries = 3
    pred = 0
    preds = ""
    thinks_new = ""
    think_ret = ""
    attempt_num = 1
    for attempt in range(max_retries):
        try:
            ret = qwen72b_inference(input_img_paths, model_name, inst,
                                    subgoals, thinks)
            print("reward output:", attempt_num, ret)
            preds = extract_reward(ret)
            thinks_new = extract_think(ret)
            if len(preds) > 0:
                pred = int(preds[0])
                if len(thinks_new) > 0:
                    think_ret = thinks_new[0]
                break
        except:
            print("just retry", attempt_num)
        attempt_num = attempt_num + 1

    if pred < 3.0:
        pred = 0.0
    elif pred < 4.0:
        pred = 1.0
    else:
        pred = 1.0

    if (not final_answer.startswith("finish")) and (
            not final_answer.startswith("calluser")):
        pred = 0.0

    format_score = format_score * 1.0 / cur_data_list_len

    model_score_old = pred
    format_score_old = format_score

    if format_score < 1.0:
        format_score = 0.0
        pred = 0.0

    all_score = pred + format_score

    debug_dict = {
        "all_score": all_score,
        "model_score": pred,
        "format_score": format_score,
        "think_ret": think_ret,
        "model_score_old": model_score_old,
        "format_score_old": format_score_old,
        "attempt_num": attempt_num
    }
    # debug_str = json.dumps(debug_dict, ensure_ascii=False)

    return all_score, debug_dict
