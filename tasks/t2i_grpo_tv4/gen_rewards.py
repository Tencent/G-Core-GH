import re
from typing import Dict, List

import torch

QWEN2_5_VL_PROMPT = """
Your role is to evaluate the aesthetic quality score of given images.
1. Bad: Extremely blurry, underexposed with significant noise, indiscernible
subjects, and chaotic composition.
2. Poor: Noticeable blur, poor lighting, washed-out colors, and awkward
composition with cut-off subjects.
3. Fair: In focus with adequate lighting, dull colors, decent composition but
lacks creativity.
4. Good: Sharp, good exposure, vibrant colors, thoughtful composition with
a clear focal point.
5. Excellent: Exceptional clarity, perfect exposure, rich colors, masterful
composition with emotional impact.

Please first provide a detailed analysis of the evaluation process, including the criteria for judging aesthetic quality, within the <Thought> tag. Then, give a final score from 1 to 5 within the <Score> tag.
<Thought>
[Analyze the evaluation process in detail here]
</Thought>
<Score>X</Score>
"""


def get_qwen2_5_vl_prompt(batched_data: Dict = None):
    n = len(batched_data['prompt'])
    msgs = []
    for i in range(n):
        msg_i = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": "You are a helpful assistant.",
                }, ],
            },
            {
                "role":
                    "user",
                "content":
                    [
                        {
                            "type": "image",
                            "image": batched_data['images'][i],
                        },
                        {
                            "type": "text",
                            "text": QWEN2_5_VL_PROMPT,
                        },
                    ],
            },
        ]
        msgs.append(msg_i)
    return msgs


def parse_qwen2_5_vl_rewards(resp_texts: List[str] = None):
    assert resp_texts is not None
    reward_scores = []
    for text in resp_texts:
        match = re.search(r'<Score>(\d+)</Score>', text)
        if match:
            reward_scores.append(float(match.group(1)) / 5)
        else:
            reward_scores.append(0)
    assert len(reward_scores) == len(resp_texts)
    return torch.tensor(reward_scores, dtype=torch.float32)


QWEN3_VL_PROMPT = """分析这张基于描述生成的图片。描述：{caption}

请从以下维度评估图片分数：
1. 语义一致性 - 图片内容与描述的匹配程度
2. 美观度 - 构图、色彩、艺术表达
3. 真实性 - 真实感和细节处理

对每个维度分别打分（1-10分），计算总分。
在<think>中详细推理，在<answer>中输出总分。

请按照以下格式输出：

1. 语义一致性（[这张图的语义一致性得分]/10）：
[你的分析过程]
2. 美观度（[这张图的美观度得分]/10）：
[你的分析过程]
3. 真实性（[这张图的真实性得分]/10）：
[你的分析过程，主要关注是否有ai生成痕迹，是否模糊崩坏，细节是否合理]

总分计算：[三个分数相加]
<answer>[总分]</answer>
"""


def get_qwen3_vl_prompt(batched_data: Dict = None):
    n = len(batched_data['prompt'])
    msgs = []
    for i in range(n):

        msg_i = [
            {
                "role":
                    "user",
                "content":
                    [
                        {
                            "type": "image",
                            "image": batched_data['images'][i],
                        },
                        {
                            "type": "text",
                            "text": QWEN3_VL_PROMPT.format(caption=batched_data['prompt'][i]),
                        },
                    ],
            },
        ]
        msgs.append(msg_i)
    return msgs


def parse_qwen3_vl_rewards(resp_texts: List[str] = None):
    # TODO 这里重复编译正则表达式，改下可以优化 cpu 消耗

    assert resp_texts is not None
    reward_scores = []
    for text in resp_texts:
        # 提取总评分（从<answer>标签内）
        match = re.search(r'<answer>([\d.]+)</answer>', text)
        r = 0.
        if match:
            try:
                r = float(match.group(1))
            except ValueError:
                r = 0
        reward_scores.append(r)
        # print(f'TRACE parse_qwen2_5_vl_rewards {torch.distributed.get_rank()} {text=} {r=}')

    assert len(reward_scores) == len(resp_texts)
    return torch.tensor(reward_scores, dtype=torch.float32)
