# copyright (c) 2025 tencent inc. all rights reserved.
# guanyouhe@tencent.com

import re
import copy
from collections import defaultdict
import numpy as np


def convert_pattern(
    user_input: str,
    image_pattern: str = '<image>',
    video_pattern: str = '<video>',
    audio_pattern: str = '<audio>'
):
    """
        Split user input into format tokenizer accepts.
    """
    pattern = r"({image}|{video}|{audio})".format(
        image=image_pattern, video=video_pattern, audio=audio_pattern
    )
    contents = []
    cur = 0
    mm_idx = defaultdict(int)
    for matched in re.finditer(pattern, user_input):
        start, end = matched.span()
        if start > cur:
            contents.append({"type": "text", "text": user_input[cur:start]})

        if matched.string[start:end][1:-1] == "audio":
            contents.append(
                {
                    "type":
                        matched.string[start:end][1:-1],
                    matched.string[start:end][1:-1]:
                        np.array(mm_idx[matched.string[start:end][1:-1]])
                }
            )
        else:
            contents.append(
                {
                    "type": matched.string[start:end][1:-1],
                    matched.string[start:end][1:-1]: str(mm_idx[matched.string[start:end][1:-1]])
                }
            )

        cur = end
        mm_idx[matched.string[start:end][1:-1]] += 1

    if cur < len(user_input):
        contents.append({"type": "text", "text": user_input[cur:len(user_input)]})

    return contents


def convert_conversations(conversations):
    res = []
    for conversation in conversations:
        new_conversation = copy.deepcopy(conversation)
        new_conversation['content'] = convert_pattern(conversation['content'])
        res.append(new_conversation)

    return res


def remove_bos(text):
    # 这里用了两次process，<bos>会多一个
    bos = '<bos>'
    assert text.startswith(bos)
    return text[len(bos):]


'''
输入的数据格式如下：
sft:
```json
{
    "conversations": [
        {
            "role": "user",
            "content": "挂在交通灯杆上的是什么？<image>"
        },
        {
            "role": "assistant",
            "content": "一个绿色的街牌挂在交通灯杆上。"
        }
    ],
    "images": [
        {
            "image_path": "7_0.png",
        }
    ],
    "meta_info": {
        "ori_idx": 859
    }
}
```
dpo:
```
{
    "conversations": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "<image>What are the key features you observe in the image?"
        }
    ],
    "chosen": {
        "role": "assistant",
        "content": "A young man standing on stage wearing a white shirt and black pants."
    },
    "rejected": {
        "role": "assistant",
        "content": "A young man standing on stage wearing white pants and shoes."
    },
    "images": [
        {
            "image_path": "rlhf-v.parquet/0_0.png"
        }
    ],
    "meta_info": {
        "ori_idx": 859
    }
}
```
grpo:
```
{
  "conversations": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "<image>Put the captcha of the image within \\boxed{}"
    }
  ],
  "label": "116OC",
  "images": [
    {
      "image_path": "train-00000-of-00003.parquet/8.png"
    }
  ],
  "meta_info": {
      "ori_idx": 859
  }
}
```
要求：
1. 可以没有图片
2. images也可以只写image_path绝对路径，使用lmdb时也可以使用相对路径
3. 每一条样本都要是有效样本，图片必须能正常读出来
4. sft的数据中：conversations[-1]默认为label
5. grpo的数据中：label是字符串，它不是必须的字段；另外grpo的conversations不能在assistant，
6. meta_info 是一些额外的信息，可以直接返回给用户，用于做统计之类的工作，"meta_info"也可以使用其它key
如果最后一个为assistant直接删除
'''


class MultiModalDatasetMap:
    def __init__(
        self,
        use_for_hf,
        use_grpo,
        tokenizer,
        max_seq_len,
        processor=None,
        image_token_id=None,
        mask_history=False,
        meta_info_key="meta_info",
        moe_pad_with_random_token=False,
    ):
        self.use_for_hf = use_for_hf
        self.use_grpo = use_grpo
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_processor = processor.image_processor if processor is not None else None
        self.max_seq_len = max_seq_len
        self.image_token_id = image_token_id
        self.mask_history = mask_history
        self.meta_info_key = meta_info_key
        self.moe_pad_with_random_token = moe_pad_with_random_token
