# copyright (c) 2025 tencent inc. all rights reserved.
# guanyouhe@tencent.com

import copy
import io
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, PngImagePlugin
from torch.utils.data import IterableDataset as TorchIterableDataset

from megatron_datasets.mega_indexed_jsonl_dataset_mm import MegaIndexedJsonlDatasetMM
from megatron_datasets.tools.lmdb_read_cli import fetch_images_from_lmdb

tar_file_cache = {}  # tar_filepath: (tar_context, access_cnt)
tar_file_cache_size = 25
tar_access_cnt = 0


def get_image_from_tar(tar_filepath, offset, size) -> Image.Image:
    global tar_file_cache
    global tar_file_cache_size
    global tar_access_cnt
    tar_access_cnt += 1
    if tar_filepath in tar_file_cache:
        tar_context = tar_file_cache[tar_filepath][0]
        tar_file_cache[tar_filepath][1] = tar_access_cnt
    else:
        with open(tar_filepath, "rb") as tar_fd:
            tar_context = tar_fd.read()
        if len(tar_file_cache) > tar_file_cache_size:
            # get the min access_cnt file
            min_key = ""
            min_cnt = tar_access_cnt * 10
            for k, v in tar_file_cache.items():
                if v[1] < min_cnt:
                    min_key = k
                    min_cnt = v[1]
            tar_file_cache.pop(min_key)
        tar_file_cache[tar_filepath] = [tar_context, tar_access_cnt]

    image = Image.open(io.BytesIO(tar_context[offset:offset + size]))
    return image


def fetch_image(
    ele: dict[str, str],
    tar_dir,
) -> Image.Image:
    image_path = ele["image_path"]
    # 目前只支持本地路径及从tar包中读取
    if "tar_name" in ele:
        image_obj = get_image_from_tar(
            os.path.join(tar_dir, ele["tar_name"]), ele["offset"], ele["size"]
        )
        image = image_obj.convert("RGB")
    else:
        image_obj = Image.open(image_path)
        image = image_obj.convert("RGB")

    return image


def fetch_images(images: list[dict], tar_dir: str, lmdb_port=None) -> list[Image.Image]:
    # NOTE(guanyouhe): 这里图片附加信息过大会导致打开图片失败，修改一下这个值就好
    if PngImagePlugin.MAX_TEXT_CHUNK < 1024 * 1024 * 1024:
        print(
            f"Warning: PngImagePlugin.MAX_TEXT_CHUNK changed from {PngImagePlugin.MAX_TEXT_CHUNK} to 1024 * 1024 * 1024"
        )
        PngImagePlugin.MAX_TEXT_CHUNK = 1024 * 1024 * 1024  # 1GB

    if lmdb_port is not None:
        img_lists = fetch_images_from_lmdb(images, lmdb_port)
    else:
        img_lists = [fetch_image(ele, tar_dir) for ele in images]
    return img_lists


def convert_pattern(
    user_input,
    image_pattern: str = '<image>',
    video_pattern: str = '<video>',
    audio_pattern: str = '<audio>'
):
    """
    Split user input into format tokenizer accepts.

    Notes
    -----
    如果 user_input 已经是结构化 content（list[dict]），直接原样返回，由上游负
    责占位符 / HTML 字面量（例如文本里真实存在的 ``<image>`` / ``<video>`` 标
    签）的区分，避免正则误匹配。
    """
    if not isinstance(user_input, str):
        return user_input

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


def refact_conversations(conversations):
    assert isinstance(
        conversations, list
    ), f"conversations should be a list but get {type(conversations)}"
    for data in conversations:
        if "tool_calls" in data.keys() and data["tool_calls"] is not None:
            tool_calls = data["tool_calls"]
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    assert isinstance(tc, dict)
                    if isinstance(tc["function"]["arguments"], dict):
                        continue
                    tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    return conversations


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
    ]
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
    ]
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
  ]
}
```
要求：
1. 可以没有图片
2. images也可以只写image_path绝对路径，使用lmdb时也可以使用相对路径
3. 每一条样本都要是有效样本，图片必须能正常读出来
4. sft的数据中：conversations[-1]默认为label
5. grpo的数据中：label是字符串，它不是必须的字段；另外grpo的conversations不能在assistant，
如果最后一个为assistant直接删除
'''


class MultiModalDataset(TorchIterableDataset):
    def __init__(
        self,
        tokenizer,
        max_seq_len,
        path_likes,
        domain_probabilities,
        domain_names,
        global_batch_size,
        train_data_consuming_progresses=None,
        rank=0,
        dp_rank=0,
        dp_size=1,
        num_workers=1,
        access_policy_interleave=False,
        shuffle_buffer_size=1000,
        seed=0,
        train=False,
        retention_rates_per_domains=None,
        unsplit_eval_data=False,
        enable_pareto=[],
        pareto_alphas=[],
        pareto_scales=[],
        pareto_score_scales=[],
        top_domains_to_cut=1,
        processor=None,
        tar_dir="/",
        lmdb_port=None,
        image_token_id=None,
        moe_pad_with_random_token=False,
    ):
        self.underlying = MegaIndexedJsonlDatasetMM(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            path_likes=path_likes,
            domain_probabilities=domain_probabilities,
            domain_names=domain_names,
            global_batch_size=global_batch_size,
            train_data_consuming_progresses=train_data_consuming_progresses,
            rank=rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            num_workers=num_workers,
            access_policy_interleave=access_policy_interleave,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            train=train,
            retention_rates_per_domains=retention_rates_per_domains,
            unsplit_eval_data=unsplit_eval_data,
            enable_pareto=enable_pareto,
            pareto_alphas=pareto_alphas,
            pareto_scales=pareto_scales,
            pareto_score_scales=pareto_score_scales,
            top_domains_to_cut=top_domains_to_cut,
        )
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_processor = processor.image_processor if processor is not None else None
        self.max_seq_len = max_seq_len
        self.tar_dir = tar_dir
        self.lmdb_port = lmdb_port
        self.image_token_id = image_token_id
        self.moe_pad_with_random_token = moe_pad_with_random_token

    def gen_label_mask(self, conversations, imgs, tools, label_role=["assistant"], rm_bos=True):
        pre_len = 0
        mask_indexs = []
        for i in range(len(conversations)):
            if conversations[i]['role'] in ['system']:
                continue
            add_generation_prompt = False
            if i + 1 < len(conversations) and conversations[i]['role'] in [
                'user'
            ] and conversations[i + 1]['role'] in ["assistant"]:
                add_generation_prompt = True
            if self.use_grpo:
                assert conversations[-1]['role'] != "assistant"
                add_generation_prompt = True

            text = self.processor.apply_chat_template(
                conversations[:i + 1],
                tools=tools,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            if rm_bos:
                text = remove_bos(text)

            text_dict = self.processor(
                text=[text],
                images=imgs,
                return_tensors="pt",
            )

            cur_len = len(text_dict['input_ids'].squeeze(0).tolist())
            if conversations[i]['role'] not in label_role:
                mask_indexs.append([pre_len, cur_len])
            pre_len = cur_len
        if self.mask_history:
            mask_indexs = [[mask_indexs[0][0], mask_indexs[-1][-1]]]
        return mask_indexs

    def __iter__(self):
        raise NotImplementedError


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
