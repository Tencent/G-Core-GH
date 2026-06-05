import copy
import json
import math
import re
from collections import defaultdict
from functools import partial
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig, Qwen2VLImageProcessorFast
from transformers.models.auto.processing_auto import AutoProcessor
from typing_extensions import override

from gpatch_v4.configs.config import RlConfig


def convert_pattern(
    user_input: str, image_pattern: str = '<image>', video_pattern: str = '<video>'
):
    """
        Split user input into format tokenizer accepts.
    """
    pattern = r"({image}|{video})".format(image=image_pattern, video=video_pattern)
    contents = []
    cur = 0
    mm_idx = defaultdict(int)
    for matched in re.finditer(pattern, user_input):
        start, end = matched.span()
        if start > cur:
            contents.append({"type": "text", "text": user_input[cur:start]})

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


instruction_following = (
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <reason> </reason> tags. The final answer MUST BE put in \boxed{}."
)


def convert_sample(sample):
    answer = sample['answer']
    problem = sample['problem']
    # write the answer in the json
    label = json.dumps(dict(
        answer=answer,
        problem=problem,
    ))
    conversation = []
    conversation.append(dict(role="system", content="You are a helpful assistant."))
    conversation.append(dict(role="user", content=problem + " " + instruction_following))
    return dict(
        conversations=conversation,
        label=label,
    )


def convert_conversations(conversations):
    res = []
    for conversation in conversations:
        new_conversation = copy.deepcopy(conversation)
        new_conversation['content'] = convert_pattern(conversation['content'])
        res.append(new_conversation)

    return res


def smart_resize(height: int, width: int, factor: int, min_pixels: int,
                 max_pixels: int) -> tuple[int, int]:

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# 这个是 slang 的限制，升级版本后可以干掉
def resize_image(
    image: Image.Image,
    size_factor: int = 28,
) -> Image.Image:

    width, height = image.size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=size_factor,
        min_pixels=4 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    )
    image = image.resize((resized_width, resized_height))

    return image


g_uniq_id = 0


def gen_unique_id(dp_rank):
    global g_uniq_id
    worker_id = get_worker_info().id
    g_uniq_id += 1
    return f"dp_rank_{dp_rank}_worke_id_{worker_id}_{g_uniq_id}"


def collate_func(
    config, processor, tokenizer, hf_config, dp_rank, mrope_index, instances: Sequence[dict]
) -> dict[str, Any]:
    assert len(instances) == 1, f"{config.training.rollout_mbs=} must be 1"

    instance = instances[0]
    instance["images"] = [resize_image(img) for img in instance["images"]]
    messages = convert_sample(instance)
    processor.image_processor.tmp_images = instance["images"]
    input_dict = processor.apply_chat_template(
        convert_conversations(messages['conversations']),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=config.training.enable_thinking,
    )

    seqlen = config.training.seq_length
    prompt_len = input_dict['input_ids'].size(-1)
    assert prompt_len <= (seqlen - config.sampler.infer_engine_configs[0].generate_max_tokens)
    input_ids = torch.nn.functional.pad(
        input_dict['input_ids'], (0, seqlen - prompt_len), value=tokenizer.pad_token_id
    )

    position_ids, _ = mrope_index.get_rope_index(
        input_ids,
        image_grid_thw=input_dict["image_grid_thw"],
        video_grid_thw=None,
        attention_mask=None,  # 这里写 None 就好，因为 grpo 要用到后面的编码
    )

    res = dict(
        # type is list
        unique_id=[gen_unique_id(dp_rank)],
        json_data_list=[messages],
        tokens=[input_ids.squeeze(0)],
        prompt_len=[torch.tensor(prompt_len, dtype=torch.int64)],
        imgs_np_array_list=[[np.array(img) for img in instance["images"]]],
        # save at mm_data_cache, type is tensor
        position_ids=position_ids,
        vision_grid_thw=input_dict["image_grid_thw"],
        image_input_mask=input_ids == hf_config.image_token_id,
        vision_data=input_dict['pixel_values'],
        cache_keys=[
            "position_ids",
            "vision_grid_thw",
            "image_input_mask",
            "vision_data",
        ],
    )

    return res


class Qwen3VlRopeIndexHelper:
    def __init__(self, config):
        from transformers import Qwen3VLModel
        self.config = config
        self.hf_class = Qwen3VLModel

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):

        return self.hf_class.get_rope_index(
            self, input_ids, image_grid_thw, video_grid_thw, attention_mask
        )


class UserQwen2VLImageProcessorFast(Qwen2VLImageProcessorFast):
    tmp_images = None

    @override
    def fetch_images(self, image_url_or_urls: Union[str, list[str], list[list[str]]]):
        return self.tmp_images


def get_dataset_and_dataloader(config: RlConfig = None, tokenizer=None, dp_rank=0, dp_size=1):
    image_processor = UserQwen2VLImageProcessorFast.from_pretrained(config.policy.hf_tokenizer_path)
    processor = AutoProcessor.from_pretrained(
        config.policy.hf_tokenizer_path, image_processor=image_processor
    )
    hf_config = AutoConfig.from_pretrained(config.policy.hf_model_path)
    mrope_index = Qwen3VlRopeIndexHelper(hf_config)

    dataset = load_dataset(config.data.data_pathes[0])
    train_dataset = dataset['train'].shuffle(seed=42)
    eval_dataset = dataset['test'].shuffle(seed=42)

    train_sampler = DistributedSampler(
        train_dataset,
        rank=dp_rank,
        num_replicas=dp_size,
        shuffle=True,
        seed=config.data.sampler_seed
    )

    eval_sampler = DistributedSampler(
        eval_dataset,
        rank=dp_rank,
        num_replicas=dp_size,
        shuffle=True,
        seed=config.data.sampler_seed
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=partial(
            collate_func, config, processor, tokenizer, hf_config, dp_rank, mrope_index
        ),
        pin_memory=config.data.dataloader_pin_memory,
        batch_size=config.training.rollout_mbs,
        num_workers=config.data.dataloader_num_workers,
        prefetch_factor=config.data.dataloader_prefetch_factor,
        drop_last=True,
    )

    eval_rollout_mbs = config.training.eval_rollout_mbs if config.training.eval_rollout_mbs else config.training.rollout_mbs
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        collate_fn=partial(
            collate_func, config, processor, tokenizer, hf_config, dp_rank, mrope_index
        ),
        pin_memory=config.data.dataloader_pin_memory,
        batch_size=eval_rollout_mbs,
        num_workers=config.data.dataloader_num_workers,
        prefetch_factor=config.data.dataloader_prefetch_factor,
        drop_last=True,
    )
    return {
        'train_dataset': train_dataset,
        'train_sampler': train_sampler,
        'train_dataloader': train_dataloader,
        'eval_dataset': eval_dataset,
        'eval_sampler': eval_sampler,
        'eval_dataloader': eval_dataloader,
    }


def verify_dataloader_func(train_dataset, train_sampler, train_dataloader):
    dl_iter = iter(train_dataloader)
    data = next(dl_iter)
    print(f"{data.keys()=}")
    print(f"{data=}")
