# copyright (c) 2024 tencent inc. all rights reserved.
# guanyouhe@tencent.com

from copy import deepcopy
import math
from dataclasses import dataclass
from typing import Dict, Sequence, Optional, Tuple
from types import SimpleNamespace

from PIL import Image
import numpy as np
from torch.utils.data.dataloader import default_collate
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers import AutoProcessor
try:
    from transformers import Qwen3VLProcessor
except:
    Qwen3VLProcessor = None

from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer

from gpatch.core.utils import qwen2vl_pad_and_split

from megatron_datasets.utils import print_rank_0, get_iterator, random_pad_list
from megatron_datasets.mm_dataset import (
    MultiModalDataset,
    fetch_images,
    convert_conversations,
)

# copy from: https://github.com/QwenLM/Qwen2-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
# 目前只保存读image的
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def resize_image(
    ele: dict[str, str],
    image: Image.Image,
    default_min_pixels: int = None,
    default_max_pixels: int = None,
    size_factor: int = IMAGE_FACTOR,
) -> Image.Image:
    # resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        default_min_pixels = default_min_pixels or MIN_PIXELS
        default_max_pixels = default_max_pixels or MAX_PIXELS
        width, height = image.size
        min_pixels = ele.get("min_pixels", default_min_pixels)
        max_pixels = ele.get("max_pixels", default_max_pixels)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


class Qwen2VLTokenizer(_HuggingFaceTokenizer):
    def qwen2vl_init(self):
        self.special_tokens_map = {
            k: v
            for k, v in zip(self._tokenizer.all_special_tokens, self._tokenizer.all_special_ids)
        }
        self.image_token = '<|image_pad|>'
        self.video_token = '<|video_pad|>'
        self.vision_start_token = '<|vision_start|>'
        self.vision_end_token = '<|vision_end|>'

    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self._tokenizer.eos_token_id

    @property
    def bos_token_id(self):
        return self._tokenizer.bos_token_id

    @property
    def image_token_id(self):
        return self.special_tokens_map[self.image_token]

    @property
    def video_token_id(self):
        return self.special_tokens_map[self.video_token]

    @property
    def vision_start_token_id(self):
        return self.special_tokens_map[self.vision_start_token]

    @property
    def vision_end_token_id(self):
        return self.special_tokens_map[self.vision_end_token]


class Qwen2VlDataset(MultiModalDataset):
    def __init__(
        self,
        train_max_seqlen,
        min_pixels_num,
        max_pixels_num,
        use_for_hf,
        mask_history,
        use_grpo,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.min_pixels_num = min_pixels_num
        self.max_pixels_num = max_pixels_num
        self.use_for_hf = use_for_hf
        self.mask_history = mask_history
        self.use_grpo = use_grpo
        self.train_max_seqlen = train_max_seqlen

    # From: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/processing_qwen2_vl.py
    def process_vision(self, images, videos=None):
        if images is not None:
            image_inputs = self.image_processor(
                images=images,
                videos=None,
                return_tensors="pt",
            )
        else:
            image_inputs = {}

        if videos is not None:
            videos_inputs = self.image_processor(
                images=None,
                videos=videos,
                return_tensors="pt",
            )
        else:
            videos_inputs = {}

        return BatchFeature(data={**image_inputs, **videos_inputs})

    # From: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/processing_qwen2_vl.py
    def padding_vision_token(self, text: str, image_grid_thw, video_grid_thw=None):
        merge_length = self.image_processor.merge_size**2
        if image_grid_thw is not None:
            index = 0
            while self.tokenizer.image_token in text:
                text = text.replace(
                    self.tokenizer.image_token,
                    "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
                )
                index += 1
            text = text.replace("<|placeholder|>", self.tokenizer.image_token)

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            while self.tokenizer.video_token in text:
                text = text.replace(
                    self.tokenizer.video_token,
                    "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length), 1
                )
                index += 1
            text = text.replace("<|placeholder|>", self.tokenizer.video_token)

        return text

    def get_image_token_cnt(self, image_grid_thw, video_grid_thw=None):
        merge_length = self.image_processor.merge_size**2
        total_cnt = torch.tensor(0, dtype=torch.long)
        if image_grid_thw is not None:
            for i in range(image_grid_thw.shape[0]):
                total_cnt += image_grid_thw[i].prod() // merge_length

        if video_grid_thw is not None:
            for i in range(video_grid_thw.shape[0]):
                total_cnt += video_grid_thw.prod() // merge_length

        return total_cnt.item()

    def convert_example(
        self,
        example,
        conversations,
        imgs,
        domain_states,
        tools=None,
        answer=None,
    ):
        # 现在都要图片大于等于1张
        media_info = self.process_vision(imgs)
        image_grid_thw = media_info.get("image_grid_thw", None)

        add_generation_prompt = False
        if self.use_grpo and conversations[-1]['role'] == "assistant":
            conversations = conversations[:-1]
        if self.use_grpo:
            assert conversations[-1]['role'] != "assistant"
            add_generation_prompt = True

        all_text = self.processor.apply_chat_template(
            conversations,
            tools=tools,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            add_vision_id=False,
        )
        all_text = self.padding_vision_token(all_text, image_grid_thw)
        all_text_tokenizer = self.tokenizer._tokenizer(all_text, padding=False)
        input_ids = all_text_tokenizer.input_ids
        attention_mask = all_text_tokenizer.attention_mask
        labels = torch.tensor(input_ids, dtype=torch.int64)
        label_mask = self.gen_label_mask(conversations, imgs, tools, rm_bos=False)
        if self.use_grpo:
            assert len(label_mask) == 1 and label_mask[0][0] == 0
        for mask in label_mask:
            labels[mask[0]:mask[1]] = -100
        prompt_len = label_mask[-1][-1]
        labels = labels.tolist()
        tmp_max_seq_len = self.max_seq_len
        if len(input_ids) > self.max_seq_len:
            # grpo train need to complete sentences
            if self.use_grpo:
                domain_states.domain_lines += example["domain_line"]
                print(f"GRPO Abort Sample at dp-rank:{self.underlying.dp_rank}[too long]")
                return None
        if self.use_grpo:
            # for privode position id
            tmp_max_seq_len = self.train_max_seqlen

        if len(input_ids) < tmp_max_seq_len + 1:
            if self.moe_pad_with_random_token:
                ban_token_ids = [
                    self.tokenizer.image_token_id, self.tokenizer.video_token_id,
                    self.tokenizer.vision_start_token_id, self.tokenizer.vision_end_token_id
                ]
                input_ids = random_pad_list(
                    input_ids, tmp_max_seq_len + 1 - len(input_ids), ban_token_ids
                )
            else:
                input_ids += [self.tokenizer._tokenizer.pad_token_id
                             ] * (tmp_max_seq_len + 1 - len(input_ids))
            labels += [-100] * (tmp_max_seq_len + 1 - len(labels))
            attention_mask += [0] * (tmp_max_seq_len + 1 - len(attention_mask))

        input_ids = input_ids[:-1]
        attention_mask = attention_mask[:-1]
        if self.use_for_hf:
            labels = labels[:-1]
        else:
            labels = labels[1:]
        if len(input_ids) > tmp_max_seq_len:
            input_ids = input_ids[-tmp_max_seq_len:]
            labels = labels[-tmp_max_seq_len:]
            attention_mask = attention_mask[-tmp_max_seq_len:]

        example["input_ids"] = torch.tensor(input_ids, dtype=torch.int64)
        example["labels"] = torch.tensor(labels, dtype=torch.int64)
        example["attention_mask"] = torch.tensor(attention_mask, dtype=torch.bool)
        example["pixel_values"] = media_info.get("pixel_values", None)
        example["image_grid_thw"] = image_grid_thw
        if self.image_token_id is not None:
            assert self.tokenizer.image_token_id == self.image_token_id
        example["image_input_mask"] = example["input_ids"] == self.tokenizer.image_token_id

        # 增加样本计数
        domain_states.domain_lines += example["domain_line"]

        sum_image_token = example["image_input_mask"].sum().cpu().item()
        total_image_token = self.get_image_token_cnt(image_grid_thw)
        if self.use_grpo:
            all_ignore = False
        else:
            all_ignore = torch.all(example["labels"] == -100).item()
        assert total_image_token >= sum_image_token
        # 跳过样本
        if total_image_token > sum_image_token or all_ignore:
            print(f"Abort Sample at dp-rank:{self.underlying.dp_rank}")
            # print(f"Abort Sample {json.dumps(json_data, ensure_ascii=False)}")
            return None

        example["domain_line"] = torch.tensor(domain_states.domain_lines, dtype=torch.int64)
        example["prompt_len"] = torch.tensor(prompt_len, dtype=torch.int64)
        domain_states.domain_lines = 0
        return example

    def __iter__(self):
        domain_states = SimpleNamespace(domain_lines=0)
        for example in self.underlying:
            # json_data 就是从jsonl读出的数据
            json_data = example["json_data"]
            imgs = None
            if 'images' in json_data and len(json_data['images']) > 0:
                imgs = fetch_images(json_data['images'], self.tar_dir, self.lmdb_port)
                imgs_valid = True
                for img in imgs:
                    if img is None:
                        imgs_valid = False
                        break
                    width, height = img.size
                    if width < IMAGE_FACTOR or height < IMAGE_FACTOR:
                        imgs_valid = False
                        break
                    if max(height, width) / min(height, width) > MAX_RATIO:
                        imgs_valid = False
                        break
                if not imgs_valid:
                    domain_states.domain_lines += example["domain_line"]
                    print(f"Abort Sample at dp-rank:{self.underlying.dp_rank}[invalid image]")
                    continue

            conversations = convert_conversations(json_data['conversations'])
            tools = None
            if 'tools' in json_data:
                tools = json_data['tools']
            answer = None
            if 'label' in json_data:
                answer = json_data['label']
            assert len(conversations) > 1
            del example["json_data"]

            # NOTE(guanyouhe): 这里 python/sglang/srt/multimodal/processors/qwen_vl.py 都做了 resize
            # qwen3vl processors 与 qwen2vl/qwen2.5vl 有所不同（应该是因为 qwen3vl 使用 Qwen2VLImageProcessorFast）
            # 所以这里得先做 resize
            if Qwen3VLProcessor is not None and isinstance(self.processor, Qwen3VLProcessor) \
            and imgs is not None:
                imgs = [
                    resize_image(ele, img, self.min_pixels_num, self.max_pixels_num)
                    for ele, img in zip(json_data['images'], imgs)
                ]

            example_copy = deepcopy(example)
            example_copy = self.convert_example(
                example_copy, conversations, imgs, domain_states, tools, answer
            )
            if example_copy is None:
                continue

            if self.use_grpo:
                example_copy["json_data"] = json_data
                imgs_np_array = None
                if imgs is not None:
                    imgs_np_array = [
                        np.array(resize_image(ele, img, self.min_pixels_num, self.max_pixels_num))
                        for ele, img in zip(json_data['images'], imgs)
                    ]
                example_copy["imgs_np_array"] = imgs_np_array
            yield example_copy


class Qwen2VlDatasetDPO(Qwen2VlDataset):
    def __iter__(self):
        domain_states = SimpleNamespace(domain_lines=0)
        for example in self.underlying:
            # json_data 就是从jsonl读出的数据
            json_data = example["json_data"]
            imgs = None
            if 'images' in json_data and len(json_data['images']) > 0:
                imgs = fetch_images(json_data['images'], self.tar_dir, self.lmdb_port)
                # TODO(guanyouhe): 这里不需要resize，因为process中会做，删除这里做一版wandb base
                imgs = [resize_image(ele, img) for ele, img in zip(json_data['images'], imgs)]
            conversations_chosen = deepcopy(json_data['conversations'])
            conversations_chosen.append(json_data['chosen'])
            conversations_rejected = deepcopy(json_data['conversations'])
            conversations_rejected.append(json_data['rejected'])
            conversations_chosen = convert_conversations(conversations_chosen)
            conversations_rejected = convert_conversations(conversations_rejected)
            assert len(conversations_rejected) > 1 and len(conversations_chosen) > 1
            tools = None
            if 'tools' in json_data:
                tools = json_data['tools']
            del example["json_data"]

            example_chosen = deepcopy(example)
            example_chosen = self.convert_example(
                example_chosen, conversations_chosen, imgs, domain_states, tools, None
            )
            example_rejected = deepcopy(example)
            example_rejected = self.convert_example(
                example_rejected, conversations_rejected, imgs, domain_states, tools
            )
            if example_chosen is None or example_rejected is None:
                assert domain_states.domain_lines >= 2 * (example["domain_line"])
                domain_states.domain_lines -= example["domain_line"]
                continue

            # example_chosen先运行，它的domain_line是正确的
            if "domain_line" in example_chosen:
                # 只需要记录一个domain_line
                example_rejected["domain_line"] = torch.tensor(0, dtype=torch.int64)
            yield example_chosen
            yield example_rejected


def get_rope_index(
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    tokenizer=None,
    spatial_merge_size=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embeddin for text part.
        Examples:
            Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [3, 4, 5, 6, 7]
            text height position_ids: [3, 4, 5, 6, 7]
            text width position_ids: [3, 4, 5, 6, 7]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    image_token_id = tokenizer.image_token_id
    video_token_id = tokenizer.video_token_id
    vision_start_token_id = tokenizer.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device
        )
        image_index, video_index = 0, 0
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1,
                                                        1).expand(-1,
                                                                  llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1,
                                                        1).expand(llm_grid_t, -1,
                                                                  llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1,
                                                        -1).expand(llm_grid_t, llm_grid_h,
                                                                   -1).flatten()
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas,
                                             device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1],
                             device=input_ids.device).view(1, 1,
                                                           -1).expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def get_rope_index_2p5(
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    tokenizer=None,
    spatial_merge_size=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embeddin for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    image_token_id = tokenizer.image_token_id
    video_token_id = tokenizer.video_token_id
    vision_start_token_id = tokenizer.vision_start_token_id
    tokens_per_second = 2  # 所有的qwen2.5vl都是一样的，写死
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * tokens_per_second

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = torch.arange(llm_grid_h).view(1, -1,
                                                        1).expand(llm_grid_t, -1,
                                                                  llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1,
                                                        -1).expand(llm_grid_t, llm_grid_h,
                                                                   -1).flatten()
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas,
                                             device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1],
                             device=input_ids.device).view(1, 1,
                                                           -1).expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


# Slightly modified from Qwen3VLModel.get_rope_index
def get_rope_index_3vl(
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

    # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                t_index = torch.arange(llm_grid_t).view(-1,
                                                        1).expand(-1,
                                                                  llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1,
                                                        1).expand(llm_grid_t, -1,
                                                                  llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1,
                                                        -1).expand(llm_grid_t, llm_grid_h,
                                                                   -1).flatten()
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas,
                                             device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1],
                             device=input_ids.device).view(1, 1,
                                                           -1).expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def get_ltor_masks_and_position_ids(
    input_ids,
    image_thw_grids,
    video_thw_grids,
    second_per_grid_ts,
    target,
    pad_token,
    ignore_index=None,
    model_arch="qwen2vl",
    tokenizer=None,
    spatial_merge_size=None,
    attention_mask=None,
    hf_config=None,
):
    """Build masks and position id for left to right model."""
    # Position ids. [3 X bs X seqlen]
    if model_arch in ["qwen2.5vl"]:
        position_ids, _ = get_rope_index_2p5(
            input_ids=input_ids,
            image_grid_thw=image_thw_grids,
            video_grid_thw=video_thw_grids,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            spatial_merge_size=spatial_merge_size,
        )
    elif model_arch in ["qwen2vl"]:
        position_ids, _ = get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_thw_grids,
            video_grid_thw=video_thw_grids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            spatial_merge_size=spatial_merge_size,
        )
    elif model_arch in ["qwen3_vl_moe", "qwen3_vl"]:
        position_ids, _ = get_rope_index_3vl(
            hf_config.vision_config.spatial_merge_size,
            hf_config.image_token_id,
            hf_config.video_token_id,
            hf_config.vision_start_token_id,
            input_ids,
            image_thw_grids,
            None,
            attention_mask,
        )
    else:
        assert False, f"not support this model arch:{model_arch}"
    # Loss mask.
    loss_mask = torch.ones(target.size(), dtype=torch.float, device=input_ids.device)
    loss_mask[target == pad_token] = 0.0  # mask paddings
    if ignore_index is not None:
        loss_mask[target == ignore_index] = 0.0  # mask prompts

    return loss_mask, position_ids


class DataCollatorForQwen2Vl(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(
        self,
        hw_factor: int = 1,
        model_arch="qwen2vl",
        tokenizer=None,
        spatial_merge_size=None,
        is_dpo=False,
        use_grpo=False,
        cp_size=1,
        hf_config=None,
    ):
        super().__init__()
        # qwen2vl所有的模型merge_size都为2，因此它本来就是2*2的倍数
        self.hw_factor = hw_factor * 4
        self.model_arch = model_arch
        self.tokenizer = tokenizer
        self.spatial_merge_size = spatial_merge_size
        self.is_dpo = is_dpo
        self.use_grpo = use_grpo
        self.cp_size = cp_size
        self.hf_config = hf_config

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if self.is_dpo:
            assert len(instances) % 2 == 0
            # 正负样本交错出现，要换好顺序
            instances = instances[::2] + instances[1::2]
        new_instances = []
        pixel_values = []
        image_grid_thws = []
        json_data_list = []
        imgs_np_array_list = []
        for instance in instances:
            if instance["pixel_values"] is not None:
                pixel_values.append(instance["pixel_values"])
                image_grid_thws.append(instance["image_grid_thw"])
            del instance["pixel_values"]
            del instance["image_grid_thw"]
            if self.use_grpo:
                json_data_list.append(instance["json_data"])
                del instance["json_data"]
                imgs_np_array_list.append(instance["imgs_np_array"])
                del instance["imgs_np_array"]
            new_instances.append(instance)

        nopad_image_grid_thw = None
        if len(image_grid_thws) > 0:
            nopad_image_grid_thw = torch.cat(image_grid_thws, dim=0)
        res = default_collate(new_instances)
        if len(pixel_values) > 0:
            # pad empty and split image for tp/sp/cp
            cp_size = 1 if self.use_grpo else self.cp_size
            pixel_values, image_grid_thws, cp_img_num, images_padded = qwen2vl_pad_and_split(
                cp_size,
                self.hw_factor,
                pixel_values,
                image_grid_thws,
            )
            if self.model_arch in ["qwen3_vl_moe", "qwen3_vl"]:
                for image_padded in images_padded:
                    assert not image_padded, "not support image padded now"
            res["pixel_values"] = torch.cat(pixel_values, dim=0)
            res["image_grid_thw"] = torch.cat(image_grid_thws, dim=0)
            res["has_image"] = torch.tensor([True], dtype=torch.bool)
            res["images_padded"] = torch.tensor(images_padded, dtype=torch.int64)
            res["cp_img_num"] = torch.tensor(cp_img_num, dtype=torch.int64)
        else:
            res["has_image"] = torch.tensor([False], dtype=torch.bool)

        second_per_grid_ts = None  # 这个参数是从视频中拿到的，现在还没有支持视频
        loss_mask, position_ids = get_ltor_masks_and_position_ids(
            res["input_ids"],
            nopad_image_grid_thw,
            None,
            second_per_grid_ts,
            res["labels"],
            self.tokenizer.pad_token_id,
            ignore_index=-100,
            model_arch=self.model_arch,
            tokenizer=self.tokenizer,
            spatial_merge_size=self.spatial_merge_size,
            hf_config=self.hf_config,
        )
        res["loss_mask"] = loss_mask
        if len(pixel_values) > 0:
            res["position_ids"] = position_ids
        else:
            res["position_ids"] = position_ids.clone()
        if self.use_grpo:
            res["json_data_list"] = json_data_list
            res["imgs_np_array_list"] = imgs_np_array_list
        return res


def get_processor(args):
    processor_path = args.processor_path
    if args.model_arch in ["qwen3_vl_moe", "qwen3_vl"]:
        min_pixels = args.min_pixels_num
        max_pixels = args.max_pixels_num
    else:
        min_pixels = args.min_pixels_num if args.min_pixels_num else MIN_PIXELS
        max_pixels = args.max_pixels_num if args.max_pixels_num else MAX_PIXELS
    init_kwargs = {
        "trust_remote_code": True,
        "cache_dir": None,
        "token": None,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "use_fast": True,
    }
    processor = AutoProcessor.from_pretrained(processor_path, **init_kwargs)
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return processor


def build_train_valid_test_datasets(
    args,
    tokenizer,
    rank=0,
    dp_rank=0,
    dp_size=1,
    use_for_hf=False,
    is_dpo=False,
):
    train_path_likes = args.data_path
    eval_path_likes = args.px_eval_data_path
    domain_probabilities = args.px_domain_probabilities
    retention_rates_per_domains = args.px_retention_rates_per_domain
    domain_names = args.px_train_data_domain_names
    enable_pareto = args.px_train_apply_pareto
    pareto_alpha = args.px_train_pareto_alpha
    pareto_scale = args.px_train_pareto_scale
    pareto_score_scale = args.train_pareto_score_scale
    processor = get_processor(args)
    mask_history = args.mask_history
    use_grpo = args.use_grpo
    max_seq_length = args.seq_length
    if use_grpo:
        assert mask_history, f"mask_history must be True when use grpo"
        max_seq_length -= args.ppo_resp_seq_len

    dataset_class = Qwen2VlDataset
    if is_dpo:
        dataset_class = Qwen2VlDatasetDPO

    print_rank_0(
        f'build_train_valid_datasets train_data_consuming_progresses {args.train_data_consuming_progresses}'
    )
    train_ds = dataset_class(
        args.seq_length,
        args.min_pixels_num,
        args.max_pixels_num,
        use_for_hf,
        mask_history,
        use_grpo,
        tokenizer,
        max_seq_length,
        train_path_likes,
        domain_probabilities,
        domain_names,
        args.global_batch_size,
        train_data_consuming_progresses=args.train_data_consuming_progresses,
        rank=rank,
        dp_rank=dp_rank,
        dp_size=dp_size,
        num_workers=args.num_workers,
        access_policy_interleave=False,
        shuffle_buffer_size=args.px_shuffle_buffer_size,
        seed=args.seed,
        train=True,
        retention_rates_per_domains=retention_rates_per_domains,
        unsplit_eval_data=False,
        enable_pareto=enable_pareto,
        pareto_alphas=pareto_alpha,
        pareto_scales=pareto_scale,
        pareto_score_scales=pareto_score_scale,
        top_domains_to_cut=args.px_top_domains_to_cut,
        processor=processor,
        tar_dir=args.tarfile_path,
        lmdb_port=args.lmdb_port,
        image_token_id=args.image_token_id,
        moe_pad_with_random_token=args.moe_pad_with_random_token,
    )

    eval_ds = None
    if eval_path_likes is not None:
        # NOTE(guanyouhe): 尝未测试每次eval是否都一样
        eval_ds = dataset_class(
            args.seq_length,
            args.min_pixels_num,
            args.max_pixels_num,
            use_for_hf,
            mask_history,
            use_grpo,
            tokenizer,
            max_seq_length,
            eval_path_likes,
            [1.0],  # 写入概率
            args.px_eval_data_domain_names,
            args.global_batch_size,
            train_data_consuming_progresses=None,
            rank=rank,
            dp_rank=dp_rank,
            dp_size=dp_size,
            num_workers=args.num_workers,
            access_policy_interleave=False,
            shuffle_buffer_size=args.px_shuffle_buffer_size,
            seed=args.seed,
            train=False,
            retention_rates_per_domains=retention_rates_per_domains,
            unsplit_eval_data=False,
            enable_pareto=enable_pareto,
            pareto_alphas=pareto_alpha,
            pareto_scales=pareto_scale,
            pareto_score_scales=pareto_score_scale,
            top_domains_to_cut=args.px_top_domains_to_cut,
            processor=processor,
            tar_dir=args.tarfile_path,
            lmdb_port=args.lmdb_port,
            image_token_id=args.image_token_id,
            moe_pad_with_random_token=args.moe_pad_with_random_token,
        )
        assert args.px_reset_dataloader_at_start_of_eval, "需要--px-reset-dataloader-at-start-of-eval来保保证每次eval的数据是一样的"
    test_ds = None

    return train_ds, eval_ds, test_ds


def build_train_valid_test_data_iter(
    args,
    tokenizer,
    rank=0,
    dp_rank=0,
    dp_size=1,
    use_for_hf=False,
    is_dpo=False,
):
    tokenizer.__class__ = Qwen2VLTokenizer
    tokenizer.qwen2vl_init()
    train_ds, eval_ds, test_ds = build_train_valid_test_datasets(
        args,
        tokenizer,
        rank,
        dp_rank,
        dp_size,
        use_for_hf=use_for_hf,
        is_dpo=is_dpo,
    )

    hw_factor = args.context_parallel_size
    if args.sequence_parallel:
        hw_factor *= args.tensor_model_parallel_size
    # grpo数据先不pad
    if args.use_grpo or args.model_arch in ["qwen3_vl_moe", "qwen3_vl"]:
        hw_factor = 1

    hf_config = None
    if args.model_arch in ["qwen3_vl_moe", "qwen3_vl"]:
        from transformers import Qwen3VLConfig
        hf_config = Qwen3VLConfig.from_pretrained(args.processor_path)
        assert hf_config.image_token_id == tokenizer.image_token_id
        assert hf_config.video_token_id == tokenizer.video_token_id
        assert hf_config.vision_start_token_id == tokenizer.vision_start_token_id
        assert hf_config.vision_end_token_id == tokenizer.vision_end_token_id

    collate_func = DataCollatorForQwen2Vl(
        hw_factor=hw_factor,
        model_arch=args.model_arch,
        tokenizer=tokenizer,
        spatial_merge_size=args.spatial_merge_size,
        is_dpo=is_dpo,
        use_grpo=args.use_grpo,
        cp_size=args.context_parallel_size,
        hf_config=hf_config,
    )

    batch_size = args.micro_batch_size
    if args.use_grpo:
        batch_size = args.ppo_rollout_micro_batch_size
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_func,
        prefetch_factor=args.px_dataloader_prefetch_factor,
    )

    eval_dataloader = None
    if eval_ds is not None:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_func,
            prefetch_factor=args.px_dataloader_prefetch_factor,
        )
    test_dataloader = None
    if test_ds is not None:
        test_dataloader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_func,
            prefetch_factor=args.px_dataloader_prefetch_factor,
        )
    if use_for_hf:
        return train_dataloader, eval_dataloader, test_dataloader
    return get_iterator(train_dataloader), get_iterator(eval_dataloader
                                                       ), get_iterator(test_dataloader)
