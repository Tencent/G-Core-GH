# copyright (c) 2024 tencent inc. all rights reserved.
# guanyouhe@tencent.com

import json
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataloader import default_collate
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.auto.processing_auto import AutoProcessor

from gdataset import GDatasetV4
from gdataset.feat import PilImageListFeat
from megatron_datasets.utils import get_iterator
from tasks.multimodal_comm.multimodal_dataset_map import (
    MultiModalDatasetMap,
    convert_conversations,
    remove_bos,
)

from megatron.training.tokenizer.tokenizer import _HuggingFaceTokenizer

from gpatch.training.v3.ppo_actor import iter_to_ppo_epoch_step


class Glm4vTokenizer(_HuggingFaceTokenizer):
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.init_glm_4v()

    def init_glm_4v(self):
        self.special_tokens_map = {
            k: v
            for k, v in zip(self._tokenizer.all_special_tokens, self._tokenizer.all_special_ids)
        }
        self.image_token = (
            "<|image|>"
            if not hasattr(self._tokenizer, "image_token") else self._tokenizer.image_token
        )
        self.video_token = (
            "<|video|>"
            if not hasattr(self._tokenizer, "video_token") else self._tokenizer.video_token
        )
        self.video_start_token = "<|begin_of_video|>"
        self.video_end_token = "<|end_of_video|>"
        self._image_token_id = self._tokenizer.convert_tokens_to_ids(self.image_token)
        self._video_token_id = self._tokenizer.convert_tokens_to_ids(self.video_token)

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
        return self._image_token_id

    @property
    def video_token_id(self):
        return self._video_token_id

    @property
    def video_start_token_id(self):
        return self.special_tokens_map[self.video_start_token]

    @property
    def video_end_token_id(self):
        return self.special_tokens_map[self.video_end_token]


class Glm4vDatasetMap(MultiModalDatasetMap):
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
        keep_text_and_image=False,  # for test
    ):
        super().__init__(
            use_for_hf=use_for_hf,
            use_grpo=use_grpo,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            processor=processor,
            image_token_id=image_token_id,
            mask_history=mask_history,
            meta_info_key=meta_info_key,
        )
        self.keep_text_and_image = keep_text_and_image

    def padding_vision_token(self, text: str, image_grid_thw, video_grid_thw=None):
        merge_length = self.image_processor.merge_size**2
        if image_grid_thw is not None:
            index = 0
            while self.tokenizer.image_token in text:
                text = text.replace(
                    self.tokenizer.image_token,
                    "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length),
                    1,
                )
                index += 1
            text = text.replace("<|placeholder|>", self.tokenizer.image_token)

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            while self.tokenizer.video_token in text:
                text = text.replace(
                    self.tokenizer.video_token,
                    "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length),
                    1,
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

    def gen_label_mask(
        self,
        conversations,
        image_grid_thw,
        tools,
        label_role=["assistant"],
        rm_bos=True,
    ):
        pre_len = 0
        mask_indexs = []
        for i in range(len(conversations)):
            if conversations[i]["role"] in ["system"]:
                continue
            add_generation_prompt = False
            if conversations[i]["role"] in ["user"]:
                add_generation_prompt = True
            text = self.processor.apply_chat_template(
                conversations[:i + 1],
                tools=tools,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            if rm_bos:
                text = remove_bos(text)

            text = self.padding_vision_token(text, image_grid_thw)
            text_tokenizer = self.tokenizer._tokenizer(text, padding=False)
            cur_len = len(text_tokenizer.input_ids)
            if conversations[i]["role"] not in label_role:
                mask_indexs.append([pre_len, cur_len])
            pre_len = cur_len
        if self.mask_history:
            mask_indexs = [[mask_indexs[0][0], mask_indexs[-1][-1]]]
        return mask_indexs

    def convert_example(
        self,
        conversations,
        imgs,
        tools=None,
        answer=None,
    ):
        # 现在都要图片大于等于1张
        add_generation_prompt = False
        if self.use_grpo and conversations[-1]["role"] == "assistant":
            conversations = conversations[:-1]
        if self.use_grpo:
            assert conversations[-1]["role"] != "assistant"
            add_generation_prompt = True

        text = self.processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            add_vision_id=False,
        )
        inputs = self.processor(
            text=[text], images=imgs, padding=False, return_mm_token_type_ids=True
        )
        input_ids = inputs.input_ids[0]
        attention_mask = inputs.attention_mask[0]
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        if pixel_values is not None and not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.from_numpy(pixel_values)
            image_grid_thw = torch.from_numpy(image_grid_thw)

        # fast and non fast processor return different shape, unify it to [patch_num, hidden_size]
        if pixel_values is not None:
            pixel_values = pixel_values.reshape((-1, pixel_values.shape[-1]))

        # print(f"pixel_values {type(pixel_values)}, {pixel_values.shape}; image_grid_thw {type((image_grid_thw))} {image_grid_thw.shape}; ", flush=True)
        labels = torch.tensor(input_ids, dtype=torch.int64)
        label_mask = self.gen_label_mask(conversations, image_grid_thw, tools, rm_bos=False)
        if self.use_grpo:
            assert len(label_mask) == 1 and label_mask[0][0] == 0
        for mask in label_mask:
            labels[mask[0]:mask[1]] = -100
        prompt_len = label_mask[-1][-1]
        tokenizer_len = len(input_ids)
        labels = labels.tolist()
        if len(input_ids) > self.max_seq_len:
            # grpo train need to complete sentences
            if self.use_grpo:
                return f"GRPO Invalid Sample: sample too long"

        if len(input_ids) < self.max_seq_len + 1:
            input_ids += [self.tokenizer._tokenizer.pad_token_id
                         ] * (self.max_seq_len + 1 - len(input_ids))
            labels += [self.tokenizer.eos_token_id] + [-100] * (self.max_seq_len - len(labels))
            attention_mask += [0] * (self.max_seq_len + 1 - len(attention_mask))

        input_ids = input_ids[:-1]
        attention_mask = attention_mask[:-1]
        if self.use_for_hf:
            labels = labels[:-1]
        else:
            labels = labels[1:]
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[-self.max_seq_len:]
            labels = labels[-self.max_seq_len:]
            attention_mask = attention_mask[-self.max_seq_len:]

        data_dict = {}
        data_dict["input_ids"] = torch.tensor(input_ids, dtype=torch.int64)
        data_dict["labels"] = torch.tensor(labels, dtype=torch.int64)
        data_dict["attention_mask"] = torch.tensor(attention_mask, dtype=torch.bool)
        data_dict["pixel_values"] = pixel_values
        data_dict["image_grid_thw"] = image_grid_thw
        if self.image_token_id is not None:
            assert (
                self.tokenizer.image_token_id == self.image_token_id
            ), f"{self.tokenizer.image_token_id} vs {self.image_token_id}"
        data_dict["image_input_mask"] = (data_dict["input_ids"] == self.tokenizer.image_token_id)

        sum_image_token = data_dict["image_input_mask"].sum().cpu().item()
        total_image_token = self.get_image_token_cnt(image_grid_thw)
        if self.use_grpo:
            all_ignore = False
        else:
            all_ignore = torch.all(data_dict["labels"] == -100).item()
        assert total_image_token >= sum_image_token
        # 跳过样本
        if total_image_token > sum_image_token or all_ignore:
            return f"Invalid Sample: image token-ids too long"

        data_dict["prompt_len"] = torch.tensor(prompt_len, dtype=torch.int64)
        data_dict["tokenizer_len"] = torch.tensor(tokenizer_len, dtype=torch.int64)
        if self.keep_text_and_image:
            data_dict["imgs"] = imgs
            data_dict["text"] = text

        return data_dict

    def process(self, example):
        imgs = example.pop("__images_feat__", [])  # read from feat
        """
        imgs_valid = True
        for img in imgs:
            assert img is not None, f"the image is invalid"
            width, height = img.size
            if width < IMAGE_FACTOR or height < IMAGE_FACTOR:
                imgs_valid = False
                break
            if max(height, width) / min(height, width) > MAX_RATIO:
                imgs_valid = False
                break
        if not imgs_valid:
            return "image is to small"
        """

        if len(imgs) == 0:
            imgs = None
        conversations = convert_conversations(example["conversations"])
        tools = None
        if "tools" in example:
            tools = example["tools"]
        answer = None
        if "label" in example:
            answer = example["label"]
        assert len(conversations) > 1

        data_dict = self.convert_example(conversations, imgs, tools, answer)
        if self.use_grpo and isinstance(data_dict, dict):
            data_dict["json_data"] = example

            imgs_np_array = None
            if imgs is not None:
                imgs_np_array = [np.array(e) for e in imgs]
            data_dict["imgs_np_array"] = imgs_np_array
            """
            imgs_np_array = None
            if imgs is not None:
                imgs_np_array = [
                    np.array(resize_image(ele, img, self.min_pixels_num, self.max_pixels_num))
                    for ele, img in zip(example['images'], imgs)
                ]
            data_dict["imgs_np_array"] = imgs_np_array
            """
        if isinstance(data_dict, dict):
            if self.meta_info_key is not None:
                data_dict["meta_info"] = example.get(self.meta_info_key, None)
            else:
                data_dict["meta_info"] = None
        return data_dict

    def __call__(self, example):
        data_dict = self.process(example)
        assert isinstance(data_dict, dict), f"Should make sure the sample is valid: {data_dict}"
        return data_dict


def get_ltor_masks_and_position_ids(
    input_ids,
    image_thw_grids,
    video_thw_grids,
    second_per_grid_ts,
    target,
    vision_config,
    ignore_index=None,
    attention_mask=None,
):
    """Build masks and position id for left to right model."""
    # Position ids. [3 X bs X seqlen]
    from mbridge.models.glm4_vl.vl_mixin import VLMixin

    vl_mixin = VLMixin()
    vl_mixin.config = vision_config
    assert video_thw_grids is None
    assert second_per_grid_ts is None
    position_ids, _ = vl_mixin.get_rope_index(input_ids, image_thw_grids)
    # Loss mask.
    loss_mask = torch.ones(target.size(), dtype=torch.float, device=input_ids.device)
    pad_token = vision_config.pad_token_id
    # eos is also pad for glm4v
    # loss_mask[target == pad_token] = 0.0  # mask paddings
    if ignore_index is not None:
        loss_mask[target == ignore_index] = 0.0  # mask prompts

    return loss_mask, position_ids


@dataclass
class DataCollatorForGlm4v(object):
    """Collate examples for supervised fine-tuning."""
    def __init__(
        self,
        vision_config,
        hw_factor: int = 1,
        tokenizer=None,
        is_dpo=False,
        use_grpo=False,
    ):
        super().__init__()
        self.vision_config = vision_config
        self.hw_factor = hw_factor * (vision_config.spatial_merge_size**2)
        self.tokenizer = tokenizer
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.is_dpo = is_dpo
        self.use_grpo = use_grpo

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if self.is_dpo:
            assert len(instances) % 2 == 0
            # 正负样本交错出现，要换好顺序
            instances = instances[::2] + instances[1::2]
        new_instances = []
        pixel_values = []
        image_grid_thws = []
        seq_len = 0
        json_data_list = []
        meta_info_list = []
        imgs_np_array_list = []
        for instance in instances:
            if instance["pixel_values"] is not None:
                seq_len += instance["pixel_values"].size(0)
                pixel_values.append(instance["pixel_values"])
                image_grid_thws.append(instance["image_grid_thw"])
            del instance["pixel_values"]
            del instance["image_grid_thw"]
            if self.use_grpo:
                json_data_list.append(instance["json_data"])
                del instance["json_data"]
                imgs_np_array_list.append(instance["imgs_np_array"])
                del instance["imgs_np_array"]
            meta_info_list.append(instance["meta_info"])
            del instance["meta_info"]
            new_instances.append(instance)
        image_padded = False
        # TODO(hessianliu): do this when vision tp is supported
        """
        image_padded = 0 != seq_len % self.hw_factor
        if image_padded:
            padded_seqlen = (
                seq_len + self.hw_factor - 1
            ) // self.hw_factor * self.hw_factor - seq_len
            assert padded_seqlen > 0 and padded_seqlen % 4 == 0
            pixel_values.append(
                torch.zeros(
                    [padded_seqlen, pixel_values[0].size(-1)],
                    dtype=pixel_values[0].dtype,
                    device=pixel_values[0].device,
                )
            )
            image_grid_thws.append(
                torch.tensor(
                    [[1, 2, padded_seqlen // 2]],
                    dtype=image_grid_thws[0].dtype,
                    device=image_grid_thws[0].device,
                )
            )
        """
        res = default_collate(new_instances)
        if len(pixel_values) > 0:
            res["pixel_values"] = torch.cat(pixel_values, dim=0)
            res["image_grid_thw"] = torch.cat(image_grid_thws, dim=0)
            res["has_image"] = torch.tensor([True], dtype=torch.bool)
        else:
            res["has_image"] = torch.tensor([False], dtype=torch.bool)
        res["image_padded"] = torch.tensor([image_padded], dtype=torch.bool)

        second_per_grid_ts = None  # 这个参数是从视频中拿到的，现在还没有支持视频
        loss_mask, position_ids = get_ltor_masks_and_position_ids(
            res["input_ids"],
            res.get("image_grid_thw", None),
            None,
            second_per_grid_ts,
            res["labels"],
            self.vision_config,
            ignore_index=-100,
        )
        res["loss_mask"] = loss_mask
        if len(pixel_values) > 0:
            res["position_ids"] = position_ids
        else:
            res["position_ids"] = position_ids.clone()
        if self.use_grpo:
            res["json_data_list"] = json_data_list
            res["imgs_np_array_list"] = imgs_np_array_list
        res["meta_info"] = meta_info_list
        return res


def load_processor(hf_model_path):
    processor = AutoProcessor.from_pretrained(hf_model_path, use_fast=True, trust_remote_code=True)
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor


def get_processor(args):
    return load_processor(args.processor_path)


def sort_by_prompt_len(sample):
    return sample["tokenizer_len"]


def build_train_valid_test_datasets(
    args,
    tokenizer,
    processor,
    rank=0,
    dp_rank=0,
    dp_size=1,
    use_for_hf=False,
    is_dpo=False,
    feats=None,
):
    assert not is_dpo, "not support now"
    train_path_like = args.gdatasetv4_train_metadata_file
    eval_path_like = args.gdatasetv4_eval_metadata_file
    mask_history = args.mask_history
    use_grpo = args.use_grpo
    if use_grpo:
        assert mask_history, f"mask_history must be True when use grpo"

    gbs = args.global_batch_size
    consumed = args.iteration * gbs
    if args.use_grpo and consumed > 0:
        gbs = args.ppo_rollout_global_batch_size
        _, ppo_step = iter_to_ppo_epoch_step(args.iteration)
        consumed = ppo_step * gbs

    smart_padding_compare_func = None
    smart_padding_buffer_size = 0
    if args.px_inputs_pad_to_longest:
        smart_padding_compare_func = sort_by_prompt_len
        smart_padding_buffer_size = args.px_smart_padding_buffer_size

    train_ds = GDatasetV4(
        train_path_like,
        dp_rank=dp_rank,
        dp_size=dp_size,
        gbs=gbs,
        shuffling_buffer_size=args.px_shuffle_buffer_size,
        consumed=consumed,
        feats=feats,
        seed=42,
        smart_padding_compare_func=smart_padding_compare_func,
        smart_padding_buffer_size=smart_padding_buffer_size,
        mbs=args.micro_batch_size,
    )
    train_map_fn = Glm4vDatasetMap(
        use_for_hf,
        use_grpo,
        tokenizer,
        args.seq_length,
        processor=processor,
        mask_history=mask_history,
    )
    train_ds.map(train_map_fn)
    train_ds.set_epoch(0)

    eval_ds = None
    if eval_path_like is not None:
        eval_gbs = gbs
        if args.use_grpo:
            eval_gbs = args.ppo_eval_rollout_global_batch_size
        eval_ds = GDatasetV4(
            eval_path_like,
            dp_rank=dp_rank,
            dp_size=dp_size,
            gbs=eval_gbs,
            shuffling_buffer_size=args.px_shuffle_buffer_size,
            consumed=0,
            feats=feats,
            seed=42,
        )
        eval_map_fn = Glm4vDatasetMap(
            use_for_hf,
            use_grpo,
            tokenizer,
            args.seq_length,
            processor=processor,
            mask_history=mask_history,
        )
        eval_ds.map(eval_map_fn)
        eval_ds.set_epoch(0)
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
    feats=None,
):
    if isinstance(tokenizer, _HuggingFaceTokenizer):
        tokenizer.__class__ = Glm4vTokenizer
        tokenizer.init_glm_4v()
    else:
        tokenizer = Glm4vTokenizer(tokenizer)

    processor = load_processor(args.processor_path)
    if feats is None:
        feats = {
            "images":
                PilImageListFeat(
                    lmdb=True,
                    return_src_data=True,
                    convert_to_rgb=True,
                    new_name="__images_feat__",
                ),
        }
    train_ds, eval_ds, test_ds = build_train_valid_test_datasets(
        args,
        tokenizer,
        processor,
        rank,
        dp_rank,
        dp_size,
        use_for_hf=use_for_hf,
        is_dpo=is_dpo,
        feats=feats,
    )

    hw_factor = args.context_parallel_size
    if args.sequence_parallel:
        hw_factor *= args.tensor_model_parallel_size
    # grpo数据先不pad
    if args.use_grpo:
        hw_factor = 1
    vision_config = {}
    vision_config["spatial_merge_size"] = processor.image_processor.merge_size
    vision_config["image_token_id"] = tokenizer.image_token_id
    vision_config["video_token_id"] = tokenizer.video_token_id
    vision_config["video_start_token_id"] = tokenizer.video_start_token_id
    vision_config["video_end_token_id"] = tokenizer.video_end_token_id
    vision_config["pad_token_id"] = tokenizer.pad_token_id
    vision_config = SimpleNamespace(**vision_config)
    collate_func = DataCollatorForGlm4v(
        vision_config,
        hw_factor=hw_factor,
        tokenizer=tokenizer,
        is_dpo=is_dpo,
        use_grpo=args.use_grpo,
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
        eval_batch_size = batch_size
        if args.use_grpo:
            eval_batch_size = args.ppo_eval_rollout_micro_batch_size
        eval_dataloader = torch.utils.data.DataLoader(
            eval_ds,
            batch_size=eval_batch_size,
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
    return (
        get_iterator(train_dataloader),
        get_iterator(eval_dataloader),
        get_iterator(test_dataloader),
    )
