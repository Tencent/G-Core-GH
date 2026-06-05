"""
这个文件引用了 QwenVlDatasetMap，它的输入格式要求如下：

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
            "image_path": "0",
        }
    ],
    "__images_feat__": [
        <PIL.Image.Image image mode=RGB size=xxx>,
        <PIL.Image.Image image mode=RGB size=xxx>
    ],

}
```
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
      "image_path": "0"
    }
  ],
  "__images_feat__": [
      <PIL.Image.Image image mode=RGB size=xxx>,
      <PIL.Image.Image image mode=RGB size=xxx>
  ]
}
```
要求：
1. 可以没有图片/视频
2. images也可以只写image_path 填个数字就好，这里只是为了兼容
3. __images_feat__ 为 PIL.Image
4. __videos_feat__ 是 shape 为 (T, C, H, W) 的 torch.Tensor
5. sft的数据中：conversations[-1]默认为label
6. grpo的数据中：label不是必须的字段；另外grpo的conversations不能在assistant，
如果最后一个为assistant直接删除


本文件做了两件事：
1. 将用户输入的 dataset 转成 QwenVlDatasetMap 要求输入的格式
2. 会将 DataCollatorForQwenVl 输出的数据转成按要求输入的数据
"""

import json
import traceback
import zipfile
from io import BytesIO
from PIL import Image
from typing import Any, Dict, Sequence
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from transformers.models.auto.processing_auto import AutoProcessor
from transformers import AutoConfig
from datasets import load_dataset

from gpatch_v4.configs.config import RlConfig
from megatron_datasets.qwenvl_dataset_map import (
    QwenVlDatasetMap,
    UserQwen2VLImageProcessorFast,
    UserQwen3VLVideoProcessor,
    DataCollatorForQwenVl,
)


class Qwen3VLSimpleDataset(Dataset):

    def __init__(self, config: RlConfig, tokenizer=None, processor=None, split="train"):
        self.config = config
        self.system_prompt = config.data.system_prompt

        assert "image_zip_path" in config.task, "image_zip_path is required"
        assert "image_dir_in_zip" in config.task, "image_dir_in_zip is required"
        assert "data_files_format" in config.task, "data_files_format is required"

        self.train_dataset = load_dataset(config.task["data_files_format"], data_files=self.config.data.data_pathes, split=split)

        self.hf_config = AutoConfig.from_pretrained(self.config.policy.hf_tokenizer_path)
        self.map_func = QwenVlDatasetMap(
            self.hf_config,
            min_pixels=None,
            max_pixels=None,
            use_grpo=False,
            tokenizer=tokenizer,
            max_seq_len=self.config.training.seq_length,
            processor=processor,
            mask_history=self.config.data.mask_history,
            moe_pad_with_random_token=False,
        )

        # 直接从 zip 读取图片
        assert config.task["image_zip_path"] is not None and len(config.task["image_zip_path"]) == 1, "暂时只支持 1 个 zip 文件"

        self.zip_path = config.task["image_zip_path"][0]
        print(f"Loading images from zip: {self.zip_path}")
        self.image_dir = config.task["image_dir_in_zip"]
        self._worker_id = None
        self._zip_file_obj = None

    def __del__(self):
        try:
            if self._zip_file_obj is not None:
                self._zip_file_obj.close()
        except Exception as e:
            print(f"Error closing zip file: {e}")

    @property
    def zip_file_obj(self):
        worker_info = get_worker_info()
        current_worker_id = worker_info.id if worker_info else -1

        if self._zip_file_obj is None or self._worker_id != current_worker_id:
            if self._zip_file_obj is not None:
                try:
                    self._zip_file_obj.close()
                except Exception as e:
                    pass
            self._zip_file_obj = zipfile.ZipFile(self.zip_path, 'r')
            self._worker_id = current_worker_id
        return self._zip_file_obj

    def _load_image_from_zip(self, image_filename: str) -> Image.Image:
        image_path = f"{self.image_dir}/{image_filename}"
        try:
            image_data = self.zip_file_obj.read(image_path)
            img = Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as e:
            traceback.print_exc()
            print(f"Error loading image {image_path}: {e}")
            raise e
        return img

    def format_to_conversation(self, example):
        # 下面处理主要针对 pmc demo 数据，业务具体数据需要根据实际情况调整
        question = example["Question"]
        answer = example["Answer"]
        choice_str = "".join([example[f'Choice {x}'] for x in ("A", "B", "C", "D")])
        
        user_content = f"You are a medical expert, please observe the following picture and answer this question accurately: {question} Choose from the following options and response with only the letter option: {choice_str} A letter of A/B/C/D is all you need to return and absolutely nothing else. <image>"

        messages = []
        if self.system_prompt is not None:
            messages.append(dict(role="system", content=self.system_prompt))
        messages.append(dict(role="user", content=user_content))
        messages.append(dict(role="assistant", content=answer))

        images = [dict(image_path=example['Figure_path'])]
        images_feat = [self._load_image_from_zip(img_path['image_path']) for img_path in images]

        return dict(
            conversations=messages,
            images=images,
            __images_feat__=images_feat,
        )

    def __len__(self) -> int:
        return len(self.train_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.train_dataset[idx]
        sample = self.format_to_conversation(example)

        data_dict = self.map_func.process(sample)
        assert isinstance(data_dict, dict), f"Should make sure the sample is valid: {data_dict}"
        return data_dict


def collate_func_rbs(collate_func_cat, instances: Sequence[Dict]):
    bs = len(instances)
    assert bs == 1
    k_map = {
        "input_ids": "tokens",
        "prompt_len": "prompt_lengths",
        "pixel_values": "vision_data",
        "image_grid_thw": "vision_grid_thw",
    }

    res_cat = collate_func_cat(instances)
    res = {}

    longest_len = res_cat["sequence_lengths"].max().item()
    for k, v in res_cat.items():
        new_k = k
        if k in k_map:
            new_k = k_map[k]

        # 可以不做的，但都打印出来太难看了
        if k == "input_ids":
            v = v[..., :longest_len]

        if torch.is_tensor(v):
            if k == "position_ids":
                res[new_k] = list(torch.split(v, 1, dim=1))
            elif k in ["pixel_values", "image_grid_thw"]:
                res[new_k] = [v]
            elif k in ["image_input_mask"]:
                res[new_k] = list(torch.split(v, 1, dim=0))
            else:
                res[new_k] = list(torch.unbind(v, dim=0))
        else:
            assert isinstance(v, list)
            res[new_k] = v
        assert len(res[new_k]) == bs, f"error: {k=} {bs=} {len(res[k])=}"
    return res


def get_dataset_and_dataloader(config: RlConfig = None, tokenizer=None, dp_rank=0, dp_size=1):
    image_processor = UserQwen2VLImageProcessorFast.from_pretrained(config.policy.hf_tokenizer_path)
    video_process = UserQwen3VLVideoProcessor.from_pretrained(config.policy.hf_tokenizer_path)
    processor = AutoProcessor.from_pretrained(config.policy.hf_tokenizer_path,
                                              image_processor=image_processor,
                                              video_processor=video_process)

    train_dataset = Qwen3VLSimpleDataset(config, tokenizer, processor)
    sampler = DistributedSampler(train_dataset,
                                 rank=dp_rank,
                                 num_replicas=dp_size,
                                 shuffle=True,
                                 seed=config.data.sampler_seed)
    collate_func = DataCollatorForQwenVl(
        hw_factor=1,
        model_arch=config.policy.model_arch,
        tokenizer=tokenizer,
        is_dpo=False,
        use_grpo=False,
        cp_size=1, # 这里不做直接的切分，因为 offdistillation 的输入要转成 rbs
        hf_config_path=config.policy.hf_tokenizer_path,
    )

    dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=partial(collate_func_rbs, collate_func),
        pin_memory=config.data.dataloader_pin_memory,
        batch_size=config.training.train_mbs,
        num_workers=config.data.dataloader_num_workers,
        prefetch_factor=config.data.dataloader_prefetch_factor,
        drop_last=True,
    )
    return {
        'train_dataset': train_dataset,
        'train_sampler': sampler,
        'train_dataloader': dataloader,
    }


def verify_dataloader_func(train_dataset, train_sampler, train_dataloader):
    print(f"{len(train_dataset)=}")
    data = train_dataset[0]
    print(f"{data.keys()=}")
    print(f"{data=}")

    data_dl = next(iter(train_dataloader))
    print(f"{data_dl.keys()=}")
    print(f"{data_dl=}")
