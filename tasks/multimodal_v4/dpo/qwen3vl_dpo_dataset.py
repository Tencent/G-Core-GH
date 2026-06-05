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

本文件为 DPO 数据集，所以一个样本会生成正反两条样本

本文件做了两件事：
1. 将用户输入的 dataset 转成 QwenVlDatasetMap 要求输入的格式
2. 会将 DataCollatorForQwenVl 输出的数据转成按要求输入的数据
"""

import json
from typing import Any, Dict, Sequence, Tuple
from functools import partial
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers.models.auto.processing_auto import AutoProcessor
from transformers import AutoConfig
from datasets import load_dataset

from gpatch_v4.configs.config import RlConfig
from megatron_datasets.qwenvl_dataset_map import (
    QwenVlDatasetMap,
    UserQwen2VLImageProcessorFast,
    UserQwen3VLVideoProcessor,
    TrainerV4DataCollatorForQwenVl,
)


class Qwen3VLDpoSimpleDataset(Dataset):
    def __init__(self, config: RlConfig, tokenizer=None, processor=None):
        self.config = config
        dataset = load_dataset(self.config.data.data_pathes[0])
        self.train_dataset = dataset['train'].shuffle(seed=42)
        self.hf_config = AutoConfig.from_pretrained(self.config.policy.hf_tokenizer_path)
        self.map_func = QwenVlDatasetMap(
            self.hf_config,
            min_pixels=None,
            max_pixels=None,
            use_grpo=False,
            tokenizer=tokenizer,
            max_seq_len=self.config.training.seq_length,
            processor=processor,
            mask_history=False,
            moe_pad_with_random_token=False,
            no_shift_label=True,
            config=self.config,
        )

    def convert_sample(self, sample):
        def convert_ele(ele):
            if ele['from'] == 'human':
                return dict(role="user", content=ele['value'])
            elif ele['from'] == 'gpt':
                return dict(role="assistant", content=ele['value'])
            else:
                raise NotImplementedError

        def convert_conversation(conversation):
            new_conversation = []
            # add system prompt
            new_conversation.append(dict(role="system", content="You are a helpful assistant."))
            for ele in conversation:
                new_conversation.append(convert_ele(ele))
            return new_conversation

        conversation = sample["conversations"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]
        images = sample["images"]
        conversation = convert_conversation(conversation)
        rejected = convert_ele(rejected)
        chosen = convert_ele(chosen)

        images = [dict(image_path=f"{i}") for i in range(len(sample["images"]))]
        return [
            dict(conversations=conversation + [chosen], images=images),
            dict(conversations=conversation + [rejected], images=images)
        ]

    def __len__(self) -> int:
        return len(self.train_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ele_data = self.train_dataset[idx]
        dpo_samples = self.convert_sample(ele_data)
        assert len(dpo_samples) == 2
        dpo_samples[0]["__images_feat__"] = ele_data["images"]
        dpo_samples[1]["__images_feat__"] = ele_data["images"]

        data_dict0 = self.map_func.process(dpo_samples[0])
        data_dict1 = self.map_func.process(dpo_samples[1])
        assert isinstance(data_dict0, dict), f"Should make sure the sample is valid: {data_dict0}"
        assert isinstance(data_dict1, dict), f"Should make sure the sample is valid: {data_dict1}"
        return (data_dict0, data_dict1)


def get_dataset_and_dataloader(config: RlConfig = None, tokenizer=None, dp_rank=0, dp_size=1):
    image_processor = UserQwen2VLImageProcessorFast.from_pretrained(config.policy.hf_tokenizer_path)
    video_process = UserQwen3VLVideoProcessor.from_pretrained(config.policy.hf_tokenizer_path)
    processor = AutoProcessor.from_pretrained(
        config.policy.hf_tokenizer_path,
        image_processor=image_processor,
        video_processor=video_process
    )

    train_dataset = Qwen3VLDpoSimpleDataset(config, tokenizer, processor)
    sampler = DistributedSampler(
        train_dataset,
        rank=dp_rank,
        num_replicas=dp_size,
        shuffle=True,
        seed=config.data.sampler_seed
    )
    collate_func = TrainerV4DataCollatorForQwenVl(
        hw_factor=1,
        model_arch=config.policy.model_arch,
        tokenizer=tokenizer,
        is_dpo=True,  # 设置这里会将所有的 chosen 样本放到上面，rejected 样本放到下面
        use_grpo=False,
        hf_config_path=config.policy.hf_tokenizer_path,
    )

    dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=collate_func,
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
    # data is a tuple of two dicts
    print(f"{data=}")

    data_dl = next(iter(train_dataloader))
    print(f"{data_dl.keys()=}")
    print(f"{data_dl=}")
