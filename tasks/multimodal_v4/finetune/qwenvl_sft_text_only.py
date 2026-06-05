"""
这个文件引用了 QwenVlDatasetMap，它的输入格式要求如下：

sft:
```json
{
    "conversations": [
        {
            "role": "user",
            "content": [{"type": "text", "text": "挂在交通灯杆上的是什么？"}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "一个绿色的街牌挂在交通灯杆上。"}]
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
      "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
      "role": "user",
      "content": [{"type": "text", "text": "Put the captcha of the image within \\boxed{}"}]
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


本文件做了一件事：
1. 将用户输入的 dataset 转成 QwenVlDatasetMap 要求输入的格式
"""

import glob
import os
from typing import Any, Dict

import torch
from datasets import load_dataset
from PIL import PngImagePlugin
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoConfig
from transformers.models.auto.processing_auto import AutoProcessor

from megatron_datasets.qwenvl_dataset_map import (
    QwenVlDatasetMap,
    TrainerV4DataCollatorForQwenVl,
    UserQwen2VLImageProcessorFast,
    UserQwen3VLVideoProcessor,
)

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.utils import log


class QwenVLSimpleTextOnlyDataset(Dataset):
    '''
    这个类用于读取 math_rl_v4 里面的 finetune 的文本数据，finetune 用
    '''
    def __init__(
        self,
        config: RlConfig,
        tokenizer=None,
        processor=None,
        json_pattern="*.jsonl",
        split="train"
    ):
        if PngImagePlugin.MAX_TEXT_CHUNK < 100 * 1024 * 1024:
            log(
                f"Warning: PngImagePlugin.MAX_TEXT_CHUNK changed from {PngImagePlugin.MAX_TEXT_CHUNK} to 1024 * 1024 * 1024"
            )
            PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # 100MB

        self.config = config
        self.system_prompt = config.data.system_prompt
        self.data_dir = config.data.data_pathes[0]

        json_files = glob.glob(os.path.join(self.data_dir, json_pattern))
        tmp_dataset = load_dataset('json', data_files=json_files, split=split)

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
            no_shift_label=True,
            config=self.config,
        )

        # dataset_v4 要求每条样本都合法，这里先做一个过滤，如果比较耗时，可以考虑离线过滤
        def is_valid(example):
            sample = self.format_to_conversation(example)
            data_dict = self.map_func.process(sample)
            return isinstance(data_dict, dict)

        len_src_ds = len(tmp_dataset)
        self.train_dataset = tmp_dataset
        print(f"filter out num: {len_src_ds - len(self.train_dataset)}")

    def format_to_conversation(self, example):
        # 下面处理主要针对 math 的数据，这个数据暂时只有单轮的，不过多轮也一样，拼进去就行
        if "problem" in example.keys():
            assert "problem" in example.keys()
            assert "solution" in example.keys()
            q_str = example['problem']
            a_str = example['solution']
        elif "question" in example.keys():
            assert "question" in example.keys()
            assert "target" in example.keys()
            q_str = example['question']
            a_str = example['target']

        messages = []
        if self.system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": self.system_prompt
                    }],
                }
            )

        messages.append({
            "role": "user",
            "content": [{
                "type": "text",
                "text": q_str
            }],
        })
        messages.append({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": a_str
            }],
        })

        images = []
        images_feat = []
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


def get_dataset_and_dataloader(config: RlConfig = None, tokenizer=None, dp_rank=0, dp_size=1):
    image_processor = UserQwen2VLImageProcessorFast.from_pretrained(config.policy.hf_tokenizer_path)
    video_process = UserQwen3VLVideoProcessor.from_pretrained(config.policy.hf_tokenizer_path)
    processor = AutoProcessor.from_pretrained(
        config.policy.hf_tokenizer_path,
        image_processor=image_processor,
        video_processor=video_process
    )

    train_dataset = QwenVLSimpleTextOnlyDataset(config, tokenizer, processor)
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
        is_dpo=False,
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
    print(f"{data.keys()=}")
    print(f"{data=}")

    data_dl = next(iter(train_dataloader))
    print(f"{data_dl.keys()=}")
    print(f"{data_dl=}")
