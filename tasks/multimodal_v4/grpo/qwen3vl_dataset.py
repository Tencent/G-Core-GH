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


本文件做了一件事：
1. 将用户输入的 dataset 转成 QwenVlDatasetMap 要求输入的格式
"""

import json
from typing import Any, Dict

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
    TrainerV4DataCollatorForQwenVlGRPO,
)


class Qwen3VLSimpleDataset(Dataset):
    def __init__(self, config: RlConfig, tokenizer=None, processor=None):
        self.config = config

        self.hf_config = AutoConfig.from_pretrained(
            self.config.policy.hf_tokenizer_path, trust_remote_code=True
        )
        self.map_func = QwenVlDatasetMap(
            self.hf_config,
            min_pixels=None,
            max_pixels=None,
            use_grpo=True,
            tokenizer=tokenizer,
            max_seq_len=self.config.training.seq_length,
            grpo_resp_length=self.config.sampler.infer_engine_configs[0].generate_max_tokens,
            processor=processor,
            mask_history=False,
            moe_pad_with_random_token=False,
            no_shift_label=True,
            config=self.config,
        )

        dataset = load_dataset(self.config.data.data_pathes[0])
        tmp_dataset = dataset['train'].shuffle(seed=42)

        # 每条样本都合法，这里先做一个过滤，如果比较耗时，可以考虑离线过滤
        def is_valid(example):
            sample = self.convert_sample(example)
            data_dict = self.map_func.process(sample)
            return isinstance(data_dict, dict)

        len_src_ds = len(tmp_dataset)
        self.train_dataset = tmp_dataset.filter(is_valid, num_proc=1)
        #TODO(guanyouhe):
        # datasets.filter(num_proc=4) 会使用 Python 的 multiprocessing 来 fork 子进程。在 Ray actor 内部 fork 进程是不安全的，容易导致死锁或资源冲突，Ray 可能因此杀掉该 actor
        # 先设置成 1
        print(f"filter out num: {len_src_ds - len(self.train_dataset)}")

    def convert_sample(self, sample):
        instruction_following = (
            r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
            r"The reasoning process MUST BE enclosed within <reason> </reason> tags. The final answer MUST BE put in \boxed{}."
        )
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

        images = [dict(image_path=f"{i}") for i in range(len(sample["images"]))]
        return dict(
            conversations=conversation,
            label=label,
            images=images,
            __images_feat__=sample["images"],
        )

    def __len__(self) -> int:
        return len(self.train_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ele_data = self.train_dataset[idx]
        sample = self.convert_sample(ele_data)

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

    train_dataset = Qwen3VLSimpleDataset(config, tokenizer, processor)
    sampler = DistributedSampler(
        train_dataset,
        rank=dp_rank,
        num_replicas=dp_size,
        shuffle=True,
        seed=config.data.sampler_seed
    )
    collate_func = TrainerV4DataCollatorForQwenVlGRPO(
        hw_factor=1,
        model_arch=config.policy.model_arch,
        tokenizer=tokenizer,
        hf_config_path=config.policy.hf_tokenizer_path,
        dp_rank=dp_rank,
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
    # data = train_dataset[0]
    # print(f"{data.keys()=}")
    # print(f"{data=}")

    data_dl = next(iter(train_dataloader))
    print(f"{data_dl.keys()=}")
    print(f"{data_dl=}")
