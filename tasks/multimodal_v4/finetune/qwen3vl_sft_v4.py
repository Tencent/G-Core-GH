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
5. __audios_feat__ 是 shape 为 () 的 torch.Tensor
6. sft的数据中：conversations[-1]默认为label
7. grpo的数据中：label不是必须的字段；另外grpo的conversations不能在assistant，
如果最后一个为assistant直接删除


本文件做了只做了一件事：
1. 将用户输入的 dataset 转成 QwenVlDatasetMap 要求输入的格式
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig
from transformers.models.auto.processing_auto import AutoProcessor

from gdataset import GDatasetV4
from gdataset.feat import PilImageListFeat
from megatron_datasets.qwenvl_dataset_map import (
    QwenVlDatasetMap,
    TrainerV4DataCollatorForQwenVl,
    UserQwen2VLImageProcessorFast,
    UserQwen3OmniFeatureExtractor,
    UserQwen3VLVideoProcessor,
    sort_by_prompt_len,
)

from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.training_backend.megatron_backend.checkpoint import (
    get_latest_checkpoint_folder,
)
from gpatch_v4.utils import log


def _get_dataset_and_dataloader(
    config: FinetuneConfig = None, tokenizer=None, dp_rank=0, dp_size=1, data_path=None
):
    image_processor = UserQwen2VLImageProcessorFast.from_pretrained(config.policy.hf_tokenizer_path)
    video_process = UserQwen3VLVideoProcessor.from_pretrained(config.policy.hf_tokenizer_path)
    extra_kwargs = {}
    if config.policy.model_arch in ["qwen3_omni", "qwen3_omni_moe"]:
        extra_kwargs['feature_extractor'] = UserQwen3OmniFeatureExtractor.from_pretrained(
            config.policy.hf_tokenizer_path
        )
    processor = AutoProcessor.from_pretrained(
        config.policy.hf_tokenizer_path,
        image_processor=image_processor,
        video_processor=video_process,
        **extra_kwargs,
    )

    hf_config = AutoConfig.from_pretrained(config.policy.hf_tokenizer_path)
    map_func = QwenVlDatasetMap(
        hf_config,
        min_pixels=None,
        max_pixels=None,
        use_grpo=False,
        tokenizer=tokenizer,
        max_seq_len=config.training.seq_length,
        processor=processor,
        mask_history=config.data.mask_history,
        moe_pad_with_random_token=False,
        no_shift_label=True,
        config=config,
    )
    feats = {
        'images':
            PilImageListFeat(
                lmdb=True,
                return_src_data=True,
                convert_to_rgb=True,
                new_name="__images_feat__",
            ),
    }

    assert "smart_padding_buffer_size" in config.task, f'{config.task=}'
    assert "shuffle_buffer_size" in config.task, f'{config.task=}'
    smart_padding_compare_func = sort_by_prompt_len
    smart_padding_buffer_size = config.task["smart_padding_buffer_size"]
    shuffle_buffer_size = config.task["shuffle_buffer_size"]
    consumed = 0
    load_latest_step = get_latest_checkpoint_folder(config.checkpoint.load_ckpt_path)
    if load_latest_step is not None:
        consumed = load_latest_step * config.training.train_gbs
    dataset = GDatasetV4(
        data_path,
        dp_rank=dp_rank,
        dp_size=dp_size,
        gbs=config.training.train_gbs,
        shuffling_buffer_size=shuffle_buffer_size,
        consumed=consumed,
        feats=feats,
        seed=42,
        smart_padding_compare_func=smart_padding_compare_func,
        smart_padding_buffer_size=smart_padding_buffer_size,
        mbs=config.training.train_mbs,
    )
    total_len_ds = len(dataset) * dp_size
    epoch = consumed // total_len_ds
    dataset.map(map_func)
    dataset.set_epoch(epoch)
    sampler = None
    collate_func = TrainerV4DataCollatorForQwenVl(
        hw_factor=1,
        model_arch=config.policy.model_arch,
        tokenizer=tokenizer,
        is_dpo=False,
        use_grpo=False,
        hf_config_path=config.policy.hf_tokenizer_path,
    )

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=collate_func,
        pin_memory=config.data.dataloader_pin_memory,
        batch_size=config.training.train_mbs,
        num_workers=config.data.dataloader_num_workers,
        prefetch_factor=config.data.dataloader_prefetch_factor,
        drop_last=True,
    )
    return dataset, sampler, dataloader


def get_dataset_and_dataloader(config: FinetuneConfig = None, tokenizer=None, dp_rank=0, dp_size=1):
    train_path = config.data.data_pathes[0]
    eval_path = config.data.data_pathes[1]

    train_dataset, sampler, train_dataloader = _get_dataset_and_dataloader(
        config, tokenizer, dp_rank, dp_size, train_path
    )
    eval_dataset, _, eval_dataloader = _get_dataset_and_dataloader(
        config, tokenizer, dp_rank, dp_size, eval_path
    )

    return {
        'train_dataset': train_dataset,
        'train_sampler': sampler,
        'train_dataloader': train_dataloader,
        'eval_dataset': eval_dataset,
        'eval_dataloader': eval_dataloader,
    }


def verify_dataloader_func(train_dataset, train_sampler, train_dataloader):
    print(f"{len(train_dataset)=}")
    data = next(iter(train_dataset))
    print(f"{data.keys()=}")
    print(f"{data=}")

    data_dl = next(iter(train_dataloader))
    print(f"{data_dl.keys()=}")
    print(f"{data_dl=}")
