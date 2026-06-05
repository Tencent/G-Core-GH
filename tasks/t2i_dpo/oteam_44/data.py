"""
adapted code from llamafactory and ms-swift to suuport llamafactory dataset
"""
import contextlib
import io
import os
import random
from functools import partial
from types import SimpleNamespace
from typing import Any, Literal, Mapping, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from megatron.core import mpu

from gpatch_v4.utils import set_seed, to_device

DATASET_NAME_MAPPING = {
    "yuvalkirstain/pickapic_v1": ("jpg_0", "jpg_1", "label_0", "caption"),
    "yuvalkirstain/pickapic_v2": ("jpg_0", "jpg_1", "label_0", "caption"),
    "xzuyn/pickapic_v2_only_some": ("jpg_0", "jpg_1", "label_0", "caption"),
}


# adapted from transformers
@contextlib.contextmanager
def main_process_first(args, local=True, desc="work"):
    """
    A context manager for torch distributed environment where on needs to do something on the main process, while
    blocking replicas, and when it's finished releasing the replicas.

    One such use is for `datasets`'s `map` feature which to be efficient should be run once on the main process,
    which upon completion saves a cached version of results and which then automatically gets loaded by the
    replicas.

    Args:
        local (`bool`, *optional*, defaults to `True`):
            if `True` first means process of rank 0 of each node if `False` first means process of rank 0 of node
            rank 0 In multi-node environment with a shared filesystem you most likely will want to use
            `local=False` so that only the main process of the first node will do the processing. If however, the
            filesystem is not shared, then the main process of each node will need to do the processing, which is
            the default behavior.
        desc (`str`, *optional*, defaults to `"work"`):
            a work description to be used in debug logs

    """
    distributed = (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)

    if distributed:
        main_process_desc = "main local process" if local else "main process"
        # local rank
        is_main_process = args.local_rank == 0 if local else dist.get_rank() == 0
        try:
            if not is_main_process:
                # tell all replicas to wait
                print(
                    f"{dist.get_rank()}-{args.local_rank}: waiting for the {main_process_desc} to perform {desc}"
                )
                dist.barrier()
            yield
        finally:
            if is_main_process:
                # the wait is over
                print(
                    f"{dist.get_rank()}-{args.local_rank} completed {desc}, releasing all replicas"
                )
                dist.barrier()
    else:
        yield


class BatchSamplerShard:
    def __init__(
        self,
        total_samples: int,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        data_seed: Optional[int],
    ):
        self.total_samples = total_samples // self.world_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.base_seed = data_seed or 0
        self.curr_seed = self.base_seed

    @property
    def rank(self):
        return dist.get_rank(self.group) if dist.is_initialized() else 0

    @property
    def world_size(self):
        return dist.get_world_size(self.group) if dist.is_initialized() else 1

    @property
    def group(self):
        # 为了对齐 llamafactory 的测试
        if mpu._DATA_PARALLEL_GROUP is not None:
            assert False
            return mpu.get_data_parallel_group()
        return None

    def __iter__(self):
        start_idx = self.rank * self.total_samples
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.curr_seed)
            total_idx = torch.randperm(self.total_samples * self.world_size,
                                       generator=generator).tolist()
            total_idx = total_idx[start_idx:start_idx + self.total_samples]
        else:
            total_idx = list(range(start_idx, start_idx + self.total_samples))

        batch = []
        # Last batch if not complete will be dropped.
        for idx in total_idx:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if not self.drop_last and len(batch) > 0:
            yield batch
        return

    def set_epoch(self, epoch: int):
        self.curr_seed = self.base_seed + epoch

    def __len__(self) -> int:
        if self.drop_last:
            return self.total_samples // self.batch_size
        else:
            return (self.total_samples + self.batch_size - 1) // self.batch_size


class DataLoaderShard(DataLoader):
    def __init__(self, dataset, device=None, **dataloader_params):
        self.device = device
        super().__init__(dataset, **dataloader_params)

    def set_epoch(self, epoch: int):
        if self.batch_sampler is not None and hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)
        elif self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        for item in super().__iter__():
            if self.device:
                item = to_device(item, self.device)
            yield item


class DataLoaderDispatcher:
    def __init__(self, base_dataloader, device=None, skip_batches: int = 0):
        self.base_dataloader = base_dataloader
        self.device = device
        self.skip_batches = skip_batches

    @property
    def rank(self):
        return dist.get_rank(self.group) if dist.is_initialized() else 0

    @property
    def world_size(self):
        return dist.get_world_size(self.group) if dist.is_initialized() else 1

    @property
    def group(self):
        return mpu.get_data_parallel_group()

    def _scatter_object_list(self, inputs):
        if not dist.is_initialized():
            return inputs[0]
        outputs = [None]
        global_src_rank = dist.get_global_rank(self.group, 0)
        dist.scatter_object_list(outputs, inputs, global_src_rank, group=self.group)
        return outputs[0]

    def _skip_batches(self, base_iter):
        if self.rank == 0 and self.skip_batches > 0:
            for _ in tqdm(range(self.skip_batches), dynamic_ncols=True, desc="Skip Batches: "):
                [next(base_iter) for _ in range(self.world_size)]

    def __iter__(self):
        base_iter = iter(self.base_dataloader)
        self._skip_batches(base_iter)
        while True:
            if self.rank == 0:
                try:
                    data = [next(base_iter) for _ in range(self.world_size)]
                except StopIteration:
                    data = [None] * self.world_size
                data = self._scatter_object_list(data)
            else:
                data = self._scatter_object_list(None)
            if data is None:
                break
            if self.device:
                data = to_device(data, self.device)
            yield data


def seed_worker(worker_id: int, num_workers: int, rank: int):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    init_seed = torch.initial_seed() % 2**32
    worker_seed = num_workers * rank + init_seed
    set_seed(worker_seed)


def build_dataloader_from_dataset(args, dataset, data_collator, batch_size, dp_rank):
    dataloader_params = {
        "collate_fn": data_collator,
        "num_workers": args.data.dataloader_num_workers,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 1,
    }
    batch_sampler_params = {
        "drop_last": True,
        "shuffle": True,
        "data_seed": args.training.seed,
    }

    if hasattr(dataset, "__len__"):
        batch_sampler = BatchSamplerShard(
            len(dataset), batch_size=batch_size, **batch_sampler_params
        )
        dataloader_params["worker_init_fn"] = partial(
            seed_worker,
            num_workers=args.data.dataloader_num_workers,
            rank=dp_rank,
        )
        dataloader_params["batch_sampler"] = batch_sampler
        dataloader = DataLoaderShard(
            dataset, device=torch.cuda.current_device(), **dataloader_params
        )
    else:
        # IterableDataset
        if dist.is_initialized() and dataloader_params["prefetch_factor"]:
            dataloader_params["prefetch_factor"] = (
                dataloader_params["prefetch_factor"] * mpu.get_data_parallel_world_size()
            )
        dataloader = DataLoader(dataset, batch_size=batch_size, **dataloader_params)
        dataloader = DataLoaderDispatcher(dataloader, torch.cuda.current_device(), skip_batches=0)
    return dataloader


def build_dataloader(args, tokenizer, image_processor=None):
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    assert args.data.dataset_name is not None
    if args.data.train_data_dir is not None:
        data_files = {}
        data_files[args.data.split] = os.path.join(args.data.train_data_dir, f"{args.data.split}*")
        dataset = load_dataset(
            "parquet",
            data_files=data_files,
            cache_dir=args.data.cache_dir,
        )
    else:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.data.dataset_name,
            args.data.dataset_config_name,
            cache_dir=args.data.cache_dir,
            data_dir=args.data.train_data_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset[args.data.split].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.data.dataset_name, None)
    if 'pickapic' in args.data.dataset_name or (args.training.train_method == 'dpo'):
        pass
    elif args.data.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.data.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.data.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.data.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.data.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.data.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.data.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        if tokenizer is not None:
            inputs = tokenizer(
                captions,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return inputs.input_ids
        else:
            return captions

    # Preprocessing the datasets.
    if image_processor is None:
        train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    args.training.resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomCrop(args.training.resolution)
                if args.data.random_crop else transforms.CenterCrop(args.training.resolution),
                transforms.Lambda(lambda x: x)
                if args.data.no_hflip else transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    else:
        train_transforms = None

    ##### START BIG OLD DATASET BLOCK #####

    #### START PREPROCESSING/COLLATION ####
    if args.training.train_method == 'dpo':
        print("Ignoring image_column variable, reading from jpg_0 and jpg_1")

        def preprocess_train(examples):
            all_pixel_values = []
            for col_name in ['jpg_0', 'jpg_1']:
                images = [
                    Image.open(io.BytesIO(im_bytes)).convert("RGB")
                    for im_bytes in examples[col_name]
                ]
                if image_processor is None:
                    pixel_values = [train_transforms(image) for image in images]
                else:
                    pixel_values = [
                        image_processor.preprocess(
                            image, args.training.resolution, args.training.resolution
                        ).squeeze(0) for image in images
                    ]
                all_pixel_values.append(pixel_values)
            # Double on channel dim, jpg_y then jpg_w
            im_tup_iterator = zip(*all_pixel_values)
            combined_pixel_values = []
            for im_tup, label_0 in zip(im_tup_iterator, examples['label_0']):
                if label_0 == 0 and (
                    not args.training.choice_model
                ):  # don't want to flip things if using choice_model for AI feedback
                    im_tup = im_tup[::-1]
                combined_im = torch.cat(im_tup, dim=0)  # no batch dim
                combined_pixel_values.append(combined_im)
            examples["pixel_values"] = combined_pixel_values
            # SDXL takes raw prompts
            if tokenizer is not None:
                examples["input_ids"] = tokenize_captions(examples)
                examples["caption"] = tokenize_captions(examples)
            return examples

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            return_d = {"pixel_values": pixel_values}
            # SDXL takes raw prompts
            if tokenizer is None:
                return_d["caption"] = [example["caption"] for example in examples]
            else:
                return_d["input_ids"] = torch.stack([example["input_ids"] for example in examples])

            if args.training.choice_model:
                # If using AIF then deliver image data for choice model to determine if should flip pixel values
                for k in ['jpg_0', 'jpg_1']:
                    return_d[k] = [
                        Image.open(io.BytesIO(example[k])).convert("RGB") for example in examples
                    ]
                return_d["caption"] = [example["caption"] for example in examples]
            return return_d

        if args.training.choice_model:
            pass
            """
            # TODO: Fancy way of doing this?
            if args.choice_model == 'hps':
                from utils.hps_utils import Selector
            elif args.choice_model == 'clip':
                from utils.clip_utils import Selector
            elif args.choice_model == 'pickscore':
                from utils.pickscore_utils import Selector
            elif args.choice_model == 'aes':
                from utils.aes_utils import Selector
            selector = Selector('cpu' if args.sdxl else torch.cuda.current_device())

            def do_flip(jpg0, jpg1, prompt):
                scores = selector.score([jpg0, jpg1], prompt)
                return scores[1] > scores[0]

            def choice_model_says_flip(batch):
                assert len(
                    batch['caption']
                ) == 1  # Can switch to iteration but not needed for nwo
                return do_flip(batch['jpg_0'][0], batch['jpg_1'][0], batch['caption'][0])
            """

    elif args.training.train_method == 'sft':

        def preprocess_train(examples):
            if 'pickapic' in args.dataset_name:
                images = []
                # Probably cleaner way to do this iteration
                for im_0_bytes, im_1_bytes, label_0 in zip(
                    examples['jpg_0'], examples['jpg_1'], examples['label_0']
                ):
                    assert label_0 in (0, 1)
                    im_bytes = im_0_bytes if label_0 == 1 else im_1_bytes
                    images.append(Image.open(io.BytesIO(im_bytes)).convert("RGB"))
            else:
                images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = tokenize_captions(examples)
            return examples

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            return_d = {"pixel_values": pixel_values}
            return_d["input_ids"] = torch.stack([example["input_ids"] for example in examples])
            return return_d

    #### END PREPROCESSING/COLLATION ####

    ### DATASET #####
    with main_process_first(args, local=False, desc="prepara_dataset"):
        if 'pickapic' in args.data.dataset_name:
            # eliminate no-decisions (0.5-0.5 labels)
            orig_len = dataset[args.data.split].num_rows
            not_split_idx = [
                i for i, label_0 in enumerate(dataset[args.data.split]['label_0'])
                if label_0 in (0, 1)
            ]
            dataset[args.data.split] = dataset[args.data.split].select(not_split_idx)
            new_len = dataset[args.data.split].num_rows
            print(f"Eliminated {orig_len - new_len}/{orig_len} split decisions for Pick-a-pic")

            # Below if if want to train on just the Dreamlike vs dreamlike pairs
            if args.data.dreamlike_pairs_only:
                orig_len = dataset[args.data.split].num_rows
                dream_like_idx = [
                    i for i, (m0, m1) in enumerate(
                        zip(
                            dataset[args.data.split]['model_0'], dataset[args.data.split]['model_1']
                        )
                    ) if (('dream' in m0) and ('dream' in m1))
                ]
                dataset[args.data.split] = dataset[args.data.split].select(dream_like_idx)
                new_len = dataset[args.data.split].num_rows
                print(
                    f"Eliminated {orig_len - new_len}/{orig_len} non-dreamlike gens for Pick-a-pic"
                )

        if args.training.max_train_samples is not None:
            dataset[args.data.split] = dataset[args.data.split].shuffle(seed=args.seed).select(
                range(args.training.max_train_samples)
            )
        # Set the training transforms
        train_dataset = dataset[args.data.split].with_transform(preprocess_train)

    # DataLoaders creation:
    train_dataloader = build_dataloader_from_dataset(
        args,
        train_dataset,
        data_collator=collate_fn,
        batch_size=args.training.train_batch_size,
        dp_rank=dist.get_rank()
    )
    return train_dataloader
