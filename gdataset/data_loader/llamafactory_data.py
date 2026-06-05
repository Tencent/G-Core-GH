"""
adapted code from llamafactory and ms-swift to suuport llamafactory dataset
"""
import contextlib
import random
from functools import partial
from types import SimpleNamespace
from typing import Any, Literal, Mapping, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from llamafactory.data.data_utils import get_dataset_module, merge_dataset, split_dataset
from llamafactory.data.loader import _get_merged_dataset
from llamafactory.extras import logging
from llamafactory.hparams import DataArguments
from torch.utils.data import DataLoader
from tqdm import tqdm

from gdataset.data_loader.processor import LengthFilter, _get_dataset_processor

from megatron.core import mpu

logger = logging.get_logger(__name__)


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
                logger.debug(
                    f"{self.process_index}: waiting for the {main_process_desc} to perform {desc}"
                )
                dist.barrier()
            yield
        finally:
            if is_main_process:
                # the wait is over
                logger.debug(
                    f"{self.process_index}: {main_process_desc} completed {desc}, releasing all replicas"
                )
                dist.barrier()
    else:
        yield


# adapted from transformers


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""Preprocesses the dataset, including format checking and tokenization."""
    if dataset is None:
        return None

    dataset_processor = _get_dataset_processor(
        data_args,
        stage,
        template,
        tokenizer,
        processor,
        do_generate=(training_args.predict_with_generate and is_eval)
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or
            (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        dataset_processor.preprocess_dataset,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            dataset_processor.print_data_example(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError(
                    "Cannot find sufficient samples, consider increasing dataset size."
                )
            else:
                raise RuntimeError(
                    "Cannot find valid samples, check `data/README.md` for the data format."
                )

    return dataset


# llamafactory get_dataset but use local dataset, bypass main_process_first of transformers
def get_dataset(
    template,
    model_args,
    data_args,
    training_args,
    rank,
    dp_size,
    dp_rank,
    stage,
    tokenizer,
    processor,
):
    dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
    eval_dataset = _get_merged_dataset(
        data_args.eval_dataset,
        model_args,
        data_args,
        training_args,
        stage,
        return_dict=data_args.eval_on_each_dataset,
    )
    dataset = _get_preprocessed_dataset(
        dataset,
        data_args,
        training_args,
        stage,
        template,
        tokenizer,
        processor,
        is_eval=False,
    )
    if isinstance(eval_dataset, dict):
        for eval_name, eval_data in eval_dataset.items():
            eval_dataset[eval_name] = _get_preprocessed_dataset(
                eval_data,
                data_args,
                training_args,
                stage,
                template,
                tokenizer,
                processor,
                is_eval=True,
            )
    else:
        eval_dataset = _get_preprocessed_dataset(
            eval_dataset,
            data_args,
            training_args,
            stage,
            template,
            tokenizer,
            processor,
            is_eval=True,
        )

    dataset_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)
    if data_args.tokenized_path is not None:  # save tokenized dataset to disk
        if training_args.should_save:
            dataset_dict.save_to_disk(data_args.tokenized_path)
            logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
            logger.info_rank0(
                f"Please launch the training with `tokenized_path: {data_args.tokenized_path}`."
            )

    return get_dataset_module(dataset_dict)


def to_device(data: Any, device: Union[str, torch.device, int], non_blocking: bool = False) -> Any:
    """Move inputs to a device"""
    if isinstance(data, Mapping):
        return type(data)({k: to_device(v, device, non_blocking) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(to_device(v, device, non_blocking) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=non_blocking)
    else:
        return data


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


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int, num_workers: int, rank: int):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    init_seed = torch.initial_seed() % 2**32
    worker_seed = num_workers * rank + init_seed
    set_seed(worker_seed)


def build_dataloader(args, dataset, data_collator, batch_size, dp_size, dp_rank):
    dataloader_params = {
        "collate_fn": data_collator,
        "num_workers": args.dataloader_num_workers,
        "pin_memory": args.dataloader_pin_memory,
        "persistent_workers": args.dataloader_persistent_workers,
        "prefetch_factor": args.dataloader_prefetch_factor,
    }
    batch_sampler_params = {
        "drop_last": args.dataloader_drop_last,
        "shuffle": args.train_dataloader_shuffle,
        "data_seed": args.data_seed,
    }

    if hasattr(dataset, "__len__"):
        batch_sampler = BatchSamplerShard(
            len(dataset), batch_size=batch_size, **batch_sampler_params
        )
        dataloader_params["worker_init_fn"] = partial(
            seed_worker,
            num_workers=args.dataloader_num_workers,
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


def filter_dataset(dataset_module, args):
    from datasets import Dataset, IterableDataset
    dataset_module_dst = {}
    length_filter = LengthFilter(args.seq_length)
    for (k, v) in dataset_module.items():
        assert isinstance(v, (Dataset, IterableDataset))
        dataset_module_dst[k] = v.filter(length_filter)
    return dataset_module_dst


def build_llama_fc_dataloaders(args, llamafactory_config, collator, rank, dp_size, dp_rank):
    dataset_module = get_dataset(
        llamafactory_config.template,
        llamafactory_config.model_args,
        llamafactory_config.data_args,
        llamafactory_config.training_args,
        rank,
        dp_size,
        dp_rank,
        stage="sft",
        **llamafactory_config.tokenizer_module
    )
    dataset_module = filter_dataset(dataset_module, args)
    train_dataloader = build_dataloader(
        llamafactory_config.training_args,
        dataset_module["train_dataset"],
        collator,
        args.micro_batch_size,
        dp_size,
        dp_rank,
    )
    eval_dataloader = None
    test_dataloader = None

    return train_dataloader, eval_dataloader, test_dataloader
