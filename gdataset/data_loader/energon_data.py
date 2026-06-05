import json
import os
from typing import List

import torch

from gdataset.data_loader.llamafactory_bridge import DATA_CONFIG, DatasetAttr, get_dataset_converter
from gdataset.data_loader.processor import (
    LengthFilter,
    ProcessorWrapper,
    StackedMap,
    _get_dataset_processor,
)

from megatron.core import parallel_state
from megatron.energon import (
    BatchDataset,
    Cooker,
    CrudeSample,
    DefaultTaskEncoder,
    FilterDataset,
    MapDataset,
    Sample,
    WorkerConfig,
    basic_sample_keys,
    edataclass,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    stateless,
)
from megatron.energon.flavors import CrudeJsonlDatasetFactory
from megatron.energon.task_encoder.base import get_stateless


@edataclass
class RawSample(Sample):
    data: dict


@stateless()
def cook_text(sample: CrudeSample) -> RawSample:
    return RawSample(**basic_sample_keys(sample), data=sample["json"])


class MapWrapper:
    def __init__(self, map_fun):
        self.map_fun = map_fun

    @stateless()
    def __call__(self, sample: RawSample):
        return RawSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,
            __subflavors__=sample.__subflavors__,
            data=self.map_fun(sample.data)
        )


class FilterWrapper:
    def __init__(self, filter):
        self.filter = filter

    @stateless()
    def __call__(self, sample: RawSample):
        return self.filter(sample.data)


class CollatorWraper:
    def __init__(self, collator):
        self.collator = collator

    @stateless()
    def __call__(self, samples: List[RawSample]):
        samples = [e.data for e in samples]
        return RawSample(__key__=None, __restore_key__=None, data=self.collator(samples))


class SimpleCookingTaskEncoder(DefaultTaskEncoder):
    cookers = [Cooker(cook=cook_text)]


def build_energon_data_loaders(args, llamafactory_config, collator, rank, dp_size, dp_rank):
    torch.manual_seed(42)
    worker_config = WorkerConfig(
        rank=dp_rank,
        world_size=dp_size,
        num_workers=args.num_workers,
        seed_offset=42,
    )

    data_args = llamafactory_config.data_args
    assert len(data_args.dataset) == 1, f"{data_args.dataset}"
    dataset_dir = data_args.dataset_dir
    dataset = data_args.dataset[0]

    config_path = os.path.join(dataset_dir, DATA_CONFIG)
    with open(config_path) as f:
        dataset_info = json.load(f)
    assert dataset in dataset_info
    assert "file_name" in dataset_info[dataset]
    dataset_attr = DatasetAttr("file", dataset_name=dataset_info[dataset]["file_name"])
    dataset_attr.join(dataset_info[dataset])

    local_path = os.path.join(dataset_dir, dataset_attr.dataset_name)

    # single sample with no batching
    train_dataset = get_train_dataset(
        local_path,
        worker_config=worker_config,
        batch_size=None,
        shuffle_buffer_size=None,
        max_samples_per_sequence=None,
        task_encoder=SimpleCookingTaskEncoder(),
    )

    stacked_map = StackedMap()
    # map format
    func = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
    stacked_map.add(func)
    # processor
    stage = "sft" if not getattr(args, "use_grpo", False) else "ppo"
    processor_args = (
        llamafactory_config.data_args, stage, llamafactory_config.template,
        llamafactory_config.tokenizer_module["tokenizer"],
        llamafactory_config.tokenizer_module["processor"]
    )

    func = ProcessorWrapper(_get_dataset_processor(*processor_args))
    stacked_map.add(func)
    stacked_map = MapWrapper(stacked_map)
    train_dataset = MapDataset(
        train_dataset, stacked_map, stateless_map_fn=True, worker_config=worker_config
    )

    # filter invalid sample
    filter_length = args.seq_length
    if stage == "ppo" and getattr(args, "ppo_resp_seq_len") is not None:
        filter_length = filter_length - args.ppo_resp_seq_len
        assert filter_length > 0

    filter = FilterWrapper(LengthFilter(filter_length))
    train_dataset = FilterDataset(train_dataset, filter_fn=filter, worker_config=worker_config)

    batch_size = args.micro_batch_size
    if args.use_grpo:
        batch_size = args.ppo_rollout_micro_batch_size

    # TODO(hessianliu): customisable map and filter by user
    collator = CollatorWraper(collator)
    train_dataset = BatchDataset(
        train_dataset,
        batch_size=batch_size,
        batcher=collator,
        batcher_stateless=True,
        drop_last=True,
        worker_config=worker_config
    )

    # filter invalid data
    # batch dataset

    loader = get_savable_loader(
        train_dataset,
        watchdog_timeout_seconds=None,
        prefetch_factor=args.px_dataloader_prefetch_factor
    )

    # TODO(hessianliu): remove dep on training
    from megatron.training.checkpointing import get_checkpoint_name

    if args.load is not None:
        if getattr(args, "dataloader_save", None):
            dp_rank = parallel_state.get_data_parallel_rank()
            data_save_name = get_checkpoint_name(
                args.dataloader_save,
                args.iteration,
                pipeline_rank=
                0,  # Only the first pipeline parallel rank stores the dataloader checkpoint.
                basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
            )
            if os.path.exists(data_save_name):
                try:
                    dataset_state_dict = torch.load(
                        data_save_name, map_location="cpu", weights_only=False
                    )
                    loader.restore_state_rank(dataset_state_dict["dataloader_state_dict"])
                    print(f"restored dataset state from {data_save_name}")
                except Exception as e:
                    print("loading dataset state failed. Skipping. " + str(e))
            else:
                print(f"dataset state {data_save_name} does not exist")

    return (EnergonDataloader(loader), None, None)


class EnergonDataloader:
    """A wrapper to use Megatron Energon dataloader with the Megatron-LM training loop."""
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iter = iter(cyclic_iter(dataloader))

    def __next__(self):
        return self._iter.__next__()

    def __iter__(self):
        return self._iter.__iter__()

    def save_state(self):
        return self._dataloader.save_state_rank()


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x.data
