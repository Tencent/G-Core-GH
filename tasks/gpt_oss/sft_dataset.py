from dataclasses import dataclass
import itertools
import random
import re
import copy
import torch
from torch.utils.data import IterableDataset as TorchIterableDataset

from megatron.training import get_args
from megatron_datasets.tasks.math_rl_v3 import map_dataset
from megatron_datasets.utils import print_rank_0

from gdataset import GDatasetV4


@dataclass
class GSftDataset(TorchIterableDataset):
    def __init__(
        self,
        tokenizer,
        seq_len,
        mbs,
        gbs,
        train=False,
        metadata_file=None,
        dp_rank=0,
        dp_size=1,
        shuffling_buffer_size=1000,
        seed=0,
        eos_token=None,
        dataset_map_fn=None,
    ):
        self.seq_len = seq_len
        self.mbs = mbs
        self.gbs = gbs
        self.tokenizer = tokenizer
        self.train = train
        # TODO(parkeychen): interleave datasets
        self.metadata_file = metadata_file
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.shuffling_buffer_size = shuffling_buffer_size
        self.seed = seed
        self.in_iter = False
        self.dataset_map_fn = dataset_map_fn

        if eos_token is None:
            self.eos_token = self.tokenizer._tokenizer.eos_token
        else:
            self.eos_token = eos_token

        if not self.train:
            assert self.shuffling_buffer_size == 0
        args = get_args()
        self.start_epoch = 0
        consumed = 0
        print_rank_0(f'GSftDataset.__init__: args.iteration {args.iteration}')
        # consumed = args.iteration * self.mbs * dp_size
        self.make_underlying(consumed=consumed, epoch=self.start_epoch)

    def make_underlying(self, consumed, epoch):
        self.underlying = GDatasetV4(
            metadata_file=self.metadata_file,
            dp_rank=self.dp_rank,
            dp_size=self.dp_size,
            gbs=self.gbs,
            shuffling_buffer_size=self.shuffling_buffer_size,
            seed=self.seed,
            consumed=consumed,
        )
        self.underlying.set_epoch(epoch)

    def iter_in_epoch(self, epoch):
        if torch.distributed.get_rank() == 0:
            print(f'GSftDataset.iter_in_epoch epoch {epoch} train {self.train}')
        for example in self.underlying:
            assert self.dataset_map_fn is not None, "for gpt-oss provide dataset_map_fn"
            o = self.dataset_map_fn(self.tokenizer, example)
            o['train'] = self.train
            o['epoch'] = epoch
            o['line'] = 1
            yield o

    def __iter__(self):
        assert not self.in_iter
        self.in_iter = True
        for epoch in itertools.count(start=self.start_epoch):
            yield from self.iter_in_epoch(epoch)
            self.make_underlying(consumed=0, epoch=epoch + 1)
        assert False, 'never reachable'
