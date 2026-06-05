# copyright (c) 2024 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

from dataclasses import dataclass
from datetime import datetime
from functools import partial
from types import SimpleNamespace
from typing import List
import copy
import json
import os
import pickle

from datasets import interleave_datasets, concatenate_datasets, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset as TorchIterableDataset
from tqdm import tqdm
import torch

from megatron_datasets.utils import print_rank_0, print_datetime


class ConsumedByThisRank:
    def __init__(self, epoch, line):
        self.epoch = epoch
        self.line = line

    def __str__(self):
        return f'(epoch={self.epoch}, line={self.line})'

    def __repr__(self):
        return str(self)

    def __eq__(self, rhs):
        return self.epoch == rhs.epoch and self.line == rhs.line

    def __lt__(self, rhs):
        if self.epoch == rhs.epoch:
            return self.line < rhs.line
        else:
            return self.epoch < rhs.epoch

    def __le__(self, rhs):
        return self < rhs or self == rhs

    def merge(self, rhs):
        # 合并两个 train data consuming progress
        # 没有 ABC，看起来不是很 OO。
        if self < rhs:
            self.epoch = rhs.epoch
            self.line = rhs.line


def get_epoch_and_line(consuming_progresses, rank):
    epoch = 0
    line = 0
    if consuming_progresses is not None:
        z = ConsumedByThisRank(epoch=0, line=0)
        prg = consuming_progresses.setdefault(rank, z)
        epoch = prg.epoch
        line = prg.line
    return epoch, line


def update_epoch_and_line(consuming_progresses, rank, data):
    # 记录数据消费进度。
    # 2. consume 了一半的 sample ，热启动会被跳过，相当于浪费 0.5 条样本。解决方案是 save 下 partially used 进度，
    # 有点小麻烦，问题很小，不会重复就好。
    if data is None:
        return
    if not data['train'][0].item():
        return
    z = ConsumedByThisRank(epoch=0, line=0)
    prg = consuming_progresses.setdefault(rank, z)
    prev_epoch = prg.epoch

    max_epoch = torch.max(data['epoch']).item()
    if max_epoch != prev_epoch:
        prg.epoch = max_epoch
        prg.line = 0
    mask = data['epoch'] != max_epoch
    line = torch.sum(data['line'].masked_fill(mask, 0))
    prg.line += torch.sum(line).item()


@dataclass
class MegaIndexedJsonlDataset(TorchIterableDataset):
    # 对比之前各种实现的六大优势：
    #  1. Iterable dataset，可以 hold、快速加载大量数据。
    #  2. json 支持任意字段，非常灵活。
    #  3. 作业死亡后热启动快速
    #  4. 预处理快速，在 gemini 上随时简单处理即可。
    #  5. train 数据热启动精确（没有数据重放、可以没有丢失）
    #  6. 不要求 data files 是 dp size 的倍数关系，gen 数据比较灵活。

    def __init__(
        self,
        path_likes,
        domain_probabilities,
        domain_names,
        dp_rank=0,
        dp_size=1,
        epoch=0,
        consumed=0,
        access_policy_interleave=False,
        shuffle_buffer_size=1000,
        seed=0,
        train=False,
        retention_rates_per_domains=[],
        unsplit_eval_data=False,
        enable_pareto=[],
        pareto_alphas=[],
        pareto_scales=[],
        pareto_score_scales=[],
        pad_and_cycle=False,
        shuffle_strategy="interleave",
    ):
        assert isinstance(path_likes, list)
        if domain_probabilities is not None:
            assert len(domain_probabilities) == len(path_likes)
        self.path_likes = path_likes  # `hf.datasets.PathLike`
        self.domain_probabilities = domain_probabilities
        self.domain_names = domain_names
        self.dp_rank = dp_rank
        self.dp_size = dp_size

        # `access_policy_interleave` 不是指混合数据集 interleave。
        self.access_policy_interleave = access_policy_interleave

        self.seed = seed
        self.shuffle_strategy = shuffle_strategy
        self.shuffle_buffer_size = shuffle_buffer_size
        self.consumed = consumed
        self.to_skip = consumed
        self.train = train
        self.retention_rates_per_domains = retention_rates_per_domains
        self.unsplit_eval_data = unsplit_eval_data
        self.flag_on_pareto_sampling = False
        self.enable_pareto = enable_pareto
        self.pareto_alphas = pareto_alphas
        self.pareto_scales = pareto_scales
        self.pareto_score_scales = pareto_score_scales
        self.pad_and_cycle = pad_and_cycle
        if len(self.enable_pareto) > 0:
            assert self.train, f"only train dataset can enable pareto"
            assert len(domain_probabilities) == len(self.enable_pareto)
            assert len(domain_probabilities) == len(self.pareto_alphas)
            assert len(domain_probabilities) == len(self.pareto_scales)
            assert len(domain_probabilities) == len(self.pareto_score_scales)
            self.flag_on_pareto_sampling = True
        self.print_domain_id_map()
        self.ds = self.make_underlying(epoch)
        self.ds_iter = self.skip()
        self.eval_file_cache = {}
        self.train_file_cache = {}

    def print_domain_id_map(self):
        domain_id_map = []
        for domain_id, path_like in enumerate(self.path_likes):
            d = {
                'domain_id': domain_id,
                'domain_name': self.domain_names[domain_id],
                'domain_path_like': path_like,
            }
            if self.domain_probabilities:
                d['domain_probabilities'] = self.domain_probabilities[domain_id]
            if self.flag_on_pareto_sampling:
                d['enable_pareto'] = self.enable_pareto[domain_id]
                d['pareto_alphas'] = self.pareto_alphas[domain_id]
                d['pareto_scales'] = self.pareto_scales[domain_id]
                d['pareto_score_scales'] = self.pareto_score_scales[domain_id]
            domain_id_map.append(d)
        domain_id_map_str = 'MegaIndexedJsonlDataset id / domain mapping ' + json.dumps(
            domain_id_map, indent=4
        )
        print_rank_0(domain_id_map_str)

    def make_underlying(self, epoch):
        def add_domain_id(domain_id, example):
            example["domain_id"] = torch.tensor(domain_id, dtype=torch.int64)
            return example

        # interleave（合并），shuffle，分布式切割数据集
        # 理论上可以改成 mapped dataset 的，index 完全可以在内存里存在，`__getitem__` 里读数据文件即可。
        # 但是这样：
        # 1. 无法做 map、filter 的操作了（小问题，`__getitem__` 随便搞搞即可）。
        # 2. 无法解决合并样本导致总长度无法计算的问题，所以也没啥用。
        subdatasets = []
        for domain_id, path_like in enumerate(self.path_likes):
            if self.train:
                sample_rate = self.retention_rates_per_domains[domain_id] if len(
                    self.retention_rates_per_domains
                ) > 0 else None
                enable_pareto, pareto_alpha, pareto_scale, pareto_score_scale = False, None, None, None
                if self.flag_on_pareto_sampling:
                    enable_pareto = self.enable_pareto[domain_id]
                    pareto_alpha = self.pareto_alphas[domain_id]
                    pareto_scale = self.pareto_scales[domain_id]
                    pareto_score_scale = self.pareto_score_scales[domain_id]
                ## 区分开需要 pareto 抽样和不需要 pareto 抽样的数据
                # 不需要抽样的数据就不再重新做预处理了，兼容老的数据
                if enable_pareto:
                    ds = load_dataset(
                        path_like,
                        split='train',
                        streaming=True,
                        trust_remote_code=True,
                        dp_rank=self.dp_rank,
                        dp_size=self.dp_size,
                        access_policy_interleave=self.access_policy_interleave,
                        sample_rate=sample_rate,
                        enable_pareto=enable_pareto,
                        pareto_alpha=pareto_alpha,
                        pareto_scale=pareto_scale,
                        pareto_score_scale=pareto_score_scale,
                    )
                else:
                    ds = load_dataset(
                        path_like,
                        split='train',
                        streaming=True,
                        trust_remote_code=True,
                        dp_rank=self.dp_rank,
                        dp_size=self.dp_size,
                        access_policy_interleave=self.access_policy_interleave,
                        sample_rate=sample_rate,
                    )
            else:
                kwargs = {}
                if self.pad_and_cycle:
                    kwargs = {"pad_and_cycle": self.pad_and_cycle}
                ds = load_dataset(
                    path_like,
                    split='train',
                    streaming=True,
                    trust_remote_code=True,
                    dp_rank=self.dp_rank,
                    dp_size=self.dp_size,
                    access_policy_interleave=self.access_policy_interleave,
                    unsplit_data=self.unsplit_eval_data,
                    **kwargs,
                )
            if self.shuffle_strategy == "intra_domain" and self.shuffle_buffer_size > 0:
                ds = ds.shuffle(buffer_size=self.shuffle_buffer_size, seed=self.seed + epoch)
            ds = ds.map(partial(add_domain_id, domain_id))
            subdatasets.append(ds)

        if self.shuffle_strategy == "interleave":
            # https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/main_classes#datasets.interleave_datasets
            # 覆盖了 eval 无 probabilities 的情况，算是合理的。
            interleaved_ds = interleave_datasets(
                subdatasets,
                probabilities=self.domain_probabilities,
                seed=self.seed,
                stopping_strategy='all_exhausted'
            )
            if self.shuffle_buffer_size > 0:
                interleaved_ds = interleaved_ds.shuffle(
                    buffer_size=self.shuffle_buffer_size, seed=self.seed + epoch
                )
            distributed_ds = interleaved_ds
        elif self.shuffle_strategy in ["intra_domain", "no_shuffle"]:
            distributed_ds = concatenate_datasets(subdatasets)
        elif self.shuffle_strategy == "inter_domain" and self.shuffle_buffer_size > 0:
            distributed_ds = concatenate_datasets(subdatasets).shuffle(
                buffer_size=self.shuffle_buffer_size, seed=self.seed + epoch
            )
        else:
            raise NotImplementedError(f"do not support shuffle_strategy: {self.shuffle_strategy}")

        # IndexedJsonlDataset 本身就是 distributed，不需要 split_dataset_by_node。
        return distributed_ds

    def read_and_parse_obj_from_jsonl(self, fname, offset, length):
        # no need for a file handler cache since it's handled by the os kernel
        if not self.train:
            file_cache = self.eval_file_cache
        else:
            file_cache = self.train_file_cache

        if fname in file_cache.keys():
            inf = file_cache[fname]
        else:
            inf = open(fname, 'rb')
            file_cache[fname] = inf
        inf.seek(offset)
        line = inf.read(length)
        obj = json.loads(line)

        return obj

    def log_skip(self):
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{time_str}] skip consumed lines' \
                + f' rank {torch.distributed.get_rank()}' \
                + f' dp_rank {self.dp_rank}' \
                + f' to_skip {self.to_skip}')

    def skip(self):
        # 跳过已经消费掉的数据（因为 iterable dataset 不是 random accessable）。
        # 由于只是 read 12 bytes 的 index，所以速度非常快。如果数据在 ssd，每个 dp worker，每秒
        # 可以 skip ~10000 jsonl。
        # TODO skip 的逻辑还可以再优化加速下
        ds_iter = iter(self.ds)
        if self.to_skip > 0:
            self.log_skip()
            while self.to_skip > 0:
                try:
                    next(ds_iter)
                    self.to_skip -= 1
                    if self.to_skip % 10000 == 0:
                        self.log_skip()
                except StopIteration:
                    break
            print(f'done to_skip rank {torch.distributed.get_rank()}')
        return ds_iter

    def __iter__(self):
        # 根据索引，从 data file 取出数据
        while True:
            try:
                idx = next(self.ds_iter)
            except StopIteration:
                break
            fname = idx['data_file_name']
            offset = idx['offset']
            length = idx['length']
            domain_id = idx['domain_id']
            json_obj = self.read_and_parse_obj_from_jsonl(fname, offset, length)

            # # 保留字段
            assert isinstance(json_obj, dict) \
                    and 'deleted' not in json_obj \
                    and 'train' not in json_obj and 'epoch' not in json_obj and 'line' not in json_obj
            # and 'domain_id' not in json_obj
            json_obj['domain_id'] = domain_id
            yield json_obj
