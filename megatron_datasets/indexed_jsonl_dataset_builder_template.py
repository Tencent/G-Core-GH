# copyright (c) 2024 tencent inc. all rights reserved.
# nrwu@tencent.com

from glob import glob
from packaging import version
from types import SimpleNamespace
import csv
import json
import os
import pprint
import struct
import sys
import random

import numpy as np
import torch
import datasets
import torch

_CITATION = "_CITATION"
_DESCRIPTION = "_DESCRIPTION"
_HOMEPAGE = ""
_LICENSE = ""
EACH_INDEX_SIZE = 12
EACH_INDEX_WITH_SCORE_SIZE = 16

logger = datasets.logging.get_logger('IndexedJsonlDataset')


class IndexedJsonlDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("3.0.1")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="d1", version=VERSION, description="This part of my dataset covers a first domain"
        ),
    ]
    DEFAULT_CONFIG_NAME = "d1"

    def __init__(
        self,
        *args,
        dp_rank=0,
        dp_size=1,
        num_workers=1,
        access_policy_interleave=False,
        sample_rate=None,
        seed=42,
        debug=False,
        unsplit_data=False,  # naming 应该是 dont_split_data，由于 kwarg 所以暂时不改了。
        consumed_in_this_domain=None,
        shuffle_buffer_size=0,
        enable_pareto=False,
        pareto_alpha=None,
        pareto_scale=None,
        pareto_score_scale=None,
        pad_and_cycle=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert version.parse(datasets.__version__) >= version.parse('3.0.1')

        self.sample_rate = sample_rate
        self.seed = seed
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.num_workers = num_workers
        self.debug = debug
        self.dont_split_data = unsplit_data
        if consumed_in_this_domain is None:
            consumed_in_this_domain = {}
        self.consumed_in_this_domain = consumed_in_this_domain
        self.shuffle_buffer_size = shuffle_buffer_size
        self.pad_and_cycle = pad_and_cycle

        # pareto sampling
        self.enable_pareto = enable_pareto
        self.pareto_alpha = pareto_alpha
        self.pareto_scale = pareto_scale
        self.pareto_score_scale = pareto_score_scale
        if self.enable_pareto:
            assert (self.pareto_alpha and self.pareto_scale and self.pareto_score_scale) is not None
            assert isinstance(self.pareto_alpha, float) and isinstance(self.pareto_scale, float) \
                    and isinstance(self.pareto_score_scale, float)
            assert self.pareto_alpha >= 0.0 and self.pareto_scale > 0.0 and self.pareto_score_scale > 0.0
        '''
        Access policy 和 interleave 概念来自 numa。一般来说，ssd 和 ram 都具有局部性，所以连续访问会更好。
        不过有些情况下，希望无论删掉一部分数据后，访问顺序仍然保持不变，因此需要打开 interleave。

        1. 默认情况下，每个 worker 都会访问 data file 的 1 个 part，例如文件分 4 part，work 0 访问
           part 0，worker 3 访问 part 3。
        2. interleave 允许类似 numa 的交替访问，worker 0 访问 line 1，line 5，line 9....，worker 3
           访问 line 4，line 8，line 12...。
        关闭 interleave 性能更好，但由于 pytorch 有 dataloader，所以也还好。

        参考：https://stackoverflow.com/questions/33315950/mongodb-in-docker-numactl-interleave-all-explanation
        '''
        assert not access_policy_interleave, 'access_policy_interleave deprecated'
        assert not (self.pad_and_cycle and self.dont_split_data), \
            "pad_and_cycle and dont_split_data cannot be True at the same time"

    def _info(self):
        if self.config.name == "d1":
            features = {
                'data_file_name': datasets.Value('string'),
                'offset': datasets.Value('int64'),
                'length': datasets.Value('int32'),
                'worker_id': datasets.Value('int32'),
            }
            features = datasets.Features(features)
        else:
            raise ValueError('wtf')
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        meta = dl_manager.download_and_extract('metadata.json')
        data_dir = os.path.abspath(os.path.dirname(meta))  # is that a good idea ?
        with open(meta, 'r') as inf:
            meta = json.load(inf)
        data_files = meta['data_files']
        idx_files = meta['idx_files']
        nums = meta['nums']
        assert len(data_files) == len(idx_files) and len(data_files) == len(nums)
        domain_name = meta.get('domain_name', 'no-domain-name')

        df_idxf_ns = []
        for fi in range(len(data_files)):
            df = data_files[fi]
            idxf = idx_files[fi]
            n = nums[fi]
            df_idxf_ns.append((df, idxf, n))

        assert len(
            df_idxf_ns
        ) > 0, f"total line in file {data_files} is lower than dp_size {self.dp_size}"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'data_dir': data_dir,
                    'domain_name': domain_name,
                    'df_idxf_ns': {
                        'this': df_idxf_ns
                    },
                    'shard_tfu': [None] * self.num_workers,
                },
            ),
        ]

    def _generate_examples(self, data_dir, domain_name, df_idxf_ns, shard_tfu):
        df_idxf_ns = df_idxf_ns['this']
        each_row_index_size = EACH_INDEX_SIZE
        if self.enable_pareto:
            each_row_index_size = EACH_INDEX_WITH_SCORE_SIZE

        rank = 0
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            worker_sz = 1
        else:
            worker_id = worker_info.id
            worker_sz = worker_info.num_workers
        assert worker_sz == self.num_workers, f'{worker_sz=} {self.num_workers=}'

        df_idxf_ns = sorted(df_idxf_ns)
        if self.debug or os.environ.get("PX_DEBUG_LOG", "0") == "1":
            logger.warn(
                    f'IndexedJsonlDataset epoch rank {rank}' \
                            + f' domain_name {domain_name}' \
                            + f' df_idxf_ns {df_idxf_ns}' \
                            + f' seed {self.seed}' \
                            + f' sample rate {self.sample_rate}'
            )

        consumed_by_this_wk = self.consumed_in_this_domain.setdefault(
            worker_id, SimpleNamespace(epoch=0, line=0)
        )

        def get_tensor_item(value):
            if torch.is_tensor(value):
                assert value.numel() == 1  # 检查张量是否只有一个元素
                return value.item()
            else:
                return value

        epoch = get_tensor_item(consumed_by_this_wk.epoch)
        to_skip = consumed_by_this_wk.line
        rng = torch.Generator("cpu").manual_seed(self.seed + epoch)
        np_rng = np.random.default_rng(self.seed + epoch)
        py_rng = random.Random(self.seed + epoch)
        buf_using = []
        buf_filling = []

        # Guard 掉数据不足的 domain 的情况，实在无法处理，杀掉进程。
        no_enough_data = True
        for _, _, n in df_idxf_ns:
            if self.dont_split_data:
                n_each_wk = n
            else:
                n_each_wk = n // (self.dp_size * worker_sz)
            if n_each_wk > 0:
                no_enough_data = False

        assert not no_enough_data, "data is much less than dp_size, pls provide more data"

        for data_fname, idx_fname, n in df_idxf_ns:
            if self.dont_split_data:
                n_each_wk = n
            elif self.pad_and_cycle:
                # round up n_each_wk
                n_each_wk = (n + self.dp_size * worker_sz - 1) // (self.dp_size * worker_sz)
            else:
                n_each_wk = n // (self.dp_size * worker_sz)

            if n_each_wk < 1:
                logger.warn(f'ignore very small df {df}')
                continue

            if self.dont_split_data:
                idx_f_off = 0
            else:
                idx_f_off = (self.dp_rank * worker_sz + worker_id) * n_each_wk * each_row_index_size

            if self.debug or os.environ.get("PX_DEBUG_LOG", "0") == "1":
                logger.warn(
                    f'IndexedJsonlDataset next file rank {rank}' \
                            + f' dp_rank {self.dp_rank}' \
                            + f' domain_name {domain_name}' \
                            + f' data_fname {data_fname}' \
                            + f' idx_f_off {idx_f_off}' \
                            + f' n_each_wk {n_each_wk}' \
                            + f' len(df_idxf_ns) {len(df_idxf_ns)}'
                )

            with open(os.path.join(data_dir, idx_fname), 'rb') as idx_file:
                abs_data_fname = os.path.join(data_dir, data_fname)
                idx_file.seek(idx_f_off)
                key = 0
                for i in range(n_each_wk):
                    packed = idx_file.read(each_row_index_size)

                    if self.pad_and_cycle and len(packed) != each_row_index_size:
                        idx_file.seek(0)
                        packed = idx_file.read(each_row_index_size)

                    assert len(
                        packed
                    ) == each_row_index_size, f'{len(packed)=} {each_row_index_size=}'
                    key += 1

                    if self.enable_pareto:
                        offset, length, score = struct.unpack('<qif', packed)
                        if not self.pareto_sample(np_rng, score):
                            continue
                        to_yield = (
                            key, {
                                'data_file_name': abs_data_fname,
                                'offset': offset,
                                'length': length,
                                'worker_id': worker_id,
                            }
                        )

                    else:
                        offset, length = struct.unpack('<qi', packed)
                        p = torch.rand(1, generator=rng)[0]
                        if self.sample_rate is not None and p > self.sample_rate:
                            continue
                        to_yield = (
                            key, {
                                'data_file_name': abs_data_fname,
                                'offset': offset,
                                'length': length,
                                'worker_id': worker_id,
                            }
                        )

                    buf_filling.append(to_yield)
                    if len(buf_filling) >= self.shuffle_buffer_size:
                        py_rng.shuffle(buf_filling)
                        buf_filling += buf_using
                        buf_using = buf_filling
                        buf_filling = []

                    if len(buf_using) > 0:
                        poped = buf_using.pop(-1)
                        if to_skip > 0:
                            to_skip -= 1
                        else:
                            yield poped

        py_rng.shuffle(buf_filling)
        buf_filling += buf_using
        buf_using = buf_filling
        buf_filling = []
        while len(buf_using) > 0:
            poped = buf_using.pop(-1)
            if to_skip > 0:
                to_skip -= 1
            else:
                yield poped

    def pareto_sample(self, rng, score):
        alpha = self.pareto_alpha
        scale = self.pareto_scale
        score_scale = self.pareto_score_scale
        s = rng.pareto(alpha, 1) * scale
        if s[0] <= score_scale * score:
            return True
        return False
