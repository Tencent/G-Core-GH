# copyright (c) 2024 tencent inc. all rights reserved.
# nrwu@tencent.com
# 老模板，备份，不要用。

from glob import glob
import csv
import json
import os
import pprint
import struct
import sys

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

    VERSION = datasets.Version("0.0.3")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="d1", version=VERSION, description="This part of my dataset covers a first domain"
        ),
    ]
    DEFAULT_CONFIG_NAME = "d1"

    def __init__(
        self,
        dp_rank=0,
        dp_size=1,
        access_policy_interleave=False,
        sample_rate=None,
        seed=42,
        debug=False,
        unsplit_data=False,
        enable_pareto=False,
        pareto_alpha=None,
        pareto_scale=None,
        pareto_score_scale=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sample_rate = sample_rate
        self.seed = seed
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.debug = debug
        self.unsplit_data = unsplit_data
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

    def _info(self):
        if self.config.name == "d1":
            features = {
                'data_file_name': datasets.Value('string'),
                'offset': datasets.Value('int64'),
                'length': datasets.Value('int32'),
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
            if self.unsplit_data:
                n_each_rank = n
                df_idxf_ns.append((df, idxf, n_each_rank))
            else:
                n_each_rank = n // self.dp_size
                if n < self.dp_size:
                    logger.warn(f'ignore very small df {df}')
                else:
                    df_idxf_ns.append((df, idxf, n_each_rank))
        assert len(
            df_idxf_ns
        ) > 0, f"total line in file {data_files} is lower than dp_size {self.dp_size}"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    'data_dir': data_dir,
                    'domain_name': domain_name,
                    'df_idxf_ns': df_idxf_ns,
                },
            ),
        ]

    def _generate_examples(self, data_dir, domain_name, df_idxf_ns):
        each_row_index_size = EACH_INDEX_SIZE
        if self.enable_pareto:
            each_row_index_size = EACH_INDEX_WITH_SCORE_SIZE
        rank = 0
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()

        df_idxf_ns = sorted(df_idxf_ns)
        if self.debug or os.environ.get("PX_DEBUG_LOG", "0") == "1":
            logger.warn(
                    f'IndexedJsonlDataset epoch rank {rank}' \
                            + f' domain_name {domain_name}' \
                            + f' df_idxf_ns {df_idxf_ns}' \
                            + f' seed {self.seed}' \
                            + f' sample rate {self.sample_rate}'
            )
        if self.enable_pareto:
            generator = np.random.default_rng(self.seed)
        else:
            generator = torch.Generator("cpu").manual_seed(self.seed)

        for data_fname, idx_fname, n_each_rank in df_idxf_ns:
            if self.unsplit_data:
                idx_f_off = 0
            else:
                idx_f_off = self.dp_rank * n_each_rank * each_row_index_size
            if self.debug or os.environ.get("PX_DEBUG_LOG", "0") == "1":
                logger.warn(
                    f'IndexedJsonlDataset next file rank {rank}' \
                            + f' dp_rank {self.dp_rank}' \
                            + f' domain_name {domain_name}' \
                            + f' data_fname {data_fname}' \
                            + f' idx_f_off {idx_f_off}' \
                            + f' n_each_rank {n_each_rank}' \
                            + f' len(df_idxf_ns) {len(df_idxf_ns)}'
                )

            with open(os.path.join(data_dir, idx_fname), 'rb') as idx_file:
                abs_data_fname = os.path.join(data_dir, data_fname)
                idx_file.seek(idx_f_off)
                key = 0
                for i in range(n_each_rank):
                    packed = idx_file.read(each_row_index_size)
                    assert len(packed) == each_row_index_size
                    key += 1
                    if self.enable_pareto:
                        offset, length, score = struct.unpack('<qif', packed)
                        if self._pareto_sample(
                            generator, score, self.pareto_alpha, self.pareto_scale,
                            self.pareto_score_scale
                        ):
                            yield key, {
                                'data_file_name': abs_data_fname,
                                'offset': offset,
                                'length': length,
                            }
                    else:
                        offset, length = struct.unpack('<qi', packed)

                        p = torch.rand(1, generator=generator)[0]
                        if self.sample_rate is not None and p > self.sample_rate:
                            continue

                        yield key, {
                            'data_file_name': abs_data_fname,
                            'offset': offset,
                            'length': length,
                        }

    def _pareto_sample(self, rng, score, alpha, scale, score_scale):
        s = rng.pareto(alpha, 1) * scale
        if s[0] <= score_scale * score:
            return True
        return False
