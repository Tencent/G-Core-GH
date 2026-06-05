# copyright (c) 2024 tencent inc. all rights reserved.
# nrwu@tencent.com, guanyouhe@tencent.com

import json
import random
from typing import Callable, Dict, List

import numpy as np
import torch
from torch.utils.data import IterableDataset as TorchIterableDataset

from gdataset.feat import Feat
from gdataset.store import store_cli_provider
'''
TODO:
- re-ordering (group by resolution)
- use async coroutine to reduce number of process
- thread mode
'''


class ComposedFunc:
    def __init__(self, tmp_func, map_fn):
        self.tmp_func = tmp_func
        self.map_fn = map_fn

    def __call__(self, *args, **kwds):
        return self.map_fn(self.tmp_func(*args, **kwds))


class GDatasetV4(TorchIterableDataset):
    '''
    GDatasetV4 是一个允许从多种存储后段（cos、uniondb 等）读取数据对象的惰性 iterable dataset。

    GDatasetV4 is a iterable dataset that allow lazy loading data objects from storage backends (e.g. COS, uniondb).

    GDatasetV4 的限制：

    - 不支持 filter 类操作（改变数据集长度），我们在数据管道过滤数据。

    - 强制 drop last

    GDatasetV4 also has limits:

    - any operation that changes length(`filter`) is not supported,

    - forced drop last.
    '''
    def __init__(
        self,
        metadata_file: str,
        dp_rank: int = -1,
        dp_size: int = -1,
        gbs: int = 1,
        shuffling_buffer_size: int = 0,
        seed: int = 0,
        consumed: int = 0,
        feats: Dict[str, Feat] = None,
        smart_padding_compare_func: Callable = None,
        smart_padding_buffer_size: int = 0,
        mbs: int = 1,
        dry_run_dataset: bool = False,
        disable_shuffle_data_files: bool = False,
    ):
        '''
        Parameters
        ----------
        metadata_file : str
            metadata file 路径
        dp_rank : int
            data parallel rank
        dp_size : int
            data parallel size
        gbs : int
            global batch size = micro batch size * gradient accumulation step * dp size
        shuffling_buffer_size:
            iterable dataset 只支持局部 shuffle
        seed : int
            随机数种子
        consumed : int
            消费掉的数据数量 = train_iters * gbs
        feats : dict[str, Feat]
            复杂数据类型，见 `gdataset.Feat`
        smart_padding_compare_func : Callable
            对局部样本数据重新排序(smart padding)的比较函数
        smart_padding_buffer_size : int
            对局部样本数据重新排序(smart padding)的样本大小
        mbs : int
            micro batch size

        Returns
        -------

        Examples
        --------

        .. highlight:: python
        .. code-block:: python

            dp_rank = torch.distributed.get_rank()
            dp_size = torch.distributed.get_world_size()

            # 训练的 micro batch size
            mbs = 2

            # 训练的 global batch size（= micro batch size x gradient accumulation step x dp size）
            gbs = 512

            # 假设程序挂掉之前，消费了 100 个 batch，启动要跳掉
            consumed = 100 * gbs

            # 由于数据太多，不可能全局 shuffle，所以一定局部 shuffle。
            shuffling_buffer_size = 32768

            # 数据的主体是 json，json 只支持 int、float、str、dict 等简单类型；
            # 复杂数据类型如 PIL image，如果通过 feats 构造。
            #
            # Feat 的概念来自 huggingface datasets（https://huggingface.co/docs/datasets/about_dataset_features），
            # datasets 通过 features 来指定 image / video 或其他更复杂类型的构造。
            #
            # Storage backend 在此处设置，内部会自行处理。cos region、secret id、secret key 信息都记录在
            # metadata.json 中。
            feats = {
                'caption': JsonFeat(nest=False),
                'cos_url': PilImageFeat(cos=True, new_name='image'),
            }

            # 快速启动；如果有数百万条数据需要跳过，大约 10+ secs 可以搞定。
            d = GDatasetV4('ft_local/metadata.json',
                           dp_rank=dp_rank,
                           dp_size=dp_size,
                           gbs=gbs,
                           shuffling_buffer_size=shuffling_buffer_size,
                           consumed=consumed,
                           feats=feats,
                           seed=42)

        A example of metadata.json

        .. highlight:: json
        .. code-block:: json

            {
              "name": "demo",
              "description": "bla bla bla",
              "cos_region": "ap-shanghai",
              "cos_secret_id": "yyy",
              "cos_secret_key": "zzz",
              "data_files": [
                { "fp": "ft_local/CSIG_pexels_clean_v2-1_0609_part001.jsonl" },
                { "fp": "ft_local/CSIG_pexels_clean_v2-1_0609_part002.jsonl" },
                { "fp": "ft_local/CSIG_pexels_clean_v2-1_0609_part012.jsonl" }
              ],
              "data_file_num_lines": [
                100000,
                100000,
                29563
              ]
            }
        '''
        assert dp_rank >= 0 and dp_size > 0
        assert shuffling_buffer_size >= 0
        assert shuffling_buffer_size % gbs == 0, f'{shuffling_buffer_size=} must be multiple of {gbs=}'
        assert gbs % dp_size == 0, f'{gbs=} must be multiple of {dp_size=}'
        assert consumed >= 0 and consumed % gbs == 0
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.gbs = gbs
        self.shuffling_buffer_size = (shuffling_buffer_size + gbs - 1) // gbs * gbs
        self.seed = seed
        self.consumed = consumed
        self.epoch = None
        self.map_func = None
        self.mbs = mbs
        self.dry_run_dataset = dry_run_dataset
        self.disable_shuffle_data_files = disable_shuffle_data_files

        with open(metadata_file, 'r') as inf:
            metadata = inf.read()
        self.metadata = json.loads(metadata)

        self.feats = {}
        if feats:
            self.feats = feats
        for _, feat in self.feats.items():
            feat.post_init(self.metadata)

        self.smart_padding_compare_func = smart_padding_compare_func
        self.smart_padding_buffer_size = smart_padding_buffer_size
        self.use_smart_padding = False

        if self.smart_padding_compare_func is not None:
            assert self.smart_padding_buffer_size > 0
            self.use_smart_padding = True
            assert self.shuffling_buffer_size % self.smart_padding_buffer_size == 0, \
            f'{shuffling_buffer_size=} must be multiple of {smart_padding_buffer_size=}'

    def __len__(self):
        '''
        数据集长度

        length of dataset (trailing samples dropped)

        注意：torch iterable dataset 不支持 `__len__`

        NOTE: `__len__` is not supported by torch iterable datasets.

        Returns
        -------
        : int
            length of dataset
        '''
        data_file_num_lines = self.metadata['data_file_num_lines']
        data_file_num_lines = [x // self.gbs * self.gbs for x in data_file_num_lines]
        total_num_lines = sum(data_file_num_lines)
        length = total_num_lines // self.dp_size
        return length

    def map(self, map_fn: Callable) -> TorchIterableDataset:
        '''
        类似 hugggingface datasets 的 map 操作（https://huggingface.co/docs/datasets/en/process#map）。

        a `map` similar to huggingface datasets(https://huggingface.co/docs/datasets/en/process#map).

        Parameters
        ----------
        map_fn : Callable
            map function

        Returns
        -------
        : torch.utils.data.IterableDataset
        '''
        if self.map_func is None:
            self.map_func = map_fn
        else:
            tmp_func = self.map_func
            self.map_func = lambda x: map_fn(tmp_func(x))

            # 如果后台进程使用 spawn 创建，就需要使用 callabke object 替代 lambda
            # 因为 local lamdba function 不能被正常序列化
            # tmp_func = self.map_func
            # composed_func = ComposedFunc(tmp_func, map_fn)
            # self.map_func = composed_func

        return self

    def set_epoch(self, epoch: int):
        '''
        set epoch.

        see `https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable` for why `set_epoch`

        Parameters
        ----------
        epoch : int
            epoch number (starting from 0)
        '''
        self.epoch = epoch

    def get_dataloader_dp(self):
        # dataloader worker_num >1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            dl_wk_id = 0
            dl_wk_sz = 1
        else:
            dl_wk_id = worker_info.id
            dl_wk_sz = worker_info.num_workers
        dl_dp_rank = self.dp_rank + dl_wk_id * self.dp_size
        dl_dp_size = self.dp_size * dl_wk_sz

        assert dl_dp_size % self.gbs == 0 or self.gbs % dl_dp_size == 0, \
            f'{self.gbs=} and {dl_dp_size=}. Either gbs must be a multiple of dl_dp_size, or dl_dp_size must be a multiple of gbs'
        if self.shuffling_buffer_size == 0:
            assert self.gbs >= dl_dp_size
        else:
            assert 0 == self.shuffling_buffer_size % (dl_dp_size * self.mbs)
        return dl_dp_rank, dl_dp_size

    def parse_fp_line(self, line):
        if isinstance(line, dict):
            jobj = line
        else:
            jobj = json.loads(line.strip())

        fnames = [_ for _ in jobj.keys()]
        for fk in fnames:
            if fk not in self.feats:
                continue
            try:
                fv = jobj[fk]
                jobj.pop(fk)
                encoded = self.feats[fk].encode_example(fk, fv)
            except Exception as e:
                print(f'error in encoding {fk=} {fv=} pkey={line} {e}')
                raise e
            jobj.update(encoded)

        return jobj

    def smart_padding_reorder(self, py_rng, lines: List[str], line_i: int):
        if not self.use_smart_padding:
            return lines
        lines = [json.loads(line) for line in lines]
        num_smart_pad = (
            len(lines) + self.smart_padding_buffer_size - 1
        ) // self.smart_padding_buffer_size

        reorder_lines = []
        for i in range(num_smart_pad):
            begin_idx = i * self.smart_padding_buffer_size
            end_idx = min(len(lines), (i + 1) * self.smart_padding_buffer_size)
            sorted_lines = sorted(lines[begin_idx:end_idx], key=self.smart_padding_compare_func)
            assert len(
                sorted_lines
            ) % self.gbs == 0, f"force drop last, the len of dataset must be n * gbs"

            num_batch = (len(sorted_lines) + self.gbs - 1) // self.gbs
            # for training diversity
            idxs = list(range(num_batch))
            _py_rng = random.Random(self.seed * self.epoch + line_i + i)
            _py_rng.shuffle(idxs)

            for idx in idxs:
                begin_i = idx * self.gbs
                end_i = min(len(sorted_lines), (idx + 1) * self.gbs)
                tmp_lines = sorted_lines[begin_i:end_i]
                reorder_lines.extend(tmp_lines)

        return reorder_lines

    # for debugging purpose
    def iter_fp_pattern(
        self, data_file_path_like, py_rng, num_lines, consumed_this_epoch, num_read_once
    ):
        n_read = 0
        n_parsed = 0
        dl_dp_rank, dl_dp_size = self.get_dataloader_dp()

        with open(data_file_path_like['fp'], 'r') as data_file:

            for line_i in range(0, num_lines, num_read_once):
                lines = []
                for line_j in range(line_i, min(num_lines, line_i + num_read_once)):
                    line = data_file.readline()
                    n_read += 1
                    assert line, f'unexpected eof {data_file_path_like=} {num_lines=}'
                    lines.append(line)
                assert len(lines) % self.gbs == 0

                # skip if consumed
                if n_read <= consumed_this_epoch:
                    n_parsed = n_read
                    continue

                # can't shuffle iterable dataset in dataloader
                # see https://huggingface.co/docs/datasets/v1.11.0/dataset_streaming.html#shuffling-the-dataset-shuffle
                if self.shuffling_buffer_size > 0:
                    _py_rng = random.Random(self.seed * self.epoch + line_i)
                    _py_rng.shuffle(lines)
                    lines = self.smart_padding_reorder(py_rng, lines, line_i)

                # NOTE: On resuming (run 2), if dp_size * worker_num > gbs, the order of data
                # might be different from previous run (run 1). But it does not matter, since no sample
                # will be replayed or missed.
                # assert self.gbs % dl_dp_size == 0, f'invalid {self.gbs=} {dl_wk_sz=} {self.dp_size=}'

                # manual dp sharding, no torch.data.sampler needed
                for start_li in range(dl_dp_rank * self.mbs, len(lines), dl_dp_size * self.mbs):
                    for li in range(start_li, start_li + self.mbs):
                        if n_parsed + li < consumed_this_epoch:
                            pass
                        else:
                            line = lines[li]
                            obj = self.parse_fp_line(line)
                            if self.map_func is not None:
                                obj = self.map_func(obj)
                            yield obj

                n_parsed += len(lines)

    def iter_udb_pattern(
        self, data_file_path_like, py_rng, num_lines, consumed_this_epoch, num_read_once
    ):
        """
        UDB data iterator with distributed sharding and resume support.
        
        Parameters
        ----------
        data_file_path_like : dict
            Configuration containing UDB connection details
        py_rng : random.Random
            Random number generator for shuffling
        num_lines : int
            Total number of data records (for progress tracking)
        consumed_this_epoch : int
            Number of already processed records
        num_read_once: int
            Batch size for primary key retrieval
        
        Yields
        ------
        : Any
            Processed data samples
        """
        # Initialize UDB client
        udb_cli = store_cli_provider(self.metadata, uniondb=True)

        # Extract connection parameters
        udb_config = data_file_path_like["uniondb"]
        columns = udb_config["columns"]
        udb_key_file_path = data_file_path_like['udb_key_file']

        n_read = 0
        n_parsed = 0
        dl_dp_rank, dl_dp_size = self.get_dataloader_dp()

        with open(udb_key_file_path, 'r') as data_file:

            for line_i in range(0, num_lines, num_read_once):
                lines = []
                for line_j in range(line_i, min(num_lines, line_i + num_read_once)):
                    line = data_file.readline()
                    n_read += 1
                    assert line, f'unexpected eof {data_file_path_like=} {num_lines=}'
                    lines.append(line)
                assert len(lines) % self.gbs == 0

                # skip if consumed
                if n_read <= consumed_this_epoch:
                    n_parsed = n_read
                    continue

                if self.shuffling_buffer_size > 0:
                    _py_rng = random.Random(self.seed * self.epoch + line_i)
                    _py_rng.shuffle(lines)
                    lines = self.smart_padding_reorder(py_rng, lines, line_i)

                for start_li in range(dl_dp_rank * self.mbs, len(lines), dl_dp_size * self.mbs):
                    for li in range(start_li, start_li + self.mbs):
                        if n_parsed + li < consumed_this_epoch:
                            pass
                        else:
                            # nested dict (in json) is not supported by uniondb yet, it returns a dict
                            # of bytes.
                            jobj = udb_cli.get(primary_key=lines[li].strip(), columns=columns)
                            fnames = [_ for _ in jobj.keys()]
                            for fk in fnames:
                                if fk not in self.feats:
                                    continue
                                try:
                                    fv = jobj[fk]
                                    jobj.pop(fk)
                                    encoded = self.feats[fk].encode_example(fk, fv)
                                except Exception as e:
                                    print(
                                        f'error in encoding {fk=} {fv=} pkey={lines[li].strip()} {e=}'
                                    )
                                    raise e
                                jobj.update(encoded)
                            if self.map_func is not None:
                                jobj = self.map_func(jobj)
                            yield jobj

                n_parsed += len(lines)

    def iter_data_file(self, data_file_path_like, py_rng, np_rng, num_lines, consumed_this_epoch):
        assert consumed_this_epoch < num_lines
        assert consumed_this_epoch % self.gbs == 0, f'unexpected {consumed_this_epoch=} {self.gbs=}'

        if self.shuffling_buffer_size == 0:
            num_read_once = self.gbs
        else:
            num_read_once = self.shuffling_buffer_size

        if "uniondb" in data_file_path_like:
            yield from self.iter_udb_pattern(
                data_file_path_like, py_rng, num_lines, consumed_this_epoch, num_read_once
            )
        else:
            yield from self.iter_fp_pattern(
                data_file_path_like, py_rng, num_lines, consumed_this_epoch, num_read_once
            )

    def __iter__(self):
        '''
        Iter throught the dataset.

        Yields
        ------
        : Any
            Processed data sample
        '''
        assert self.epoch is not None
        seed = self.seed + self.epoch
        py_rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        data_files = self.metadata['data_files']
        num_data_files = len(data_files)
        data_file_num_lines = self.metadata['data_file_num_lines']
        data_file_num_lines = [x // self.gbs * self.gbs for x in data_file_num_lines]

        # shuffle data file list
        if not self.disable_shuffle_data_files:
            indices = py_rng.sample(range(num_data_files), num_data_files)
            data_files = [data_files[i] for i in indices]
            data_file_num_lines = [data_file_num_lines[i] for i in indices]

        # consumed
        total_num_lines = sum(data_file_num_lines)
        if self.consumed <= self.epoch * total_num_lines:
            consumed_this_epoch = 0
        else:
            assert self.consumed // total_num_lines == self.epoch
            consumed_this_epoch = self.consumed % total_num_lines

        for i, data_file_path_like in enumerate(data_files):
            num_lines = data_file_num_lines[i]
            if consumed_this_epoch >= num_lines:
                consumed_this_epoch -= num_lines
            else:
                yield from self.iter_data_file(
                    data_file_path_like, py_rng, np_rng, num_lines, consumed_this_epoch
                )
                consumed_this_epoch = 0
