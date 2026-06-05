import os
import json
from dataclasses import dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class ExtraDataConfig(MappingProtocol):
    processor_path: Optional[str] = field(default=None, metadata={"help": "processor_path"})

    min_pixels_num: Optional[int] = field(default=None, metadata={"help": "min image width * height"})
    max_pixels_num: Optional[int] = field(default=None, metadata={"help": "max image width * height"})
    video_min_frames: Optional[int] = field(default=None, metadata={"help": "min video frames"})
    video_max_frames: Optional[int] = field(default=None, metadata={"help": "max video frames"})
    video_min_pixels: Optional[int] = field(default=None, metadata={"help": "min video frame num_frame * width * height"})
    video_max_pixels: Optional[int] = field(default=None, metadata={"help": "max video frame num_frame * width * height"})

    lmdb_port: Optional[int] = field(default=None, metadata={"help": "lmdb server port"})
    mask_history: bool = field(default=False, metadata={"help": "多轮对话只取最后一轮对话为label"})


@dataclass
class DatasetV3Config(MappingProtocol):
    train_data_path: Optional[list[str]] = field(default_factory=list)
    train_probability: Optional[list[float]] = field(default_factory=list)
    train_data_domain_names: Optional[list[str]] = field(default_factory=list)
    train_total_nums: Optional[list[int]] = field(default_factory=list)

    eval_data_path: Optional[list[str]] = field(default_factory=list)
    eval_sample_nums_per_domain: Optional[list[int]] = field(default_factory=list)
    eval_data_domain_names: Optional[list[str]] = field(default_factory=list)
    eval_total_nums: Optional[list[int]] = field(default_factory=list)
    # eval_iters_per_domain: Optional[list[int]] = field(default_factory=list)


@dataclass
class DataConfig(MappingProtocol):
    py_path: Optional[str] = field(default=None, metadata={"help": "dataset impl python file"})
    fn_name: Optional[str] = field(default=None, metadata={"help": "get dataloader function name"})
    get_batch_fn_name: Optional[str] = field(default=None, metadata={"help": "get batch function name"})
    data_pathes: Optional[List[str]] = field(default=None, metadata={"help": "Pathes to the data files (or configs)"})
    dataset_v3_config_path: Optional[str] = field(default=None, metadata={"help": "dataset v3 json config path"})
    sampler_seed: int = field(default=42, metadata={"help": "Seed for the sampler"})
    system_prompt: Optional[str] = field(default=None, metadata={"help": "System prompt"})
    dataloader_num_workers: int = field(default=1, metadata={"help": "Number of workers for the dataloader"})
    dataloader_pin_memory: bool = field(default=True, metadata={"help": "Whether to pin memory for the dataloader"})
    dataloader_prefetch_factor: Optional[int] = field(default=None, metadata={"help": "Number of batches loaded"})
    shuffle_buffer_size: int = field(default=1000000, metadata={"help": "shuffle buffle size"})

    data_extra: ExtraDataConfig = field(default_factory=ExtraDataConfig)
    dataset_v3: DatasetV3Config = field(default_factory=DatasetV3Config)

    def _config_dataset_v3(self):
        with open(self.dataset_v3_config_path, 'r') as f:
            data_config = json.load(f)

            if "train_data_infos" in data_config.keys():
                for key, values in data_config["train_data_infos"].items():
                    self.dataset_v3.train_data_path.append(values["path"])
                    self.dataset_v3.train_probability.append(float(values["probability"]))
                    self.dataset_v3.train_data_domain_names.append(key)
                    with open(os.path.join(values["path"], 'metadata.json'), 'r') as f:
                        metadata = json.load(f)
                    self.dataset_v3.train_total_nums.append(metadata['total_num'])
            # mega eval_iters <= 0 有特殊含义，即不做 eval；不要加入。
            # TODO(guanyouhe): 这里都加入了，因为不知道 eval_iters 是多少
            # 后面有一个eval_iters_per_domain 与 eval_iters 的判断也得加上
            # 可以直接 eval_iters/global_batch_size... 加到函数参数中
            if "eval_data_infos" in data_config.keys():
                for key, values in data_config["eval_data_infos"].items():
                    self.dataset_v3.eval_data_path.append(values["path"])
                    # 兼容性考虑做保留，默认为 -1。
                    self.dataset_v3.eval_sample_nums_per_domain.append(values.get("eval_samples_num", -1))
                    self.dataset_v3.eval_data_domain_names.append(key)
                    with open(os.path.join(values["path"], 'metadata.json'), 'r') as f:
                        metadata = json.load(f)
                    self.dataset_v3.eval_total_nums.append(metadata['total_num'])

        # 排序 domains（一开始为了 doremi 做的，后来发现排序起来有些代码处理简单）
        # TODO(@xiaotaoliu)：这 6 个 args naming 有点谜，没什么规则。不过有空再说吧，先做高优先的...
        # train
        if self.dataset_v3.train_probability is not None:
            self.dataset_v3.train_probability = [
                x for _, x in sorted(zip(self.dataset_v3.train_data_path, self.dataset_v3.train_probability))
            ]
            self.dataset_v3.train_data_domain_names = [
                x for _, x in sorted(zip(self.dataset_v3.train_data_path, self.dataset_v3.train_data_domain_names))
            ]
        self.dataset_v3.train_data_path = sorted(self.dataset_v3.train_data_path)

        # eval
        if self.dataset_v3.eval_data_path:
            self.dataset_v3.eval_sample_nums_per_domain = [
                x for _, x in sorted(zip(self.dataset_v3.eval_data_path, self.dataset_v3.eval_sample_nums_per_domain))
            ]
            self.dataset_v3.eval_data_domain_names = [
                x for _, x in sorted(zip(self.dataset_v3.eval_data_path, self.dataset_v3.eval_data_domain_names))
            ]
            self.dataset_v3.eval_data_path = sorted(self.dataset_v3.eval_data_path)

        assert self.dataset_v3.train_probability is None or len(self.dataset_v3.train_data_path
                                                          ) == len(self.dataset_v3.train_probability)

        if self.dataset_v3.eval_data_path:
            assert self.dataset_v3.eval_data_domain_names is not None
            assert self.dataset_v3.eval_sample_nums_per_domain is not None
            assert len(self.dataset_v3.eval_data_path) == len(self.dataset_v3.eval_sample_nums_per_domain)
            assert len(self.dataset_v3.eval_data_path) == len(self.dataset_v3.eval_data_domain_names)
            '''
            实际上对于 on-the-fly 处理 + 攒 seq 的情况（例如 pretrain，sft 不 pad），无法提前确定 train-iter 和 eval-iter，
            只能是近似。对于 train 而言，最多是多点数据少点数据的区别，问题不大。

            这里回归 megatron-lm 原版的逻辑：
            1. 用户给出 eval-iter，指定错了就错了。
            2. 如果 eval-iter 太小了，没有消费完，那么会留到下一次 eval，导致 eval 数据略有不同。有 log 可以观察。
            3. 如果 `args.eval_iters` 太大了，会重放数据，不会挂掉。有 log 可以观察。

            说实话这里 eval-iters 没处理好，但 nvidia 原版也是这样的 bug，先不处理了。如果你需要定制化的 eval-iter
            逻辑再联系 nrwu and xiaotaoliu。

            下面是之前 xiaotaoliu 做 pretrain 留下的逻辑，也是正确的。每个 domain 的 eval sample 个数。这里之前没考虑好，
            如果是 on-the-fly 处理，map 后的 sample 个数无法确定，而且绑定了 tokenizer。
            '''
            
            if len(self.dataset_v3.eval_sample_nums_per_domain
                  ) > 0 and all([c >= 0 for c in self.dataset_v3.eval_sample_nums_per_domain]):
                pass
            # TODO(guanyouhe): 不知道 eval_iters/global_batch_size 是多少
                # for eval_sample_nums in self.dataset_v3.eval_sample_nums_per_domain:
                #     self.dataset_v3.eval_iters_per_domain.append(eval_sample_nums // args.global_batch_size)


    def __post_init__(self):
        if self.dataset_v3_config_path is not None:
            assert self.data_pathes is None
            self._config_dataset_v3()

