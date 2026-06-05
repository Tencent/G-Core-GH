import copy
from typing import Any, Literal, Mapping, Optional, Union

from .supervised import SupervisedDatasetProcessor
from .unsupervised import UnsupervisedDatasetProcessor


class ProcessorWrapper:
    def __init__(self, processor):
        self._processor = processor

    def __call__(self, example):
        return self._processor.preprocess_example(example)


class LengthFilter:
    def __init__(self, max_length, check=False):
        self.max_length = max_length
        self.check = check

    def __call__(self, example):
        ans = len(example['input_ids']) <= self.max_length
        if self.check:
            assert ans
        if not ans:
            print(f"filtered {self.max_length} {len(example['input_ids'])}", flush=True)
        return ans


class StackedMap:
    def __init__(self):
        self.map_fns = []

    def add(self, map_func):
        self.map_fns.append(map_func)

    def __call__(self, sample):
        json_data = copy.copy(sample)
        json_data.pop("__images_feat__", [])
        for map_func in self.map_fns:
            sample = map_func(sample)
        sample["json_data"] = json_data
        return sample


def _get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> "DatasetProcessor":
    r"""Return the corresponding dataset processor."""
    assert stage in ["sft", "ppo"]

    if stage == "sft" and not do_generate:
        dataset_processor_class = SupervisedDatasetProcessor
    else:
        dataset_processor_class = UnsupervisedDatasetProcessor

    return dataset_processor_class(
        template=template, tokenizer=tokenizer, processor=processor, data_args=data_args
    )
