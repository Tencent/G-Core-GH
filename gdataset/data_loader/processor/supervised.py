from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import torch
from llamafactory.data.processor.supervised import (
    SupervisedDatasetProcessor as LlamaFactorySupervisedDatasetProcessor,
)
from llamafactory.extras.constants import IGNORE_INDEX


@dataclass
class SupervisedDatasetProcessor(LlamaFactorySupervisedDatasetProcessor):

    # for llama factory dataset
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".
                    format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    # for gcore dataset
    def preprocess_example(self, example):
        if len(example["_prompt"]) % 2 != 1 or len(example["_response"]) != 1:
            assert False

        model_inputs = {}
        input_ids, labels = self._encode_data_example(
            prompt=example["_prompt"],
            response=example["_response"],
            system=example["_system"],
            tools=example["_tools"],
            images=example["_images"] or [],
            videos=example["_videos"] or [],
            audios=example["_audios"] or [],
        )
        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(input_ids)
        input_ids_len = len(input_ids)
        model_inputs["tokenizer_len"] = torch.tensor(input_ids_len, dtype=torch.int64)
        # left shift lable
        model_inputs["labels"] = labels[1:] + [IGNORE_INDEX]
        model_inputs["images"] = example["_images"]
        model_inputs["videos"] = example["_videos"]
        model_inputs["audios"] = example["_audios"]
        return model_inputs
