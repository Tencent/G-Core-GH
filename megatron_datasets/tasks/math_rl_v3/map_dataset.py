import numpy as np
import io
import copy
import json
import torch
try:
    from megatron import get_args
except:
    from megatron.training import get_args
from datasets import load_dataset
from tqdm import tqdm

from megatron.training import get_tokenizer


class PogDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_padding_length):
        self.tokenizer = get_tokenizer()
        self.IGNORE_INDEX = -100
        self.pad_token_id = self.tokenizer._tokenizer.pad_token_id
        self.max_padding_length = max_padding_length
        print(f"PogDataset path {path}")
        train_dataset = self.preprocess(path[0])
        self.input_ids = np.array(train_dataset['input_ids'])
        self.labels = np.array(train_dataset['labels'])
        self.samples = []
        for inputs, labels in tqdm(zip(self.input_ids, self.labels)):
            self.samples.append([inputs, labels])
        print('  >> total number of samples: {}'.format(len(self.samples)))

    def preprocess(self, path):
        data = open(path, 'r', encoding='utf-8')
        data = data.readlines()
        prompt_template = "###{Human}\n### Response:"
        # prompt_template = "<|begin_of_text|>instruction:\n{instruction}\ninput:\n{input}\noutput:\n"

        input_ids = []
        labels = []
        cnt = 0
        for example in data:
            example = json.loads(example)
            if 'Human' not in example or example['Human'].strip() == "":
                continue

            if 'Assistant' not in example or example['Assistant'].strip() == "":
                continue
            source = prompt_template.format_map(example)
            target = example['Assistant']
            # target = example['output']
            source = source
            full = source + f"{target}{self.tokenizer._tokenizer.eos_token}"
            source = self.tokenizer.tokenize(source)
            full = self.tokenizer.tokenize(full)
            label = copy.deepcopy(full)
            #TODO(jeffhong): 加上下面这个会导致 label 在 eos token 的地方变成 -100，
            # 后面偏移的时候会导致缺失一位有效位, 比如，记 eos_token_id = 99, pad_token_id = 100
            # tokens:   0, 1, 2,    3,  99, pad_token, labels: 0, 1, 2, 3, -100, -100
            # 偏移labels:1, 2, 3, -100, -100
            # 3 本来应该预测 eos_token， 加上下面这行变成了预测 -100
            # label[-1] = self.IGNORE_INDEX

            #TODO(jintaosu): fix bug
            # for t1, t2 in zip(source, full):
            #     assert t1 == t2, "The user input_ids are not a prefix of the full input_ids! Please check the template."

            if len(full) >= self.max_padding_length:
                full = full[:self.max_padding_length]
                label = label[:self.max_padding_length]

            if self.max_padding_length > len(full):
                label = label + [self.IGNORE_INDEX] * (self.max_padding_length - len(label))
                full = full + [self.pad_token_id] * (self.max_padding_length - len(full))

            # NOTE: in get_batch_on_this_tp_rank_original, tokens use [:-1] and labels use [1:]
            # we add an extra token to use the old api
            # TODO: update get_batch_on_this_tp_rank_original and replace the following line with
            # label = [self.IGNORE_INDEX] * (len(source) - 1) + full[len(input_ids):] + [self.IGNORE_INDEX]
            full = full + [self.pad_token_id]
            label = [self.IGNORE_INDEX] * len(source) + label[len(source):] + [self.IGNORE_INDEX]

            flag = True
            for i in range(len(label)):
                if label[i] != self.IGNORE_INDEX:
                    flag = False
                    break
            if flag:
                continue
            cnt += 1
            # if cnt > 1000:
            #     break
            input_ids.append(full)
            labels.append(label)

        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        return self.gpt_convert_example_to_feature(raw_sample)

    def gpt_convert_example_to_feature(self, sample):
        """
        Convert a single sample containing input_id, label and loss_mask into a format suitable for GPT training.
        """
        input_ids, labels = sample
        train_sample = {'input_ids': input_ids, 'labels': labels}

        return train_sample


def build_pretrain_dataset_from_original(args):
    print(f"build_pretrain_dataset_from_original {args.data_path}")

    train_dataset = PogDataset(args.data_path, args.seq_length)
    return train_dataset, None, None
