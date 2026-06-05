# coding=utf-8
# copyright (c) 2024 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

from functools import partial
import os
import sys
import json

import torch
from torch import Tensor

from megatron.training import get_args
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core import mpu
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.transformer.spec_utils import import_module
from megatron.training import pretrain
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.training.utils import get_ltor_masks_and_position_ids

from gpatch.core.utils import split_data_cp_rank
from gpatch.training.arguments import gpatch_extra_args
from gpatch.training.utils import print_with_rank_and_datetime
from gpatch.training.v3.default_model_provider import default_infer_dpo_model_provider
from gpatch.patch_mcore import init_gpatch_for_mcore

from megatron_datasets.args import parse_dataset_config

from tasks.dpo.dpo_order_dataset import build_train_valid_test_datasets

# 默认调用这个 model provider，如果有什么特定参数需要修改再重写 model_provider
model_provider = default_infer_dpo_model_provider


def get_batch(data_iterator):
    args = get_args()
    keys = ['input_ids', 'labels']
    datatype = torch.int64

    # Broadcast data.
    src_json = None
    domain_name = None
    jsonl_file = None
    if data_iterator is not None:
        data = next(data_iterator)

        assert args.micro_batch_size % 2 == 0
        assert torch.all(data['is_chosen'][0::2]).item()
        assert torch.all(data['is_rejected'][1::2]).item()
        new_data = {}
        for k, v in data.items():
            if k not in ['src_json', 'domain_name', 'jsonl_file']:
                new_v = torch.cat((v[0::2], v[1::2]), dim=0)
            else:
                new_v = v[0::2] + v[1::2]
            new_data[k] = new_v
        data = new_data
        src_json = data['src_json']
        domain_name = data['domain_name']
        jsonl_file = data['jsonl_file']
    else:
        data = None

    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['input_ids'].long()
    labels = data_b['labels'].long()
    labels = labels.contiguous()
    tokens = tokens.contiguous()

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        labels, -100, args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss
    )
    loss_mask_full = loss_mask.detach()

    if args.context_parallel_size > 1:
        tokens = split_data_cp_rank(tokens, mpu.get_context_parallel_world_size(), 1)
        labels = split_data_cp_rank(labels, mpu.get_context_parallel_world_size(), 1)
        loss_mask = split_data_cp_rank(loss_mask, mpu.get_context_parallel_world_size(), 1)
        attention_mask = split_data_cp_rank(
            attention_mask, mpu.get_context_parallel_world_size(), 2
        )
        position_ids = split_data_cp_rank(position_ids, mpu.get_context_parallel_world_size(), 1)

    return tokens, labels, loss_mask, loss_mask_full, attention_mask, position_ids, src_json, domain_name, jsonl_file


def loss_func(loss_mask: torch.Tensor, metrics, output_tensor: torch.Tensor):
    args = get_args()
    loss = output_tensor.mean()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    metrics['dpo-metrics/loss'] = loss
    for k, v in metrics.items():
        averaged = average_losses_across_data_parallel_group([v])
        print(f"trace metrics {k=} {averaged} {averaged.shape}")
        metrics[k] = averaged.squeeze()
    return loss * args.context_parallel_size, metrics


def write_res_to_file(src_json, output_tensor, domain_name, jsonl_file, args):
    src_json_list = []
    for one_src in src_json:
        src_json_list.append(json.loads(one_src))
    mbs = len(src_json_list)
    reward_margin = output_tensor.tolist()
    assert mbs % 2 == 0, f'mbs must a multiple of 2'
    assert len(reward_margin) == mbs, f'the margin output must be equal mbs'

    for i in range(mbs // 2):
        assert src_json[i] == src_json[i + mbs // 2], 'chosen sample must be equal rejected smaple'
        assert domain_name[i] == domain_name[i + mbs // 2]
        assert jsonl_file[i] == jsonl_file[i + mbs // 2]
        if args.dpo_model_using != 'both':
            if 'ref-model-logps' not in src_json_list[i]:
                src_json_list[i]['ref-model-logps'] = {}
            src_json_list[i]['ref-model-logps']['chosen'] = reward_margin[i]
            src_json_list[i]['ref-model-logps']['rejected'] = reward_margin[i + mbs // 2]
        else:
            if 'modpo-chosen-margin' not in src_json_list[i]:
                src_json_list[i]['modpo-chosen-margin'] = {}
            if 'modpo-rejected-margin' not in src_json_list[i]:
                src_json_list[i]['modpo-rejected-margin'] = {}
            src_json_list[i]['modpo-chosen-margin'][args.dpo_gen_margin_key] = reward_margin[i]
            src_json_list[i]['modpo-rejected-margin'][args.dpo_gen_margin_key
                                                     ] = reward_margin[i + mbs // 2]

        filepath = os.path.join(args.dpo_gen_margin_path, domain_name[i])
        if not os.path.exists(filepath):
            os.makedirs(filepath, exist_ok=True)

        filename = os.path.join(filepath, jsonl_file[i])
        with open(filename, "a", encoding="utf-8") as file:
            json.dump(src_json_list[i], file, ensure_ascii=False)
            file.write("\n")


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    assert args.dpo
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, loss_mask_full, attention_mask, position_ids, src_json, domain_name, jsonl_file = get_batch(
        data_iterator
    )
    timers('batch-generator').stop()
    output_tensor, metrics = model(
        tokens,
        position_ids,
        loss_mask_full,
        attention_mask,
        labels=labels,
    )

    # save the result in jsonl file
    if src_json is not None and mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank(
    ) == 0 and mpu.get_context_parallel_rank() == 0:
        write_res_to_file(src_json, output_tensor, domain_name, jsonl_file, args)

    return output_tensor, partial(loss_func, loss_mask, metrics)


def train_valid_test_datasets_provider(train_val_test_num_samples=None):
    """Build train, valid, and test datasets."""
    args = get_args()
    tokenizer = get_tokenizer()

    if not getattr(args, 'px_parsed_dataset_config', False):
        parse_dataset_config(args)
        args.px_parsed_dataset_config = True

    if args.dpo_model_using == 'both':
        assert args.dpo_gen_margin_key is not None
    assert args.dpo_gen_margin_path is not None
    if not os.path.exists(args.dpo_gen_margin_path) and torch.distributed.get_rank() == 0:
        os.makedirs(args.dpo_gen_margin_path)

    print_rank_0('> building train, validation, and test datasets ...')
    if args.model_arch in ["qwen2-72b", "qwen2.5-1.5b"]:
        prompt_format = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        eos_token = tokenizer._tokenizer.eos_token
    elif args.model_arch in ["dsr1-distill-qwen2.5-32b"]:
        prompt_format = "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>{input}<｜Assistant｜>"
        eos_token = tokenizer._tokenizer.eos_token
    else:
        prompt_format = None
        eos_token = None

    print_rank_0(f"building dataset with prompt_format {prompt_format} eos_token {eos_token}")
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        args,
        tokenizer,
        prompt_format=prompt_format,
        eos_token=eos_token,
    )
    from megatron_datasets.utils import get_iterator
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.micro_batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True
    )
    train_dl = get_iterator(train_dl, dataloader_type='cyclic')
    if valid_ds is None:
        valid_dl = None
    else:
        valid_dl = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=args.micro_batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True
        )
        valid_dl = get_iterator(valid_dl, dataloader_type='cyclic')

    if test_ds is None:
        test_dl = None
    else:
        test_dl = torch.utils.data.DataLoader(
            test_ds,
            batch_size=args.micro_batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True
        )
        test_dl = get_iterator(test_dl, dataloader_type='cyclic')

    print_rank_0(f"> finished creating datasets ...")
    return train_dl, valid_dl, test_dl


if __name__ == "__main__":
    init_gpatch_for_mcore()
    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=gpatch_extra_args,
    )
