# coding=utf-8
# copyright (c) 2024 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

from functools import partial
import os
import sys

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

from gpatch.training.arguments import gpatch_extra_args
from gpatch.core.utils import split_data_cp_rank
from gpatch.training.utils import print_with_rank_and_datetime
from gpatch.training.v3.default_model_provider import default_dpo_model_provider
from gpatch.patch_mcore import init_gpatch_for_mcore

from megatron_datasets.args import parse_dataset_config

from tasks.dpo.dpo_dataset import (
    build_train_valid_test_datasets,
    update_epoch_and_line,
    DpoDataCollator,
)

# 默认调用这个 model provider，如果有什么特定参数需要修改再重写 model_provider
model_provider = default_dpo_model_provider


def get_batch(data_iterator):
    args = get_args()
    keys = ['input_ids', 'labels']
    if len(args.dpo_margin_keys) > 0:
        keys = ['input_ids', 'labels', 'margin', 'w0']
    if args.dpo_model_using != 'both':
        keys.append('ref_logps')
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
        assert args.micro_batch_size % 2 == 0
        assert torch.all(data['is_chosen'][0::2]).item()
        assert torch.all(data['is_rejected'][1::2]).item()
        new_data = {}
        for k, v in data.items():
            new_v = torch.cat((v[0::2], v[1::2]), dim=0)
            new_data[k] = new_v
        data = new_data
    else:
        data = None

    update_epoch_and_line(args.train_data_consuming_progresses, torch.distributed.get_rank(), data)
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['input_ids'].long()
    labels = data_b['labels'].long()
    w0_weights = None
    reward_margins = None
    if len(args.dpo_margin_keys) > 0:
        w0_weights = (data_b['w0'].long() / 1000.0).to(torch.float)
        reward_margins = (data_b['margin'].long() / 1000.0).to(torch.float)
    ref_logps = None
    if args.dpo_model_using != 'both':
        ref_logps = (data_b['ref_logps'].long() / 1000.0).to(torch.float)
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
        # 下面其实都是 none，先 assert 出去，如果不是 None再做处理, 因为 seq_dim 不确定是1还是2
        assert w0_weights is None
        assert reward_margins is None
        # w0_weights = get_tensosplit_data_cp_rankr_on_this_cp_rank(w0_weights, mpu.get_context_parallel_world_size(), 1)
        # reward_margins = get_tensor_on_this_cp_rank(reward_margins, mpu.get_context_parallel_world_size(), 1)

    if os.environ.get("PX_DEBUG_TRAIN_LOG", "0") == "1":
        tokens_non_pad_lengths = (tokens != get_tokenizer().pad_token_id).sum(dim=1)
        labels_non_pad_lengths = (labels != -100).sum(dim=1)
        first_non_pad_indices = (labels != -100).max(dim=1).indices
        tokens_sum = [v[:tokens_non_pad_lengths[i]].sum() for i, v in enumerate(tokens)]
        labels_sum = [
            v[first_non_pad_indices[i]:first_non_pad_indices[i] + labels_non_pad_lengths[i]].sum()
            for i, v in enumerate(labels)
        ]
        real_tokens_ = [tokens[i][:tokens_non_pad_lengths[i]] for i in range(tokens.shape[0])]
        real_labels_ = [
            v[first_non_pad_indices[i]:first_non_pad_indices[i] + labels_non_pad_lengths[i]]
            for i, v in enumerate(labels)
        ]
        print(
            f"trace input {torch.distributed.get_rank()} {tokens_non_pad_lengths=} {labels_non_pad_lengths=} sum {tokens_sum} {labels_sum} {tokens.shape}"
        )

    return tokens, labels, loss_mask, loss_mask_full, attention_mask, position_ids, w0_weights, reward_margins, ref_logps


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
        metrics[k] = averaged
    return loss * args.context_parallel_size, metrics


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    assert args.dpo
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, loss_mask_full, attention_mask, position_ids, w0_weights, reward_margins, ref_logps = get_batch(
        data_iterator
    )
    timers('batch-generator').stop()

    output_tensor, metrics = model(
        tokens,
        position_ids,
        loss_mask_full,
        attention_mask,
        labels=labels,
        w0_weights=w0_weights,
        reward_margins=reward_margins,
        ref_logps=ref_logps,
    )
    return output_tensor, partial(loss_func, loss_mask, metrics)


def train_valid_test_datasets_provider(train_val_test_num_samples=None):
    """Build train, valid, and test datasets."""
    args = get_args()
    tokenizer = get_tokenizer()

    if not getattr(args, 'px_parsed_dataset_config', False):
        parse_dataset_config(args)
        args.px_parsed_dataset_config = True

    print_rank_0('> building train, validation, and test datasets ...')
    print_rank_0(f"You should define your own prompt_format according to your model")
    #TODO: 用户需要根据自己的 model / 任务确定你的 prompt_format
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
        rank=torch.distributed.get_rank(),
        dp_rank=mpu.get_data_parallel_rank(),
        dp_size=mpu.get_data_parallel_world_size(),
        prompt_format=prompt_format,
        eos_token=eos_token,
        only_using_policy=args.dpo_model_using != 'both',
    )

    if args.px_inputs_pad_to_longest:
        print_rank_0("train with dynamic length")
        collate_fn = DpoDataCollator(
            tokenizer=tokenizer,
            seq_len=args.seq_length,
            train_with_dynamic_len=args.px_inputs_pad_to_longest,
            pad_to_multiple_of=args.px_pad_to_multiple_of
        )
        train_dl, valid_dl, test_dl = create_data_loader([train_ds, valid_ds, test_ds], collate_fn)
        print_rank_0(f"> finished creating data loader ...")
        return train_dl, valid_dl, test_dl
    else:
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


def create_data_loader(datasets, collate_fn):
    args = get_args()
    data_loaders = []
    for dataset in datasets:
        if dataset is None:
            data_loaders.append(None)
            continue

        assert isinstance(dataset, torch.utils.data.IterableDataset)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.micro_batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn
        )
        data_loaders.append(data_loader)
    return data_loaders


def set_return_dataloder_func(args, train_valid_test_datasets_provider):
    if args.px_inputs_pad_to_longest:
        setattr(train_valid_test_datasets_provider, "return_dataloaders", True)


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
