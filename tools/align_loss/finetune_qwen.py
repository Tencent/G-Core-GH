# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

import os
import torch
import importlib
from typing import Optional
from functools import partial
from packaging.version import Version

from megatron.training import get_args
try:
    from megatron.training import inprocess_restart
except ImportError:
    inprocess_restart = None
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import package_info
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.utils import StragglerDetector
from megatron.core import tensor_parallel
from megatron.training.utils import get_ltor_masks_and_position_ids, average_losses_across_data_parallel_group

from gpatch.training.arguments import gpatch_extra_args
from gpatch.core.utils import split_data_cp_rank
from gpatch.patch_mcore import init_gpatch_for_mcore

from megatron_datasets.tasks.math_rl_v3.sft_dataset import (
    build_train_valid_test_datasets,
    update_epoch_and_line,
    SftDataCollator,
)

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_provider import model_provider as model_provider_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()
# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10
mcore_version = Version(package_info.__version__)


def add_demo_extra_args(parser):
    """Extra arguments."""
    parser = gpatch_extra_args(parser)
    if has_nvidia_modelopt:
        parser = add_modelopt_args(parser)

    group = parser.add_argument_group(title='align arguments')
    group.add_argument("--use-map-dataset", action='store_true', help="map dataset")
    return parser


def get_batch(data_iterator):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    # Items and their type.
    keys = ['input_ids', 'labels']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    # assert not args.packed_seq, 'not implemented yet'
    if args.px_use_indexed_jsonl_dataset:
        update_epoch_and_line(
            args.train_data_consuming_progresses, torch.distributed.get_rank(), data
        )
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['input_ids'].long()
    labels_ = data_b['labels'].long()
    if args.px_use_indexed_jsonl_dataset:
        labels = labels_.contiguous()
        tokens = tokens_.contiguous()
    elif args.use_map_dataset:
        labels = labels_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()
    else:
        raise NotImplementedError("error")

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        labels, -100, args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss
    )

    print(f"trace tokens {torch.distributed.get_rank()} {tokens.sum()} {labels.sum()}", flush=True)

    if args.context_parallel_size > 1:
        tokens = split_data_cp_rank(tokens, mpu.get_context_parallel_world_size(), 1)
        labels = split_data_cp_rank(labels, mpu.get_context_parallel_world_size(), 1)
        loss_mask = split_data_cp_rank(loss_mask, mpu.get_context_parallel_world_size(), 1)
        attention_mask = split_data_cp_rank(
            attention_mask, mpu.get_context_parallel_world_size(), 2
        )
        position_ids = split_data_cp_rank(position_ids, mpu.get_context_parallel_world_size(), 1)

    if os.environ.get("PX_DEBUG_TRAIN_LOG", "0") == "1":
        tokens_non_pad_lengths = (tokens != get_tokenizer()._tokenizer.pad_token_id).sum(dim=1)
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
            f"trace input {torch.distributed.get_rank()} {tokens_non_pad_lengths=} {first_non_pad_indices=} {labels_non_pad_lengths=} sum {tokens_sum} {labels_sum} {tokens.shape}"
        )

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    # 注意，要对齐 transformers 的 loss，需要用这个方式来计算 loss。保留这个 script
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )
    print(f"trace loss {torch.distributed.get_rank()} {loss=}", flush=True)
    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model: GPTModel):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples=None):
    """Build train, valid, and test datasets."""
    args = get_args()
    tokenizer = get_tokenizer()

    print_rank_0('> building train, validation, and test datasets ...')

    if args.model_arch in ["qwen2-72b", "qwen1.5-moe"]:
        prompt_format = "<|im_start|>system\nyou are a helpful assistant<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
        eos_token = tokenizer.eos_token
    elif args.model_arch in ["qwen2.5-math-rm-72b", "qwen2.5-math-1.5b"]:
        prompt_format = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"
        eos_token = tokenizer.eos_token
    elif args.model_arch == "yi_9b":
        prompt_format = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
        eos_token = None
    else:
        prompt_format = None
        eos_token = None
    print_rank_0(f"building dataset with prompt_format {prompt_format} eos_token {eos_token}")
    if args.px_use_indexed_jsonl_dataset:
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            args,
            tokenizer,
            rank=torch.distributed.get_rank(),
            dp_rank=mpu.get_data_parallel_rank(),
            dp_size=mpu.get_data_parallel_world_size(),
            prompt_format=prompt_format,
            eos_token=eos_token,
            custom_target_key=args.custom_dataset_target_key,
            custom_question_key=args.custom_dataset_question_key
        )
    elif args.use_map_dataset:
        from megatron_datasets.tasks.math_rl_v3.map_dataset import build_pretrain_dataset_from_original
        train_ds, valid_ds, test_ds = build_pretrain_dataset_from_original(args)
    else:
        raise Exception("should use px_use_indexed_jsonl_dataset")

    if args.px_inputs_pad_to_longest:
        assert not args.use_map_dataset
        print_rank_0("train with dynamic length")
        collate_fn = SftDataCollator(
            tokenizer=tokenizer,
            seq_len=args.seq_length,
            train_with_dynamic_len=args.px_inputs_pad_to_longest,
            pad_to_multiple_of=args.px_pad_to_multiple_of
        )
        train_dl, valid_dl, test_dl = create_data_loader([train_ds, valid_ds, test_ds], collate_fn)
        print_rank_0(f"> finished creating data loader ...")
        return train_dl, valid_dl, test_dl

    print_rank_0(f"> finished creating datasets ...")
    return train_ds, valid_ds, test_ds


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


# 这样的写法出现报错
# assert out._base is None, "counter-productive to free a view of another tensor
# 应该是默认值不对？
def model_provider(pre_process=True, post_process=True):
    args = get_args()
    if args.load_model_provider is not None:
        mod = importlib.import_module(args.load_model_provider)
        print(f"[Import] {mod=} {mod.model_provider=}")
        return mod.model_provider(pre_process, post_process)
    else:
        if mcore_version < Version("0.13.0"):
            from gpatch.training.v3.default_model_provider import default_sft_model_provider
        else:
            from gpatch.training.v3.default_model_provider_0_13 import default_sft_model_provider
        return default_sft_model_provider(pre_process, pre_process)


if __name__ == "__main__":
    init_gpatch_for_mcore()

    print(f"{mcore_version=} {Version('0.13.0')} {mcore_version < Version('0.13.0')}")
    if mcore_version < Version("0.13.0"):
        from gpatch.training.v3.default_model_provider import default_sft_model_provider
        extra_args = {}
    else:
        from gpatch.training.v3.default_model_provider_0_13 import default_sft_model_provider
        # Optionally enable inprocess restart on pretrain
        pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
        extra_args = {"store": store}

    model_provider = default_sft_model_provider

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_demo_extra_args,
        **extra_args,
    )
