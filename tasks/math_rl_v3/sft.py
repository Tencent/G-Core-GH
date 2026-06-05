# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

import os
import torch
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
from megatron.training.utils import get_ltor_masks_and_position_ids

from gpatch.training.arguments import gpatch_extra_args
from gpatch.core.utils import split_data_cp_rank
from gpatch.patch_mcore import init_gpatch_for_mcore
from megatron_datasets.args import parse_dataset_config
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

    group = parser.add_argument_group(title='demo extra args')
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
        raise NotImplementedError("somgthing wrong")

    if mcore_version < Version("0.14.0"):
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            labels, -100, args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss
        )
    else:
        # 这个是 dev branch 用到的
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            labels, -100, 0, args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss,
            False
        )
    if args.context_parallel_size > 1:
        tokens = split_data_cp_rank(tokens, mpu.get_context_parallel_world_size(), 1)
        labels = split_data_cp_rank(labels, mpu.get_context_parallel_world_size(), 1)
        loss_mask = split_data_cp_rank(loss_mask, mpu.get_context_parallel_world_size(), 1)
        attention_mask = split_data_cp_rank(
            attention_mask, mpu.get_context_parallel_world_size(), 2
        )
        position_ids = split_data_cp_rank(position_ids, mpu.get_context_parallel_world_size(), 1)

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
            f"trace input {torch.distributed.get_rank()} {tokens_non_pad_lengths=} {first_non_pad_indices=} "
            f"{labels_non_pad_lengths=} sum {tokens_sum} {labels_sum} {tokens.shape}"
        )

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func_lt_0_13(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(rerun_state_machine.is_spiky_loss, threshold=SPIKY_LOSS_PERC),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )
    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)

    # TODO
    # 1. nv mlm 0.12 改了这个行为，之前的代码都要修改
    # 2. nv mlm 0.12 te 1.13 mbs>1 cp>1 的 grad 不对，原因不明
    if Version(package_info.__version__) < Version("0.12.1"):
        bwd_loss = loss[0] * args.context_parallel_size
    else:
        bwd_loss = loss[0].clone()
    return (
        bwd_loss,
        local_num_tokens,
        {
            'lm loss': (reporting_loss[0], reporting_loss[1])
        },
    )


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    # mcore 不在这里做 cp 的 allreduce, 因为 megatron 把 allreduce 环节放到了 train_step 外面
    # 但是如果训练数据里 pad token 太多，某一个 cp 全是 pad 出来，出来的 loss 为 0.0，num_tokens 也为 0.0
    # 这样算出来的梯度会变成 nan，导致检查不过
    total_tokens = loss_mask.sum()
    loss = torch.cat([loss.view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(
            loss, group=mpu.get_context_parallel_group(), op=torch.distributed.ReduceOp.AVG
        )

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    reporting_loss = loss.clone().detach()
    local_num_tokens = loss[1].sum().clone().detach().to(torch.int)
    return (
        loss[0].clone(),
        local_num_tokens,
        {
            'lm loss': reporting_loss
        },
    )


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    with stimer:
        if mcore_version < Version("0.13.0") or args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            output_tensor = model(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )
    if mcore_version < Version("0.13.0"):
        return output_tensor, partial(loss_func_lt_0_13, loss_mask)
    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def train_valid_test_datasets_provider(train_val_test_num_samples=None):
    """Build train, valid, and test datasets."""
    args = get_args()
    tokenizer = get_tokenizer()

    print_rank_0('> building train, validation, and test datasets ...')

    if not getattr(args, 'px_parsed_dataset_config', False):
        parse_dataset_config(args)
        args.px_parsed_dataset_config = True
    if mpu.get_tensor_model_parallel_rank() != 0:
        return None, None, None
    '''
    # 下面 if 分支这样写是有历史原因的，不过后面建议不要再这样写，逐渐替换成在脚本的 args 里定义下面两个参数
        --px-apply-chat-template \
        --px-system-prompt 根据你的模型或者任务来定义 \
    # 比如 "qwen2.5-math-1.5b"
        --px-apply-chat-template \
        --px-system-prompt "Please reason step by step, and put your final answer within \\boxed{}." \
    '''
    if args.px_apply_chat_template:
        prompt_format = None
        eos_token = tokenizer._tokenizer.eos_token
    elif args.model_arch in ["qwen2-72b", "qwen1.5-moe"]:
        prompt_format = "<|im_start|>system\nyou are a helpful assistant<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
        eos_token = tokenizer._tokenizer.eos_token
    elif args.model_arch in ["qwen2.5-math-rm-72b", "qwen2.5-math-1.5b"]:
        prompt_format = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"
        eos_token = tokenizer._tokenizer.eos_token
    elif args.model_arch == "yi_9b":
        prompt_format = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"
        eos_token = None
    elif args.model_arch in ["qwen3", "qwen3-moe", "qwen3-next-moe"]:
        if torch.distributed.get_rank() == 0:
            print(f"define your own prompt_format according to your task")
        #TODO：使用 apply chat template
        prompt_format = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"
        eos_token = tokenizer._tokenizer.eos_token
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
        )
        assert isinstance(train_ds, torch.utils.data.IterableDataset)
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
        train_dl, valid_dl, test_dl = create_data_loader_with_collator(
            [train_ds, valid_ds, test_ds], collate_fn
        )
    else:
        if args.use_map_dataset:
            return train_ds, valid_ds, test_ds

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

    print_rank_0(f"> finished creating data loader ...")
    return train_dl, valid_dl, test_dl


def create_data_loader_with_collator(datasets, collate_fn):
    from megatron_datasets.utils import get_iterator
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
        data_loaders.append(get_iterator(data_loader, dataloader_type='cyclic'))
    return data_loaders


if __name__ == "__main__":
    init_gpatch_for_mcore()

    # Temporary for transition to core datasetss
    train_valid_test_datasets_provider.is_distributed = True

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
        args_defaults={},
        extra_args_provider=add_demo_extra_args,
        **extra_args,
    )
