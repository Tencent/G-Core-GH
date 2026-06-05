# coding=utf-8
# copyright (c) 2026 tencent inc. all rights reserved.
# xiaotaoliu@tencent.com, nrwu@tencent.com

import os
import sys
import copy
import torch
from typing import Optional, Union, List
from dataclasses import fields
from functools import partial
from packaging.version import Version

import torch.distributed
from transformers import AutoConfig
from megatron.training import get_args
try:
    from megatron.training import inprocess_restart
except ImportError:
    inprocess_restart = None
from megatron.core import parallel_state
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import package_info
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.training import pretrain
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import StragglerDetector
from megatron.core import tensor_parallel
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron.core.transformer.multi_token_prediction import get_mtp_ranks
from megatron.core.extended_models.welm_v4.oe_embedding import compute_welm_v4_ngram_hashes

from gpatch.training.arguments import gpatch_extra_args
from gpatch.patch_mcore import init_gpatch_for_mcore
from megatron_datasets.args import parse_dataset_config

from mbridge import AutoBridge

try:
    from megatron.post_training.arguments import add_modelopt_args

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()
# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10
mcore_version = Version(package_info.__version__)

from gpatch.core.utils import split_data_cp_rank, freeze_moe_router
from gpatch.training.utils import gpatch_core_transformer_config_from_args
from gpatch.core.transformer.transformer_config import GpatchTransformerConfig
from megatron.core.extended_models.welm_v4.transformer_config import WelmV4TransformerConfig
from megatron.core.extended_models.welm_v4.gpt_model import WelmV4Model
from tasks.welm_v4.welmv4_sft_dataset import (
    build_train_valid_test_datasets,
    update_epoch_and_line,
    SftDataCollator,
)
from tasks.welm_v4.utils import merge_config

mcore_version = Version(package_info.__version__)


def add_demo_extra_args(parser):
    """Extra arguments."""
    parser = gpatch_extra_args(parser)
    if has_nvidia_modelopt:
        parser = add_modelopt_args(parser)

    group = parser.add_argument_group(title='demo extra args')
    group.add_argument("--use-map-dataset", action='store_true', help="map dataset")

    # 在 args 临时多加一个参数
    group.add_argument(
        "--oe-grams",
        nargs='*',
        type=int,
        default=None,
        help="The ngrams to use for OE ngram embedding"
    )
    return parser


def model_provider(
    pre_process=True,
    post_process=True,
    vp_stage: Optional[int] = None,
    config=None,
    pg_collection=None
) -> Union[WelmV4Model]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[WelmV4Model]: The returned model
    """
    args = get_args()
    bridge = AutoBridge.from_pretrained(args.hf_model_path, trust_remote_code=True)
    config = gpatch_core_transformer_config_from_args(args, GpatchTransformerConfig)
    config = merge_config(bridge.config, config)

    hf_config = bridge.hf_config
    bridge.config = config
    model = bridge._model_provider([])(pre_process=pre_process, post_process=post_process)
    if args.freeze_moe_router:
        freeze_moe_router(model)

    assert hf_config.vocab_size == model.vocab_size, f"vocab size mismatch: {hf_config.vocab_size} vs {model.vocab_size}"
    assert args.padded_vocab_size == model.vocab_size, f"vocab size mismatch: {args.padded_vocab_size} vs {model.vocab_size}"
    model.bridge = bridge

    args.oe_grams = bridge.config.oe_grams

    print_rank_0(
        f"Model create with config {bridge.config.__dict__} {bridge.config.kv_mirror_imitated_layers}"
    )
    return model


def get_batch(data_iterator):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    if (not parallel_state.is_pipeline_first_stage(ignore_virtual=True)
       ) and (not parallel_state.is_pipeline_last_stage(ignore_virtual=True)):
        return None, None, None, None, None, None, None

    args = get_args()
    tokenizer = get_tokenizer()
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    if args.px_use_indexed_jsonl_dataset:
        update_epoch_and_line(
            args.train_data_consuming_progresses,
            torch.distributed.get_rank(),
            data,
        )

    keys_to_broadcast = ['input_ids', 'labels']
    datatype = torch.int64
    data_b = tensor_parallel.broadcast_data(keys_to_broadcast, data, datatype)
    tokens = data_b['input_ids'].long()
    labels = data_b['labels'].long()

    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        labels, -100, tokenizer._tokenizer.pad_token, args.reset_position_ids,
        args.reset_attention_mask, True, True
    )

    # 为避免 cp 切分后再去算 input_ids_ngram 存在边界问题，所以要提前处理好
    input_ids_ngram = None
    vocab_size = args.padded_vocab_size
    assert args.oe_grams is not None
    if args.oe_grams is not None and len(args.oe_grams) > 0:
        input_ids_ngram = compute_welm_v4_ngram_hashes(
            input_ids=tokens.clone(),
            vocab_size=vocab_size,
            oe_grams=args.oe_grams,
        )
        # [max(self.oe_grams) - 1, batch_size, seq_len]
        input_ids_ngram = torch.stack(input_ids_ngram, dim=0)
        if input_ids_ngram.dtype != torch.int64:
            input_ids_ngram = input_ids_ngram.to(torch.int64)

    if args.context_parallel_size > 1:
        tokens = split_data_cp_rank(tokens, mpu.get_context_parallel_world_size(), 1)
        attention_mask = split_data_cp_rank(
            attention_mask, mpu.get_context_parallel_world_size(), 2
        )
        position_ids = split_data_cp_rank(position_ids, mpu.get_context_parallel_world_size(), 1)
        loss_mask = split_data_cp_rank(loss_mask, mpu.get_context_parallel_world_size(), 1)
        labels = split_data_cp_rank(labels, mpu.get_context_parallel_world_size(), 1)

        # input_ids_ngram (num_ngrams, batch, seq_len)
        assert input_ids_ngram is not None
        input_ids_ngram = split_data_cp_rank(
            input_ids_ngram, mpu.get_context_parallel_world_size(), 2
        )

    return tokens, labels, loss_mask, attention_mask, position_ids, None, input_ids_ngram


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[WelmV4Model] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    loss = torch.cat([loss.view(1), num_tokens.view(1)])
    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(
            loss, group=mpu.get_context_parallel_group(), op=torch.distributed.ReduceOp.AVG
        )

    report = {'lm loss': loss.detach().clone()}

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

    local_num_tokens = loss[1].sum().clone().detach().to(torch.int)
    return loss[0].clone(), local_num_tokens, report


def forward_step(data_iterator, model: WelmV4Model, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (WelmV4Model): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids, _, input_ids_ngram = get_batch(
            data_iterator
        )
    timers('batch-generator').stop()

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            output_tensor = model(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
                loss_mask=loss_mask,
                packed_seq_params=None,
                input_ids_ngram=input_ids_ngram
            )

    return output_tensor, partial(loss_func, loss_mask, model=model)


def train_valid_test_datasets_provider(train_valid_test_num_samples=None):
    """Build train, valid, and test datasets."""
    args = get_args()
    tokenizer = get_tokenizer()

    hf_config = AutoConfig.from_pretrained(args.hf_model_path, trust_remote_code=True)
    print_rank_0('> building train, validation, and test datasets ...')

    if not getattr(args, 'px_parsed_dataset_config', False):
        parse_dataset_config(args)
        args.px_parsed_dataset_config = True
    if mpu.get_tensor_model_parallel_rank() != 0:
        return None, None, None

    prompt_format = None
    eos_token = tokenizer._tokenizer.eos_token

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
    else:
        raise Exception("should use px_use_indexed_jsonl_dataset")

    if args.px_inputs_pad_to_longest:
        assert args.px_smart_pad_by_truncated, f"px_inputs_pad_to_longest requires px_smart_pad_by_truncated"
        assert not args.use_map_dataset
        print_rank_0("train with dynamic length")
        collate_fn = SftDataCollator(
            tokenizer=tokenizer,
            seq_len=args.seq_length,
            train_with_dynamic_len=args.px_inputs_pad_to_longest,
            pad_to_multiple_of=args.px_pad_to_multiple_of,
            moe_pad_with_random_token=args.moe_pad_with_random_token,
            smart_pad_by_truncated=args.px_smart_pad_by_truncated,
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
    for di, dataset in enumerate(datasets):
        if dataset is None:
            data_loaders.append(None)
            continue
        assert isinstance(dataset, torch.utils.data.IterableDataset)

        extra_args = {}
        if di == 0:
            # 只让 train_dataset 设置
            extra_args["collate_fn"] = collate_fn
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.micro_batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            **extra_args,
        )
        data_loaders.append(get_iterator(data_loader, dataloader_type='cyclic'))
    return data_loaders


def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        config = gpatch_core_transformer_config_from_args(args)
        mtp_ranks = get_mtp_ranks(pp_ranks, config)
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks


if __name__ == "__main__":
    init_gpatch_for_mcore()

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True
    print(f"{mcore_version=}", flush=True)

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={},
        extra_args_provider=add_demo_extra_args,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
