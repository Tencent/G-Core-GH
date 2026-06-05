# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# guanyouhe@tencent.com

import os
import copy
from dataclasses import fields
from functools import partial
from packaging.version import Version

import torch

from megatron.core import mpu
from megatron.core import package_info
from megatron.core.enums import ModelType
from megatron.training import get_args, get_tokenizer, print_rank_0, pretrain, get_timers
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron.core import tensor_parallel
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.core import ModelParallelConfig
from megatron.core.transformer import TransformerConfig

from mbridge import AutoBridge

from megatron_datasets.args import parse_dataset_config
from megatron_datasets.mega_indexed_jsonl_dataset_v3 import update_consumed
from megatron_datasets.internvl_dataset import IMG_CONTEXT_TOKEN

from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.training.arguments import gpatch_extra_args
from gpatch.core.transformer.transformer_config import GpatchTransformerConfig

mcore_version = Version(package_info.__version__)


def merge_config(hf_config, mg_config):
    assert hf_config.num_layers == mg_config.num_layers
    config_merged = copy.deepcopy(mg_config)
    hf_fields = {e.name for e in fields(hf_config)}
    mg_fields = {e.name for e in fields(TransformerConfig)}
    # config fields TransformerConfig - TransformerConfig
    diff = hf_fields - mg_fields
    # generally, use parallel config from megatron config, use model config from hf config
    for f in diff:
        setattr(config_merged, f, getattr(hf_config, f))

    return config_merged


def model_provider(
    pre_process=True,
    post_process=True,
    add_encoder=True,
    add_decoder=True,
    parallel_output=True,
):
    args = get_args()
    assert args.context_parallel_size == 1, "only support cp=1 now"
    assert not args.dpo, f'not support dpo now'

    bridge = AutoBridge.from_pretrained(
        args.hf_model_path, trust_remote_code=True, make_vocab_size_divisible_by=256
    )
    config = core_transformer_config_from_args(args, GpatchTransformerConfig)
    config = merge_config(bridge.config, config)
    bridge.config = config
    model = bridge._model_provider([])(pre_process=pre_process, post_process=post_process)

    model.model_type = ModelType.encoder_and_decoder
    model.freeze(
        freeze_language_model=args.mm_freeze_llm,
        freeze_vision_model=args.mm_freeze_vision_encoder,
        freeze_vision_projection=args.mm_freeze_projector,
    )
    # attach bridge
    model.bridge = bridge

    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    # NOTE(guanyouhe): 这里直接broadcast train_data_consuming_progresses，可以消除保存ckpt的warning
    update_consumed(args.train_data_consuming_progresses, torch.distributed.get_rank(), data)

    keys = ["input_ids", "labels", "has_imgs"]
    data_b = tensor_parallel.broadcast_data(keys, data, torch.int64)
    tokens = data_b["input_ids"].long().contiguous()
    labels = data_b["labels"].long().contiguous()
    has_imgs = data_b["has_imgs"].long().contiguous().tolist()[0]

    if has_imgs > 0:
        keys = ["pixel_values"]
        data_b = tensor_parallel.broadcast_data(keys, data, torch.bfloat16)
        imgs = data_b["pixel_values"].type(torch.bfloat16).contiguous()
    else:
        imgs = torch.tensor([], dtype=torch.bfloat16, device=tokens.device)

    attention_mask = None
    _, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        labels,
        -100,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    return (
        tokens,
        labels,
        loss_mask,
        position_ids,
        imgs,
        attention_mask,
    )


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    loss = torch.stack([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan().any(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    averaged_loss = average_losses_across_data_parallel_group(loss)
    averaged_loss = averaged_loss[0] / averaged_loss[1]

    return (loss[0] / loss[1]) * args.context_parallel_size, {"lm loss": averaged_loss}


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()
    # Get the batch.
    timers("batch-generator", log_level=1).start()
    (
        tokens,
        labels,
        loss_mask,
        position_ids,
        imgs,
        attention_mask,
    ) = get_batch(data_iterator)
    timers("batch-generator").stop()

    timers("model-forward-only", log_level=1).start()
    image_token_id = get_tokenizer()._tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    output_tensor = model(
        images=imgs,
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
        image_token_index=image_token_id,
    )

    timers("model-forward-only").stop()

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_data_iter_provider(train_val_test_num_samples=None):
    """Build multimodal train, validation and test dataloaders."""
    args = get_args()
    if args.data_path is None:
        parse_dataset_config(args)
    # tp-rank != 0 返回空，但在use_grpo时，每个tp-rank都会创建dataloader
    if not args.use_grpo and mpu.get_tensor_model_parallel_rank() != 0:
        return None, None, None
    tokenizer = get_tokenizer()

    print_rank_0('> building train, validation, and test dataloader ...')
    from megatron_datasets.internvl_dataset import build_train_valid_test_data_iter
    train_iter, valid_iter, test_iter = build_train_valid_test_data_iter(
        args,
        tokenizer,
        rank=torch.distributed.get_rank(),
        dp_rank=mpu.get_data_parallel_rank(),
        dp_size=mpu.get_data_parallel_world_size(),
        cp_rank=mpu.get_context_parallel_rank(),
        cp_size=mpu.get_context_parallel_world_size(),
        is_dpo=args.dpo,
    )

    print(
        f"> world size {mpu.get_data_parallel_world_size()} rank {mpu.get_data_parallel_rank()} "
        f"finished creating dataloader ..."
    )
    return train_iter, valid_iter, test_iter


def add_extra_args(parser):
    """Extra arguments."""
    parser = gpatch_extra_args(parser)

    group = parser.add_argument_group(title='InternVL arguments')

    group.add_argument("--internvl-template", type=str, default="internvl2_5", help="")
    group.add_argument("--tarfile-path", type=str, default="/", help="")
    group.add_argument("--downsample-ratio", type=float, default=0.5, help="")
    group.add_argument("--lmdb-port", type=int, default=None, help="lmdb server port")
    group.add_argument("--mask-history", action='store_true', help="多轮对话只取最后一轮对话为label")
    group.add_argument("--initializer-factor", type=float, default=0.1, help="")
    group.add_argument("--drop-path-rate", type=float, default=0.1, help="")
    group.add_argument("--max-num", type=int, default=12, help="图片切分的最大数量")
    group.add_argument("--processor-path", type=str, default=None, help="")
    return parser


if __name__ == "__main__":
    # 每个tp-rank都要运行parse_dataset_config
    init_gpatch_for_mcore()
    setattr(train_valid_test_data_iter_provider, "is_distributed", True)
    print(f"{mcore_version=} {Version('0.13.0')} {mcore_version < Version('0.13.0')}")
    if mcore_version < Version("0.13.0"):
        extra_args = {}
    else:
        from megatron.training import inprocess_restart
        pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
        extra_args = {"store": store}

    pretrain(
        train_valid_test_data_iter_provider,
        model_provider,
        ModelType.encoder_and_decoder,
        forward_step,
        extra_args_provider=add_extra_args,
        **extra_args,
    )
