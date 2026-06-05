# coding=utf-8

# Copyright (c) 2024 Tencent Inc. All Rights Reserved.
# Author: xiaotaoliu@tencent.com, nrwu@tencent.com
#
# 本文件来自 Megatron-LM：https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/utils.py
# ，仅仅包含了简单的 helper 函数，避免直接依赖具体版本的 Megatron。

from datetime import datetime
from typing import Tuple
import random

import torch

IGNORE_INDEX = -100


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_datetime(string):
    """Note that this call will sync across all ranks."""
    time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print_rank_0('[' + string + '] datetime: {} '.format(time_str))


def _get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
):
    """Build masks and position id for left to right model.

    Args:
        data (torch.Tensor): The data tenor that holds the tokens from the dataset

        eod_token (int): ID of the token to that is considered the EOD

        reset_position_ids (bool): Switch to reset the document position ID's

        reset_attention_mask (bool): Switch to reset the attention mask

        eod_mask_loss (bool): Switch to enable the EOD mask loss

    Returns:
        torch.Tensor : Attention mask needed to be used for Attention

        torch.Tensor : The mask used for loss value during training

        torch.Tensor : The position ID's of the token
    """
    seq_length = data.numel()

    attention_mask = torch.tril(torch.ones((seq_length, seq_length),
                                           device=data.device)).unsqueeze(0)

    # Loss mask.
    loss_mask = torch.ones(seq_length, dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Find indices where EOD token is.
        eod_index = position_ids[data == eod_token]
        # Detach indices from positions if going to modify positions.
        if reset_position_ids:
            eod_index = eod_index.clone()

        # Loop through EOD indices:
        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]
            # Mask attention loss.
            if reset_attention_mask:
                attention_mask[0, (i + 1):, :(i + 1)] = 0
            # Reset positions.
            if reset_position_ids:
                position_ids[(i + 1):] -= i + 1 - prev_index
                prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def _print_args(title, args):
    """Print arguments."""
    if args.rank == 0:
        print(f'------------------------ {title} ------------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {title} ---------------------', flush=True)


DEFAULT_TOOL_PROMPT = (
    "You have access to the following tools:\n{tool_text}"
    "Use the following format if using a tool:\n"
    "```\n"
    "Action: tool name (one of [{tool_names}])\n"
    "Action Input: the input to the tool, in a JSON format representing the kwargs "
    """(e.g. ```{{"input": "hello world", "num_beams": 5}}```)\n"""
    "```\n"
)


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:
    r"""
    Computes the real sequence length after truncation by the cutoff_len.
    """
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


def get_image_seqlen(config) -> int:
    r"""
    Computes the number of special tokens per image.
    """
    model_type = getattr(config, "model_type", None)
    if model_type == "llava":
        image_seqlen = (config.vision_config.image_size // config.vision_config.patch_size)**2
        if getattr(
            config, "vision_feature_select_strategy", "default"
        ) == "full":  # add [CLS] token
            image_seqlen += 1
    elif model_type == "paligemma":
        image_seqlen = config.vision_config.num_image_tokens
    else:
        image_seqlen = -1

    return image_seqlen


def get_patch_size(config, processor) -> int:
    r"""
    Computes the patch size of the vit.
    """
    patch_size = getattr(config.vision_config, "patch_size", getattr(processor, "patch_size", -1))
    return patch_size


def get_vision_feature_select_strategy(config, processor) -> int:
    r"""
    Get the vision_feature_select_strategy.
    """
    vision_feature_select_strategy = getattr(
        config, "vision_feature_select_strategy",
        getattr(processor, "vision_feature_select_strategy", "default")
    )
    return vision_feature_select_strategy


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def get_iterator(dataloader, dataloader_type="cyclic"):
    """Return dataset iterator."""
    from megatron.core.rerun_state_machine import RerunDataIterator

    if dataloader is None:
        return dataloader
    if dataloader_type == "single":
        return RerunDataIterator(iter(dataloader))
    elif dataloader_type == "cyclic":
        return RerunDataIterator(iter(cyclic_iter(dataloader)))
    else:
        raise RuntimeError("unexpected dataloader type")


def random_pad_list(lst, pad_len, ban_token_ids=None):
    assert pad_len >= 0, f'maybe max_seq_len calc wrong {pad_len}'
    if pad_len == 0:
        return lst
    else:
        if ban_token_ids is not None:
            filtered_lst = [x for x in lst if x not in ban_token_ids]
            padding = random.choices(filtered_lst, k=pad_len)
        else:
            padding = random.choices(lst, k=pad_len)
        return lst + padding
