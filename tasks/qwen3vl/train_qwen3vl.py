import copy
import os
from dataclasses import fields
from functools import partial
from typing import Optional

import torch
import torch.distributed
from packaging.version import Version

from megatron_datasets.args import parse_dataset_config
from megatron_datasets.mega_indexed_jsonl_dataset_v3 import update_consumed

from mbridge import AutoBridge
from megatron.core import mpu, package_info, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    pretrain,
    print_rank_0,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import average_losses_across_data_parallel_group

from gpatch.core.device_type import is_wxacc2
from gpatch.core.transformer.transformer_config import GpatchTransformerConfig
from gpatch.core.utils import split_data_cp_rank
from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.training.arguments import gpatch_extra_args

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


def freeze_moe_router(model):
    for layer in model.decoder.layers:
        if hasattr(layer.mlp, "router"):
            if hasattr(layer.mlp.router, "weight"):
                layer.mlp.router.weight.requires_grad = False
            if hasattr(layer.mlp.router, "bias") and layer.mlp.router.bias is not None:
                layer.mlp.router.bias.requires_grad = False


def model_provider(
    pre_process=True,
    post_process=True,
    add_encoder=True,
    add_decoder=True,
    vp_stage: Optional[int] = None,
    config=None,
    pg_collection=None,
):
    args = get_args()
    assert not args.dpo, f'not support dpo now'

    bridge = AutoBridge.from_pretrained(args.hf_model_path)
    config = core_transformer_config_from_args(args, GpatchTransformerConfig)
    config = merge_config(bridge.config, config)
    bridge.config = config
    model = bridge._model_provider([])(pre_process=pre_process, post_process=post_process)

    model.freeze(
        freeze_language_model=args.mm_freeze_llm,
        freeze_vision_model=args.mm_freeze_vision_encoder,
        freeze_vision_projection=args.mm_freeze_projector,
    )
    if args.freeze_moe_router:
        freeze_moe_router(model.language_model)

    # attach bridge
    model.bridge = bridge
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    imgs = None
    tokens = None
    labels = None
    loss_mask = None
    attention_mask = None
    position_ids = None
    use_llamafactory_ds = args.use_new_dataloader and (args.dataset_impl == "llamafactory")

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.is_cpu:
                data[k] = v.cuda(non_blocking=True)
    else:
        data = None

    if args.px_data_config_path is not None:
        update_consumed(args.train_data_consuming_progresses, torch.distributed.get_rank(), data)

    keys = ["image_input_mask", "has_image"]
    data_b = tensor_parallel.broadcast_data(keys, data, torch.bool)
    attention_mask = None
    image_input_mask = data_b["image_input_mask"].bool().contiguous()
    has_image = data_b["has_image"].bool()[0].item()

    keys = ["input_ids", "labels", "position_ids"]
    if has_image:
        keys.extend(["image_grid_thw", "images_padded", "cp_img_num"])
    if use_llamafactory_ds:
        keys.remove("position_ids")
        if "images_padded" in keys:
            keys.remove("images_padded")
            keys.remove("cp_img_num")
    data_b = tensor_parallel.broadcast_data(keys, data, torch.int64)
    tokens = data_b["input_ids"].long().contiguous()
    labels = data_b["labels"].long().contiguous()
    image_grid_thw = data_b.get("image_grid_thw", None)
    images_padded = data_b.get("images_padded", None)
    cp_img_num = data_b.get("cp_img_num", None)
    if has_image and not use_llamafactory_ds:
        cp_img_num = cp_img_num.long().tolist()
        images_padded = images_padded.bool().tolist()
        for image_padded in images_padded:
            assert not image_padded, "not support image padded now"

    position_ids = None
    if not use_llamafactory_ds:
        position_ids = data_b["position_ids"].long().contiguous()

    keys = ["loss_mask"]
    if has_image:
        keys.append("pixel_values")
    data_b = tensor_parallel.broadcast_data(keys, data, torch.float32)
    if has_image:
        imgs = data_b["pixel_values"].float().squeeze(0).contiguous()
        imgs = imgs.type(torch.bfloat16)
    else:
        imgs = None
    loss_mask = data_b["loss_mask"].float().contiguous()

    # llamafactory dataset 没有对 label 及 loss_mask 有位移
    if use_llamafactory_ds:
        labels = labels.roll(shifts=-1, dims=-1)
        loss_mask = loss_mask.roll(shifts=-1, dims=-1)

    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"

    if args.context_parallel_size > 1:
        # tokens不可以切分，因为它要完整生成embeding
        # position_ids不可以切分，它在生成位置编码后再切分
        labels = split_data_cp_rank(labels, args.context_parallel_size, 1)
        loss_mask = split_data_cp_rank(loss_mask, args.context_parallel_size, 1)
        assert attention_mask is None, "if attention_mask is not None, it should be split too"
    return (
        tokens, labels, loss_mask, attention_mask, position_ids, imgs, image_grid_thw,
        image_input_mask, images_padded, cp_img_num
    )


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()
    use_llamafactory_ds = args.use_new_dataloader and (args.dataset_impl == "llamafactory")

    real_seqlen = torch.tensor(
        output_tensor.shape[-1] * args.context_parallel_size, dtype=torch.float
    )

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

    bwd_loss = loss[0] / loss[1]
    if use_llamafactory_ds:
        averaged_loss = bwd_loss.clone().detach()
        torch.distributed.all_reduce(
            averaged_loss,
            group=mpu.get_data_parallel_group(),
            op=torch.distributed.ReduceOp.AVG,
        )
    else:
        averaged_loss = average_losses_across_data_parallel_group(loss)
        averaged_loss = averaged_loss[0] / averaged_loss[1]

    return bwd_loss, {"lm loss": averaged_loss, "real-seqlen": real_seqlen}


def square_averaging_loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()
    use_llamafactory_ds = args.use_new_dataloader and (args.dataset_impl == "llamafactory")

    real_seqlen = torch.tensor(
        output_tensor.shape[-1] * args.context_parallel_size, dtype=torch.float
    )

    losses = output_tensor.float()
    loss_weight = loss_mask.sum(dim=-1).float()

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss_weight, group=mpu.get_context_parallel_group())

    loss_weight = 1 / loss_weight.sqrt()
    loss_weight = torch.where(
        loss_mask == 1, loss_weight.unsqueeze(1),
        torch.tensor(0.0, dtype=loss_weight.dtype, device=loss_weight.device)
    )
    loss_weights_sum = loss_weight.sum()
    torch.distributed.all_reduce(
        loss_weights_sum, op=torch.distributed.ReduceOp.AVG, group=mpu.get_data_parallel_group()
    )

    losses = losses * loss_weight
    losses = losses.sum() / loss_weights_sum
    loss = losses.view(1).clone()

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan().any(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    bwd_loss = loss
    if use_llamafactory_ds:
        averaged_loss = bwd_loss.clone().detach()
        torch.distributed.all_reduce(
            averaged_loss,
            group=mpu.get_data_parallel_group(),
            op=torch.distributed.ReduceOp.AVG,
        )
    else:
        averaged_loss = average_losses_across_data_parallel_group(loss)[0]
    return bwd_loss, {"lm loss": averaged_loss, "real-seqlen": real_seqlen}


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()
    # Get the batch.
    timers("batch-generator", log_level=1).start()
    (
        tokens, labels, loss_mask, attention_mask, position_ids, pixel_values, image_grid_thw,
        image_input_mask, images_padded, cp_img_num
    ) = get_batch(data_iterator)

    timers("batch-generator").stop()

    timers("model-forward-only", log_level=1).start()
    output_tensor = model(
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
        image_grid_thw=image_grid_thw,
        pixel_values=pixel_values,
        image_input_mask=image_input_mask,
        images_padded=images_padded,
        cp_img_num=cp_img_num,
    )
    timers("model-forward-only").stop()
    if args.apply_square_averaging_loss:
        return output_tensor, partial(square_averaging_loss_func, loss_mask)

    return output_tensor, partial(loss_func, loss_mask)


def new_train_valid_test_data_iter_provider(train_val_test_num_samples=None):
    """Build multimodal train, validation and test dataloaders."""
    from gdataset.data_loader.data_loader_builder import DefaultDataItersBuilder
    args = get_args()
    # # 保存 args 为了让 llamafactory 直接 load
    # os.makedirs("ckpt_ds", exist_ok=True)
    # torch.save(args, f"ckpt_ds/{torch.distributed.get_rank()}.pt")
    builder = DefaultDataItersBuilder(args)

    rank = torch.distributed.get_rank()
    dp_rank = mpu.get_data_parallel_rank()
    dp_size = mpu.get_data_parallel_world_size()
    print(f"init data rank {rank}, dp_rank {dp_rank}, dp_size {dp_size}", flush=True)
    train_iter, valid_iter, test_iter = builder.build(rank, dp_size, dp_rank)

    return train_iter, valid_iter, test_iter


def train_valid_test_data_iter_provider(train_val_test_num_samples=None):
    """Build multimodal train, validation and test dataloaders."""
    args = get_args()
    if args.use_new_dataloader:
        return new_train_valid_test_data_iter_provider(train_val_test_num_samples)

    using_dataset_v4 = True
    if args.px_data_config_path is not None:
        using_dataset_v4 = False
        if not args.use_grpo:
            parse_dataset_config(args)
    # tp-rank != 0 返回空，但在use_grpo时，每个tp-rank都会创建dataloader
    if not args.use_grpo and mpu.get_tensor_model_parallel_rank() != 0:
        return None, None, None
    tokenizer = get_tokenizer()

    print_rank_0(f'> building train, validation, and test dataloader ... {using_dataset_v4=}')
    if not using_dataset_v4:
        from megatron_datasets.qwen2vl_dataset import build_train_valid_test_data_iter
    else:
        from tasks.qwen2vl.qwen2vl_dataset_map import build_train_valid_test_data_iter
    train_iter, valid_iter, test_iter = build_train_valid_test_data_iter(
        args,
        tokenizer,
        rank=torch.distributed.get_rank(),
        dp_rank=mpu.get_data_parallel_rank(),
        dp_size=mpu.get_data_parallel_world_size(),
        is_dpo=args.dpo,
    )

    print(
        f"> world size {mpu.get_data_parallel_world_size()} rank {mpu.get_data_parallel_rank()} "
        f"finished creating dataloader ..."
    )
    return train_iter, valid_iter, test_iter


def add_qwen3vl_extra_args(parser):
    parser = gpatch_extra_args(parser)
    """Extra arguments."""
    group = parser.add_argument_group(title='qwen3vl arguments')

    group.add_argument("--processor-path", type=str, default=None, help="")
    group.add_argument("--tarfile-path", type=str, default="/", help="")
    group.add_argument("--min-pixels-num", type=int, default=None, help="min image width * height")
    group.add_argument("--max-pixels-num", type=int, default=None, help="max image width * height")
    group.add_argument("--video-min-frames", type=int, default=None, help="min video frames")
    group.add_argument("--video-max-frames", type=int, default=None, help="max video frames")
    group.add_argument(
        "--video-min-pixels",
        type=int,
        default=None,
        help="min video frame num_frame * width * height"
    )
    group.add_argument(
        "--video-max-pixels",
        type=int,
        default=None,
        help="max video frame num_frame * width * height"
    )
    group.add_argument("--lmdb-port", type=int, default=None, help="lmdb server port")
    group.add_argument('--spatial-merge-size', type=int, default=2, help='spatial merge size')
    group.add_argument("--mask-history", action='store_true', help="多轮对话只取最后一轮对话为label")

    group.add_argument(
        '--ppo-skip-special-tokens',
        action="store_true",
        help="whether to tokenizer decode skip special tokens or not"
    )
    return parser


if __name__ == "__main__":
    # 每个tp-rank都要运行parse_dataset_config
    init_gpatch_for_mcore()
    assert mcore_version >= Version("0.13.0")
    setattr(train_valid_test_data_iter_provider, "is_distributed", True)

    from megatron.training import inprocess_restart
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_data_iter_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_qwen3vl_extra_args,
        store=store,
    )
