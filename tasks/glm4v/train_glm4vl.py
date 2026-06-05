import copy
import os
import sys
from dataclasses import fields
from functools import partial
from typing import Union

import torch
import torch.distributed
from packaging.version import Version

from megatron_datasets.args import parse_dataset_config
from megatron_datasets.mega_indexed_jsonl_dataset_v3 import update_consumed

from mbridge import AutoBridge
from megatron.core import ModelParallelConfig, mpu, package_info, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import average_losses_across_data_parallel_group

from gpatch.core.device_type import is_wxacc2
from gpatch.core.models.multimodal.qwen2vl_model import Qwen2VLModel
from gpatch.core.transformer.transformer_config import GpatchTransformerConfig
from gpatch.core.utils import split_data_cp_rank, split_data_ulysses_cp_rank
from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.training.arguments import gpatch_extra_args

PARARELL_CONFIG = [""]
MODEL_CONFIG = [""]

# config key in model config (TransformerConfig) that has parallel implications
_PARALLEL_CONFIG_KEYS_ = [
    "num_layers_in_first_pipeline_stage",
    "num_layers_in_last_pipeline_stage",
    "account_for_embedding_in_pipeline_split",
    "account_for_loss_in_pipeline_split",
    "recompute_granularity",
    "recompute_method",
    "recompute_num_layers",
    "distribute_saved_activations",
    "recompute_modules",
    "init_model_with_meta_device",
    "moe_shared_expert_overlap",
    "moe_layer_recompute",
    "moe_token_dispatcher_type",
    "moe_enable_deepep",
    "moe_router_bias_update_rate"
    "moe_router_dtype",
    "moe_permute_fusion",
    "bias_dropout_fusion",
    "cp_comm_type",
    "clone_scatter_output_in_embedding",
    "config_logger_dir",
]


def merge_config(hf_config, mg_config):
    assert hf_config.num_layers == mg_config.num_layers
    config_merged = copy.deepcopy(mg_config)
    hf_fields = {e.name for e in fields(hf_config)}
    mg_fields = {e.name for e in fields(ModelParallelConfig)}
    # config fields TransformerConfig - ModelParallelConfig
    diff = hf_fields - mg_fields
    # generally, use parallel config from megatron config, use model config from hf config
    for f in diff:
        setattr(config_merged, f, getattr(hf_config, f))

    # some config in model config (TransformerConfig) has parallel implications, so we use megatron config
    for k in _PARALLEL_CONFIG_KEYS_:
        if hasattr(mg_config, k):
            setattr(config_merged, k, getattr(mg_config, k))

    return config_merged


def freeze_moe_router(model):
    for layer in model.decoder.layers:
        if hasattr(layer.mlp, "router"):
            if hasattr(layer.mlp.router, "weight"):
                layer.mlp.router.weight.requires_grad = False
            if hasattr(layer.mlp.router, "bias") and layer.mlp.router.bias is not None:
                layer.mlp.router.bias.requires_grad = False
        if hasattr(layer.mlp, "shared_experts"):
            if hasattr(layer.mlp.shared_experts, "gate_weight") and \
                    layer.mlp.shared_experts.gate_weight is not None:
                layer.mlp.shared_experts.gate_weight.requires_grad = False
            if hasattr(layer.mlp.shared_experts, "gate_bias"):
                layer.mlp.shared_experts.gate_bias.requires_grad = False


def model_provider(
    pre_process=True,
    post_process=True,
):
    args = get_args()
    bridge = AutoBridge.from_pretrained(args.processor_path)
    config = core_transformer_config_from_args(args, GpatchTransformerConfig)
    config = merge_config(bridge.config, config)
    # args => config, should be processed by transformer_from_args, direct copy for now
    config.recompute_ve = args.recompute_ve
    bridge.config = config
    model = bridge._model_provider([])(pre_process=pre_process, post_process=post_process)
    if not args.mm_freeze_vision_encoder:
        assert args.tensor_model_parallel_size == 1, f"when vision_encoder is tuned, tp must be 1"

    model.freeze(args.mm_freeze_llm, args.mm_freeze_vision_encoder, args.mm_freeze_projector)

    if args.freeze_moe_router:
        freeze_moe_router(model)

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

    keys = ["has_image"]
    data_b = tensor_parallel.broadcast_data(keys, data, torch.bool)
    attention_mask = None
    #image_input_mask = data_b["image_input_mask"].bool().contiguous()
    #image_padded = data_b["image_padded"].bool()[0].item()
    has_image = data_b["has_image"].bool()[0].item()

    keys = ["input_ids", "labels", "position_ids"]
    if has_image:
        keys.append("image_grid_thw")
    data_b = tensor_parallel.broadcast_data(keys, data, torch.int64)
    tokens = data_b["input_ids"].long().contiguous()
    labels = data_b["labels"].long().contiguous()
    image_grid_thw = data_b.get("image_grid_thw", None)
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
    video_input_mask = None

    assert (tokens.shape == labels.shape), f"tokens: {tokens.shape} != labels: {labels.shape}"

    if args.context_parallel_size > 1:
        assert False, "not supported yet"

    return (
        tokens,
        labels,
        loss_mask,
        attention_mask,
        position_ids,
        imgs,
        image_grid_thw,
    )


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()
    real_seqlen = output_tensor.shape[-1] * args.context_parallel_size

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
    if Version(package_info.__version__) < Version("0.12.1"):
        bwd_loss = (loss[0] / loss[1]) * args.context_parallel_size
    else:
        bwd_loss = loss[0] / loss[1]

    return bwd_loss, {"lm loss": averaged_loss, "real-seqlen": real_seqlen}


def forward_step(data_iterator, model: Qwen2VLModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()
    # Get the batch.
    timers("batch-generator", log_level=1).start()
    (tokens, labels, loss_mask, attention_mask, position_ids, pixel_values,
     image_grid_thw) = get_batch(data_iterator)

    timers("batch-generator").stop()

    timers("model-forward-only", log_level=1).start()
    output_tensor_or_metric = model(
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        labels=labels,
        image_grid_thw=image_grid_thw,
        pixel_values=pixel_values,
    )
    timers("model-forward-only").stop()

    if isinstance(output_tensor_or_metric, tuple):
        assert args.dpo
        output_tensor, metric = output_tensor_or_metric
        return output_tensor, partial(dpo_loss_func, metric)

    assert not args.dpo
    output_tensor = output_tensor_or_metric
    return output_tensor, partial(loss_func, loss_mask)


def old_train_valid_test_data_iter_provider(train_val_test_num_samples=None):
    """Build multimodal train, validation and test dataloaders."""
    args = get_args()
    using_dataset_v4 = True
    if args.px_data_config_path is not None:
        using_dataset_v4 = False
        if not args.use_grpo:
            parse_dataset_config(args)
    assert using_dataset_v4
    # tp-rank != 0 返回空，但在use_grpo时，每个tp-rank都会创建dataloader
    if not args.use_grpo and mpu.get_tensor_model_parallel_rank() != 0:
        return None, None, None
    tokenizer = get_tokenizer()

    print_rank_0(f"> building train, validation, and test dataloader ... {using_dataset_v4=}")
    from tasks.glm4v.glm4vl_dataset_map import build_train_valid_test_data_iter

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


def new_train_valid_test_data_iter_provider(train_val_test_num_samples=None):
    """Build multimodal train, validation and test dataloaders."""
    from gdataset.data_loader.data_loader_builder import DefaultDataItersBuilder
    args = get_args()
    builder = DefaultDataItersBuilder(args)

    rank = torch.distributed.get_rank()
    dp_rank = mpu.get_data_parallel_rank()
    dp_size = mpu.get_data_parallel_world_size()
    print(f"init data rank {rank}, dp_rank {dp_rank}, dp_size {dp_size}", flush=True)
    train_iter, valid_iter, test_iter = builder.build(rank, dp_size, dp_rank)

    save_state = hasattr(train_iter, "save_state")

    return train_iter, valid_iter, test_iter


def train_valid_test_data_iter_provider(train_val_test_num_samples=None):
    args = get_args()
    if args.use_new_dataloader:
        return new_train_valid_test_data_iter_provider(train_val_test_num_samples)
    else:
        return old_train_valid_test_data_iter_provider(train_val_test_num_samples)


def add_glm4vl_extra_args(parser):
    parser = gpatch_extra_args(parser)
    """Extra arguments."""
    group = parser.add_argument_group(title="qwen2vl/qwen2.5vl arguments")
    group.add_argument("--processor-path", type=str, default=None, help="")
    group.add_argument("--tarfile-path", type=str, default="/", help="")
    group.add_argument("--lmdb-port", type=int, default=None, help="lmdb server port")
    group.add_argument("--mask-history", action="store_true", help="多轮对话只取最后一轮对话为label")
    group.add_argument(
        "--ppo-skip-special-tokens",
        action="store_true",
        help="whether to tokenizer decode skip special tokens or not",
    )
    group.add_argument(
        "--recompute-ve",
        action="store_true",
        help="whether to recompute vision encoder",
    )

    return parser


if __name__ == "__main__":
    # 每个tp-rank都要运行parse_dataset_config
    init_gpatch_for_mcore()
    setattr(train_valid_test_data_iter_provider, "is_distributed", True)

    pretrain(
        train_valid_test_data_iter_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_glm4vl_extra_args,
    )
