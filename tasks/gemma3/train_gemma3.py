# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# guanyouhe@tencent.com

import os
import warnings
from copy import deepcopy
from functools import partial
from dataclasses import asdict
from typing import Union
from packaging.version import Version

import torch

from gpatch.patch_mcore import init_gpatch_for_mcore
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, get_tokenizer, print_rank_0, pretrain, get_timers
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron.core import tensor_parallel
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.core import package_info

from megatron_datasets.args import parse_dataset_config
from megatron_datasets.mega_indexed_jsonl_dataset_v3 import update_consumed
from megatron.training.activations import fast_gelu

from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.training.arguments import gpatch_extra_args
from gpatch.core.models.vision.multimodal_projector import get_projector_module_spec_te
from gpatch.core.transformer.transformer_config import Gemma3TransformerConfig
from gpatch.core.models.multimodal.gemma3_model import Gemma3Model
from gpatch.core.models.multimodal.llava_model_dpo import Gemma3ModelDPO
from gpatch.core.models.multimodal.layer_specs import (
    get_gemma3_layer_spec_te,
    get_layer_spec_te,
)
from gpatch.core.utils import split_data_cp_rank


def get_vision_projection_config(config: Gemma3TransformerConfig):
    proj_config = Gemma3TransformerConfig(**asdict(config))
    proj_config.image_size = 896
    proj_config.patch_size = 14
    proj_config.mm_tokens_per_image = 256
    proj_config.layernorm_zero_centered_gamma = True
    proj_config.add_bias_linear = False
    return proj_config


def get_siglip_model_config(config: Gemma3TransformerConfig, apply_query_key_layer_scaling):
    config.num_layers = 27
    config.num_attention_heads = 16
    config.add_bias_linear = True
    config.add_qkv_bias = True
    config.hidden_size = 1152
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.ffn_hidden_size = 4304
    config.gated_linear_unit = False
    config.activation_func = fast_gelu
    config.kv_channels = 72
    config.num_query_groups = 16
    config.layernorm_zero_centered_gamma = False
    config.apply_query_key_layer_scaling = apply_query_key_layer_scaling
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.normalization = 'LayerNorm'
    config.apply_rope_fusion = False
    config.qk_layernorm = False
    config.layernorm_epsilon = 1e-6
    return config


def get_llava_model_configs(args):
    base_config = core_transformer_config_from_args(args, Gemma3TransformerConfig)
    base_config.sliding_window = args.sliding_window
    base_config.embed_scale = base_config.hidden_size**0.5
    base_config.hf_vocab_size = get_tokenizer().vocab_size

    language_config = deepcopy(base_config)
    language_config.activation_func = torch.nn.functional.gelu

    language_transformer_layer_spec = get_gemma3_layer_spec_te(is_vit=False)

    vision_config = deepcopy(base_config)
    vision_config = get_siglip_model_config(
        vision_config, apply_query_key_layer_scaling=args.apply_query_key_layer_scaling
    )

    vision_transformer_layer_spec = get_layer_spec_te(is_vit=True)
    vision_projection_config = get_vision_projection_config(base_config)

    # --encoder-pipeline-model-parallel-size 1 will enable a separate pipeline stage for the vision model.
    if args.encoder_pipeline_model_parallel_size > 0:
        assert (
            args.encoder_pipeline_model_parallel_size == 1
        ), "vision model and projection can only live on 1 pipeline stage."

        if args.encoder_tensor_model_parallel_size > 0:
            vision_config.tensor_model_parallel_size = args.encoder_tensor_model_parallel_size
            vision_projection_config.tensor_model_parallel_size = (
                args.encoder_tensor_model_parallel_size
            )

    # Make sure vision model pipeline parallel size is not inherited from the language model pipeline parallel size.
    # 0 is not a valid for the config value, hence max(1, ).
    vision_config.pipeline_model_parallel_size = max(1, args.encoder_pipeline_model_parallel_size)
    vision_projection_config.pipeline_model_parallel_size = vision_config.pipeline_model_parallel_size

    # Make sure the vision model does not inherit first and last pipeline num layers from the language model.
    vision_config.num_layers_in_first_pipeline_stage = vision_config.num_layers_in_last_pipeline_stage = None

    vision_projection_layer_spec = get_projector_module_spec_te()

    # Toggle --recompute* for the vision and language model separately.
    vision_config.recompute_granularity = None
    vision_config.recompute_method = None
    vision_config.recompute_num_layers = None

    vision_projection_config.recompute_granularity = None
    vision_projection_config.recompute_method = None
    vision_projection_config.recompute_num_layers = None

    return (
        language_config,
        language_transformer_layer_spec,
        vision_config,
        vision_transformer_layer_spec,
        vision_projection_config,
        vision_projection_layer_spec,
    )


def check_model(args):
    assert args.encoder_pipeline_model_parallel_size <= 1, "LLaVA does not support pp>1 for encoder on it's own pipeline rank"
    assert args.qk_layernorm, f"you should add --qk-layernorm"
    assert args.transformer_impl == "transformer_engine", "Gemma3 only supports TE now"
    print_rank_0('building a multimodal model ...')

    assert (
        args.decoder_seq_length is not None
    ), "Please provide --decoder-seq-length to set the language model sequence length"
    if args.decoder_seq_length > args.max_position_embeddings:
        args.max_position_embeddings = args.decoder_seq_length
        warnings.warn(
            f"Expanded max_position_embeddings to {args.max_position_embeddings} to accommodate the maximum language model sequence length"
        )


def sft_model_provider(
    pre_process=True,
    post_process=True,
    add_encoder=True,
    add_decoder=True,
    parallel_output=True
) -> Gemma3Model:
    """Build the model."""
    args = get_args()
    check_model(args)

    (
        language_config,
        language_transformer_layer_spec,
        vision_config,
        vision_transformer_layer_spec,
        vision_projection_config,
        vision_projection_layer_spec,
    ) = get_llava_model_configs(args)

    model = Gemma3Model(
        language_transformer_config=language_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.decoder_seq_length,
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_transformer_layer_spec,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_layer_spec,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        img_h=args.img_h,
        img_w=args.img_w,
        patch_dim=args.patch_dim,
        language_rotary_base=args.rotary_base,
        language_rope_scaling=args.use_rope_scaling,
    )

    model.freeze(
        freeze_language_model=args.mm_freeze_llm,
        freeze_vision_model=args.mm_freeze_vision_encoder,
        freeze_vision_projection=args.mm_freeze_projector,
    )

    return model


def dpo_model_provider(
    pre_process=True,
    post_process=True,
    add_encoder=True,
    add_decoder=True,
    parallel_output=True
) -> Gemma3ModelDPO:
    """Build the model."""
    args = get_args()
    check_model(args)

    (
        language_config,
        language_transformer_layer_spec,
        vision_config,
        vision_transformer_layer_spec,
        vision_projection_config,
        vision_projection_layer_spec,
    ) = get_llava_model_configs(args)

    model = Gemma3ModelDPO(
        # llava model param
        language_transformer_config=language_config,
        language_transformer_layer_spec=language_transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.decoder_seq_length,
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_transformer_layer_spec,
        vision_projection_config=vision_projection_config,
        vision_projection_layer_spec=vision_projection_layer_spec,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        pre_process=pre_process,
        post_process=post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        llava_model_class=Gemma3Model,
        # dpo param
        beta=args.dpo_beta,
        label_smoothing=args.dpo_label_smoothing,
        ftx_gamma=args.dpo_ftx_gamma,
        # llava model extra param
        img_h=args.img_h,
        img_w=args.img_w,
        patch_dim=args.patch_dim,
        language_rotary_base=args.rotary_base,
        language_rope_scaling=args.use_rope_scaling,
    )

    model.freeze(
        freeze_language_model=args.mm_freeze_llm,
        freeze_vision_model=args.mm_freeze_vision_encoder,
        freeze_vision_projection=args.mm_freeze_projector,
    )

    return model


def model_provider(
    pre_process=True,
    post_process=True,
    add_encoder=True,
    add_decoder=True,
    parallel_output=True,
) -> Union[Gemma3Model, Gemma3ModelDPO]:
    args = get_args()
    if args.dpo:
        return dpo_model_provider(
            pre_process, post_process, add_encoder, add_decoder, parallel_output
        )

    return sft_model_provider(pre_process, post_process, add_encoder, add_decoder, parallel_output)


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

    keys = ["attention_mask", "sliding_window_attention_mask"]
    if has_imgs > 0:
        keys.append("pixel_values")
    data_b = tensor_parallel.broadcast_data(keys, data, torch.bfloat16)
    attn_mask = data_b["attention_mask"].type(torch.bfloat16).contiguous()
    sliding_window_attention_mask = data_b["sliding_window_attention_mask"].type(torch.bfloat16
                                                                                ).contiguous()

    if has_imgs > 0:
        # shape: num_imgs x c x h x w
        imgs = data_b["pixel_values"].type(torch.bfloat16).contiguous()
    else:
        imgs = torch.tensor([], dtype=torch.bfloat16, device=tokens.device)

    _, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        labels,
        -100,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    if args.context_parallel_size > 1:
        labels = split_data_cp_rank(labels, args.context_parallel_size, 1)
        loss_mask = split_data_cp_rank(loss_mask, args.context_parallel_size, 1)
        if has_imgs > 0:
            imgs = split_data_cp_rank(imgs, args.context_parallel_size, 2)

    return (
        tokens,
        labels,
        loss_mask,
        attn_mask,
        sliding_window_attention_mask,
        position_ids,
        imgs,
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
    if Version(package_info.__version__) < Version("0.12.1"):
        bwd_loss = (loss[0] / loss[1]) * args.context_parallel_size
    else:
        bwd_loss = loss[0] / loss[1]

    return bwd_loss, {"lm loss": averaged_loss}


def dpo_loss_func(metrics, output_tensor: torch.Tensor):
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

    if Version(package_info.__version__) < Version("0.12.1"):
        bwd_loss = loss * args.context_parallel_size
    else:
        bwd_loss = loss
    return bwd_loss, metrics


def forward_step(data_iterator, model: Union[Gemma3Model, Gemma3ModelDPO]):
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
        tokens,
        labels,
        loss_mask,
        attn_mask,
        sliding_window_attention_mask,
        position_ids,
        imgs,
    ) = get_batch(data_iterator)
    timers("batch-generator").stop()

    timers("model-forward-only", log_level=1).start()
    image_token_id = get_tokenizer()._tokenizer.image_token_id
    output_tensor_or_metric = model(
        images=imgs,
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=(attn_mask, sliding_window_attention_mask),
        labels=labels,
        image_token_index=image_token_id,
    )
    timers("model-forward-only").stop()

    if args.dpo:
        output_tensor, metric = output_tensor_or_metric
        return output_tensor, partial(dpo_loss_func, metric)
    output_tensor = output_tensor_or_metric
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
    from megatron_datasets.gemma3_dataset import build_train_valid_test_data_iter
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

    group = parser.add_argument_group(title='gemma3 arguments')
    group.add_argument("--processor-path", type=str, default=None, help="")
    group.add_argument("--tarfile-path", type=str, default="/", help="")
    group.add_argument("--lmdb-port", type=int, default=None, help="lmdb server port")
    group.add_argument("--sliding-window", type=int, default=1024, help="Gemma3 sliding window")
    group.add_argument(
        "--query-pre-attn-scalar", type=int, default=256, help="query_pre_attn_scalar"
    )
    group.add_argument("--mask-history", action='store_true', help="多轮对话只取最后一轮对话为label")

    group.add_argument(
        '--ppo-skip-special-tokens',
        action="store_true",
        help="whether to tokenizer decode skip special tokens or not"
    )
    return parser


if __name__ == "__main__":

    init_gpatch_for_mcore()
    mcore_version = Version(package_info.__version__)
    print(f"{mcore_version=} {Version('0.13.0')} {mcore_version < Version('0.13.0')}")
    if mcore_version < Version("0.13.0"):
        extra_args = {}
    else:
        from megatron.training import inprocess_restart
        pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
        extra_args = {"store": store}

    # 每个tp-rank都要运行parse_dataset_config
    setattr(train_valid_test_data_iter_provider, "is_distributed", True)

    pretrain(
        train_valid_test_data_iter_provider,
        model_provider,
        ModelType.encoder_and_decoder,
        forward_step,
        extra_args_provider=add_extra_args,
        **extra_args,
    )
