import os
import sys
from functools import partial
from typing import Union
from packaging.version import Version

from gpatch.core.device_type import is_wxacc2

import torch
import torch.distributed

from megatron.core import mpu

from megatron.core.enums import ModelType
from megatron.training import get_args, get_timers, pretrain, print_rank_0, get_tokenizer
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec, get_gpt_layer_local_spec
)
from megatron.core import mpu, tensor_parallel
from megatron.core import package_info

from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.core.models.multimodal.qwen2vl_model import Qwen2VLModel
from gpatch.core.models.multimodal.llava_model_dpo import Qwen2VLModelDPO
from gpatch.core.transformer.transformer_config import GpatchTransformerConfig
from gpatch.core.utils import split_data_cp_rank
from gpatch.core.models.vision.qwen2vl_vit_model import (
    Qwen2VisionModel,
    Qwen2P5VisionModel,
    Qwen2VLTransformerConfig,
    Qwen2P5VLTransformerConfig,
)

from gpatch.core.models.vision.mulitmodal_vit_spec import (
    get_qwen2vl_vision_local_spec,
    get_qwen2vl_vision_with_transformer_engine_spec,
    get_proj_mlp_module_spec,
    get_qwen2vl_vision_with_transformer_engine_spec_lora,
    get_qwen2vl_vision_with_local_spec_lora,
    get_proj_mlp_module_spec_lora,
)
from gpatch.core.models.vision.config import (
    get_qwen2vl_vision_model_config,
    get_qwen2vl_vision_projection_config,
    get_qwen2p5vl_vision_model_config,
    get_qwen2p5vl_vision_projection_config,
)
from gpatch.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec_lora, get_gpt_layer_with_local_spec_lora
)
from gpatch.training.arguments import gpatch_extra_args

from megatron_datasets.args import parse_dataset_config
from megatron_datasets.mega_indexed_jsonl_dataset_v3 import update_consumed

mcore_version = Version(package_info.__version__)


def get_qwen2vl_config(args):
    # Config of vit, llm and projector
    tf_config = core_transformer_config_from_args(args, GpatchTransformerConfig)
    config = core_transformer_config_from_args(args, Qwen2VLTransformerConfig)
    tf_config.mrope_section = [16, 24, 24]
    config.mrope_section = [16, 24, 24]

    if args.rotary_seq_len_interpolation_factor is not None or args.rotary_seq_len_interpolation_factor != 1:
        print_rank_0('Multimodal RoPE currently not support RoPE interpolation, set to None...')
        args.rotary_seq_len_interpolation_factor = None

    vision_config = get_qwen2vl_vision_model_config(tf_config, args.seq_length)
    vision_projector_config = get_qwen2vl_vision_projection_config(
        tf_config, vision_config.hidden_size, args.spatial_merge_size
    )

    print_rank_0("building Qwen2-VL model")
    if args.enable_lora:
        # Layer Specs of vit, llm and projector
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec_lora(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            gated_linear_unit=config.gated_linear_unit,
        )
        vision_model_spec = get_qwen2vl_vision_with_transformer_engine_spec_lora(
            args.context_parallel_size,
            False,
            gated_linear_unit=vision_config.gated_linear_unit,
        )
        vision_projector_spec = get_proj_mlp_module_spec_lora(
            use_te=True,
            add_norm=False,
        ).submodules
    else:
        # Layer Specs of vit, llm and projector
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
        )
        vision_model_spec = get_qwen2vl_vision_with_transformer_engine_spec(
            args.context_parallel_size,
            False,
        )
        vision_projector_spec = get_proj_mlp_module_spec(use_te=True, add_norm=False).submodules

    return config, vision_config, vision_projector_config, vision_model_spec, transformer_layer_spec, vision_projector_spec, Qwen2VisionModel


def get_qwen2p5vl_config(args):
    # Config of vit, llm and projector
    tf_config = core_transformer_config_from_args(args, GpatchTransformerConfig)
    config = core_transformer_config_from_args(args, Qwen2P5VLTransformerConfig)
    tf_config.mrope_section = [16, 24, 24]
    config.mrope_section = [16, 24, 24]

    vision_config = get_qwen2p5vl_vision_model_config(tf_config, args.seq_length)
    vision_projector_config = get_qwen2p5vl_vision_projection_config(
        tf_config, vision_config.hidden_size, args.spatial_merge_size
    )
    if not is_wxacc2():
        if args.enable_lora:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec_lora(
                args.num_experts,
                args.moe_grouped_gemm,
                args.qk_layernorm,
                gated_linear_unit=config.gated_linear_unit,
            )
            vision_model_spec = get_qwen2vl_vision_with_transformer_engine_spec_lora(
                args.context_parallel_size,
                True,
                gated_linear_unit=vision_config.gated_linear_unit,
            )
            vision_projector_spec = get_proj_mlp_module_spec_lora(
                use_te=True,
                add_norm=False,
            ).submodules
        else:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                args.num_experts,
                args.moe_grouped_gemm,
                args.qk_layernorm,
            )
            vision_model_spec = get_qwen2vl_vision_with_transformer_engine_spec(
                args.context_parallel_size,
                True,
            )
            vision_projector_spec = get_proj_mlp_module_spec(use_te=True, add_norm=False).submodules
    else:
        assert args.context_parallel_size == 1, "currently context parallel is not supported"
        if args.enable_lora:
            transformer_layer_spec = get_gpt_layer_with_local_spec_lora(
                args.num_experts,
                args.moe_grouped_gemm,
                args.qk_layernorm,
                gated_linear_unit=config.gated_linear_unit,
            )
            vision_model_spec = get_qwen2vl_vision_with_local_spec_lora(
                args.context_parallel_size,
                True,
                gated_linear_unit=vision_config.gated_linear_unit,
            )
            vision_projector_spec = get_proj_mlp_module_spec_lora(
                use_te=False,
                add_norm=False,
            ).submodules
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(
                args.num_experts,
                args.moe_grouped_gemm,
                args.qk_layernorm,
            )
            vision_model_spec = get_qwen2vl_vision_local_spec(args.model_arch == "qwen2.5vl")
            vision_projector_spec = get_proj_mlp_module_spec(
                use_te=False, add_norm=False
            ).submodules

    return config, vision_config, vision_projector_config, vision_model_spec, transformer_layer_spec, vision_projector_spec, Qwen2P5VisionModel


def sft_model_provider(
    pre_process=True,
    post_process=True,
    add_encoder=True,
    add_decoder=True,
    parallel_output=True,
) -> Qwen2VLModel:
    args = get_args()
    print_rank_0("start building qwen2v./qwen2.5vl model ...")
    if args.rotary_seq_len_interpolation_factor is not None or args.rotary_seq_len_interpolation_factor != 1:
        print_rank_0('Multimodal RoPE currently not support RoPE interpolation, set to None...')
        args.rotary_seq_len_interpolation_factor = None

    if args.model_arch == "qwen2vl":
        config, vision_config, vision_projector_config, vision_model_spec, \
            transformer_layer_spec, vision_projector_spec, vision_model_class = get_qwen2vl_config(args)
    elif args.model_arch == "qwen2.5vl":
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError as e:
            raise ImportError("Please install transformers>=4.49.0.dev to support Qwen2.5VL")
        config, vision_config, vision_projector_config, vision_model_spec, \
            transformer_layer_spec, vision_projector_spec, vision_model_class = get_qwen2p5vl_config(args)
    else:
        raise ValueError(f"本文件只支持qwen2vl/qwen2.5vl {args.model_arch=}")

    model = Qwen2VLModel(
        language_transformer_config=config,
        language_transformer_layer_spec=transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_model_spec,
        vision_projection_config=vision_projector_config,
        vision_projection_layer_spec=vision_projector_spec,
        vision_projection_type='mlp',
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,
        pre_process=pre_process,
        post_process=post_process,
        add_decoder=add_decoder,
        add_encoder=add_encoder,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        vision_model_class=vision_model_class,
    )

    model.freeze(
        freeze_language_model=args.mm_freeze_llm,
        freeze_vision_model=args.mm_freeze_vision_encoder,
        freeze_vision_projection=args.mm_freeze_projector,
    )

    # freeze llm when use lora
    if args.enable_lora:
        assert (not args.freeze_lora) and args.mm_freeze_llm and args.mm_freeze_vision_encoder \
            and args.mm_freeze_projector, "use LoRA should freeze all but not LoRA"
        for pname, params in model.named_parameters():
            if '.lora_a.' in pname or '.lora_b.' in pname:
                params.requires_grad_(True)

    return model


def dpo_model_provider(
    pre_process=True,
    post_process=True,
    add_encoder=True,
    add_decoder=True,
    parallel_output=True,
) -> Qwen2VLModelDPO:
    args = get_args()
    print_rank_0("start building qwen2v./qwen2.5vl model ...")
    if args.rotary_seq_len_interpolation_factor is not None or args.rotary_seq_len_interpolation_factor != 1:
        print_rank_0('Multimodal RoPE currently not support RoPE interpolation, set to None...')
        args.rotary_seq_len_interpolation_factor = None

    if args.model_arch == "qwen2vl":
        config, vision_config, vision_projector_config, vision_model_spec, \
            transformer_layer_spec, vision_projector_spec, vision_model_class = get_qwen2vl_config(args)
    elif args.model_arch == "qwen2.5vl":
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError as e:
            raise ImportError("Please install transformers>=4.49.0.dev to support Qwen2.5VL")
        config, vision_config, vision_projector_config, vision_model_spec, \
            transformer_layer_spec, vision_projector_spec, vision_model_class = get_qwen2p5vl_config(args)
    else:
        raise ValueError(f"本文件只支持qwen2vl/qwen2.5vl {args.model_arch=}")

    model = Qwen2VLModelDPO(
        # qwen2vl model param
        language_transformer_config=config,
        language_transformer_layer_spec=transformer_layer_spec,
        language_vocab_size=args.padded_vocab_size,
        language_max_sequence_length=args.max_position_embeddings,
        vision_transformer_config=vision_config,
        vision_transformer_layer_spec=vision_model_spec,
        vision_projection_config=vision_projector_config,
        vision_projection_layer_spec=vision_projector_spec,
        vision_projection_type='mlp',
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        language_position_embedding_type=args.position_embedding_type,
        language_rotary_percent=args.rotary_percent,
        language_rotary_base=args.rotary_base,
        pre_process=pre_process,
        post_process=post_process,
        add_decoder=add_decoder,
        add_encoder=add_encoder,
        llava_model_class=Qwen2VLModel,
        # dpo param
        beta=args.dpo_beta,
        label_smoothing=args.dpo_label_smoothing,
        ftx_gamma=args.dpo_ftx_gamma,
        # qwen2vl model extra param
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        vision_model_class=vision_model_class,
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
) -> Union[Qwen2VLModel, Qwen2VLModelDPO]:
    args = get_args()
    if args.context_parallel_size > 1:
        assert mcore_version >= Version(
            '0.13.0'
        ), f"mcore version should >= 0.13.0, but now {mcore_version=}"
    if args.decoder_seq_length is None:
        args.decoder_seq_length = args.seq_length
    if args.dpo:
        return dpo_model_provider(
            pre_process, post_process, add_encoder, add_decoder, parallel_output
        )

    return sft_model_provider(pre_process, post_process, add_encoder, add_decoder, parallel_output)


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

    keys = ["image_input_mask", "has_image"]
    data_b = tensor_parallel.broadcast_data(keys, data, torch.bool)
    attention_mask = None
    image_input_mask = data_b["image_input_mask"].bool().contiguous()
    has_image = data_b["has_image"].bool()[0].item()

    keys = ["input_ids", "labels", "position_ids"]
    if has_image:
        keys.extend(["image_grid_thw", "images_padded", "cp_img_num"])
    data_b = tensor_parallel.broadcast_data(keys, data, torch.int64)
    tokens = data_b["input_ids"].long().contiguous()
    labels = data_b["labels"].long().contiguous()
    image_grid_thw = data_b.get("image_grid_thw", None)
    images_padded = data_b.get("images_padded", None)
    cp_img_num = data_b.get("cp_img_num", None)
    if has_image:
        cp_img_num = cp_img_num.long().tolist()
        images_padded = images_padded.bool().tolist()
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

    assert tokens.shape == labels.shape, f"tokens: {tokens.shape} != labels: {labels.shape}"

    if args.context_parallel_size > 1:
        # tokens不可以切分，因为它要完整生成embeding
        # position_ids不可以切分，它在生成位置编码后再切分
        labels = split_data_cp_rank(labels, args.context_parallel_size, 1)
        loss_mask = split_data_cp_rank(loss_mask, args.context_parallel_size, 1)
        assert attention_mask is None, "if attention_mask is not None, it should be split too"

    return (
        tokens, labels, loss_mask, attention_mask, position_ids, imgs, image_grid_thw,
        image_input_mask, video_input_mask, images_padded, cp_img_num
    )


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()
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

    averaged_loss = average_losses_across_data_parallel_group(loss)
    averaged_loss = averaged_loss[0] / averaged_loss[1]
    if Version(package_info.__version__) < Version("0.12.1"):
        bwd_loss = (loss[0] / loss[1]) * args.context_parallel_size
    else:
        bwd_loss = loss[0] / loss[1]

    return bwd_loss, {"lm loss": averaged_loss, "real-seqlen": real_seqlen}


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
    (
        tokens, labels, loss_mask, attention_mask, position_ids, imgs, thw_grids, image_input_mask,
        video_input_mask, images_padded, cp_img_num
    ) = get_batch(data_iterator)
    assert video_input_mask is None, "有video的输入qwen2vl与qwen2.5vl不同, 暂时不支持"
    timers("batch-generator").stop()

    timers("model-forward-only", log_level=1).start()
    output_tensor_or_metric = model(
        input_ids=tokens,
        position_ids=position_ids,
        vision_data=imgs,
        vision_grid_thw=thw_grids,
        video_start_index=image_input_mask.sum().cpu().item(),
        image_input_mask=image_input_mask,
        video_input_mask=video_input_mask,
        attention_mask=attention_mask,
        labels=labels,
        images_padded=images_padded,
        cp_img_num=cp_img_num,
    )
    timers("model-forward-only").stop()

    if isinstance(output_tensor_or_metric, tuple):
        assert args.dpo
        output_tensor, metric = output_tensor_or_metric
        return output_tensor, partial(dpo_loss_func, metric)

    assert not args.dpo
    output_tensor = output_tensor_or_metric
    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_data_iter_provider(train_val_test_num_samples=None):
    """Build multimodal train, validation and test dataloaders."""
    args = get_args()
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


def add_qwen2vl_extra_args(parser):
    parser = gpatch_extra_args(parser)
    """Extra arguments."""
    group = parser.add_argument_group(title='qwen2vl/qwen2.5vl arguments')
    group.add_argument("--processor-path", type=str, default=None, help="")
    group.add_argument("--tarfile-path", type=str, default="/", help="")
    group.add_argument("--min-pixels-num", type=int, default=None, help="min image width * height")
    group.add_argument("--max-pixels-num", type=int, default=None, help="max image width * height")
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
        extra_args_provider=add_qwen2vl_extra_args,
        **extra_args,
    )
