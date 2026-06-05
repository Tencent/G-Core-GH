import os
import gc

import torch

from megatron.core import mpu
from megatron.legacy import fused_kernels
from megatron.training import get_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_global_variables
from megatron.training.initialize import _set_random_seed, _initialize_distributed
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core import mpu
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.spec_utils import import_module


def sft_provider(pre_process=True, post_process=True):
    args = get_args()
    config = core_transformer_config_from_args(get_args())
    assert args.use_mcore_models
    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            args.num_experts, args.moe_grouped_gemm
        )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.px_rope_base,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
    )
    return model


def load_sft():
    model = sft_provider(
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=mpu.is_pipeline_last_stage(),
    )
    _, _ = load_checkpoint([model], None, None)
    sd = model.state_dict()
    cpu_sd = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        cpu_sd[k] = v.cpu()

    del sd
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return cpu_sd


def load_semantic_rank(state_dict):
    from tasks.semantic_rank.finetune_semantic_rank_bog import model_provider
    rank_model = model_provider(
        pre_process=mpu.is_pipeline_first_stage(),
        post_process=mpu.is_pipeline_last_stage(),
    )

    rank_model.load_state_dict(state_dict, strict=False)
    return rank_model


def main():
    from tasks.semantic_rank.finetune_semantic_rank_bog import add_embed_recall_args
    args = parse_args(extra_args_provider=add_embed_recall_args)

    args = validate_args(args)
    set_global_variables(args, build_tokenizer=True)
    args = get_args()
    _initialize_distributed()
    _set_random_seed(args.seed, args.data_parallel_random_init)
    fused_kernels.load(args)
    torch.distributed.barrier()

    sd = load_sft()
    torch.distributed.barrier()
    rank_model = load_semantic_rank(sd)
    torch.distributed.barrier()
    save_checkpoint(1, [rank_model], None, None, num_floating_point_operations_so_far=0)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        os.rename(os.path.join(args.save, 'iter_0000001'), os.path.join(args.save, 'release'))
        with open(os.path.join(args.save, 'latest_checkpointed_iteration.txt'), 'w') as outf:
            outf.write('release')


if __name__ == "__main__":
    main()
