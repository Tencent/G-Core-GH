import os
from pdb import run
import time
import inspect

import torch
from tqdm import tqdm
from transformers import AutoConfig

from megatron.training.checkpointing import save_checkpoint
from megatron.training import get_args

from gpatch.patch_mcore import init_gpatch_for_mcore
from tools.px_ckpt_conv import px_ckpt_conv, hf_to_mlm, mlm_to_hf
from tasks.gemma3.train_gemma3 import add_extra_args


def convert_siglip_hf_to_mlm(run_args, model_config, hf_vision_model, mlm_vision_model):
    # embedding
    mlm_vision_model.position_embeddings.weight.copy_(
        hf_vision_model.embeddings.position_embedding.weight
    )
    mlm_vision_model.conv1.weight.copy_(hf_vision_model.embeddings.patch_embedding.weight)
    mlm_vision_model.conv1.bias.copy_(hf_vision_model.embeddings.patch_embedding.bias)
    # post_layernorm
    mlm_vision_model.ln_post.weight.copy_(hf_vision_model.post_layernorm.weight)
    mlm_vision_model.ln_post.bias.copy_(hf_vision_model.post_layernorm.bias)
    # decoder
    num_layers = model_config['num_hidden_layers']

    args = get_args()
    use_te = (args.transformer_impl == 'transformer_engine')
    assert use_te, "only support transformer_engine now"
    model_arch = run_args.model_arch
    run_args.model_arch = "siglip"
    for layer_idx in tqdm(range(num_layers), "decoder layer states"):
        hf_layer = hf_vision_model.encoder.layers[layer_idx]
        lm_layer = mlm_vision_model.decoder.layers[layer_idx]

        # attn
        hf_to_mlm.set_hf2lm_attn_state(run_args, model_config, lm_layer, hf_layer)
        # layer_norm1
        if use_te:
            lm_layer.self_attention.linear_qkv.layer_norm_weight.copy_(hf_layer.layer_norm1.weight)
            lm_layer.self_attention.linear_qkv.layer_norm_bias.copy_(hf_layer.layer_norm1.bias)
        else:
            lm_layer.input_layernorm.weight.copy_(hf_layer.layer_norm1.weight)
            lm_layer.input_layernorm.bias.copy_(hf_layer.layer_norm1.bais)

        # mlp
        hf_to_mlm.set_hf2lm_mlp_state(run_args, lm_layer, hf_layer)

        # layer_norm2
        if use_te:
            lm_layer.mlp.linear_fc1.layer_norm_weight.copy_(hf_layer.layer_norm2.weight)
            lm_layer.mlp.linear_fc1.layer_norm_bias.copy_(hf_layer.layer_norm2.bias)
        else:
            lm_layer.pre_mlp_layernorm.copy_(hf_layer.layer_norm2.weight)
            lm_layer.pre_mlp_layernorm.copy_(hf_layer.layer_norm2.bias)
    run_args.model_arch = model_arch


def convert_gemma3_projector_hf_to_mlm(hf_model, mlm_model):
    hf_projector = hf_model.multi_modal_projector
    mlm_projector = mlm_model.vision_projection

    mlm_projector.mm_input_projection.layer_norm_weight.copy_(hf_projector.mm_soft_emb_norm.weight)
    mlm_projector.mm_input_projection.weight.copy_(hf_projector.mm_input_projection_weight.T)


def convert_gemma3_projector_mlm_to_hf(hf_model, mlm_model):
    hf_projector = hf_model.multi_modal_projector
    mlm_projector = mlm_model.vision_projection

    hf_projector.mm_soft_emb_norm.weight.copy_(mlm_projector.mm_input_projection.layer_norm_weight)
    hf_projector.mm_input_projection_weight.copy_(mlm_projector.mm_input_projection.weight.T)


def save_mlm_checkpoint(run_args, mlm_model):
    save_checkpoint(1, [mlm_model], None, None, num_floating_point_operations_so_far=0)

    old_name = os.path.join(run_args.megatron_save_dir, "iter_0000001")
    new_name = os.path.join(run_args.megatron_save_dir, "release")
    latesest_file = os.path.join(run_args.megatron_save_dir, "latest_checkpointed_iteration.txt")
    os.rename(old_name, new_name)
    with open(latesest_file, 'w') as f:
        f.write('release')
    print("successfully convert hf ckpt to megatron ckpt")


def convert_gemma3_hf_to_mlm(run_args, model_config, hf_model, mlm_model, with_save=False):
    # vision partment
    old_kv_channels = run_args.kv_channels
    run_args.kv_channels = None
    convert_siglip_hf_to_mlm(
        run_args=run_args,
        model_config=model_config['vision_config'],
        hf_vision_model=hf_model.vision_tower.vision_model,
        mlm_vision_model=mlm_model.vision_model,
    )
    run_args.kv_channels = old_kv_channels

    # projector partment
    convert_gemma3_projector_hf_to_mlm(
        hf_model=hf_model,
        mlm_model=mlm_model,
    )
    # text partment
    print(f"{mlm_model.language_model=} {hf_model.language_model=}")
    hf_to_mlm.convert_hf_to_mlm(
        run_args=run_args,
        model_config=model_config,
        lm_model=mlm_model.language_model,
        hf_model=hf_model.language_model,
        with_save=False,
    )

    if with_save:
        save_mlm_checkpoint(run_args, mlm_model)


### mlm -> hf
def convert_siglip_mlm_to_hf(run_args, model_config, hf_model, mlm_model):
    # embedding
    hf_model.embeddings.position_embedding.weight.copy_(mlm_model.position_embeddings.weight)
    hf_model.embeddings.patch_embedding.weight.copy_(mlm_model.conv1.weight)
    hf_model.embeddings.patch_embedding.bias.copy_(mlm_model.conv1.bias)
    # post_layernorm
    hf_model.post_layernorm.weight.copy_(mlm_model.ln_post.weight)
    hf_model.post_layernorm.bias.copy_(mlm_model.ln_post.bias)
    # decoder
    num_layers = model_config['num_hidden_layers']

    args = get_args()
    use_te = (args.transformer_impl == 'transformer_engine')
    model_arch = run_args.model_arch
    run_args.model_arch = "siglip"
    for layer_idx in tqdm(range(num_layers), "decoder layer states"):
        hf_layer = hf_model.encoder.layers[layer_idx]
        lm_layer = mlm_model.decoder.layers[layer_idx]

        # attn
        mlm_to_hf.set_lm2hf_attn_state(run_args, model_config, lm_layer, hf_layer)

        # layer_norm1
        if use_te:
            hf_layer.layer_norm1.weight.copy_(lm_layer.self_attention.linear_qkv.layer_norm_weight)
            hf_layer.layer_norm1.bias.copy_(lm_layer.self_attention.linear_qkv.layer_norm_bias)
        else:
            hf_layer.layer_norm1.weight.copy_(lm_layer.input_layernorm.weight)
            hf_layer.layer_norm1.bais.copy_(lm_layer.input_layernorm.bias)

        # mlp
        mlm_to_hf.set_lm2hf_mlp_state(run_args, lm_layer, hf_layer)

        # layer_norm2
        if use_te:
            hf_layer.layer_norm2.weight.copy_(lm_layer.mlp.linear_fc1.layer_norm_weight)
            hf_layer.layer_norm2.bias.copy_(lm_layer.mlp.linear_fc1.layer_norm_bias)
        else:
            hf_layer.layer_norm2.weight.copy_(lm_layer.pre_mlp_layernorm)
            hf_layer.layer_norm2.bias.copy_(lm_layer.pre_mlp_layernorm)
    run_args.model_arch = model_arch


def convert_gemma3_mlm_to_hf(run_args, model_config, hf_model, mlm_model):
    # vision partment
    old_kv_channels = run_args.kv_channels
    run_args.kv_channels = None
    convert_siglip_mlm_to_hf(
        run_args=run_args,
        model_config=model_config['vision_config'],
        hf_model=hf_model.vision_tower.vision_model,
        mlm_model=mlm_model.vision_model,
    )
    run_args.kv_channels = old_kv_channels
    # projector partment
    convert_gemma3_projector_mlm_to_hf(
        hf_model=hf_model,
        mlm_model=mlm_model,
    )
    # llm partment
    mlm_to_hf.convert_mlm_to_hf(
        run_args=run_args,
        model_config=model_config,
        lm_model=mlm_model.language_model,
        hf_model=hf_model.language_model,
        hf_tokenizer=None,
        save_ckpt=False,
    )
    if model_config['tie_word_embeddings']:
        delattr(hf_model.language_model, 'lm_head')
    # save huggingface format model
    t1 = time.time()
    print('HF model saving pretrained...')
    hf_model.save_pretrained(run_args.hf_save_dir, safe_serialization=False)
    cmd = f"cp {run_args.hf_config_json} {run_args.hf_save_dir}"
    os.system(cmd)
    generator_path = os.path.join(
        os.path.dirname(run_args.hf_config_json), 'generation_config.json'
    )
    cmd = f"cp {generator_path} {run_args.hf_save_dir}"
    os.system(cmd)

    t2 = time.time()
    print(f'converted MLM ckpt to HF ckpt successfully save:{t2 - t1}s')


def add_convert_gemma3_args(parser):
    group = parser.add_argument_group(title='qwen2vl dpo args')

    group.add_argument("--gemma3_dpo", action='store_true', help="dpo")
    group.add_argument(
        "--gemma3_dpo_choice_model",
        type=str,
        default="policy",
        choices=["policy", "reference"],
        help="选择dpo中的哪个模型来转换",
    )
    group.add_argument(
        "--gemma3_dpo_hf_ref_model",
        type=str,
        default='',
        help="""--gemma3_dpo_hf_ref_model：用于ref_model的hf路径
        --hf_load_dir：用于policy_model的路径""",
    )
    return parser


if __name__ == "__main__":
    print(f"\033[93m Now only support CUDA device \033[0m")
    init_gpatch_for_mcore()

    run_args = px_ckpt_conv.get_run_args(add_convert_gemma3_args)
    torch.set_grad_enabled(False)

    from tasks.gemma3.train_gemma3 import model_provider
    print(f"\033[93m {inspect.getsourcefile(model_provider)=} \033[0m")

    # TODO(jeffhong): remove tokenizer hardcode
    extra_argv = [
        "--model-arch",
        "gemma3",
        "--tokenizer-model",
        run_args.tokenizer_path,
    ]
    if run_args.gemma3_dpo:
        extra_argv.append("--dpo")

    if run_args.convert_way == "hf_to_mlm":
        mlm_model, model_config = px_ckpt_conv.create_mlm_model(
            run_args,
            only_create=True,
            model_provider_func=model_provider,
            extra_argv=extra_argv,
            extra_args_provider=add_extra_args,
            build_tokenizer=True,
        )
        from transformers import Gemma3ForConditionalGeneration
        hf_model = px_ckpt_conv.create_hf_model(
            run_args,
            only_create=False,
            model_class=Gemma3ForConditionalGeneration,
        )
        if not run_args.gemma3_dpo:
            convert_gemma3_hf_to_mlm(run_args, model_config, hf_model, mlm_model, True)
        else:
            assert run_args.gemma3_dpo_hf_ref_model != '', "--gemma3_dpo_hf_ref_model must be set"
            convert_gemma3_hf_to_mlm(
                run_args, model_config, hf_model, mlm_model.policy_model, False
            )
            src_hf_load_dir = run_args.hf_load_dir
            run_args.hf_load_dir = run_args.gemma3_dpo_hf_ref_model
            ref_hf_model = px_ckpt_conv.create_hf_model(
                run_args,
                only_create=False,
                model_class=Gemma3ForConditionalGeneration,
            )
            convert_gemma3_hf_to_mlm(
                run_args, model_config, ref_hf_model, mlm_model.ref_model, False
            )
            save_mlm_checkpoint(run_args, mlm_model)
            run_args.hf_load_dir = src_hf_load_dir
    elif run_args.convert_way == "mlm_to_hf":
        lm_model, model_config = px_ckpt_conv.create_mlm_model(
            run_args,
            only_create=False,
            model_provider_func=model_provider,
            extra_argv=extra_argv,
            extra_args_provider=add_extra_args,
            build_tokenizer=True,
        )
        from transformers import Gemma3ForConditionalGeneration
        hf_model = px_ckpt_conv.create_hf_model(
            run_args,
            only_create=True,
            model_class=Gemma3ForConditionalGeneration,
        )
        if not run_args.gemma3_dpo:
            convert_gemma3_mlm_to_hf(run_args, model_config, hf_model, lm_model)
        else:
            assert run_args.gemma3_dpo_choice_model in ["policy", "reference"]
            if run_args.gemma3_dpo_choice_model == "policy":
                convert_gemma3_mlm_to_hf(run_args, model_config, hf_model, lm_model.policy_model)
            else:
                convert_gemma3_mlm_to_hf(run_args, model_config, hf_model, lm_model.ref_model)
    else:
        raise NotImplementedError(f"convert way {run_args.convert_way} is not supported")
