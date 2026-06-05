import os
import time
import inspect
import math
from packaging import version

import torch
import torch.distributed
from tqdm import tqdm
import torch.nn.init as init
import transformers
from transformers import AutoConfig

from megatron.core.utils import get_model_config
from megatron.training.checkpointing import save_checkpoint
from tools.px_ckpt_conv import px_ckpt_conv, hf_to_mlm, mlm_to_hf
from gpatch.core.models.gpt.weight_conversion.utils import merge_hf_lora_weight


def set_hf2lm_attn_state(run_args, model_config, lm_layer, hf_layer):
    lm_attn = lm_layer.self_attention
    hf_attn = hf_layer.attn

    vision_hidden_size = model_config['embed_dim']
    # num_query_groups 与 num_attention_heads一样
    vision_num_query_groups = model_config['num_heads']
    vision_head_dim = vision_hidden_size // model_config['num_heads']

    lm_attn.linear_qkv.weight.copy_(
        hf_attn.qkv.weight.view(
            3, vision_num_query_groups, -1, vision_head_dim, vision_hidden_size
        ).transpose(0, 1).flatten(1, 2).reshape(-1, vision_hidden_size).contiguous()
    )
    lm_attn.linear_qkv.bias.copy_(
        hf_attn.qkv.bias.view(3, vision_num_query_groups,
                              -1).transpose(0, 1).flatten(1, 2).view(-1).contiguous()
    )
    lm_attn.linear_proj.weight.copy_(hf_attn.proj.weight)
    lm_attn.linear_proj.bias.copy_(hf_attn.proj.bias)


def set_lm2hf_attn_state(run_args, model_config, lm_layer, hf_layer):
    lm_attn = lm_layer.self_attention
    hf_attn = hf_layer.attn

    vision_hidden_size = model_config['embed_dim']
    # num_query_groups 与 num_attention_heads一样
    vision_num_query_groups = model_config['num_heads']
    vision_head_dim = vision_hidden_size // model_config['num_heads']

    hf_attn.qkv.weight.copy_(
        lm_attn.linear_qkv.weight.view(
            vision_num_query_groups, 3, -1, vision_head_dim, vision_hidden_size
        ).transpose(0, 1).reshape(-1, vision_hidden_size).contiguous()
    )
    hf_attn.qkv.bias.copy_(
        lm_attn.linear_qkv.bias.view(vision_num_query_groups, 3,
                                     -1).transpose(0, 1).reshape(-1).contiguous()
    )
    hf_attn.proj.weight.copy_(lm_attn.linear_proj.weight)
    hf_attn.proj.bias.copy_(lm_attn.linear_proj.bias)


def set_hf2lm_mlp_state(lm_layer, hf_layer):
    lm_mlp = lm_layer.mlp
    hf_mlp = hf_layer.mlp

    lm_mlp.linear_fc1.weight.copy_(hf_mlp.fc1.weight)
    lm_mlp.linear_fc1.bias.copy_(hf_mlp.fc1.bias)
    lm_mlp.linear_fc2.weight.copy_(hf_mlp.fc2.weight)
    lm_mlp.linear_fc2.bias.copy_(hf_mlp.fc2.bias)


def set_lm2hf_mlp_state(lm_layer, hf_layer):
    lm_mlp = lm_layer.mlp
    hf_mlp = hf_layer.mlp

    hf_mlp.fc1.weight.copy_(lm_mlp.linear_fc1.weight)
    hf_mlp.fc1.bias.copy_(lm_mlp.linear_fc1.bias)
    hf_mlp.fc2.weight.copy_(lm_mlp.linear_fc2.weight)
    hf_mlp.fc2.bias.copy_(lm_mlp.linear_fc2.bias)


def convert_qwen2vl_vision_hf_to_mlm(run_args, model_config, hf_model, mlm_model):
    # embedding
    mlm_model.patch_embed.proj.weight.copy_(hf_model.patch_embed.proj.weight)

    # decoder
    num_layers = model_config['depth']
    for layer_idx in tqdm(range(num_layers), "decoder layer states"):
        hf_layer = hf_model.blocks[layer_idx]
        lm_layer = mlm_model.decoder.layers[layer_idx]
        # attn
        set_hf2lm_attn_state(run_args, model_config, lm_layer, hf_layer)
        # layer_norm1
        lm_layer.self_attention.linear_qkv.layer_norm_weight.copy_(hf_layer.norm1.weight)
        lm_layer.self_attention.linear_qkv.layer_norm_bias.copy_(hf_layer.norm1.bias)
        # mlp
        set_hf2lm_mlp_state(lm_layer, hf_layer)
        # layer_norm2
        lm_layer.mlp.linear_fc1.layer_norm_weight.copy_(hf_layer.norm2.weight)
        lm_layer.mlp.linear_fc1.layer_norm_bias.copy_(hf_layer.norm2.bias)


def convert_qwen2vl_vision_mlm_to_hf(run_args, model_config, hf_model, mlm_model):
    # embedding
    hf_model.patch_embed.proj.weight.copy_(mlm_model.patch_embed.proj.weight)

    # decoder
    num_layers = model_config['depth']
    for layer_idx in tqdm(range(num_layers), "decoder layer states"):
        hf_layer = hf_model.blocks[layer_idx]
        lm_layer = mlm_model.decoder.layers[layer_idx]
        # attn
        set_lm2hf_attn_state(run_args, model_config, lm_layer, hf_layer)
        # layer_norm1
        hf_layer.norm1.weight.copy_(lm_layer.self_attention.linear_qkv.layer_norm_weight)
        hf_layer.norm1.bias.copy_(lm_layer.self_attention.linear_qkv.layer_norm_bias)
        # mlp
        set_lm2hf_mlp_state(lm_layer, hf_layer)
        # layer_norm2
        hf_layer.norm2.weight.copy_(lm_layer.mlp.linear_fc1.layer_norm_weight)
        hf_layer.norm2.bias.copy_(lm_layer.mlp.linear_fc1.layer_norm_bias)


def convert_qwen2vl_projector_hf_to_mlm(hf_model, mlm_model):
    hfprojector = hf_model.merger
    mgprojector = mlm_model.projection
    mlm_model.decoder.final_layernorm.weight.copy_(hfprojector.ln_q.weight)
    mlm_model.decoder.final_layernorm.bias.copy_(hfprojector.ln_q.bias)

    mgprojector.encoder.linear_fc1.weight.copy_(hfprojector.mlp[0].weight)
    mgprojector.encoder.linear_fc1.bias.copy_(hfprojector.mlp[0].bias)
    mgprojector.encoder.linear_fc2.weight.copy_(hfprojector.mlp[2].weight)
    mgprojector.encoder.linear_fc2.bias.copy_(hfprojector.mlp[2].bias)


def convert_qwen2vl_projector_mlm_to_hf(hf_model, mlm_model):
    hfprojector = hf_model.merger
    mgprojector = mlm_model.projection
    hfprojector.ln_q.weight.copy_(mlm_model.decoder.final_layernorm.weight)
    hfprojector.ln_q.bias.copy_(mlm_model.decoder.final_layernorm.bias)

    hfprojector.mlp[0].weight.copy_(mgprojector.encoder.linear_fc1.weight)
    hfprojector.mlp[0].bias.copy_(mgprojector.encoder.linear_fc1.bias)
    hfprojector.mlp[2].weight.copy_(mgprojector.encoder.linear_fc2.weight)
    hfprojector.mlp[2].bias.copy_(mgprojector.encoder.linear_fc2.bias)


def save_mlm_checkpoint(run_args, mlm_model):
    # init lora param
    if run_args.enable_lora:
        for name, param in mlm_model.named_parameters():
            if '.lora_a.' in name:
                init.kaiming_uniform_(param, a=math.sqrt(5))
            if '.lora_b.' in name:
                init.zeros_(param)

    # save
    save_checkpoint(1, [mlm_model], None, None, num_floating_point_operations_so_far=0)

    old_name = os.path.join(run_args.megatron_save_dir, "iter_0000001")
    new_name = os.path.join(run_args.megatron_save_dir, "release")
    latesest_file = os.path.join(run_args.megatron_save_dir, "latest_checkpointed_iteration.txt")
    os.rename(old_name, new_name)
    with open(latesest_file, 'w') as f:
        f.write('release')
    print("successfully convert hf ckpt to megatron ckpt")


def convert_qwen2vl_hf_to_mlm(run_args, model_config, hf_model, mlm_model, with_save=True):
    # llm partment
    hf_to_mlm.convert_hf_to_mlm(
        run_args=run_args,
        model_config=model_config,
        lm_model=mlm_model.language_model,
        hf_model=hf_model,
        with_save=False,
    )
    # vision partment
    convert_qwen2vl_vision_hf_to_mlm(
        run_args=run_args,
        model_config=model_config['vision_config'],
        hf_model=hf_model.visual,
        mlm_model=mlm_model.vision_model,
    )
    # projector partment
    convert_qwen2vl_projector_hf_to_mlm(
        hf_model=hf_model.visual,
        mlm_model=mlm_model.vision_model,
    )
    if with_save:
        save_mlm_checkpoint(run_args, mlm_model)


def convert_qwen2vl_mlm_to_hf(run_args, model_config, hf_model, mlm_model):
    # merge the lora matric to src matric: m = m + lora_a * lora_b
    if run_args.enable_lora:
        # llm partment
        lm_model_config = get_model_config(lm_model.language_model)
        state_dict = lm_model.language_model.state_dict()
        for name, param in lm_model.language_model.named_parameters():
            merge_param = merge_hf_lora_weight(name, param, state_dict, lm_model_config)
            param.copy_(merge_param)
        # vision partment
        lm_model_config = get_model_config(lm_model.vision_model)
        state_dict = lm_model.vision_model.state_dict()
        for name, param in lm_model.vision_model.named_parameters():
            merge_param = merge_hf_lora_weight(name, param, state_dict, lm_model_config)
            param.copy_(merge_param)

    # llm partment
    mlm_to_hf.convert_mlm_to_hf(
        run_args=run_args,
        model_config=model_config,
        lm_model=mlm_model.language_model,
        hf_model=hf_model,
        hf_tokenizer=None,
        save_ckpt=False,
    )
    # vision partment
    convert_qwen2vl_vision_mlm_to_hf(
        run_args=run_args,
        model_config=model_config['vision_config'],
        hf_model=hf_model.visual,
        mlm_model=mlm_model.vision_model,
    )
    # projector partment
    convert_qwen2vl_projector_mlm_to_hf(
        hf_model=hf_model.visual,
        mlm_model=mlm_model.vision_model,
    )
    if model_config['tie_word_embeddings']:
        delattr(hf_model, 'lm_head')
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


def add_convert_qwen2vl_args(parser):
    group = parser.add_argument_group(title='qwen2vl dpo args')

    group.add_argument("--qwen2vl_dpo", action='store_true', help="dpo")
    group.add_argument(
        "--qwen2vl_dpo_choice_model",
        type=str,
        default="policy",
        choices=["policy", "reference"],
        help="选择dpo中的哪个模型来转换",
    )
    group.add_argument(
        "--qwen2vl_dpo_hf_ref_model",
        type=str,
        default='',
        help="""--qwen2vl_dpo_hf_ref_model：用于ref_model的hf路径
        --hf_load_dir：用于policy_model的路径""",
    )
    return parser


if __name__ == "__main__":
    print(f"\033[93m Now only support CUDA device \033[0m")
    run_args = px_ckpt_conv.get_run_args(add_convert_qwen2vl_args)
    torch.set_grad_enabled(False)

    assert version.parse(transformers.__version__) <= version.parse("4.51.3")
    # NOTE(guanyouhe): 这个是相对引入
    from tasks.qwen2vl.train_qwen2vl import model_provider, add_qwen2vl_extra_args
    print(f"\033[93m {inspect.getsourcefile(model_provider)=} \033[0m")

    config = AutoConfig.from_pretrained(os.path.dirname(run_args.hf_config_json))
    model_config = config.to_dict()

    extra_argv = ["--model-arch", "qwen2vl"]
    if run_args.qwen2vl_dpo:
        extra_argv.append("--dpo")
    if run_args.enable_lora:
        extra_argv.extend(
            ["--mm-freeze-llm", "--mm-freeze-vision-encoder", "--mm-freeze-projector"]
        )
    if run_args.convert_way == "hf_to_mlm":
        lm_model, model_config = px_ckpt_conv.create_mlm_model(
            run_args,
            only_create=True,
            model_provider_func=model_provider,
            extra_argv=extra_argv,
            extra_args_provider=add_qwen2vl_extra_args,
        )
        # qwen huggingface model
        from transformers import Qwen2VLForConditionalGeneration
        hf_model = px_ckpt_conv.create_hf_model(
            run_args,
            only_create=False,
            model_class=Qwen2VLForConditionalGeneration,
        )
        if not run_args.qwen2vl_dpo:
            convert_qwen2vl_hf_to_mlm(run_args, model_config, hf_model, lm_model, True)
        else:
            assert run_args.qwen2vl_dpo_hf_ref_model != '', "--qwen2vl_dpo_hf_ref_model must be set"
            convert_qwen2vl_hf_to_mlm(
                run_args, model_config, hf_model, lm_model.policy_model, False
            )
            src_hf_load_dir = run_args.hf_load_dir
            run_args.hf_load_dir = run_args.qwen2vl_dpo_hf_ref_model
            ref_hf_model = px_ckpt_conv.create_hf_model(
                run_args,
                only_create=False,
                model_class=Qwen2VLForConditionalGeneration,
            )
            convert_qwen2vl_hf_to_mlm(
                run_args, model_config, ref_hf_model, lm_model.ref_model, False
            )
            save_mlm_checkpoint(run_args, lm_model)
            run_args.hf_load_dir = src_hf_load_dir
    elif run_args.convert_way == "mlm_to_hf":
        lm_model, model_config = px_ckpt_conv.create_mlm_model(
            run_args,
            only_create=False,
            model_provider_func=model_provider,
            extra_argv=extra_argv,
            extra_args_provider=add_qwen2vl_extra_args,
        )
        # qwen huggingface model
        from transformers import Qwen2VLForConditionalGeneration
        hf_model = px_ckpt_conv.create_hf_model(
            run_args,
            only_create=True,
            model_class=Qwen2VLForConditionalGeneration,
        )
        if not run_args.qwen2vl_dpo:
            convert_qwen2vl_mlm_to_hf(run_args, model_config, hf_model, lm_model)
        else:
            assert run_args.qwen2vl_dpo_choice_model in ["policy", "reference"]
            if run_args.qwen2vl_dpo_choice_model == "policy":
                convert_qwen2vl_mlm_to_hf(run_args, model_config, hf_model, lm_model.policy_model)
            else:
                convert_qwen2vl_mlm_to_hf(run_args, model_config, hf_model, lm_model.ref_model)
    else:
        raise NotImplementedError(f"convert way {run_args.convert_way} is not supported")
