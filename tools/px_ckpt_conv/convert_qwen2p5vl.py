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
import xxhash

from megatron.core.utils import get_model_config
from megatron.training.checkpointing import save_checkpoint

from tools.px_ckpt_conv import px_ckpt_conv, hf_to_mlm, mlm_to_hf
from gpatch.core.models.gpt.weight_conversion.utils import merge_hf_lora_weight
from gpatch.core.device_type import is_wxacc2


class Qwen2p5VLOldVersion:
    model = None


def set_hf2lm_attn_state(run_args, model_config, lm_layer, hf_layer):
    lm_attn = lm_layer.self_attention
    hf_attn = hf_layer.attn

    vision_hidden_size = model_config['hidden_size']
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

    vision_hidden_size = model_config['hidden_size']
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

    lm_mlp.linear_fc1.weight.copy_(
        torch.cat(
            [
                hf_mlp.gate_proj.weight,
                hf_mlp.up_proj.weight,
            ],
            dim=0,
        )
    )
    lm_mlp.linear_fc2.weight.copy_(hf_mlp.down_proj.weight)

    lm_mlp.linear_fc1.bias.copy_(
        torch.cat(
            [
                hf_mlp.gate_proj.bias,
                hf_mlp.up_proj.bias,
            ],
            dim=0,
        )
    )
    lm_mlp.linear_fc2.bias.copy_(hf_mlp.down_proj.bias)


def set_lm2hf_mlp_state(lm_layer, hf_layer):
    lm_mlp = lm_layer.mlp
    hf_mlp = hf_layer.mlp

    assert lm_mlp.linear_fc1.weight.shape[0] % 2 == 0
    split_size = lm_mlp.linear_fc1.weight.shape[0] // 2

    linear_fc1_weight = torch.split(lm_mlp.linear_fc1.weight, split_size)
    hf_mlp.gate_proj.weight.copy_(linear_fc1_weight[0])
    hf_mlp.up_proj.weight.copy_(linear_fc1_weight[1])
    hf_mlp.down_proj.weight.copy_(lm_mlp.linear_fc2.weight)

    assert lm_mlp.linear_fc1.bias.shape[0] % 2 == 0
    split_size = lm_mlp.linear_fc1.bias.shape[0] // 2

    linear_fc1_bias = torch.split(lm_mlp.linear_fc1.bias, split_size)
    hf_mlp.gate_proj.bias.copy_(linear_fc1_bias[0])
    hf_mlp.up_proj.bias.copy_(linear_fc1_bias[1])
    hf_mlp.down_proj.bias.copy_(lm_mlp.linear_fc2.bias)


def convert_qwen2p5vl_vision_hf_to_mlm(run_args, model_config, hf_model, mlm_model):
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
        if is_wxacc2():
            lm_layer.input_layernorm.weight.copy_(hf_layer.norm1.weight)
        else:
            lm_layer.self_attention.linear_qkv.layer_norm_weight.copy_(hf_layer.norm1.weight)
        # mlp
        set_hf2lm_mlp_state(lm_layer, hf_layer)
        # layer_norm2
        if is_wxacc2():
            lm_layer.pre_mlp_layernorm.weight.copy_(hf_layer.norm2.weight)
        else:
            lm_layer.mlp.linear_fc1.layer_norm_weight.copy_(hf_layer.norm2.weight)


def convert_qwen2p5vl_vision_mlm_to_hf(run_args, model_config, hf_model, mlm_model):
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
        if is_wxacc2():
            hf_layer.norm1.weight.copy_(lm_layer.input_layernorm.weight)
        else:
            hf_layer.norm1.weight.copy_(lm_layer.self_attention.linear_qkv.layer_norm_weight)
        # mlp
        set_lm2hf_mlp_state(lm_layer, hf_layer)
        # layer_norm2
        if is_wxacc2():
            hf_layer.norm2.weight.copy_(lm_layer.pre_mlp_layernorm.weight)
        else:
            hf_layer.norm2.weight.copy_(lm_layer.mlp.linear_fc1.layer_norm_weight)


def convert_qwen2p5vl_projector_hf_to_mlm(hf_model, mlm_model):
    hfprojector = hf_model.merger
    mgprojector = mlm_model.projection
    mlm_model.decoder.final_layernorm.weight.copy_(hfprojector.ln_q.weight)

    mgprojector.encoder.linear_fc1.weight.copy_(hfprojector.mlp[0].weight)
    mgprojector.encoder.linear_fc1.bias.copy_(hfprojector.mlp[0].bias)
    mgprojector.encoder.linear_fc2.weight.copy_(hfprojector.mlp[2].weight)
    mgprojector.encoder.linear_fc2.bias.copy_(hfprojector.mlp[2].bias)


def convert_qwen2p5vl_projector_mlm_to_hf(hf_model, mlm_model):
    hfprojector = hf_model.merger
    mgprojector = mlm_model.projection
    hfprojector.ln_q.weight.copy_(mlm_model.decoder.final_layernorm.weight)

    hfprojector.mlp[0].weight.copy_(mgprojector.encoder.linear_fc1.weight)
    hfprojector.mlp[0].bias.copy_(mgprojector.encoder.linear_fc1.bias)
    hfprojector.mlp[2].weight.copy_(mgprojector.encoder.linear_fc2.weight)
    hfprojector.mlp[2].bias.copy_(mgprojector.encoder.linear_fc2.bias)


def save_mlm_checkpoint(run_args, mlm_model):
    # init lora param
    with torch.random.fork_rng():  # this ensures local random
        if run_args.enable_lora:
            for name, param in mlm_model.named_parameters():
                if '.lora_a.' in name:
                    # a stable hashing that returns a consistent seed given the same param_name
                    seed = xxhash.xxh64(name).intdigest() % (2**32)
                    torch.manual_seed(seed)
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


def convert_qwen2p5vl_hf_to_mlm(run_args, model_config, hf_model, mlm_model, with_save=True):
    # llm partment、
    hf_llm_model = hf_model
    if version.parse(transformers.__version__) >= version.parse("4.52.0"):
        hf_llm_model = Qwen2p5VLOldVersion()
        # 这里是为了兼容 transformers > 4.52.0 版本的模型结构，
        # 开发这个用的是 transformers == 4.51.3 的结构
        hf_llm_model.model = hf_model.model.language_model
    hf_to_mlm.convert_hf_to_mlm(
        run_args=run_args,
        model_config=model_config,
        lm_model=mlm_model.language_model,
        hf_model=hf_llm_model,
        with_save=False,
    )
    # vision partment
    if version.parse(transformers.__version__) >= version.parse("4.52.0"):
        hf_vit_model = hf_model.model.visual
    else:
        hf_vit_model = hf_model.visual
    convert_qwen2p5vl_vision_hf_to_mlm(
        run_args=run_args,
        model_config=model_config['vision_config'],
        hf_model=hf_vit_model,
        mlm_model=mlm_model.vision_model,
    )
    # projector partment
    convert_qwen2p5vl_projector_hf_to_mlm(
        hf_model=hf_vit_model,
        mlm_model=mlm_model.vision_model,
    )
    if with_save:
        save_mlm_checkpoint(run_args, mlm_model)


def convert_qwen2p5vl_mlm_to_hf(run_args, model_config, hf_model, mlm_model):
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
    hf_llm_model = hf_model
    if version.parse(transformers.__version__) >= version.parse("4.52.0"):
        hf_llm_model = Qwen2p5VLOldVersion()
        # 这里是为了兼容 transformers < 4.52.0 版本的模型结构，因为开发这个置换时用的正是这个结构
        hf_llm_model.model = hf_model.model.language_model
    mlm_to_hf.convert_mlm_to_hf(
        run_args=run_args,
        model_config=model_config,
        lm_model=mlm_model.language_model,
        hf_model=hf_llm_model,
        hf_tokenizer=None,
        save_ckpt=False,
    )
    # vision partment
    if version.parse(transformers.__version__) >= version.parse("4.52.0"):
        hf_vit_model = hf_model.model.visual
    else:
        hf_vit_model = hf_model.visual
    convert_qwen2p5vl_vision_mlm_to_hf(
        run_args=run_args,
        model_config=model_config['vision_config'],
        hf_model=hf_vit_model,
        mlm_model=mlm_model.vision_model,
    )
    # projector partment
    convert_qwen2p5vl_projector_mlm_to_hf(
        hf_model=hf_vit_model,
        mlm_model=mlm_model.vision_model,
    )
    if model_config['tie_word_embeddings']:
        delattr(hf_model, 'lm_head')
    # save huggingface format model
    t1 = time.time()
    print('HF model saving pretrained...')
    hf_model.save_pretrained(run_args.hf_save_dir, safe_serialization=False)

    # copy the original configs for vllm inference
    original_hf_dir = run_args.hf_config_json.replace('config.json', '')
    jsons_to_cp = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.json",\
                   "preprocessor_config.json", "vocab.json", 'generation_config.json']
    for json_to_cp in jsons_to_cp:
        cmd = f"cp {original_hf_dir}/{json_to_cp} {run_args.hf_save_dir}"
        os.system(cmd)

    if os.environ.get("PX_INSPECET_MODEL", "0") == "1":
        if torch.distributed.get_rank() == 0:
            for pname, param in hf_model.named_parameters():
                print(f"after convert {pname=} {param.shape} {param.sum()} {param}")

    t2 = time.time()
    print(f'converted MLM ckpt to HF ckpt successfully save:{t2 - t1}s')


if __name__ == "__main__":
    from tools.px_ckpt_conv.convert_qwen2vl import add_convert_qwen2vl_args
    run_args = px_ckpt_conv.get_run_args(add_convert_qwen2vl_args)
    torch.set_grad_enabled(False)

    assert version.parse(transformers.__version__) <= version.parse("4.51.3")
    # NOTE(guanyouhe): 这个是相对引入
    from tasks.qwen2vl.train_qwen2vl import model_provider, add_qwen2vl_extra_args
    print(f"\033[93m {inspect.getsourcefile(model_provider)=} \033[0m")

    config = AutoConfig.from_pretrained(os.path.dirname(run_args.hf_config_json))
    model_config = config.to_dict()

    extra_argv = ["--model-arch", "qwen2.5vl"]
    if run_args.enable_lora:
        extra_argv.extend(
            ["--mm-freeze-llm", "--mm-freeze-vision-encoder", "--mm-freeze-projector"]
        )
    if run_args.qwen2vl_dpo:
        extra_argv.append("--dpo")
    if run_args.convert_way == "hf_to_mlm":
        lm_model, model_config = px_ckpt_conv.create_mlm_model(
            run_args,
            only_create=True,
            model_provider_func=model_provider,
            extra_argv=extra_argv,
            extra_args_provider=add_qwen2vl_extra_args,
        )
        # qwen huggingface model
        from transformers import Qwen2_5_VLForConditionalGeneration
        hf_model = px_ckpt_conv.create_hf_model(
            run_args,
            only_create=False,
            model_class=Qwen2_5_VLForConditionalGeneration,
        )
        if not run_args.qwen2vl_dpo:
            convert_qwen2p5vl_hf_to_mlm(run_args, model_config, hf_model, lm_model)
        else:
            assert run_args.qwen2vl_dpo_hf_ref_model != '', "--qwen2vl_dpo_hf_ref_model must be set"
            convert_qwen2p5vl_hf_to_mlm(
                run_args, model_config, hf_model, lm_model.policy_model, False
            )
            src_hf_load_dir = run_args.hf_load_dir
            run_args.hf_load_dir = run_args.qwen2vl_dpo_hf_ref_model
            ref_hf_model = px_ckpt_conv.create_hf_model(
                run_args,
                only_create=False,
                model_class=Qwen2_5_VLForConditionalGeneration,
            )
            convert_qwen2p5vl_hf_to_mlm(
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
        from transformers import Qwen2_5_VLForConditionalGeneration
        hf_model = px_ckpt_conv.create_hf_model(
            run_args,
            only_create=True,
            model_class=Qwen2_5_VLForConditionalGeneration,
        )
        if not run_args.qwen2vl_dpo:
            convert_qwen2p5vl_mlm_to_hf(run_args, model_config, hf_model, lm_model)
        else:
            assert run_args.qwen2vl_dpo_choice_model in ["policy", "reference"]
            if run_args.qwen2vl_dpo_choice_model == "policy":
                convert_qwen2p5vl_mlm_to_hf(run_args, model_config, hf_model, lm_model.policy_model)
            else:
                convert_qwen2p5vl_mlm_to_hf(run_args, model_config, hf_model, lm_model.ref_model)
    else:
        raise NotImplementedError(f"convert way {run_args.convert_way} is not supported")
