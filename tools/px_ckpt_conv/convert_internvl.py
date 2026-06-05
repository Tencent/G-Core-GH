import os
from pdb import run
import time
import inspect

import torch
from tqdm import tqdm
from transformers import AutoModel

from megatron.training.checkpointing import save_checkpoint
from megatron.training import get_args

from gpatch.patch_mcore import init_gpatch_for_mcore
from tools.px_ckpt_conv import px_ckpt_conv, hf_to_mlm, mlm_to_hf
from tasks.internvl.train_internvl import add_extra_args
from tools.px_ckpt_conv import convert_qwen2p5vl


def set_hf2lm_mlp_state(lm_layer, hf_layer):
    lm_mlp = lm_layer.mlp
    hf_mlp = hf_layer.mlp

    lm_mlp.linear_fc1.weight.copy_(hf_mlp.fc1.weight)
    lm_mlp.linear_fc2.weight.copy_(hf_mlp.fc2.weight)

    lm_mlp.linear_fc1.bias.copy_(hf_mlp.fc1.bias)
    lm_mlp.linear_fc2.bias.copy_(hf_mlp.fc2.bias)


def set_mlm2hf_mlp_state(lm_layer, hf_layer):
    lm_mlp = lm_layer.mlp
    hf_mlp = hf_layer.mlp

    hf_mlp.fc1.weight.copy_(lm_mlp.linear_fc1.weight)
    hf_mlp.fc2.weight.copy_(lm_mlp.linear_fc2.weight)

    hf_mlp.fc1.bias.copy_(lm_mlp.linear_fc1.bias)
    hf_mlp.fc2.bias.copy_(lm_mlp.linear_fc2.bias)


def convert_internvit_hf_to_mlm(run_args, model_config, hf_vision_model, mlm_vision_model):
    # embedding
    mlm_vision_model.class_token.data.copy_(hf_vision_model.embeddings.class_embedding.data)
    mlm_vision_model.position_embedding.data.copy_(
        hf_vision_model.embeddings.position_embedding.data
    )
    mlm_vision_model.conv1.weight.copy_(hf_vision_model.embeddings.patch_embedding.weight)
    mlm_vision_model.conv1.bias.copy_(hf_vision_model.embeddings.patch_embedding.bias)

    # decoder
    num_layers = model_config['num_hidden_layers']
    model_config['num_heads'] = model_config['num_attention_heads']

    args = get_args()
    use_te = (args.transformer_impl == 'transformer_engine')
    assert use_te, "only support transformer_engine now"
    model_arch = run_args.model_arch
    run_args.model_arch = "internvit"
    for layer_idx in tqdm(range(num_layers), "internvl decoder layer states"):
        hf_layer = hf_vision_model.encoder.layers[layer_idx]
        lm_layer = mlm_vision_model.decoder.layers[layer_idx]

        # attn
        convert_qwen2p5vl.set_hf2lm_attn_state(run_args, model_config, lm_layer, hf_layer)
        # layer_norm1
        lm_layer.self_attention.linear_qkv.layer_norm_weight.copy_(hf_layer.norm1.weight)
        lm_layer.self_attention.linear_qkv.layer_norm_bias.copy_(hf_layer.norm1.bias)
        # mlp
        set_hf2lm_mlp_state(lm_layer, hf_layer)
        # layer_norm2
        lm_layer.mlp.linear_fc1.layer_norm_weight.copy_(hf_layer.norm2.weight)
        lm_layer.mlp.linear_fc1.layer_norm_bias.copy_(hf_layer.norm2.bias)
        # ls1, ls2
        lm_layer.ls1.copy_(hf_layer.ls1)
        lm_layer.ls2.copy_(hf_layer.ls2)
    run_args.model_arch = model_arch


def convert_internvl_projector_hf_to_mlm(hf_model, mlm_model):
    hf_projector = hf_model.mlp1
    mlm_projector = mlm_model.vision_projection.encoder

    mlm_projector.linear_fc1.layer_norm_weight.copy_(hf_projector[0].weight)
    mlm_projector.linear_fc1.layer_norm_bias.copy_(hf_projector[0].bias)
    mlm_projector.linear_fc1.weight.copy_(hf_projector[1].weight)
    mlm_projector.linear_fc1.bias.copy_(hf_projector[1].bias)
    mlm_projector.linear_fc2.weight.copy_(hf_projector[3].weight)
    mlm_projector.linear_fc2.bias.copy_(hf_projector[3].bias)


def convert_internvl_projector_mlm_to_hf(hf_model, mlm_model):
    hf_projector = hf_model.mlp1
    mlm_projector = mlm_model.vision_projection.encoder

    hf_projector[0].weight.copy_(mlm_projector.linear_fc1.layer_norm_weight)
    hf_projector[0].bias.copy_(mlm_projector.linear_fc1.layer_norm_bias)
    hf_projector[1].weight.copy_(mlm_projector.linear_fc1.weight)
    hf_projector[1].bias.copy_(mlm_projector.linear_fc1.bias)
    hf_projector[3].weight.copy_(mlm_projector.linear_fc2.weight)
    hf_projector[3].bias.copy_(mlm_projector.linear_fc2.bias)


def save_mlm_checkpoint(run_args, mlm_model):
    save_checkpoint(1, [mlm_model], None, None, num_floating_point_operations_so_far=0)

    old_name = os.path.join(run_args.megatron_save_dir, "iter_0000001")
    new_name = os.path.join(run_args.megatron_save_dir, "release")
    latesest_file = os.path.join(run_args.megatron_save_dir, "latest_checkpointed_iteration.txt")
    os.rename(old_name, new_name)
    with open(latesest_file, 'w') as f:
        f.write('release')
    print("successfully convert hf ckpt to megatron ckpt")


def convert_internvl_hf_to_mlm(run_args, model_config, hf_model, mlm_model, with_save=False):
    # vision partment
    old_kv_channels = run_args.kv_channels
    run_args.kv_channels = None
    convert_internvit_hf_to_mlm(
        run_args=run_args,
        model_config=model_config['vision_config'],
        hf_vision_model=hf_model.vision_model,
        mlm_vision_model=mlm_model.vision_model,
    )
    run_args.kv_channels = old_kv_channels

    # projector partment
    convert_internvl_projector_hf_to_mlm(
        hf_model=hf_model,
        mlm_model=mlm_model,
    )
    # text partment
    old_model_arch = run_args.model_arch
    run_args.model_arch = "qwen2vl"
    hf_to_mlm.convert_hf_to_mlm(
        run_args=run_args,
        model_config=model_config,
        lm_model=mlm_model.language_model,
        hf_model=hf_model.language_model,
        with_save=False,
    )
    run_args.model_arch = old_model_arch

    if with_save:
        save_mlm_checkpoint(run_args, mlm_model)


### mlm -> hf
def convert_internvit_mlm_to_hf(run_args, model_config, hf_vision_model, mlm_vision_model):
    # embedding
    hf_vision_model.embeddings.class_embedding.data.copy_(mlm_vision_model.class_token.data)
    hf_vision_model.embeddings.position_embedding.data.copy_(
        mlm_vision_model.position_embedding.data
    )
    hf_vision_model.embeddings.patch_embedding.weight.copy_(mlm_vision_model.conv1.weight)
    hf_vision_model.embeddings.patch_embedding.bias.copy_(mlm_vision_model.conv1.bias)

    # decoder
    num_layers = model_config['num_hidden_layers']
    model_config['num_heads'] = model_config['num_attention_heads']

    args = get_args()
    use_te = (args.transformer_impl == 'transformer_engine')
    assert use_te, "only support transformer_engine now"
    model_arch = run_args.model_arch
    run_args.model_arch = "internvit"
    for layer_idx in tqdm(range(num_layers), "internvl decoder layer states"):
        hf_layer = hf_vision_model.encoder.layers[layer_idx]
        lm_layer = mlm_vision_model.decoder.layers[layer_idx]

        # attn
        convert_qwen2p5vl.set_lm2hf_attn_state(run_args, model_config, lm_layer, hf_layer)
        # layer_norm1
        hf_layer.norm1.weight.copy_(lm_layer.self_attention.linear_qkv.layer_norm_weight)
        hf_layer.norm1.bias.copy_(lm_layer.self_attention.linear_qkv.layer_norm_bias)
        # mlp
        set_mlm2hf_mlp_state(lm_layer, hf_layer)
        # layer_norm2
        hf_layer.norm2.weight.copy_(lm_layer.mlp.linear_fc1.layer_norm_weight)
        hf_layer.norm2.bias.copy_(lm_layer.mlp.linear_fc1.layer_norm_bias)
        # ls1, ls2
        hf_layer.ls1.copy_(lm_layer.ls1)
        hf_layer.ls2.copy_(lm_layer.ls2)
    run_args.model_arch = model_arch


def convert_internvl_mlm_to_hf(run_args, model_config, hf_model, mlm_model):
    # vision partment
    old_kv_channels = run_args.kv_channels
    run_args.kv_channels = None
    convert_internvit_mlm_to_hf(
        run_args=run_args,
        model_config=model_config['vision_config'],
        hf_vision_model=hf_model.vision_model,
        mlm_vision_model=mlm_model.vision_model,
    )
    run_args.kv_channels = old_kv_channels
    # projector partment
    convert_internvl_projector_mlm_to_hf(
        hf_model=hf_model,
        mlm_model=mlm_model,
    )
    # llm partment
    old_model_arch = run_args.model_arch
    run_args.model_arch = "qwen2vl"
    mlm_to_hf.convert_mlm_to_hf(
        run_args=run_args,
        model_config=model_config,
        lm_model=mlm_model.language_model,
        hf_model=hf_model.language_model,
        hf_tokenizer=None,
        save_ckpt=False,
    )
    run_args.model_arch = old_model_arch
    if run_args.tie_word_embeddings:
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


if __name__ == "__main__":
    print(f"\033[93m Now only support CUDA device \033[0m")
    init_gpatch_for_mcore()

    run_args = px_ckpt_conv.get_run_args()
    torch.set_grad_enabled(False)

    from tasks.internvl.train_internvl import model_provider
    print(f"\033[93m {inspect.getsourcefile(model_provider)=} \033[0m")

    extra_argv = [
        "--model-arch",
        "internvl",
        "--add-qkv-bias",
        "--hf-model-path",
        run_args.tokenizer_path,
    ]
    if run_args.convert_way == "hf_to_mlm":
        hf_model = px_ckpt_conv.create_hf_model(
            run_args,
            only_create=False,
            model_class=AutoModel,
        )
        mlm_model, model_config = px_ckpt_conv.create_mlm_model(
            run_args,
            only_create=True,
            model_provider_func=model_provider,
            extra_argv=extra_argv,
            extra_args_provider=add_extra_args,
            build_tokenizer=True,
        )
        convert_internvl_hf_to_mlm(run_args, model_config, hf_model, mlm_model, True)
    elif run_args.convert_way == "mlm_to_hf":
        hf_model = px_ckpt_conv.create_hf_model(
            run_args,
            only_create=True,
            model_class=None,
        )
        lm_model, model_config = px_ckpt_conv.create_mlm_model(
            run_args,
            only_create=False,
            model_provider_func=model_provider,
            extra_argv=extra_argv,
            extra_args_provider=add_extra_args,
            build_tokenizer=True,
        )
        assert model_config['tie_word_embeddings'] == run_args.tie_word_embeddings
        convert_internvl_mlm_to_hf(run_args, model_config, hf_model, lm_model)
    else:
        raise NotImplementedError(f"convert way {run_args.convert_way} is not supported")
