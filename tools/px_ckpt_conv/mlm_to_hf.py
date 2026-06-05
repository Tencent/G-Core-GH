import argparse
import json
import os
import re
import shutil
import sys
import time
import tempfile

# import multi_device.platform

from accelerate import init_empty_weights
from tqdm import tqdm
from transformers.modeling_utils import no_init_weights
import torch
import transformers

from gpatch.core.device_type import is_wxacc1, is_wxacc2
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.legacy import fused_kernels
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args, set_global_variables
from megatron.training.initialize import _set_random_seed, _initialize_distributed
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
from megatron.training.utils import unwrap_model
from megatron.core.utils import get_model_config

from gpatch.core.models.gpt.weight_conversion.utils import merge_hf_lora_weight

from utils import read_json
from moe_mlm_to_hf import convert_moe_mlm_to_hf


def set_lm2hf_attn_state(run_args, model_config, lm_layer, hf_layer):
    lm_attn = lm_layer.self_attention
    if run_args.model_arch.lower() in [
        'llama',
        'bog',
        'baichuan-7b',
        'yi-9b',
        'qwen2-72b',
        'qwen2.5-1.5b',
        'qwen2vl',
        'dsr1-distill-qwen2.5-32b',
        'qwen2.5-math-rm-72b',
        'qwen2.5-math-1.5b',
        'qwen2.5-3b-instruct',
        'qwq-32b',
        'siglip',
        'gemma3',
        'qwen3',
    ]:
        hf_attn = hf_layer.self_attn
    elif run_args.model_arch.lower() in ['welm_19b']:
        hf_attn = hf_layer.attention
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')

    num_query_head = model_config["num_attention_heads"]
    num_kv_head = model_config.get("num_key_value_heads", num_query_head)
    dim = run_args.kv_channels
    if dim is None:
        dim = model_config['hidden_size'] // num_query_head
    assert num_query_head % num_kv_head == 0
    total_dim = 2 * dim + (dim * num_query_head // num_kv_head)

    linear_qkv = lm_attn.linear_qkv.weight.reshape((num_kv_head, total_dim, -1))

    wq = linear_qkv.narrow(1, 0,
                           dim * num_query_head // num_kv_head).reshape((dim * num_query_head, -1))
    wk = linear_qkv.narrow(1, dim * num_query_head // num_kv_head,
                           dim).reshape((dim * num_kv_head, -1))
    wv = linear_qkv.narrow(1, dim * num_query_head // num_kv_head + dim,
                           dim).reshape((dim * num_kv_head, -1))
    if run_args.model_arch.lower() in [
        "llama",
        'bog',
        'yi-9b',
        'qwen2-72b',
        'qwen2.5-1.5b',
        'qwen2vl',
        'dsr1-distill-qwen2.5-32b',
        'qwen2.5-math-rm-72b',
        'qwen2.5-math-1.5b',
        'qwen2.5-3b-instruct',
        'qwq-32b',
        'siglip',
        'gemma3',
        'qwen3',
    ]:
        hf_attn.q_proj.weight.data.copy_(wq)
        hf_attn.k_proj.weight.data.copy_(wk)
        hf_attn.v_proj.weight.data.copy_(wv)
        if (run_args.model_arch.lower() == 'siglip'):
            hf_attn.out_proj.weight.data.copy_(lm_attn.linear_proj.weight)
        else:
            hf_attn.o_proj.weight.data.copy_(lm_attn.linear_proj.weight)

        if run_args.model_arch.lower() in [
            'qwen2-72b',
            'qwen2.5-1.5b',
            'qwen2vl',
            'dsr1-distill-qwen2.5-32b',
            'qwen2.5-math-rm-72b',
            'qwen2.5-math-1.5b',
            'qwen2.5-3b-instruct',
            'qwq-32b',
            'siglip',
        ]:
            mlm_bias = lm_attn.linear_qkv.bias.reshape((num_kv_head, total_dim))
            q_bias = mlm_bias.narrow(1, 0, dim * num_query_head // num_kv_head).reshape(
                (dim * num_query_head)
            )
            k_bias = mlm_bias.narrow(1, dim * num_query_head // num_kv_head,
                                     dim).reshape((dim * num_kv_head))
            v_bias = mlm_bias.narrow(1, dim * num_query_head // num_kv_head + dim,
                                     dim).reshape((dim * num_kv_head))
            hf_attn.q_proj.bias.data.copy_(q_bias)
            hf_attn.k_proj.bias.data.copy_(k_bias)
            hf_attn.v_proj.bias.data.copy_(v_bias)
            if (run_args.model_arch.lower() == 'siglip'):
                hf_attn.out_proj.bias.data.copy_(lm_attn.linear_proj.bias)

    elif run_args.model_arch.lower() in ["baichuan-7b"]:
        hf_attn.W_pack.weight.data.copy_(torch.cat([wq, wk, wv], dim=0))
        hf_attn.o_proj.weight.data.copy_(lm_attn.linear_proj.weight)
    elif run_args.model_arch.lower() in ["welm_19b"]:
        hf_attn.query_key_value.weight.data.copy_(lm_attn.linear_qkv.weight)
        hf_attn.query_key_value.bias.data.copy_(lm_attn.linear_qkv.bias)
        hf_attn.dense.weight.data.copy_(lm_attn.linear_proj.weight)
        hf_attn.dense.bias.data.copy_(lm_attn.linear_proj.bias)
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')


def set_lm2hf_mlp_state(run_args, lm_layer, hf_layer):
    lm_mlp = lm_layer.mlp
    hf_mlp = hf_layer.mlp
    if run_args.model_arch.lower() in [
        'llama',
        'bog',
        'baichuan-7b',
        'yi-9b',
        'qwen2-72b',
        'qwen2.5-1.5b',
        'qwen2vl',
        'dsr1-distill-qwen2.5-32b',
        'qwen2.5-math-rm-72b',
        'qwen2.5-math-1.5b',
        'qwen2.5-3b-instruct',
        'qwq-32b',
        'gemma3',
        'qwen3',
    ]:
        assert lm_mlp.linear_fc1.weight.data.shape[0] % 2 == 0
        split_size = lm_mlp.linear_fc1.weight.data.shape[0] // 2

        linear_fc1_weight = torch.split(lm_mlp.linear_fc1.weight, split_size)
        hf_mlp.gate_proj.weight.data.copy_(linear_fc1_weight[0])
        hf_mlp.up_proj.weight.data.copy_(linear_fc1_weight[1])
        hf_mlp.down_proj.weight.data.copy_(lm_mlp.linear_fc2.weight)
    elif run_args.model_arch.lower() in ['welm_19b']:
        hf_mlp.dense_h_to_4h.weight.data.copy_(lm_mlp.linear_fc1.weight)
        hf_mlp.dense_h_to_4h.bias.data.copy_(lm_mlp.linear_fc1.bias)
        hf_mlp.dense_4h_to_h.weight.data.copy_(lm_mlp.linear_fc2.weight)
        hf_mlp.dense_4h_to_h.bias.data.copy_(lm_mlp.linear_fc2.bias)
    elif run_args.model_arch.lower() in ['siglip']:
        hf_mlp_fc1 = hf_mlp.fc1
        hf_mlp_fc2 = hf_mlp.fc2
        hf_mlp_fc1.weight.data.copy_(lm_mlp.linear_fc1.weight)
        hf_mlp_fc1.bias.data.copy_(lm_mlp.linear_fc1.bias)
        hf_mlp_fc2.weight.data.copy_(lm_mlp.linear_fc2.weight)
        hf_mlp_fc2.bias.data.copy_(lm_mlp.linear_fc2.bias)
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')


@torch.no_grad()
def convert_mlm_to_hf(run_args, model_config, lm_model, hf_model, hf_tokenizer, save_ckpt=True):
    print('HF tokenizer saving pretrained...')
    if hf_tokenizer is not None:
        hf_tokenizer.save_pretrained(run_args.hf_save_dir)

    # merge the lora matric to src matric: m = m + lora_a * lora_b
    if run_args.enable_lora:
        lm_model_config = get_model_config(lm_model)
        state_dict = lm_model.state_dict()
        for name, param in lm_model.named_parameters():
            merge_param = merge_hf_lora_weight(name, param, state_dict, lm_model_config)
            param.copy_(merge_param)

    if run_args.model_arch.lower() in ['bog-moe', 'welm_moe_32b', 'qwen1.5-moe', 'qwen3-moe']:
        assert not run_args.enable_lora, f"Not supported yet, can be support later"
        return convert_moe_mlm_to_hf(
            run_args, model_config, lm_model=lm_model, hf_model=hf_model, save_ckpt=save_ckpt
        )

    t0 = time.time()

    print('copying parameters...')
    # embedding
    if run_args.model_arch.lower() in [
        'llama',
        'bog',
        'baichuan-7b',
        'yi-9b',
        'qwen2-72b',
        'qwen2.5-1.5b',
        'qwen2vl',
        'dsr1-distill-qwen2.5-32b',
        'qwen2.5-math-rm-72b',
        'qwen2.5-math-1.5b',
        'qwen2.5-3b-instruct',
        'qwq-32b',
        'gemma3',
        'qwen3',
    ]:
        hf_vocab_size = hf_model.model.embed_tokens.weight.shape[0]
        hf_model.model.embed_tokens.weight.data.copy_(
            lm_model.embedding.word_embeddings.weight[:hf_vocab_size]
        )
    elif run_args.model_arch.lower() in ["welm_19b"]:
        hf_vocab_size = hf_model.lm.embed_in.weight.shape[0]
        hf_model.lm.embed_in.weight.data.copy_(
            lm_model.embedding.word_embeddings.weight[:hf_vocab_size]
        )
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')

    # decoder
    if run_args.model_arch.lower() in [
        'llama',
        'bog',
        'baichuan-7b',
        'yi-9b',
        'qwen2-72b',
        'qwen2.5-1.5b',
        'qwen2vl',
        'dsr1-distill-qwen2.5-32b',
        'qwen2.5-math-rm-72b',
        'qwen2.5-math-1.5b',
        'qwen2.5-3b-instruct',
        'qwq-32b',
        'gemma3',
        'qwen3',
    ]:
        num_layers = model_config["num_hidden_layers"]
    elif run_args.model_arch.lower() in ["welm_19b"]:
        num_layers = model_config["num_layers"]
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')
    with_qk_norm = model_config.get("qk_layernorm", False)
    if run_args.model_arch.lower() in ["gemma3"]:
        with_qk_norm = True
    if with_qk_norm:
        assert run_args.model_arch.lower() in [
            "bog",
            "gemma3",
            'qwen3',
        ]

    for layer_idx in tqdm(range(num_layers), "copy decoder layer states"):
        lm_layer = lm_model.decoder.layers[layer_idx]
        if run_args.model_arch.lower() in [
            'llama',
            'bog',
            'baichuan-7b',
            'yi-9b',
            'qwen2-72b',
            'qwen2.5-1.5b',
            'qwen2vl',
            'dsr1-distill-qwen2.5-32b',
            'qwen2.5-math-rm-72b',
            'qwen2.5-math-1.5b',
            'qwen2.5-3b-instruct',
            'qwq-32b',
            'gemma3',
            'qwen3',
        ]:
            hf_layer = hf_model.model.layers[layer_idx]
        elif run_args.model_arch.lower() in ["welm_19b"]:
            hf_layer = hf_model.lm.layers[layer_idx]
        else:
            raise ValueError(f'unknown arch {run_args.model_arch}')

        set_lm2hf_attn_state(run_args, model_config, lm_layer, hf_layer)
        set_lm2hf_mlp_state(run_args, lm_layer, hf_layer)
        if is_wxacc1() or is_wxacc2():
            hf_layer.input_layernorm.weight.data.copy_(lm_layer.input_layernorm.weight)
            if run_args.model_arch.lower() in ["welm_19b"]:
                hf_layer.input_layernorm.bias.data.copy_(lm_layer.input_layernorm.bias)
            hf_layer.post_attention_layernorm.weight.data.copy_(lm_layer.pre_mlp_layernorm.weight)
            if run_args.model_arch.lower() in ["welm_19b"]:
                hf_layer.post_attention_layernorm.bias.data.copy_(lm_layer.pre_mlp_layernorm.bias)
        else:
            hf_layer.input_layernorm.weight.data.copy_(
                lm_layer.self_attention.linear_qkv.layer_norm_weight
            )
            if run_args.model_arch.lower() in ["welm_19b"]:
                hf_layer.input_layernorm.bias.data.copy_(
                    lm_layer.self_attention.linear_qkv.layer_norm_bias
                )
            hf_layer.post_attention_layernorm.weight.data.copy_(
                lm_layer.mlp.linear_fc1.layer_norm_weight
            )
            if run_args.model_arch.lower() in ["welm_19b"]:
                hf_layer.post_attention_layernorm.bias.data.copy_(
                    lm_layer.mlp.linear_fc1.layer_norm_bias
                )
        if with_qk_norm:
            if run_args.model_arch.lower() in ['bog']:
                hf_layer.self_attn.q_prenorm.weight.data.copy_(
                    lm_layer.self_attention.q_layernorm.weight
                )
                hf_layer.self_attn.k_prenorm.weight.data.copy_(
                    lm_layer.self_attention.k_layernorm.weight
                )
            elif run_args.model_arch.lower() in ['gemma3']:
                hf_layer.self_attn.q_norm.weight.copy_(lm_layer.self_attention.q_layernorm.weight)
                hf_layer.self_attn.k_norm.weight.copy_(lm_layer.self_attention.k_layernorm.weight)
            elif run_args.model_arch.lower() in ['qwen3']:
                hf_layer.self_attn.q_norm.weight.data.copy_(
                    lm_layer.self_attention.q_layernorm.weight
                )
                hf_layer.self_attn.k_norm.weight.data.copy_(
                    lm_layer.self_attention.k_layernorm.weight
                )
            else:
                raise NotImplementedError(f"arch {run_args.model_arch}")

        # gemma3多了两个norm及位置不一样
        if run_args.model_arch.lower() in ['gemma3']:
            hf_layer.pre_feedforward_layernorm.weight.copy_(
                lm_layer.mlp.linear_fc1.layer_norm_weight
            )
            hf_layer.post_attention_layernorm.weight.copy_(lm_layer.post_attention_layernorm.weight)
            hf_layer.post_feedforward_layernorm.weight.copy_(
                lm_layer.post_feedforward_layernorm.weight
            )

    # output layer
    if run_args.model_arch.lower() in [
        'llama',
        'bog',
        'baichuan-7b',
        'yi-9b',
        'qwen2-72b',
        'qwen2.5-1.5b',
        'qwen2vl',
        'dsr1-distill-qwen2.5-32b',
        'qwen2.5-math-rm-72b',
        'qwen2.5-math-1.5b',
        'qwen2.5-3b-instruct',
        'qwq-32b',
        'gemma3',
        'qwen3',
    ]:
        hf_model.model.norm.weight.data.copy_(lm_model.decoder.final_layernorm.weight)
    elif run_args.model_arch.lower() in ['welm_19b']:
        hf_model.lm.final_layer_norm.weight.data.copy_(lm_model.decoder.final_layernorm.weight)
        hf_model.lm.final_layer_norm.bias.data.copy_(lm_model.decoder.final_layernorm.bias)
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')

    has_embed_out = hasattr(hf_model, 'embed_out')
    if not run_args.tie_word_embeddings and not model_config.get('tie_word_embeddings', False):
        if run_args.model_arch.lower() in [
            'llama',
            'bog',
            'baichuan-7b',
            'yi-9b',
            'qwen2-72b',
            'qwen2.5-1.5b',
            'qwen2vl',
            'dsr1-distill-qwen2.5-32b',
            'qwq-32b',
            'qwen3',
        ]:
            hf_model.lm_head.weight.data.copy_(lm_model.output_layer.weight[:hf_vocab_size])
        elif run_args.model_arch.lower() in ['welm_19b']:
            if has_embed_out:
                hf_model.embed_out.weight.data.copy_(lm_model.output_layer.weight[:hf_vocab_size])
        else:
            raise ValueError(f'unknown arch {run_args.model_arch}')

    if not save_ckpt:
        return

    t1 = time.time()
    print('HF model saving pretrained...')

    hf_model.save_pretrained(run_args.hf_save_dir)

    if os.environ.get("PX_INSPECET_MODEL", "0") == "1":
        if torch.distributed.get_rank() == 0:
            for pname, param in hf_model.named_parameters():
                print(f"after convert {pname=} {param.shape} {param.sum()} {param}")

    t2 = time.time()
    print(
        f'''converted MLM ckpt to HF ckpt successfully
t1 - t0 {t1 - t0}
t2 - t1 {t2 - t1}
    '''
    )
