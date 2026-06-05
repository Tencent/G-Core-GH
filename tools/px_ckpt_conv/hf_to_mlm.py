import argparse
import json
import os
import re
import shutil
import sys
import time
import tempfile
import math

from accelerate import init_empty_weights
from tqdm import tqdm
from transformers.modeling_utils import no_init_weights
import torch
import transformers
import torch.nn.init as init

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.legacy import fused_kernels
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args, set_global_variables
from megatron.training.initialize import _set_random_seed, _initialize_distributed
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
from megatron.training.utils import unwrap_model
from gpatch.core.device_type import is_wxacc1

from gpatch.core.device_type import is_wxacc1, is_wxacc2
from moe_hf_to_mlm import convert_moe_hf_to_mlm
from utils import read_json


def set_hf2lm_attn_state(run_args, model_config, lm_layer, hf_layer):
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
    hidden_size = model_config["hidden_size"]
    assert num_query_head % num_kv_head == 0

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
        lm_attn.linear_qkv.weight.data.copy_(
            torch.cat(
                [
                    hf_attn.q_proj.weight.reshape(
                        (num_kv_head, dim * num_query_head // num_kv_head, hidden_size)
                    ),
                    hf_attn.k_proj.weight.reshape((num_kv_head, dim, hidden_size)),
                    hf_attn.v_proj.weight.reshape((num_kv_head, dim, hidden_size)),
                ],
                dim=1
            ).reshape(-1, hidden_size)
        )
        if run_args.model_arch.lower() in ['siglip']:
            lm_attn.linear_proj.weight.data.copy_(hf_attn.out_proj.weight)
        else:
            lm_attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)
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
            lm_attn.linear_qkv.bias.data.copy_(
                torch.cat(
                    [
                        hf_attn.q_proj.bias.reshape(
                            (num_kv_head, dim * num_query_head // num_kv_head)
                        ),
                        hf_attn.k_proj.bias.reshape((num_kv_head, dim)),
                        hf_attn.v_proj.bias.reshape((num_kv_head, dim)),
                    ],
                    dim=1
                ).reshape(-1)
            )
            if run_args.model_arch.lower() in [
                'siglip',
            ]:
                lm_attn.linear_proj.bias.copy_(hf_attn.out_proj.bias)
    elif run_args.model_arch.lower() in ["baichuan-7b"]:
        W_pack = hf_attn.W_pack.weight
        concate_hidden_size = W_pack.shape[0]

        Wq = W_pack[0:concate_hidden_size // 3, :]
        Wk = W_pack[concate_hidden_size // 3:concate_hidden_size // 3 * 2, :]
        Wv = W_pack[-concate_hidden_size // 3:, :]

        lm_attn.linear_qkv.weight.data.copy_(
            torch.cat(
                [
                    Wq.reshape((num_kv_head, dim * num_query_head // num_kv_head, -1)),
                    Wk.reshape((num_kv_head, dim, -1)),
                    Wv.reshape((num_kv_head, dim, -1)),
                ],
                dim=1
            ).reshape(-1, hidden_size)
        )
        lm_attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)
    elif run_args.model_arch.lower() in ["welm_19b"]:
        lm_attn.linear_qkv.weight.data.copy_(hf_attn.query_key_value.weight)
        lm_attn.linear_qkv.bias.data.copy_(hf_attn.query_key_value.bias)
        lm_attn.linear_proj.weight.data.copy_(hf_attn.dense.weight)
        lm_attn.linear_proj.bias.data.copy_(hf_attn.dense.bias)
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')


def set_hf2lm_mlp_state(run_args, lm_layer, hf_layer):
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
        lm_mlp.linear_fc1.weight.data.copy_(
            torch.cat([
                hf_mlp.gate_proj.weight,
                hf_mlp.up_proj.weight,
            ], dim=0)
        )
        lm_mlp.linear_fc2.weight.data.copy_(hf_mlp.down_proj.weight)
    elif run_args.model_arch.lower() in ['welm_19b']:
        lm_mlp.linear_fc1.weight.data.copy_(hf_mlp.dense_h_to_4h.weight)
        lm_mlp.linear_fc1.bias.data.copy_(hf_mlp.dense_h_to_4h.bias)
        lm_mlp.linear_fc2.weight.data.copy_(hf_mlp.dense_4h_to_h.weight)
        lm_mlp.linear_fc2.bias.data.copy_(hf_mlp.dense_4h_to_h.bias)
    elif run_args.model_arch.lower() in ['siglip']:
        hf_mlp_fc1 = hf_mlp.fc1
        hf_mlp_fc2 = hf_mlp.fc2
        lm_mlp.linear_fc1.weight.data.copy_(hf_mlp_fc1.weight)
        lm_mlp.linear_fc1.bias.data.copy_(hf_mlp_fc1.bias)
        lm_mlp.linear_fc2.weight.data.copy_(hf_mlp_fc2.weight)
        lm_mlp.linear_fc2.bias.data.copy_(hf_mlp_fc2.bias)
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')


def convert_hf_to_mlm(run_args, model_config, hf_model, lm_model, with_save=True):

    if run_args.model_arch.lower() in ['bog-moe', 'welm_moe_32b', 'qwen1.5-moe', 'qwen3-moe']:
        assert not run_args.enable_lora, f"Not supported yet, can be support later"
        return convert_moe_hf_to_mlm(
            run_args, model_config, hf_model=hf_model, lm_model=lm_model, with_save=with_save
        )

    # embedding
    if run_args.model_arch.lower() in [
        'llama',
        'bog',
        'baichuan-7b',
        'yi-9b',
        'qwen2-72b',
        'qwen2.5-1.5b',
        'qwen2vl',
        'siglip',
        'dsr1-distill-qwen2.5-32b',
        'qwen2.5-math-rm-72b',
        'qwen2.5-math-1.5b',
        'qwen2.5-3b-instruct',
        'qwq-32b',
        'gemma3',
        'qwen3',
    ]:
        extra_vocab = lm_model.embedding.word_embeddings.weight.data.shape[
            0] - hf_model.model.embed_tokens.weight.shape[0]
        embed_dim = hf_model.model.embed_tokens.weight.shape[1]
        extra_zeros = torch.zeros(
            (extra_vocab, embed_dim), dtype=hf_model.model.embed_tokens.weight.dtype
        )
        padded_embed = torch.cat((hf_model.model.embed_tokens.weight, extra_zeros), dim=0)
    elif run_args.model_arch.lower() in ['welm_19b']:
        extra_vocab = lm_model.embedding.word_embeddings.weight.data.shape[
            0] - hf_model.lm.embed_in.weight.shape[0]
        embed_dim = hf_model.lm.embed_in.weight.shape[1]
        extra_zeros = torch.zeros((extra_vocab, embed_dim), dtype=hf_model.lm.embed_in.weight.dtype)
        padded_embed = torch.cat((hf_model.lm.embed_in.weight, extra_zeros), dim=0)
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')
    lm_model.embedding.word_embeddings.weight.data.copy_(padded_embed)

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

    for layer_idx in tqdm(range(num_layers), "decoder layer states"):
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

        set_hf2lm_attn_state(run_args, model_config, lm_layer, hf_layer)
        set_hf2lm_mlp_state(run_args, lm_layer, hf_layer)

        if is_wxacc1() or is_wxacc2():
            lm_layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight)
            if run_args.model_arch.lower() in ['welm_19b']:
                lm_layer.input_layernorm.bias.data.copy_(hf_layer.input_layernorm.bias)
            lm_layer.pre_mlp_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)
            if run_args.model_arch.lower() in ['welm_19b']:
                lm_layer.pre_mlp_layernorm.bias.data.copy_(hf_layer.post_attention_layernorm.bias)
        else:
            lm_layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(
                hf_layer.input_layernorm.weight
            )
            if run_args.model_arch.lower() in ['welm_19b']:
                lm_layer.self_attention.linear_qkv.layer_norm_bias.data.copy_(
                    hf_layer.input_layernorm.bias
                )
            lm_layer.mlp.linear_fc1.layer_norm_weight.data.copy_(
                hf_layer.post_attention_layernorm.weight
            )
            if run_args.model_arch.lower() in ['welm_19b']:
                lm_layer.mlp.linear_fc1.layer_norm_bias.data.copy_(
                    hf_layer.post_attention_layernorm.bias
                )
        if with_qk_norm:
            if run_args.model_arch.lower() in ['bog']:
                lm_layer.self_attention.q_layernorm.weight.data.copy_(
                    hf_layer.self_attn.q_prenorm.weight
                )
                lm_layer.self_attention.k_layernorm.weight.data.copy_(
                    hf_layer.self_attn.k_prenorm.weight
                )
            elif run_args.model_arch.lower() in ['gemma3']:
                lm_layer.self_attention.q_layernorm.weight.copy_(hf_layer.self_attn.q_norm.weight)
                lm_layer.self_attention.k_layernorm.weight.copy_(hf_layer.self_attn.k_norm.weight)
            elif run_args.model_arch.lower() in ['qwen3']:
                lm_layer.self_attention.q_layernorm.weight.data.copy_(
                    hf_layer.self_attn.q_norm.weight
                )
                lm_layer.self_attention.k_layernorm.weight.data.copy_(
                    hf_layer.self_attn.k_norm.weight
                )
            else:
                raise NotImplementedError(f"{run_args.model_arch.lower()} has no qknorm")
        # gemma3多了两个norm及位置不一样
        if run_args.model_arch.lower() in ['gemma3']:
            lm_layer.mlp.linear_fc1.layer_norm_weight.copy_(
                hf_layer.pre_feedforward_layernorm.weight
            )
            lm_layer.post_attention_layernorm.weight.copy_(hf_layer.post_attention_layernorm.weight)
            lm_layer.post_feedforward_layernorm.weight.copy_(
                hf_layer.post_feedforward_layernorm.weight
            )

    # output layer
    # TODO(@xiaotaoliu): assert not args.untie_embeddings_and_output_weights
    if run_args.model_arch.lower() in [
        'llama',
        'bog',
        'baichuan-7b',
        'yi-9b',
        'qwen2-72b',
        'qwen2vl',
        'dsr1-distill-qwen2.5-32b',
        'qwq-32b',
        'gemma3',
        'qwen3',
    ]:
        lm_model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)

        if not run_args.tie_word_embeddings and not model_config.get('tie_word_embeddings', False):
            extra_vocab = lm_model.output_layer.weight.data.shape[
                0] - hf_model.lm_head.weight.shape[0]
            embed_dim = hf_model.lm_head.weight.shape[1]
            extra_zeros = torch.zeros((extra_vocab, embed_dim), dtype=hf_model.lm_head.weight.dtype)
            padded_output_layer = torch.cat((hf_model.lm_head.weight, extra_zeros), dim=0)
            lm_model.output_layer.weight.data.copy_(padded_output_layer)
    elif run_args.model_arch.lower() in ["welm_19b"]:
        lm_model.decoder.final_layernorm.weight.data.copy_(hf_model.lm.final_layer_norm.weight)
        lm_model.decoder.final_layernorm.bias.data.copy_(hf_model.lm.final_layer_norm.bias)
        # welm19b is tie_embeddings_and_output_weights
        assert run_args.tie_word_embeddings
    elif run_args.model_arch.lower() in [
        'qwen2.5-1.5b',
        'qwen2.5-math-1.5b',
        'qwen2.5-math-rm-72b',
        'qwen2.5-3b-instruct',
    ]:
        lm_model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')

    if not with_save:
        return

    # init lora param
    if run_args.enable_lora:
        for name, param in lm_model.named_parameters():
            if '.lora_a.' in name:
                init.kaiming_uniform_(param, a=math.sqrt(5))
            if '.lora_b.' in name:
                init.zeros_(param)

    save_checkpoint(1, [lm_model], None, None, num_floating_point_operations_so_far=0)
    if os.environ.get("PX_INSPECET_MODEL", "0") == "1":
        if torch.distributed.get_rank() == 0:
            for pname, param in lm_model.named_parameters():
                print(f"after convert {pname=} {param.shape} {param.sum()} {param}")

    old_name = os.path.join(run_args.megatron_save_dir, "iter_0000001")
    new_name = os.path.join(run_args.megatron_save_dir, "release")
    latesest_file = os.path.join(run_args.megatron_save_dir, "latest_checkpointed_iteration.txt")
    os.rename(old_name, new_name)
    with open(latesest_file, 'w') as f:
        f.write('release')
    print("successfully convert hf ckpt to megatron ckpt")
