import os
# import multi_device.platform

from tqdm import tqdm
import torch

from gpatch.core.device_type import is_wxacc1
from megatron.training.checkpointing import save_checkpoint


def set_moe_hf2lm_attn_state(run_args, model_config, lm_layer, hf_layer):
    lm_attn = lm_layer.self_attention
    if run_args.model_arch.lower() in ['bog-moe', 'welm_moe_32b', 'qwen1.5-moe', 'qwen3-moe']:
        hf_attn = hf_layer.self_attn
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')

    num_query_head = model_config["num_attention_heads"]
    num_kv_head = model_config.get("num_key_value_heads", num_query_head)
    dim = run_args.kv_channels
    if dim is None:
        dim = model_config['hidden_size'] // num_query_head
    hidden_size = model_config["hidden_size"]
    assert num_query_head % num_kv_head == 0

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
    lm_attn.linear_proj.weight.data.copy_(hf_attn.o_proj.weight)

    if run_args.model_arch.lower() in ['welm_moe_32b', 'qwen1.5-moe']:
        lm_attn.linear_qkv.bias.data.copy_(
            torch.cat(
                [
                    hf_attn.q_proj.bias.reshape((num_kv_head, dim * num_query_head // num_kv_head)),
                    hf_attn.k_proj.bias.reshape((num_kv_head, dim)),
                    hf_attn.v_proj.bias.reshape((num_kv_head, dim)),
                ],
                dim=1
            ).reshape(-1)
        )


def set_moe_hf2lm_mlp_state(run_args, model_config, lm_layer, hf_layer):
    num_shared_experts_key = "n_shared_experts" if run_args.model_arch.lower() in [
        'bog-moe'
    ] else "num_shared_experts"
    num_experts_key = "n_routed_experts" if run_args.model_arch.lower() in [
        'bog-moe'
    ] else "num_experts"
    num_experts = model_config[num_experts_key]
    px_has_shared_experts = model_config.get(num_shared_experts_key, 0) > 0
    px_has_shared_expert_gate = model_config.get('has_shared_expert_gate', False)
    lm_mlp = lm_layer.mlp
    hf_mlp = hf_layer.mlp

    # gate
    lm_mlp.router.weight.data.copy_(hf_mlp.gate.weight)
    if run_args.escore_bias_inherit:
        lm_mlp.router.e_score_correction_bias.data.copy_(hf_mlp.gate.e_score_correction_bias)

    # experts
    if run_args.use_te_grouped_gemm:
        # TE grouped gemm
        for expert_i in range(num_experts):
            w1 = torch.cat(
                [
                    hf_mlp.experts[expert_i].gate_proj.weight,
                    hf_mlp.experts[expert_i].up_proj.weight
                ],
                dim=0
            )
            w2 = hf_mlp.experts[expert_i].down_proj.weight
            getattr(lm_mlp.experts.linear_fc1, f"weight{expert_i}").data.copy_(w1)
            getattr(lm_mlp.experts.linear_fc2, f"weight{expert_i}").data.copy_(w2)
    elif not is_wxacc1():
        # deprecated
        # MLM grouped gemm
        expert_fc1_weights = []
        expert_fc2_weights = []
        for expert_i in range(num_experts):
            w1 = torch.cat(
                [
                    hf_mlp.experts[expert_i].gate_proj.weight,
                    hf_mlp.experts[expert_i].up_proj.weight
                ],
                dim=0
            )
            w2 = hf_mlp.experts[expert_i].down_proj.weight
            expert_fc1_weights.append(w1)
            expert_fc2_weights.append(w2)
        lm_mlp.experts.weight1.data.copy_(torch.cat(expert_fc1_weights, dim=0).T)
        lm_mlp.experts.weight2.data.copy_(torch.cat(expert_fc2_weights, dim=1).T)
    else:
        # no group gemm
        for expert_i in range(num_experts):
            w1 = torch.cat(
                [
                    hf_mlp.experts[expert_i].gate_proj.weight,
                    hf_mlp.experts[expert_i].up_proj.weight
                ],
                dim=0
            )
            w2 = hf_mlp.experts[expert_i].down_proj.weight
            lm_mlp.experts.local_experts[expert_i].linear_fc1.weight.data.copy_(w1)
            lm_mlp.experts.local_experts[expert_i].linear_fc2.weight.data.copy_(w2)

    if px_has_shared_experts:
        assert run_args.model_arch.lower() not in ['qwen3-moe']
        shared_expert_name = "shared_experts" if run_args.model_arch.lower() in [
            'bog-moe'
        ] else "shared_expert"
        shared_expert_obj = getattr(hf_mlp, f"{shared_expert_name}")
        w_g = shared_expert_obj.gate_proj.weight
        w_u = shared_expert_obj.up_proj.weight
        w_d = shared_expert_obj.down_proj.weight

        lm_mlp.shared_experts.linear_fc1.weight.data.copy_(torch.cat([
            w_g,
            w_u,
        ], dim=0))
        lm_mlp.shared_experts.linear_fc2.weight.data.copy_(w_d)
        if px_has_shared_expert_gate:
            assert run_args.model_arch.lower() not in ['bog-moe']
            if is_wxacc1():
                lm_mlp.shared_experts.gate_weight.data.copy_(hf_mlp.shared_expert_gate.weight)
            else:
                lm_mlp.shared_expert_gate.weight.data.copy_(hf_mlp.shared_expert_gate.weight)


def set_moe_hf2lm_dense_mlp_state(run_args, model_config, lm_layer, hf_layer):
    lm_mlp = lm_layer.mlp
    hf_mlp = hf_layer.mlp

    if run_args.model_arch.lower() in ['bog-moe']:
        lm_mlp.linear_fc1.weight.data.copy_(
            torch.cat([
                hf_mlp.gate_proj.weight,
                hf_mlp.up_proj.weight,
            ], dim=0)
        )
        lm_mlp.linear_fc2.weight.data.copy_(hf_mlp.down_proj.weight)
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')


def set_hf2mlm_layernorm(run_args, model_config, lm_layer, hf_layer, with_qk_norm=False):
    if is_wxacc1():
        lm_layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight)
        lm_layer.pre_mlp_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)
    else:
        lm_layer.self_attention.linear_qkv.layer_norm_weight.data.copy_(
            hf_layer.input_layernorm.weight
        )
        lm_layer.pre_mlp_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)
    # qk norm
    if with_qk_norm:
        if run_args.model_arch.lower() in ['bog-moe']:
            # bog moe 为 rmsnorm, 所以没有 bias赋值
            lm_layer.self_attention.q_layernorm.weight.data.copy_(
                hf_layer.self_attn.q_prenorm.weight
            )
            lm_layer.self_attention.k_layernorm.weight.data.copy_(
                hf_layer.self_attn.k_prenorm.weight
            )
        elif run_args.model_arch.lower() in ['qwen3-moe']:
            lm_layer.self_attention.q_layernorm.weight.data.copy_(hf_layer.self_attn.q_norm.weight)
            lm_layer.self_attention.k_layernorm.weight.data.copy_(hf_layer.self_attn.k_norm.weight)
        else:
            raise NotImplementedError(f"{run_args.model_arch.lower()} has no qknorm")


def convert_moe_hf_to_mlm(run_args, model_config, hf_model, lm_model, with_save=True):
    # embedding
    if run_args.model_arch.lower() in ['bog-moe', 'welm_moe_32b', 'qwen1.5-moe', 'qwen3-moe']:
        extra_vocab = lm_model.embedding.word_embeddings.weight.data.shape[
            0] - hf_model.model.embed_tokens.weight.shape[0]
        embed_dim = hf_model.model.embed_tokens.weight.shape[1]
        extra_zeros = torch.zeros(
            (extra_vocab, embed_dim), dtype=hf_model.model.embed_tokens.weight.dtype
        )
        padded_embed = torch.cat((hf_model.model.embed_tokens.weight, extra_zeros), dim=0)
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')
    lm_model.embedding.word_embeddings.weight.data.copy_(padded_embed)

    # decoder
    if run_args.model_arch.lower() in ["bog-moe", "welm_moe_32b", "qwen1.5-moe", "qwen3-moe"]:
        num_layers = model_config["num_hidden_layers"]
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')
    moe_first_k_dense_replace = model_config.get('first_k_dense_replace', 0)
    num_layers -= moe_first_k_dense_replace

    with_qk_norm = model_config.get("qk_layernorm", False)
    if with_qk_norm:
        assert run_args.model_arch.lower() in ["bog-moe", "qwen3-moe"]

    for layer_idx in tqdm(
        range(moe_first_k_dense_replace), "copy decoder first_k_dense_layer states"
    ):
        lm_layer = lm_model.decoder.first_k_dense_layers[layer_idx]
        if run_args.model_arch.lower() in ['bog-moe']:
            hf_layer = hf_model.model.layers[layer_idx]
        else:
            raise ValueError(f'unknown arch {run_args.model_arch}')
        set_moe_hf2lm_attn_state(run_args, model_config, lm_layer, hf_layer)
        set_moe_hf2lm_dense_mlp_state(run_args, model_config, lm_layer, hf_layer)
        set_hf2mlm_layernorm(run_args, model_config, lm_layer, hf_layer, with_qk_norm=with_qk_norm)

    for layer_idx in tqdm(range(num_layers), "decoder layer states"):
        lm_layer = lm_model.decoder.layers[layer_idx]
        if run_args.model_arch.lower() in ['bog-moe', 'welm_moe_32b', 'qwen1.5-moe', 'qwen3-moe']:
            hf_layer = hf_model.model.layers[layer_idx + moe_first_k_dense_replace]
        else:
            raise ValueError(f'unknown arch {run_args.model_arch}')

        set_moe_hf2lm_attn_state(run_args, model_config, lm_layer, hf_layer)
        set_moe_hf2lm_mlp_state(run_args, model_config, lm_layer, hf_layer)
        set_hf2mlm_layernorm(run_args, model_config, lm_layer, hf_layer, with_qk_norm=with_qk_norm)

    # output layer
    # TODO(@xiaotaoliu): assert not args.untie_embeddings_and_output_weights
    if run_args.model_arch.lower() in ['bog-moe', 'welm_moe_32b', 'qwen1.5-moe', 'qwen3-moe']:
        lm_model.decoder.final_layernorm.weight.data.copy_(hf_model.model.norm.weight)

        # 当转 rm model 时候，没有 lm_head 层，跳过赋值
        has_lm_head = hasattr(hf_model, 'lm_head')
        if has_lm_head:
            extra_vocab = lm_model.output_layer.weight.data.shape[
                0] - hf_model.lm_head.weight.shape[0]
            embed_dim = hf_model.lm_head.weight.shape[1]
            extra_zeros = torch.zeros((extra_vocab, embed_dim), dtype=hf_model.lm_head.weight.dtype)
            padded_output_layer = torch.cat((hf_model.lm_head.weight, extra_zeros), dim=0)
            lm_model.output_layer.weight.data.copy_(padded_output_layer)
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')

    if not with_save:
        return

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
    print("successfully convert moe hf ckpt to megatron ckpt")

    pass
