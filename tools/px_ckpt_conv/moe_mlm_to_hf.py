import os
import time
# import multi_device.platform

from tqdm import tqdm
import torch

from gpatch.core.device_type import is_wxacc1


def set_moe_lm2hf_attn_state(run_args, model_config, lm_layer, hf_layer):
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
    assert num_query_head % num_kv_head == 0
    total_dim = 2 * dim + (dim * num_query_head // num_kv_head)

    linear_qkv = lm_attn.linear_qkv.weight.reshape((num_kv_head, total_dim, -1))

    wq = linear_qkv.narrow(1, 0,
                           dim * num_query_head // num_kv_head).reshape((dim * num_query_head, -1))
    wk = linear_qkv.narrow(1, dim * num_query_head // num_kv_head,
                           dim).reshape((dim * num_kv_head, -1))
    wv = linear_qkv.narrow(1, dim * num_query_head // num_kv_head + dim,
                           dim).reshape((dim * num_kv_head, -1))

    hf_attn.q_proj.weight.data.copy_(wq)
    hf_attn.k_proj.weight.data.copy_(wk)
    hf_attn.v_proj.weight.data.copy_(wv)
    hf_attn.o_proj.weight.data.copy_(lm_attn.linear_proj.weight)

    if run_args.model_arch.lower() in ['welm_moe_32b', 'qwen1.5-moe']:
        linear_qkv_bias = lm_attn.linear_qkv.bias.reshape((num_kv_head, total_dim))
        wq_bias = linear_qkv_bias.narrow(1, 0, dim * num_query_head // num_kv_head).reshape(
            (dim * num_query_head)
        )
        wk_bias = linear_qkv_bias.narrow(1, dim * num_query_head // num_kv_head,
                                         dim).reshape((dim * num_kv_head))
        wv_bias = linear_qkv_bias.narrow(1, dim * num_query_head // num_kv_head + dim,
                                         dim).reshape((dim * num_kv_head))
        hf_attn.q_proj.bias.data.copy_(wq_bias)
        hf_attn.k_proj.bias.data.copy_(wk_bias)
        hf_attn.v_proj.bias.data.copy_(wv_bias)


def set_moe_lm2hf_mlp_state(run_args, model_config, lm_layer, hf_layer):
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
    if run_args.model_arch.lower() in ['bog-moe', 'welm_moe_32b', 'qwen1.5-moe', 'qwen3-moe']:
        hf_mlp = hf_layer.mlp
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')

    # gate
    hf_mlp.gate.weight.data.copy_(lm_mlp.router.weight)
    if run_args.escore_bias_inherit:
        hf_mlp.gate.e_score_correction_bias.data.copy_(lm_mlp.router.e_score_correction_bias)

    # experts
    if run_args.use_te_grouped_gemm:
        for expert_i in range(num_experts):
            linear_fc1_size = getattr(lm_mlp.experts.linear_fc1, f"weight{expert_i}").data.shape[0]
            assert linear_fc1_size % 2 == 0
            split_size = linear_fc1_size // 2
            linear_fc1_weight = torch.split(
                getattr(lm_mlp.experts.linear_fc1, f"weight{expert_i}"), split_size
            )
            hf_mlp.experts[expert_i].gate_proj.weight.data.copy_(linear_fc1_weight[0])
            hf_mlp.experts[expert_i].up_proj.weight.data.copy_(linear_fc1_weight[1])
            hf_mlp.experts[expert_i].down_proj.weight.data.copy_(
                getattr(lm_mlp.experts.linear_fc2, f"weight{expert_i}")
            )
    elif not is_wxacc1():
        raise NotImplementedError("not implemented")
    else:
        for expert_i in range(num_experts):
            linear_fc1_size = lm_mlp.experts.local_experts[expert_i].linear_fc1.weight.data.shape[0]
            assert linear_fc1_size % 2 == 0
            split_size = linear_fc1_size // 2
            linear_fc1_weight = torch.split(
                lm_mlp.experts.local_experts[expert_i].linear_fc1.weight, split_size
            )
            hf_mlp.experts[expert_i].gate_proj.weight.data.copy_(linear_fc1_weight[0])
            hf_mlp.experts[expert_i].up_proj.weight.data.copy_(linear_fc1_weight[1])
            hf_mlp.experts[expert_i].down_proj.weight.data.copy_(
                lm_mlp.experts.local_experts[expert_i].linear_fc2.weight
            )

    # shared experts
    if px_has_shared_experts:
        assert run_args.model_arch.lower() not in ['qwen3-moe']
        shared_expert_name = "shared_experts" if run_args.model_arch.lower() in [
            'bog-moe'
        ] else "shared_expert"
        shared_expert_obj = getattr(hf_mlp, f"{shared_expert_name}")

        shared_expert_fc1_size = lm_mlp.shared_experts.linear_fc1.weight.data.shape[0]
        assert shared_expert_fc1_size % 2 == 0
        split_size = shared_expert_fc1_size // 2
        shared_expert_linear_fc1_weight = torch.split(
            lm_mlp.shared_experts.linear_fc1.weight, split_size
        )
        shared_expert_obj.gate_proj.weight.data.copy_(shared_expert_linear_fc1_weight[0])
        shared_expert_obj.up_proj.weight.data.copy_(shared_expert_linear_fc1_weight[1])
        shared_expert_obj.down_proj.weight.data.copy_(lm_mlp.shared_experts.linear_fc2.weight)

        if px_has_shared_expert_gate:
            assert run_args.model_arch.lower() not in ['bog-moe']
            if is_wxacc1():
                hf_mlp.shared_expert_gate.weight.data.copy_(lm_mlp.shared_experts.gate_weight)
            else:
                hf_mlp.shared_expert_gate.weight.data.copy_(lm_mlp.shared_expert_gate.weight)


def set_moe_lm2hf_dense_mlp_state(run_args, model_config, lm_layer, hf_layer):
    lm_mlp = lm_layer.mlp
    hf_mlp = hf_layer.mlp
    if run_args.model_arch.lower() in ['bog-moe']:
        assert lm_mlp.linear_fc1.weight.data.shape[0] % 2 == 0
        split_size = lm_mlp.linear_fc1.weight.data.shape[0] // 2

        linear_fc1_weight = torch.split(lm_mlp.linear_fc1.weight, split_size)
        hf_mlp.gate_proj.weight.data.copy_(linear_fc1_weight[0])
        hf_mlp.up_proj.weight.data.copy_(linear_fc1_weight[1])
        hf_mlp.down_proj.weight.data.copy_(lm_mlp.linear_fc2.weight)
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')


def set_lm2hf_layernorm(run_args, model_config, lm_layer, hf_layer, with_qk_norm=False):
    if is_wxacc1():
        hf_layer.input_layernorm.weight.data.copy_(lm_layer.input_layernorm.weight)
        hf_layer.post_attention_layernorm.weight.data.copy_(lm_layer.pre_mlp_layernorm.weight)
    else:
        hf_layer.input_layernorm.weight.data.copy_(
            lm_layer.self_attention.linear_qkv.layer_norm_weight
        )
        hf_layer.post_attention_layernorm.weight.data.copy_(lm_layer.pre_mlp_layernorm.weight)
    # qknorm
    if with_qk_norm:
        if run_args.model_arch.lower() in ['bog-moe']:
            # bog moe 为 rmsnorm
            hf_layer.self_attn.q_prenorm.weight.data.copy_(
                lm_layer.self_attention.q_layernorm.weight
            )
            hf_layer.self_attn.k_prenorm.weight.data.copy_(
                lm_layer.self_attention.k_layernorm.weight
            )
        elif run_args.model_arch.lower() in ['qwen3-moe']:
            hf_layer.self_attn.q_norm.weight.data.copy_(lm_layer.self_attention.q_layernorm.weight)
            hf_layer.self_attn.k_norm.weight.data.copy_(lm_layer.self_attention.k_layernorm.weight)
        else:
            raise NotImplementedError(f"arch {run_args.model_arch}")


def convert_moe_mlm_to_hf(run_args, model_config, lm_model, hf_model, save_ckpt=True):
    t0 = time.time()
    # embedding
    if run_args.model_arch.lower() in ['bog-moe', 'welm_moe_32b', 'qwen1.5-moe', 'qwen3-moe']:
        hf_vocab_size = hf_model.model.embed_tokens.weight.shape[0]
        hf_model.model.embed_tokens.weight.data.copy_(
            lm_model.embedding.word_embeddings.weight[:hf_vocab_size]
        )
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')

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

    # first_k_dense_layers convert
    for layer_idx in tqdm(
        range(moe_first_k_dense_replace), "copy decoder first_k_dense_layer states"
    ):
        lm_layer = lm_model.decoder.first_k_dense_layers[layer_idx]
        if run_args.model_arch.lower() in ['bog-moe']:
            hf_layer = hf_model.model.layers[layer_idx]
        else:
            raise ValueError(f'unknown arch {run_args.model_arch}')
        set_moe_lm2hf_attn_state(run_args, model_config, lm_layer, hf_layer)
        set_moe_lm2hf_dense_mlp_state(run_args, model_config, lm_layer, hf_layer)
        set_lm2hf_layernorm(run_args, model_config, lm_layer, hf_layer, with_qk_norm=with_qk_norm)

    for layer_idx in tqdm(range(num_layers), "copy decoder layer states"):
        lm_layer = lm_model.decoder.layers[layer_idx]
        if run_args.model_arch.lower() in ['bog-moe', 'welm_moe_32b', 'qwen1.5-moe', 'qwen3-moe']:
            hf_layer = hf_model.model.layers[layer_idx + moe_first_k_dense_replace]
        else:
            raise ValueError(f'unknown arch {run_args.model_arch}')

        set_moe_lm2hf_attn_state(run_args, model_config, lm_layer, hf_layer)
        set_moe_lm2hf_mlp_state(run_args, model_config, lm_layer, hf_layer)
        set_lm2hf_layernorm(run_args, model_config, lm_layer, hf_layer, with_qk_norm=with_qk_norm)

    # output layer
    if run_args.model_arch.lower() in ['bog-moe', 'welm_moe_32b', 'qwen1.5-moe', 'qwen3-moe']:
        hf_model.model.norm.weight.data.copy_(lm_model.decoder.final_layernorm.weight)

        # 当转 rm model 时候，没有 lm_head 层，跳过赋值
        has_lm_head = hasattr(hf_model, 'lm_head')
        if has_lm_head:
            hf_vocab_size = hf_model.lm_head.weight.shape[0]
            hf_model.lm_head.weight.data.copy_(lm_model.output_layer.weight)
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
