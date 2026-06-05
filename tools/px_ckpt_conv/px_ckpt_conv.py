import argparse
import json
import pathlib
import os
import re
import shutil
import sys
import time
import tempfile
from contextlib import nullcontext

from importlib.metadata import version
from pkg_resources import packaging
from accelerate import init_empty_weights
from transformers.modeling_utils import no_init_weights
from transformers import AutoConfig
from tqdm import tqdm
import torch
import transformers

from megatron.core import __version__
from gpatch.core.device_type import is_wxacc1, is_wxacc2
from gpatch.patch_mcore import init_gpatch_for_mcore
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.legacy import fused_kernels
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args, set_global_variables
from megatron.training.initialize import _set_random_seed, _initialize_distributed
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
from megatron.training.utils import unwrap_model

try:
    from lubanml.api.common import get_file_from_luban
except:
    get_file_from_luban = lambda x: x

from mlm_to_hf import convert_mlm_to_hf
from hf_to_mlm import convert_hf_to_mlm
from gpatch.training.arguments import gpatch_extra_args


def get_te_version():
    try:
        import transformer_engine as te

        def get_te_version_str():
            if hasattr(te, '__version__'):
                return str(te.__version__)
            else:
                return version("transformer-engine")

        return packaging.version.Version(get_te_version_str())
    except ImportError:
        return None


def create_mlm_model(
    run_args,
    only_create=False,
    extra_argv=[],
    model_provider_func=None,
    build_tokenizer=False,
    extra_args_provider=None,
    force_mlm_untie_embeddings_and_output_weights=None,  # for RM model
):
    print('creating MLM model...')
    t0 = time.time()
    # TODO: 将下面使用到的 AutoConfig.from_pretrained 统一到这里来
    config = AutoConfig.from_pretrained(
        os.path.dirname(run_args.hf_config_json), trust_remote_code=True
    )
    model_config = config.to_dict()
    if run_args.model_arch in ["internvl"]:
        model_config = model_config['llm_config']
    elif run_args.model_arch in ["gemma3"]:
        model_config = model_config['text_config']

    # 不再对外暴露，这是 HF config
    run_args.tie_word_embeddings = model_config['tie_word_embeddings']

    # Megatron 的 kv head dim，根据 megatron args named，之所以存在是因为 qwen3 的 head dim
    # 比较特殊，不能直接计算出来。然而，让用户传递也比较难用，因此，不再要求用户传递，而是在代码内自己判断
    # 出来。
    # 不再对外暴露，这是 HF config
    if run_args.model_arch in ['qwen3', 'qwen3-moe']:
        kv_channels = model_config['head_dim']
    elif run_args.model_arch in [
        'llama',
        'bog',
        'baichuan-7b',
        'welm_19b',
        'yi-9b',
        'qwen2-72b',
        'qwen2.5-1.5b',
        'bog-moe',
        'welm_moe_32b',  # 以后统一用 xx-yy-zz 的格式，这里一开始被 xt 这么写了，就不动了。
        'qwen1.5-moe',
        'qwen2vl',
        'qwen2p5vl',
        'dsr1-distill-qwen2.5-32b',
        'qwen2.5-math-rm-72b',
        'qwen2.5-math-1.5b',
        'qwen2.5-3b-instruct',
        'qwq-32b',
        'internvl',
    ]:
        kv_channels = model_config['hidden_size'] // model_config['num_attention_heads']
    elif run_args.model_arch in ["gemma3"]:
        kv_channels = model_config['text_config']['head_dim']
    else:
        raise NotImplementedError()
    run_args.kv_channels = kv_channels

    sys.argv = [
        'placeholder.py',
        '--no-load-optim',
        '--no-load-rng',
        '--no-save-optim',
        '--no-save-rng',
        '--no-initialization',
        '--use-cpu-initialization',
        '--micro-batch-size',
        '1',
        '--save-interval',
        '1',
    ]
    if run_args.enable_lora:
        lora_arv = ["--enable-lora", '--lora-r', f"{run_args.lora_r}"]
        sys.argv.extend(lora_arv)
    sys.argv += extra_argv
    if run_args.model_arch.lower() in ['bog-moe', 'welm_moe_32b', 'qwen1.5-moe', 'qwen3-moe']:
        sys.argv += [
            '--untie-embeddings-and-output-weights',
            '--position-embedding-type',
            'rope',
            '--normalization',
            'RMSNorm',
            '--swiglu',
            '--disable-bias-linear',
            '--no-masked-softmax-fusion',
            '--no-bias-gelu-fusion',
            '--no-bias-dropout-fusion',
            '--attention-dropout',
            '0',
            '--hidden-dropout',
            '0',
            '--no-async-tensor-model-parallel-allreduce',
            '--group-query-attention',
        ]
        if run_args.model_arch.lower() in ['welm_moe_32b', 'qwen1.5-moe']:
            sys.argv += [
                '--add-qkv-bias',
            ]
    elif run_args.model_arch.lower() in [
        'llama',
        'bog',
        'baichuan-7b',
        'yi-9b',
        'qwen2-72b',
        'qwen2.5-1.5b',
        'dsr1-distill-qwen2.5-32b',
        'qwen2.5-math-rm-72b',
        'qwen2.5-math-1.5b',
        'qwen2.5-3b-instruct',
        'qwq-32b',
        'qwen3',
    ]:
        sys.argv += [
            '--disable-bias-linear',
            '--position-embedding-type',
            'rope',
            '--normalization',
            'RMSNorm',
            '--no-masked-softmax-fusion',
            '--swiglu',
            '--no-bias-swiglu-fusion',
            '--no-bias-dropout-fusion',
            '--attention-dropout',
            '0',
            '--hidden-dropout',
            '0',
            '--no-async-tensor-model-parallel-allreduce',
            '--group-query-attention',
        ]
        if run_args.model_arch.lower() in [
            'llama',
            'bog',
            'baichuan-7b',
            'yi-9b',
            'qwen2-72b',
            'dsr1-distill-qwen2.5-32b',
            'qwen2.5-math-rm-72b',
            'qwq-32b',
        ]:
            sys.argv += [
                '--untie-embeddings-and-output-weights',
            ]
        if run_args.model_arch.lower() in [
            'qwen2-72b',
            'qwen2.5-1.5b',
            'dsr1-distill-qwen2.5-32b',
            'qwen2.5-math-rm-72b',
            'qwen2.5-math-1.5b',
            'qwen2.5-3b-instruct',
            'qwq-32b',
        ]:
            sys.argv += [
                '--add-qkv-bias',
            ]
    elif run_args.model_arch.lower() in ['welm_19b']:
        sys.argv += [
            '--use-rotary-position-embeddings',
            '--position-embedding-type',
            'rope',
            'LayerNorm',
            '--add-qkv-bias',
            '--no-masked-softmax-fusion',
            '--no-bias-gelu-fusion',
            '--no-bias-dropout-fusion',
            '--attention-dropout',
            '0',
            '--hidden-dropout',
            '0',
            '--no-async-tensor-model-parallel-allreduce',
        ]
    elif run_args.model_arch.lower() in ['qwen2vl']:
        # TODO(@guanyouhe)：没有 px-rope-variant / px-qwen-extra-vocab-size 了
        sys.argv += [
            '--swiglu',
            '--normalization',
            'RMSNorm',
            '--use-rotary-position-embeddings',
            '--position-embedding-type',
            'rope',
            '--disable-bias-linear',
            '--add-qkv-bias',
            '--rotary-percent',
            '1.0',
            '--rotary-base',
            '1000000',
            '--rotary-seq-len-interpolation-factor',
            '1',
            '--padded-vocab-size',
            '151936',
            '--attention-dropout',
            '0',
            '--hidden-dropout',
            '0',
            '--group-query-attention',
            '--micro-batch-size',
            '4',
            '--decoder-seq-length',
            '4096',
            '--seq-length',
            '4096',
        ]
    elif run_args.model_arch.lower() in ['gemma3']:
        sys.argv += [
            "--normalization",
            "RMSNorm",
            "--disable-bias-linear",
            "--position-embedding-type",
            "rope",
            "--rotary-percent",
            "1.0",
            "--rotary-base",
            "1000000",
            "--swiglu",
            "--attention-dropout",
            "0.0",
            "--hidden-dropout",
            "0.0",
            "--group-query-attention",
            "--no-masked-softmax-fusion",
            "--img-h",
            "896",
            "--img-w",
            "896",
            "--patch-dim",
            "14",
            "--decoder-seq-length",
            "4096",
        ]
    elif run_args.model_arch.lower() in ['internvl']:
        sys.argv += [
            "--normalization",
            "RMSNorm",
            "--disable-bias-linear",
            "--position-embedding-type",
            "rope",
            "--rotary-percent",
            "1.0",
            "--rotary-base",
            "1000000",
            "--swiglu",
            "--attention-dropout",
            "0.0",
            "--hidden-dropout",
            "0.0",
            "--group-query-attention",
            "--no-masked-softmax-fusion",
            "--img-h",
            "448",
            "--img-w",
            "448",
            "--patch-dim",
            "14",
            "--decoder-seq-length",
            "4096",
        ]
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')

    if is_wxacc1() or is_wxacc2():
        sys.argv += [
            '--no-gradient-accumulation-fusion',
        ]

    args = parse_args(extra_args_provider=extra_args_provider)
    args.bf16 = run_args.bf16
    args.fp16 = run_args.fp16
    args.clip_grad = 1.0
    args.seed = 1111
    args.model_type = ModelType.encoder_or_decoder
    if run_args.model_arch.lower() in ['gemma3']:
        args.model_type = ModelType.encoder_and_decoder

    if run_args.model_arch.lower() in ['bog-moe']:
        num_layers_key = 'num_hidden_layers'
        ffn_hidden_size_key = 'moe_intermediate_size'
        norm_epsilon_key = 'rms_norm_eps'
        num_query_groups_key = 'num_key_value_heads'
        px_rope_base_key = 'rope_theta'
        num_shared_experts_key = 'n_shared_experts'
        num_experts_key = "n_routed_experts"
        topk_key = "num_experts_per_tok"
        # illegal keys
        shared_expert_gate_key = 'has_shared_expert_gate'
        shared_expert_intermediate_size_key = 'shared_expert_intermediate_size'

    elif run_args.model_arch.lower() in ['welm_moe_32b', 'qwen1.5-moe', 'qwen3-moe']:
        num_layers_key = 'num_hidden_layers'
        ffn_hidden_size_key = 'moe_intermediate_size'
        norm_epsilon_key = 'rms_norm_eps'
        num_query_groups_key = 'num_key_value_heads'
        px_rope_base_key = 'rope_theta'
        num_shared_experts_key = 'num_shared_experts'
        shared_expert_gate_key = 'has_shared_expert_gate'
        shared_expert_intermediate_size_key = 'shared_expert_intermediate_size'
        num_experts_key = "num_experts"
        topk_key = "num_experts_per_tok"
    elif run_args.model_arch.lower() in [
        'llama',
        'bog',
        'baichuan-7b',
        'yi-9b',
        'qwen2-72b',
        'qwen2.5-1.5b',
        'dsr1-distill-qwen2.5-32b',
        'qwen2.5-math-rm-72b',
        'qwen2.5-math-1.5b',
        'qwen2.5-3b-instruct',
        'qwq-32b',
        'qwen3',
    ]:
        num_layers_key = 'num_hidden_layers'
        ffn_hidden_size_key = 'intermediate_size'
        norm_epsilon_key = 'rms_norm_eps'
        num_query_groups_key = 'num_key_value_heads'
        px_rope_base_key = 'rope_theta'
        if run_args.model_arch.lower() == 'bog':
            px_rope_base_key = 'rope_base'
    elif run_args.model_arch.lower() in ['qwen2vl']:
        # NOTE(guanyouhe): 这种方式可能比较直接read_json更好，它会带上默认参数
        config = AutoConfig.from_pretrained(
            os.path.dirname(run_args.hf_config_json), trust_remote_code=True
        )
        model_config = config.to_dict()
        num_layers_key = 'num_hidden_layers'
        ffn_hidden_size_key = 'intermediate_size'
        norm_epsilon_key = 'rms_norm_eps'
        num_query_groups_key = 'num_key_value_heads'
        px_rope_base_key = 'rope_theta'
        head_dim = model_config.get('head_dim', run_args.kv_channels)
        assert head_dim == run_args.kv_channels, (
            f"FATAL: head_dim {head_dim} is not equal to kv_channels {run_args.kv_channels} !!!"
        )
    elif run_args.model_arch.lower() in ['welm_19b']:
        num_layers_key = 'num_layers'
        ffn_hidden_size_key = 'ffn_hidden_size'
        norm_epsilon_key = 'layernorm_epsilon'
        num_query_groups_key = 'num_kv_heads'
        px_rope_base_key = 'rotary_emb_base'
    elif run_args.model_arch.lower() in ['gemma3']:
        # NOTE(guanyouhe): 这种方式可能比较直接read_json更好，它会带上默认参数
        config = AutoConfig.from_pretrained(os.path.dirname(run_args.hf_config_json))
        config_dict = config.to_dict()
        model_config = config_dict['text_config']
        model_config["vision_config"] = config_dict["vision_config"]
        model_config["qk_layernorm"] = True
        num_layers_key = 'num_hidden_layers'
        ffn_hidden_size_key = 'intermediate_size'
        norm_epsilon_key = 'rms_norm_eps'
        num_query_groups_key = 'num_key_value_heads'
        px_rope_base_key = 'rope_theta'
        head_dim = model_config.get('head_dim', run_args.kv_channels)
        assert head_dim == run_args.kv_channels, (
            f"FATAL: head_dim {head_dim} is not equal to kv_channels {run_args.kv_channels} !!!"
        )
    elif run_args.model_arch.lower() in ['internvl']:
        config = AutoConfig.from_pretrained(
            os.path.dirname(run_args.hf_config_json), trust_remote_code=True
        )
        config_dict = config.to_dict()
        model_config = config_dict['llm_config']
        model_config["vision_config"] = config_dict["vision_config"]
        num_layers_key = 'num_hidden_layers'
        ffn_hidden_size_key = 'intermediate_size'
        norm_epsilon_key = 'rms_norm_eps'
        num_query_groups_key = 'num_key_value_heads'
        px_rope_base_key = 'rope_theta'
        head_dim = model_config.get('head_dim', run_args.kv_channels)
        assert head_dim == run_args.kv_channels, (
            f"FATAL: head_dim {head_dim} is not equal to kv_channels {run_args.kv_channels} !!!"
        )
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')

    args.num_layers = model_config[num_layers_key]
    args.hidden_size = model_config["hidden_size"]
    args.ffn_hidden_size = model_config[ffn_hidden_size_key]
    args.norm_epsilon = model_config[norm_epsilon_key]
    args.num_attention_heads = model_config["num_attention_heads"]
    args.num_query_groups = model_config.get(num_query_groups_key, args.num_attention_heads)
    args.max_position_embeddings = model_config["max_position_embeddings"]
    args.seq_length = model_config.get("max_length", args.max_position_embeddings)
    args.init_method_std = model_config["initializer_range"]
    args.px_rope_base = model_config.get(px_rope_base_key, 10000)
    args.kv_channels = run_args.kv_channels

    if force_mlm_untie_embeddings_and_output_weights is not None:
        args.untie_embeddings_and_output_weights = force_mlm_untie_embeddings_and_output_weights
    else:
        args.untie_embeddings_and_output_weights = not model_config['tie_word_embeddings']

    args.qk_layernorm = model_config.get("qk_layernorm", False)
    args.rm_head_arch = "multi_layers" if run_args.rm_multi_layers else "single_layer"

    if run_args.model_arch.lower() in ['bog', 'bog-moe']:
        # bog 系列 的 qk_norm 为 rmsnorm
        args.qk_norm_type = "rmsnorm"
        if run_args.model_arch.lower() in ['bog-moe']:
            topk_method = model_config.get("topk_method", None)
            if topk_method == "noaux_tc":
                balancing_type = "aux_loss_free"
            elif topk_method == "aux_loss":
                balancing_type = "aux_loss"
            else:
                raise NotImplementedError(f"unknown topk_method {topk_method}")
            args.moe_router_load_balancing_type = balancing_type

        args.moe_first_k_dense_replace = model_config.get('first_k_dense_replace', 0)
        args.moe_first_k_dense_layer_ffn_hidden_size = model_config.get('intermediate_size', None)

    elif run_args.model_arch in ['qwen3', 'qwen3-moe']:
        model_config["qk_layernorm"] = True
        args.qk_layernorm = True
        args.add_qkv_bias = model_config.get("attention_bias", False)

    if run_args.escore_bias_inherit:
        assert args.moe_router_load_balancing_type in ["aux_loss_free", "aux_loss"]

    # MoE args
    if run_args.model_arch.lower() in ['bog-moe', 'welm_moe_32b', 'qwen1.5-moe', 'qwen3-moe']:
        aux_loss_key = "aux_loss_alpha" if run_args.model_arch.lower(
        ) == "bog-moe" else "router_aux_loss_coef"
        args.moe_aux_loss_coeff = model_config.get(aux_loss_key, 1e-3)
        args.num_experts = model_config.get(num_experts_key, 1)
        args.moe_router_topk = model_config.get(topk_key, 1)
        args.moe_grouped_gemm = run_args.moe_grouped_gemm
        #TODO: 删除 moe_shared_expert_gate 和 px_has_shared_experts 等 mcore moe 无用的 args
        n_shared_experts = model_config.get(num_shared_experts_key, 0)
        all_shared_experts_intermediate_size = n_shared_experts * model_config[ffn_hidden_size_key]
        assert isinstance(all_shared_experts_intermediate_size, int)

        moe_shared_expert_intermediate_size = model_config.get(
            shared_expert_intermediate_size_key, None
        )
        if moe_shared_expert_intermediate_size is not None and moe_shared_expert_intermediate_size > 0:
            args.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size

        if run_args.model_arch.lower() in ['bog-moe']:
            args.moe_shared_expert_intermediate_size = all_shared_experts_intermediate_size

    if run_args.model_arch in ['qwen3', 'qwen3-moe']:
        # qwen3 moe 没有 shared_expert
        assert not is_wxacc1()

    args.train_data_consuming_progresses = {}  # 记录数据消费进度
    args.load = run_args.megatron_load_dir
    if args.load[-1] == '/':
        args.load = args.load[:-1]
    args.save = run_args.megatron_save_dir
    args.tokenizer_type = run_args.tokenizer_type
    args.tokenizer_model = run_args.tokenizer_path

    args.padded_vocab_size = _vocab_size_with_padding(model_config["vocab_size"], args)
    args.transformer_impl = "transformer_engine" if not is_wxacc1() and not is_wxacc2() else "local"
    if (args.transformer_impl == "transformer_engine"):
        args.use_te = True

    # key point
    args.use_legacy_models = False
    args.ckpt_format = run_args.dist_ckpt_format
    args.world_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
    args = validate_args(args)
    print(f"MLM args {args}")

    set_global_variables(args, build_tokenizer=build_tokenizer)
    mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(args.pipeline_model_parallel_size)
    mpu.set_virtual_pipeline_model_parallel_world_size(args.virtual_pipeline_model_parallel_size)

    if not is_wxacc1() and not is_wxacc2():
        fused_kernels.load(args)

    mcore_version = packaging.version.parse(__version__)
    extra_args = {}
    if mcore_version >= packaging.version.parse('0.13.0'):
        extra_args = {"store": None}
    _initialize_distributed(None, None, **extra_args)
    _set_random_seed(args.seed, args.data_parallel_random_init)

    # Get first pipe stage.
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)

    if run_args.model_arch.lower() in [
        'llama',
        'bog',
        'baichuan-7b',
        'welm_19b',
        'yi-9b',
        'qwen2-72b',
        'qwen2.5-1.5b',
        'bog-moe',
        'welm_moe_32b',
        'qwen1.5-moe',
        'qwen2vl',
        'dsr1-distill-qwen2.5-32b',
        'qwen2.5-math-rm-72b',
        'qwen2.5-math-1.5b',
        'qwen2.5-3b-instruct',
        'qwq-32b',
        'gemma3',
        'qwen3',
        'qwen3-moe',
        'internvl',
    ]:
        if model_provider_func is not None:
            model = model_provider_func(True, True).to(args.params_dtype)
        else:
            from gpatch.training.v3.default_model_provider import default_sft_model_provider
            model = default_sft_model_provider(True, True).to(args.params_dtype)
    else:
        raise ValueError(f'unknown arch {run_args.model_arch}')

    t1 = time.time()
    if not only_create:
        args.load = os.path.abspath(args.load)
        base_name = os.path.basename(args.load)
        assert os.path.isdir(args.load), "args.load should be a directory"
        # create conv temp dir
        convert_tmp_dir = os.path.join(os.path.dirname(args.load), f"convert_temp_dir_{base_name}")
        pathlib.Path(convert_tmp_dir).mkdir(parents=False, exist_ok=True)
        os.symlink(args.load, os.path.join(convert_tmp_dir, base_name))
        lastest_filename = os.path.join(convert_tmp_dir, "latest_checkpointed_iteration.txt")
        try:
            if base_name != "release":
                pattern = r'^iter_\d+$'
                assert bool(
                    re.match(pattern, base_name)
                ), f"the megatron load dir should be release or iter_xxxx: {args.load}"
                iter_name = str(int(re.findall(r'\d+', base_name)[0]))
            else:
                assert base_name == "release", f"the megatron load dir should be release or iter_xxxx: {args.load}"
                iter_name = "release"
            with open(lastest_filename, "w") as f:
                f.write(iter_name)
            args.load = convert_tmp_dir
            iteration, _ = load_checkpoint([model], None, None)
        except Exception as e:
            print(f"Error loading ckpt from {args.load}: {e}")
        finally:
            os.unlink(os.path.join(convert_tmp_dir, base_name))
            os.remove(lastest_filename)
            os.rmdir(convert_tmp_dir)
        print(f"Loaded ckpt from iteration {iteration}")
        # # For ckpt inspection
        if os.environ.get("PX_INSPECET_MODEL", "0") == "1":
            if torch.distributed.get_rank() == 0:
                for k, v in model.named_parameters():
                    print(k, v.shape, v.sum().item(), v)
    t2 = time.time()
    print(f'''MLM model created
t1 - t0 {t1 - t0}
t2 - t1 {t2 - t1}
    ''')
    return model, model_config


def create_hf_model(run_args, only_create=False, model_class=None):
    print('creating HF model...')
    t0 = time.time()
    torch_dtype = torch.float
    if run_args.bf16:
        torch_dtype = torch.bfloat16
    if run_args.fp16:
        torch_dtype = torch.float16
    run_args.hf_load_dir = get_file_from_luban(run_args.hf_load_dir)

    if only_create:
        if not os.path.exists(run_args.hf_save_dir):
            os.makedirs(run_args.hf_save_dir)
        cmd = f"cp {run_args.hf_config_json} {run_args.hf_save_dir}"
        os.system(cmd)
        cmd = f"cp {run_args.hf_py_source_file}/*.py {run_args.hf_save_dir}"
        os.system(cmd)

        hf_config = transformers.AutoConfig.from_pretrained(
            run_args.hf_save_dir, trust_remote_code=True
        )
        if is_wxacc1():
            auto_ctx = nullcontext()
        else:
            auto_ctx = no_init_weights()
        if run_args.no_init_hf_weights:
            auto_ctx = no_init_weights()

        with auto_ctx:
            if model_class is not None:
                hf_model = model_class(hf_config).to(torch_dtype)
            else:
                hf_model = transformers.AutoModelForCausalLM.from_config(
                    hf_config, trust_remote_code=True, torch_dtype=torch_dtype
                )
    else:
        if model_class is None:
            model_class = transformers.AutoModelForCausalLM
        hf_model = model_class.from_pretrained(
            run_args.hf_load_dir, trust_remote_code=True, torch_dtype=torch_dtype, device_map="cpu"
        )
        # # For ckpt inspection
        if os.environ.get("PX_INSPECET_MODEL", "0") == "1":
            if torch.distributed.get_rank() == 0:
                for k, v in hf_model.named_parameters():
                    print("hf model ", k, v.shape, v.sum().item(), v)
    t1 = time.time()
    print(f'''HF model created
t1 - t0 {t1 - t0}
    ''')
    return hf_model


def get_run_args(extra_args_provider=None):
    parser = argparse.ArgumentParser(
        description="Megatron and Hf Checkpoint Utility Arguments",
        allow_abbrev=False,
        conflict_handler='resolve'
    )
    parser.add_argument(
        '--convert_way',
        type=str,
        required=True,
        choices=["mlm_to_hf", "hf_to_mlm"],
        help="The action of convertion"
    )
    parser.add_argument(
        '--megatron_load_dir',
        type=str,
        required=True,
        help='Directory to load megatron model checkpoint from'
    )
    parser.add_argument(
        '--megatron_save_dir',
        type=str,
        required=True,
        help='Directory to save megatron model checkpoint to'
    )
    parser.add_argument(
        '--hf_load_dir', type=str, required=True, help='Directory to load hf model checkpoint from'
    )
    parser.add_argument(
        '--hf_save_dir', type=str, required=True, help='Directory to save hf model checkpoint to'
    )
    parser.add_argument(
        '--hf_py_source_file', type=str, default=None, help='Directory of hf py source file'
    )
    parser.add_argument(
        '--hf_config_json', type=str, required=True, help='The config json of hf model'
    )
    parser.add_argument('--tokenizer_path', type=str, required=True, help="The tokenizer path")
    parser.add_argument(
        '--tokenizer_type',
        type=str,
        default='HuggingFaceTokenizer',
        choices=[
            # 其他暂时都没啥用，就这个了
            'HuggingFaceTokenizer',
            'MultimodalTokenizer',
        ],
        help='What type of tokenizer to use'
    )
    parser.add_argument('--fp16', action='store_true', help='Run model in fp16 mode.')
    parser.add_argument('--bf16', action='store_true', help='Run model in bfloat16 mode.')
    parser.add_argument(
        '--model_arch',
        type=str,
        required=True,
        choices=[
            'llama',
            'bog',
            'baichuan-7b',
            'welm_19b',
            'yi-9b',
            'qwen2-72b',
            'qwen2.5-1.5b',
            'bog-moe',
            'welm_moe_32b',  # 以后统一用 xx-yy-zz 的格式，这里一开始被 xt 这么写了，就不动了。
            'qwen1.5-moe',
            'qwen2vl',
            'qwen2p5vl',
            'gemma3',
            'dsr1-distill-qwen2.5-32b',
            'qwen2.5-math-rm-72b',
            'qwen2.5-math-1.5b',
            'qwen2.5-3b-instruct',
            'qwq-32b',
            'qwen3',
            'qwen3-moe',
            'internvl',
        ],
        help='Type of the model'
    )
    parser.add_argument(
        '--dist_ckpt_format',
        type=str,
        required=True,
        default='torch_dist',
        choices=['zarr', 'torch_dist'],
        help='Distributed checkpoint format to use.'
    )
    parser.add_argument(
        '--moe_grouped_gemm',
        action='store_true',
        help=
        'When there are multiple experts per rank, compress multiple local (potentially small) gemms in a single kernel launch to improve the utilization and performance by leveraging the Grouped GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).'
    )
    parser.add_argument(
        '--use_te_grouped_gemm', action='store_true', help='whether to apply use_te_grouped_gemm'
    )
    parser.add_argument(
        '--no_init_hf_weights', action='store_true', help='whether to use no_init_weights of hf'
    )
    # LoRA
    parser.add_argument('--enable_lora', action='store_true', help="enable LoRA")
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    # TODO ?
    parser.add_argument('--escore_bias_inherit', action='store_true', help="ecore bias inherit")
    parser.add_argument('--rm_multi_layers', action='store_true', help="rm_multi_layers or not")
    parser.add_argument(
        '--hf_auto_model_class_name', type=str, default=None, help="auto_model_cls_name"
    )
    parser.add_argument(
        '--mlm_model_provider_module_name',
        type=str,
        default=None,
        help='module that defines model_provider'
    )

    # TODO drop it ?
    # parser.add_argument('--reward_model_extra_args',
    #                     type=str,
    #                     default=None,
    #                     help='reward model extra arguments')

    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    args = parser.parse_args()

    if args.convert_way == "mlm_to_hf":
        assert args.hf_py_source_file is not None
        assert args.tokenizer_path is not None
    assert not args.fp16 or not args.bf16, f"Do not set --bf16 and --fp16 at the same time"
    return args


if __name__ == "__main__":
    init_gpatch_for_mcore()
    run_args = get_run_args()
    if run_args.use_te_grouped_gemm:
        _te_version = get_te_version()
        assert _te_version is not None and _te_version >= packaging.version.Version("1.9.0.dev0")
    if is_wxacc1():
        assert not run_args.use_te_grouped_gemm

    hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
        run_args.tokenizer_path, trust_remote_code=True
    )

    if run_args.convert_way == "hf_to_mlm":
        lm_model, model_config = create_mlm_model(
            run_args,
            only_create=True,
            extra_args_provider=gpatch_extra_args,
        )
        hf_model = create_hf_model(run_args, only_create=False)
        convert_hf_to_mlm(
            run_args=run_args, model_config=model_config, lm_model=lm_model, hf_model=hf_model
        )
    elif run_args.convert_way == "mlm_to_hf":
        lm_model, model_config = create_mlm_model(
            run_args,
            only_create=False,
            extra_args_provider=gpatch_extra_args,
        )
        hf_model = create_hf_model(run_args, only_create=True)
        convert_mlm_to_hf(
            run_args=run_args,
            model_config=model_config,
            lm_model=lm_model,
            hf_model=hf_model,
            hf_tokenizer=hf_tokenizer
        )
    else:
        raise NotImplementedError(f"convert way {run_args.convert_way} is not supported")
