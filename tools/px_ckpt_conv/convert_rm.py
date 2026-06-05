from pkg_resources import packaging
import importlib
import os
import time

from transformers import AutoModelForCausalLM, AutoConfig, AutoModel
from transformers.modeling_utils import no_init_weights
import torch
import torch.distributed

from gpatch.core.device_type import is_wxacc1
from gpatch.patch_mcore import init_gpatch_for_mcore
from gpatch.training.arguments import gpatch_extra_args
from tools.px_ckpt_conv import px_ckpt_conv, mlm_to_hf, hf_to_mlm


def create_rm_hf_model(run_args, only_create=True):
    torch_dtype = torch.float
    if run_args.fp16:
        torch_dtype = torch.float16
    if run_args.bf16:
        torch_dtype = torch.bfloat16

    hf_auto_model_class_name = AutoModelForCausalLM
    if run_args.hf_auto_model_class_name is not None:
        hf_auto_model_class_name = eval(run_args.hf_auto_model_class_name)

    if only_create:
        with no_init_weights():
            hf_config = AutoConfig.from_pretrained(
                run_args.hf_py_source_file, trust_remote_code=True
            )
            model = hf_auto_model_class_name.from_config(
                hf_config,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            )
    else:
        hf_config = None
        model = hf_auto_model_class_name.from_pretrained(
            run_args.hf_load_dir, trust_remote_code=True, torch_dtype=torch_dtype, device_map="cpu"
        )
    return model, hf_config


def create_rm_mlm_model(run_args, only_create=True):
    # TODO(guanyouhe): 这里引用的是其它的model_provider
    assert run_args.mlm_model_provider_module_name is not None, "mlm_model_provider_module_name must be set when using reward model"

    mod = importlib.import_module(run_args.mlm_model_provider_module_name)
    model_provider = mod.model_provider

    # TODO why ?
    # get_tasks_args = None
    # if run_args.reward_model_extra_args is not None:
    #     extra_args = importlib.import_module(run_args.reward_model_extra_args)
    #     get_tasks_args = extra_args.get_tasks_args

    lm_model, model_config = px_ckpt_conv.create_mlm_model(
        run_args,
        only_create=only_create,
        extra_argv=[
            '--rm-use-avg-pool',
        ],
        model_provider_func=model_provider,
        build_tokenizer=True,
        extra_args_provider=gpatch_extra_args,
        force_mlm_untie_embeddings_and_output_weights=False,  # 省掉 MLM 一个 output layer
    )
    return lm_model, model_config


def convert_rm_mlm2hf(run_args, hf_config, mlm_config, hf_model, mlm_model):
    # llm
    mlm_to_hf.convert_mlm_to_hf(
        run_args=run_args,
        model_config=mlm_config,
        lm_model=mlm_model,
        hf_model=hf_model,
        hf_tokenizer=None,
        save_ckpt=False
    )
    # rm_head
    hf_model.rm_head.weight.copy_(mlm_model.rm_head.weight)

    # inspect model
    if os.environ.get("PX_INSPECET_MODEL", "0") == "1":
        for pname, param in mlm_model.named_parameters():
            print(f"mlm ckpt {pname=} {param.sum()} {param.shape} {param}")

        for pname, param in hf_model.named_parameters():
            print(f"hf ckpt {pname=} {param.sum()} {param.shape} {param}")

    # save huggingface format model
    t1 = time.time()
    print('HF model saving pretrained...')
    hf_config.save_pretrained(run_args.hf_save_dir)
    hf_model.save_pretrained(run_args.hf_save_dir)

    src_path = os.path.join(run_args.hf_py_source_file, '*.py')
    os.system(f'cp {src_path} {run_args.hf_save_dir}')

    t2 = time.time()
    print(f'converted MLM ckpt to HF ckpt successfully save:{t2 - t1}s')


def convert_rm_hf2mlm(run_args, hf_config, mlm_config, hf_model, mlm_model):
    assert hf_config is None
    # llm model
    hf_to_mlm.convert_hf_to_mlm(
        run_args=run_args,
        model_config=mlm_config,
        lm_model=mlm_model,
        hf_model=hf_model,
        with_save=False
    )

    # rm head
    if not run_args.rm_multi_layers:
        mlm_model.rm_head.weight.data.copy_(hf_model.rm_head.weight)
    else:
        hf_rm_head = hf_model.score
        mlm_rm_head = mlm_model.rm_head.score
        assert len(hf_rm_head) == len(mlm_rm_head)
        for i in range(len(mlm_rm_head)):
            if hasattr(hf_rm_head[i], 'weight'):
                mlm_rm_head[i].weight.data.copy_(hf_rm_head[i].weight)
            if hasattr(hf_rm_head[i], 'bias'):
                mlm_rm_head[i].bias.data.copy_(hf_rm_head[i].bias)

    # inspect model
    if os.environ.get("PX_INSPECET_MODEL", "0") == "1":
        for pname, param in mlm_model.named_parameters():
            print(f"mlm ckpt {pname=} {param.sum()} {param.shape} {param}")

        for pname, param in hf_model.named_parameters():
            print(f"hf ckpt {pname=} {param.sum()} {param.shape} {param}")

    # save mlm format ckpt
    t1 = time.time()
    print('MLM model saving ...')
    px_ckpt_conv.save_checkpoint(1, [mlm_model], None, None, num_floating_point_operations_so_far=0)

    # rename to release
    old_name = os.path.join(run_args.megatron_save_dir, "iter_0000001")
    new_name = os.path.join(run_args.megatron_save_dir, "release")
    latesest_file = os.path.join(run_args.megatron_save_dir, "latest_checkpointed_iteration.txt")
    os.rename(old_name, new_name)
    with open(latesest_file, 'w') as f:
        f.write('release')
    t2 = time.time()
    print(f"successfully convert moe hf ckpt to megatron ckpt. time {t2 - t1}")


if __name__ == "__main__":
    init_gpatch_for_mcore()
    run_args = px_ckpt_conv.get_run_args()
    torch.set_grad_enabled(False)
    if run_args.use_te_grouped_gemm:
        _te_version = px_ckpt_conv.get_te_version()
        assert _te_version is not None and _te_version >= packaging.version.Version("1.9.0.dev0")
    if is_wxacc1():
        assert not run_args.use_te_grouped_gemm

    if run_args.convert_way == "hf_to_mlm":
        hf_model, hf_config = create_rm_hf_model(run_args, only_create=False)
        lm_model, model_config = create_rm_mlm_model(run_args, only_create=True)
        convert_rm_hf2mlm(run_args, hf_config, model_config, hf_model=hf_model, mlm_model=lm_model)
    elif run_args.convert_way == "mlm_to_hf":
        lm_model, model_config = create_rm_mlm_model(run_args, only_create=False)
        hf_model, hf_config = create_rm_hf_model(run_args, only_create=True)
        convert_rm_mlm2hf(run_args, hf_config, model_config, hf_model=hf_model, mlm_model=lm_model)
    else:
        raise NotImplementedError(f"convert way {run_args.convert_way} is not supported")
