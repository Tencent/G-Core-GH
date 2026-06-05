import sys
import os
import time
from datetime import timedelta
import pathlib
import re
import shutil
import inspect

import json
import torch

from transformers import AutoConfig
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training import initialize_megatron, get_args
from megatron.training.global_vars import set_global_variables
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint

from mbridge import AutoBridge
from mbridge.utils.post_creation_callbacks import freeze_moe_router, make_value_model

from tools.convert_fp8.convert_json import convert_config_json_file, convert_index_json_file
from gpatch.core.parallel_state import cpu_barrier
from gpatch.training.utils import print_with_rank_and_datetime
from mpatch.training.checkpointing import save_args_json, copy_extra_file


def init_distributed(tp=2, pp=1, cp=1, vpp=1, ep=1, etp=None):
    """Initialize distributed environment"""
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=300))
    print(
        f"[Init] global_rank {torch.distributed.get_rank()} local_rank {torch.distributed.get_node_local_rank()}"
    )
    # print(f"{os.environ}=")
    torch.cuda.set_device(torch.distributed.get_node_local_rank())
    if pp <= 1:
        vpp = None
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
    )
    model_parallel_cuda_manual_seed(0)


def add_ckpt_args(parser):
    parser.add_argument(
        "--convert_way",
        type=str,
        required=True,
        choices=["mlm_to_hf", "hf_to_mlm"],
        help="The action of convertion"
    )
    parser.add_argument(
        "--hf_dir",
        type=str,
        required=True,
        help=
        "Path to the Hugging Face model directory for config.json and model.safetensors.index.json",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        required=True,
        help="Path to the load model directory",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        required=True,
        help="Path to the save model directory",
    )
    parser.add_argument(
        "--num_layers_in_first_pipeline_stage",
        type=int,
        default=None,
        help="Number of layers in the first pipeline stage",
    )
    parser.add_argument(
        "--num_layers_in_last_pipeline_stage",
        type=int,
        default=None,
        help="Number of layers in the last pipeline stage",
    )
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=4, help="Pipeline parallel size")
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
    parser.add_argument("--vpp", type=int, default=None, help="Virtual pipeline parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel size")
    parser.add_argument("--etp", type=int, default=None, help="Expert tensor parallel size")

    parser.add_argument(
        "--override_args_from_ckpt", action='store_true', help="override the args from checkpoint"
    )
    parser.add_argument(
        "--mbridge_distributed_filesystem", action='store_true', help="加速保存 ckpt，但需要使用 dfs"
    )
    parser.add_argument(
        "--auto_compute_first_last_pp_layers",
        action='store_true',
        help="自动计算--num_layers_in_first_pipeline_stage --num_layers_in_last_pipeline_stage",
    )
    parser.add_argument(
        "--no_check_export",
        action='store_true',
        help="no chect export for save time",
    )
    parser.add_argument('--remove_fp8', action='store_true', help='Remove FP8 config to bf16')
    parser.add_argument('--remove_mtp', action='store_true', help='Remove MTP from json')
    return parser


def convert_hf_to_mlm(args, bridge, model):
    t = time.time()
    bridge.load_weights(model, args.load_model_path, memory_efficient=True)
    if torch.distributed.get_rank() == 0:
        print(
            f"[rank {torch.distributed.get_rank()}] HF Model loaded, elapsed {time.time() - t:.2f} seconds, proceeding with post-processing ..."
        )

    t = time.time()
    #########################################################
    ## if you want to save distributed_checkpoint, you need to save it here
    ## note: it is not verified
    #########################################################
    save_checkpoint(
        iteration=1,
        model=model,
        optimizer=None,
        opt_param_scheduler=None,
        num_floating_point_operations_so_far=0
    )
    if torch.distributed.get_rank() == 0:
        print(
            f"[rank {torch.distributed.get_rank()}] Save checkpoint succ, elapsed {time.time() - t:.2f} seconds, proceeding with post-processing ..."
        )

    # verify if the weights are loaded correctly
    t = time.time()
    if not args.no_check_export:
        for k, v in bridge.export_weights(model):
            if torch.distributed.get_rank() != 0:
                continue
            # .to(v.dtype) qwen2.5vl-32b 的vit部分是 float32 的
            gt = bridge.safetensor_io.load_one_hf_weight(k).to(v.device).to(v.dtype)
            if k != "lm_head.weight":
                assert v.shape == gt.shape, f"mismatch of {k} {v.shape=} {gt.shape=}"
                assert v.sum().item() == gt.sum().item(), f"mismatch of {k} {v=} {gt=}"
            else:
                if v.shape[0] == 1:
                    print(f"this is a value model, {k} {v.shape=} {gt.shape=}")
            if torch.distributed.get_rank() == 0:
                print(k, "export ok")

    if torch.distributed.get_rank() == 0:
        # rename
        old_name = os.path.join(args.save, "iter_0000001")
        new_name = os.path.join(args.save, "release")
        latesest_file = os.path.join(args.save, "latest_checkpointed_iteration.txt")
        os.rename(old_name, new_name)
        with open(latesest_file, 'w') as f:
            f.write('release')
        print(
            f"[rank {torch.distributed.get_rank()}] Verify succ, elapsed {time.time() - t:.2f} seconds, proceeding with post-processing ..."
        )


def convert_mlm_to_hf(args, bridge, model: list):
    t = time.time()
    args.load = os.path.abspath(args.load)
    base_name = os.path.basename(args.load)
    assert os.path.isdir(args.load), "args.load should be a directory"
    # create conv temp dir and link the load folder
    convert_tmp_dir = os.path.join(os.path.dirname(args.load), f"convert_temp_dir_{base_name}")
    if torch.distributed.get_rank() == 0:
        if os.path.exists(convert_tmp_dir):
            print(f"Remove existing convert_tmp_dir {convert_tmp_dir}")
            shutil.rmtree(convert_tmp_dir)
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
        except:
            os.unlink(os.path.join(convert_tmp_dir, base_name))
            os.remove(lastest_filename)
            os.rmdir(convert_tmp_dir)
    try:
        args.load = convert_tmp_dir
        torch.distributed.barrier()
        iteration, _ = load_checkpoint(model, None, None)
        print(
            f"[rank {torch.distributed.get_rank()}] MLM Model loaded iter {iteration} succ, elapsed {time.time() - t:.2f} seconds, proceeding with post-processing ..."
        )
    finally:
        if torch.distributed.get_rank() == 0:
            os.unlink(os.path.join(convert_tmp_dir, base_name))
            os.remove(lastest_filename)
            os.rmdir(convert_tmp_dir)
    t = time.time()
    bridge.safetensor_io = bridge._get_safetensor_io(args.hf_dir)

    # # export weights
    # for k, v in bridge.export_weights(model):
    #     if torch.distributed.get_rank() != 0:
    #         continue
    #     print(k, "export ok")
    # if torch.distributed.get_rank() == 0:
    #     print(f"[rank {torch.distributed.get_rank()}] Export weights succ, elapsed {time.time() - t:.2f} seconds.")

    t = time.time()
    save_func_sig = inspect.signature(bridge.save_weights)
    save_weights_kwargs = {}
    if "distributed_filesystem" in save_func_sig.parameters:
        save_weights_kwargs["distributed_filesystem"] = args.mbridge_distributed_filesystem
    bridge.save_weights(model, args.save_model_path, memory_efficient=True, **save_weights_kwargs)
    save_args_json(args, args.save_model_path)
    if torch.distributed.get_rank() == 0:
        print(
            f"[rank {torch.distributed.get_rank()}] Save weights succ, elapsed {time.time() - t:.2f} seconds."
        )


def main():

    # Megatron parallelism parameters
    # 1. Initialize distributed environment
    args = parse_args(extra_args_provider=add_ckpt_args)

    if args.override_args_from_ckpt:
        load_args_file = os.path.join(args.load_model_path, "common.pt")
        assert os.path.isfile(load_args_file), f"{load_args_file=}"
        load_args = torch.load(load_args_file, weights_only=False)['args']
        for key, value in vars(load_args).items():
            if key not in [
                'convert_way',
                'load_model_path',
                'save_model_path',
                'decoder_first_pipeline_num_layers',
                'decoder_last_pipeline_num_layers',
                'num_layers_in_first_pipeline_stage',
                'num_layers_in_last_pipeline_stage',
                'mbridge_distributed_filesystem',
                'tp',
                'pp',
                'cp',
                'vpp',
                'ep',
                'etp',
            ] and "_parallel_size" not in key:
                setattr(args, key, value)

    if args.auto_compute_first_last_pp_layers:
        assert args.num_layers_in_first_pipeline_stage is None
        assert args.num_layers_in_last_pipeline_stage is None

    args.finetune = True
    # mock for pass check
    args.data_parallel_size = 1
    args.micro_batch_size = 1
    args.global_batch_size = 1
    args.use_dist_ckpt = True
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0

    # override the tp/pp/cp/vpp/ep/etp
    args.tensor_model_parallel_size = args.tp
    args.pipeline_model_parallel_size = args.pp
    args.context_parallel_size = args.cp
    args.virtual_pipeline_model_parallel_size = args.vpp
    args.expert_model_parallel_size = args.ep
    args.expert_tensor_parallel_size = args.etp

    # for mcore dist ckpt load / save
    args.load = args.load_model_path
    args.save = args.save_model_path

    set_global_variables(args=args, build_tokenizer=False)
    init_distributed(tp=args.tp, pp=args.pp, cp=args.cp, vpp=args.vpp, ep=args.ep, etp=args.etp)

    # 2. build bridge and mcore model
    cpu_barrier()
    if torch.distributed.get_rank() == 0:
        print("Building distributed group succ.")
    if args.convert_way == "mlm_to_hf":
        if torch.distributed.get_rank() == 0:
            assert os.path.exists(args.hf_dir), f"{args.hf_dir} does not exist"
            if not os.path.exists(args.save_model_path):
                print(f"Create directory {args.save_model_path}.")
                os.makedirs(args.save_model_path, exist_ok=True)
            copy_extra_file(args.hf_dir, args.save_model_path)

            if args.remove_fp8 or args.remove_mtp:
                convert_config_json_file(
                    args.hf_dir, args.save_model_path, args.remove_fp8, args.remove_mtp
                )
                convert_index_json_file(
                    args.hf_dir, args.save_model_path, args.remove_fp8, args.remove_mtp
                )
        cpu_barrier()
    # args.hf_dir = args.save_model_path
    config = AutoConfig.from_pretrained(args.hf_dir, trust_remote_code=True)
    bridge = AutoBridge.from_config(config)

    # set extra args
    extra_kwargs = {}
    if args.auto_compute_first_last_pp_layers and args.pp > 1:
        num_layers = bridge.config.num_layers
        first_last_layer = num_layers - (num_layers + args.pp - 1) // args.pp * (args.pp - 2)
        assert first_last_layer > 1
        args.num_layers_in_first_pipeline_stage = first_last_layer // 2
        args.num_layers_in_last_pipeline_stage = (first_last_layer + 1) // 2

    if args.num_layers_in_first_pipeline_stage is not None:
        extra_kwargs["num_layers_in_first_pipeline_stage"] = args.num_layers_in_first_pipeline_stage
    if args.num_layers_in_last_pipeline_stage is not None:
        extra_kwargs["num_layers_in_last_pipeline_stage"] = args.num_layers_in_last_pipeline_stage
    bridge.set_extra_args(**extra_kwargs)
    # bridge.config.mtp_num_layers = 0
    model = bridge.get_model(post_model_creation_callbacks=[], wrap_with_ddp=False)

    # maintain router bias dtype
    for m in model:
        from mbridge.core.util import unwrap_model
        m = unwrap_model(m)
        if hasattr(m, "decoder"):
            for l in m.decoder.layers:
                if (
                    hasattr(l, "mlp") and hasattr(l.mlp, "router") and
                    hasattr(l.mlp.router, "_maintain_float32_expert_bias")
                ):
                    # print(f"maintain router bias dtype for {l.mlp.router}")
                    l.mlp.router._maintain_float32_expert_bias()
        if hasattr(m, "mtp"):
            for l in m.mtp.layers:
                l = l.transformer_layer
                if (
                    hasattr(l, "mlp") and hasattr(l.mlp, "router") and
                    hasattr(l.mlp.router, "_maintain_float32_expert_bias")
                ):
                    print(f"maintain mtp router bias dtype for {l.mlp.router}")
                    l.mlp.router._maintain_float32_expert_bias()

    # 3. do convert
    if args.convert_way == "hf_to_mlm":
        convert_hf_to_mlm(args, bridge, model)
    elif args.convert_way == "mlm_to_hf":
        convert_mlm_to_hf(args, bridge, model)

    if torch.distributed.get_rank() == 0:
        print(
            f"Args convert_way {args.convert_way} Load {args.load_model_path} Save {args.save_model_path} succ!"
        )

    # Synchronize all processes to ensure workflow completes
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return True


if __name__ == "__main__":
    main()
