import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    _init_optim_state,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor

from gpatch_v4.core.parallel_state import cpu_barrier

try:
    from gpatch_v4.models.oteam4_4.transformer_flux import (
        FluxTransformer2DModel as Oteam4_4Transformer2DModel,
    )
except ImportError:
    Oteam4_4Transformer2DModel = None
from gpatch_v4.training_backend.fsdp2_backend.ema_model_t2i import T2iEmaModel
from gpatch_v4.utils import clear_memory, log, logging_rank0, sync_cuda_and_get_time

# TODO 具体模型绑定，显然不合理
MODEL_CHECKPOINT = "diffusion_pytorch_model.bin"
OPTIM_CHECKPOINT = "optim_state_dict.pt"
LR_SCHEDULER_CHECKPOINT = "lr_scheduler_state_dict.pt"
PARAMS = "params"


def get_latest_checkpoint_folder(path):
    if path is None:
        return None
    if torch.distributed.get_rank() == 0:
        latest_checkpoint_file = os.path.join(path, "latest_checkpoint.txt")

        if os.path.exists(latest_checkpoint_file):
            with open(latest_checkpoint_file, "r") as f:
                step = int(f.read().strip())
        else:
            step = -1
    else:
        step = 0
    step_tensor = torch.tensor(step, dtype=torch.int64, device=torch.cuda.current_device())
    torch.distributed.all_reduce(step_tensor, op=torch.distributed.ReduceOp.SUM)
    latest_ckpt_step = step_tensor.item()
    if latest_ckpt_step == -1:
        return None
    return latest_ckpt_step


def load_model_with_mmap(model_dir):
    model_path = Path(model_dir)

    bin_file = model_path / MODEL_CHECKPOINT
    if bin_file.exists():
        return torch.load(str(bin_file), mmap=True, weights_only=True, map_location="cpu")

    safetensors_files = list(model_path.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    if safetensors_files:
        from safetensors.torch import load_file

        state_dict = {}
        for file in sorted(safetensors_files):
            state_dict.update(load_file(str(file), device="cpu"))
        return state_dict

    raise FileNotFoundError("未找到模型文件")


def load_model(
    model: FSDPModule,
    last_model_checkpoint: str,
    dcp_api: bool = False,
):
    t1 = sync_cuda_and_get_time()
    full_sd = load_model_with_mmap(last_model_checkpoint)

    if dcp_api:
        set_model_state_dict(
            model=model,
            model_state_dict=full_sd,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )
        return model
    t2 = sync_cuda_and_get_time()

    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    t_start_load = time.time()
    t_load = 0
    t_dist = 0
    for param_name, full_tensor in full_sd.items():
        full_tensor = full_tensor.to(torch.float32)
        cpu_barrier()
        t_end_load = time.time()
        t_load += t_end_load - t_start_load

        t_start_dist = time.time()
        sharded_meta_param = meta_sharded_sd.get(param_name)
        try:
            sharded_tensor = distribute_tensor(
                full_tensor,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
            )
            sharded_sd[param_name] = nn.Parameter(sharded_tensor)
        except Exception as e:
            log(f"Error in loading {param_name}: {e}")

        t_end_dist = time.time()
        t_dist += t_end_dist - t_start_dist
        t_start_load = time.time()

    t3 = sync_cuda_and_get_time()

    # choose `assign=True` since we cannot call `copy_` on meta tensor
    model.load_state_dict(sharded_sd, strict=False, assign=True)
    for p in model.parameters():
        assert p.dtype == torch.float32, f"{p.dtype} != torch.float32"

    t4 = sync_cuda_and_get_time()
    logging_rank0(
        f"perf load model time total: {t4 - t1:.3f} shard {t3 - t2:.3f} load {t2 - t1:.3f} {t_load=} {t_dist=}"
    )
    return model


def load_ema_model(config, load_path, subfolder, dit, load_weight_from_ema_ckpt=True):
    diff_model_cls = None
    if config.train_config.oteam_version == "4.4":
        diff_model_cls = Oteam4_4Transformer2DModel
    elif config.train_config.oteam_version == "4.3":
        diff_model_cls = FluxTransformer2DModel

    ema_dit_net = dit
    if load_weight_from_ema_ckpt:
        assert load_path is not None and subfolder is not None
        log(f"loading ema model from {load_path}/{subfolder}", rank=0)
        load_model(ema_dit_net, os.path.join(load_path, subfolder))
    else:
        log(f"assign dit to ema model", rank=0)

    ema_dit_net = T2iEmaModel(
        ema_dit_net.parameters(),
        decay=config.ema_config.ema_decay,
        model_cls=diff_model_cls,
        model_config=ema_dit_net.config
    )

    # TODO: 只有更新的地方用到了，看起来似乎没太大必要放到显存上？
    ema_dit_net.to(torch.cuda.current_device())
    return ema_dit_net


def load_optimizer(
    model: FSDPModule,
    opt: torch.optim.Optimizer,
    load_path: str,
    latest_step: int = None,
    dcp_api: bool = False
):
    begin_t = sync_cuda_and_get_time()
    assert latest_step is not None
    ckpt_path = f"{load_path}/step-{latest_step}"
    last_optim_checkpoint = f"{ckpt_path}/{OPTIM_CHECKPOINT}"

    full_sd = torch.load(last_optim_checkpoint, mmap=True, weights_only=True, map_location="cpu")
    if dcp_api:
        set_optimizer_state_dict(
            model=model,
            optimizers=opt,
            optim_state_dict=full_sd,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )
        return
    _init_optim_state(opt)
    param_groups = opt.state_dict()["param_groups"]
    state = opt.state_dict()["state"]

    full_param_groups = full_sd["param_groups"]
    full_state = full_sd["state"]

    for param_group, full_param_group in zip(param_groups, full_param_groups):
        for key, value in full_param_group.items():
            if key == PARAMS:
                continue
            param_group[key] = value
        for pid, full_pid in zip(param_group[PARAMS], full_param_group[PARAMS]):
            if pid not in state:
                continue
            param_state = state[pid]
            full_param_state = full_state[full_pid]
            for attr, full_tensor in full_param_state.items():
                sharded_tensor = param_state[attr]
                if isinstance(sharded_tensor, DTensor):
                    # exp_avg is DTensor
                    param_state[attr] = distribute_tensor(
                        full_tensor,
                        sharded_tensor.device_mesh,
                        sharded_tensor.placements,
                    )
                else:
                    # step is plain tensor
                    param_state[attr] = full_tensor
    opt.load_state_dict({
        "param_groups": param_groups,
        "state": state,
    })
    end_t = sync_cuda_and_get_time()
    cpu_barrier()
    log(
        f"successfully loaded optimizer from {ckpt_path} using_time"
        f" {end_t - begin_t} seconds.",
        rank=0
    )
    return opt


def load_scheduler(lr_scheduler, load_dir, latest_step):
    assert latest_step is not None
    ckpt_path = f"{load_dir}/step-{latest_step}"

    state_dict = torch.load(f"{ckpt_path}/{LR_SCHEDULER_CHECKPOINT}")
    lr_scheduler.load_state_dict(state_dict)
    cpu_barrier()
    log(f"successfully loaded scheduler from {ckpt_path}", rank=0)
    return lr_scheduler


def get_full_model_state_dict(model: FSDPModule, dcp_api: bool = False):
    if dcp_api:
        return get_model_state_dict(
            model=model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )

    sharded_sd = model.state_dict()
    cpu_state_dict = {}
    for param_name, sharded_param in sharded_sd.items():
        full_param = sharded_param.full_tensor()
        if torch.distributed.get_rank() == 0:
            cpu_state_dict[param_name] = full_param.cpu()
        else:
            del full_param
    return cpu_state_dict


def get_full_optimizer_state_dict(
    model: FSDPModule, opt: torch.optim.Optimizer, dcp_api: bool = False
):
    if dcp_api:
        return get_optimizer_state_dict(
            model=model,
            optimizers=opt,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
    is_rank_zero = torch.distributed.get_rank() == 0
    sharded_sd = opt.state_dict()
    sharded_state = sharded_sd["state"]
    full_state = {}
    for group_id, sharded_group in sharded_state.items():
        group_state = {}
        for attr, sharded_tensor in sharded_group.items():
            if isinstance(sharded_tensor, DTensor):
                # "exp_avg" in AdamW is `DTensor`
                full_tensor = sharded_tensor.full_tensor()
            else:
                # "step" in AdamW is plain tensor
                full_tensor = sharded_tensor
            if is_rank_zero:
                group_state[attr] = full_tensor.cpu()
            else:
                del full_tensor
        if is_rank_zero:
            full_state[group_id] = group_state
        else:
            del group_state
    if is_rank_zero:
        return {
            "param_groups": sharded_sd["param_groups"],
            "state": full_state,
        }
    else:
        return {}


def save_checkpoint(
    config,
    model: FSDPModule,
    optim: torch.optim.Optimizer,
    lr_scheduler,
    step: int,
    dcp_api: bool = False,
    ema_dit=None,
):

    ck_config = config.checkpoint
    save_dir = ck_config.save_ckpt_path
    if save_dir is None:
        log(f"checkpoint saving ignored due to empty save_ckpt_path", rank=0)
        return
    new_checkpoint_folder = f"{save_dir}/step-{step}"
    log(f"saving checkpoint to {new_checkpoint_folder}", rank=0)

    # get model and optim state dict
    begin_t = sync_cuda_and_get_time()
    model_state_dict = get_full_model_state_dict(model, dcp_api=dcp_api)
    if not ck_config.no_save_optim:
        optim_state_dict = get_full_optimizer_state_dict(model, optim, dcp_api=dcp_api)
    t2 = sync_cuda_and_get_time()

    if torch.distributed.get_rank() == 0:
        new_model_checkpoint = f"{new_checkpoint_folder}/{MODEL_CHECKPOINT}"
        new_optim_checkpoint = f"{new_checkpoint_folder}/{OPTIM_CHECKPOINT}"
        new_lr_scheduler_checkpoint = f"{new_checkpoint_folder}/{LR_SCHEDULER_CHECKPOINT}"
        os.makedirs(new_checkpoint_folder, exist_ok=True)
        torch.save(model_state_dict, new_model_checkpoint)
        if not ck_config.no_save_optim:
            torch.save(optim_state_dict, new_optim_checkpoint)
            torch.save(lr_scheduler.state_dict(), new_lr_scheduler_checkpoint)

    cpu_barrier()
    t3 = sync_cuda_and_get_time()
    if config.ema.use_ema:
        assert ema_dit is not None
        ema_dit.save_pretrained(os.path.join(new_checkpoint_folder, "transformer_ema"))

    end_t = sync_cuda_and_get_time()
    if torch.distributed.get_rank() == 0:
        latest_checkpoint_file = os.path.join(save_dir, "latest_checkpoint.txt")
        with open(latest_checkpoint_file, "w") as f:
            f.write(str(step))
        log(
            f"successfully saved checkpoint to {new_checkpoint_folder} using_time {end_t - begin_t} seconds"
        )
    log(f"perf time gather params {t2 - begin_t} save {t3 - t2} ema {end_t - t3}", rank=0)
    cpu_barrier()
    clear_memory()
