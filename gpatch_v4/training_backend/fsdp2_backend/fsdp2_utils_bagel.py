import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file, save_file
from torch.distributed import distributed_c10d as c10d
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    _init_optim_state,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_init import _init_default_fully_shard_mesh
from torch.distributed.tensor import DTensor, distribute_tensor

from gpatch_v4.core.device import get_device_backend_name
from gpatch_v4.models.bagel.modeling.bagel.modeling_utils import (
    MLPconnector,
    PositionEmbedding,
    TimestepEmbedder,
)
from gpatch_v4.models.bagel.modeling.bagel.qwen2_navit import (
    Qwen2DecoderLayer,
    Qwen2MoEDecoderLayer,
    Qwen2MoTDecoderLayer,
)
from gpatch_v4.models.bagel.modeling.bagel.siglip_navit import (
    SiglipEncoderLayer,
    SiglipVisionTransformer,
)

MODEL_CHECKPOINT = "model.safetensors"
OPTIM_CHECKPOINT = "optim_state_dict.pt"
EMA_CHECKPOINT = "ema.safetensors"
PARAMS = "params"


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy,
        backward_prefetch,
        cpu_offload,
        num_replicate,
        num_shard=8,
        num_to_forward_prefetch=0,
        torch_dist_timeout_minutes=30
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard
        self.num_to_forward_prefetch = num_to_forward_prefetch
        self.torch_dist_timeout_minutes = torch_dist_timeout_minutes


def grad_checkpoint_check_fn(model_args, module):

    recompute_layer = model_args.recompute_layer
    module_options = (
        Qwen2DecoderLayer, SiglipEncoderLayer, MLPconnector, Qwen2MoEDecoderLayer,
        Qwen2MoTDecoderLayer
    )
    if not isinstance(module, module_options):
        return False

    if isinstance(module, (Qwen2DecoderLayer, Qwen2MoEDecoderLayer, Qwen2MoTDecoderLayer)):
        if module.layer_idx >= recompute_layer:
            return False

    return True


def print_model(model, name):
    for (k, v) in model.named_parameters():
        """
        sharded_meta_param.device_mesh,
        sharded_meta_param.placements,
        """
        print(f"{name} {k}: {v.placements}")


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p, alpha=1 - decay)


def _init_device_mesh(fsdp_config):
    torch_dist_timeout_minutes = fsdp_config.torch_dist_timeout_minutes
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            get_device_backend_name(),
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
        c10d._set_pg_timeout(
            timedelta(minutes=torch_dist_timeout_minutes), device_mesh["shard"].get_group()
        )
        c10d._set_pg_timeout(
            timedelta(minutes=torch_dist_timeout_minutes), device_mesh["replicate"].get_group()
        )
        c10d._set_pg_timeout(
            timedelta(minutes=torch_dist_timeout_minutes), c10d._get_default_group()
        )
    else:
        device_mesh = _init_default_fully_shard_mesh()
        c10d._set_pg_timeout(
            timedelta(minutes=torch_dist_timeout_minutes), c10d._get_default_group()
        )
    return device_mesh


def fsdp_wrapper(model, device_mesh, fsdp_config):

    fsdp_kwargs = {"mesh": device_mesh}
    print(f"{model}", flush=True)
    fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    if model.config.visual_gen:
        fully_shard(model.time_embedder, **fsdp_kwargs)
        fully_shard(model.latent_pos_embed, **fsdp_kwargs)

    if model.config.visual_und:
        for layer in model.vit_model.vision_model.encoder.layers:
            fully_shard(layer, **fsdp_kwargs)
        fully_shard(model.vit_model.vision_model, **fsdp_kwargs)
        fully_shard(model.connector, **fsdp_kwargs)
        fully_shard(model.vit_pos_embed, **fsdp_kwargs)

    for layer in model.language_model.model.layers:
        fully_shard(layer, **fsdp_kwargs)

    # fully_shard(model.language_model)
    fully_shard(model, **fsdp_kwargs)

    return model


def fsdp_ema_setup(ema_model, device_mesh, fsdp_config):
    for param in ema_model.parameters():
        param.requires_grad = False
    ema_model = fsdp_wrapper(ema_model, device_mesh, fsdp_config)
    return ema_model


def get_latest_checkpoint_folder(path):
    max_num = None
    if not os.path.exists(path):
        return max_num
    for name in os.listdir(path):
        folder_path = os.path.join(path, name)
        if os.path.isdir(folder_path):
            try:
                num = int(name)
                if max_num is None or num > max_num:
                    max_num = num
            except ValueError:
                pass  # Skip non-numeric folder names
    return max_num


def load_model(ckpt_dir, model: FSDPModule, dcp_api):
    last_model_checkpoint = f"{ckpt_dir}/{MODEL_CHECKPOINT}"
    full_sd = torch.load(last_model_checkpoint, mmap=True, weights_only=True, map_location="cpu")
    if dcp_api:
        set_model_state_dict(
            model=model,
            model_state_dict=full_sd,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )
        return
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    for param_name, full_tensor in full_sd.items():
        sharded_meta_param = meta_sharded_sd.get(param_name)
        sharded_tensor = distribute_tensor(
            full_tensor,
            sharded_meta_param.device_mesh,
            sharded_meta_param.placements,
        )
        sharded_sd[param_name] = nn.Parameter(sharded_tensor)
    # choose `assign=True` since we cannot call `copy_` on meta tensor
    model.load_state_dict(sharded_sd, strict=False, assign=True)


def load_optim(ckpt_dir, model: FSDPModule, opt: torch.optim.Optimizer, dcp_api):
    last_optim_checkpoint = f"{ckpt_dir}/{OPTIM_CHECKPOINT}"
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


def get_full_model_state_dict(model: FSDPModule, dcp_api):
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
        # FSDP2 model.state_dict() can contain a mix of:
        #   - DTensor-sharded parameters
        #   - plain Tensor buffers (e.g. MoE router correction bias)
        # Only DTensor exposes .full_tensor(); plain Tensor should be used directly.
        if isinstance(sharded_param, DTensor):
            full_param = sharded_param.full_tensor()
        else:
            full_param = sharded_param
        if torch.distributed.get_rank() == 0:
            cpu_state_dict[param_name] = full_param.cpu()
        else:
            del full_param
    return cpu_state_dict


def get_full_optimizer_state_dict(
    model: FSDPModule,
    opt: torch.optim.Optimizer,
    dcp_api,
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


def save(folder, cur_step, model: FSDPModule, optim: torch.optim.Optimizer):
    model_state_dict = get_full_model_state_dict(model, False)
    optim_state_dict = get_full_optimizer_state_dict(model, optim, True)
    if torch.distributed.get_rank() == 0:
        new_checkpoint_folder = f"{folder}/{cur_step:07d}"
        new_model_checkpoint = f"{new_checkpoint_folder}/{MODEL_CHECKPOINT}"
        new_optim_checkpoint = f"{new_checkpoint_folder}/{OPTIM_CHECKPOINT}"
        os.makedirs(new_checkpoint_folder, exist_ok=True)
        save_file(model_state_dict, new_model_checkpoint)
        torch.save(optim_state_dict, new_optim_checkpoint)


def save_ema(foldr, ema_model):
    if ema_model is None:
        return
    ema_state_dict = get_full_model_state_dict(ema_model, False)
    if torch.distributed.get_rank() == 0:
        ckpt = f"{foldr}/{EMA_CHECKPOINT}"
        save_file(ema_state_dict, ckpt)


def load_state_dict(file_path, device):
    full_sd = torch.load(file_path, mmap=True, weights_only=True, map_location=device)
    return full_sd


class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir,
        train_steps,
        model,
        ema_model,
        optimizer,
        scheduler,
        data_status,
        logger,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        # ema
        save_ema(save_path, ema_model)
        # model and optimizer
        save(ckpt_dir, train_steps, model, optimizer)

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if dist.get_rank() == 0 and data_status is not None:
            torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        dist.barrier()
        return

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            if resume_from_ema:
                model_state_dict_path = os.path.join(resume_from, EMA_CHECKPOINT)
            else:
                model_state_dict_path = os.path.join(resume_from, MODEL_CHECKPOINT)

            model_state_dict = load_file(model_state_dict_path, device="cpu")
            # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
            # which makes it easier to adapt to different resolutions.
            model_state_dict.pop('latent_pos_embed.pos_embed')
            model_state_dict.pop('vit_pos_embed.pos_embed')
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(msg)
            del model_state_dict

            if ema_model is not None:
                ema_state_dict_path = os.path.join(resume_from, EMA_CHECKPOINT)
                if not os.path.exists(ema_state_dict_path):
                    logger.info(f"replicaing ema model from {model_state_dict_path}.")
                    ema_state_dict_path = model_state_dict_path
                ema_state_dict = load_file(ema_state_dict_path, device="cpu")
                # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
                # which makes it easier to adapt to different resolutions.
                ema_state_dict.pop('latent_pos_embed.pos_embed')
                ema_state_dict.pop('vit_pos_embed.pos_embed')
                msg = ema_model.load_state_dict(ema_state_dict, strict=False)
                logger.info(msg)
                del ema_state_dict
        else:
            logger.info(f"Training from scratch.")
        return model, ema_model

    @staticmethod
    def try_load_train_state(resume_from, model, optimizer, scheduler):
        if resume_from is not None and os.path.exists(resume_from):

            # load optimizer
            load_optim(resume_from, model, optimizer, True)

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(
                scheduler_state_dict_path, weights_only=True, map_location="cpu"
            )
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status
