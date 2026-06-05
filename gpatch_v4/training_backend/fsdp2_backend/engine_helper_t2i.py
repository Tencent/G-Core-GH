# TODO rename it fsdp2_engine_mixin_t2i.py
import os

import torch
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor

from gpatch_v4.core import constants
from gpatch_v4.core.parallel_state import cpu_barrier
from gpatch_v4.models import REGISTER_MODEL_CLS
from gpatch_v4.training_backend.fsdp2_backend.checkpoint_t2i import load_optimizer, load_scheduler


def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    layers_list = [model.transformer_blocks, model.single_transformer_blocks]
    for layers in layers_list:
        for i, layer in enumerate(layers):
            if i >= len(layers) - num_to_forward_prefetch:
                break
            layers_to_prefetch = [layers[i + j] for j in range(1, num_to_forward_prefetch + 1)]
            layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    layers_list = [model.transformer_blocks, model.single_transformer_blocks]
    for layers in layers_list:
        for i, layer in enumerate(layers):
            if i < num_to_backward_prefetch:
                continue
            layers_to_prefetch = [layers[i - j] for j in range(1, num_to_backward_prefetch + 1)]
            layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def setup_optimizer(config, model, latest_step=None):
    params_to_optimize = model.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer_config = config.optimizer
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=optimizer_config.lr,
        betas=(optimizer_config.adam_beta1, optimizer_config.adam_beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.adam_epsilon,
    )
    if latest_step is not None and (not config.checkpoint.no_load_optim):
        optimizer = load_optimizer(model, optimizer, config.checkpoint.load_ckpt_path, latest_step)
    return optimizer


def setup_lr_scheduler(config, optimizer, total_step, latest_step=None):
    optimizer_config = config.optimizer
    lr_scheduler = get_scheduler(
        optimizer_config.lr_decay_style,
        optimizer=optimizer,
        num_warmup_steps=optimizer_config.lr_warmup_steps,
        num_training_steps=total_step,
        num_cycles=optimizer_config.lr_num_cycles,
        power=optimizer_config.lr_power,
    )
    if latest_step is not None and (not config.checkpoint.no_load_optim):
        lr_scheduler = load_scheduler(lr_scheduler, config.checkpoint.load_ckpt_path, latest_step)
    return lr_scheduler


class Fsdp2EngineMixinT2i:
    def _create_device_mesh(self, ):
        dist_config = self.policy_config.dist_config
        if not dist_config.enable_custom_device_mesh:
            return None

        assert dist_config.custom_device_mesh is not None
        device_mesh = tuple(dist_config.custom_device_mesh)
        assert len(device_mesh
                  ) == 2 and device_mesh[0] * device_mesh[1] == torch.distributed.get_world_size()
        device_type = "cuda"
        mesh_2d = init_device_mesh(
            device_type, mesh_shape=device_mesh, mesh_dim_names=("replicate", "shard")
        )
        return mesh_2d

    # TODO use config subfolder
    def _create_model(
        self,
        subfolder="transformer",
        mix_precision=True,
        enable_fwd_bwd_prefetch=True,
        check_weight_type=torch.float32,
        freeze_weights=False,
        device_mesh=None,
        post_meta_init_fn=None,
    ):
        policy_config = self.policy_config
        dist_config = policy_config.dist_config
        enable_recompute = self.config.training.recompute
        model_path = policy_config.hf_model_path

        # FSDP2 不需要将 cast 到 bf16，能自己用类似 megatron 的方式处理好 mixed precision
        # https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#enabling-mixed-precision
        diff_model_cls = None
        if policy_config.model_arch in REGISTER_MODEL_CLS.keys():
            diff_model_cls = REGISTER_MODEL_CLS[policy_config.model_arch]
        else:
            raise NotImplementedError(f'unknown {policy_config.model_arch=}')

        config = diff_model_cls.load_config(os.path.join(model_path, subfolder))
        with torch.device('meta'):
            dit = diff_model_cls.from_config(config)

        # Optional post-meta-device-init hook (e.g. re-materializing RoPE freqs).
        # Passed in from the extended pipeline that knows model-specific details.
        if post_meta_init_fn is not None:
            post_meta_init_fn(dit)

        if enable_recompute:
            dit.enable_gradient_checkpointing()

        if freeze_weights:
            dit.requires_grad_(False)

        fsdp_kwargs = {}
        if mix_precision:
            fsdp_kwargs['mp_policy'] = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False
            )

        # 下面参数打开 zero2，老 oom，算了，用回 zero3
        # fsdp_kwargs['reshard_after_forward'] = 128

        # 在所有卡上开一个整个的 zero3 会因为通信压力太大而影响训练的吞吐，跟厂商的支持一起优化后，
        # 设置一个 2d 的 device mesh 可以降低通信压力，下面是 12B 模型的两个经验数字
        # 1024 分辨率 -> mesh (4, 256)
        # 256 -> mesh (32, 32)
        if device_mesh is None and dist_config.enable_custom_device_mesh:
            device_mesh = self._create_device_mesh()
        fsdp_kwargs["mesh"] = device_mesh
        for layer in dit.transformer_blocks:
            fully_shard(layer, **fsdp_kwargs)
        if hasattr(dit, 'single_transformer_blocks'):
            # qwen image edit 没有 single_transformer_blocks，可能做成 by model 的更好，先这样子吧。
            for layer in dit.single_transformer_blocks:
                fully_shard(layer, **fsdp_kwargs)
        fully_shard(dit, **fsdp_kwargs)
        # sharded parameters are float32
        for p in dit.parameters():
            assert p.dtype == check_weight_type, f"{p.dtype} != {check_weight_type}"

        if enable_fwd_bwd_prefetch:
            if dist_config.fsdp2_num_to_forward_prefetch > 1:
                set_modules_to_forward_prefetch(
                    dit, num_to_forward_prefetch=dist_config.fsdp2_num_to_forward_prefetch
                )
            if dist_config.fsdp2_num_to_backward_prefetch > 1:
                set_modules_to_backward_prefetch(
                    dit, num_to_backward_prefetch=dist_config.fsdp2_num_to_backward_prefetch
                )
        cpu_barrier()
        return dit
