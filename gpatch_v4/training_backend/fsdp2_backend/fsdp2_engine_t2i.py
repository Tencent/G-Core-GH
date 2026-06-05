import math
import os

import torch
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import randn_tensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor

from gpatch_v4.core.parallel_state import cpu_barrier
from gpatch_v4.training_backend.common.swap_mixin import EngineSwapMixin
from gpatch_v4.training_backend.fsdp2_backend.checkpoint_t2i import (
    get_latest_checkpoint_folder,
    load_ema_model,
    load_model,
    save_checkpoint,
)
from gpatch_v4.training_backend.fsdp2_backend.engine_helper_t2i import (
    Fsdp2EngineMixinT2i,
    setup_lr_scheduler,
    setup_optimizer,
)
from gpatch_v4.training_backend.fsdp2_backend.fsdp2_swap_impl import Fsdp2SwapImpl
from gpatch_v4.utils import log, logging_rank0


class Fsdp2EngineT2i(Fsdp2EngineMixinT2i, EngineSwapMixin):
    def __init__(self, config, policy_config, post_meta_init_fn=None):
        self.config = config
        self.policy_config = policy_config
        self.post_meta_init_fn = post_meta_init_fn
        # only create meta model now
        self.device_mesh = self._create_device_mesh()
        self.model = self._create_model(
            device_mesh=self.device_mesh, post_meta_init_fn=post_meta_init_fn
        )
        self.ema_model = None
        self.ref_model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.hf_model_config = None
        self.swap_impl = Fsdp2SwapImpl()

        if not self.policy_config.without_ref:
            self.ref_model = self._create_model(
                freeze_weights=True,
                device_mesh=self.device_mesh,
                post_meta_init_fn=post_meta_init_fn
            )

        if self.config.ema.use_ema:
            self.ema_model = self._create_model(post_meta_init_fn=post_meta_init_fn)

    def eval_mode(self):
        self.model.eval()

    def train_mode(self):
        self.model.train()

    def setup_model_and_optimizer(self):
        policy_config = self.policy_config
        checkpoint_config = self.config.checkpoint
        latest_step = get_latest_checkpoint_folder(checkpoint_config.load_ckpt_path)

        latest_saved_step = 0
        # load model and ema model
        if latest_step is None:
            logging_rank0(
                f"not found step in {checkpoint_config.load_ckpt_path}, start from original path"
                f" {policy_config.hf_model_path}"
            )
            logging_rank0("begin load_model")
            self.model = load_model(
                self.model, os.path.join(policy_config.hf_model_path, "transformer")
            )
            logging_rank0("end load_model 1")
            cpu_barrier()
            logging_rank0("end load_model 2")
            if self.config.ema.use_ema:
                self.ema_model = load_ema_model(
                    self.config, None, "transformer", self.model, load_weight_from_ema_ckpt=False
                )
            cpu_barrier()
            logging_rank0(f"successfully start from {policy_config.hf_model_path}")
        else:
            latest_saved_step = latest_step
            ckpt_path = f"{checkpoint_config.load_ckpt_path}/step-{latest_step}"
            logging_rank0(f"Loading checkpoint from {ckpt_path}")
            self.model = load_model(self.model, ckpt_path)
            cpu_barrier()
            if self.config.ema.use_ema:
                self.ema_model = load_ema_model(
                    self.config,
                    ckpt_path,
                    "transformer_ema",
                    self.ema_model,
                    load_weight_from_ema_ckpt=True
                )
            cpu_barrier()
            logging_rank0(f"successfully start from {ckpt_path}")

        # TODO(hessianliu): load ref model
        if self.ref_model is not None:
            self.sync_model(self.ref_model, self.model)

        # optimizer
        if self.config.debug.debug_no_optim:
            optimizer = None
        else:
            optimizer = setup_optimizer(self.config, self.model, latest_step)
        self.optimizer = optimizer

        # lr scheduler
        # 如果是 RL 就会有 total_ppo_step，否则就是 SFT，用的 train step。
        if hasattr(self.config.training, 'total_ppo_step'):
            self.lr_scheduler = setup_lr_scheduler(
                self.config, self.optimizer, self.config.training.total_ppo_step, latest_step
            )
        else:
            self.lr_scheduler = setup_lr_scheduler(
                self.config, self.optimizer, self.config.training.total_training_step, latest_step
            )

        self.hf_model_config = self.model.config
        return latest_saved_step

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def update_ema(self):
        if self.config.ema.use_ema:
            assert self.ema_model is not None
            self.ema_model.step(self.model.parameters())

    def clip_grad_norm_(self):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.config.optimizer.max_grad_norm
        )
        grad_norm = grad_norm.full_tensor()
        return grad_norm

    def save(self, curr_step):
        save_checkpoint(
            self.config,
            self.model,
            self.optimizer,
            self.lr_scheduler,
            step=curr_step,
            ema_dit=self.ema_model
        )

    def sync_model(self, dst_model, src_model):
        dst_model.to_empty(device="cuda")
        dst_params = []
        src_params = []
        for (dst_p, p) in zip(dst_model.parameters(), src_model.parameters()):
            dst_params.append(dst_p)
            src_params.append(p)
        # torch._foreach_copy_(dst_params, src_params) do not work for dtensor
        torch._foreach_zero_(dst_params)
        torch._foreach_add_(dst_params, src_params, alpha=1.0)
