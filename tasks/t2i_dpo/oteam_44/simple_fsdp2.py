import contextlib
import dataclasses
import functools
import gc
import os
import random

import diffusers
import numpy as np
import torch
import torch.distributed as dist
import transformers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from packaging import version
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_init import _init_default_fully_shard_mesh
from torchvision import transforms
from torchvision.transforms.functional import crop
from transformers import AutoTokenizer, CLIPTextModel, CLIPTokenizer, PretrainedConfig
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from tasks.t2i_dpo.oteam_44.data import build_dataloader

from gpatch_v4.configs.config import T2iDpoConfig
from gpatch_v4.trainer.t2i_dpo_trainer import BaseDpoTrainer


class SimpleDpoTrainer(BaseDpoTrainer):
    config_cls = T2iDpoConfig

    def wrap_model(self, model):
        if dist.get_rank() == 0:
            print(f"{model}", flush=True)
        fsdp_kwargs = {"mesh": self.device_mesh}
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=self.weight_dtype,
            reduce_dtype=torch.float32,
        )
        for layer in model.down_blocks:
            fully_shard(layer, **fsdp_kwargs)
        fully_shard(model.mid_block, **fsdp_kwargs)
        for layer in model.up_blocks:
            fully_shard(layer, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)
        return model

    def build_train_valid_test_data_iter(self):
        args = self.args
        tokenizer = self.tokenizer
        return build_dataloader(args, tokenizer)

    def prepare_batch(self, batch):
        vae = self.vae
        args = self.args
        text_encoder = self.text_encoder
        weight_dtype = self.weight_dtype
        noise_scheduler = self.noise_scheduler
        # Convert images to latent space
        if args.training.train_method == 'dpo':
            # y_w and y_l were concatenated along channel dimension
            feed_pixel_values = batch["pixel_values"]
            print(f"pixel_values {feed_pixel_values.shape}", flush=True)
            feed_pixel_values = torch.cat(feed_pixel_values.chunk(2, dim=1))
            # If using AIF then we haven't ranked yet so do so now
            # Only implemented for BS=1 (assert-protected)
            if args.training.choice_model:
                assert False
        elif args.training.train_method == 'sft':
            feed_pixel_values = batch["pixel_values"]

        #### Diffusion Stuff ####
        # encode pixels --> latents
        with torch.no_grad():
            latents = vae.encode(feed_pixel_values.to(weight_dtype)).latent_dist.sample()
            print(f"latents {latents.shape}", flush=True)
            latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        # variants of noising
        if args.training.noise_offset:  # haven't tried yet
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += args.training.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
            )
        if args.training.input_perturbation:  # haven't tried yet
            new_noise = noise + args.input_perturbation * torch.randn_like(noise)

        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz, ), device=latents.device
        )
        timesteps = timesteps.long()
        if args.training.train_method == 'dpo':  # make timesteps and noise same for pairs in DPO
            timesteps = timesteps.chunk(2)[0].repeat(2)
            noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)

        noisy_latents = noise_scheduler.add_noise(
            latents, new_noise if args.training.input_perturbation else noise, timesteps
        )
        ### START PREP BATCH ###
        # only support sd1.5
        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(batch["input_ids"])[0]
        if args.training.train_method == 'dpo':
            encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)

        prepared_batch = {
            "noisy_latents": noisy_latents,
            "timesteps": timesteps,
            "encoder_hidden_states": encoder_hidden_states,
        }
        return noise, prepared_batch

    def model_forward(self, model, batch):
        """ unet of ref model"""

        noisy_latents = batch["noisy_latents"]
        timesteps = batch["timesteps"]
        encoder_hidden_states = batch["encoder_hidden_states"]

        model_batch_args = (noisy_latents, timesteps, encoder_hidden_states)
        added_cond_kwargs = None

        model_pred = model(*model_batch_args, added_cond_kwargs=added_cond_kwargs).sample

        if dist.get_rank() == 0:
            print(
                f"noisy_latents {noisy_latents.shape}; timesteps {timesteps.shape};  model_pred {model_pred.shape}",
                flush=True
            )
        return model_pred

    def build_model_and_optimizer(self):
        args = self.args

        weight_dtype = torch.float32
        if args.training.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.training.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        ### START DIFFUSION BOILERPLATE ###
        # Load scheduler, tokenizer and models.
        noise_scheduler = DDPMScheduler.from_pretrained(
            args.training.pretrained_model_name_or_path, subfolder="scheduler"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            args.training.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.training.revision
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            args.training.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.training.revision
        )

        self.text_encoders = None
        self.tokenizers = None
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        # Can custom-select VAE (used in original SDXL tuning)
        vae_path = (
            args.training.pretrained_model_name_or_path
            if args.training.pretrained_vae_model_name_or_path is None else
            args.training.pretrained_vae_model_name_or_path
        )
        vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae" if args.training.pretrained_vae_model_name_or_path is None else None,
            revision=args.training.revision
        )
        # clone of model
        ref_unet = UNet2DConditionModel.from_pretrained(
            args.training.unet_init
            if args.training.unet_init else args.training.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.training.revision
        )

        if args.training.unet_init:
            print("Initializing unet from", args.unet_init)

        unet = UNet2DConditionModel.from_pretrained(
            args.training.unet_init
            if args.training.unet_init else args.training.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.training.revision
        )

        text_encoder.to("cuda", dtype=weight_dtype)
        if args.training.train_method == 'dpo':
            ref_unet.to("cuda")

        unet = unet.to("cuda")
        vae.to("cuda", dtype=weight_dtype)

        # Freeze vae, text_encoder(s), reference unet
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        if args.training.train_method == 'dpo':
            ref_unet.requires_grad_(False)

        unet = self.wrap_model(unet)
        ref_unet = self.wrap_model(ref_unet)

        if args.training.recompute:  #  (args.sdxl and ('turbo' not in args.pretrained_model_name_or_path) ):
            print(
                "Enabling gradient checkpointing, either because you asked for this or because you're using SDXL"
            )
            unet.enable_gradient_checkpointing()

        optimizer = torch.optim.AdamW(
            unet.parameters(),
            lr=args.optimizer.lr,
            betas=(args.optimizer.adam_beta1, args.optimizer.adam_beta2),
            weight_decay=args.optimizer.weight_decay,
            eps=args.optimizer.adam_epsilon,
        )

        if args.optimizer.lr_decay_style == 'cosine':
            assert args.max_train_steps is not None
            lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=args.optimizer.lr_warmup_steps,
                num_training_steps=args.optimizer.max_train_steps,
                min_lr=args.optimizer.min_lr,
            )
        elif args.optimizer.lr_decay_style == 'constant_with_warmup':
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer, num_warmup_steps=args.optimizer.lr_warmup_steps
            )
        elif args.optimizer.lr_decay_style == "constant":
            lr_scheduler = get_constant_schedule(optimizer=optimizer)
        else:
            raise ValueError

        grad_scaler = None
        if weight_dtype == torch.float16:
            grad_scaler = GradScaler()

        self.ref_unet = ref_unet
        self.unet = unet
        self.vae = vae
        self.optimizer = optimizer
        self.noise_scheduler = noise_scheduler
        self.grad_scaler = grad_scaler
        self.lr_scheduler = lr_scheduler

        self.load_ckpt()

    def save_ckpt(self):
        pass

    def load_ckpt(self):
        args = self.args
        global_step = 0
        self.global_step = global_step
