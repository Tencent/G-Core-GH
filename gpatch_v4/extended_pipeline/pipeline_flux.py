import math
import os
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.extended_pipeline.mixin import ExtendedPipelineMixin, FluxLikePipelineMixin
from gpatch_v4.extended_pipeline.pipeline_base import ExtendedPipelineAbc
from gpatch_v4.training_backend import TrainingEngineFactory
from gpatch_v4.utils import (
    average_losses_across_data_parallel_group,
    extend_value_to_dict,
    get_iterator_k_split_list,
    log,
    repeat_interleave_tensor_or_list,
    sync_cuda_and_get_time,
    unbind_tensor_to_list,
)


class FluxPipeline(ExtendedPipelineAbc, ExtendedPipelineMixin, FluxLikePipelineMixin):
    """Extended pipeline for Flux-based text-to-image diffusion models.

    Parameters
    ----------
    config : object
    """
    def __init__(self, config):
        self.config = config
        extra_args = {"policy_config": config.policy}
        self.model = TrainingEngineFactory.get_training_engine(config, **extra_args)
        self.vae = None
        self.scheduler = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.tokenizer = None
        self.tokenizer_2 = None
        self.image_processor = None

    @override
    def setup_pipeline(self):
        # Disable cuDNN SDPA backend to avoid MHA graph execution failures
        # during backward pass with bf16 mixed precision (FSDP2).
        # The cuDNN backend's MHA graph can produce CUBLAS_STATUS_EXECUTION_FAILED.
        # Flash Attention and math backends are unaffected.
        # https://github.com/pytorch/pytorch/issues/134001
        # https://github.com/pytorch/pytorch/issues/138581
        torch.backends.cuda.enable_cudnn_sdp(False)

        model_path = self.config.policy.hf_model_path
        begin_t = sync_cuda_and_get_time()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_2")
        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ).to(torch.cuda.current_device())

        self.text_encoder = CLIPTextModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        ).to(torch.cuda.current_device())
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16,
        ).to(torch.cuda.current_device())
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        end_t = sync_cuda_and_get_time()
        log(f"loading encoder takes {end_t - begin_t:.2f} s", rank=0)

        # TODO: 两个 encoder 模型是否有必要做 fully_shard，还是直接做 offload？
        # self.text_encoder = fsdp2_fully_shard(text_encoder)
        # self.text_encoder_2 = fsdp2_fully_shard(text_encoder_2)

        prev_ppo_step = self.model.setup_model_and_optimizer()

        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels) -
                                    1) if getattr(self, "vae", None) else 8
        # 注意 oteam image_processor 设置的 vae_scale_factor 是 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )

        # set some config
        self.tokenizer_max_length = self.tokenizer.model_max_length
        self.num_channels_latents = self.vae.config.latent_channels

        log(
            f"FluxPipeline.setup_pipeline {self.num_channels_latents=} {self.vae_scale_factor=}",
            rank=0
        )
        return prev_ppo_step

    @override
    def encode_prompt(
        self,
        prompt: Union[str, List[str]] = None,
        **kwargs,
    ):
        with torch.no_grad():
            prompt = [prompt] if isinstance(prompt, str) else prompt
            device = torch.cuda.current_device()
            if hasattr(self.text_encoder, "module"):
                dtype = self.text_encoder.module.dtype
            else:
                dtype = self.text_encoder.dtype

            pooled_prompt_embeds = self._encode_prompt_with_clip(prompt)
            prompt_embeds = self._encode_prompt_with_t5(prompt)
            # 都是 0，所以 flux 刚好不出问题
            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "text_ids": text_ids,
        }

    @override
    def repeat_interleave_tensor_or_list(
        self,
        rb: Dict[str, Union[List[Any], torch.Tensor]],
        repeat: int,
    ):
        for k, v in rb.items():
            if k != "text_ids":
                rb[k] = repeat_interleave_tensor_or_list(v, repeat)

    @override
    def permute_timesteps(
        self,
        rb: Dict[str, List[Any]],
    ):
        expected_len = len(rb["timesteps"])

        # for debugging purpose, you can simplify the ordering as following:
        # ```
        # perms = [torch.arange(rb["timesteps"][0].shape[0]) for _ in range(expected_len)]
        # ```
        perms = [torch.randperm(rb["timesteps"][0].shape[0]) for _ in range(expected_len)]

        perm_keys = ["timesteps", "latents", "next_latents", "log_probs"]
        if self.config.training.disable_cfg_uncond_grad:
            perm_keys.append("cfg_neg_preds")
        if self.config.training.t2i_cfg_logps_impl_v2:
            perm_keys.append("cfg_pos_latents")

        # 只对有 sampling_step 的 tensor 做 randperm
        for key in perm_keys:
            assert key in rb, f'{key} not in {rb.keys()}'
            rb[key] = [rb[key][i][perms[i]] for i in range(expected_len)]
        rb["perms"] = perms

    def prepare_latent_image_ids(self, batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        # FIXME 如果 height / width 太大，这里容易溢出 bf16 的 7 个 significant bits。
        return latent_image_ids.to(device=device, dtype=dtype)

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)
        repeat_n = self.config.training.sampling_repeat_n
        assert batch_size % repeat_n == 0
        assert latents is None

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # latent 统一用 fp32
        if self.config.training.init_same_noise:
            latents = []
            for tmp_i in range(batch_size // repeat_n):
                tmp = randn_tensor(
                    (num_channels_latents, height, width), device=device, dtype=torch.float32
                )
                for tmp_j in range(repeat_n):
                    latents.append(tmp)
            latents = torch.stack(latents)
            assert latents.shape == shape
        else:
            latents = randn_tensor(shape, device=device, dtype=torch.float32)

        latents = self.pack_latents(latents, batch_size, num_channels_latents, height, width)
        latent_image_ids = self.prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype
        )
        return latents, latent_image_ids

    @torch.no_grad()
    @override
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        **kwargs,
    ):
        batch_size = prompt_embeds.shape[0]
        device = torch.device(torch.cuda.current_device())
        repeat_n = self.config.training.sampling_repeat_n
        model_mbs = self.config.training.rollout_model_mbs

        latents, latent_image_ids = self.prepare_latents(
            batch_size,
            self.num_channels_latents,
            self.config.training.height,
            self.config.training.width,
            prompt_embeds.dtype,
            device=device,
            generator=generator,
        )

        self.setup_scheduler_and_timesteps()

        all_latents = [latents]
        all_log_probs = []

        if self.model.hf_model_config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        self.scheduler.set_begin_index(0)
        self.model.eval_mode()
        for si in range(self.config.training.sampling_steps):
            timestep_value = self.timesteps[si]
            timesteps = torch.full(
                [latents.shape[0]], timestep_value, device=latents.device, dtype=torch.long
            )
            assert latents.dtype == torch.float32

            if self.config.training.use_torch_autocast:
                # TODO dtype config
                autocast_ctx = torch.autocast("cuda", torch.bfloat16)
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()
            with autocast_ctx:

                # Typical shapes of Flux, just in case U need it.
                # ```
                # latents.shape=torch.Size([b, s, 64])
                # prompt_embeds.shape=torch.Size([b, s_t, 4096])
                # timesteps.shape=torch.Size([b])
                # text_ids.shape=torch.Size([s_t, 3])
                # pooled_prompt_embeds.shape=torch.Size([b, 768])
                # latent_image_ids.shape=torch.Size([s, 3])
                # noise_pred.shape=torch.Size([b, s, 64])
                # ```
                if model_mbs is None:
                    # 一次性计算提高效率，但会导致与 train 轻微的 logps 计算误差；如果 clip
                    # 比较严格，会导致问题。
                    noise_pred = self.model(
                        hidden_states=latents.bfloat16(),
                        encoder_hidden_states=prompt_embeds,
                        timestep=timesteps / 1000,
                        guidance=guidance,
                        txt_ids=text_ids,
                        pooled_projections=pooled_prompt_embeds,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]
                else:
                    assert repeat_n % model_mbs == 0
                    noise_pred = []
                    for model_mbi in range(repeat_n // model_mbs):
                        start_idx = model_mbi * model_mbs
                        end_idx = (model_mbi + 1) * model_mbs
                        _noise_pred = self.model(
                            hidden_states=latents[start_idx:end_idx].bfloat16(),
                            encoder_hidden_states=prompt_embeds[start_idx:end_idx],
                            timestep=(timesteps / 1000)[start_idx:end_idx],
                            guidance=guidance[start_idx:end_idx],
                            txt_ids=text_ids,
                            pooled_projections=pooled_prompt_embeds[start_idx:end_idx],
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=None,
                            return_dict=False,
                        )[0]
                        noise_pred.append(_noise_pred)
                    noise_pred = torch.cat(noise_pred, dim=0)

            assert noise_pred.dtype == torch.bfloat16
            assert latents.dtype == torch.float32

            if model_mbs is None:
                latents, ode_latents, log_prob, _ = self.step_noise_scheduler(
                    noise_pred,
                    latents,
                    self.sigma_schedule,
                    prev_sample=None,
                    added_noise=None,
                    eta=self.config.training.eta,
                    index=si,
                )
            else:
                assert repeat_n % model_mbs == 0
                assert self.config.training.rollout_mbs == 1
                tmp_latents = []
                ode_latents = []
                log_prob = []
                for model_mbi in range(repeat_n // model_mbs):
                    start_idx = model_mbi * model_mbs
                    end_idx = (model_mbi + 1) * model_mbs
                    _latents, _ode_latents, _log_prob, _ = self.step_noise_scheduler(
                        noise_pred[start_idx:end_idx],
                        latents[start_idx:end_idx],
                        self.sigma_schedule,
                        prev_sample=None,
                        added_noise=None,
                        eta=self.config.training.eta,
                        index=si,
                    )
                    tmp_latents.append(_latents)
                    ode_latents.append(_ode_latents)
                    log_prob.append(_log_prob)
                latents = torch.cat(tmp_latents, dim=0)
                ode_latents = torch.cat(ode_latents, dim=0)
                log_prob = torch.cat(log_prob, dim=0)

            assert latents.dtype == torch.float32
            all_latents.append(latents)
            all_log_probs.append(log_prob)

        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1)

        # decode to images
        # vae.enable_tiling() 原先代码这么写，但 xt 没加上去，对齐一波之后，先不想动了
        images = self.decode_to_images(ode_latents)

        # change batch into list and return List[Dict[str, List[Any]]]
        batch_size = latents.shape[0]
        batch_latents_lst = unbind_tensor_to_list(
            all_latents[:, :-1][:, :-1].detach().clone().cpu()
        )
        next_latents_lst = unbind_tensor_to_list(all_latents[:, 1:][:, :-1].detach().clone().cpu())
        all_log_probs_lst = unbind_tensor_to_list(all_log_probs[:, :-1].detach().clone().cpu())

        latent_image_ids_lst = [latent_image_ids.cpu() for _ in range(batch_size)]
        timesteps = self.timesteps[:self.sampling_steps - 1].detach().clone().cpu()
        timestep_lst = [timesteps for _ in range(batch_size)]
        text_ids_lst = [text_ids.cpu() for _ in range(batch_size)]

        # 虽然有点怪，但是 FluxPipeline 的 prompt 是 list，images 也是 list
        # TODO rename next_latents to prev_latents
        rb = {
            'images': images,
            'encoder_hidden_states': unbind_tensor_to_list(prompt_embeds.cpu()),
            'pooled_prompt_embeds': unbind_tensor_to_list(pooled_prompt_embeds.cpu()),
            'text_ids': text_ids_lst,
            'image_ids': latent_image_ids_lst,
            'latents': batch_latents_lst,
            'next_latents': next_latents_lst,
            'log_probs': all_log_probs_lst,
            'timesteps': timestep_lst,
        }
        rm_req_rb = {
            'images': images,
            'prompt': prompt,
        }
        return rb, rm_req_rb

    def prepare_data_for_training(self, micro_rollout_batch: List[Dict[str, Any]]):
        batch = {
            "encoder_hidden_states": [],
            "pooled_prompt_embeds": [],
            "text_ids": [],
            "image_ids": [],
            "latents": [],
            "next_latents": [],
            "log_probs": [],
            "timesteps": [],
            "advantages": [],
            "perms": [],
        }
        batch_keys = batch.keys()
        for data in micro_rollout_batch:
            for key in batch_keys:
                batch[key].append(data[key])

        for key in batch_keys:
            batch[key] = torch.stack(batch[key]).cuda(non_blocking=True)
        return batch

    @override
    def ppo_train_step(self, rollout_batches: List[Dict[str, Any]]):
        device = torch.device(torch.cuda.current_device())
        training_config = self.config.training
        gas_wo_t = training_config.train_gas_wo_timestep
        num_microbatches = training_config.train_gbs_wo_timestep // (
            training_config.train_mbs * mpu.get_data_parallel_world_size()
        )
        assert training_config.train_mbs == 1, f"temporally force {training_config.train_mbs=} == 1"

        self.model.train_mode()
        self.model.optimizer.zero_grad()

        if self.model.hf_model_config.guidance_embeds:
            guidance = torch.full(
                [1], training_config.guidance_scale, device=device, dtype=torch.float32
            )
            guidance = guidance.expand(training_config.train_mbs)
        else:
            guidance = None

        metrics = {}
        data_iter = get_iterator_k_split_list(rollout_batches, num_microbatches)
        for mb_i in range(num_microbatches):
            micro_batch_data = next(data_iter)
            data = self.prepare_data_for_training(micro_batch_data)

            perm = data["perms"]
            advantages = data["advantages"]
            train_timesteps = int(len(data["timesteps"][0]) * training_config.timestep_fraction)

            for ti in range(train_timesteps):
                input_latent = data["latents"][:, ti]
                next_latent = data["next_latents"][:, ti]
                prev_log_probs = data["log_probs"][:, ti]
                timestep = data["timesteps"][:, ti]
                assert input_latent.dtype == torch.float32 and next_latent.dtype == torch.float32

                if self.config.training.use_torch_autocast:
                    # TODO dtype config
                    autocast_ctx = torch.autocast("cuda", torch.bfloat16)
                else:
                    from contextlib import nullcontext
                    autocast_ctx = nullcontext()
                with autocast_ctx:
                    noise_pred = self.model(
                        hidden_states=input_latent.bfloat16(),
                        encoder_hidden_states=data["encoder_hidden_states"],
                        timestep=timestep / 1000,
                        guidance=guidance,
                        txt_ids=data["text_ids"][0].squeeze(0),
                        pooled_projections=data['pooled_prompt_embeds'],
                        img_ids=data["image_ids"][0].squeeze(0),
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]

                curr_log_probs = []
                for ii in range(training_config.train_mbs):
                    _, _, logps, _ = self.step_noise_scheduler(
                        noise_pred[ii].unsqueeze(0),
                        input_latent[ii].unsqueeze(0),
                        self.sigma_schedule,
                        prev_sample=next_latent[ii].unsqueeze(0),
                        added_noise=None,
                        eta=self.config.training.eta,
                        index=perm[ii][ti],
                    )
                    curr_log_probs.append(logps)

                curr_log_probs = torch.cat(curr_log_probs, dim=0)

                loss, ratio_detached = self.calculate_grpo_loss(
                    advantages,
                    curr_log_probs,
                    prev_log_probs,
                    gas_wo_timestep=gas_wo_t,
                    train_timesteps=train_timesteps,
                )
                loss.backward()

                avg_loss = loss.detach().clone()
                loss_mean = average_losses_across_data_parallel_group([avg_loss, ratio_detached])
                ratio_detached = loss_mean[1]
                loss_mean = loss_mean[0]
                extend_value_to_dict(
                    metrics, {
                        f"policy/loss": [loss_mean.item()],
                        f"policy/ppo_ratio": [ratio_detached.item()],
                    }
                )

            # 原来这里写的是 GAS，这不 make sense，容易造成误解，但是要求用户配置 timestep keep fraction 配置 GBS
            # 也有点难用... 所以实事求是写 GAS w/o timestep，即容易用，也不混淆。
            if (mb_i + 1) % gas_wo_t == 0:
                grad_norm = self.model.clip_grad_norm_()
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()
                extend_value_to_dict(metrics, {f"policy/grad_norm": [grad_norm.item()]})

        return metrics

    # PRIVATE FUNCS BELOW

    def _encode_prompt_with_t5(self, prompt):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        device = torch.cuda.current_device()
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.
            model_max_length,  # flux 实际上就是 512，但 oteam44 是 hubery hardcode 的 277
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder_2(text_input_ids.to(device))[0]

        if hasattr(self.text_encoder_2, "module"):
            dtype = self.text_encoder_2.module.dtype
        else:
            dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds

    def _encode_prompt_with_clip(self, prompt):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        device = torch.cuda.current_device()

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)
        if hasattr(self.text_encoder, "module"):
            dtype = self.text_encoder.module.dtype
        else:
            dtype = self.text_encoder.dtype
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds

    @override
    def sft_train_step(self, rollout_batches: List[Dict[str, Any]]):
        raise NotImplementedError()
