import math
from typing import Dict, List, Optional, Union

import torch
import torch.distributed

from gpatch_v4.core.parallel_state import cpu_barrier
from gpatch_v4.training_backend.fsdp2_backend.swap import offload_model, onload_model
from gpatch_v4.utils.common_utils import log, profile_memory_and_time


class ExtendedPipelineMixin:
    """Mixin placeholder for extended pipeline shared utilities."""
    pass


def sd3_time_shift(shift, timesteps):
    """Apply the SD3-style time shift to a schedule.

    Parameters
    ----------
    shift : float
    timesteps : torch.Tensor
        Normalized timestep schedule in [0, 1].

    Returns
    -------
    torch.Tensor
        Shifted timesteps.
    """
    return (shift * timesteps) / (1 + (shift - 1) * timesteps)


def dancegrpo_sigma_scheduler_timestep(shift, sampling_steps):
    """Compute the sigma schedule and timesteps for DanceGRPO.

    Parameters
    ----------
    shift : float
    sampling_steps : int

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(sigma_schedule, timesteps)``.
    """
    # TODO 如果有其他 diffus 的方法，做成 diffusers 的 noise scheduler。
    sigma_schedule = torch.linspace(1, 0, sampling_steps + 1)

    sigma_schedule = sd3_time_shift(shift, sigma_schedule)
    timesteps = (sigma_schedule * 1000).to(torch.long)
    return sigma_schedule, timesteps


class T2IGrpoMixin:
    """Mixin providing diffusion scheduler, noise step, log-prob
    computation, and GRPO loss for T2I tasks.
    """
    def setup_scheduler_and_timesteps(self):
        """Initialize the noise scheduler and timestep schedule."""
        shift = self.config.training.shift
        self.sampling_steps = self.config.training.sampling_steps

        # TODO: 在这里判断后面使用 flowgrpo 的还是 dancegrpo
        sigma_schedule, timestep = dancegrpo_sigma_scheduler_timestep(shift, self.sampling_steps)
        self.sigma_schedule = sigma_schedule
        self.timesteps = timestep

    def step_noise_scheduler(
        self,
        model_output: torch.Tensor,
        latents: torch.Tensor,
        sigmas: torch.Tensor,
        prev_sample: torch.Tensor,
        added_noise: torch.Tensor,
        eta: float,
        index: int,
    ):
        """Perform one denoising step and compute log probability.

        Parameters
        ----------
        model_output : torch.Tensor
            Model (velocity / noise) prediction.
        latents : torch.Tensor
            Current noisy latents.
        sigmas : torch.Tensor
            Full sigma schedule.
        prev_sample : torch.Tensor or None
            ``None`` → sample with noise.
        added_noise : torch.Tensor or None
            ``None`` → sample fresh noise.
        eta : float
            Stochastic noise strength.
        index : int
            Current timestep index into ``sigmas``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ``(prev_sample, pred_original_sample, log_prob, added_noise)``.
        """
        model_output = model_output.to(torch.float32)
        assert latents.dtype == torch.float32
        if prev_sample is not None:
            assert prev_sample.dtype == torch.float32

        sigma = sigmas[index]
        dsigma = sigmas[index + 1] - sigma
        prev_sample_mean = latents + dsigma * model_output

        pred_original_sample = latents - sigma * model_output

        delta_t = sigma - sigmas[index + 1]
        std_dev_t = eta * math.sqrt(delta_t)

        score_estimate = -(latents - pred_original_sample * (1 - sigma)) / sigma**2
        log_term = -0.5 * eta**2 * score_estimate
        prev_sample_mean = prev_sample_mean + log_term * dsigma

        if prev_sample is None:
            if added_noise is None:
                added_noise = torch.randn_like(prev_sample_mean)
            assert prev_sample_mean.shape == added_noise.shape and prev_sample_mean.dtype == added_noise.dtype
            prev_sample = prev_sample_mean + added_noise * std_dev_t

        log_prob = self.compute_log_probs(prev_sample, prev_sample_mean, std_dev_t)
        return prev_sample, pred_original_sample, log_prob, added_noise

    def compute_log_probs(
        self, prev_sample: torch.Tensor, prev_sample_mean: torch.Tensor, std_dev_t: float
    ):
        """Compute the Gaussian log probability of ``prev_sample``.

        Parameters
        ----------
        prev_sample : torch.Tensor
        prev_sample_mean : torch.Tensor
        std_dev_t : float

        Returns
        -------
        torch.Tensor
            Per-sample log probability (reduced over spatial dims).
        """
        # log prob of prev_sample given prev_sample_mean and std_dev_t
        log_prob = (
            -((prev_sample.detach().to(torch.float32) - prev_sample_mean.to(torch.float32))**2) /
            (2 * (std_dev_t**2)) - math.log(std_dev_t) -
            torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
        return log_prob

    def calculate_grpo_loss(
        self, advantages: torch.Tensor, curr_log_probs: torch.Tensor, prev_log_probs: torch.Tensor,
        gas_wo_timestep: int, train_timesteps: int
    ):
        """Calculate the clipped GRPO policy-gradient loss.

        Parameters
        ----------
        advantages : torch.Tensor
        curr_log_probs : torch.Tensor
        prev_log_probs : torch.Tensor
        gas_wo_timestep : int
            Gradient accumulation steps (without timestep dim).
        train_timesteps : int

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(loss, ratio_detached)``.
        """
        ppo_config = self.config.ppo

        # DEBUG: assuming on policy
        # prev_log_probs = curr_log_probs.detach().clone()

        advantages = torch.clamp(advantages, ppo_config.adv_clip_min, ppo_config.adv_clip_max)
        ratio = torch.exp(curr_log_probs - prev_log_probs)
        clip_ratio_low = ppo_config.ppo_clip_ratio_low if ppo_config.ppo_clip_ratio_low is not None else ppo_config.ppo_ratio_eps
        clip_ratio_high = ppo_config.ppo_clip_ratio_high if ppo_config.ppo_clip_ratio_high is not None else ppo_config.ppo_ratio_eps

        ratio_clamped = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
        loss1 = -advantages * ratio
        loss2 = -advantages * ratio_clamped

        loss = torch.mean(torch.maximum(loss1, loss2)) / (gas_wo_timestep * train_timesteps)
        return loss, ratio.detach()


class FluxLikePipelineMixin(T2IGrpoMixin):
    """Mixin for Flux-like diffusion pipelines.

    Provides latent packing / unpacking, VAE decoding, and encoder
    offload / onload utilities.
    """
    def pack_latents(self, latents, batch_size, num_channels_latents, height, width):
        """Pack spatial latents into a sequence of patches.

        Parameters
        ----------
        latents : torch.Tensor
            Shape ``(B, C, H, W)``.
        batch_size : int
        num_channels_latents : int
        height : int
        width : int

        Returns
        -------
        torch.Tensor
            Packed latents of shape ``(B, num_patches, C*4)``.
        """
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )
        return latents

    def unpack_latents(self, latents, height, width, vae_scale_factor):
        """Unpack a sequence of patches back to spatial latents.

        Parameters
        ----------
        latents : torch.Tensor
            Packed latents of shape ``(B, num_patches, C)``.
        height : int
            Target image height.
        width : int
            Target image width.
        vae_scale_factor : int
            VAE spatial compression factor.

        Returns
        -------
        torch.Tensor
            Spatial latents of shape ``(B, C', H, W)``.
        """
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def decode_to_images(self, latents, output_type='pil'):
        """Decode latents to images through the VAE.

        Parameters
        ----------
        latents : torch.Tensor
        output_type : str, optional
            ``'pil'`` / ``'pt'`` / etc.; default ``'pil'``.

        Returns
        -------
        list
            Decoded images.
        """
        height = self.config.training.height
        width = self.config.training.width

        if self.config.training.use_torch_autocast:
            # TODO dtype config
            autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()
        with autocast_ctx:
            latents = self.unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            latents = latents.to(dtype=self.vae.dtype)
            images = self.vae.decode(latents, return_dict=False)[0]
            images = self.image_processor.postprocess(images, output_type=output_type)
        return images

    def offload_encoder(self, ):
        """Offload text encoders from GPU to CPU."""
        with profile_memory_and_time(
            f"offload encoder", rank=torch.distributed.get_world_size() - 1
        ):
            offload_model(self.text_encoder)
            offload_model(self.text_encoder_2)
        cpu_barrier()

    def onload_encoder(self):
        """Onload text encoders from CPU back to GPU."""
        with profile_memory_and_time(
            f"onload encoder", rank=torch.distributed.get_world_size() - 1
        ):
            onload_model(self.text_encoder)
            onload_model(self.text_encoder_2)
        cpu_barrier()
