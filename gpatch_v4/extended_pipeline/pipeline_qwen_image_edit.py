import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import AutoencoderKLQwenImage
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen2TokenizerFast,
    Qwen2VLProcessor,
    T5ForConditionalGeneration,
)
from typing_extensions import override

from megatron.core import mpu

from gpatch_v4.core.parallel_state import cpu_barrier
from gpatch_v4.extended_pipeline.mixin import ExtendedPipelineMixin
from gpatch_v4.extended_pipeline.pipeline_base import ExtendedPipelineAbc
from gpatch_v4.training_backend import TrainingEngineFactory
from gpatch_v4.training_backend.fsdp2_backend.swap import offload_model, onload_model
from gpatch_v4.utils import (
    average_losses_across_data_parallel_group,
    extend_value_to_dict,
    get_iterator_k_split_list,
    log,
    logging_rank0,
    profile_memory_and_time,
    sync_cuda_and_get_time,
    unbind_tensor_to_list,
)

CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


def materialize_rope_freqs(model):
    """Re-create pos_freqs/neg_freqs on CPU for RoPE modules after meta-device init.

    These are complex-valued tensors stored as plain attributes (not nn.Parameter
    or register_buffer) because register_buffer drops the imaginary part. Since
    they are not in state_dict, checkpoint loading never materializes them, so we
    must do it explicitly.
    """
    for module in model.modules():
        if hasattr(module,
                   'pos_freqs') and hasattr(module,
                                            'neg_freqs') and hasattr(module, 'rope_params'):
            axes_dim = module.axes_dim
            theta = module.theta
            pos_index = torch.arange(4096)
            neg_index = torch.arange(4096).flip(0) * -1 - 1
            module.pos_freqs = torch.cat(
                [
                    module.rope_params(pos_index, axes_dim[0], theta),
                    module.rope_params(pos_index, axes_dim[1], theta),
                    module.rope_params(pos_index, axes_dim[2], theta),
                ],
                dim=1,
            )
            module.neg_freqs = torch.cat(
                [
                    module.rope_params(neg_index, axes_dim[0], theta),
                    module.rope_params(neg_index, axes_dim[1], theta),
                    module.rope_params(neg_index, axes_dim[2], theta),
                ],
                dim=1,
            )


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height


class QwenImageEditPipeline(ExtendedPipelineAbc, ExtendedPipelineMixin):
    # TODO separate edit and inpaint

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = TrainingEngineFactory.get_training_engine(
            config,
            policy_config=config.policy,
            post_meta_init_fn=materialize_rope_freqs,
        )
        self.vae = None
        self.tokenizer = None
        self.processor = None
        self.text_encoder = None

    @override
    def setup_pipeline(self):
        model_path = self.config.policy.hf_model_path
        begin_t = sync_cuda_and_get_time()

        self.vae = AutoencoderKLQwenImage.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ).to(torch.cuda.current_device())
        self.vae.requires_grad_(False)

        # TODO check padding side?
        self.tokenizer = Qwen2Tokenizer.from_pretrained(os.path.join(model_path, 'tokenizer'))
        self.processor = Qwen2VLProcessor.from_pretrained(os.path.join(model_path, 'processor'))
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            os.path.join(model_path, 'text_encoder'),
            torch_dtype=torch.bfloat16,
            device_map='cpu',
        )
        end_t = sync_cuda_and_get_time()
        log(f"loading encoder takes {end_t - begin_t:.2f} s", rank=0)

        prev_step = self.model.setup_model_and_optimizer()

        self.vae_scale_factor = 2**len(self.vae.temperal_downsample
                                      ) if getattr(self, "vae", None) else 8
        self.latent_channels = self.vae.config.z_dim if getattr(self, "vae", None) else 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        self.tokenizer_max_length = 1024
        self.prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 64
        self.default_sample_size = 128

        return prev_step

    @override
    def encode_prompt(self, **kwargs):
        raise NotImplementedError()

    @override
    def repeat_interleave_tensor_or_list(
        self,
        rb: Dict[str, Union[List[Any], torch.Tensor]],
        repeat: int,
    ):
        raise NotImplementedError()

    @override
    def permute_timesteps(
        self,
        rb: Dict[str, List[Any]],
    ):
        raise NotImplementedError()

    @torch.no_grad()
    @override
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        byt5_embeds: Optional[torch.FloatTensor] = None,
        hidden_states_mask: Optional[torch.FloatTensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        guidance_scale: float = 3.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        **kwargs,
    ):
        raise NotImplementedError('not implemented yet')

    def prepare_data_for_training(self, micro_rollout_batch: List[Dict[str, Any]]):
        batch = {
            "encoder_hidden_states": [],
            "pooled_prompt_embeds": [],
            "byt5_embeds": [],
            "hidden_states_mask": [],
            "text_ids": [],
            "image_ids": [],
            "latents": [],
            "next_latents": [],
            "log_probs": [],
            "timesteps": [],
            "advantages": [],
            "perms": [],
        }
        if self.config.training.disable_cfg_uncond_grad:
            batch["cfg_neg_preds"] = []
        if self.config.training.t2i_cfg_logps_impl_v2:
            batch["cfg_pos_latents"] = []
        batch_keys = batch.keys()
        for data in micro_rollout_batch:
            for key in batch_keys:
                batch[key].append(data[key])

        for key in batch_keys:
            batch[key] = torch.stack(batch[key]).cuda(non_blocking=True)
        return batch

    @override
    def ppo_train_step(self, rollout_batches: List[Dict[str, Any]]):
        raise NotImplementedError()

    @override
    def offload_encoder(self):
        with profile_memory_and_time(
            f"offload encoder", rank=torch.distributed.get_world_size() - 1
        ):
            offload_model(self.text_encoder, non_blocking=True, do_clear_memory=False)
            offload_model(self.vae, non_blocking=True, do_clear_memory=True)
        cpu_barrier()

    @override
    def onload_encoder(self):
        with profile_memory_and_time(
            f"onload encoder", rank=torch.distributed.get_world_size() - 1
        ):
            onload_model(self.text_encoder, non_blocking=True, do_clear_memory=False)
            onload_model(self.vae, non_blocking=True, do_clear_memory=True)
        cpu_barrier()

    # ---- helpers adapted from QwenImageEditPlusPipeline ----

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )
        return latents

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    @torch.no_grad()
    def _encode_vae_image(self, image: torch.Tensor):
        """Encode image tensor to normalized latent. image: (B, C, 1, H, W) or (B, C, H, W)."""
        if image.ndim == 4:
            image = image.unsqueeze(2)  # (B, C, 1, H, W)
        image_latents = self.vae.encode(image).latent_dist.mode()
        latents_mean = (
            torch.tensor(
                self.vae.config.latents_mean
            ).view(1, self.latent_channels, 1, 1, 1).to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(
                self.vae.config.latents_std
            ).view(1, self.latent_channels, 1, 1, 1).to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std
        return image_latents

    @torch.no_grad()
    def _get_qwen_prompt_embeds(self, prompt, image, device):
        """
        On-the-fly encode prompt + condition images into text embeddings.
        Adapted from QwenImageEditPlusPipeline._get_qwen_prompt_embeds.

        Args:
            prompt: list of str
            image: list of list of PIL.Image (per-sample cond images)
            device: torch device
        """
        dtype = self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt

        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx

        # build per-sample text with image placeholders
        txt = []
        flat_images = []
        for i, p in enumerate(prompt):
            sample_imgs = image[i]
            base_img_prompt = ""
            for j, img in enumerate(sample_imgs):
                base_img_prompt += img_prompt_template.format(j + 1)
                flat_images.append(img)
            txt.append(template.format(base_img_prompt + p))

        model_inputs = self.processor(
            text=txt,
            images=flat_images if flat_images else None,
            padding=True,
            return_tensors="pt",
        ).to(device)

        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(
            hidden_states, model_inputs.attention_mask
        )
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states
        ]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
                for u in split_hidden_states
            ]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    @torch.no_grad()
    def prepare_sft_data_from_images(
        self,
        micro_batch_data: List[Dict[str, Any]],
    ):
        """
        On-the-fly: cond_images + target_image + prompts -> all tensors needed for SFT.
        Dataset is responsible for producing cond_images (list of PIL) and target_image (PIL).
        This method is task-agnostic.
        """

        device = torch.device(torch.cuda.current_device())
        target_images = [d["target_image"] for d in micro_batch_data]
        cond_images_per_sample = [d["cond_images"] for d in micro_batch_data]  # list of list of PIL
        prompts = [d["prompt"] for d in micro_batch_data]
        batch_size = len(target_images)

        condition_images_for_encoder = []  # for text encoder (resized to CONDITION_IMAGE_SIZE)
        vae_cond_images = []  # for VAE encoding (resized to VAE_IMAGE_SIZE)
        vae_cond_sizes = []
        target_height = self.config.training.height
        target_width = self.config.training.width

        for cond_imgs in cond_images_per_sample:
            enc_imgs = []
            sample_vae_imgs = []
            sample_vae_sizes = []
            for cond_img in cond_imgs:
                cond_w, cond_h = cond_img.size

                # condition image for text encoder
                enc_w, enc_h = calculate_dimensions(CONDITION_IMAGE_SIZE, cond_w / cond_h)
                enc_imgs.append(self.image_processor.resize(cond_img, enc_h, enc_w))

                # condition image for VAE latent encoding
                vae_w, vae_h = calculate_dimensions(VAE_IMAGE_SIZE, cond_w / cond_h)
                sample_vae_sizes.append((vae_w, vae_h))
                sample_vae_imgs.append(
                    self.image_processor.preprocess(cond_img, vae_h,
                                                    vae_w).unsqueeze(2)  # (1, C, 1, H, W)
                )
            condition_images_for_encoder.append(enc_imgs)
            vae_cond_images.append(sample_vae_imgs)
            vae_cond_sizes.append(sample_vae_sizes)

        # --- text embeddings (on the fly) ---
        prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
            prompts,
            condition_images_for_encoder,
            device,
        )

        # --- target latent: encode target_image via VAE, pack ---
        lat_h = 2 * (target_height // (self.vae_scale_factor * 2))
        lat_w = 2 * (target_width // (self.vae_scale_factor * 2))

        target_pixel = self.image_processor.preprocess(
            target_images[0], target_height, target_width
        )
        for i in range(1, batch_size):
            target_pixel = torch.cat(
                [
                    target_pixel,
                    self.image_processor.preprocess(target_images[i], target_height, target_width),
                ],
                dim=0
            )
        target_pixel = target_pixel.unsqueeze(2).to(
            device=device, dtype=torch.bfloat16
        )  # (B, C, 1, H, W)
        target_latents = self._encode_vae_image(target_pixel)  # (B, C, 1, lat_h, lat_w)
        target_latents = target_latents.squeeze(2)  # (B, C, lat_h, lat_w)
        target_latents = self._pack_latents(
            target_latents,
            batch_size,
            self.latent_channels,
            lat_h,
            lat_w,
        )  # (B, S_target, D)

        # --- cond latent: encode each cond image via VAE, pack ---
        all_cond_latents = []
        for i in range(batch_size):
            sample_latents = []
            for vae_img in vae_cond_images[i]:
                vae_img = vae_img.to(device=device, dtype=torch.bfloat16)  # (1, C, 1, H, W)
                img_lat = self._encode_vae_image(vae_img)  # (1, C, 1, vae_h, vae_w)
                img_lat = img_lat.squeeze(2)
                _, _, ih, iw = img_lat.shape
                img_lat = self._pack_latents(
                    img_lat, 1, self.latent_channels, ih, iw
                )  # (1, S_cond_i, D)
                sample_latents.append(img_lat)
            # concat all cond image latents for this sample
            all_cond_latents.append(torch.cat(sample_latents, dim=1))  # (1, S_cond_total, D)
        image_latents = torch.cat(all_cond_latents, dim=0)  # (B, S_cond_total, D)

        # --- img_shapes ---
        img_shapes = []
        for i in range(batch_size):
            shapes = [(1, lat_h // 2, lat_w // 2)]
            for vae_w, vae_h in vae_cond_sizes[i]:
                shapes.append(
                    (
                        1,
                        int(vae_h) // self.vae_scale_factor // 2,
                        int(vae_w) // self.vae_scale_factor // 2,
                    )
                )
            img_shapes.append(shapes)

        # --- compute txt_seq_lens from prompt_embeds_mask ---
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

        ret = {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "target_latents": target_latents,
            "image_latents": image_latents,
            "img_shapes": img_shapes,
            "txt_seq_lens": txt_seq_lens,
        }
        return ret

    @override
    def sft_train_step(self, rollout_batches: List[Dict[str, Any]]):
        training_config = self.config.training
        gas = training_config.gradient_accumulation_steps
        num_microbatches = len(rollout_batches)
        assert num_microbatches == gas

        # --- onload encoder & vae, prepare all microbatch data, then offload ---
        self.onload_encoder()
        all_data = []
        data_iter = get_iterator_k_split_list(rollout_batches, num_microbatches)
        for mb_i in range(num_microbatches):
            micro_batch_data = next(data_iter)
            all_data.append(self.prepare_sft_data_from_images(micro_batch_data))
        self.offload_encoder()

        self.model.train_mode()
        self.model.optimizer.zero_grad()
        '''
        # 看看下面这段代码逻辑能不能对上...
        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps) 
        '''

        metrics = {}
        for mb_i in range(num_microbatches):
            data = all_data[mb_i]

            # clean latents from VAE encoding of GT image (packed)
            target_latents = data["target_latents"]  # (B, S, D)

            # sample random timestep for flow matching
            bsz = target_latents.shape[0]
            timesteps = torch.randint(0, 1000, (bsz, ), device='cuda', dtype=torch.long)
            sigmas = (timesteps / 1000).float()  # (B,)

            # add noise: noisy = (1 - sigma) * clean + sigma * noise
            noise = torch.randn_like(target_latents)
            sigmas_expanded = sigmas[:, None, None]  # (B, 1, 1)
            noisy_latents = (1.0 - sigmas_expanded
                            ) * target_latents.float() + sigmas_expanded * noise.float()

            # flow matching velocity target: v = noise - clean
            target = noise.float() - target_latents.float()

            # concat cond latents to noisy latents
            image_latents = data["image_latents"]  # (B, S_cond, D)
            latent_model_input = torch.cat(
                [noisy_latents.bfloat16(), image_latents.bfloat16()],
                dim=1,
            )  # (B, S + S_cond, D)

            if training_config.use_torch_autocast:
                autocast_ctx = torch.autocast("cuda", torch.bfloat16)
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()

            with autocast_ctx:
                _noise_pred = self.model(
                    hidden_states=latent_model_input,
                    timestep=(timesteps / 1000).bfloat16(),
                    guidance=None,
                    encoder_hidden_states_mask=data["prompt_embeds_mask"],
                    encoder_hidden_states=data["prompt_embeds"],
                    img_shapes=data["img_shapes"],
                    txt_seq_lens=data['txt_seq_lens'],
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
                # only take the noise prediction for the target latent part
                noise_pred = _noise_pred[:, :target_latents.size(1)]

            # MSE loss
            loss = torch.nn.functional.mse_loss(noise_pred.float(), target, reduction="mean")
            loss = loss / gas
            loss.backward()

            # to bypass "dead code" none grad of last layer caused by double stream attn
            for tmp_p in self.model.model.parameters():
                if tmp_p.requires_grad and tmp_p.grad is None:
                    tmp_p.grad = torch.zeros_like(tmp_p)

            avg_loss = (loss * gas).detach().clone()
            loss_mean = average_losses_across_data_parallel_group([avg_loss])
            extend_value_to_dict(metrics, {
                f"policy/loss": [loss_mean[0].item()],
            })

            if (mb_i + 1) % gas == 0:
                grad_norm = self.model.clip_grad_norm_()
                self.model.optimizer.step()
                self.model.optimizer.zero_grad()
                extend_value_to_dict(metrics, {f"policy/grad_norm": [grad_norm.item()]})

        return metrics
