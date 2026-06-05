"""
Shared inference methods for omni multimodal models (Bagel, WGOv3, …).

Both Bagel and WGOv3 mix in ``OmniInferenceMixin`` to inherit common
inference logic (text / VAE data preparation, autoregressive decoding,
flow-matching image generation with optional CFG + TaylorSeer caching).

Model-specific methods (ViT processing, chat) remain in each model class.

Requires the mixed-in class to provide the following attributes:
  language_model   – with .model.embed_tokens, .lm_head, .forward_inference
  hidden_size      – int
  config.visual_gen
  latent_patch_size, latent_channel, latent_downsample, max_latent_size
  vae2llm, llm2vae, time_embedder, latent_pos_embed
  get_flattened_position_ids(H, W, downsample, max_num_patches_per_side)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from gpatch_v4.models.bagel.modeling.bagel.cache_utils.taylorseer import cache_init
except ImportError:
    cache_init = None


class OmniInferenceMixin:
    """Shared inference logic for omni multimodal models (Bagel / WGOv3).

    Both models use MoT (Mixture of Transformers), so ``forward_inference``
    always receives ``mode`` ("und" or "gen") and gen-specific token indexes.
    """
    def _inference_device(self):
        return self.language_model.model.embed_tokens.weight.device

    def _move_to_inference_device(self, *values):
        device = self._inference_device()
        moved = []
        for value in values:
            if isinstance(value, torch.Tensor):
                moved.append(value.to(device=device))
            else:
                moved.append(value)
        return moved

    def _maybe_anchor_noise_packed_position_ids(
        self,
        packed_position_ids: Optional[torch.LongTensor],
        packed_noise_token_indexes: Optional[torch.Tensor],
        sample_lens: Optional[torch.Tensor | List[int]] = None,
    ) -> Optional[torch.LongTensor]:
        """Optionally re-anchor denoising/noise RoPE positions before text.

        When ``config.noise_1d_rope_anchor`` is set, only noise/VAE query tokens are
        remapped. Token order, KV indices, and attention masks stay unchanged.

        For a single image noise block this collapses all noise tokens to the anchor
        (e.g. ``-1``). If a sample contains multiple distinct noise positions (such
        as multi-frame/video style inputs), we preserve their relative offsets by
        shifting the whole noise span so that its last position lands on the anchor.
        """
        anchor = getattr(getattr(self, "config", None), "noise_1d_rope_anchor", None)
        if anchor is None or packed_position_ids is None or packed_noise_token_indexes is None:
            return packed_position_ids

        if packed_noise_token_indexes.dtype == torch.bool:
            packed_noise_token_indexes = torch.nonzero(packed_noise_token_indexes,
                                                       as_tuple=False).flatten()
        else:
            packed_noise_token_indexes = packed_noise_token_indexes.to(
                device=packed_position_ids.device,
                dtype=torch.long,
            )

        if packed_noise_token_indexes.numel() == 0:
            return packed_position_ids

        if sample_lens is None:
            sample_lens_list = [int(packed_position_ids.shape[0])]
        elif isinstance(sample_lens, torch.Tensor):
            sample_lens_list = [int(length) for length in sample_lens.tolist()]
        else:
            sample_lens_list = [int(length) for length in sample_lens]

        anchored_position_ids = packed_position_ids.clone()
        anchor_tensor = anchored_position_ids.new_tensor(anchor)

        start = 0
        for sample_len in sample_lens_list:
            end = start + sample_len
            sample_mask = (packed_noise_token_indexes >= start) & (packed_noise_token_indexes < end)
            if torch.any(sample_mask):
                sample_noise_indexes = packed_noise_token_indexes[sample_mask]
                sample_noise_position_ids = packed_position_ids[sample_noise_indexes]
                anchored_position_ids[sample_noise_indexes] = (
                    anchor_tensor + sample_noise_position_ids - sample_noise_position_ids.max()
                )
            start = end

        return anchored_position_ids

    # ------------------------------------------------------------------ #
    #                     Text helpers                                    #
    # ------------------------------------------------------------------ #

    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = []
        packed_text_position_ids = []
        text_token_lens = []
        packed_text_indexes = []
        packed_key_value_indexes = []

        curr = 0
        newlens, new_rope_out = [], []
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(
                range(curr_position_id, curr_position_id + len(text_ids))
            )
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope_out.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }
        return generation_input, newlens, new_rope_out

    @torch.no_grad()
    def forward_cache_update_text(
        self,
        past_key_values,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        (
            packed_text_ids,
            packed_text_position_ids,
            text_token_lens,
            packed_text_indexes,
            packed_key_value_indexes,
            key_values_lens,
        ) = self._move_to_inference_device(
            packed_text_ids,
            packed_text_position_ids,
            text_token_lens,
            packed_text_indexes,
            packed_key_value_indexes,
            key_values_lens,
        )
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            mode="und",
        )
        return output.past_key_values

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = [], []
        packed_query_position_ids = []

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids['bos_token_id'])
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }
        return generation_input

    @torch.no_grad()
    def generate_text(
        self,
        past_key_values,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        (
            packed_key_value_indexes,
            key_values_lens,
            packed_start_tokens,
            packed_query_position_ids,
        ) = self._move_to_inference_device(
            packed_key_value_indexes,
            key_values_lens,
            packed_start_tokens,
            packed_query_position_ids,
        )
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0,
                len(key_values_lens),
                device=key_values_lens.device,
                dtype=key_values_lens.dtype,
            )

            unpacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(unpacked)):
                unpacked[i] += i
            packed_key_value_indexes = torch.cat(unpacked, dim=0)

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                mode="und",
            )
            past_key_values = output.past_key_values
            pred_logits = self.language_model.lm_head(output.packed_query_sequence)

            if do_sample:
                probs = F.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            unpacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(unpacked)):
                unpacked[i] = torch.cat(
                    [unpacked[i],
                     torch.tensor([unpacked[i][-1] + 1], device=unpacked[i].device)],
                    dim=0,
                )
            packed_key_value_indexes = torch.cat(unpacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id:
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    # ------------------------------------------------------------------ #
    #                     VAE helpers                                     #
    # ------------------------------------------------------------------ #

    def prepare_vae_images(
        self,
        curr_kvlens,
        curr_rope,
        images,
        transforms,
        new_token_ids,
        timestep=0,
    ):
        patchified_vae_latent_shapes, packed_vae_position_ids = [], []
        packed_vae_token_indexes = []
        packed_text_ids, packed_text_indexes = [], []
        packed_seqlens, packed_position_ids, packed_indexes = [], [], []
        packed_key_value_indexes = []

        _curr = curr = 0
        vae_image_tensors = []
        newlens, new_rope_out = [], []
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1),
                image_tensor.size(2),
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size,
            )
            packed_vae_position_ids.append(vae_position_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope_out.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }
        return generation_input, newlens, new_rope_out

    @torch.no_grad()
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        (
            packed_vae_position_ids,
            packed_timesteps,
            packed_vae_token_indexes,
            packed_text_ids,
            packed_text_indexes,
            packed_position_ids,
            packed_seqlens,
            packed_indexes,
            key_values_lens,
            packed_key_value_indexes,
        ) = self._move_to_inference_device(
            packed_vae_position_ids,
            packed_timesteps,
            packed_vae_token_indexes,
            packed_text_ids,
            packed_text_indexes,
            packed_position_ids,
            packed_seqlens,
            packed_indexes,
            key_values_lens,
            packed_key_value_indexes,
        )
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        vae_param = next(vae_model.parameters())
        padded_images = padded_images.to(device=vae_param.device, dtype=vae_param.dtype)
        padded_latent = vae_model.encode(padded_images)

        p = self.latent_patch_size
        packed_latent = []
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(
                -1,
                p * p * self.latent_channel,
            )
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        # Some VAE backends still emit fp32 latents under bf16 inference, so align
        # all VAE-bridge inputs to the bridge layer dtype before the linear op.
        vae2llm_dtype = self.vae2llm.weight.dtype
        packed_latent = packed_latent.to(dtype=vae2llm_dtype)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids).to(dtype=vae2llm_dtype)
        packed_timestep_embeds = self.time_embedder(packed_timesteps).to(dtype=vae2llm_dtype)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype or packed_latent.device != packed_sequence.device:
            packed_latent = packed_latent.to(
                device=packed_sequence.device, dtype=packed_sequence.dtype
            )
        packed_sequence[packed_vae_token_indexes] = packed_latent

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            mode="gen",
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        return output.past_key_values

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids):
        packed_text_ids, packed_text_indexes = [], []
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = [], [], []
        packed_position_ids, packed_seqlens, packed_indexes = [], [], []
        packed_key_value_indexes = []

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_position_ids = self.get_flattened_position_ids(
                H,
                W,
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size,
            )
            packed_vae_position_ids.append(vae_position_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_init_noises.append(
                torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size**2)
            )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }
        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = [], [], []

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids":
                torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens":
                torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes":
                torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes":
                torch.tensor(
                    packed_key_value_indexes,
                    dtype=torch.long,
                ),
        }
        return generation_input

    # ------------------------------------------------------------------ #
    #           Image generation (flow matching + optional TaylorSeer)   #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = None,
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values=None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values=None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # TaylorSeer (Bagel only; ignored when False / cache_init unavailable)
        enable_taylorseer: bool = False,
        show_progress: bool = True,
    ):
        (
            packed_text_ids,
            packed_text_indexes,
            packed_init_noises,
            packed_vae_position_ids,
            packed_vae_token_indexes,
            packed_seqlens,
            packed_position_ids,
            packed_indexes,
            key_values_lens,
            packed_key_value_indexes,
            cfg_text_packed_query_indexes,
            cfg_text_packed_position_ids,
            cfg_text_key_values_lens,
            cfg_text_packed_key_value_indexes,
            cfg_img_packed_query_indexes,
            cfg_img_packed_position_ids,
            cfg_img_key_values_lens,
            cfg_img_packed_key_value_indexes,
        ) = self._move_to_inference_device(
            packed_text_ids,
            packed_text_indexes,
            packed_init_noises,
            packed_vae_position_ids,
            packed_vae_token_indexes,
            packed_seqlens,
            packed_position_ids,
            packed_indexes,
            key_values_lens,
            packed_key_value_indexes,
            cfg_text_packed_query_indexes,
            cfg_text_packed_position_ids,
            cfg_text_key_values_lens,
            cfg_text_packed_key_value_indexes,
            cfg_img_packed_query_indexes,
            cfg_img_packed_position_ids,
            cfg_img_key_values_lens,
            cfg_img_packed_key_value_indexes,
        )
        if cfg_interval is None:
            cfg_interval = [0, 1]

        if enable_taylorseer:
            assert cache_init is not None, "TaylorSeer requires cache_utils"
            self.language_model.model.enable_taylorseer = True
            model_pred_cache_dic, model_pred_current = cache_init(self, num_timesteps)
            model_pred_text_cache_dic, model_pred_text_current = cache_init(self, num_timesteps)
            model_pred_img_cache_dic, model_pred_img_current = cache_init(self, num_timesteps)
        else:
            self.language_model.model.enable_taylorseer = False
            model_pred_cache_dic = model_pred_current = None
            model_pred_text_cache_dic = model_pred_text_current = None
            model_pred_img_cache_dic = model_pred_img_current = None

        x_t = packed_init_noises

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), disable=not show_progress):
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
                model_pred_cache_dic=model_pred_cache_dic,
                model_pred_current=model_pred_current,
                model_pred_text_cache_dic=model_pred_text_cache_dic,
                model_pred_text_current=model_pred_text_current,
                model_pred_img_cache_dic=model_pred_img_cache_dic,
                model_pred_img_current=model_pred_img_current,
            )
            x_t = x_t - v_t.to(x_t.device) * dts[i]

        if enable_taylorseer:
            del model_pred_cache_dic, model_pred_current
            del model_pred_text_cache_dic, model_pred_text_current
            del model_pred_img_cache_dic, model_pred_img_current

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    @torch.no_grad()
    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values=None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values=None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # TaylorSeer caches (optional)
        model_pred_cache_dic=None,
        model_pred_current=None,
        model_pred_text_cache_dic=None,
        model_pred_text_current=None,
        model_pred_img_cache_dic=None,
        model_pred_img_current=None,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        vae2llm_dtype = self.vae2llm.weight.dtype
        x_t_for_bridge = x_t.to(dtype=vae2llm_dtype)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids).to(dtype=vae2llm_dtype)
        packed_timestep_embeds = self.time_embedder(timestep).to(dtype=vae2llm_dtype)
        x_t_embedded = self.vae2llm(x_t_for_bridge) + packed_timestep_embeds + packed_pos_embed
        if x_t_embedded.dtype != packed_sequence.dtype or x_t_embedded.device != packed_sequence.device:
            x_t_embedded = x_t_embedded.to(
                device=packed_sequence.device, dtype=packed_sequence.dtype
            )
        packed_sequence[packed_vae_token_indexes] = x_t_embedded

        packed_position_ids = self._maybe_anchor_noise_packed_position_ids(
            packed_position_ids,
            packed_vae_token_indexes,
            sample_lens=packed_seqlens,
        )
        cfg_text_packed_position_ids = self._maybe_anchor_noise_packed_position_ids(
            cfg_text_packed_position_ids,
            packed_vae_token_indexes,
            sample_lens=packed_seqlens,
        )
        cfg_img_packed_position_ids = self._maybe_anchor_noise_packed_position_ids(
            cfg_img_packed_position_ids,
            packed_vae_token_indexes,
            sample_lens=packed_seqlens,
        )

        _ts = getattr(self.language_model.model, 'enable_taylorseer', False)

        if _ts and model_pred_cache_dic is not None:
            self.language_model.model.cache_dic = model_pred_cache_dic
            self.language_model.model.current = model_pred_current

        gen_fwd_kwargs = dict(
            mode="gen",
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **gen_fwd_kwargs,
        )
        v_t = self.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]

        if cfg_text_scale > 1.0:
            if _ts and model_pred_text_cache_dic is not None:
                self.language_model.model.cache_dic = model_pred_text_cache_dic
                self.language_model.model.current = model_pred_text_current
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **gen_fwd_kwargs,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            if _ts and model_pred_img_cache_dic is not None:
                self.language_model.model.cache_dic = model_pred_img_cache_dic
                self.language_model.model.current = model_pred_img_current
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **gen_fwd_kwargs,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        # CFG renormalization
        if cfg_text_scale > 1.0:
            if cfg_renorm_type == "text_channel":
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(
                    min=cfg_renorm_min,
                    max=1.0,
                )
                v_t_text = v_t_text_ * scale
                if cfg_img_scale > 1.0:
                    v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                else:
                    v_t = v_t_text
            else:
                v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                if cfg_img_scale > 1.0:
                    v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                else:
                    v_t_ = v_t_text_

                if cfg_renorm_type == "global":
                    norm_v_t = torch.norm(v_t)
                    norm_v_t_ = torch.norm(v_t_)
                elif cfg_renorm_type == "channel":
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                else:
                    raise NotImplementedError(f"{cfg_renorm_type} is not supported")
                scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(
                    min=cfg_renorm_min,
                    max=1.0,
                )
                v_t = v_t_ * scale

        return v_t
