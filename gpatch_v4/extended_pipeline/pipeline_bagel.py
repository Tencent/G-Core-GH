import gc
import math
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from typing_extensions import override

from gpatch_v4.core.device import get_device_backend_name, get_device_module
from gpatch_v4.extended_pipeline.mixin import T2IGrpoMixin
from gpatch_v4.extended_pipeline.pipeline_fsdp2_omni_base import FSDP2EngineBase
from gpatch_v4.models.bagel.data.data_utils import (
    add_special_tokens,
    get_flattened_position_ids_extrapolate,
    get_flattened_position_ids_interpolate,
)
from gpatch_v4.models.bagel.modeling.autoencoder import load_ae
from gpatch_v4.models.bagel.modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from gpatch_v4.models.bagel.modeling.bagel.qwen2 import Qwen2Tokenizer
from gpatch_v4.training_backend.common.omni_training_utils import count_parameters
from gpatch_v4.training_backend.fsdp2_backend.fsdp2_utils_bagel import (
    FSDPCheckpoint,
    fsdp_ema_setup,
    fsdp_wrapper,
    grad_checkpoint_check_fn,
)
from gpatch_v4.utils import (
    extend_value_to_dict,
    get_iterator_k_split_list,
    logging_rank0,
    repeat_interleave_tensor_or_list,
    unbind_tensor_to_list,
)


class FSDP2EngineBagel(FSDP2EngineBase, T2IGrpoMixin):
    def build_model_and_optimizer(self):
        """Build the Bagel model engine."""

        training_args = self.training_args
        model_args = self.model_args
        logger = self.logger

        resume_from, resume_model_only, finetune_from_ema = self._resolve_resume()

        if training_args.finetune_from_hf:
            llm_config = Qwen2Config.from_json_file(
                os.path.join(model_args.model_path, "llm_config.json")
            )
        else:
            llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
        llm_config.layer_module = model_args.layer_module
        llm_config.qk_norm = model_args.llm_qk_norm
        llm_config.tie_word_embeddings = model_args.tie_word_embeddings
        llm_config.freeze_und = training_args.freeze_und
        llm_config.use_flash_mask = training_args.use_flash_mask

        if training_args.finetune_from_hf:
            language_model = Qwen2ForCausalLM(llm_config)
        else:
            language_model = Qwen2ForCausalLM.from_pretrained(
                model_args.llm_path, config=llm_config
            )

            # Initialize ONLY qk_norm layers (new layers not in checkpoint)
            # Must use named_modules() to check names, cannot use apply()
            logging_rank0("Initializing qk_norm layers (not in checkpoint)...")
            for name, module in language_model.named_modules():
                if 'q_norm' in name or 'k_norm' in name:
                    if hasattr(module, 'weight'):
                        # 直接初始化为全1（RMSNorm 的标准初始化）
                        with torch.no_grad():
                            module.weight.fill_(1.0)
                        logging_rank0(f"  Initialized {name} to ones")

            # Handle lm_head for tie_word_embeddings mismatch
            if not llm_config.tie_word_embeddings:
                logging_rank0("Copying embed_tokens.weight to lm_head.weight")
                with torch.no_grad():
                    language_model.lm_head.weight.copy_(language_model.model.embed_tokens.weight)
        if training_args.copy_init_moe:
            language_model.init_moe()

        vit_config = None
        vit_model = None
        if training_args.visual_und:
            if training_args.finetune_from_hf:
                vit_config = SiglipVisionConfig.from_json_file(
                    os.path.join(model_args.model_path, "vit_config.json")
                )
            else:
                vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
            vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
            vit_config.rope = model_args.vit_rope
            if training_args.finetune_from_hf:
                vit_model = SiglipVisionModel(vit_config)
            else:
                vit_model = SiglipVisionModel.from_pretrained(
                    model_args.vit_path, config=vit_config
                )

        vae_config = None
        vae_model = None
        if training_args.visual_gen:
            vae_model, vae_config = load_ae(
                local_path=os.path.join(model_args.model_path, "ae.safetensors") if training_args.
                finetune_from_hf else model_args.vae_path
            )
            vae_model.to(get_device_module().current_device())

        config = BagelConfig(
            visual_gen=training_args.visual_gen,
            visual_und=training_args.visual_und,
            llm_config=llm_config,
            vit_config=vit_config if training_args.visual_und else None,
            vae_config=vae_config if training_args.visual_gen else None,
            latent_patch_size=model_args.latent_patch_size,
            max_latent_size=model_args.max_latent_size,
            vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
            connector_act=model_args.connector_act,
            interpolate_pos=model_args.interpolate_pos,
            timestep_shift=training_args.timestep_shift,
            vae_mask=training_args.vae_mask,
        )
        model = Bagel(language_model, vit_model if training_args.visual_und else None, config)
        if training_args.visual_und:
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

        total_param_count = count_parameters(model)
        lm_param_count = count_parameters(model.language_model)
        logger.info(
            f"Model parameter count: {total_param_count / 1e9:.2f}B (LM-only: {lm_param_count / 1e9:.2f}B)"
        )

        # Setup tokenizer for model:
        tokenizer = Qwen2Tokenizer.from_pretrained(
            model_args.model_path if training_args.finetune_from_hf else model_args.llm_path
        )
        tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
        # 只在 tokenizer 真的比模型 vocab 大时才扩展 embedding。
        # num_new_tokens > 0 不能作为判断条件：add_special_tokens 会把已存在于
        # added_tokens 的 token 注册为 special token 并返回数量（如 3），
        # 但 len(tokenizer) 不变（仍是 151665 < model vocab_size 151936），
        # 用 num_new_tokens > 0 会错误地触发 resize_token_embeddings(151665)，
        # 把 embed 从 151936 截断到 151665。
        current_vocab_size = model.language_model.config.vocab_size
        if len(tokenizer) > current_vocab_size:
            model.language_model.resize_token_embeddings(len(tokenizer))
            model.config.llm_config.vocab_size = len(tokenizer)
            model.language_model.config.vocab_size = len(tokenizer)

        # --- Freeze, FSDP, optimizer (shared logic from base class) ---
        self._freeze_modules(model, vae_model, vit_model)

        fsdp_model, ema_model = self._setup_fsdp(
            model,
            fsdp_wrapper,
            grad_checkpoint_check_fn,
            fsdp_ema_setup,
            FSDPCheckpoint,
            resume_from,
            finetune_from_ema,
        )

        optimizer, scheduler = self._build_optimizer_scheduler(fsdp_model)

        self.vit_config = vit_config
        self.vit_model = vit_model
        self.vae_config = vae_config
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids

        self._store_state(
            fsdp_model,
            ema_model,
            optimizer,
            scheduler,
            language_model.config,
            resume_from,
            resume_model_only,
            FSDPCheckpoint,
        )

    @override
    def forward_backward_step(self, data, loss_scale=None):
        training_args = self.training_args
        device = get_device_module().current_device()

        ce_loss_weights = data.pop('ce_loss_weights', None)
        with torch.amp.autocast(get_device_backend_name(), enabled=True, dtype=torch.bfloat16):
            if training_args.visual_gen and 'padded_images' in data:
                with torch.no_grad():
                    data['padded_latent'] = self.vae_model.encode(data.pop('padded_images'))
            loss_dict = self.fsdp_model(**data)
            # model_preds is for rl
            loss_dict.pop("model_preds", None)

        loss = 0
        ce = loss_dict["ce"]
        mse = loss_dict["mse"]
        total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device) \
            if ce is not None else torch.tensor(0, device=device)
        total_mse_tokens = torch.tensor(len(data['mse_loss_indexes']), device=device) \
            if mse is not None else torch.tensor(0, device=device)
        dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)

        if ce is not None:

            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            # assert not training_args.visual_und
            loss_dict["ce"] = torch.tensor(0.0, device=device)

        if mse is not None:
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
            loss_dict["mse"] = mse.detach()
            loss = loss + mse * training_args.mse_weight
        else:
            loss_dict["mse"] = torch.tensor(0.0, device=device)

        if loss_scale:
            loss = loss * loss_scale
        loss.backward()
        return loss.detach(), loss_dict, total_mse_tokens, total_ce_tokens

    # optimize_step, save_ckpt, set_grad_sync_flag: inherited from FSDP2EngineBase

    @override
    def setup_pipeline(self):
        from gpatch_v4.training_backend.common.omni_training_utils import LoggerAdaptor
        self.set_logger(LoggerAdaptor())
        self.build_model_and_optimizer()
        if hasattr(
            self.config.training, "guidance_scale"
        ) and self.config.training.guidance_scale > 1:
            self.uncond_feats = self.encode_prompt("")
        return self.train_step

    @override
    def encode_prompt(self, prompt: Union[str, List[str]] = None, **kwargs):

        device = get_device_module().current_device()
        if not isinstance(prompt, list):
            assert isinstance(prompt, str)
            prompt = [prompt]

        sample_lens = []
        packed_text_ids = []
        packed_text_indexes = []
        packed_vae_token_indexes = []
        packed_position_ids = []
        split_lens_list = []
        attn_modes_list = []

        for p in prompt:
            split_lens = []
            attn_modes = []
            cur_index = 0
            cur_rope = 0
            text_ids = self.tokenizer.encode(p)
            text_len = len(text_ids)
            # we assume empty promt is cfg uncond_feats
            if text_len > 0:
                text_ids = [self.new_token_ids["bos_token_id"]
                           ] + text_ids + [self.new_token_ids['eos_token_id']]
                text_len = text_len + 2
                split_lens.append(text_len)
                attn_modes.append("causal")

            text_index = list(range(len(text_ids)))
            position_ids = list(range(len(text_ids)))
            cur_index += len(text_ids)
            cur_rope += len(text_ids)

            # start of image
            text_ids.append(self.new_token_ids['start_of_image'])
            text_index.append(cur_index)
            cur_index += 1

            H, W = (self.config.training.height, self.config.training.width)
            h = H // self.fsdp_model.latent_downsample
            w = W // self.fsdp_model.latent_downsample
            num_img_tokens = w * h
            vae_token_indexes = list(range(cur_index, cur_index + num_img_tokens))

            cur_index += num_img_tokens
            text_ids.append(self.new_token_ids['end_of_image'])
            text_index.append(cur_index)
            position_ids.extend([cur_rope] * (num_img_tokens + 2))

            sample_lens.append(text_len + num_img_tokens + 2)
            packed_text_ids.append(torch.tensor(text_ids, dtype=torch.long))
            packed_text_indexes.append(torch.tensor(text_index, dtype=torch.long))
            packed_vae_token_indexes.append(torch.tensor(vae_token_indexes, dtype=torch.long))
            packed_position_ids.append(torch.tensor(position_ids, dtype=torch.long))
            split_lens.append(num_img_tokens + 2)
            attn_modes.append("noise")
            split_lens_list.append(split_lens)
            attn_modes_list.append(attn_modes)

        return dict(
            sample_lens=sample_lens,
            packed_text_ids=packed_text_ids,
            packed_text_indexes=packed_text_indexes,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_position_ids=packed_position_ids,
            split_lens=split_lens_list,
            attn_modes=attn_modes_list,
        )

    @override
    def repeat_interleave_tensor_or_list(
        self,
        rb: Dict[str, Union[List[Any], torch.Tensor]],
        repeat: int,
    ):
        for (k, v) in rb.items():
            rb[k] = repeat_interleave_tensor_or_list(v, repeat)
        return rb

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
        # 只对有 sampling_step 的 tensor 做 randperm
        for key in perm_keys:
            assert key in rb, f'{key} not in {rb.keys()}'
            rb[key] = [rb[key][i][perms[i]] for i in range(expected_len)]
        rb["perms"] = perms

    def prepare_data_for_training(self, micro_rollout_batch: List[Dict[str, Any]]):
        batch = {
            "sample_lens": [],
            "packed_text_ids": [],
            "packed_text_indexes": [],
            "packed_vae_token_indexes": [],
            "packed_position_ids": [],
            "split_lens": [],
            "attn_modes": [],
            "latent_position_ids": [],
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

        stack_keys = [
            "latent_position_ids", "latents", "next_latents", "log_probs", "timesteps",
            "advantages", "perms"
        ]

        for key in stack_keys:
            batch[key] = torch.stack(batch[key]).cuda(non_blocking=True)
        return batch

    def pack_latents(self, latents, batch_size, latent_channel, height, width):
        latent_patch_size = self.model_args.latent_patch_size
        latents = latents.reshape(
            batch_size, latent_channel, height // latent_patch_size, latent_patch_size,
            width // latent_patch_size, latent_patch_size
        )
        latents = torch.einsum("bchpwq->bhwpqc", latents).reshape(
            batch_size, -1, latent_patch_size * latent_patch_size * latent_channel
        )
        return latents

    def unpack_latents(
        self,
        latents,
        height,
        width,
    ):
        latent_channel = self.fsdp_model.latent_channel
        latent_patch_size = self.model_args.latent_patch_size
        down_sample = self.vae_config.downsample
        height = (int(height) // down_sample)
        width = (int(width) // down_sample)

        latents = latents.reshape(
            -1, height // latent_patch_size, width // latent_patch_size, latent_patch_size,
            latent_patch_size, latent_channel
        )
        latents = torch.einsum("bhwpqc->bchpwq", latents)
        latents = latents.reshape(-1, latent_channel, height, width)
        return latents

    def prepare_packed_latent_position_ids(self, batch_size, height, width, device):
        vae_image_downsample = self.model_args.latent_patch_size * self.vae_config.downsample
        get_flattened_position_ids = get_flattened_position_ids_interpolate if self.model_args.interpolate_pos else get_flattened_position_ids_extrapolate
        return get_flattened_position_ids(
            height, width, vae_image_downsample, self.model_args.max_latent_size
        ).to(device, non_blocking=True).view([1, -1]).repeat_interleave(batch_size, dim=0)

    def prepare_latents(
        self,
        batch_size,
        height,
        width,
        device,
        generator=None,
        latents=None,
    ):
        img_h = height
        img_w = width

        latent_channel = self.fsdp_model.latent_channel
        down_sample = self.vae_config.downsample
        latent_patch_size = self.model_args.latent_patch_size

        height = (int(height) // down_sample)
        width = (int(width) // down_sample)

        shape = (batch_size, latent_channel, height, width)
        repeat_n = self.config.training.sampling_repeat_n
        assert batch_size % repeat_n == 0
        assert latents is None

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # latent 统一用 fp32
        # TODO reuse this piece
        if self.config.training.init_same_noise:
            latents = []
            for tmp_i in range(batch_size // repeat_n):
                tmp = randn_tensor(
                    (latent_channel, height, width), device=device, dtype=torch.float32
                )
                for tmp_j in range(repeat_n):
                    latents.append(tmp)
            latents = torch.stack(latents)
            assert latents.shape == shape
        else:
            latents = randn_tensor(shape, device=device, dtype=torch.float32)
        latents = latents.reshape(
            batch_size, latent_channel, height // latent_patch_size, latent_patch_size,
            width // latent_patch_size, latent_patch_size
        )
        latents = torch.einsum("bchpwq->bhwpqc", latents).reshape(
            -1, latent_patch_size * latent_patch_size * latent_channel
        )
        packed_latent_position_ids = self.prepare_packed_latent_position_ids(
            batch_size, img_h, img_w, device
        )
        latents = self.pack_latents(latents, batch_size, latent_channel, height, width)
        return latents, packed_latent_position_ids

    def decode_to_images(self, latents, output_type='pil'):
        height = self.config.training.height
        width = self.config.training.width
        latents = self.unpack_latents(latents, height, width)
        images = self.vae_model.decode(latents)
        images = (images * 0.5 + 0.5).clamp(0, 1).permute(0, 2, 3, 1) * 255
        images = (images).to(torch.uint8).cpu()
        images = unbind_tensor_to_list(images)
        images = [Image.fromarray(e.numpy()) for e in images]
        return images

    def pack_batch(
        self,
        sample_lens: List[int] = None,
        packed_text_ids: List[torch.LongTensor] = None,
        packed_text_indexes: List[torch.LongTensor] = None,
        packed_vae_token_indexes: List[torch.LongTensor] = None,
        packed_position_ids: List[torch.LongTensor] = None,
        split_lens: List[List[int]] = None,
        attn_modes: List[List[str]] = None,
    ):
        split_lens_list = []
        attn_modes_list = []
        packed_text_indexes_list = []
        packed_vae_token_indexes_list = []
        l_acc = 0
        for (i, l) in enumerate(sample_lens):
            packed_text_indexes_list.append(packed_text_indexes[i] + l_acc)
            packed_vae_token_indexes_list.append(packed_vae_token_indexes[i] + l_acc)
            split_lens_list.extend(split_lens[i])
            attn_modes_list.extend(attn_modes[i])
            l_acc += l

        packed_text_ids = torch.cat(packed_text_ids, dim=0)
        packed_text_indexes = torch.cat(packed_text_indexes_list, dim=0)
        packed_vae_token_indexes = torch.cat(packed_vae_token_indexes_list, dim=0)
        packed_position_ids = torch.cat(packed_position_ids, dim=0)

        return sample_lens, packed_text_ids, packed_text_indexes, packed_vae_token_indexes, packed_position_ids, split_lens_list, attn_modes_list

    def merge_uncond_predict(self, model_pred_uncond, model_pred):
        """Align with the logic in bagel _forward_flow."""
        cfg_guidance_scale = self.config.training.guidance_scale
        # b, s, d
        noise_pred = model_pred_uncond + cfg_guidance_scale * (model_pred - model_pred_uncond)
        assert noise_pred.ndim == 3
        if self.config.training.cfg_renorm_type == "global":
            norm_model_pred = torch.norm(model_pred, dim=(1, 2), keepdim=True)
            norm_noise_pred = torch.norm(noise_pred, dim=(1, 2), keepdim=True)
        elif self.config.training.cfg_renorm_type == "channel":
            norm_model_pred = torch.norm(model_pred, dim=-1, keepdim=True)
            norm_noise_pred = torch.norm(noise_pred, dim=-1, keepdim=True)
        else:
            assert False, f"not supported type {self.config.training.cfg_renorm_type}"

        scale = (norm_model_pred /
                 (norm_noise_pred + 1e-8)).clamp(min=self.config.training.cfg_renorm_min, max=1.0)
        noise_pred = noise_pred * scale
        return noise_pred

    @override
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        sample_lens: List[int] = None,
        packed_text_ids: List[torch.LongTensor] = None,
        packed_text_indexes: List[torch.LongTensor] = None,
        packed_vae_token_indexes: List[torch.LongTensor] = None,
        packed_position_ids: List[torch.LongTensor] = None,
        split_lens: List[List[int]] = None,
        attn_modes: List[List[str]] = None,
        guidance_scale: float = 3.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        **kwargs,
    ):
        # rollout 需使用训练模式的 forward_train 接口（即便 no_grad）
        self.fsdp_model.train()

        batch_size = len(sample_lens)
        device = torch.device(get_device_module().current_device())
        repeat_n = self.config.training.sampling_repeat_n
        model_mbs = self.config.training.rollout_model_mbs
        rollout_mbs = self.config.training.rollout_mbs
        if model_mbs is None:
            model_mbs = rollout_mbs * repeat_n

        latents, latent_position_ids = self.prepare_latents(
            batch_size,
            self.config.training.height,
            self.config.training.width,
            device=device,
            generator=generator,
        )

        self.setup_scheduler_and_timesteps()

        all_latents = [latents]
        all_log_probs = []
        assert latents.ndim == 3
        # self.model.eval()
        for si in range(self.config.training.sampling_steps):
            timestep_value = self.timesteps[si]
            # b, num tokens
            timesteps = torch.full(
                [latents.shape[0], latents.shape[1]],
                timestep_value,
                device=latents.device,
                dtype=torch.long
            )
            assert latents.dtype == torch.float32

            if self.config.training.use_torch_autocast:
                # TODO dtype config
                autocast_ctx = torch.autocast("cuda", torch.bfloat16)
            else:
                from contextlib import nullcontext
                autocast_ctx = nullcontext()

            # 由于 train 的时候，pos / neg prompt
            # 一起计算，为了对齐，这里也要一起计算，写起来有点难受了。我高度怀疑 clip range
            # 改一下就好了。
            neg_noise_pred = []
            pos_noise_pred = []
            noise_pred = []

            assert rollout_mbs * repeat_n % model_mbs == 0
            for model_mbi in range(rollout_mbs * repeat_n // model_mbs):
                start_idx = model_mbi * model_mbs
                end_idx = (model_mbi + 1) * model_mbs

                if self.config.training.guidance_scale > 1:
                    _latents = latents[start_idx:end_idx].repeat(2, 1, 1)
                    _latent_position_ids = latent_position_ids[start_idx:end_idx].repeat(2, 1).view(
                        [-1]
                    )
                    _timesteps = timesteps[start_idx:end_idx].repeat(2, 1).view([-1])
                    tmp = repeat_interleave_tensor_or_list(
                        self.uncond_feats["sample_lens"], model_mbs
                    )
                    _sample_lens = tmp + sample_lens[start_idx:end_idx]
                    tmp = repeat_interleave_tensor_or_list(
                        self.uncond_feats["packed_text_ids"], model_mbs
                    )
                    _packed_text_ids = tmp + packed_text_ids[start_idx:end_idx]
                    tmp = repeat_interleave_tensor_or_list(
                        self.uncond_feats["packed_text_indexes"], model_mbs
                    )
                    _packed_text_indexes = tmp + packed_text_indexes[start_idx:end_idx]
                    tmp = repeat_interleave_tensor_or_list(
                        self.uncond_feats["packed_vae_token_indexes"], model_mbs
                    )
                    _packed_vae_token_indexes = tmp + packed_vae_token_indexes[start_idx:end_idx]
                    tmp = repeat_interleave_tensor_or_list(
                        self.uncond_feats["packed_position_ids"], model_mbs
                    )
                    _packed_position_ids = tmp + packed_position_ids[start_idx:end_idx]
                    tmp = repeat_interleave_tensor_or_list(
                        self.uncond_feats["split_lens"], model_mbs
                    )
                    _split_lens = tmp + split_lens[start_idx:end_idx]
                    tmp = repeat_interleave_tensor_or_list(
                        self.uncond_feats["attn_modes"], model_mbs
                    )
                    _attn_modes = tmp + attn_modes[start_idx:end_idx]
                else:
                    _latents = latents[start_idx:end_idx]
                    _latent_position_ids = latent_position_ids[start_idx:end_idx].view([-1])
                    _timesteps = timesteps[start_idx:end_idx].view([-1])
                    _sample_lens = sample_lens[start_idx:end_idx]
                    _packed_text_ids = packed_text_ids[start_idx:end_idx]
                    _packed_text_indexes = packed_text_indexes[start_idx:end_idx]
                    _packed_vae_token_indexes = packed_vae_token_indexes[start_idx:end_idx]
                    _packed_position_ids = packed_position_ids[start_idx:end_idx]
                    _split_lens = split_lens[start_idx:end_idx]
                    _attn_modes = attn_modes[start_idx:end_idx]

                (
                    _sample_lens, _packed_text_ids, _packed_text_indexes, _packed_vae_token_indexes,
                    _packed_position_ids, _split_lens, _attn_modes
                ) = self.pack_batch(
                    _sample_lens, _packed_text_ids, _packed_text_indexes, _packed_vae_token_indexes,
                    _packed_position_ids, _split_lens, _attn_modes
                )
                with autocast_ctx:
                    _noise_pred = self.fsdp_model(
                        sequence_length=sum(_sample_lens),
                        packed_text_ids=_packed_text_ids,
                        packed_text_indexes=_packed_text_indexes,
                        sample_lens=_sample_lens,
                        packed_position_ids=_packed_position_ids,
                        split_lens=_split_lens,
                        attn_modes=_attn_modes,
                        packed_latent_position_ids=_latent_position_ids,
                        packed_vae_token_indexes=_packed_vae_token_indexes,
                        packed_timesteps=(_timesteps / 1000).bfloat16(),
                        mse_loss_indexes=_packed_vae_token_indexes,
                        packed_latent=_latents,
                    )["model_preds"]
                    _noise_pred = _noise_pred.reshape(*(_latents.shape))

                # don't forget about cfg here
                if self.config.training.guidance_scale > 1:
                    _neg_noise_pred, _pos_noise_pred = _noise_pred.chunk(2)
                    neg_noise_pred.append(_neg_noise_pred)
                    pos_noise_pred.append(_pos_noise_pred)
                else:
                    noise_pred.append(_noise_pred)

            if self.config.training.guidance_scale > 1:
                noise_pred = neg_noise_pred + pos_noise_pred
            noise_pred = torch.cat(noise_pred, dim=0)

            if self.config.training.guidance_scale > 1:
                model_pred_uncond, model_pred_text = noise_pred.chunk(2)
                noise_pred = self.merge_uncond_predict(model_pred_uncond, model_pred_text)

            assert noise_pred.dtype == torch.bfloat16
            assert latents.dtype == torch.float32

            assert rollout_mbs * repeat_n % model_mbs == 0
            tmp_latents = []
            ode_latents = []
            log_prob = []
            for model_mbi in range(rollout_mbs * repeat_n // model_mbs):
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
        images = self.decode_to_images(ode_latents)
        # decode to images
        # self.vae.enable_tiling()
        # change batch into list and return List[Dict[str, List[Any]]]
        batch_size = latents.shape[0]
        batch_latents_lst = unbind_tensor_to_list(
            all_latents[:, :-1][:, :-1].detach().clone().cpu()
        )
        next_latents_lst = unbind_tensor_to_list(all_latents[:, 1:][:, :-1].detach().clone().cpu())
        all_log_probs_lst = unbind_tensor_to_list(all_log_probs[:, :-1].detach().clone().cpu())
        latent_position_ids = unbind_tensor_to_list(latent_position_ids.detach().clone().cpu())
        timesteps = self.timesteps[:self.sampling_steps - 1].detach().clone().cpu()
        timestep_lst = [timesteps for _ in range(batch_size)]

        # 虽然有点怪，但是 FluxPipeline 的 prompt 是 list，images 也是 list
        # TODO rename next_latents to prev_latents
        rb = {
            'images': images,
            "sample_lens": sample_lens,
            "packed_text_ids": packed_text_ids,
            "packed_text_indexes": packed_text_indexes,
            "packed_vae_token_indexes": packed_vae_token_indexes,
            "packed_position_ids": packed_position_ids,
            "split_lens": split_lens,
            "attn_modes": attn_modes,
            'latent_position_ids': latent_position_ids,
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

    @override
    def ppo_train_step(self, rollout_batches: List[Dict[str, Any]]):

        training_config = self.config.training
        gas_wo_t = training_config.train_gas_wo_timestep
        num_microbatches = training_config.train_gbs_wo_timestep // (
            training_config.train_mbs * dist.get_world_size()
        )
        assert training_config.train_mbs == 1, f"temporally force {training_config.train_mbs=} == 1"
        assert len(
            rollout_batches
        ) == num_microbatches * training_config.train_mbs, f"num_microbatches {num_microbatches}"
        # Ensure model/ema/optimizer are in consistent train state
        self.prepare_for_train()

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
                if self.config.training.guidance_scale > 1:
                    _latents = input_latent.repeat(2, 1, 1)
                    _latent_position_ids = data["latent_position_ids"].repeat(2, 1).view([-1])
                    _timesteps = timestep.view([-1, 1]).repeat(2, input_latent.shape[1]).view([-1])

                    def pack_cfg_data_l(key):
                        tmp = repeat_interleave_tensor_or_list(
                            self.uncond_feats[key], training_config.train_mbs
                        )
                        return tmp + data[key]

                    _sample_lens = pack_cfg_data_l("sample_lens")
                    _packed_text_ids = pack_cfg_data_l("packed_text_ids")
                    _packed_text_indexes = pack_cfg_data_l("packed_text_indexes")
                    _packed_vae_token_indexes = pack_cfg_data_l("packed_vae_token_indexes")
                    _packed_position_ids = pack_cfg_data_l("packed_position_ids")
                    _split_lens = pack_cfg_data_l("split_lens")
                    _attn_modes = pack_cfg_data_l("attn_modes")
                else:
                    _latents = input_latent
                    _latent_position_ids = data["latent_position_ids"].view([-1])
                    _timesteps = timestep.view([-1, 1]).repeat(1, input_latent.shape[1]).view([-1])
                    _sample_lens = data["sample_lens"]
                    _packed_text_ids = data["packed_text_ids"]
                    _packed_text_indexes = data["packed_text_indexes"]
                    _packed_vae_token_indexes = data["packed_vae_token_indexes"]
                    _packed_position_ids = data["packed_position_ids"]
                    _split_lens = data["split_lens"]
                    _attn_modes = data["attn_modes"]

                (
                    _sample_lens, _packed_text_ids, _packed_text_indexes, _packed_vae_token_indexes,
                    _packed_position_ids, _split_lens, _attn_modes
                ) = self.pack_batch(
                    _sample_lens, _packed_text_ids, _packed_text_indexes, _packed_vae_token_indexes,
                    _packed_position_ids, _split_lens, _attn_modes
                )

                if self.config.training.use_torch_autocast:
                    # TODO dtype config
                    autocast_ctx = torch.autocast("cuda", torch.bfloat16)
                else:
                    from contextlib import nullcontext
                    autocast_ctx = nullcontext()
                with autocast_ctx:
                    noise_pred = self.fsdp_model(
                        sequence_length=sum(_sample_lens),
                        packed_text_ids=_packed_text_ids,
                        packed_text_indexes=_packed_text_indexes,
                        sample_lens=_sample_lens,
                        packed_position_ids=_packed_position_ids,
                        split_lens=_split_lens,
                        attn_modes=_attn_modes,
                        packed_latent_position_ids=_latent_position_ids,
                        packed_vae_token_indexes=_packed_vae_token_indexes,
                        packed_timesteps=(_timesteps / 1000).bfloat16(),
                        mse_loss_indexes=_packed_vae_token_indexes,
                        packed_latent=_latents,
                    )["model_preds"]
                    noise_pred = noise_pred.reshape(*(_latents.shape))

                if self.config.training.guidance_scale > 1:
                    model_pred_uncond, model_pred_text = noise_pred.chunk(2)
                    noise_pred = self.merge_uncond_predict(model_pred_uncond, model_pred_text)

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
                loss_mean = [
                    avg_loss, ratio_detached
                ]  #average_losses_across_data_parallel_group([avg_loss, ratio_detached])
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
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.fsdp_model.parameters(), self.config.optimizer.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                extend_value_to_dict(metrics, {f"policy/grad_norm": [total_norm.item()]})

        return metrics

    def offload_encoder(self):
        pass

    def onload_encoder(self):
        pass

    @override
    def sft_train_step(self, rollout_batches: List[Dict[str, Any]]):
        pass
