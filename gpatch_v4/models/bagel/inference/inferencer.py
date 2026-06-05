# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from ..data.data_utils import pil_img2rgb
from ..modeling.bagel.qwen2_navit import NaiveCache

VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer.
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image.
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''


class InterleaveInferencer:
    """
    Bagel 模型的交错式推理器

    支持多模态输入（文本和图像交错）和输出（文本理解或图像生成）

    Args:
        model: Bagel 模型实例
        vae_model: VAE 自编码器模型
        tokenizer: Qwen2 分词器
        vae_transform: VAE 图像变换
        vit_transform: ViT 图像变换
        new_token_ids: 特殊 token ID 字典
    """
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        self.model = model
        self.vae_model = vae_model
        self.tokenizer = tokenizer
        self.vae_transform = vae_transform
        self.vit_transform = vit_transform
        self.new_token_ids = new_token_ids

    def init_gen_context(self):
        """初始化生成上下文，包含 KV cache 和位置信息"""
        gen_context = {
            'kv_lens': [0],
            'ropes': [0],
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        """
        用文本更新生成上下文

        Args:
            text: 输入文本
            gen_context: 当前生成上下文

        Returns:
            更新后的生成上下文
        """
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            prompts=[text],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )

        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values

        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        """
        用图像更新生成上下文

        Args:
            image: 输入图像 (PIL.Image)
            gen_context: 当前生成上下文
            vae: 是否使用 VAE 编码
            vit: 是否使用 ViT 编码

        Returns:
            更新后的生成上下文
        """
        assert vae or vit
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        if vae:
            # 更新 VAE 特征
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vae_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(
                self.vae_model, past_key_values, **generation_input
            )

        if vit:
            # 更新 ViT 特征
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes,
                images=[image],
                transforms=self.vit_transform,
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(
                past_key_values, **generation_input
            )

        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values

        return gen_context

    @torch.no_grad()
    def gen_image(
        self,
        image_shape,
        gen_context,
        cfg_text_scale=4.0,
        cfg_img_scale=1.5,
        cfg_text_precontext=None,
        cfg_img_precontext=None,
        cfg_interval=(0.4, 1.0),
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        num_timesteps=50,
        timestep_shift=3.0
    ):
        """
        生成图像

        Args:
            image_shape: 图像尺寸 (H, W)
            gen_context: 生成上下文
            cfg_text_scale: 文本 CFG 强度
            cfg_img_scale: 图像 CFG 强度
            cfg_text_precontext: 文本 CFG 预上下文
            cfg_img_precontext: 图像 CFG 预上下文
            cfg_interval: CFG 应用的时间步区间
            cfg_renorm_min: CFG 重归一化最小值
            cfg_renorm_type: CFG 重归一化类型
            num_timesteps: 扩散步数
            timestep_shift: 时间步偏移

        Returns:
            生成的图像 (PIL.Image)
        """
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes,
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )

        # 文本 CFG
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )

        # 图像 CFG
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg,
            image_sizes=[image_shape],
        )

        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text[
                'cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'
                                                                     ],
        )

        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

    def decode_image(self, latent, image_shape):
        """
        解码 latent 为图像

        Args:
            latent: latent 向量
            image_shape: 目标图像尺寸 (H, W)

        Returns:
            PIL.Image
        """
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample

        latent = latent.reshape(
            1, h, w, self.model.latent_patch_size, self.model.latent_patch_size,
            self.model.latent_channel
        )
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(
            1, self.model.latent_channel, h * self.model.latent_patch_size,
            w * self.model.latent_patch_size
        )
        image = self.vae_model.decode(
            latent.to(
                next(self.vae_model.parameters()).device,
                dtype=next(self.vae_model.parameters()).dtype
            )
        )
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(
        self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0
    ):
        """
        生成文本

        Args:
            gen_context: 生成上下文
            max_length: 最大生成长度
            do_sample: 是否采样
            temperature: 采样温度

        Returns:
            生成的文本
        """
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        output = self.tokenizer.decode(unpacked_latent[:, 0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,
        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(512, 512),
    ) -> List[Union[str, Image.Image]]:
        """
        交错式推理：支持文本和图像交错输入，输出文本或图像

        Args:
            input_lists: 输入列表，可包含文本（str）和图像（PIL.Image）
            think: 是否启用思考模式
            understanding_output: 输出类型（True: 文本理解, False: 图像生成）
            max_think_token_n: 思考阶段最大 token 数
            do_sample: 文本生成是否采样
            text_temperature: 文本生成温度
            cfg_text_scale: 文本 CFG 强度
            cfg_img_scale: 图像 CFG 强度
            cfg_interval: CFG 应用区间
            timestep_shift: 时间步偏移
            num_timesteps: 扩散步数
            cfg_renorm_min: CFG 重归一化最小值
            cfg_renorm_type: CFG 重归一化类型
            image_shapes: 输出图像尺寸 (H, W)

        Returns:
            输出列表，包含生成的文本或图像
        """
        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            for input_term in input_lists:
                if isinstance(input_term, str):
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(
                        input_term, gen_context, vae=not understanding_output
                    )

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"Unsupported input type: {type(input_term)}")

            if understanding_output:
                gen_text = self.gen_text(
                    gen_context,
                    do_sample=do_sample,
                    temperature=text_temperature,
                    max_length=max_think_token_n
                )
                output_list.append(gen_text)

            else:
                if think:
                    gen_text = self.gen_text(
                        gen_context,
                        do_sample=do_sample,
                        temperature=text_temperature,
                        max_length=max_think_token_n
                    )
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                img = self.gen_image(
                    image_shapes,
                    gen_context,
                    cfg_text_precontext=cfg_text_context,
                    cfg_img_precontext=cfg_img_context,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    timestep_shift=timestep_shift,
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )

                output_list.append(img)

        return output_list

    def __call__(
        self,
        image: Optional[Union[Image.Image, List[Image.Image]]] = None,
        text: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        便捷调用接口

        Args:
            image: 输入图像（单张或列表）
            text: 输入文本
            **kwargs: 传递给 interleave_inference 的其他参数

        Returns:
            字典，包含 'image' 和 'text' 两个键
        """
        output_dict = {'image': None, 'text': None}

        if image is None and text is None:
            print('Please provide at least one input: either an image or text.')
            return output_dict

        input_list = []
        if image is not None:
            if isinstance(image, Image.Image):
                input_list.append(image)
            elif isinstance(image, list):
                for img in image:
                    assert isinstance(
                        img, Image.Image
                    ), f"image must be a list of Image.Image, but got {type(img)}"
                input_list.extend(image)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

        if text is not None:
            input_list.append(text)

        output_list = self.interleave_inference(input_list, **kwargs)

        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
        return output_dict
