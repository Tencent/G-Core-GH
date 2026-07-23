import uuid
from typing import Any, Dict, List, Set

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig
from transformers.models.auto.processing_auto import AutoProcessor
from typing_extensions import override

from megatron_datasets.mm_dataset import convert_conversations

from gpatch_v4.configs.config import RlConfig
from gpatch_v4.extended_model.base import (
    ApplySamplingRolloutAttrBase,
    PrepareDataForward,
    SamplerGenerateFunc,
)
from gpatch_v4.generation_backend.routed_experts_utils import process_routed_experts
from gpatch_v4.utils import BroadcastUtils, log


class ApplySamplingRolloutAttrMultiModal(ApplySamplingRolloutAttrBase):
    """Multi-modal rollout attribute handler with data caching for OOM prevention.

    Parameters
    ----------
    config : RlConfig
    """
    def __init__(self, config: RlConfig):
        self.config = config
        self.mm_data_cache: Dict[str, Dict[str, Any]] = {}
        self.mm_data_used: Set[str] = set()

    @override
    def remove_rollout_attr_before_sampling(self, rollout_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Strip cache_keys from the batch before sending to samplers (avoid OOM).

        Parameters
        ----------
        rollout_batch : Dict[str, List[Any]]

        Returns
        -------
        Dict[str, List[Any]]
        """
        assert "unique_id" in rollout_batch
        assert "cache_keys" in rollout_batch

        unique_id = rollout_batch["unique_id"][0]
        cache_keys = rollout_batch.pop("cache_keys")
        assert unique_id not in self.mm_data_cache
        self.mm_data_cache[unique_id] = {}
        for key in cache_keys:
            assert key in rollout_batch
            self.mm_data_cache[unique_id][key] = rollout_batch[key]
            del rollout_batch[key]

        return rollout_batch

    @override
    def remove_rollout_attr(self, rollout_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Second pass: strip cached fields again before next send (avoid OOM).

        Parameters
        ----------
        rollout_batch : Dict[str, List[Any]]

        Returns
        -------
        Dict[str, List[Any]]
        """
        assert "unique_id" in rollout_batch
        unique_id = rollout_batch["unique_id"][0]
        cache_keys = self.mm_data_cache[unique_id].keys()
        for key in cache_keys:
            assert key in rollout_batch
            del rollout_batch[key]

        return rollout_batch

    @override
    def add_back_rollout_attr_after_sampling(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
    ) -> List[Dict[str, List[Any]]]:
        """Add back attrs removed in ``remove_rollout_attr_before_sampling``.

        Parameters
        ----------
        rollout_batches : List[Dict[str, List[Any]]]

        Returns
        -------
        List[Dict[str, List[Any]]]
        """
        # get the self.mm_data_used
        # Broadcast mm_data_cache to TP/PP non-head ranks.
        # Skip when running in a single-process context (e.g. RolloutController)
        # where torch.distributed is not initialized.
        if torch.distributed.is_initialized():
            self.mm_data_cache = BroadcastUtils.broadcast_object_within_mp_and_cp(
                self.mm_data_cache, make_recursive_clone_in_case_of_view=True
            )
        for rollout_batch in rollout_batches:
            assert "unique_id" in rollout_batch
            uniq_ids = rollout_batch.pop("unique_id")
            for unique_id in uniq_ids:
                assert unique_id in self.mm_data_cache, f"error: {unique_id=} {self.mm_data_cache.keys()=}"
                self.mm_data_used.add(unique_id)
                for k, v in self.mm_data_cache[unique_id].items():
                    if k not in rollout_batch:
                        rollout_batch[k] = []
                    rollout_batch[k].append(v)

        return rollout_batches

    @override
    def add_back_rollout_attr(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
    ) -> List[Dict[str, List[Any]]]:
        """Add back attrs removed in ``remove_rollout_attr_before_sampling``.

        Parameters
        ----------
        rollout_batches : List[Dict[str, List[Any]]]

        Returns
        -------
        List[Dict[str, List[Any]]]
        """
        # 相比 add_back_rollout_attr_after_sampling，没有添加 self.mm_data_used 标志
        # 也没有去除 unique_id
        for rollout_batch in rollout_batches:
            assert "unique_id" in rollout_batch
            uniq_ids = rollout_batch["unique_id"]
            for unique_id in uniq_ids:
                assert unique_id in self.mm_data_cache, f"error: {unique_id=} {self.mm_data_cache.keys()=}"
                for k, v in self.mm_data_cache[unique_id].items():
                    if k not in rollout_batch:
                        rollout_batch[k] = []
                    rollout_batch[k].append(v)

        return rollout_batches

    @override
    def replay_rollout_batch(self, rollout_batch):
        # TODO(guanyouhe): 该路径未经测试
        assert "unique_id" in rollout_batch

        unique_id = rollout_batch["unique_id"][0]
        if unique_id in self.mm_data_used:
            self.mm_data_used.remove(unique_id)

    @override
    def clear_data_cache(self):
        data_used_tmp = list(self.mm_data_used)
        for k in data_used_tmp:
            del self.mm_data_cache[k]
            self.mm_data_used.remove(k)


class ApplySamplingRolloutAttrQwen3_5(ApplySamplingRolloutAttrMultiModal):
    """Qwen3.5 rollout attribute handler.

    相比 ``ApplySamplingRolloutAttrMultiModal``，修复了 ``rollout_batch[key]``
    本身就是 per-sample ``list`` 时被当成一条样本整份缓存的 bug：
    只有当值是长度等于样本数的 ``list`` 时才按 ``idx`` 拆分；其余情况
    （预拼好的 tensor、``None``、跨样本共享值）保留旧行为整份缓存，避免
    ``None[idx]`` 或误拆 tensor 的 batch 维。

    Parameters
    ----------
    config : RlConfig
    """
    @override
    def remove_rollout_attr_before_sampling(self, rollout_batch: Dict[str, Any]) -> Dict[str, Any]:
        assert "unique_id" in rollout_batch
        assert "cache_keys" in rollout_batch

        unique_id_list = rollout_batch["unique_id"]
        cache_keys = rollout_batch.pop("cache_keys")
        n = len(unique_id_list)
        for idx, unique_id in enumerate(unique_id_list):
            assert unique_id not in self.mm_data_cache
            self.mm_data_cache[unique_id] = {}
            for key in cache_keys:
                assert key in rollout_batch
                val = rollout_batch[key]
                if isinstance(val, list) and len(val) == n:
                    self.mm_data_cache[unique_id][key] = val[idx]
                else:
                    self.mm_data_cache[unique_id][key] = val

        for key in cache_keys:
            del rollout_batch[key]
        return rollout_batch

    @override
    def replay_rollout_batch(self, rollout_batch):
        # TODO: 该路径未经测试
        assert "unique_id" in rollout_batch
        for unique_id in rollout_batch["unique_id"]:
            if unique_id in self.mm_data_used:
                self.mm_data_used.remove(unique_id)


class ApplySamplingRolloutAttrQwen3_5_MOE(ApplySamplingRolloutAttrQwen3_5):
    """Qwen3.5 MoE rollout attribute handler.

    与 :class:`ApplySamplingRolloutAttrQwen3_5` 行为一致，单独保留一个类型
    是为了方便后续按架构单独定制。
    """


class SamplerGenerateFuncMultiModal(SamplerGenerateFunc):
    def __init__(self):
        self.processor = None
        self.hf_config = None

    def get_batch(self, batch, config):

        json_data_list = batch["json_data_list"]
        imgs_np_array_list = batch["imgs_np_array_list"]
        audios_np_array_list = batch.get("audios_np_array_list", None)

        prompt_texts_or_ids = []
        raw_images = []
        raw_audios = []
        labels = []
        assert len(imgs_np_array_list) == len(json_data_list)
        if audios_np_array_list is not None:
            assert len(audios_np_array_list) == len(json_data_list)
        for i, (json_data, imgs_np_array) in enumerate(zip(json_data_list, imgs_np_array_list)):
            imgs = None
            if imgs_np_array is not None:
                imgs = [Image.fromarray(img) for img in imgs_np_array]
            raw_images.append(imgs)
            audios = None
            if audios_np_array_list is not None and audios_np_array_list[i] is not None:
                audios = list(audios_np_array_list[i])
            raw_audios.append(audios)

            # pre
            conversations = convert_conversations(json_data['conversations'])
            add_generation_prompt = False
            if conversations[-1]['role'] == "assistant":
                conversations = conversations[:-1]

            assert conversations[-1]['role'] != "assistant"
            add_generation_prompt = True
            all_text = self.processor.apply_chat_template(
                conversations,
                tools=json_data.get("tools", None),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=config.training.enable_thinking,
            )

            prompt_texts_or_id = self.processor.tokenizer([all_text])["input_ids"][0]
            prompt_texts_or_ids.append(prompt_texts_or_id)

            labels.append(json_data['label'])

        return prompt_texts_or_ids, raw_images, raw_audios, labels, batch

    @override
    async def __call__(self, config, infer_engine, idx, tokenizer, batched_data,
                       sampling_repeat_n) -> Dict[str, List[Any]]:
        if self.processor is None:
            model_info = config.sampler.model_info[idx]
            self.processor = AutoProcessor.from_pretrained(model_info.hf_model_path)
            self.hf_config = AutoConfig.from_pretrained(model_info.hf_model_path)
        sampling_params = infer_engine.get_sampling_params_from_config(
            config.sampler.infer_engine_configs[idx],
            tokenizer.eos_token_id,
        )

        prompt_texts_or_ids, raw_images, raw_audios, labels, prompt_data = self.get_batch(
            batched_data, config
        )
        rank_unique_ids = prompt_data["unique_id"]
        tokens_from_dataset = prompt_data["tokens"]
        prompt_len_from_dataset = prompt_data["prompt_len"]
        json_data_list = batched_data["json_data_list"]
        teacher_tokens_from_dataset = prompt_data.get("teacher_tokens", None)
        teacher_prompt_len_from_dataset = prompt_data.get("teacher_prompt_len", None)

        # Audio sample rate must match what the training-side feature extractor uses;
        # this is fixed to 16kHz for Qwen3-Omni (Whisper extractor).
        audio_sample_rate = 16000

        gens = []
        for i, (prompt, image, audios) in enumerate(
            zip(prompt_texts_or_ids, raw_images, raw_audios)
        ):
            llm_input = dict(prompt_token_ids=prompt)
            mm_data = {}
            if image is not None:
                mm_data["image"] = image
            if audios is not None and len(audios) > 0:
                # vLLM expects each audio as (np.ndarray, sample_rate) tuple so it
                # can resample to the model-expected rate. We always use the same
                # numpy arrays that the training-side Whisper extractor consumes
                # to guarantee identical mel features.
                mm_data["audio"] = [
                    (np.asarray(a, dtype=np.float32), audio_sample_rate) for a in audios
                ]
            if mm_data:
                llm_input["multi_modal_data"] = mm_data
            for j in range(sampling_repeat_n):
                tmp_sampling_params = infer_engine.copy_sampling_params_with_seed_offset(
                    sampling_params, i * sampling_repeat_n + j
                )
                gen = infer_engine.async_generate(
                    llm_input,
                    tmp_sampling_params,
                    str(uuid.uuid4().hex),
                    return_routed_experts=config.training.moe_router_replay,
                )
                gens.append(gen)

        gen_outputs = await infer_engine.wait_and_get_async_generate_output(gens)

        tokens = []
        sequence_lengths = []
        prompt_lengths = []
        rollout_log_probs = []
        routed_experts_list = []
        pad_token_id = tokenizer.pad_token_id
        teacher_tokens = []
        teacher_squence_lengths = []
        teacher_prompt_lengths = []

        hf_cfg = config.policy.hf_config
        # for qwen3-omni
        if hasattr(hf_cfg, "thinker_config") and hasattr(hf_cfg.thinker_config, "text_config"):
            text_cfg = hf_cfg.thinker_config.text_config
        # for qwen3-vl
        elif hasattr(hf_cfg, "text_config"):
            text_cfg = hf_cfg.text_config
        else:
            text_cfg = hf_cfg
        moe_router_topk = getattr(text_cfg, "num_experts_per_tok", None)
        num_layers = text_cfg.num_hidden_layers

        for gi, _ in enumerate(gens):
            i = gi // sampling_repeat_n
            j = gi % sampling_repeat_n

            one_sample = gen_outputs[gi]
            one_output = one_sample.outputs[0]
            # the tokenizer should be same between actor and sampler
            assert one_output.prompt_len == prompt_len_from_dataset[
                i], f"{one_output.prompt_len=} {prompt_len_from_dataset[i]=}"
            one_prompt_token_ids = tokens_from_dataset[i][:prompt_len_from_dataset[i]].tolist()

            output_token_ids = list(one_output.token_ids)
            rollout_log_prob = one_output.output_logprobs
            assert len(output_token_ids) == len(rollout_log_prob)
            # Pitfall: possibly image/audio token id contained (perhaps due to the bad capability of model itself).
            mm_token_ids = set()
            for attr in ("image_token_id", "video_token_id", "audio_token_id"):
                tid = getattr(self.hf_config, attr, None)
                if tid is not None:
                    mm_token_ids.add(tid)
            for i in range(len(output_token_ids)):
                if output_token_ids[i] in mm_token_ids:
                    log(f"Warning: unexpect token ids!")
                    output_token_ids[i] = pad_token_id

            token = one_prompt_token_ids + output_token_ids
            assert len(token) <= config.training.seq_length

            sequence_lengths.append(torch.tensor(len(token), dtype=torch.long))
            tokens.append(torch.tensor(token, dtype=torch.long))
            prompt_lengths.append(torch.tensor(len(one_prompt_token_ids), dtype=torch.long))

            # 组装 rollout logps
            gen_lp = torch.tensor(rollout_log_prob, dtype=torch.float32)
            prompt_len = len(one_prompt_token_ids)
            full_lp = torch.ones(len(token), dtype=torch.float32)
            gen_len = gen_lp.size(0)
            assert len(token) == prompt_len + gen_len
            full_lp[prompt_len - 1:prompt_len + gen_len - 1] = gen_lp
            rollout_log_probs.append(full_lp)

            routed_experts = process_routed_experts(one_output, num_layers, moe_router_topk)
            routed_experts_list.append(routed_experts)

            # 组装 teacher tokens
            if teacher_tokens_from_dataset is not None:
                i = gi // sampling_repeat_n
                teacher_prompt_len = teacher_prompt_len_from_dataset[i]
                one_teacher_token_ids = teacher_tokens_from_dataset[i][:teacher_prompt_len].tolist()
                token_ids = one_teacher_token_ids + output_token_ids
                assert len(token_ids) <= config.training.seq_length
                teacher_tokens.append(torch.tensor(token_ids, dtype=torch.long))
                teacher_squence_lengths.append(torch.tensor(len(token_ids), dtype=torch.long))
                teacher_prompt_lengths.append(
                    torch.tensor(len(one_teacher_token_ids), dtype=torch.long)
                )

        labels = [label for label in labels for _ in range(sampling_repeat_n)]
        rank_unique_ids = [
            unique_id for unique_id in rank_unique_ids for _ in range(sampling_repeat_n)
        ]
        assert len(labels) == len(tokens)
        assert len(labels) == len(sequence_lengths)
        assert len(labels) == len(prompt_lengths)
        assert len(labels) == len(rank_unique_ids)

        rollout_batch = dict(
            tokens=tokens,
            sequence_lengths=sequence_lengths,
            prompt_lengths=prompt_lengths,
            labels=labels,
            rollout_log_probs=rollout_log_probs,
            unique_id=rank_unique_ids,
        )

        if teacher_tokens_from_dataset is not None:
            assert len(labels) == len(teacher_tokens)
            assert len(labels) == len(teacher_squence_lengths)
            assert len(labels) == len(teacher_prompt_lengths)
            rollout_batch["teacher_tokens"] = teacher_tokens
            rollout_batch["teacher_sequence_lengths"] = teacher_squence_lengths
            rollout_batch["teacher_prompt_lengths"] = teacher_prompt_lengths

        if config.training.moe_router_replay:
            rollout_batch["routed_experts"] = routed_experts_list

        if config.training.use_gen_rm_reward:
            json_data_repeats = [
                json_data for json_data in json_data_list for _ in range(sampling_repeat_n)
            ]
            assert len(labels) == len(json_data_repeats)
            rollout_batch["json_data_repeats"] = json_data_repeats

            # Images are not repeated as this may cause memory OOM
            imgs_np_array_list = prompt_data["imgs_np_array_list"]
            imgs_repeats = []
            for imgs in imgs_np_array_list:
                for i in range(sampling_repeat_n):
                    if i == 0:
                        if imgs is None:
                            imgs_repeats.append([np.array([])])
                        else:
                            imgs_repeats.append(imgs)
                    else:
                        imgs_repeats.append([np.array([])])
            rollout_batch["imgs_repeats"] = imgs_repeats

        return rollout_batch
