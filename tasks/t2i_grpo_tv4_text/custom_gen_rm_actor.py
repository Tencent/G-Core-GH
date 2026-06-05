import asyncio
import copy
import uuid
from typing import Any, Dict, List

import torch

from gpatch_v4.actor.t2i_grpo_gen_rm_actor import T2iGrpoGenRmActor
from gpatch_v4.utils import log, unbind_tensor_to_list

# isort: off
from tasks.t2i_grpo_tv4_text.reward_util import (
    build_character_ocr_message,
    build_dsg_entity_message,
    build_dsg_entity_score_message,
    build_ocr_message,
    calculate_match_score,
    compute_character_score,
    extract_text_gt_from_prompt,
    safe_json_loads,
)
# isort: on
"""
oteam text 一个 model, 多个 generative reward, 索性定制化了
"""


class OteamTextGrpoGenRmActor(T2iGrpoGenRmActor):
    def post_init(self, rm_idx):
        pass

    def process_conversation(self, message):
        prompt = self.processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = self.tokenizer(prompt)["input_ids"]
        return token_ids

    async def textreward_char(self, batched_data: Dict[str, List[Any]], sampling_params):

        repeat_n = self.config.training.sampling_repeat_n
        rollout_mbs = self.config.training.rollout_mbs
        message = build_character_ocr_message("image")
        token_ids = self.process_conversation(message)
        async_gens = []
        for i in range(repeat_n * rollout_mbs):
            image = batched_data["images"][i]
            vlm_inputs = {
                'prompt_token_ids': token_ids,
                'multi_modal_data': {
                    "image": image
                },
            }
            tmp_sampling_params = self.infer_engine.copy_sampling_params_with_seed_offset(
                sampling_params, i
            )
            gen = self.infer_engine.async_generate(
                vlm_inputs, tmp_sampling_params, str(uuid.uuid4().hex)
            )
            async_gens.append(gen)

        gen_outputs = await self.infer_engine.wait_and_get_async_generate_output(async_gens)
        output_token_ids = []
        for gi in range(len(async_gens)):
            sample_output = gen_outputs[gi]
            assert len(sample_output.outputs) == 1
            sample_output = sample_output.outputs[0]
            output_token_ids.append(list(sample_output.token_ids))

        resp_texts = self.tokenizer.batch_decode(output_token_ids, skip_special_tokens=False)
        character_scores = []
        for (caption, ocr_text) in zip(batched_data['prompt'], resp_texts):
            ocr_text = ocr_text.strip() if ocr_text else "无"
            char_score_result = compute_character_score(ocr_text, caption)
            character_score = char_score_result.get("score", 0.0)
            character_scores.append(character_score)
        return torch.tensor(character_scores, dtype=torch.float32)

    async def call_dsg_score(self, image, caption, questions, sampling_params):
        message = build_dsg_entity_score_message(image, caption, questions)
        token_ids = self.process_conversation(message)
        vlm_inputs = {
            'prompt_token_ids': token_ids,
            'multi_modal_data': {
                "image": image
            },
        }

        gen = self.infer_engine.async_generate(vlm_inputs, sampling_params, str(uuid.uuid4().hex))

        gen_outputs = await self.infer_engine.wait_and_get_async_generate_output([gen])
        sample_output = gen_outputs[0].outputs[0]

        res = self.tokenizer.decode(list(sample_output.token_ids))
        parsed = safe_json_loads(res, default=None)
        if parsed is None or not isinstance(parsed, dict):
            return {"answers": [], "score": 0.0}
        # 兼容新旧格式：新格式直接有score，旧格式在summary中
        if "score" not in parsed and "summary" in parsed:
            parsed["score"] = parsed["summary"].get("score", 0.0)

        return parsed

    async def textreward_dsg(self, batched_data: Dict[str, List[Any]], sampling_params):
        async_gens = []
        prompts = batched_data['prompt']
        # build identity
        for i, caption in enumerate(prompts):
            message = build_dsg_entity_message(caption)
            token_ids = self.process_conversation(message)
            vlm_inputs = {
                'prompt_token_ids': token_ids,
            }
            gen = self.infer_engine.async_generate(
                vlm_inputs, sampling_params, str(uuid.uuid4().hex)
            )
            async_gens.append(gen)

        gen_outputs = await self.infer_engine.wait_and_get_async_generate_output(async_gens)
        output_token_ids = []
        for gi in range(len(async_gens)):
            sample_output = gen_outputs[gi]
            assert len(sample_output.outputs) == 1
            sample_output = sample_output.outputs[0]
            output_token_ids.append(list(sample_output.token_ids))
        resp_texts = self.tokenizer.batch_decode(output_token_ids, skip_special_tokens=False)

        # dsg score
        async def dsg_score_task(image, caption, questions):
            if len(questions) > 0:
                dsg_score_result = await self.call_dsg_score(
                    image, caption, questions, sampling_params
                )
                # 兼容新旧格式
                dsg_score = dsg_score_result.get(
                    "score",
                    dsg_score_result.get("summary", {}).get("score", 0.0)
                )
            else:
                dsg_score = 0.0
            return dsg_score

        async_scores = []
        for i, res in enumerate(resp_texts):
            questions = safe_json_loads(res, default={"questions": []}).get("questions", [])
            caption = prompts[i]
            image = batched_data["images"][i]
            async_scores.append(dsg_score_task(image, caption, questions))

        scores = await asyncio.gather(*async_scores)
        return torch.tensor(scores, dtype=torch.float32)

    async def textreward_ocr(self, batched_data: Dict[str, List[Any]], sampling_params):
        repeat_n = self.config.training.sampling_repeat_n
        rollout_mbs = self.config.training.rollout_mbs
        message = build_ocr_message("image")
        token_ids = self.process_conversation(message)
        async_gens = []
        for i in range(repeat_n * rollout_mbs):
            image = batched_data["images"][i]
            vlm_inputs = {
                'prompt_token_ids': token_ids,
                'multi_modal_data': {
                    "image": image
                },
            }
            tmp_sampling_params = self.infer_engine.copy_sampling_params_with_seed_offset(
                sampling_params, i
            )
            gen = self.infer_engine.async_generate(
                vlm_inputs, tmp_sampling_params, str(uuid.uuid4().hex)
            )
            async_gens.append(gen)

        gen_outputs = await self.infer_engine.wait_and_get_async_generate_output(async_gens)
        output_token_ids = []
        for gi in range(len(async_gens)):
            sample_output = gen_outputs[gi]
            assert len(sample_output.outputs) == 1
            sample_output = sample_output.outputs[0]
            output_token_ids.append(list(sample_output.token_ids))

        resp_texts = self.tokenizer.batch_decode(output_token_ids, skip_special_tokens=False)
        ocr_scores = []
        for (caption, ocr_text) in zip(batched_data['prompt'], resp_texts):
            text_gt = extract_text_gt_from_prompt(caption)
            ocr_score, _, _ = calculate_match_score(text_gt, ocr_text)
            ocr_scores.append(ocr_score)
        return torch.tensor(ocr_scores, dtype=torch.float32)

    async def generate_rewards(self, req_dict: Dict[str, Any]):
        batched_data: Dict[str, List[Any]] = req_dict["batched_data"]
        repeat_n = self.config.training.sampling_repeat_n
        rollout_mbs = self.config.training.rollout_mbs
        for k, v in batched_data.items():
            assert isinstance(v, list) and len(
                v
            ) == rollout_mbs * repeat_n, f'unexpected {k=} {v=} {rollout_mbs=} {repeat_n=} {len(v)=}'

        sampling_params = self.infer_engine.get_sampling_params_from_config(
            self.config.gen_rm.infer_engine_configs[self.rm_idx], self.tokenizer.eos_token_id
        )
        # textreward_char
        rewards = {}
        log("textreward_char")
        textreward_char_scores = await self.textreward_char(batched_data, sampling_params)
        textreward_char_scores = unbind_tensor_to_list(textreward_char_scores.view(-1, 1))
        rewards["textreward_char"] = {}
        rewards["textreward_char"]["rewards"] = textreward_char_scores
        rewards["textreward_char"]["weight"] = 0.4

        log("textreward_ocr")
        textreward_ocr_scores = await self.textreward_ocr(batched_data, sampling_params)
        textreward_ocr_scores = unbind_tensor_to_list(textreward_ocr_scores.view(-1, 1))
        rewards["textreward_ocr"] = {}
        rewards["textreward_ocr"]["rewards"] = textreward_ocr_scores
        rewards["textreward_ocr"]["weight"] = 0.4

        log("textreward_dsg")
        textreward_dsg_scores = await self.textreward_dsg(batched_data, sampling_params)
        textreward_dsg_scores = unbind_tensor_to_list(textreward_dsg_scores.view(-1, 1))
        rewards["textreward_dsg"] = {}
        rewards["textreward_dsg"]["rewards"] = textreward_dsg_scores
        rewards["textreward_dsg"]["weight"] = 0.2

        ret_dict = {
            f"reward_gen_rm_{self.rm_idx}": rewards,
        }

        return ret_dict
