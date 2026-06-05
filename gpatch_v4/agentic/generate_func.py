import copy
import uuid
from typing import Any, Dict, List, Tuple

from typing_extensions import override

from gpatch_v4.extended_model.base import SamplerGenerateFunc


class SamplerGenerateFuncAgentic(SamplerGenerateFunc):
    @override
    async def __call__(self, config, infer_engine, idx, tokenizer, batched_data,
                       sampling_repeat_n) -> Dict[str, List[Any]]:

        assert sampling_repeat_n == 1
        prompt_token_ids = batched_data["prompt_token_ids"]
        assert isinstance(prompt_token_ids, list)
        assert len(prompt_token_ids) == 1
        prompt_token_ids = prompt_token_ids[0].tolist()

        images = batched_data.get("images", None)
        if images is not None:
            assert isinstance(images, list)
            assert len(images) == 1
            images = images[0]
        logit_bias = {}
        # 不产出多模态 token
        if hasattr(config.policy.hf_config, "image_token_id"):
            image_token_id = config.policy.hf_config.image_token_id
            logit_bias[image_token_id] = -100
        if hasattr(config.policy.hf_config, "video_token_id"):
            video_token_id = config.policy.hf_config.video_token_id
            logit_bias[video_token_id] = -100

        sampling_params = infer_engine.get_sampling_params_from_config(
            config.sampler.infer_engine_configs[idx],
            tokenizer.eos_token_id,
            logit_bias=logit_bias,
        )

        tmp_sampling_params = copy.deepcopy(sampling_params)

        # 支持 caller (e.g. TrajEnvManager._make_decision) 按需 cap 单次请求的 max_new_tokens，
        # 用来从源头保证 trajectory 累计 tokens 不超过 seq_length。
        # 与引擎默认值取 min: caller 只能"压低"，不能"放宽"。
        override_max_new_tokens = batched_data.get("max_new_tokens", None)
        if override_max_new_tokens is not None:
            assert isinstance(override_max_new_tokens, list) and len(override_max_new_tokens) == 1
            override = int(override_max_new_tokens[0])
            assert override > 0, f"max_new_tokens must be positive, got {override}"
            tmp_sampling_params.max_new_tokens = min(tmp_sampling_params.max_new_tokens, override)

        llm_input = dict(prompt_token_ids=prompt_token_ids)
        if images is not None:
            llm_input.update({
                "multi_modal_data": {
                    "image": images,
                },
            })
        gen = infer_engine.async_generate(llm_input, tmp_sampling_params, str(uuid.uuid4().hex))

        gen_outputs = await infer_engine.wait_and_get_async_generate_output([gen])
        assert len(gen_outputs) == 1
        gen_out = gen_outputs[0]
        resp_tokens = list(gen_out.outputs[0].token_ids)
        prompt_len = gen_out.outputs[0].prompt_len
        output_logprobs = gen_out.outputs[0].output_logprobs
        rollout_batch = {
            "response_ids": resp_tokens,
            "prompt_len": prompt_len,
            "output_logprobs": output_logprobs
        }
        return rollout_batch


generate_func = SamplerGenerateFuncAgentic()
