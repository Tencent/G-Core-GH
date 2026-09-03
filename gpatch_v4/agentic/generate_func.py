import uuid
from typing import Any, Dict, List

from typing_extensions import override

from gpatch_v4.extended_model.base import SamplerGenerateFunc
from gpatch_v4.generation_backend.routed_experts_utils import process_routed_experts


class SamplerGenerateFuncAgentic(SamplerGenerateFunc):
    @override
    async def __call__(
        self,
        config,
        infer_engine,
        idx,
        tokenizer,
        batched_data,
        sampling_repeat_n,
        is_eval: bool = False
    ) -> Dict[str, List[Any]]:

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
            is_eval=is_eval,
        )

        seed_offset_raw = batched_data.get("seed_offset", None)
        if seed_offset_raw is not None:
            assert isinstance(seed_offset_raw, list) and len(seed_offset_raw) == 1
            seed_offset = int(seed_offset_raw[0])
        else:
            seed_offset = None
        tmp_sampling_params = infer_engine.copy_sampling_params_with_seed_offset(
            sampling_params, seed_offset
        )

        # 支持 caller (e.g. TrajEnvManager._make_decision) 按需 cap 单次请求的 max_new_tokens，
        # 用来从源头保证 trajectory 累计 tokens 不超过 seq_length。
        # 与引擎默认值取 min: caller 只能"压低"，不能"放宽"。
        override_max_new_tokens = batched_data.get("max_new_tokens", None)
        if override_max_new_tokens is not None:
            assert isinstance(override_max_new_tokens, list) and len(override_max_new_tokens) == 1
            override = int(override_max_new_tokens[0])
            assert override > 0, f"max_new_tokens must be positive, got {override}"
            # vllm 用 max_tokens，sglang 用 max_new_tokens，自动兼容两种 backend
            _max_tokens_field = "max_new_tokens" if hasattr(
                tmp_sampling_params, "max_new_tokens"
            ) else "max_tokens"
            setattr(
                tmp_sampling_params, _max_tokens_field,
                min(getattr(tmp_sampling_params, _max_tokens_field), override)
            )

        llm_input = dict(prompt_token_ids=prompt_token_ids)
        if images is not None:
            llm_input.update({
                "multi_modal_data": {
                    "image": images,
                },
            })
        return_routed_experts = config.training.moe_router_replay
        if return_routed_experts and config.sampler.backend != "vllm":
            raise NotImplementedError(
                "Agentic MoE router replay currently supports only the vLLM backend"
            )
        gen = infer_engine.async_generate(
            llm_input,
            tmp_sampling_params,
            str(uuid.uuid4().hex),
            return_routed_experts=return_routed_experts,
        )

        gen_outputs = await infer_engine.wait_and_get_async_generate_output([gen])
        assert len(gen_outputs) == 1
        gen_out = gen_outputs[0]
        completion = gen_out.outputs[0]
        resp_tokens = list(completion.token_ids)
        prompt_len = completion.prompt_len
        output_logprobs = completion.output_logprobs
        engine_finish_reason = getattr(completion, "finish_reason", None)
        if engine_finish_reason is None:
            meta_info = getattr(completion, "meta_info", None)
            finish_info = meta_info.get("finish_reason") if isinstance(meta_info, dict) else None
            if isinstance(finish_info, dict):
                engine_finish_reason = finish_info.get("type")
            elif isinstance(finish_info, str):
                engine_finish_reason = finish_info
        rollout_batch = {
            "response_ids": resp_tokens,
            "prompt_len": prompt_len,
            "output_logprobs": output_logprobs,
            # Preserve the backend termination signal. In particular, SGLang
            # reports max_new_tokens exhaustion as "length"; dropping it makes
            # the Agent layer misclassify a truncated answer as a normal stop.
            "engine_finish_reason": engine_finish_reason,
        }

        if return_routed_experts and config.sampler.backend == "vllm":
            num_layers = config.training.moe_router_replay_num_layers
            moe_router_topk = config.training.moe_router_replay_topk
            assert num_layers is not None and moe_router_topk is not None, (
                "MoE router replay shape must be resolved before sampler actors "
                "are created"
            )
            num_layers = int(num_layers)
            moe_router_topk = int(moe_router_topk)

            routed_experts = process_routed_experts(
                gen_out.outputs[0], num_layers, moe_router_topk
            )
            assert routed_experts is not None, (
                "moe_router_replay is enabled, but the sampler did not return "
                "routed_experts; check the vLLM/SGLang version and sampler startup flags"
            )
            expected_seq_len = prompt_len + len(resp_tokens)
            assert routed_experts.ndim == 3, (
                "routed_experts must be [seq, layer, topk], got "
                f"{routed_experts.shape=}"
            )
            assert routed_experts.shape == (
                expected_seq_len,
                num_layers,
                moe_router_topk,
            ), (
                "agentic sampler routing is not token-aligned: "
                f"{routed_experts.shape=} != "
                f"({expected_seq_len}, {num_layers}, {moe_router_topk})"
            )
            rollout_batch["routed_experts"] = routed_experts
        return rollout_batch


generate_func = SamplerGenerateFuncAgentic()
