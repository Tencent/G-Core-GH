"""Custom text-only rollout generation function for Gemma4 GRPO.

Mirrors :class:`SamplerGenerateFuncLLM` but uses ``gt_label`` instead of
the multimodal ``json_data_list`` path, avoiding the prompt-length assertion
that fails when vLLM internally strips tokens for Gemma4.
"""

import uuid
from typing import Any, Dict, List

import torch

from gpatch_v4.generation_backend.routed_experts_utils import process_routed_experts


async def gemma4_text_gen_rollout(
    config, infer_engine, idx, tokenizer, batched_data, sampling_repeat_n
) -> Dict[str, List[Any]]:
    rank_unique_ids = batched_data["unique_id"]
    prompt_token_ids = batched_data["prompt_token_ids"]
    prompt_lens = batched_data["prompt_lens"]
    gt_label = batched_data["gt_label"]

    sampling_params = infer_engine.get_sampling_params_from_config(
        config.sampler.infer_engine_configs[idx], tokenizer.eos_token_id
    )
    async_gens = []
    for i in range(len(prompt_token_ids)):
        for j in range(sampling_repeat_n):
            tmp_sampling_params = infer_engine.copy_sampling_params_with_seed_offset(
                sampling_params, i * sampling_repeat_n + j
            )
            gen = infer_engine.async_generate(
                prompt_token_ids[i],
                tmp_sampling_params,
                str(uuid.uuid4().hex),
                return_routed_experts=config.training.moe_router_replay,
            )
            async_gens.append(gen)

    gen_outputs = await infer_engine.wait_and_get_async_generate_output(async_gens)

    tokens_lst = []
    seq_length_lst = []
    prompt_len_lst = []
    label_lst = []
    routed_experts_list = []
    unique_id_list = []
    rollout_log_probs_lst = []

    hf_config = config.policy.hf_config
    if hasattr(hf_config, "text_config"):
        num_layers = hf_config.text_config.num_hidden_layers
        moe_router_topk = getattr(hf_config.text_config, "num_experts_per_tok", None)
    else:
        num_layers = hf_config.num_hidden_layers
        moe_router_topk = getattr(hf_config, "num_experts_per_tok", None)

    for gi, gen_out in enumerate(gen_outputs):
        i = gi // sampling_repeat_n
        assert len(gen_out.outputs) == 1

        one_output = gen_out.outputs[0]
        resp_tokens = list(one_output.token_ids)
        one_prompt_token_ids = prompt_token_ids[i]['prompt_token_ids']
        assert one_output.prompt_len == len(one_prompt_token_ids), (
            f"prompt_len mismatch: engine={one_output.prompt_len}, "
            f"dataset={len(one_prompt_token_ids)}. "
            f"The dataset tokenization must match the inference engine."
        )
        token = one_prompt_token_ids + resp_tokens
        assert len(token) <= config.training.seq_length
        tokens_lst.append(torch.tensor(token, dtype=torch.long))
        seq_length_lst.append(torch.tensor(len(token), dtype=torch.long))
        prompt_len_lst.append(prompt_lens[i])
        label_lst.append(gt_label[i])
        unique_id_list.append(rank_unique_ids[i])

        rollout_log_prob = one_output.output_logprobs
        assert len(resp_tokens
                  ) == len(rollout_log_prob), f"{len(resp_tokens)=} {len(rollout_log_prob)=}"
        gen_lp = torch.tensor(rollout_log_prob, dtype=torch.float32)
        prompt_len = len(one_prompt_token_ids)
        full_lp = torch.ones(len(token), dtype=torch.float32)
        gen_len = gen_lp.size(0)
        assert len(token) == prompt_len + gen_len
        full_lp[prompt_len - 1:prompt_len + gen_len - 1] = gen_lp
        rollout_log_probs_lst.append(full_lp)

        routed_experts = process_routed_experts(one_output, num_layers, moe_router_topk)
        routed_experts_list.append(routed_experts)

    rollout_batch = dict(
        tokens=tokens_lst,
        sequence_lengths=seq_length_lst,
        prompt_lengths=prompt_len_lst,
        labels=label_lst,
        unique_id=unique_id_list,
        rollout_log_probs=rollout_log_probs_lst,
    )
    if config.training.moe_router_replay:
        rollout_batch["routed_experts"] = routed_experts_list
    return rollout_batch
