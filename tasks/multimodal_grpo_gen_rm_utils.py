# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# guanyouhe@tencent.com, xiaotaoliu@tencent.com, nrwu@tencent.com

import json
import re
import uuid
import copy
import random
from typing import Dict, List, Any, Optional

import torch
from PIL import Image
from transformers import AutoProcessor

from megatron.training.global_vars import get_args

from gpatch.training.global_vars import get_actor_tokenizer, get_rm_tokenizer
from gpatch.core.utils import list_for_tensor_tolist
from gpatch.core.aligner_helper import expand_rollout_batch
from gpatch.core.utils import load_and_call

from tasks.multimodal_grpo_critic_utils import rule_based_rm
from gpatch.core.utils import load_and_call
from tasks.multimodal_comm.multimodal_dataset_map import convert_conversations


def get_sampling_params(engine, eos_token_id):
    args = get_args()

    # Create a sampling params object.
    return engine.get_sampling_params(
        n=1,  # by passing vllm async llm issues
        temperature=args.ppo_gen_rm_temperature,
        top_k=args.ppo_gen_rm_top_k if args.ppo_gen_rm_top_k > 0 else -1,
        top_p=args.ppo_gen_rm_top_p,
        max_tokens=args.ppo_gen_rm_resp_seq_len,
        stop_token_ids=[eos_token_id],
        seed=args.seed,
        repetition_penalty=args.ppo_gen_rm_repetition_penalty,
    )


def get_rule_based_rm(batches: List[Dict[str, List[Any]]] = None):
    args = get_args()
    sequence_lengths_list = []
    prompt_lengths_list = []
    inputs_list = []
    labels = []
    for batch in batches:
        sequence_lengths_list.extend(batch["sequence_lengths"])
        prompt_lengths_list.extend(batch["prompt_lengths"])
        inputs_list.extend(batch["tokens"])
        labels.extend(batch["labels"])
    seq_len_cpu = torch.stack(sequence_lengths_list).view(-1).tolist()
    prompt_len_cpu = torch.stack(prompt_lengths_list).view(-1).tolist()
    tokens_cpu = list_for_tensor_tolist(inputs_list, False)

    return rule_based_rm(
        seq_len_cpu,
        prompt_len_cpu,
        tokens_cpu,
        labels,
        args.ppo_mm_rule_type,
        args.ppo_custom_rule_file,
        args.ppo_fmt_factor,
        args.ppo_skip_special_tokens,
        get_actor_tokenizer(),
    )


def parse_rm_model_output(content):
    result = ""
    has_input_text_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    if has_input_text_match:
        result = has_input_text_match.group(1).strip().lower()

    reason = ""
    summaary_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    if summaary_match:
        reason = summaary_match.group(1).strip()

    return result.strip(), reason.strip()


def replace_last_str(input_str: str, replace_str: str, custom_str: str):
    parts = input_str.rsplit(replace_str, 1)
    if len(parts) > 1:
        return parts[0] + custom_str + parts[1]
    return input_str.replace(replace_str, custom_str, 1)


g_rm_processor = None


def get_rm_processor(rm_tokenizer_model: str):
    global g_rm_processor
    if g_rm_processor is None:
        g_rm_processor = AutoProcessor.from_pretrained(rm_tokenizer_model)
    return g_rm_processor


# return the prompt text and image list
def prepare_data_for_gen_rm_geo3k(
    sample: Dict[str, Any],
    actor_tokenizer,
    processor,
    **kwargs,
) -> tuple[str, list[Image.Image]]:
    label = json.loads(sample["labels"])
    problem = label["problem"]
    src_imgs = [Image.fromarray(img_array) for img_array in sample["imgs_repeats"]]
    sampler_tokens = sample["tokens"][sample["prompt_lengths"]:sample["sequence_lengths"]]
    sampler_resp_strs = actor_tokenizer._tokenizer.decode(sampler_tokens, skip_special_tokens=False)
    summary_tag_pattern = r'<think>(.*?)</think>'
    reasoning_match = re.search(summary_tag_pattern, sampler_resp_strs, re.DOTALL)
    reasoning = ""
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    alignment_judge_promot = """
Role Setting:
You are a geometry expert, skilled in analyzing and validating solutions to geometry problems.

Task Description:
You need to evaluate whether Model reasoning process for a given geometry problem is correct. The input includes the problem itself and Model reasoning process. Follow these steps:

- Analyze the Problem: Read the problem carefully to understand its requirements and given conditions.
- Evaluate the Reasoning: Check Model reasoning step-by-step to see if it is logical, rigorous, and leads to a correct conclusion.
- Generate Judgment: Based on your analysis, determine whether Model reasoning is correct.

Output Format Requirements:

- Place your reasoning process between <think> and </think>.
- Place the final judgment (only yes or no) between <answer> and </answer>.

Example Input:
Problem: In triangle ABC, angle A=30° and angle B=60°. Find the measure of angle C.
Model reasoning process: Since the sum of angles in a triangle is 180°, angle C=180°-angle A-angle B=180°-30°-60°=90°.

Example Output:
<think>
The problem provides two angles of triangle ABC: angle A=30° and angle B=60°. According to the triangle angle sum theorem, the sum of the three angles is 180°. Therefore, angle C=180°-30°-60°=90°. Model reasoning is entirely correct.
</think>
<answer>yes</answer>

Here is the real input data:
- problem: {problem}
- Model reasoning process: {reasoning}
"""
    # the {problem} has <image> label
    messages = [
        {
            'role': 'user',
            'content': replace_last_str(alignment_judge_promot, "{problem}", problem),
        }
    ]
    messages = convert_conversations(messages)

    assert 'rm_tokenizer_models' in kwargs
    rm_processor = get_rm_processor(kwargs['rm_tokenizer_models'][0])
    prompt = rm_processor.apply_chat_template(
        messages,
        tools=None,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt = replace_last_str(prompt, "{reasoning}", reasoning)

    return prompt, src_imgs


# return the prompt text and image list
def prepare_data_for_gen_rm_pmc_vqa(
    sample: Dict[str, Any],
    actor_tokenizer,
    processor,
    **kwargs,
) -> tuple[str, list[Image.Image]]:
    label = json.loads(sample["labels"])
    src_imgs = [Image.fromarray(img_array) for img_array in sample["imgs_repeats"]]
    sampler_tokens = sample["tokens"][sample["prompt_lengths"]:sample["sequence_lengths"]]
    sampler_resp_strs = actor_tokenizer._tokenizer.decode(sampler_tokens, skip_special_tokens=False)

    content = """**Instructions:**:

You will be given the following information:
1. **The Question:** {question}
2. **The Image:** <image>
3. **The Image Caption:** {caption}
4. **The Options: **
{A}
{B}
{C}
{D}
5. **The Ground Truth Answer:** {answer}
6. **The Model's Response:** {sampler_resp}

**Evaluation Criteria:**
You must evaluate the model's response based on two criteria:

1. **Reasoning Correctness:** Analyze the text before the final <answer> tag. This reasoning must be logically sound, factually consistent with the provided Image Caption, and directly address the Question. It should not contain significant errors or contradictions.
2. **Answer Correctness:** The text inside the final `<answer>...</answer>` tags must exactly match the provided Ground Truth Answer.

**Final Judgment Rule:**
The response is considered correct only if both of the following are true:
- The **Reasoning Correctness** is satisfactory.
- The **Answer Correctness** is satisfactory.

Output Format:
You must output **only** one of the following two lines, with no other text, explanations, or formatting.

- If the response is correct (both criteria are met), output:
  `<answer>yes</answer>`

If the response is incorrect (either criterion is not met), output:
  `<answer>no</answer>`
"""
    messages = [
        {
            "role":
                "system",
            "content":
                """**Role:**: You are a precise medical imaging evaluation assistant. Your task is to judge the quality of another model's response based on the provided ground truth and context."""
        }, {
            "role": "user",
            "content": content
        }
    ]
    messages = convert_conversations(messages)

    assert 'rm_tokenizer_models' in kwargs
    rm_processor = get_rm_processor(kwargs['rm_tokenizer_models'][0])
    prompt = rm_processor.apply_chat_template(
        messages,
        tools=None,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt = prompt.format(
        question=label["question"],
        caption=label["caption"],
        A=label["A"],
        B=label["B"],
        C=label["C"],
        D=label["D"],
        answer=label["answer"],
        sampler_resp=sampler_resp_strs,
    )

    return prompt, src_imgs


def gen_rm_compute_score_geok3k(
    answer_contents: list[str],
    rule_rewards: torch.Tensor,
    reward_extra_info: dict[str, torch.Tensor],
    gen_rm_repeat: int,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rm_rewards = []
    rm_reward = 0.0
    for i, answer_content in enumerate(answer_contents):
        result, reason = parse_rm_model_output(answer_content)
        if result == "yes":
            rm_reward += 1.0
        # 求平均
        if i % gen_rm_repeat == (gen_rm_repeat - 1):
            rm_reward /= gen_rm_repeat
            rm_rewards.append(rm_reward)
            rm_reward = 0.0

    rm_rewards = torch.tensor(rm_rewards, dtype=torch.float32, device="cpu").view(-1, 1)
    rewards = rm_rewards * 1.0 / 3.0 + rule_rewards * 2.0 / 3.0
    reward_extra_info["rm_rewards"] = rm_rewards
    return rewards, reward_extra_info


def gen_rm_compute_score_pmc_vqa(
    answer_contents: list[str],
    rule_rewards: torch.Tensor,
    reward_extra_info: dict[str, torch.Tensor],
    gen_rm_repeat: int,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    rm_rewards = []
    rm_reward = 0.0
    for i, answer_content in enumerate(answer_contents):
        has_input_text_match = re.search(r'<answer>(.*?)</answer>', answer_content, re.DOTALL)
        if has_input_text_match:
            result = has_input_text_match.group(1).strip().lower()
            if result == "yes":
                rm_reward += 1
        # 求平均
        if i % gen_rm_repeat == (gen_rm_repeat - 1):
            rm_reward /= gen_rm_repeat
            rm_rewards.append(rm_reward)
            rm_reward = 0.0

    rm_rewards = torch.tensor(rm_rewards, dtype=torch.float32, device="cpu").view(-1, 1)
    rewards = rm_rewards * 1.0 / 3.0 + rule_rewards * 2.0 / 3.0
    reward_extra_info["rm_rewards"] = rm_rewards
    return rewards, reward_extra_info


def prepare_data_for_gen_rm(
    sample: Dict[str, Any],
    actor_tokenizer,
    processor,
    rule_type: str,
    rule_file: str,
    **kwargs,
) -> tuple[str, list[Image.Image]]:
    if rule_type in ['geometry3k']:
        return prepare_data_for_gen_rm_geo3k(
            sample=sample,
            actor_tokenizer=actor_tokenizer,
            processor=processor,
            **kwargs,
        )
    elif rule_type in ['pmc_vqa']:
        return prepare_data_for_gen_rm_pmc_vqa(
            sample=sample,
            actor_tokenizer=actor_tokenizer,
            processor=processor,
            **kwargs,
        )
    elif rule_type == 'import_file':
        return load_and_call(
            rule_file,
            "prepare_data_for_gen_rm",
            sample=sample,
            actor_tokenizer=actor_tokenizer,
            processor=processor,
            **kwargs,
        )
    else:
        print(f"not support to this type: {rule_type}")
        raise NotImplementedError()


def gen_rm_compute_score(
    answer_contents: list[str],
    prompt_strs: list[str],
    labels_list: list[str],
    rule_rewards: torch.Tensor,
    reward_extra_info: dict[str, torch.Tensor],
    gen_rm_repeat: int,
    rule_type: str,
    rule_file: str,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if rule_type in ['geometry3k']:
        return gen_rm_compute_score_geok3k(
            answer_contents=answer_contents,
            rule_rewards=rule_rewards,
            reward_extra_info=reward_extra_info,
            gen_rm_repeat=gen_rm_repeat,
            **kwargs,
        )
    elif rule_type in ['pmc_vqa']:
        return gen_rm_compute_score_pmc_vqa(
            answer_contents=answer_contents,
            rule_rewards=rule_rewards,
            reward_extra_info=reward_extra_info,
            gen_rm_repeat=gen_rm_repeat,
            **kwargs,
        )
    elif rule_type == 'import_file':
        return load_and_call(
            rule_file,
            "gen_rm_compute_score",
            answer_contents=answer_contents,
            rule_rewards=rule_rewards,
            reward_extra_info=reward_extra_info,
            gen_rm_repeat=gen_rm_repeat,
            prompt_strs=prompt_strs,
            labels_list=labels_list,
            **kwargs,
        )
    else:
        print(f"not support to this type: {rule_type}")
        raise NotImplementedError()


g_processor = None
g_print_rng = None


@torch.no_grad()
async def gen_rm_func(
    engine,
    batch: Optional[Dict[str, List[Any]]],
    get_processor_func=None,
):
    rule_rewards, reward_extra_info = get_rule_based_rm([batch])

    args = get_args()
    actor_tokenizer = get_actor_tokenizer()
    rm_tokenizer = get_rm_tokenizer()
    global g_processor, g_print_rng
    if g_processor is None:
        g_processor = get_processor_func(args)
    sampling_params = get_sampling_params(engine, rm_tokenizer._tokenizer.eos_token_id)
    use_vllm = args.infer_engine_impl == "vllm"
    assert not use_vllm, f"not support vllm now"
    for i in range(len(batch["imgs_repeats"])):
        batch["imgs_repeats"][i] = batch["imgs_repeats"][0]

    exbatch = expand_rollout_batch(batch)
    prompt_strs = []
    labels_list = []
    gens = []
    for i, sample in enumerate(exbatch):
        prompt, src_imgs = prepare_data_for_gen_rm(
            sample,
            actor_tokenizer,
            g_processor,
            args.ppo_mm_rule_type,
            args.ppo_custom_rule_file,
            rm_tokenizer=rm_tokenizer,
            rm_tokenizer_models=args.rm_tokenizer_models,
        )
        if not use_vllm:
            prompt_token_ids = rm_tokenizer._tokenizer.encode(prompt)

        if use_vllm:
            llm_input = dict(prompt=prompt)
        else:
            llm_input = dict(prompt_token_ids=prompt_token_ids)

        llm_input.update({
            "multi_modal_data": {
                "image": src_imgs,
            },
        })

        for j in range(args.ppo_gen_rm_repeat):
            prompt_strs.append(prompt)
            labels_list.append(sample["labels"])
            tmp_sampling_params = copy.deepcopy(sampling_params)
            tmp_sampling_params.seed += i * args.ppo_gen_rm_repeat + j
            gen = engine.async_generate(llm_input, tmp_sampling_params, str(uuid.uuid4().hex))
            gens.append(gen)

    gen_outputs = await engine.wait_and_get_async_generate_output(gens)
    output_token_ids = []
    for gi, _ in enumerate(gens):
        one_sample = gen_outputs[gi]
        if use_vllm:
            assert len(one_sample.outputs) == 1
            one_output = one_sample.outputs[0]
        else:
            assert len(one_sample.outputs) == 1
            one_output = one_sample.outputs[0]
        output_token_ids.append(list(one_output.token_ids))
    answer_contents = rm_tokenizer._tokenizer.batch_decode(
        output_token_ids, skip_special_tokens=args.ppo_skip_special_tokens
    )
    if g_print_rng is None:
        g_print_rng = random.Random(args.seed)
    # 为了看不同模型的 tokenizer 是否正常，这里打印一下输出
    if g_print_rng.randint(0, 10) % 10 == 0:
        print(f"gen-rm output:{answer_contents[0]}")

    rewards, reward_extra_info = gen_rm_compute_score(
        answer_contents,
        prompt_strs,
        labels_list,
        rule_rewards,
        reward_extra_info,
        args.ppo_gen_rm_repeat,
        args.ppo_mm_rule_type,
        args.ppo_custom_rule_file,
        exbatch=exbatch,
    )

    # tensor split into list
    rewards = [e.squeeze(0) for e in rewards.chunk(rewards.shape[0])]
    assert len(rewards) in [args.ppo_sampling_keep, args.ppo_eval_sampling_repeat]
    sampling_repeat = len(rewards)
    for key in list(reward_extra_info):
        value = reward_extra_info[key]
        reward_extra_info[key] = [e.squeeze(0) for e in value.chunk(value.shape[0])]
        assert len(reward_extra_info[key]) == sampling_repeat

    rollout_batch = dict(
        tokens=batch["tokens"],
        sequence_lengths=batch["sequence_lengths"],
        prompt_lengths=batch["prompt_lengths"],
        labels=batch["labels"],
        unique_id=batch["unique_id"],
        rewards=rewards,
        per_token_rewards=[None] * len(rewards),
    )
    rollout_batch.update(reward_extra_info)
    return rollout_batch
