import copy
import random
import uuid
from typing import Dict

import torch


# gen rm
def extract_by_split(tag_name, text, eos_token):
    start_tag = f"<|im_start|>{tag_name}\n"
    end_tag = eos_token

    try:
        content = text.split(start_tag)[1].split(end_tag)[0]
        return content.strip()
    except IndexError:
        return None


# gen rm
def get_question_and_answer(text, actor_eos_token):
    user_content = extract_by_split('user', text, '<|im_end|>')
    assistant_content = extract_by_split('assistant', text, actor_eos_token)
    assert user_content is not None
    assert assistant_content is not None
    return (user_content, assistant_content)


def get_gen_rm_prompt(batched_data: Dict, actor_tokenizer, rm_tokenizer):
    gen_rm_prompt = """<|im_start|>You are a math teacher. Grade the Solution, verifying correctness step by step. Use Expected Answer to find any erroneous step in the Solution. At the end of the Solution verification, when you give your final grade, write it in the form "Verification: Is the answer correct (Yes/No)? X",  where X is either Yes or No.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"""
    tokens_list = batched_data["tokens"]  # List[Tensor]
    seq_len_list = batched_data["sequence_lengths"]  # List[Tensor]

    resp_toks_as_l = [token[:length] for token, length in zip(tokens_list, seq_len_list)]
    resp_strs = actor_tokenizer.batch_decode(resp_toks_as_l, skip_special_tokens=False)
    rm_prompts = []
    actor_eos_token = actor_tokenizer.eos_token or "<|im_end|>"
    for resp in resp_strs:
        question, solution = get_question_and_answer(resp, actor_eos_token)
        content = f"Question: {question}\nSolution: {solution}"
        rm_prompt = gen_rm_prompt.format(content)
        rm_prompts.append(rm_prompt)

    rm_tokens = []
    for p in rm_prompts:
        rm_tokens.append(rm_tokenizer(p, add_special_tokens=True).input_ids)

    return rm_tokens


# gen rm
def get_rm_verification(text):
    # 粗暴的方式，佛了。因为 demo 使用的 model 指令跟随能力的问题，无法完全跟随指令要求,
    # 所以拿到的结果不是很准，先用这个简单粗暴的方式，具体业务需要自己训练好 gen-rm model
    answer_lst = text.split("Is the answer correct (Yes/No)")
    if len(answer_lst) < 2:
        return 0
    answer = answer_lst[1].lower()
    yes_pos = answer.find("yes")
    no_pos = answer.find("no")

    # 本来应该使用正则表达式匹配，受限于 model 指令跟随能力。有时候可能抽取不出来 yes / no
    # pattern = r"Is the answer correct \(Yes/No\)\?\s*(Yes|No)"
    # match = re.search(pattern, text, re.IGNORECASE)

    # if match:
    #     answer = match.group(1).lower()
    #     return 1 if answer == "yes" else -1
    # else:
    #     return 0
    if yes_pos == -1 and no_pos == -1:
        return 0
    elif yes_pos == -1:
        return -1
    elif no_pos == -1:
        return 1
    elif yes_pos < no_pos:
        return 1
    else:
        return -1


# ---------------------------------------------------------------------------
# parse_reward_fn – called by GrpoGenRmActor.generate_rewards
# ---------------------------------------------------------------------------
async def generate_rewards(
    config, rm_infer_engine, rm_idx, rm_tokenizer, actor_tokenizer, batched_data, reward_repeat_n
):
    repeat_n = config.training.sampling_repeat_n
    rollout_mbs = config.training.rollout_mbs
    for k, v in batched_data.items():
        assert isinstance(v, list) and len(
            v
        ) == rollout_mbs * repeat_n, f'unexpected {k=} {v=} {rollout_mbs=} {repeat_n=} {len(v)=}'

    sampling_params = rm_infer_engine.get_sampling_params_from_config(
        config.gen_rm.infer_engine_configs[rm_idx], rm_tokenizer.eos_token_id
    )

    token_ids = get_gen_rm_prompt(batched_data, actor_tokenizer, rm_tokenizer)
    assert len(token_ids) == rollout_mbs * repeat_n

    gens = []
    for i in range(rollout_mbs * repeat_n):
        for j in range(reward_repeat_n):
            tmp_sampling_params = rm_infer_engine.copy_sampling_params_with_seed_offset(
                sampling_params, i * reward_repeat_n + j
            )
            input_dict = {
                'prompt_token_ids': token_ids[i],
            }
            gen = rm_infer_engine.async_generate(
                input_dict, tmp_sampling_params, str(uuid.uuid4().hex)
            )
            gens.append(gen)

    gen_outputs = await rm_infer_engine.wait_and_get_async_generate_output(gens)

    gen_rewards = [0. for _ in range(rollout_mbs * repeat_n)]
    output_strs = []
    for gi, gen in enumerate(gens):
        i = gi // reward_repeat_n
        j = gi % reward_repeat_n
        assert len(gen_outputs[gi].outputs) == 1
        output_tokens = gen_outputs[gi].outputs[0].token_ids
        resp_i_j = rm_tokenizer.batch_decode([output_tokens], skip_special_tokens=False)[0]
        output_strs.append(resp_i_j)

        reward_i_j = get_rm_verification(resp_i_j)
        gen_rewards[i] += torch.tensor(reward_i_j / reward_repeat_n, dtype=torch.float32)

    random_idx = random.randint(0, len(output_strs) - 1)
    print(f"gen-rm random output: {output_strs[random_idx]}")

    ret_dict = {
        "rm_rewards": gen_rewards,
        "rewards": gen_rewards,
        "per_token_rewards": [None] * len(gen_rewards)
    }
    return ret_dict
