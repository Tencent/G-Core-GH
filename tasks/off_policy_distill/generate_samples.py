import copy
import uuid

import torch


async def off_policy_generate_func(
    config, infer_engine, idx, tokenizer, student_tokenizer, batched_data, sampling_repeat_n
):
    prompt_token_ids = batched_data["prompt_token_ids"]
    prompt_lens = batched_data["prompt_lens"]

    # offpolicy 蒸馏，且后面设计 ce loss 的话， 实际上相当于数据蒸馏，要支持 teacher 和 student 的 tokenizer 不同情况？
    # 先简单粗暴判断
    is_same_tokenizer = tokenizer.get_vocab() == student_tokenizer.get_vocab()

    tea_prompt_token_ids = []
    if is_same_tokenizer:
        tea_prompt_token_ids = prompt_token_ids
    else:
        tokens_lst = []
        for i in range(len(prompt_token_ids)):
            tokens_lst.append(prompt_token_ids[i]['prompt_token_ids'])

        text_lst = student_tokenizer.batch_decode(tokens_lst, skip_special_tokens=False)
        tea_token_lst = tokenizer(text_lst, add_special_tokens=False).input_ids
        assert len(tea_token_lst) == len(tokens_lst), f"{len(tea_token_lst)=} != {len(tokens_lst)=}"
        for i in range(len(tea_token_lst)):
            tea_prompt_token_ids.append({'prompt_token_ids': tea_token_lst[i]})

    sampling_params = infer_engine.get_sampling_params_from_config(
        config.sampler.infer_engine_configs[idx], tokenizer.eos_token_id
    )
    async_gens = []
    for i in range(len(tea_prompt_token_ids)):
        for j in range(sampling_repeat_n):
            tmp_sampling_params = infer_engine.copy_sampling_params_with_seed_offset(
                sampling_params, i * sampling_repeat_n + j
            )
            gen = infer_engine.async_generate(
                tea_prompt_token_ids[i], tmp_sampling_params, str(uuid.uuid4().hex)
            )
            async_gens.append(gen)

    gen_outputs = await infer_engine.wait_and_get_async_generate_output(async_gens)

    tokens_lst = []
    prompt_len_lst = []
    label_lst = []

    tea_resp_tokens = []
    for gi, gen_out in enumerate(gen_outputs):
        i = gi // sampling_repeat_n
        j = gi % sampling_repeat_n
        assert len(gen_out.outputs) == 1

        resp_tokens = list(gen_out.outputs[0].token_ids)
        tea_resp_tokens.append(resp_tokens)

        prompt_len_lst.append(prompt_lens[i])
        tokens_lst.append(prompt_token_ids[i]['prompt_token_ids'])

    if is_same_tokenizer:
        label_lst = tea_resp_tokens
    else:
        resp_texts = tokenizer.batch_decode(tea_resp_tokens, skip_special_tokens=False)
        label_lst = student_tokenizer(resp_texts, add_special_tokens=False).input_ids

    seq_len_lst = []
    for i in range(len(label_lst)):
        seq_len_lst.append(prompt_len_lst[i] + len(label_lst[i]))
        assert len(tokens_lst[i]
                  ) == prompt_len_lst[i], f"{len(tokens_lst[i])=} != {prompt_len_lst[i]=}"

        tokens_lst[i] = torch.tensor(tokens_lst[i] + label_lst[i], dtype=torch.long)
        label_lst[i] = torch.tensor([-100] * prompt_len_lst[i] + label_lst[i], dtype=torch.long)

    assert len(seq_len_lst) == len(label_lst)
    assert len(
        seq_len_lst
    ) == config.training.train_mbs * sampling_repeat_n, f"{len(seq_len_lst)=} != {config.training.train_mbs * sampling_repeat_n}"

    rollout_batch = {
        'tokens': tokens_lst,
        'sequence_lengths': seq_len_lst,
        'prompt_lengths': prompt_len_lst,
        'labels': label_lst,
    }
    return rollout_batch
