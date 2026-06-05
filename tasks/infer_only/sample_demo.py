import copy
import time
import uuid

_global_stats = {
    'total_samples': 0,
    'total_prompt_tokens': 0,
    'total_output_tokens': 0,
    'total_time': 0.0,
    'first_call_time': None,
}


async def generate_func(config, infer_engine, idx, tokenizer, batched_data, sampling_repeat_n):
    global _global_stats

    start_time = time.time()

    if _global_stats['first_call_time'] is None:
        _global_stats['first_call_time'] = start_time

    queries = batched_data.get("query", [])
    search_res_list = batched_data.get("search_res", [])

    batch_size = len(queries)

    enable_think_mode = False
    if hasattr(config, 'infer_result') and hasattr(config.infer_result, 'enable_think_mode'):
        enable_think_mode = config.infer_result.enable_think_mode

    # concatenate query and search_res
    prompts = []
    for query, search_res in zip(queries, search_res_list):
        prompt = f"问题：{query}\n\n参考资料：\n{search_res}\n\n请基于上述参考资料回答问题："

        if enable_think_mode:
            prompt += "<think>"

        prompts.append(prompt)

    # tokenize prompts
    prompt_token_ids_list = []
    for prompt in prompts:
        encoded = tokenizer(prompt, add_special_tokens=True)
        if hasattr(encoded, 'input_ids'):
            token_ids = encoded.input_ids
        else:
            token_ids = encoded
        prompt_token_ids_list.append(token_ids)

    sampling_params = infer_engine.get_sampling_params_from_config(
        config.sampler.infer_engine_configs[idx], tokenizer.eos_token_id
    )

    print(
        f"[DEBUG] Sampling params: temperature={sampling_params.temperature}, "
        f"max_tokens={sampling_params.max_new_tokens if hasattr(sampling_params, 'max_new_tokens') else sampling_params.max_tokens}, "
        f"top_p={sampling_params.top_p} eos_token_id={tokenizer.eos_token_id}"
    )

    async_gens = []
    for i in range(batch_size):
        for j in range(sampling_repeat_n):
            tmp_sampling_params = infer_engine.copy_sampling_params_with_seed_offset(
                sampling_params, i * sampling_repeat_n + j
            )
            if config.infer_result.constant_seed is not None:
                tmp_sampling_params = infer_engine.set_sampling_params_seed(
                    tmp_sampling_params, config.infer_result.constant_seed + j
                )
            gen = infer_engine.async_generate(
                {'prompt_token_ids': prompt_token_ids_list[i]}, tmp_sampling_params,
                str(uuid.uuid4().hex)
            )
            async_gens.append(gen)

    gen_outputs = await infer_engine.wait_and_get_async_generate_output(async_gens)

    prompt_lst = []
    response_lst = []
    prompt_ids_lst = []
    response_ids_lst = []
    query_lst = []
    search_res_lst = []

    batch_prompt_tokens = 0
    batch_output_tokens = 0

    for gi, gen_out in enumerate(gen_outputs):
        i = gi // sampling_repeat_n
        j = gi % sampling_repeat_n

        assert len(gen_out.outputs) == 1
        resp_tokens = list(gen_out.outputs[0].token_ids)

        response_text = tokenizer.decode(resp_tokens, skip_special_tokens=True)
        print(
            f"[DEBUG] Response text: {len(response_text)} tokens: {len(resp_tokens)} prompt: {len(prompts[i])}"
        )
        prompt_lst.append(prompts[i])
        response_lst.append(response_text)
        prompt_ids_lst.append(prompt_token_ids_list[i])
        response_ids_lst.append(resp_tokens)
        query_lst.append(queries[i])
        search_res_lst.append(search_res_list[i])

        batch_prompt_tokens += len(prompt_token_ids_list[i])
        batch_output_tokens += len(resp_tokens)

    # statistics
    end_time = time.time()
    batch_time = end_time - start_time
    batch_samples = len(gen_outputs)

    _global_stats['total_samples'] += batch_samples
    _global_stats['total_prompt_tokens'] += batch_prompt_tokens
    _global_stats['total_output_tokens'] += batch_output_tokens
    _global_stats['total_time'] += batch_time

    avg_e2e_time = _global_stats['total_time'] / _global_stats['total_samples'] if _global_stats[
        'total_samples'] > 0 else 0
    avg_prompt_len = _global_stats['total_prompt_tokens'] / _global_stats[
        'total_samples'] if _global_stats['total_samples'] > 0 else 0
    avg_output_len = _global_stats['total_output_tokens'] / _global_stats[
        'total_samples'] if _global_stats['total_samples'] > 0 else 0
    elapsed_time = end_time - _global_stats['first_call_time']

    print(
        f"[STATS] samples={_global_stats['total_samples']} | avg_e2e={avg_e2e_time:.3f}s | avg_prompt_len={avg_prompt_len:.1f} | avg_output_len={avg_output_len:.1f} | elapsed_time={elapsed_time:.2f}s | qps={_global_stats['total_samples']/elapsed_time:.2f}samples/s"
    )

    result = {
        'prompt': prompt_lst,
        'response': response_lst,
        'prompt_ids': prompt_ids_lst,
        'response_ids': response_ids_lst,
        'query': query_lst,
        'search_res': search_res_lst,
    }

    return result
