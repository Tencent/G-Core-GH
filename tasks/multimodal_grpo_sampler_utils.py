# coding=utf-8
# copyright (c) 2025 tencent inc. all rights reserved.
# guanyouhe@tencent.com, xiaotaoliu@tencent.com, nrwu@tencent.com

import copy
import uuid

import torch
from PIL import Image
import numpy as np

try:
    import sglang as sgl
except ImportError:
    pass

from megatron.training.global_vars import get_args, get_tokenizer

from gpatch.core.utils import print_with_rank_and_datetime
from megatron_datasets.mm_dataset import convert_conversations, remove_bos
from gpatch.core.sampler_v3.routed_experts_utils import process_routed_experts

g_log_once_flag = False


def get_sampling_params(engine, eos_token_id, forbiden_token_ids=[], args=None):

    # TODO(hessianliu): remove deps on  global_vars
    if not args:
        args = get_args()
    logit_bias = {}
    for e in forbiden_token_ids:
        if engine.infer_engine_impl == 'sglang':
            logit_bias[str(e)] = -100
        else:
            logit_bias[e] = -100
    # disable logit_bias
    if engine.infer_engine_impl == 'sglang':
        if engine.sglang_version in ['0.4.6.post5']:
            global g_log_once_flag
            if not g_log_once_flag:
                g_log_once_flag = True
                print(f"unexpected Ignore logit_bias:{logit_bias} in sglang-0.4.6.post5")
            logit_bias = None
    # Create a sampling params object.
    return engine.get_sampling_params(
        n=1,  # by passing vllm async llm issues
        temperature=args.ppo_rollout_temperature,
        top_k=args.ppo_rollout_top_k if args.ppo_rollout_top_k > 0 else -1,
        top_p=args.ppo_rollout_top_p,
        max_tokens=args.ppo_resp_seq_len,
        stop_token_ids=[eos_token_id],
        seed=args.seed,
        repetition_penalty=args.ppo_rollout_repetition_penalty,
        logit_bias=logit_bias,
    )


class SamplerGetBatch(object):
    def __init__(self, get_processor_func, rm_bos: bool = True, check_pre_computed_tokens=False):
        self.get_processor_func = get_processor_func
        self.processor = None
        self.rm_bos = rm_bos
        self.check_pre_computed_tokens = check_pre_computed_tokens

    def __call__(self, batch, use_vllm, args=None):
        json_data_list = batch["json_data_list"]
        imgs_np_array_list = batch["imgs_np_array_list"]
        tokens_for_gen = batch.get("tokens_for_gen", [])
        if not args:
            args = get_args()
        if self.processor is None:
            self.processor = self.get_processor_func(args)
        prompt_texts_or_ids = []
        raw_images = []
        labels = []
        assert len(imgs_np_array_list) == len(json_data_list)
        for i, (json_data, imgs_np_array) in enumerate(zip(json_data_list, imgs_np_array_list)):
            imgs = None
            if imgs_np_array is not None:
                imgs = [Image.fromarray(img) for img in imgs_np_array]
            raw_images.append(imgs)

            pre_computed_tokens_for_gen = tokens_for_gen[i].tolist() if tokens_for_gen else None
            if tokens_for_gen and (not use_vllm) and not self.check_pre_computed_tokens:
                prompt_texts_or_ids.append(tokens_for_gen[i].tolist())
            else:
                # pre
                conversations = convert_conversations(json_data['conversations'])
                add_generation_prompt = False
                if conversations[-1]['role'] == "assistant":
                    conversations = conversations[:-1]

                assert conversations[-1]['role'] != "assistant"
                add_generation_prompt = True
                all_text = self.processor.apply_chat_template(
                    conversations,
                    tools=None,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
                if self.rm_bos:
                    all_text = remove_bos(all_text)
                if use_vllm:
                    prompt_texts_or_ids.append(all_text)
                else:
                    prompt_texts_or_id = self.processor.tokenizer([all_text])["input_ids"][0]
                    if pre_computed_tokens_for_gen:
                        assert pre_computed_tokens_for_gen == prompt_texts_or_id
                    prompt_texts_or_ids.append(prompt_texts_or_id)

            labels.append(json_data['label'])

        return prompt_texts_or_ids, raw_images, labels, batch


@torch.no_grad()
async def gen_rollouts(
    engine,
    batch,
    get_batch_obj: SamplerGetBatch,
    sampling_repeat: int,
    args=None,
    tokenizer=None,
):
    if not args:
        args = get_args()
    if not tokenizer:
        tokenizer = get_tokenizer()
    image_token_id = args.image_token_id
    assert image_token_id is not None
    # print(f"tokenizer.image_token_id {image_token_id}")
    sampling_params = get_sampling_params(
        engine, tokenizer._tokenizer.eos_token_id, [image_token_id]
    )
    use_vllm = args.infer_engine_impl == "vllm"
    prompt_texts_or_ids, raw_images, labels, prompt_data = get_batch_obj(batch, use_vllm)
    rank_unique_ids = prompt_data["unique_id"]
    tokens_from_dataset = prompt_data["tokens"]
    prompt_len_from_dataset = prompt_data["prompt_len"]
    json_data_list = batch["json_data_list"]

    gens = []
    # print(f"{prompt_texts_or_ids=} {raw_images=}")
    for i, (prompt, image) in enumerate(zip(prompt_texts_or_ids, raw_images)):
        if use_vllm:
            llm_input = dict(prompt=prompt)
        else:
            llm_input = dict(prompt_token_ids=prompt)
        if image is not None:
            llm_input.update({
                "multi_modal_data": {
                    "image": image,
                },
            })
        for j in range(sampling_repeat):
            tmp_sampling_params = copy.deepcopy(sampling_params)
            tmp_sampling_params.seed += i * sampling_repeat + j
            gen = engine.async_generate(
                llm_input,
                tmp_sampling_params,
                str(uuid.uuid4().hex),
                return_routed_experts=args.moe_router_replay
            )
            gens.append(gen)

    gen_outputs = await engine.wait_and_get_async_generate_output(gens)

    tokens = []
    sequence_lengths = []
    prompt_lengths = []
    routed_experts_list = []
    rollout_log_probs = []
    pad_token_id = tokenizer._tokenizer.pad_token_id
    image_token_id = args.image_token_id

    for gi, _ in enumerate(gens):
        i = gi // sampling_repeat
        j = gi % sampling_repeat

        one_sample = gen_outputs[gi]
        routed_experts = None
        if use_vllm:
            one_prompt_token_ids = one_sample.prompt_token_ids
            assert len(one_sample.outputs) == 1
            one_output = one_sample.outputs[0]
            # the tokenizer should be same between actor and sampler
            assert len(one_prompt_token_ids) == prompt_len_from_dataset[
                i], f"{len(one_prompt_token_ids)=} {prompt_len_from_dataset[i]=}"
            assert torch.equal(
                tokens_from_dataset[i][:prompt_len_from_dataset[i]],
                torch.tensor(one_prompt_token_ids, dtype=torch.long),
            ), f"{tokens_from_dataset[i][:prompt_len_from_dataset[i]].tolist()=} {one_prompt_token_ids=}"
        else:
            one_output = one_sample.outputs[0]
            # the tokenizer should be same between actor and sampler
            assert one_output.prompt_len == prompt_len_from_dataset[
                i], f"{one_output.prompt_len=} {prompt_len_from_dataset[i]=}"

            one_prompt_token_ids = tokens_from_dataset[i][:prompt_len_from_dataset[i]].tolist()
            routed_experts = process_routed_experts(
                one_output, args.num_layers, args.moe_router_topk
            )

        output_token_ids = list(one_output.token_ids)
        rollout_log_prob = one_output.output_logprobs
        assert len(output_token_ids) == len(rollout_log_prob)
        # Pitfall: possibly image token id contained (perhaps due to the bad capability of model itself).
        for i in range(len(output_token_ids)):
            if output_token_ids[i] == image_token_id:
                print_with_rank_and_datetime(f"unexpect token ids!")
                output_token_ids[i] = pad_token_id
        token = one_prompt_token_ids + output_token_ids
        if len(token) > args.decoder_seq_length:
            truncate_len = len(token) - args.decoder_seq_length
            token = token[:args.decoder_seq_length]
            rollout_log_prob = rollout_log_prob[:-truncate_len]
            if routed_experts is not None:
                routed_experts = routed_experts[:-truncate_len]
            print_with_rank_and_datetime(f"Warning you should filter this sample")
        sequence_lengths.append(torch.tensor(len(token), dtype=torch.long))
        tokens.append(torch.tensor(token, dtype=torch.long))
        prompt_lengths.append(torch.tensor(len(one_prompt_token_ids), dtype=torch.long))
        routed_experts_list.append(routed_experts)

        # 组装 rollout logps
        gen_lp = torch.tensor(rollout_log_prob, dtype=torch.float32)
        prompt_len = len(one_prompt_token_ids)
        full_lp = torch.ones(len(token), dtype=torch.float32)
        gen_len = gen_lp.size(0)
        assert len(token) == prompt_len + gen_len
        full_lp[prompt_len - 1:prompt_len + gen_len - 1] = gen_lp

        rollout_log_probs.append(full_lp)

    labels = [label for label in labels for _ in range(sampling_repeat)]
    rank_unique_ids = [unique_id for unique_id in rank_unique_ids for _ in range(sampling_repeat)]
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

    if args.moe_router_replay:
        rollout_batch["routed_experts"] = routed_experts_list

    if args.use_gen_rm:
        json_data_repeats = [
            json_data for json_data in json_data_list for _ in range(sampling_repeat)
        ]
        assert len(labels) == len(json_data_repeats)
        rollout_batch["json_data_repeats"] = json_data_repeats

        # Images are not repeated as this may cause memory OOM
        imgs_np_array_list = batch["imgs_np_array_list"]
        imgs_repeats = []
        for imgs in imgs_np_array_list:
            for i in range(sampling_repeat):
                if i == 0:
                    if imgs is None:
                        imgs_repeats.append([np.array([])])
                    else:
                        imgs_repeats.append(imgs)
                else:
                    imgs_repeats.append([np.array([])])
        rollout_batch["imgs_repeats"] = imgs_repeats
    return rollout_batch
