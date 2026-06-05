import copy
import re
import shutil
import uuid
from collections import defaultdict
from typing import Any, Dict, List

import datasets
import torch
from mathruler.grader import extract_boxed_content, grade_answer
from PIL import Image
from transformers import AutoConfig
from transformers.models.auto.processing_auto import AutoProcessor

from megatron.core import mpu

from gpatch_v4.utils import log

g_processor = None
g_hf_config = None


def convert_pattern(
    user_input: str, image_pattern: str = '<image>', video_pattern: str = '<video>'
):
    """
        Split user input into format tokenizer accepts.
    """
    pattern = r"({image}|{video})".format(image=image_pattern, video=video_pattern)
    contents = []
    cur = 0
    mm_idx = defaultdict(int)
    for matched in re.finditer(pattern, user_input):
        start, end = matched.span()
        if start > cur:
            contents.append({"type": "text", "text": user_input[cur:start]})

        contents.append(
            {
                "type": matched.string[start:end][1:-1],
                matched.string[start:end][1:-1]: str(mm_idx[matched.string[start:end][1:-1]])
            }
        )

        cur = end
        mm_idx[matched.string[start:end][1:-1]] += 1

    if cur < len(user_input):
        contents.append({"type": "text", "text": user_input[cur:len(user_input)]})

    return contents


def convert_conversations(conversations):
    res = []
    for conversation in conversations:
        new_conversation = copy.deepcopy(conversation)
        new_conversation['content'] = convert_pattern(conversation['content'])
        res.append(new_conversation)

    return res


instruction_following = (
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within <reason> </reason> tags. The final answer MUST BE put in \boxed{}."
)


def convert_sample(problem):
    # write the answer in the json
    conversation = []
    conversation.append(dict(role="system", content="You are a helpful assistant."))
    conversation.append(dict(role="user", content=problem + " " + instruction_following))
    return conversation


def get_batch(batch, config):
    global g_processor

    images = batch["images"]
    problems = batch["problems"]
    answers = batch["answers"]

    prompt_ids = []
    raw_images = []
    assert len(problems) == len(images)
    assert len(problems) == len(answers)
    assert len(problems) == 1, "mbs must be 1"
    for problem, image, answer in zip(problems, images, answers):
        imgs = None
        if image is not None:
            imgs = [Image.fromarray(img) for img in image]
        raw_images.append(imgs)

        # pre
        conversations = convert_sample(problem)
        conversations = convert_conversations(conversations)
        add_generation_prompt = False
        if conversations[-1]['role'] == "assistant":
            conversations = conversations[:-1]

        assert conversations[-1]['role'] != "assistant"
        add_generation_prompt = True
        all_text = g_processor.apply_chat_template(
            conversations,
            tools=None,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=config.training.enable_thinking,
        )

        prompt_ids.append(g_processor.tokenizer([all_text])["input_ids"][0])

    return prompt_ids, raw_images, images, problems, answers


async def offline_generate_func(
    config, infer_engine, idx, tokenizer, student_tokenizer, batched_data, sampling_repeat_n
) -> Dict[str, List[Any]]:
    assert config.sampler.backend == "sglang", f"only support sglang now"
    is_same_tokenizer = tokenizer.get_vocab() == student_tokenizer.get_vocab()
    assert is_same_tokenizer, f"tokenizer is not the same"

    global g_processor, g_hf_config
    if g_processor is None:
        model_info = config.sampler.model_info[idx]
        g_processor = AutoProcessor.from_pretrained(model_info.hf_model_path)
        g_hf_config = AutoConfig.from_pretrained(model_info.hf_model_path)

    sampling_params = infer_engine.get_sampling_params_from_config(
        config.sampler.infer_engine_configs[idx],
        tokenizer.eos_token_id,
    )

    prompt_ids, raw_images, images, problems, answers = get_batch(batched_data, config)

    gens = []
    for i, (prompt, image) in enumerate(zip(prompt_ids, raw_images)):
        llm_input = dict(prompt_token_ids=prompt)
        if image is not None:
            llm_input.update({
                "multi_modal_data": {
                    "image": image,
                },
            })

        for j in range(sampling_repeat_n):
            tmp_sampling_params = infer_engine.copy_sampling_params_with_seed_offset(
                sampling_params, i * sampling_repeat_n + j
            )
            gen = infer_engine.async_generate(
                llm_input,
                tmp_sampling_params,
                str(uuid.uuid4().hex),
            )
            gens.append(gen)

    gen_outputs = await infer_engine.wait_and_get_async_generate_output(gens)

    output_token_ids_list = []
    pad_token_id = tokenizer.pad_token_id

    for gi, gen_out in enumerate(gen_outputs):
        i = gi // sampling_repeat_n
        j = gi % sampling_repeat_n
        assert len(gen_out.outputs) == 1

        output_token_ids = list(gen_out.outputs[0].token_ids)
        # Pitfall: possibly image token id contained (perhaps due to the bad capability of model itself).
        for i in range(len(output_token_ids)):
            if output_token_ids[i] == g_hf_config.image_token_id:
                log(f"Warning: unexpect token ids!")
                output_token_ids[i] = pad_token_id

        output_token_ids_list.append(output_token_ids)

    answers = [answer for answer in answers for _ in range(sampling_repeat_n)]
    problems = [problem for problem in problems for _ in range(sampling_repeat_n)]
    images = [img for img in images for _ in range(sampling_repeat_n)]
    assert len(answers) == len(output_token_ids_list)
    assert len(problems) == len(output_token_ids_list)
    assert len(images) == len(output_token_ids_list)

    output_texts = tokenizer.batch_decode(output_token_ids_list, skip_special_tokens=True)
    rollout_batch = dict(
        gen_text=output_texts,
        problem=problems,
        answer=answers,
        images=images,
    )

    return rollout_batch


def geo3k_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<reason>.*</reason>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


def geo3k_acc_reward(predict_str: str, label_answer: str) -> float:
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, label_answer) else 0.0


def save_rollout_samples(config=None, tokenizer=None, rbs: List[Dict[str, Any]] = None):
    features = datasets.Features(
        {
            "images": datasets.Sequence(datasets.Image()),
            "answer": datasets.Value("string"),
            "problem": datasets.Value("string"),
            "gen_text": datasets.Value("string"),
        }
    )

    # filter rbs
    rbs_filter = []
    for rb in rbs:
        fmt = geo3k_format_reward(rb["gen_text"])
        acc = geo3k_acc_reward(rb["gen_text"], rb["answer"])
        if fmt > 0 and acc > 0:
            rbs_filter.append(rb)

    # save dataset in each dp rank
    dataset = datasets.Dataset.from_list(rbs_filter, features=features)
    save_dir = config.evaluate_result.output_dir
    prefix = config.evaluate_result.output_prefix
    dp_rank = mpu.get_data_parallel_rank()
    dp_size = mpu.get_data_parallel_world_size()
    dataset.to_parquet(f"{save_dir}/tmp/{prefix}_{dp_rank}.parquet")
    torch.distributed.barrier(group=mpu.get_data_parallel_group())

    # merge the dataset
    if dp_rank == 0:
        data_files = [f"{save_dir}/tmp/{prefix}_{i}.parquet" for i in range(dp_size)]
        dataset_dict = datasets.load_dataset("parquet", data_files={"train": data_files})
        dataset = dataset_dict["train"]
        dataset.to_parquet(f"{save_dir}/{prefix}.parquet")
        shutil.rmtree(f"{save_dir}/tmp")
    torch.distributed.barrier(group=mpu.get_data_parallel_group())


def eval_rollout_samples(config=None, tokenizer=None, rbs: List[Dict[str, Any]] = None):
    accs = []
    fmts = []
    for rb in rbs:
        gen_text = rb["gen_text"]
        answer = rb["answer"]
        fmts.append(geo3k_format_reward(gen_text))
        accs.append(geo3k_acc_reward(gen_text, answer))

    # all gather the accs and fmts in dp group
    list_fmt_acc = [None] * mpu.get_data_parallel_world_size()
    torch.distributed.all_gather_object(
        list_fmt_acc, [fmts, accs], group=mpu.get_data_parallel_group()
    )

    # merge the dataset
    fmts = []
    accs = []
    if mpu.get_data_parallel_rank() == 0:
        for fmt_acc in list_fmt_acc:
            fmts.extend(fmt_acc[0])
            accs.extend(fmt_acc[1])
        print(f"acc:{sum(accs)/len(accs)} fmt:{sum(fmts)/len(fmts)} {len(fmts)=} {len(accs)=}")
    torch.distributed.barrier(group=mpu.get_data_parallel_group())
